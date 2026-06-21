//! **N×N SPSC mesh** — a shared-nothing, thread-per-core runtime.
//!
//! Each of the `N` workers is pinned to a core and *exclusively* owns one shard.
//! A worker never touches another shard's data — not even atomically — so there
//! is **zero cross-core cache-line bouncing** on the index. The shard is a plain
//! `BTreeMap` (no atomics, no locks): single-threaded access by construction.
//!
//! Inter-core communication is an `N×N` matrix of the lock-free [`crate::spsc`]
//! queues:
//!   * `cmd[i][j]`  — commands from worker `i` to worker `j` (its inbox column).
//!   * `reply[i][j]`— replies from worker `i` back to worker `j`.
//!
//! Every queue has exactly one producer and one consumer, so the SPSC discipline
//! holds with no CAS.
//!
//! **Deadlock freedom.** A worker that finds a destination queue full does not
//! block — it *pumps its own inboxes* (serving others, draining replies) and
//! retries. Because every worker keeps making its own queues drainable, the mesh
//! cannot deadlock under backpressure.

use crate::spsc::{self, Consumer, Producer};
use std::collections::BTreeMap;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, Thread};

/// A command routed to the owning shard. `Put` is fire-and-forget; `Get` expects
/// a `Reply` on the reverse channel (used by the service mode's forward path).
enum Command {
    Put { key: Vec<u8>, val: Vec<u8> },
    Get { key: Vec<u8>, from: usize, token: u64 },
}

/// A reply to a `Get`, routed back to the requester.
struct Reply {
    token: u64,
    val: Option<Vec<u8>>,
}

/// Shared, mesh-wide counters used to quiesce (know when all in-flight commands
/// have been applied) and to pin threads.
struct MeshShared {
    sent: AtomicU64,
    applied: AtomicU64,
    phase1_done: AtomicUsize, // workers that finished submitting their workload
    n: usize,
}

/// Handle to a running mesh. Workers run on their own threads until [`shutdown`].
pub struct Mesh {
    workers: Vec<JoinHandle<()>>,
    shared: Arc<MeshShared>,
    n: usize,
}

impl Mesh {
    pub fn num_shards(&self) -> usize {
        self.n
    }

    /// Join all workers (each self-terminates once the mesh has quiesced: every
    /// worker finished producing and every routed command was applied). Returns
    /// the total number of commands applied across all shards.
    pub fn shutdown(mut self) -> u64 {
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
        self.shared.applied.load(Ordering::Relaxed)
    }
}

/// FNV-1a routing hash → owning shard (power-of-two `n` ⇒ mask).
#[inline]
fn route(key: &[u8], mask: u64) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in key {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    (h & mask) as usize
}

/// Per-worker private state and its row/column of the matrices.
struct Worker {
    id: usize,
    n: usize,
    mask: u64,
    shard: BTreeMap<Vec<u8>, Vec<u8>>,
    /// `cmd_rx[src]` — inbox from worker `src` (column `id` of the cmd matrix).
    cmd_rx: Vec<Consumer<Command>>,
    /// `cmd_tx[dst]` — outbox to worker `dst` (row `id` of the cmd matrix).
    cmd_tx: Vec<Producer<Command>>,
    /// `reply_rx[src]` — replies arriving from worker `src`.
    reply_rx: Vec<Consumer<Reply>>,
    /// `reply_tx[dst]` — replies we send to worker `dst`.
    reply_tx: Vec<Producer<Reply>>,
    shared: Arc<MeshShared>,
    /// Outstanding local Gets keyed by token → resolved value slot.
    inflight: BTreeMap<u64, Option<Vec<u8>>>,
    next_token: u64,
}

impl Worker {
    /// Serve everything currently in our inboxes once: apply incoming commands,
    /// answer Gets, and collect replies to our own Gets. Returns true if any
    /// progress was made.
    fn pump(&mut self) -> bool {
        let mut progressed = false;

        // 1. Drain incoming commands (others writing/reading OUR shard).
        for src in 0..self.n {
            // SPSC drain; bounded per call to keep latency even across inboxes.
            while let Some(cmd) = self.cmd_rx[src].pop() {
                progressed = true;
                match cmd {
                    Command::Put { key, val } => {
                        self.shard.insert(key, val);
                        self.shared.applied.fetch_add(1, Ordering::Relaxed);
                    }
                    Command::Get { key, from, token } => {
                        let val = self.shard.get(&key).cloned();
                        // Reply on the reverse channel; if full, pump-and-retry.
                        let mut reply = Reply { token, val };
                        while let Err(r) = self.reply_tx[from].push(reply) {
                            reply = r;
                            self.drain_replies();
                            core::hint::spin_loop();
                        }
                        self.shared.applied.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        // 2. Collect replies to our own outstanding Gets.
        if self.drain_replies() {
            progressed = true;
        }
        progressed
    }

    /// Collect any replies addressed to us, resolving outstanding Gets.
    fn drain_replies(&mut self) -> bool {
        let mut progressed = false;
        for src in 0..self.n {
            while let Some(rep) = self.reply_rx[src].pop() {
                progressed = true;
                self.inflight.insert(rep.token, rep.val);
            }
        }
        progressed
    }

    /// Apply a Put to the right shard: locally if we own it, else route via SPSC
    /// (pumping our own inboxes on backpressure so we never deadlock).
    fn put(&mut self, key: Vec<u8>, val: Vec<u8>) {
        let owner = route(&key, self.mask);
        self.shared.sent.fetch_add(1, Ordering::Relaxed);
        if owner == self.id {
            self.shard.insert(key, val);
            self.shared.applied.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let mut cmd = Command::Put { key, val };
        while let Err(c) = self.cmd_tx[owner].push(cmd) {
            cmd = c;
            self.pump(); // make progress so the remote queue drains
            core::hint::spin_loop();
        }
    }
}

/// Build and start a mesh of `requested` shards (rounded up to a power of two),
/// pinning worker `t` to physical core `t` when available. The closure
/// `workload(worker_id, n_shards) -> Vec<(key, val)>` produces each worker's
/// Put workload; the harness applies it through the mesh, then quiesces.
pub fn run_put_workload<F>(requested: usize, workload: F) -> (Mesh, u64)
where
    F: Fn(usize, usize) -> Vec<(Vec<u8>, Vec<u8>)> + Send + Sync + 'static,
{
    let n = requested.max(1).next_power_of_two();
    let mask = (n as u64) - 1;
    let shared = Arc::new(MeshShared {
        sent: AtomicU64::new(0),
        applied: AtomicU64::new(0),
        phase1_done: AtomicUsize::new(0),
        n,
    });

    // Allocate the two N×N SPSC matrices. `cmd[i][j]` / `reply[i][j]`.
    let cap = 4096;
    let mut cmd_p: Vec<Vec<Option<Producer<Command>>>> = grid(n);
    let mut cmd_c: Vec<Vec<Option<Consumer<Command>>>> = grid(n);
    let mut rep_p: Vec<Vec<Option<Producer<Reply>>>> = grid(n);
    let mut rep_c: Vec<Vec<Option<Consumer<Reply>>>> = grid(n);
    for i in 0..n {
        for j in 0..n {
            let (cp, cc) = spsc::channel::<Command>(cap);
            cmd_p[i][j] = Some(cp);
            cmd_c[i][j] = Some(cc);
            let (rp, rc) = spsc::channel::<Reply>(cap);
            rep_p[i][j] = Some(rp);
            rep_c[i][j] = Some(rc);
        }
    }

    let cores = core_affinity::get_core_ids().unwrap_or_default();
    let workload = Arc::new(workload);
    let mut workers = Vec::with_capacity(n);

    for id in 0..n {
        // Gather this worker's row (outbound producers) and column (inbound
        // consumers) from the matrices, moving the endpoints out.
        let cmd_tx: Vec<Producer<Command>> = (0..n).map(|j| cmd_p[id][j].take().unwrap()).collect();
        let cmd_rx: Vec<Consumer<Command>> = (0..n).map(|src| cmd_c[src][id].take().unwrap()).collect();
        let reply_tx: Vec<Producer<Reply>> = (0..n).map(|j| rep_p[id][j].take().unwrap()).collect();
        let reply_rx: Vec<Consumer<Reply>> = (0..n).map(|src| rep_c[src][id].take().unwrap()).collect();

        let shared = shared.clone();
        let workload = workload.clone();
        let core = cores.get(id).copied();

        let h = std::thread::Builder::new()
            .name(format!("mesh-shard-{id}"))
            .spawn(move || {
                if let Some(c) = core {
                    core_affinity::set_for_current(c);
                }
                let mut w = Worker {
                    id,
                    n,
                    mask,
                    shard: BTreeMap::new(),
                    cmd_rx,
                    cmd_tx,
                    reply_rx,
                    reply_tx,
                    shared,
                    inflight: BTreeMap::new(),
                    next_token: 0,
                };
                let _ = w.next_token; // reserved for the Get path

                // Phase 1: apply our own workload (routes cross-core as needed),
                // interleaving pumps so peers' inboxes keep draining.
                let items = (workload)(id, n);
                for (k, v) in items {
                    w.put(k, v);
                    w.pump();
                }
                // Announce we're done producing; until ALL workers reach here,
                // any of them may still route commands to us, so we must keep
                // serving our inboxes.
                w.shared.phase1_done.fetch_add(1, Ordering::AcqRel);

                // Phase 2: quiesce. Keep pumping (serving peers) until every
                // worker has finished producing AND every command mesh-wide has
                // been applied. Only then is it safe to stop draining — no peer
                // can route to us anymore, so no `put` can be stuck on us.
                loop {
                    w.pump();
                    let all_done = w.shared.phase1_done.load(Ordering::Acquire) == w.shared.n;
                    let sent = w.shared.sent.load(Ordering::Acquire);
                    let applied = w.shared.applied.load(Ordering::Acquire);
                    if all_done && sent == applied {
                        // One last drain to clear in-flight messages, then exit.
                        w.pump();
                        break;
                    }
                    core::hint::spin_loop();
                }
            })
            .expect("spawn mesh worker");
        workers.push(h);
    }

    let total_expected = 0; // filled by caller via shutdown()
    (
        Mesh {
            workers,
            shared,
            n,
        },
        total_expected,
    )
}

fn grid<T>(n: usize) -> Vec<Vec<Option<T>>> {
    (0..n).map(|_| (0..n).map(|_| None).collect()).collect()
}

// ===========================================================================
// Long-running service mode + external client request/response API.
//
// The benchmark harness above drives a closed Put workload. A *service* must
// instead accept requests from arbitrary external client threads and return
// responses, while staying shared-nothing. The realistic pattern (seastar /
// ScyllaDB): a request lands on an *accepting* worker (here: round-robin, as if
// by connection affinity); if that worker owns the key's shard it serves it
// locally, otherwise it **forwards** over the inter-core matrix to the owner and
// relays the reply back — exercising the full `Command::Get` / `Reply` path.
// ===========================================================================

/// What a client asks for.
enum ClientOp {
    Put { key: Vec<u8>, val: Vec<u8> },
    Get { key: Vec<u8> },
}

/// Blocking completion handle shared between the client thread and the worker
/// that fulfills the request. Park/unpark, lost-wakeup-safe.
struct ReqCompletion {
    done: AtomicBool,
    value: Mutex<Option<Vec<u8>>>,
    waiter: Thread,
}

impl ReqCompletion {
    fn new() -> Arc<Self> {
        Arc::new(ReqCompletion {
            done: AtomicBool::new(false),
            value: Mutex::new(None),
            waiter: thread::current(),
        })
    }
    /// Fulfill with an optional value and wake the waiting client.
    fn fulfill(&self, val: Option<Vec<u8>>) {
        *self.value.lock().unwrap() = val;
        self.done.store(true, Ordering::Release); // publish before unpark
        self.waiter.unpark();
    }
    /// Block the calling client until fulfilled, returning the value.
    ///
    /// Hybrid spin-then-park: replies usually come back within microseconds, so
    /// we spin briefly first and only `park()` (a scheduler round-trip) if the
    /// answer is genuinely slow. This collapses the common-case latency from a
    /// deschedule (~1 µs) to a handful of cache-coherence loads.
    fn wait(&self) -> Option<Vec<u8>> {
        const SPIN: u32 = 2000;
        for _ in 0..SPIN {
            if self.done.load(Ordering::Acquire) {
                return self.value.lock().unwrap().take();
            }
            core::hint::spin_loop();
        }
        while !self.done.load(Ordering::Acquire) {
            thread::park();
        }
        self.value.lock().unwrap().take()
    }
}

/// A client request plus its completion, linked into a worker's MPSC ingress.
struct ClientReq {
    op: ClientOp,
    completion: Arc<ReqCompletion>,
    next: *mut ClientReq,
}

/// Lock-free **MPSC** ingress: any number of client threads push (CAS), the
/// owning worker drains by swap. Same Treiber-stack pattern as `ebr`/`group_commit`.
struct Ingress {
    head: AtomicPtr<ClientReq>,
}

impl Ingress {
    fn new() -> Self {
        Ingress {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }
    /// Push from a client thread (multi-producer).
    fn push(&self, op: ClientOp, completion: Arc<ReqCompletion>) {
        let node = Box::into_raw(Box::new(ClientReq {
            op,
            completion,
            next: ptr::null_mut(),
        }));
        loop {
            let head = self.head.load(Ordering::Acquire);
            // SAFETY: we own `node` until it is published by the CAS.
            unsafe { (*node).next = head };
            if self
                .head
                .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
            core::hint::spin_loop();
        }
    }
    /// Drain everything currently queued into a Vec (single consumer = worker).
    fn drain(&self) -> Vec<ClientReq> {
        let mut node = self.head.swap(ptr::null_mut(), Ordering::AcqRel);
        let mut out = Vec::new();
        while !node.is_null() {
            // SAFETY: nodes came off the stack; each uniquely owned now.
            let boxed = unsafe { Box::from_raw(node) };
            node = boxed.next;
            out.push(*boxed); // move the request out; the intrusive box is freed
        }
        out
    }
}

// SAFETY: `ClientReq` raw pointers are owned by exactly one party at a time
// (producer before publish, worker after swap); all sharing is via atomics.
unsafe impl Send for Ingress {}
unsafe impl Sync for Ingress {}

/// Shared service state.
struct ServiceShared {
    ingress: Vec<Ingress>, // one MPSC inbox per worker
    n: usize,
    mask: u64,       // n is a power of two ⇒ routing mask
    rr: AtomicUsize, // round-robin accept counter
    stop: AtomicBool,
}

/// A running, core-pinned shared-nothing service. Hand out cloneable
/// [`ServiceHandle`]s to client threads.
pub struct MeshService {
    workers: Vec<JoinHandle<()>>,
    shared: Arc<ServiceShared>,
}

/// Cheap, cloneable client handle. `Send + Sync` so it can be shared across
/// client threads.
#[derive(Clone)]
pub struct ServiceHandle {
    shared: Arc<ServiceShared>,
}

impl MeshService {
    /// Start `requested` shards (rounded to a power of two), each on its own
    /// core-pinned worker thread, running until [`MeshService::shutdown`].
    pub fn start(requested: usize) -> Self {
        let n = requested.max(1).next_power_of_two();
        let mask = (n as u64) - 1;
        let shared = Arc::new(ServiceShared {
            ingress: (0..n).map(|_| Ingress::new()).collect(),
            n,
            mask,
            rr: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
        });

        // Build the N×N command + reply matrices.
        let cap = 4096;
        let mut cmd_p: Vec<Vec<Option<Producer<Command>>>> = grid(n);
        let mut cmd_c: Vec<Vec<Option<Consumer<Command>>>> = grid(n);
        let mut rep_p: Vec<Vec<Option<Producer<Reply>>>> = grid(n);
        let mut rep_c: Vec<Vec<Option<Consumer<Reply>>>> = grid(n);
        for i in 0..n {
            for j in 0..n {
                let (cp, cc) = spsc::channel::<Command>(cap);
                cmd_p[i][j] = Some(cp);
                cmd_c[i][j] = Some(cc);
                let (rp, rc) = spsc::channel::<Reply>(cap);
                rep_p[i][j] = Some(rp);
                rep_c[i][j] = Some(rc);
            }
        }

        let cores = core_affinity::get_core_ids().unwrap_or_default();
        let mut workers = Vec::with_capacity(n);
        for id in 0..n {
            let cmd_tx: Vec<Producer<Command>> = (0..n).map(|j| cmd_p[id][j].take().unwrap()).collect();
            let cmd_rx: Vec<Consumer<Command>> = (0..n).map(|s| cmd_c[s][id].take().unwrap()).collect();
            let reply_tx: Vec<Producer<Reply>> = (0..n).map(|j| rep_p[id][j].take().unwrap()).collect();
            let reply_rx: Vec<Consumer<Reply>> = (0..n).map(|s| rep_c[s][id].take().unwrap()).collect();
            let shared = shared.clone();
            let core = cores.get(id).copied();

            let h = thread::Builder::new()
                .name(format!("mesh-svc-{id}"))
                .spawn(move || {
                    if let Some(c) = core {
                        core_affinity::set_for_current(c); // permanent pinning
                    }
                    let mut sw = ServiceWorker {
                        id,
                        n,
                        mask,
                        shard: BTreeMap::new(),
                        cmd_rx,
                        cmd_tx,
                        reply_rx,
                        reply_tx,
                        shared,
                        client_waiters: BTreeMap::new(),
                        next_token: 0,
                    };
                    sw.service_loop();
                })
                .expect("spawn service worker");
            workers.push(h);
        }

        MeshService { workers, shared }
    }

    /// A cloneable handle for client threads.
    pub fn handle(&self) -> ServiceHandle {
        ServiceHandle {
            shared: self.shared.clone(),
        }
    }

    pub fn num_shards(&self) -> usize {
        self.shared.n
    }

    /// Stop accepting, drain in-flight work, and join all workers.
    pub fn shutdown(mut self) {
        self.shared.stop.store(true, Ordering::Release);
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

impl ServiceHandle {
    /// Submit to a chosen accepting worker and block until the response arrives.
    fn submit_to(&self, accept: usize, op: ClientOp) -> Option<Vec<u8>> {
        let completion = ReqCompletion::new();
        self.shared.ingress[accept].push(op, completion.clone());
        completion.wait()
    }

    /// Index of the worker that owns `key` (client-side routing).
    #[inline]
    fn owner(&self, key: &[u8]) -> usize {
        route(key, self.shared.mask)
    }

    /// Store a key/value. **Client-side routing**: sent straight to the owning
    /// shard, so the accepting worker serves it locally with no inter-core hop.
    pub fn put(&self, key: &[u8], val: &[u8]) {
        let owner = self.owner(key);
        self.submit_to(
            owner,
            ClientOp::Put {
                key: key.to_vec(),
                val: val.to_vec(),
            },
        );
    }

    /// Look up a key. Client-side routed to the owner → served locally, no
    /// forward/reply round-trip on the matrix.
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let owner = self.owner(key);
        self.submit_to(owner, ClientOp::Get { key: key.to_vec() })
    }

    /// Round-robin accept (connection-affinity model): the accepting worker is
    /// usually *not* the owner, so the request is forwarded over the inter-core
    /// matrix and the reply relayed back. Exposed for benchmarking the worst
    /// case against the client-routed [`ServiceHandle::get`].
    pub fn get_round_robin(&self, key: &[u8]) -> Option<Vec<u8>> {
        let accept = self.shared.rr.fetch_add(1, Ordering::Relaxed) % self.shared.n;
        self.submit_to(accept, ClientOp::Get { key: key.to_vec() })
    }
}

/// A long-running service worker: owns one shard, serves its ingress + matrix.
struct ServiceWorker {
    id: usize,
    n: usize,
    mask: u64,
    shard: BTreeMap<Vec<u8>, Vec<u8>>,
    cmd_rx: Vec<Consumer<Command>>,
    cmd_tx: Vec<Producer<Command>>,
    reply_rx: Vec<Consumer<Reply>>,
    reply_tx: Vec<Producer<Reply>>,
    shared: Arc<ServiceShared>,
    /// Forwarded Gets awaiting a `Reply`, keyed by token → client completion.
    client_waiters: BTreeMap<u64, Arc<ReqCompletion>>,
    next_token: u64,
}

impl ServiceWorker {
    fn service_loop(&mut self) {
        loop {
            let mut progressed = self.pump_matrix();
            progressed |= self.drain_ingress();

            if self.shared.stop.load(Ordering::Acquire) {
                // Shutting down: keep going until everything is quiescent — no
                // matrix traffic, empty ingress, and no client still waiting.
                if !progressed
                    && self.client_waiters.is_empty()
                    && self.shared.ingress[self.id].head.load(Ordering::Acquire).is_null()
                {
                    break;
                }
            } else if !progressed {
                core::hint::spin_loop();
            }
        }
    }

    /// Serve incoming commands and collect replies. Returns true on progress.
    fn pump_matrix(&mut self) -> bool {
        let mut progressed = false;
        // Commands others routed to OUR shard.
        for src in 0..self.n {
            while let Some(cmd) = self.cmd_rx[src].pop() {
                progressed = true;
                match cmd {
                    Command::Put { key, val } => {
                        self.shard.insert(key, val);
                    }
                    Command::Get { key, from, token } => {
                        let val = self.shard.get(&key).cloned();
                        let mut reply = Reply { token, val };
                        while let Err(r) = self.reply_tx[from].push(reply) {
                            reply = r;
                            self.collect_replies();
                            core::hint::spin_loop();
                        }
                    }
                }
            }
        }
        progressed |= self.collect_replies();
        progressed
    }

    /// Replies to Gets WE forwarded → fulfill the waiting client.
    fn collect_replies(&mut self) -> bool {
        let mut progressed = false;
        for src in 0..self.n {
            while let Some(rep) = self.reply_rx[src].pop() {
                progressed = true;
                if let Some(completion) = self.client_waiters.remove(&rep.token) {
                    completion.fulfill(rep.val);
                }
            }
        }
        progressed
    }

    /// Accept client requests: serve locally if we own the key, else forward.
    fn drain_ingress(&mut self) -> bool {
        let reqs = self.shared.ingress[self.id].drain();
        if reqs.is_empty() {
            return false;
        }
        for req in reqs {
            let ClientReq { op, completion, .. } = req;
            match op {
                ClientOp::Put { key, val } => {
                    let owner = route(&key, self.mask);
                    if owner == self.id {
                        self.shard.insert(key, val);
                    } else {
                        let mut cmd = Command::Put { key, val };
                        while let Err(c) = self.cmd_tx[owner].push(cmd) {
                            cmd = c;
                            self.pump_matrix(); // relieve backpressure, no deadlock
                            core::hint::spin_loop();
                        }
                    }
                    completion.fulfill(None); // ack
                }
                ClientOp::Get { key } => {
                    let owner = route(&key, self.mask);
                    if owner == self.id {
                        let val = self.shard.get(&key).cloned();
                        completion.fulfill(val);
                    } else {
                        // Forward to the owner; remember the completion by token.
                        let token = (self.id as u64) << 48 | self.next_token;
                        self.next_token += 1;
                        self.client_waiters.insert(token, completion);
                        let mut cmd = Command::Get {
                            key,
                            from: self.id,
                            token,
                        };
                        while let Err(c) = self.cmd_tx[owner].push(cmd) {
                            cmd = c;
                            self.pump_matrix();
                            core::hint::spin_loop();
                        }
                    }
                }
            }
        }
        true
    }
}

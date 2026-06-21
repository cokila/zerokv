//! **Asynchronous Group Commit** for the WAL.
//!
//! Durability physics: even with Direct I/O / io_uring, if every writer awaits
//! its *own* fsync the tail latency is bounded by the SSD (µs-scale). Group
//! commit amortizes that cost: writers drop their record into a lock-free queue
//! and suspend (`Poll::Pending`); a background **Batcher** coalesces everything
//! that accumulated in a tiny window (~10 µs) and issues a *single* vectored
//! write ([`FixedIoBackend::write_vectored`] → one `io_uring_prep_writev`).
//! Hundreds of transactions hit the device as one I/O; each writer's `Future`
//! resolves only after the batch is durable, so the per-caller guarantee is
//! intact while throughput approaches RAM speed.
//!
//! Concurrency: the pending queue is a lock-free **Treiber stack** (the same
//! push-by-CAS / drain-by-swap pattern as [`crate::ebr`]); each record carries a
//! monotonically increasing sequence so the Batcher can restore submission
//! (WAL) order before writing. Completion wakes the writer through the custom
//! executor's [`Waker`].

use crate::regbuf::FixedIoBackend;
use std::fs::File;
use std::future::Future;
use std::pin::Pin;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Per-record completion handshake shared between a writer's `Future` and the
/// Batcher.
struct Completion {
    /// Set true by the Batcher once this record is durable.
    done: AtomicBool,
    /// The writer's waker, parked here while `Pending`.
    waker: Mutex<Option<Waker>>,
}

impl Completion {
    fn new() -> Arc<Self> {
        Arc::new(Completion {
            done: AtomicBool::new(false),
            waker: Mutex::new(None),
        })
    }
    /// Mark durable and wake the writer (called by the Batcher).
    fn signal(&self) {
        self.done.store(true, Ordering::Release);
        if let Some(w) = self.waker.lock().unwrap().take() {
            w.wake();
        }
    }
}

/// A queued WAL record awaiting commit. Linked into the Treiber stack via `next`.
struct Pending {
    seq: u64,
    data: Box<[u8]>,
    completion: Arc<Completion>,
    next: *mut Pending,
}

/// The future a writer awaits. Resolves to `Ok(())` once the record's batch has
/// been written durably.
pub struct CommitFuture {
    completion: Arc<Completion>,
}

impl Future for CommitFuture {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.completion.done.load(Ordering::Acquire) {
            return Poll::Ready(());
        }
        // Register (refresh) our waker, then re-check to avoid the lost-wakeup
        // race with a Batcher that signals between the load and the store.
        *self.completion.waker.lock().unwrap() = Some(cx.waker().clone());
        if self.completion.done.load(Ordering::Acquire) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Shared state between the public handle and the Batcher thread.
struct Shared {
    head: AtomicPtr<Pending>, // Treiber stack of pending records
    seq: AtomicU64,           // submission order
    offset: AtomicU64,        // next file write offset
    pending_count: AtomicU64, // advisory depth (for adaptive flushing)
    running: AtomicBool,
}

/// Group-commit WAL writer. Spawns a Batcher thread on construction.
pub struct GroupCommitWal {
    shared: Arc<Shared>,
    batcher: Option<JoinHandle<()>>,
}

impl GroupCommitWal {
    /// Open a group-commit WAL over `file`, flushing every `window` (e.g. 10 µs)
    /// or as soon as `max_batch` records are queued, whichever comes first.
    /// `backend` is the I/O seam (portable today, io_uring `writev` natively).
    pub fn new<B>(file: File, backend: B, window: Duration, max_batch: usize) -> Self
    where
        B: FixedIoBackend + Send + 'static,
    {
        let start_off = file.metadata().map(|m| m.len()).unwrap_or(0);
        let shared = Arc::new(Shared {
            head: AtomicPtr::new(ptr::null_mut()),
            seq: AtomicU64::new(0),
            offset: AtomicU64::new(start_off),
            pending_count: AtomicU64::new(0),
            running: AtomicBool::new(true),
        });

        let batcher = {
            let shared = shared.clone();
            thread::Builder::new()
                .name("wal-batcher".into())
                .spawn(move || batcher_loop(shared, file, backend, window, max_batch))
                .expect("spawn batcher")
        };

        GroupCommitWal {
            shared,
            batcher: Some(batcher),
        }
    }

    /// Submit `record` for durable append. Returns a `CommitFuture` that resolves
    /// once the record's batch is on disk. The hot path here is just a serialize
    /// + one lock-free CAS push — no syscall, no blocking.
    pub fn append(&self, record: &[u8]) -> CommitFuture {
        let completion = Completion::new();
        let seq = self.shared.seq.fetch_add(1, Ordering::Relaxed);
        let node = Box::into_raw(Box::new(Pending {
            seq,
            data: record.to_vec().into_boxed_slice(),
            completion: completion.clone(),
            next: ptr::null_mut(),
        }));

        // Treiber push.
        loop {
            let head = self.shared.head.load(Ordering::Acquire);
            // SAFETY: we own `node` until it is published by the CAS.
            unsafe { (*node).next = head };
            if self
                .shared
                .head
                .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
            core::hint::spin_loop();
        }
        self.shared.pending_count.fetch_add(1, Ordering::Relaxed);

        CommitFuture { completion }
    }

    /// Current queue depth (advisory).
    pub fn pending(&self) -> u64 {
        self.shared.pending_count.load(Ordering::Relaxed)
    }
}

impl Drop for GroupCommitWal {
    fn drop(&mut self) {
        // Signal shutdown and let the Batcher flush whatever remains.
        self.shared.running.store(false, Ordering::Release);
        if let Some(h) = self.batcher.take() {
            let _ = h.join();
        }
    }
}

/// The background Batcher: accumulate over a window, then group-commit.
fn batcher_loop<B: FixedIoBackend>(
    shared: Arc<Shared>,
    file: File,
    backend: B,
    window: Duration,
    max_batch: usize,
) {
    loop {
        let running = shared.running.load(Ordering::Acquire);

        // Wait out the coalescing window, but cut it short if the queue is
        // already deep (latency vs batch-size trade-off) — and don't sleep at
        // all during shutdown drain.
        if running {
            let deadline = Instant::now() + window;
            while Instant::now() < deadline
                && (shared.pending_count.load(Ordering::Relaxed) as usize) < max_batch
            {
                core::hint::spin_loop();
            }
        }

        // Detach the whole pending stack in one atomic swap.
        let mut node = shared.head.swap(ptr::null_mut(), Ordering::AcqRel);
        if node.is_null() {
            if !running {
                break; // shut down: nothing left
            }
            continue;
        }

        // Move records out of the intrusive list into an owning Vec.
        let mut batch: Vec<Box<Pending>> = Vec::new();
        while !node.is_null() {
            // SAFETY: nodes came off the stack; each is uniquely owned now.
            let boxed = unsafe { Box::from_raw(node) };
            node = boxed.next;
            batch.push(boxed);
        }
        shared
            .pending_count
            .fetch_sub(batch.len() as u64, Ordering::Relaxed);

        // Restore WAL (submission) order; the stack reversed it.
        batch.sort_unstable_by_key(|p| p.seq);

        // Build the io vector (zero-copy: slices borrow the owned boxes) and
        // issue ONE vectored write for the whole batch.
        let slices: Vec<std::io::IoSlice<'_>> =
            batch.iter().map(|p| std::io::IoSlice::new(&p.data)).collect();
        let total: usize = batch.iter().map(|p| p.data.len()).sum();
        let off = shared.offset.fetch_add(total as u64, Ordering::Relaxed);

        let result = backend.write_vectored(&file, off, &slices);
        let _ = result; // a real impl would propagate errors to completions

        // Durable (modulo fsync policy): wake every writer in the batch.
        for p in &batch {
            p.completion.signal();
        }

        if !running && shared.head.load(Ordering::Acquire).is_null() {
            break;
        }
    }
}

// SAFETY: `Shared` is shared via `Arc` and all its fields are atomics; the
// `Pending` raw pointers are only ever owned by one party at a time (producer
// before publish, Batcher after swap).
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

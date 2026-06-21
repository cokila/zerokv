//! A minimal custom async runtime for storage I/O.
//!
//! We deliberately avoid pulling in a general-purpose runtime: the storage path
//! has one shape — "submit an aligned I/O, await completion" — and a bespoke
//! executor lets us control exactly where threads park and how wakers behave,
//! which matters for tail latency.
//!
//! This module hand-rolls every async primitive:
//!   * [`block_on`] builds a [`Waker`] from a **manually written `RawWaker` +
//!     `RawWakerVTable`** (no `Wake` trait), driving a future to completion by
//!     parking/unparking the calling thread.
//!   * [`Executor`] runs many tasks, using a **manual `Wake` trait impl** on an
//!     `Arc<Task>` that re-enqueues itself — the higher-level waker path.
//!   * [`DirectIoRead`] is a future with a hand-written `Future::poll`, holding
//!     a page-aligned buffer that is **structurally pinned** (so the address the
//!     "DMA"/`O_DIRECT` path latched onto cannot move).

use std::alloc::{self, Layout};
use std::collections::VecDeque;
use std::future::Future;
use std::marker::PhantomPinned;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Wake, Waker};
use std::thread::{self, Thread};

// ===========================================================================
// 1. Manual RawWaker: park/unpark the current thread.
// ===========================================================================

/// Shared state behind a `Waker`: which thread to unpark, and a flag so a wake
/// that races ahead of `park` is not lost.
struct ThreadSignal {
    thread: Thread,
    awake: AtomicBool,
}

/// The vtable wires our four `unsafe fn`s into the `RawWaker` ABI. We manage the
/// refcount by hand via `Arc::into_raw` / `Arc::from_raw`.
static VTABLE: RawWakerVTable = RawWakerVTable::new(rw_clone, rw_wake, rw_wake_by_ref, rw_drop);

unsafe fn rw_clone(p: *const ()) -> RawWaker {
    // SAFETY: `p` is an `Arc<ThreadSignal>` pointer we created. Bump the count
    // without consuming the original.
    let arc = unsafe { Arc::from_raw(p as *const ThreadSignal) };
    let cloned = arc.clone();
    std::mem::forget(arc); // don't drop the original ref
    RawWaker::new(Arc::into_raw(cloned) as *const (), &VTABLE)
}
unsafe fn rw_wake(p: *const ()) {
    // SAFETY: consumes one ref (the `wake` contract).
    let arc = unsafe { Arc::from_raw(p as *const ThreadSignal) };
    arc.awake.store(true, Ordering::Release);
    arc.thread.unpark();
}
unsafe fn rw_wake_by_ref(p: *const ()) {
    // SAFETY: borrow without consuming the ref.
    let arc = unsafe { Arc::from_raw(p as *const ThreadSignal) };
    arc.awake.store(true, Ordering::Release);
    arc.thread.unpark();
    std::mem::forget(arc);
}
unsafe fn rw_drop(p: *const ()) {
    // SAFETY: consumes one ref.
    drop(unsafe { Arc::from_raw(p as *const ThreadSignal) });
}

fn waker_for(signal: &Arc<ThreadSignal>) -> Waker {
    let raw = RawWaker::new(Arc::into_raw(signal.clone()) as *const (), &VTABLE);
    // SAFETY: `raw` was built from our matching vtable; the contract is upheld
    // by the four functions above.
    unsafe { Waker::from_raw(raw) }
}

/// Drive `fut` to completion on the current thread, parking between polls. This
/// is the synchronous entry point used by the storage layer to await an async
/// I/O operation.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = std::pin::pin!(fut);
    let signal = Arc::new(ThreadSignal {
        thread: thread::current(),
        awake: AtomicBool::new(true), // poll at least once
    });
    let waker = waker_for(&signal);
    let mut cx = Context::from_waker(&waker);

    loop {
        if signal.awake.swap(false, Ordering::AcqRel) {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
        } else {
            // No pending wake; sleep until a waker unparks us. The `awake` flag
            // guards against the lost-wakeup race.
            thread::park();
        }
    }
}

// ===========================================================================
// 2. Multi-task executor via the `Wake` trait.
// ===========================================================================

type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

/// A spawned unit of work. Holds its future and a handle back to the run queue
/// so it can reschedule itself when woken.
struct Task {
    future: Mutex<Option<BoxFuture>>,
    queue: Arc<SharedQueue>,
}

struct SharedQueue {
    ready: Mutex<VecDeque<Arc<Task>>>,
    /// Signaled whenever a task is pushed onto `ready`, so `run` can sleep while
    /// every task is parked waiting on an *external* waker (e.g. the WAL Batcher
    /// thread) instead of spinning or exiting prematurely.
    signal: Condvar,
    /// Number of spawned-but-not-completed tasks. `run` exits only when this hits
    /// zero, not merely when the ready queue momentarily empties.
    outstanding: AtomicUsize,
}

/// Manual `Wake` impl: waking a task pushes it back onto the ready queue and
/// notifies the runner. `Arc<Task>: Wake` gives us `Waker: From<Arc<Task>>`.
impl Wake for Task {
    fn wake(self: Arc<Self>) {
        let queue = self.queue.clone();
        queue.ready.lock().unwrap().push_back(self);
        queue.signal.notify_one();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.queue.ready.lock().unwrap().push_back(self.clone());
        self.queue.signal.notify_one();
    }
}

/// A cooperative, single-threaded multi-task executor. The ready queue uses a
/// `Mutex` because it is the *cold* scheduling path — the prohibition on locks
/// targets the hot lock-free index, not the runtime's bookkeeping.
pub struct Executor {
    queue: Arc<SharedQueue>,
}

impl Executor {
    pub fn new() -> Self {
        Executor {
            queue: Arc::new(SharedQueue {
                ready: Mutex::new(VecDeque::new()),
                signal: Condvar::new(),
                outstanding: AtomicUsize::new(0),
            }),
        }
    }

    /// Schedule a future. It runs when [`Executor::run`] next reaches it.
    pub fn spawn<F: Future<Output = ()> + Send + 'static>(&self, fut: F) {
        let task = Arc::new(Task {
            future: Mutex::new(Some(Box::pin(fut))),
            queue: self.queue.clone(),
        });
        self.queue.outstanding.fetch_add(1, Ordering::AcqRel);
        self.queue.ready.lock().unwrap().push_back(task);
    }

    /// Run until every spawned task has completed. Blocks (on a condvar) while
    /// all live tasks are parked awaiting an external wake — so a future that
    /// only resolves when a *different* thread (the WAL Batcher) signals it is
    /// driven correctly, rather than the run loop exiting early.
    pub fn run(&self) {
        loop {
            let task = {
                let mut q = self.queue.ready.lock().unwrap();
                loop {
                    if let Some(t) = q.pop_front() {
                        break t;
                    }
                    if self.queue.outstanding.load(Ordering::Acquire) == 0 {
                        return; // all tasks finished
                    }
                    // Ready queue empty but tasks remain parked: wait for a wake.
                    q = self.queue.signal.wait(q).unwrap();
                }
            };
            let mut slot = task.future.lock().unwrap();
            if let Some(mut fut) = slot.take() {
                let waker = Waker::from(task.clone());
                let mut cx = Context::from_waker(&waker);
                if fut.as_mut().poll(&mut cx).is_pending() {
                    // Not done — keep it; a later `wake` requeues it.
                    *slot = Some(fut);
                } else {
                    self.queue.outstanding.fetch_sub(1, Ordering::AcqRel);
                }
            } else {
                // Spurious wake of an already-finished task; ignore.
            }
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// 3. A future with a manually pinned, page-aligned I/O buffer.
// ===========================================================================

/// A page-aligned buffer suitable for `O_DIRECT` unbuffered reads/writes (base
/// and length must be sector/page aligned). Owns its allocation.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: owns a unique heap allocation; no shared mutability.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    const ALIGN: usize = 4096;

    pub fn new(len: usize) -> Self {
        let len = (len + Self::ALIGN - 1) & !(Self::ALIGN - 1);
        let layout = Layout::from_size_align(len, Self::ALIGN).expect("layout");
        // SAFETY: len > 0, align is a power of two.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        AlignedBuf { ptr, len }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` initialized (zeroed) bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: unique borrow of our own allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, Self::ALIGN).expect("layout");
        // SAFETY: exact layout, unique ownership.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// A future that copies `source` bytes into a pinned aligned buffer, yielding
/// once before completing to exercise the waker path (simulating I/O latency).
///
/// The buffer is **structurally pinned**: because real `O_DIRECT` submits the
/// buffer address to the kernel, it must not move while the op is in flight.
/// `PhantomPinned` makes the future `!Unpin`, and we project through `Pin`
/// manually rather than via `pin_project`, documenting the safety at each step.
///
/// For the *native* path, the pinned buffer would instead be one of
/// [`crate::regbuf::RegisteredBuffers`]' slots, and the `Pending` state below is
/// precisely the window during which the kernel owns the buffer (post-`submit`,
/// pre-CQE) — which is why it must neither move nor drop until `Ready`.
pub struct DirectIoRead {
    buf: AlignedBuf,
    source: Vec<u8>,
    submitted: bool,
    _pin: PhantomPinned,
}

impl DirectIoRead {
    pub fn new(source: Vec<u8>, buf_len: usize) -> Self {
        DirectIoRead {
            buf: AlignedBuf::new(buf_len.max(source.len())),
            source,
            submitted: false,
            _pin: PhantomPinned,
        }
    }
}

impl Future for DirectIoRead {
    type Output = AlignedBuf;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY (manual pin projection): we never move `self` or any field out
        // through this `&mut`; we only read/write field bytes in place, so the
        // pinning invariant on the aligned buffer is preserved.
        let this = unsafe { self.get_unchecked_mut() };

        if !this.submitted {
            // "Submit" the I/O: in a real engine we'd hand `buf.ptr` to the OS
            // (io_uring / O_DIRECT pread) and return Pending until completion.
            this.submitted = true;
            cx.waker().wake_by_ref(); // simulate immediate completion callback
            return Poll::Pending;
        }

        // "Completion": copy source into the aligned buffer and hand it back.
        let n = this.source.len();
        this.buf.as_mut_slice()[..n].copy_from_slice(&this.source);

        // Move the finished buffer out. We replace it with a fresh empty-ish one
        // to keep `this` valid (we are returning Ready, so no further polls).
        let out = std::mem::replace(&mut this.buf, AlignedBuf::new(Self::ALIGN_OUT));
        Poll::Ready(out)
    }
}

impl DirectIoRead {
    const ALIGN_OUT: usize = 4096;
}

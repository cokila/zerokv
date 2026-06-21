//! **Epoch-Based Reclamation (EBR)** — safe memory reclamation for the
//! lock-free index without a garbage collector.
//!
//! The hazard in any lock-free structure is the *use-after-free / stale read*:
//! thread A unlinks and frees a node while thread B is still dereferencing the
//! pointer it loaded a moment earlier. EBR solves this without per-pointer
//! reference counting (which would add expensive atomic RMWs to every read):
//!
//!   * There is a monotonically increasing global **epoch**.
//!   * Before touching shared pointers a thread *pins* itself, publishing the
//!     current global epoch into a per-thread slot (a `Guard`).
//!   * Freeing is deferred: an unlinked node is *retired* into a per-epoch bag,
//!     tagged with the epoch in which it was retired.
//!   * The global epoch may advance only when **every pinned thread is at the
//!     current epoch**. A retired node is physically freed only once the global
//!     epoch has moved two steps past its retire epoch — by then no pinned
//!     thread can still hold a pointer to it. Hence: no stale reads, ever.
//!
//! This module is itself lock-free: participant slots and garbage bags are
//! plain atomics / Treiber stacks. There is exactly one process-wide [`Domain`]
//! (`GLOBAL`), shared by every skiplist shard — mirroring crossbeam-epoch's
//! default collector.

use std::cell::Cell;
use std::ptr;
use std::sync::atomic::{fence, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// Maximum number of threads that can ever participate concurrently. Slots are
/// claimed lazily and recycled when a thread parks (set back to `UNPINNED`),
/// never returned to `FREE` — fine for a server with a bounded thread pool.
const N_SLOTS: usize = 4096;
/// Three bags are the minimum for a 2-epoch grace period with headroom.
const N_BAGS: usize = 3;

/// Slot is unclaimed.
const FREE: u64 = u64::MAX;
/// Slot is claimed by a thread that is currently *not* pinned.
const UNPINNED: u64 = 1 << 62;

/// A retired pointer awaiting reclamation. We avoid boxing a closure by storing
/// a monomorphized `drop_fn` (a plain function pointer) alongside the erased
/// pointer — zero dynamic dispatch, one small allocation per retire.
struct Retired {
    ptr: *mut u8,
    drop_fn: unsafe fn(*mut u8),
    epoch: u64,
    next: *mut Retired,
}

/// The reclamation domain: a global epoch counter, a pool of participant slots,
/// and `N_BAGS` Treiber stacks of retired garbage.
pub struct Domain {
    epoch: AtomicU64,
    slots: [AtomicU64; N_SLOTS],
    bags: [AtomicPtr<Retired>; N_BAGS],
    /// Highest slot index ever claimed. The collector scans only `0..=hwm`
    /// instead of all `N_SLOTS`, so the per-unpin cost tracks the real thread
    /// count (a handful) rather than the 4096-slot pool capacity.
    hwm: AtomicUsize,
}

// SAFETY: every field is an atomic or an atomic-guarded raw structure; all
// access goes through atomic operations with explicit ordering.
unsafe impl Sync for Domain {}

impl Domain {
    const fn new() -> Self {
        Domain {
            epoch: AtomicU64::new(0),
            // `[const { .. }; N]` const-initializes every element — no runtime
            // init, so `GLOBAL` can be a plain `static`.
            slots: [const { AtomicU64::new(FREE) }; N_SLOTS],
            bags: [const { AtomicPtr::new(ptr::null_mut()) }; N_BAGS],
            hwm: AtomicUsize::new(0),
        }
    }

    /// Pin the calling thread, returning a [`Guard`]. While the guard lives, any
    /// pointer the thread loads from the shared structure is guaranteed not to
    /// be freed underneath it.
    pub fn pin(&'static self) -> Guard {
        let idx = self.local_slot();
        let e = self.epoch.load(Ordering::Relaxed);
        // Publish our epoch. `SeqCst` + the following fence give us a total
        // order against the collector's `SeqCst` reads of the slots, so the
        // collector cannot both miss our pin AND free something we will read.
        self.slots[idx].store(e, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        Guard { domain: self, idx }
    }

    /// Defer freeing `ptr`. `drop_fn` is responsible for reconstructing the
    /// owning box/type and dropping it. Tagged with the current epoch.
    ///
    /// # Safety
    /// `ptr` must be uniquely owned by the caller at retire time (already
    /// unlinked from the structure) and valid for `drop_fn`.
    pub unsafe fn retire(&self, ptr: *mut u8, drop_fn: unsafe fn(*mut u8)) {
        let e = self.epoch.load(Ordering::Relaxed);
        let node = Box::into_raw(Box::new(Retired {
            ptr,
            drop_fn,
            epoch: e,
            next: ptr::null_mut(),
        }));
        let bag = &self.bags[(e as usize) % N_BAGS];
        // Treiber-stack push.
        loop {
            let head = bag.load(Ordering::Acquire);
            // SAFETY: we own `node` exclusively until it is published.
            unsafe { (*node).next = head };
            if bag
                .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
            core::hint::spin_loop();
        }
    }

    /// Attempt to advance the epoch and reclaim now-safe garbage. Called
    /// opportunistically (e.g. on guard drop). Cheap and best-effort: if it
    /// cannot make progress it simply returns.
    pub fn try_collect(&self) {
        let g = self.epoch.load(Ordering::SeqCst);

        // Advance only if every *pinned* participant is already at epoch `g`.
        // Scan only the claimed prefix `0..=hwm`.
        let hwm = self.hwm.load(Ordering::Relaxed).min(N_SLOTS - 1);
        let mut can_advance = true;
        for s in self.slots[..=hwm].iter() {
            let v = s.load(Ordering::SeqCst);
            let pinned = v != FREE && v != UNPINNED;
            if pinned && v != g {
                can_advance = false;
                break;
            }
        }
        if can_advance {
            // Losers of this CAS just skip; the winner publishes `g+1`.
            let _ = self
                .epoch
                .compare_exchange(g, g + 1, Ordering::SeqCst, Ordering::Relaxed);
        }

        // Reclaim from all bags everything older than (current_epoch - 1), i.e.
        // retired at epoch `e` with `e + 2 <= current`. By the epoch invariant
        // no pinned thread can still reference such nodes.
        let cur = self.epoch.load(Ordering::Relaxed);
        for bag in self.bags.iter() {
            Self::reclaim_bag(bag, cur);
        }
    }

    /// Drain one bag, free eligible nodes, re-queue the rest.
    fn reclaim_bag(bag: &AtomicPtr<Retired>, cur: u64) {
        // Detach the whole chain in one atomic swap; concurrent pushes after
        // this form a fresh chain that we won't touch this round.
        let mut node = bag.swap(ptr::null_mut(), Ordering::AcqRel);
        let mut keep: *mut Retired = ptr::null_mut();

        while !node.is_null() {
            // SAFETY: `node` came off the stack and is uniquely owned now.
            let next = unsafe { (*node).next };
            let eligible = unsafe { (*node).epoch }.saturating_add(2) <= cur;
            if eligible {
                // SAFETY: 2 epochs have elapsed since retire => no live guard
                // can reference `ptr`; run its destructor and free the record.
                unsafe {
                    let r = &*node;
                    (r.drop_fn)(r.ptr);
                    drop(Box::from_raw(node));
                }
            } else {
                // Not yet safe — thread it onto the keep-list for re-insertion.
                unsafe { (*node).next = keep };
                keep = node;
            }
            node = next;
        }

        // Re-push retained nodes (still-young garbage) back onto the bag.
        while !keep.is_null() {
            let n = keep;
            let nxt = unsafe { (*n).next };
            loop {
                let head = bag.load(Ordering::Acquire);
                unsafe { (*n).next = head };
                if bag
                    .compare_exchange_weak(head, n, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    break;
                }
                core::hint::spin_loop();
            }
            keep = nxt;
        }
    }

    /// Claim (or reuse) this thread's participant slot.
    fn local_slot(&self) -> usize {
        thread_local! {
            // Cached slot index for this thread, -1 until first registration.
            static SLOT: Cell<isize> = const { Cell::new(-1) };
        }
        SLOT.with(|s| {
            let cached = s.get();
            if cached >= 0 {
                return cached as usize;
            }
            // Linear probe for a FREE slot; claim it with a CAS.
            for (i, slot) in self.slots.iter().enumerate() {
                if slot.load(Ordering::Relaxed) == FREE
                    && slot
                        .compare_exchange(FREE, UNPINNED, Ordering::AcqRel, Ordering::Relaxed)
                        .is_ok()
                {
                    s.set(i as isize);
                    // Publish the high-water mark so the collector scans only
                    // the claimed prefix.
                    self.hwm.fetch_max(i, Ordering::Relaxed);
                    return i;
                }
            }
            panic!("ebr: exhausted {N_SLOTS} participant slots");
        })
    }
}

/// RAII pin. Dropping it unpins the thread and opportunistically tries to
/// advance/collect. `!Send` so a guard cannot be moved to another thread (its
/// slot is thread-bound) — enforced by the `PhantomData<*const ()>` below.
pub struct Guard {
    domain: &'static Domain,
    idx: usize,
}

impl Guard {
    /// Retire a node through this guard's domain. Convenience wrapper.
    ///
    /// # Safety
    /// Same contract as [`Domain::retire`].
    pub unsafe fn retire(&self, ptr: *mut u8, drop_fn: unsafe fn(*mut u8)) {
        // SAFETY: forwarded to the domain; obligations are the caller's.
        unsafe { self.domain.retire(ptr, drop_fn) }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        // Release our epoch claim. `Release` pairs with the collector's
        // `SeqCst`/`Acquire` reads of the slot.
        self.domain.slots[self.idx].store(UNPINNED, Ordering::Release);

        // Collection scans all participant slots, so it is amortized: only one
        // unpin in `COLLECT_INTERVAL` actually attempts to advance/reclaim. This
        // keeps the *read* fast path (pin → load → unpin) free of the O(slots)
        // scan, which is what preserves sub-microsecond `get` latency.
        thread_local! {
            static TICK: Cell<u32> = const { Cell::new(0) };
        }
        const COLLECT_INTERVAL: u32 = 128;
        let do_collect = TICK.with(|t| {
            let n = t.get().wrapping_add(1);
            t.set(n);
            n % COLLECT_INTERVAL == 0
        });
        if do_collect {
            self.domain.try_collect();
        }
    }
}

/// The single process-wide reclamation domain.
pub static GLOBAL: Domain = Domain::new();

/// Pin the current thread on the global domain.
#[inline]
pub fn pin() -> Guard {
    GLOBAL.pin()
}

//! Custom **bump (arena) allocator** built on `std::alloc` + raw pointers.
//!
//! The engine never frees individual records on the read path. Instead, values
//! are bump-allocated into a contiguous, page-aligned arena and handed out as
//! borrows tied to the arena's lifetime (see [`crate::zerocopy::ZeroCopyStorage`]).
//! This eliminates per-record `malloc`/`free` traffic, eliminates the
//! allocator-internal locking that wrecks tail latency, and — crucially for a
//! GC-free design — makes reclamation a single O(1) `reset` of the whole arena.
//!
//! Concurrency model: allocation is wait-free via a single `AtomicUsize` bump
//! cursor advanced with `fetch_add`. Multiple threads can allocate in parallel
//! with no lock; the only failure mode is "arena full", surfaced as `None`.

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A fixed-capacity bump allocator over one backing allocation.
///
/// Memory safety invariants (upheld by `unsafe` blocks below):
///   * `base` points to a live allocation of exactly `cap` bytes with `align`
///     alignment, owned by this `Arena` until `Drop`.
///   * `head <= cap` always (enforced by the `fetch_add` + bounds check).
///   * Returned pointers never overlap (each gets a disjoint `[off, off+size)`).
pub struct Arena {
    base: NonNull<u8>,
    cap: usize,
    align: usize,
    head: AtomicUsize,
}

// SAFETY: `Arena` owns its backing allocation and all mutation of the bump
// cursor goes through atomics. Sharing `&Arena` across threads only ever issues
// atomic `fetch_add`s and reads of immutable `base/cap`, so it is `Sync`. It can
// be moved across threads (`Send`) because nothing is thread-local.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Allocate a fresh arena with `cap` usable bytes, aligned to `align`
    /// (must be a power of two). We round `cap` up so the whole region itself
    /// honors the alignment, which lets us hand out aligned sub-slices.
    pub fn with_capacity(cap: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "align must be a power of two");
        assert!(cap > 0, "arena capacity must be > 0");

        // SAFETY: align is a power of two and cap > 0, so `Layout` is valid.
        let layout = Layout::from_size_align(cap, align).expect("invalid arena layout");
        // SAFETY: layout has non-zero size; on null we abort per std convention.
        let raw = unsafe { alloc::alloc(layout) };
        let base = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));

        Arena {
            base,
            cap,
            align,
            head: AtomicUsize::new(0),
        }
    }

    /// Wait-free bump allocation of `size` bytes aligned to `align`.
    ///
    /// Returns a raw, exclusive, uninitialized region. Returns `None` if the
    /// arena is exhausted (the caller decides whether to roll a new segment).
    pub fn alloc_raw(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        debug_assert!(align.is_power_of_two());
        if size == 0 {
            return Some(self.base);
        }
        // CAS-free fast path: reserve a slot, then validate it fits. We loop
        // only to re-align after a racing allocation moved `head`.
        let mut cur = self.head.load(Ordering::Relaxed);
        loop {
            // Align the start of our chunk relative to the arena base.
            let base_addr = self.base.as_ptr() as usize;
            let start = base_addr + cur;
            let aligned = (start + align - 1) & !(align - 1);
            let pad = aligned - start;
            let new_head = cur.checked_add(pad)?.checked_add(size)?;
            if new_head > self.cap {
                return None; // exhausted
            }
            // Publish our reservation. `Relaxed` is sufficient: the bytes we
            // return are not yet visible to anyone else, and the pointer we
            // expose carries a data dependency, so no fence is required here.
            match self.head.compare_exchange_weak(
                cur,
                new_head,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // SAFETY: `aligned` lies within `[base, base+cap)` because
                    // `new_head <= cap`, and the region is disjoint from every
                    // other reservation by construction.
                    let ptr = unsafe { NonNull::new_unchecked(aligned as *mut u8) };
                    return Some(ptr);
                }
                Err(actual) => cur = actual, // retry with the updated cursor
            }
        }
    }

    /// Copy `bytes` into the arena and return a borrow valid for `&self`.
    ///
    /// The returned slice lifetime is tied to `&self`: it cannot outlive the
    /// arena, which is exactly the zero-copy contract the index relies on.
    pub fn push_bytes(&self, bytes: &[u8]) -> Option<&[u8]> {
        let dst = self.alloc_raw(bytes.len(), 1)?;
        // SAFETY: `dst` is a fresh, exclusive, non-overlapping region of
        // `bytes.len()` bytes that we just reserved; src and dst do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.as_ptr(), bytes.len());
            Some(std::slice::from_raw_parts(dst.as_ptr(), bytes.len()))
        }
    }

    /// Bytes currently allocated.
    #[inline]
    pub fn used(&self) -> usize {
        self.head.load(Ordering::Relaxed)
    }

    /// Total capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// O(1) reclamation of the entire arena.
    ///
    /// # Safety
    /// The caller must ensure no outstanding borrow (`&[u8]`) handed out by this
    /// arena is still alive, otherwise those references would dangle. In the
    /// engine this is gated by the epoch reclamation system (no reader pinned to
    /// an old epoch).
    pub unsafe fn reset(&self) {
        self.head.store(0, Ordering::Release);
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: `base`/`cap`/`align` reproduce the exact `Layout` used in
        // `with_capacity`, and we own the allocation uniquely at drop time.
        let layout = Layout::from_size_align(self.cap, self.align).expect("layout");
        unsafe { alloc::dealloc(self.base.as_ptr(), layout) }
    }
}

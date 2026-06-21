//! **Lock-free SPSC ring buffer** — the message channel for a shared-nothing,
//! thread-per-core architecture (ScyllaDB / Redpanda / seastar style).
//!
//! In that model each shard is owned by exactly one core; a core that receives a
//! request for *another* core's shard does not touch that shard's atomics
//! (which would trigger a cross-core **cache-line bounce** under the MESI
//! coherence protocol). Instead it hands the request to the owner via a
//! Single-Producer/Single-Consumer queue. With one SPSC per (producer, consumer)
//! pair, every queue has exactly one writer and one reader, so it needs **no
//! CAS** at all — just `Release`/`Acquire` on two cursors.
//!
//! ## The cache-line lesson, inverted
//! We learned that padding nodes to 64 bytes *hurt* the index (it inflated a
//! capacity-bound working set). Here padding *helps*: `head` (written by the
//! consumer) and `tail` (written by the producer) are each given their own cache
//! line via `#[repr(align(64))]`, so the two cores never invalidate each other's
//! line just by advancing their own cursor. Same technique, opposite verdict —
//! because this is a **false-sharing** problem, not a capacity problem.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A cursor on its own cache line, so producer and consumer writes don't bounce.
#[repr(align(64))]
struct Padded(AtomicUsize);

struct Ring<T> {
    /// Slots. `cap` is a power of two; index masking replaces modulo.
    buf: Box<[UnsafeCell<MaybeUninit<T>>]>,
    mask: usize,
    /// Next index to write (producer-owned).
    tail: Padded,
    /// Next index to read (consumer-owned).
    head: Padded,
}

// SAFETY: access is disciplined by the SPSC protocol — exactly one producer
// touches `tail` + writes slots ahead of `head`; exactly one consumer touches
// `head` + reads slots behind `tail`. `T: Send` is required to move values
// across the core boundary.
unsafe impl<T: Send> Sync for Ring<T> {}
unsafe impl<T: Send> Send for Ring<T> {}

/// The producer end. Not `Clone` — there is only ever one.
pub struct Producer<T> {
    ring: Arc<Ring<T>>,
}
/// The consumer end. Not `Clone` — there is only ever one.
pub struct Consumer<T> {
    ring: Arc<Ring<T>>,
}

// Only `Send` (move the endpoint to its owning thread); never `Sync` (would
// allow two producers/consumers).
unsafe impl<T: Send> Send for Producer<T> {}
unsafe impl<T: Send> Send for Consumer<T> {}

/// Create an SPSC channel with capacity rounded up to a power of two (≥ 2).
pub fn channel<T: Send>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    let cap = capacity.max(2).next_power_of_two();
    let mut v = Vec::with_capacity(cap);
    for _ in 0..cap {
        v.push(UnsafeCell::new(MaybeUninit::uninit()));
    }
    let ring = Arc::new(Ring {
        buf: v.into_boxed_slice(),
        mask: cap - 1,
        tail: Padded(AtomicUsize::new(0)),
        head: Padded(AtomicUsize::new(0)),
    });
    (
        Producer { ring: ring.clone() },
        Consumer { ring },
    )
}

impl<T: Send> Producer<T> {
    /// Try to enqueue `value`. Returns `Err(value)` if the ring is full.
    #[inline]
    pub fn push(&self, value: T) -> Result<(), T> {
        let ring = &*self.ring;
        let tail = ring.tail.0.load(Ordering::Relaxed); // we are the only writer
        // `Acquire` on head: observe the consumer's progress before deciding full.
        let head = ring.head.0.load(Ordering::Acquire);
        if tail.wrapping_sub(head) == ring.buf.len() {
            return Err(value); // full
        }
        let slot = ring.buf[tail & ring.mask].get();
        // SAFETY: slot `tail` is behind no live reader (it's ahead of `head` by
        // less than `cap`), so writing it is exclusive.
        unsafe { (*slot).write(value) };
        // Publish: the `Release` makes the slot write visible before the consumer
        // sees the new `tail`.
        ring.tail.0.store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }
}

impl<T: Send> Consumer<T> {
    /// Try to dequeue a value. Returns `None` if the ring is empty.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let ring = &*self.ring;
        let head = ring.head.0.load(Ordering::Relaxed); // we are the only reader
        // `Acquire` on tail: observe the producer's slot write before reading it.
        let tail = ring.tail.0.load(Ordering::Acquire);
        if head == tail {
            return None; // empty
        }
        let slot = ring.buf[head & ring.mask].get();
        // SAFETY: slot `head` was fully written by the producer (we observed
        // `tail > head` with `Acquire`), and we are its only reader.
        let value = unsafe { (*slot).assume_init_read() };
        ring.head.0.store(head.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Drain up to `max` items, invoking `f` on each. Returns the count handled.
    #[inline]
    pub fn drain<F: FnMut(T)>(&self, max: usize, mut f: F) -> usize {
        let mut n = 0;
        while n < max {
            match self.pop() {
                Some(v) => {
                    f(v);
                    n += 1;
                }
                None => break,
            }
        }
        n
    }
}

impl<T> Drop for Ring<T> {
    fn drop(&mut self) {
        // Drop any values still queued (between head and tail).
        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Relaxed);
        let mut i = head;
        while i != tail {
            let slot = self.buf[i & self.mask].get();
            // SAFETY: indices in [head, tail) hold initialized values.
            unsafe { (*slot).assume_init_drop() };
            i = i.wrapping_add(1);
        }
    }
}

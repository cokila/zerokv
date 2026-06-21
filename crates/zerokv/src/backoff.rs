//! Exponential backoff with jitter for CAS retry loops.
//!
//! When many writer threads contend on the same atomic (e.g. the head array of
//! a skiplist), naively retrying a failed `compare_exchange` immediately makes
//! things *worse*: every retry issues a fresh RFO (Read-For-Ownership) on the
//! cache line, saturating the inter-core coherence bus and starving forward
//! progress. The cure is twofold:
//!
//!   * **`core::hint::spin_loop()`** — emits `PAUSE` on x86/x86-64 and `YIELD`
//!     on AArch64. It tells the CPU "I'm spin-waiting": it de-pipelines the
//!     speculative loads so a memory-order violation does not flush the
//!     pipeline, and it lowers power draw / frees SMT sibling resources.
//!
//!   * **Exponential backoff + jitter** — each failed attempt waits roughly
//!     `2^k` spin iterations, capped, *plus a random perturbation*. Jitter
//!     de-synchronizes threads that started contending in lockstep, breaking
//!     the "thundering herd" where everyone backs off and retries together.

use core::cell::Cell;

/// A cheap, allocation-free, non-cryptographic PRNG (xorshift64*) used purely
/// to spread out retry timing. Seeded per-`Backoff` from the stack address so
/// two threads spinning on the same line diverge immediately.
#[derive(Debug)]
struct Xorshift64 {
    state: Cell<u64>,
}

impl Xorshift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point of xorshift.
        Xorshift64 {
            state: Cell::new(seed | 1),
        }
    }

    #[inline]
    fn next(&self) -> u64 {
        let mut x = self.state.get();
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state.set(x);
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// The exponent at which we stop spinning and start yielding the OS thread.
/// 2^6 = 64 spins is a good crossover on modern x86 before a syscall-cheap
/// `yield_now` becomes preferable to burning the core.
const SPIN_LIMIT: u32 = 6;
/// Hard cap on the exponent so the spin count never explodes.
const YIELD_LIMIT: u32 = 10;

/// Backoff state machine. Construct once per CAS loop; call [`Backoff::spin`]
/// after each failed attempt.
pub struct Backoff {
    step: Cell<u32>,
    rng: Xorshift64,
}

impl Backoff {
    #[inline]
    pub fn new() -> Self {
        // Seed from a stack address: distinct per thread/frame, no syscall.
        let seed = (&Cell::new(0u8) as *const _ as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        Backoff {
            step: Cell::new(0),
            rng: Xorshift64::new(seed),
        }
    }

    /// Pure spin-wait phase: use inside tight lock-free loops where we expect
    /// the contended operation to complete in nanoseconds and must NOT
    /// deschedule (descheduling under a held coherence line would be a latency
    /// catastrophe). Returns `true` while still in the cheap spin regime.
    #[inline]
    pub fn spin(&self) -> bool {
        let step = self.step.get();
        // Base = 2^min(step, SPIN_LIMIT); jitter in [0, base/2).
        let base = 1u64 << step.min(SPIN_LIMIT);
        // Jitter in [0, base/2): mask with (base/2 - 1) since base is a power of
        // two and >= 1. This de-synchronizes threads that began contending in
        // lockstep, breaking the thundering herd.
        let jitter = self.rng.next() & (base.max(2) / 2 - 1);
        let iters = base + jitter;

        for _ in 0..iters {
            // PAUSE / YIELD — see module docs.
            core::hint::spin_loop();
        }

        if step <= YIELD_LIMIT {
            self.step.set(step + 1);
        }
        step <= SPIN_LIMIT
    }

    /// Cooperative phase for longer waits (e.g. waiting on another thread that
    /// may be descheduled). Falls back to `std::thread::yield_now` once we have
    /// exceeded the spin regime, surrendering the core to the scheduler.
    #[inline]
    pub fn snooze(&self) {
        if self.step.get() <= SPIN_LIMIT {
            self.spin();
        } else {
            std::thread::yield_now();
            if self.step.get() <= YIELD_LIMIT {
                self.step.set(self.step.get() + 1);
            }
        }
    }

    /// True once we have backed off enough that an OS-level park would be
    /// cheaper than continuing to spin.
    #[inline]
    pub fn is_completed(&self) -> bool {
        self.step.get() > YIELD_LIMIT
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

/// A tiny, fast, per-thread random level generator for the skiplist, sharing
/// the same xorshift core. Kept here so the skiplist module stays focused.
pub(crate) fn random_geometric_level(rng_state: &Cell<u64>, max_level: usize) -> usize {
    // Each level continues with probability 1/4 (p = 0.25). Using two random
    // bits per level halves the expected node height vs. p = 0.5, which means
    // fewer atomic CAS operations per `insert` (see Adaptive-Level rationale).
    let mut x = rng_state.get();
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state.set(x);
    let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);

    let mut level = 1usize;
    let mut bits = r;
    while level < max_level && (bits & 0b11) == 0b11 {
        level += 1;
        bits >>= 2;
    }
    level
}

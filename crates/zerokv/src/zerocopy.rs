//! Zero-copy type layer: the `Pod` marker, the `ZeroCopy` trait that the derive
//! macro implements, and the **GAT-based** [`ZeroCopyStorage`] trait that lets a
//! backing store return borrows tied to its own lifetime with no heap copy.

use crate::arena::Arena;

/// **Plain Old Data** marker. A type is `Pod` iff *every* bit pattern of its
/// size is a valid value, it is `Copy`, it owns no resources / pointers, and it
/// has no `Drop`. Reinterpreting arbitrary bytes as `&T` is only sound for such
/// types, so `Pod` is the linchpin of zero-copy reads.
///
/// It is `unsafe` to implement: the compiler cannot verify these properties, so
/// the implementor asserts them. We provide blanket impls only for types we
/// know satisfy the contract.
///
/// # Safety
/// Implementors must guarantee that `Self`:
/// * is `Copy` and contains no padding bytes;
/// * is valid for **every** bit pattern of its size (no niche/enum invariants);
/// * owns no pointers, handles, or `Drop` glue.
///
/// Violating any of these makes the zero-copy `view`/`decode` paths unsound.
pub unsafe trait Pod: Copy + 'static {}

macro_rules! impl_pod {
    ($($t:ty),* $(,)?) => { $( unsafe impl Pod for $t {} )* };
}
// Integers/floats have no invalid bit patterns and no padding.
impl_pod!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);

// Fixed-size arrays of POD are POD (no padding between equal-typed elements).
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

/// Implemented (via `#[derive(ZeroCopy)]`) for user record structs. `unsafe`
/// because `view` hands out a typed reference into untrusted bytes; the derive
/// macro discharges the obligations (POD fields + no padding) at compile time.
///
/// # Safety
/// Implementors must guarantee `Self` is `Pod`-equivalent: padding-free, valid
/// for any bit pattern of length `SERIALIZED_SIZE`.
pub unsafe trait ZeroCopy: Sized {
    /// Exact on-disk / in-arena byte size. Equals `size_of::<Self>()`.
    const SERIALIZED_SIZE: usize;
    /// Serialize `self` into `dst` (must be `>= SERIALIZED_SIZE`).
    fn encode(&self, dst: &mut [u8]);
    /// Deserialize by value (a single `memcpy`).
    fn decode(src: &[u8]) -> Self;
    /// Borrow the bytes *in place* as `&Self` if length and alignment permit —
    /// the true zero-copy read. Returns `None` on a too-short or misaligned
    /// slice rather than risking UB.
    fn view(src: &[u8]) -> Option<&Self>;
}

/// Storage abstraction that returns **borrows bound to the store's lifetime**,
/// expressed with a *Generic Associated Type* (`Ref<'a>`).
///
/// The GAT is what makes the zero-copy contract expressible in the type system:
/// `get` cannot return a `Ref<'static>` or smuggle the borrow past `&'a self`,
/// because `Ref<'a>: 'a` and the lifetime is threaded through the method. The
/// borrow checker therefore guarantees no returned reference outlives the arena
/// that owns the bytes — without any runtime cost.
pub trait ZeroCopyStorage {
    /// The borrow type, parameterized by the borrow's lifetime. Implementors
    /// typically set this to `&'a [u8]` or `&'a T`.
    type Ref<'a>: 'a
    where
        Self: 'a;

    /// Look up `key`, returning a borrow valid only while `&'a self` is held.
    fn get<'a>(&'a self, key: &[u8]) -> Option<Self::Ref<'a>>;
}

/// A minimal arena-backed store used to demonstrate the GAT contract end to end.
/// Keys are linear-scanned (this is purely a teaching vehicle for the trait —
/// the production index is the lock-free skiplist); the point is the *signature*
/// of `get`, which returns `&'a [u8]` straight out of arena memory.
pub struct ArenaKv {
    arena: Arena,
    // (key_bytes_in_arena, value_bytes_in_arena) as raw spans.
    entries: std::sync::Mutex<Vec<(*const u8, usize, *const u8, usize)>>,
}

// SAFETY: the raw pointers only ever address bytes inside `arena`, which is
// `Send + Sync` and lives as long as `self`. The `Mutex` makes the `Vec`
// mutation thread-safe; the pointers are never dereferenced mutably.
unsafe impl Send for ArenaKv {}
unsafe impl Sync for ArenaKv {}

impl ArenaKv {
    pub fn new(cap: usize) -> Self {
        ArenaKv {
            arena: Arena::with_capacity(cap, 64),
            entries: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Insert by copying both key and value into the arena once.
    pub fn put(&self, key: &[u8], val: &[u8]) -> bool {
        let k = match self.arena.push_bytes(key) {
            Some(k) => k,
            None => return false,
        };
        let v = match self.arena.push_bytes(val) {
            Some(v) => v,
            None => return false,
        };
        self.entries
            .lock()
            .unwrap()
            .push((k.as_ptr(), k.len(), v.as_ptr(), v.len()));
        true
    }
}

impl ZeroCopyStorage for ArenaKv {
    // The crux: the associated borrow type is a slice of arena bytes whose
    // lifetime is exactly that of `&self`.
    type Ref<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn get<'a>(&'a self, key: &[u8]) -> Option<Self::Ref<'a>> {
        let entries = self.entries.lock().unwrap();
        for &(kp, kl, vp, vl) in entries.iter() {
            // SAFETY: (kp,kl) spans live arena bytes; the borrow we materialize
            // is immediately compared and dropped before the lock is released.
            let k = unsafe { std::slice::from_raw_parts(kp, kl) };
            if k == key {
                // SAFETY: (vp,vl) spans live arena bytes owned by `self.arena`,
                // which outlives `'a`. We launder the lifetime to `'a` — sound
                // because the arena is never reset while `&'a self` is held.
                let v: &'a [u8] = unsafe { std::slice::from_raw_parts(vp, vl) };
                return Some(v);
            }
        }
        None
    }
}

//! **Sharded index**: `K` independent lock-free skiplists keyed by a hash of
//! the key. A single global skiplist serializes writers on its head array under
//! mixed read/write load; partitioning by `hash(key) % K` cuts the probability
//! of two writers colliding on the same head pointers by a factor of ~`K`,
//! while reads pay only an O(1), fully-predicted hash + route before the usual
//! sub-100 ns skiplist descent.
//!
//! `K` must be a power of two so routing is a mask, not a division.

use crate::ebr::{pin, Guard};
use crate::skiplist::SkipList;

/// FNV-1a — a fast, allocation-free, well-dispersing non-crypto hash. Good
/// enough to balance shards; key distribution, not collision resistance, is the
/// goal.
#[inline]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// A partitioned ordered key-value index.
pub struct ShardedIndex {
    shards: Box<[SkipList]>,
    mask: u64,
}

impl ShardedIndex {
    /// `num_shards` is rounded up to a power of two. `capacity_hint` is the
    /// *total* expected item count; per-shard height adapts to `hint / K`.
    pub fn new(num_shards: usize, capacity_hint: usize) -> Self {
        let k = num_shards.max(1).next_power_of_two();
        let per_shard = (capacity_hint / k).max(2);
        let mut v = Vec::with_capacity(k);
        for _ in 0..k {
            v.push(SkipList::with_capacity_hint(per_shard));
        }
        ShardedIndex {
            shards: v.into_boxed_slice(),
            mask: (k as u64) - 1,
        }
    }

    #[inline]
    fn shard(&self, key: &[u8]) -> &SkipList {
        // `& mask` because `len` is a power of two.
        &self.shards[(fnv1a(key) & self.mask) as usize]
    }

    /// Insert/update. Pins an epoch guard for the duration so any value cell we
    /// displace stays alive for in-flight readers.
    pub fn insert(&self, key: &[u8], val: &[u8]) -> bool {
        let guard = pin();
        self.shard(key).insert(key, val, &guard)
    }

    /// Lookup with a caller-managed guard, enabling true zero-copy: the returned
    /// borrow is valid exactly while `guard` is held.
    pub fn get<'g>(&self, key: &[u8], guard: &'g Guard) -> Option<&'g [u8]> {
        self.shard(key).get(key, guard)
    }

    /// Convenience lookup that copies the value out (guard is internal).
    pub fn get_owned(&self, key: &[u8]) -> Option<Vec<u8>> {
        let guard = pin();
        self.shard(key).get(key, &guard).map(|b| b.to_vec())
    }

    pub fn remove(&self, key: &[u8]) -> bool {
        let guard = pin();
        self.shard(key).remove(key, &guard)
    }

    /// Approximate total length across shards.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }
}

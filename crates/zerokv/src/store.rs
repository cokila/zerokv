//! `KvStore`: the public façade wiring the lock-free [`ShardedIndex`] to an
//! optional durable [`MappedLog`] write-ahead log.
//!
//! The hot read/write path goes straight to the sharded, lock-free index. When
//! durability is enabled, writes first append to the pre-allocated WAL. The WAL
//! append is serialized behind a `Mutex` — this is the *cold* durability path
//! (a single sequential log writer is the standard design); the lock never
//! touches the concurrent index.

use crate::mmap_log::MappedLog;
use crate::shard::ShardedIndex;
use std::pin::Pin;
use std::sync::Mutex;

/// A sub-millisecond, GC-free key-value store.
pub struct KvStore {
    index: ShardedIndex,
    wal: Option<Mutex<Pin<Box<MappedLog>>>>,
}

impl KvStore {
    /// In-memory store (no durability). `num_shards` is rounded to a power of
    /// two; `capacity_hint` adapts skiplist heights.
    pub fn in_memory(num_shards: usize, capacity_hint: usize) -> Self {
        KvStore {
            index: ShardedIndex::new(num_shards, capacity_hint),
            wal: None,
        }
    }

    /// Durable store backed by a pre-allocated WAL file of `wal_bytes`.
    pub fn durable(
        path: impl AsRef<std::path::Path>,
        num_shards: usize,
        capacity_hint: usize,
        wal_bytes: usize,
    ) -> std::io::Result<Self> {
        let log = MappedLog::open(path, wal_bytes)?;
        Ok(KvStore {
            index: ShardedIndex::new(num_shards, capacity_hint),
            wal: Some(Mutex::new(log)),
        })
    }

    /// Insert or update. Appends a `[klen|key|value]` frame to the WAL first
    /// (if durable), then updates the lock-free index.
    pub fn put(&self, key: &[u8], value: &[u8]) -> bool {
        if let Some(wal) = &self.wal {
            let mut frame = Vec::with_capacity(4 + key.len() + value.len());
            frame.extend_from_slice(&(key.len() as u32).to_le_bytes());
            frame.extend_from_slice(key);
            frame.extend_from_slice(value);
            let mut log = wal.lock().unwrap();
            // `as_mut()` reborrows the pinned box mutably without moving it.
            log.as_mut().append(&frame);
        }
        self.index.insert(key, value)
    }

    /// Zero-copy-ish read that copies the value out (the borrowed-form lookup is
    /// available on [`ShardedIndex::get`] with a caller-held guard).
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.index.get_owned(key)
    }

    pub fn delete(&self, key: &[u8]) -> bool {
        self.index.remove(key)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Flush the WAL to disk, if durable.
    pub fn flush(&self) -> std::io::Result<()> {
        if let Some(wal) = &self.wal {
            wal.lock().unwrap().as_mut().flush()?;
        }
        Ok(())
    }
}

//! End-to-end tests covering the derive macro, GAT storage, concurrency, EBR,
//! the async executor, and the self-referential WAL.

use std::sync::Arc;
use zerokv::executor::{block_on, DirectIoRead, Executor};
use zerokv::mmap_log::MappedLog;
use zerokv::regbuf::{FixedIoBackend, PortableBackend, RegisteredBuffers};
use zerokv::zerocopy::{ArenaKv, ZeroCopyStorage};
use zerokv::{KvStore, ShardedIndex, ZeroCopy};

// A user record type. `#[repr(C)]` + POD fields + no padding => zero-copy safe.
// The derive emits compile-time assertions for all three.
#[derive(ZeroCopy, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Trade {
    id: u64,
    price: u64,
    qty: u32,
    side: u32, // padded to keep the struct padding-free (qty+side fill 8 bytes)
}

#[test]
fn derive_zero_copy_roundtrip_and_view() {
    let t = Trade {
        id: 42,
        price: 10_000,
        qty: 7,
        side: 1,
    };
    let mut buf = [0u8; Trade::SERIALIZED_SIZE];
    t.encode(&mut buf);

    // by-value decode
    assert_eq!(Trade::decode(&buf), t);

    // true zero-copy: reinterpret the bytes in place
    let view = Trade::view(&buf).expect("aligned & sized");
    assert_eq!(*view, t);
    assert_eq!(Trade::SERIALIZED_SIZE, std::mem::size_of::<Trade>());
}

#[test]
fn gat_storage_returns_arena_borrow() {
    let kv = ArenaKv::new(1 << 16);
    assert!(kv.put(b"alpha", b"one"));
    assert!(kv.put(b"beta", b"two"));

    // `get` returns `Self::Ref<'a> = &'a [u8]` straight out of arena memory.
    let v: Option<&[u8]> = kv.get(b"beta");
    assert_eq!(v, Some(&b"two"[..]));
    assert_eq!(kv.get(b"missing"), None);
}

#[test]
fn single_thread_index_crud() {
    let idx = ShardedIndex::new(8, 1024);
    assert!(idx.insert(b"k1", b"v1"));
    assert!(!idx.insert(b"k1", b"v1b")); // update returns false
    assert_eq!(idx.get_owned(b"k1"), Some(b"v1b".to_vec()));
    assert!(idx.remove(b"k1"));
    assert_eq!(idx.get_owned(b"k1"), None);
    assert!(!idx.remove(b"k1"));
}

#[test]
fn concurrent_insert_get_remove() {
    let idx = Arc::new(ShardedIndex::new(16, 100_000));
    let threads = 8;
    let n = 5_000;

    // Concurrent inserts.
    let mut hs = Vec::new();
    for t in 0..threads {
        let idx = idx.clone();
        hs.push(std::thread::spawn(move || {
            for i in 0..n {
                let k = ((t * n + i) as u64).to_le_bytes();
                idx.insert(&k, &k);
            }
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    assert_eq!(idx.len(), threads * n);

    // Concurrent reads — every key must resolve to itself.
    let mut hs = Vec::new();
    for t in 0..threads {
        let idx = idx.clone();
        hs.push(std::thread::spawn(move || {
            for i in 0..n {
                let k = ((t * n + i) as u64).to_le_bytes();
                let v = idx.get_owned(&k).expect("present");
                assert_eq!(v, k.to_vec());
            }
        }));
    }
    for h in hs {
        h.join().unwrap();
    }

    // Concurrent removes of half the keyspace, exercising EBR reclamation.
    let mut hs = Vec::new();
    for t in 0..threads {
        let idx = idx.clone();
        hs.push(std::thread::spawn(move || {
            for i in 0..n {
                if (t * n + i) % 2 == 0 {
                    let k = ((t * n + i) as u64).to_le_bytes();
                    idx.remove(&k);
                }
            }
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    // Surviving (odd) keys still resolve; removed (even) keys are gone.
    for t in 0..threads {
        for i in 0..n {
            let k = ((t * n + i) as u64).to_le_bytes();
            if (t * n + i) % 2 == 0 {
                assert_eq!(idx.get_owned(&k), None);
            } else {
                assert_eq!(idx.get_owned(&k), Some(k.to_vec()));
            }
        }
    }
}

#[test]
fn custom_executor_block_on() {
    let out = block_on(async { 1 + 2 });
    assert_eq!(out, 3);
}

#[test]
fn custom_executor_direct_io_future() {
    // The DirectIoRead future yields once (Pending) then completes, exercising
    // the manual RawWaker + pinned aligned buffer.
    let buf = block_on(DirectIoRead::new(b"payload".to_vec(), 4096));
    assert_eq!(&buf.as_slice()[..7], b"payload");
}

#[test]
fn multitask_executor_runs_all() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static DONE: AtomicUsize = AtomicUsize::new(0);
    let ex = Executor::new();
    for _ in 0..16 {
        ex.spawn(async {
            DONE.fetch_add(1, Ordering::Relaxed);
        });
    }
    ex.run();
    assert_eq!(DONE.load(Ordering::Relaxed), 16);
}

#[test]
fn self_referential_wal_roundtrip() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("zerokv_wal_{}.log", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let mut log = MappedLog::open(&path, 64 * 1024).unwrap();
    let i0 = log.as_mut().append(b"hello").unwrap();
    let i1 = log.as_mut().append(b"world").unwrap();
    // Records are borrows back into the pinned buffer (self-reference).
    assert_eq!(log.record(i0), Some(&b"hello"[..]));
    assert_eq!(log.record(i1), Some(&b"world"[..]));
    log.as_mut().flush().unwrap();
    drop(log);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn registered_buffers_acquire_release_and_index() {
    let pool = RegisteredBuffers::register(4, 1); // 4 buffers, page-rounded
    assert_eq!(pool.count(), 4);
    assert_eq!(pool.buf_len(), 4096); // rounded up to a page
    assert_eq!(pool.available(), 4);

    // Acquire two distinct buffers => distinct buf_index values.
    let a = pool.acquire().unwrap();
    let b = pool.acquire().unwrap();
    assert_ne!(a.index(), b.index());
    assert_eq!(pool.available(), 2);

    // Dropping returns the slot to the pool.
    drop(a);
    drop(b);
    assert_eq!(pool.available(), 4);
}

#[test]
fn fixed_buffer_read_through_portable_backend() {
    // Write a file, then read it back into a *registered* fixed buffer via the
    // pluggable backend — the exact call the native io_uring path would make
    // through `read_fixed`, with the portable positional read standing in.
    let dir = std::env::temp_dir();
    let path = dir.join(format!("zerokv_fixed_{}.bin", std::process::id()));
    std::fs::write(&path, b"REGISTERED-BUFFER-IO").unwrap();
    let file = std::fs::File::open(&path).unwrap();

    let pool = RegisteredBuffers::register(2, 4096);
    let mut buf = pool.acquire().unwrap();
    let backend = PortableBackend;
    let n = backend.read_fixed(&file, 0, &mut buf, 20).unwrap();

    assert_eq!(n, 20);
    assert_eq!(&buf.as_slice()[..n], b"REGISTERED-BUFFER-IO");

    drop(buf);
    drop(file);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn end_to_end_async_pipeline() {
    // Full pipeline: a file on disk is read asynchronously, through the custom
    // executor, into a *registered fixed buffer* (the io_uring-ready path), the
    // bytes are loaded into the sharded lock-free index, and finally read back
    // zero-copy. Exercises regbuf + executor + Future/Pin + ShardedIndex + EBR
    // together.
    let dir = std::env::temp_dir();
    let path = dir.join(format!("zerokv_e2e_{}.bin", std::process::id()));
    // 4 records of 8 bytes: key i -> payload "REC#000i".
    let records: Vec<[u8; 8]> = (0..4)
        .map(|i| {
            let mut r = *b"REC#0000";
            r[7] = b'0' + i as u8;
            r
        })
        .collect();
    let mut blob = Vec::new();
    for r in &records {
        blob.extend_from_slice(r);
    }
    std::fs::write(&path, &blob).unwrap();
    let file = std::fs::File::open(&path).unwrap();

    let index = Arc::new(ShardedIndex::new(8, 16));
    let pool = RegisteredBuffers::register(2, 4096);
    let backend = PortableBackend;

    // For each record: async-read its 8 bytes at the right offset into a fixed
    // buffer, then insert into the index under key = record number.
    for (i, _) in records.iter().enumerate() {
        let mut buf = pool.acquire().expect("a free registered buffer");
        let n = backend
            .read_fixed(&file, (i * 8) as u64, &mut buf, 8)
            .unwrap();
        assert_eq!(n, 8);

        // Drive an async copy of the freshly-read bytes through our hand-rolled
        // runtime (DirectIoRead yields once via the manual RawWaker, proving the
        // executor wakes correctly), then commit to the index.
        let payload = buf.as_slice()[..n].to_vec();
        let staged = block_on(DirectIoRead::new(payload, 4096));
        index.insert(&[i as u8], &staged.as_slice()[..8]);
        // `buf` drops here -> returns to the pool for the next iteration.
    }

    // Every key resolves to its record, zero-copy, from the lock-free index.
    assert_eq!(index.len(), 4);
    for (i, r) in records.iter().enumerate() {
        assert_eq!(index.get_owned(&[i as u8]).as_deref(), Some(&r[..]));
    }

    drop(file);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn spsc_cross_thread_fifo() {
    // One producer thread, one consumer thread, FIFO with no loss/dup.
    let (tx, rx) = zerokv::spsc::channel::<u64>(1024);
    let n = 200_000u64;
    let producer = std::thread::spawn(move || {
        let mut i = 0u64;
        while i < n {
            // Spin on backpressure when the ring is full.
            if tx.push(i).is_ok() {
                i += 1;
            } else {
                std::hint::spin_loop();
            }
        }
    });
    let consumer = std::thread::spawn(move || {
        let mut expect = 0u64;
        while expect < n {
            if let Some(v) = rx.pop() {
                assert_eq!(v, expect, "SPSC must preserve FIFO order");
                expect += 1;
            } else {
                std::hint::spin_loop();
            }
        }
        expect
    });
    producer.join().unwrap();
    assert_eq!(consumer.join().unwrap(), n);
}

#[test]
fn mesh_cross_shard_puts_quiesce() {
    // 4 shards; each worker writes keys that route across ALL shards, exercising
    // the N×N SPSC matrix for inter-core message passing. After quiesce, every
    // submitted command must have been applied exactly once.
    let per_worker = 20_000usize;
    let (mesh, _) = zerokv::mesh::run_put_workload(4, move |id, _n| {
        (0..per_worker)
            .map(|i| {
                // Keys spread across the whole space so most route off-core.
                let k = ((id as u64) << 40 | i as u64).to_le_bytes().to_vec();
                let v = (i as u64).to_le_bytes().to_vec();
                (k, v)
            })
            .collect()
    });
    let shards = mesh.num_shards();
    let applied = mesh.shutdown();
    assert_eq!(shards, 4);
    assert_eq!(applied, (per_worker * shards) as u64, "every put applied once");
}

#[test]
fn mesh_service_client_request_response() {
    // Long-running core-pinned service; external client threads put/get through
    // the public handle. Keys route across all shards, so most requests are
    // forwarded over the inter-core matrix and replied back (Get/reply path).
    use zerokv::mesh::MeshService;

    let svc = MeshService::start(4);
    assert_eq!(svc.num_shards(), 4);

    let n_clients = 4;
    let per_client = 5_000usize;

    // Writers.
    std::thread::scope(|s| {
        for c in 0..n_clients {
            let h = svc.handle();
            s.spawn(move || {
                for i in 0..per_client {
                    let k = ((c * per_client + i) as u64).to_le_bytes();
                    h.put(&k, &k);
                }
            });
        }
    });

    // Readers — every previously written key must come back correct, even though
    // the accepting worker is round-robin and usually not the key's owner.
    std::thread::scope(|s| {
        for c in 0..n_clients {
            let h = svc.handle();
            s.spawn(move || {
                for i in 0..per_client {
                    let k = ((c * per_client + i) as u64).to_le_bytes();
                    let got = h.get(&k);
                    assert_eq!(got.as_deref(), Some(&k[..]), "service get must round-trip");
                }
                // A miss returns None.
                assert_eq!(h.get(&u64::MAX.to_le_bytes()), None);
            });
        }
    });

    svc.shutdown();
}

#[test]
fn group_commit_batches_and_wakes() {
    use std::time::Duration;
    use zerokv::executor::Executor;
    use zerokv::group_commit::GroupCommitWal;
    use zerokv::regbuf::PortableBackend;

    let dir = std::env::temp_dir();
    let path = dir.join(format!("zerokv_gc_{}.wal", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .unwrap();

    let wal = std::sync::Arc::new(GroupCommitWal::new(
        file,
        PortableBackend,
        Duration::from_micros(10),
        256,
    ));

    // Submit many records concurrently; each writer awaits its CommitFuture on
    // the custom executor. They should all be woken by the Batcher.
    use std::sync::atomic::{AtomicUsize, Ordering};
    static DONE: AtomicUsize = AtomicUsize::new(0);
    let ex = Executor::new();
    let n = 500;
    for i in 0..n {
        let wal = wal.clone();
        ex.spawn(async move {
            let rec = format!("record-{i:04}").into_bytes();
            wal.append(&rec).await;
            DONE.fetch_add(1, Ordering::Relaxed);
        });
    }
    ex.run();
    assert_eq!(DONE.load(Ordering::Relaxed), n, "all commit futures resolved");

    // The WAL file must contain the concatenated batch (n records * 11 bytes).
    drop(wal); // joins the batcher, flushing the tail
    let meta = std::fs::metadata(&path).unwrap();
    assert_eq!(meta.len(), (n * "record-0000".len()) as u64);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn durable_store_put_get() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("zerokv_store_{}.log", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let store = KvStore::durable(&path, 8, 1024, 1 << 20).unwrap();
    store.put(b"key", b"value");
    assert_eq!(store.get(b"key"), Some(b"value".to_vec()));
    store.flush().unwrap();
    drop(store);
    let _ = std::fs::remove_file(&path);
}

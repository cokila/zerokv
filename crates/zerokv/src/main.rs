// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright © 2026 gerardo. Part of zerokv; see LICENSE.

//! `zerokv` demo & benchmark binary.
//!
//! Run in release for meaningful numbers:
//!   cargo run --release --bin zerokv -- [threads] [ops_per_thread]
//!
//! It prints a short functional demo (KV ops + `#[derive(ZeroCopy)]`) and then
//! measures, on the lock-free sharded index:
//!   * write throughput (concurrent inserts),
//!   * read throughput — owned (copying) and zero-copy (borrowed),
//!   * a 90/10 mixed read/write workload,
//!   * single-op read latency percentiles (timer-overhead corrected).

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use zerokv::ebr::pin;
use zerokv::{KvStore, ShardedIndex, ZeroCopy};

#[derive(ZeroCopy, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Trade {
    id: u64,
    price: u64,
    qty: u32,
    side: u32,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let threads = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
    let per_thread: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let total = threads * per_thread;

    println!("zerokv — sub-millisecond lock-free KV engine");
    println!("============================================");
    println!("threads = {threads}, ops/thread = {per_thread}, total = {total}\n");

    functional_demo();
    derive_demo();

    println!("\n--- performance (release) ---\n");
    // Leva 2 — sharding scaled to the writer count to relieve head-array
    // contention. (Going *more* aggressive than this helps cache-resident sets
    // and high thread counts, but adds sentinel/locality overhead on a cold,
    // RAM-bound, single-thread access pattern — so we scale with threads.)
    let shards = (threads * 8).next_power_of_two().max(32);
    println!("sharding: {shards} partitions (~{} keys/shard)\n", total / shards);
    let idx = Arc::new(ShardedIndex::new(shards, total));

    let w = bench_writes(&idx, threads, per_thread);
    report("WRITE  (concurrent insert)", total, w);

    let r = bench_reads_owned(&idx, threads, per_thread);
    report("READ   (owned / copying)  ", total, r);

    let rz = bench_reads_zerocopy(&idx, threads, per_thread);
    report("READ   (zero-copy borrow) ", total, rz);

    let m = bench_mixed(&idx, threads, per_thread);
    report("MIXED  (90% get / 10% put)", total, m);

    println!(
        "\nNote: with {total} keys the working set is hundreds of MB (>> CPU cache),\n\
         so throughput above is bound by RAM latency — every skiplist hop is a cache miss."
    );

    // Cold (large) vs hot (cache-resident) read latency, to separate the
    // index's algorithmic cost from memory-subsystem cost.
    latency_percentiles("cold (full dataset, RAM-bound)", &idx, total.min(2_000_000));

    let hot_n = 50_000;
    let hot = Arc::new(ShardedIndex::new(256, hot_n));
    for i in 0..hot_n {
        let k = (i as u64).to_le_bytes();
        hot.insert(&k, &k);
    }
    latency_percentiles("hot (50k keys, cache-resident)", &hot, hot_n);

    shared_nothing_ab(threads, per_thread.min(500_000));
    mesh_service_bench(threads, per_thread.min(200_000));
    group_commit_bench(per_thread.min(200_000));
}

/// Throughput of the long-running `MeshService` through the *public client API*
/// (round-robin accept + cross-shard forward + reply). This is end-to-end
/// request/response latency, not raw index speed: every op crosses the ingress
/// MPSC and usually the inter-core SPSC matrix.
fn mesh_service_bench(client_threads: usize, per_client: usize) {
    use zerokv::mesh::MeshService;
    let n = client_threads.max(1);
    let total = n * per_client;
    println!("\n--- MeshService (shared-nothing, public client req/resp) ---");
    let svc = MeshService::start(n);
    println!("  {} shards, {n} client threads, {per_client} ops each", svc.num_shards());

    // Populate.
    let tw = {
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for c in 0..n {
                let h = svc.handle();
                s.spawn(move || {
                    for i in 0..per_client {
                        let k = ((c * per_client + i) as u64).to_le_bytes();
                        h.put(&k, &k);
                    }
                });
            }
        });
        t0.elapsed()
    };
    // GET, client-side routed to the owner (served locally, no matrix hop).
    let tr = bench_gets(&svc, n, per_client, false);
    // GET, round-robin accept (worst case: forwarded over the matrix + reply).
    let trr = bench_gets(&svc, n, per_client, true);
    svc.shutdown();

    let line = |label: &str, d: std::time::Duration| {
        println!(
            "  {label}  {:>6.2} Mops/s   {:>6.1} ns/op (end-to-end req/resp)",
            total as f64 / d.as_secs_f64() / 1e6,
            d.as_nanos() as f64 / total as f64
        );
    };
    line("PUT (client-routed)      ", tw);
    line("GET (client-routed/local)", tr);
    line("GET (round-robin/forward)", trr);
}

fn bench_gets(
    svc: &zerokv::mesh::MeshService,
    n: usize,
    per_client: usize,
    round_robin: bool,
) -> std::time::Duration {
    let _ = svc.num_shards();
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for c in 0..n {
            let h = svc.handle();
            s.spawn(move || {
                let mut acc = 0usize;
                for i in 0..per_client {
                    let k = ((c * per_client + i) as u64).to_le_bytes();
                    let got = if round_robin { h.get_round_robin(&k) } else { h.get(&k) };
                    if let Some(v) = got {
                        acc = acc.wrapping_add(v.len());
                    }
                }
                black_box(acc);
            });
        }
    });
    t0.elapsed()
}

/// Throughput of the Group-Commit WAL: many concurrent writers, each awaiting a
/// durable `CommitFuture`, coalesced by the Batcher into vectored writes.
fn group_commit_bench(records: usize) {
    use std::time::Duration;
    use zerokv::executor::Executor;
    use zerokv::group_commit::GroupCommitWal;
    use zerokv::regbuf::PortableBackend;

    let path = std::env::temp_dir().join(format!("zerokv_gcbench_{}.wal", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let file = std::fs::OpenOptions::new()
        .read(true).write(true).create(true).truncate(true)
        .open(&path).unwrap();

    println!("\n--- Group-Commit WAL (10µs window, vectored batches) ---");
    let wal = Arc::new(GroupCommitWal::new(file, PortableBackend, Duration::from_micros(10), 1024));
    let ex = Executor::new();
    for i in 0..records {
        let wal = wal.clone();
        ex.spawn(async move {
            let rec = (i as u64).to_le_bytes();
            wal.append(&rec).await;
        });
    }
    let t0 = Instant::now();
    ex.run();
    let d = t0.elapsed();
    drop(wal);
    let _ = std::fs::remove_file(&path);

    println!(
        "  {records} durable commits in {:?}  =>  {:.2} Mops/s   {:.1} ns/op (amortized)",
        d,
        records as f64 / d.as_secs_f64() / 1e6,
        d.as_nanos() as f64 / records as f64
    );
}

/// A/B test of proposal #1 (shared-nothing): same number of *core-pinned*
/// threads, but data either **shared** (every thread routes by hash across all
/// shards → cross-core access) or **shard-local** (thread `t` owns shard `t`
/// exclusively → no cross-core atomic contention, working set stays in that
/// core's private cache). Isolates the architectural effect from everything else.
fn shared_nothing_ab(threads: usize, per_thread: usize) {
    use zerokv::ebr::pin;
    use zerokv::skiplist::SkipList;

    let cores = core_affinity::get_core_ids().unwrap_or_default();
    let n = threads.max(1);
    let total = n * per_thread;
    println!(
        "\n--- Shared-Nothing A/B ({} physical cores detected, {n} pinned threads) ---",
        cores.len()
    );

    let pin_to = |t: usize| {
        if let Some(c) = cores.get(t).copied() {
            core_affinity::set_for_current(c);
        }
    };

    // ---- SHARED: one index, all threads hash across all shards (cross-core) --
    let shared = Arc::new(ShardedIndex::new(n.next_power_of_two(), total));
    let sw = {
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for t in 0..n {
                let idx = shared.clone();
                let pin_to = &pin_to;
                s.spawn(move || {
                    pin_to(t);
                    for i in 0..per_thread {
                        let k = ((t * per_thread + i) as u64).to_le_bytes();
                        idx.insert(&k, &k);
                    }
                });
            }
        });
        t0.elapsed()
    };
    let sr = {
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for t in 0..n {
                let idx = shared.clone();
                let pin_to = &pin_to;
                s.spawn(move || {
                    pin_to(t);
                    let mut acc = 0usize;
                    for i in 0..per_thread {
                        // Stride across the WHOLE keyspace → other cores' data.
                        let key = ((t.wrapping_mul(7919) + i.wrapping_mul(104729)) % total) as u64;
                        let g = pin();
                        if let Some(v) = idx.get(&key.to_le_bytes(), &g) {
                            acc = acc.wrapping_add(v.len());
                        }
                    }
                    black_box(acc);
                });
            }
        });
        t0.elapsed()
    };

    // ---- SHARD-LOCAL: N independent skiplists, thread t touches only its own -
    let local: Arc<Vec<SkipList>> =
        Arc::new((0..n).map(|_| SkipList::with_capacity_hint(per_thread)).collect());
    let lw = {
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for t in 0..n {
                let local = local.clone();
                let pin_to = &pin_to;
                s.spawn(move || {
                    pin_to(t);
                    let g = pin();
                    let sl = &local[t];
                    for i in 0..per_thread {
                        let k = (i as u64).to_le_bytes();
                        sl.insert(&k, &k, &g);
                    }
                });
            }
        });
        t0.elapsed()
    };
    let lr = {
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for t in 0..n {
                let local = local.clone();
                let pin_to = &pin_to;
                s.spawn(move || {
                    pin_to(t);
                    let sl = &local[t];
                    let mut acc = 0usize;
                    for i in 0..per_thread {
                        // Stride within OUR OWN shard only → core-local data.
                        let key = (i.wrapping_mul(2_654_435_761) % per_thread) as u64;
                        let g = pin();
                        if let Some(v) = sl.get(&key.to_le_bytes(), &g) {
                            acc = acc.wrapping_add(v.len());
                        }
                    }
                    black_box(acc);
                });
            }
        });
        t0.elapsed()
    };

    let ops = total as f64;
    let line = |label: &str, d: std::time::Duration| {
        println!(
            "{label}  {:>7.2} Mops/s   {:>6.1} ns/op",
            ops / d.as_secs_f64() / 1e6,
            d.as_nanos() as f64 / ops
        );
    };
    println!();
    line("WRITE  shared (cross-core) ", sw);
    line("WRITE  shard-local         ", lw);
    line("READ   shared (cross-core) ", sr);
    line("READ   shard-local         ", lr);
}

// ---------------------------------------------------------------------------
// Functional demos
// ---------------------------------------------------------------------------

fn functional_demo() {
    println!("[functional] KvStore put/get/delete");
    let store = KvStore::in_memory(8, 1024);
    store.put(b"user:1", b"alice");
    store.put(b"user:2", b"bob");
    println!("  get user:1 -> {:?}", as_str(store.get(b"user:1")));
    store.put(b"user:1", b"alice-v2"); // update
    println!("  get user:1 -> {:?} (after update)", as_str(store.get(b"user:1")));
    store.delete(b"user:2");
    println!("  get user:2 -> {:?} (after delete)", as_str(store.get(b"user:2")));
    println!("  len = {}", store.len());
}

fn derive_demo() {
    println!("[functional] #[derive(ZeroCopy)] round-trip & in-place view");
    let t = Trade { id: 7, price: 10_125, qty: 50, side: 1 };
    let mut buf = [0u8; Trade::SERIALIZED_SIZE];
    t.encode(&mut buf);
    let view = Trade::view(&buf).unwrap(); // reinterpret bytes, no copy
    println!("  encoded {} bytes; view == original: {}", Trade::SERIALIZED_SIZE, *view == t);
}

fn as_str(v: Option<Vec<u8>>) -> Option<String> {
    v.map(|b| String::from_utf8_lossy(&b).into_owned())
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_writes(idx: &Arc<ShardedIndex>, threads: usize, per_thread: usize) -> std::time::Duration {
    let t0 = Instant::now();
    scoped(threads, |t| {
        for i in 0..per_thread {
            let k = key(t, i, per_thread);
            idx.insert(&k, &k);
        }
    });
    t0.elapsed()
}

fn bench_reads_owned(idx: &Arc<ShardedIndex>, threads: usize, per_thread: usize) -> std::time::Duration {
    let t0 = Instant::now();
    scoped(threads, |t| {
        for i in 0..per_thread {
            let k = key(t, i, per_thread);
            black_box(idx.get_owned(&k));
        }
    });
    t0.elapsed()
}

fn bench_reads_zerocopy(idx: &Arc<ShardedIndex>, threads: usize, per_thread: usize) -> std::time::Duration {
    let t0 = Instant::now();
    scoped(threads, |t| {
        let mut acc = 0usize;
        for i in 0..per_thread {
            let k = key(t, i, per_thread);
            // One pin per op (realistic): the borrow is valid only while pinned.
            let g = pin();
            if let Some(v) = idx.get(&k, &g) {
                acc = acc.wrapping_add(v.len());
            }
        }
        black_box(acc);
    });
    t0.elapsed()
}

fn bench_mixed(idx: &Arc<ShardedIndex>, threads: usize, per_thread: usize) -> std::time::Duration {
    let t0 = Instant::now();
    scoped(threads, |t| {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
        for _ in 0..per_thread {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let slot = (rng as usize) % per_thread;
            let k = key(t, slot, per_thread);
            if rng & 0b1001 == 0b1001 {
                // ~10% writes
                idx.insert(&k, &k);
            } else {
                black_box(idx.get_owned(&k));
            }
        }
    });
    t0.elapsed()
}

/// Single-threaded read latency histogram, corrected for timer overhead.
fn latency_percentiles(label: &str, idx: &Arc<ShardedIndex>, samples: usize) {
    // Estimate Instant::now() overhead so we can report the net op cost.
    let mut sink = 0u64;
    let cal = Instant::now();
    for _ in 0..200_000 {
        sink = sink.wrapping_add(Instant::now().elapsed().as_nanos() as u64);
    }
    black_box(sink);
    let timer_overhead_ns = (cal.elapsed().as_nanos() as f64 / 200_000.0).round() as u64;

    let mut lat = Vec::with_capacity(samples);
    for i in 0..samples {
        let k = key(0, i, samples);
        let g = pin();
        let t = Instant::now();
        let v = idx.get(&k, &g);
        let ns = t.elapsed().as_nanos() as u64;
        black_box(v);
        lat.push(ns.saturating_sub(timer_overhead_ns));
    }
    lat.sort_unstable();

    let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
    println!(
        "\nREAD latency [{label}] (zero-copy, 1 thread, {} samples, ~{} ns timer subtracted):",
        lat.len(),
        timer_overhead_ns
    );
    println!("  p50  = {:>5} ns", pct(0.50));
    println!("  p90  = {:>5} ns", pct(0.90));
    println!("  p99  = {:>5} ns", pct(0.99));
    println!("  p99.9= {:>5} ns", pct(0.999));
    println!("  max  = {:>5} ns  (< 1 ms = {})", lat[lat.len() - 1], lat[lat.len() - 1] < 1_000_000);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic 8-byte key for (thread, i).
#[inline]
fn key(t: usize, i: usize, per_thread: usize) -> [u8; 8] {
    ((t * per_thread + i) as u64).to_le_bytes()
}

/// Run `f(thread_id)` on `threads` OS threads and join them.
fn scoped<F: Fn(usize) + Sync>(threads: usize, f: F) {
    std::thread::scope(|s| {
        for t in 0..threads {
            let f = &f;
            s.spawn(move || f(t));
        }
    });
}

fn report(label: &str, ops: usize, d: std::time::Duration) {
    let secs = d.as_secs_f64();
    println!(
        "{label}  {:>10.2} ms   {:>7.2} Mops/s   {:>6.1} ns/op",
        secs * 1e3,
        ops as f64 / secs / 1e6,
        d.as_nanos() as f64 / ops as f64
    );
}

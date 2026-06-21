//! Ad-hoc throughput / latency probe (custom harness, no criterion dependency).
//! Run with: `cargo run --release --bench throughput`.

use std::sync::Arc;
use std::time::Instant;
use zerokv::ShardedIndex;

fn main() {
    let threads = 8usize;
    let per_thread = 200_000usize;
    let total = threads * per_thread;

    let idx = Arc::new(ShardedIndex::new(32, total));

    // --- Write throughput ---------------------------------------------------
    let t0 = Instant::now();
    let mut handles = Vec::new();
    for t in 0..threads {
        let idx = idx.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..per_thread {
                let k = ((t * per_thread + i) as u64).to_le_bytes();
                idx.insert(&k, &k);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let w = t0.elapsed();
    println!(
        "WRITE  {total} ops in {:?}  =>  {:.2} Mops/s  ({:.1} ns/op)",
        w,
        total as f64 / w.as_secs_f64() / 1e6,
        w.as_nanos() as f64 / total as f64
    );

    // --- Read latency -------------------------------------------------------
    let t1 = Instant::now();
    let mut hits = 0u64;
    let mut handles = Vec::new();
    for t in 0..threads {
        let idx = idx.clone();
        handles.push(std::thread::spawn(move || {
            let mut local_hits = 0u64;
            for i in 0..per_thread {
                let k = ((t * per_thread + i) as u64).to_le_bytes();
                if idx.get_owned(&k).is_some() {
                    local_hits += 1;
                }
            }
            local_hits
        }));
    }
    for h in handles {
        hits += h.join().unwrap();
    }
    let r = t1.elapsed();
    println!(
        "READ   {total} ops in {:?}  =>  {:.2} Mops/s  ({:.1} ns/op), hits={hits}",
        r,
        total as f64 / r.as_secs_f64() / 1e6,
        r.as_nanos() as f64 / total as f64
    );
    assert_eq!(hits as usize, total, "every inserted key must be readable");
}

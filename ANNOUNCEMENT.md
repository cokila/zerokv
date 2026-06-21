# 🚀 Show HN: zerokv — a 200ns Rust KV engine to kill the LLM KV-Cache data-path bottleneck

While the AI industry pours billions into ever-faster GPUs, local inference servers (llama.cpp, ds4) and distributed clusters are choking on the **data path**.

With long contexts (32k–1M tokens) or MoE (Mixture-of-Experts) architectures, the GPU spends an enormous amount of time *idle*, waiting for the CPU to shuttle KV-Cache data or load expert state. Lean on RocksDB or the plain filesystem and you inherit micro-to-millisecond latencies, background compactions, and atomic contention that wreck throughput.

`zerokv` attacks this at the root, bringing **High-Frequency-Trading (HFT) systems discipline** to LLM infrastructure.

## 📊 The Numbers (A/B measured, 16 cores / 2M keys, cold set)

| Path | Latency |
|---|---|
| Direct `ShardedIndex` READ (zero-copy) | **~207 ns/op** |
| `MeshService` PUT (client-routed) | **~231 ns/op** (4.8 Mops/s) |
| `MeshService` GET (client-routed / local) | **~344 ns/op** |
| Tail latency **p99.9** | **746 ns** (solid sub-microsecond, fully tamed tail, zero stuttering) |

> Throughput on the full 2M-key set is RAM-latency bound (working set ≫ CPU cache);
> the p99.9 figure is from the cache-resident hot path. Reproduce:
> `cargo run --release --bin zerokv -- 16 125000`.

## 🧠 How it squeezes the hardware: the architecture

To get under the 200ns barrier on the hot path, we had to tear down conventional software abstractions and work *with* the silicon:

- **Pointer-chasing elimination (inline tower / DST).** Classic skiplists thrash the L3 cache: every node points to forward-link arrays scattered across the heap. `zerokv` unifies the node into a hand-built **Dynamically Sized Type**, allocated manually via `Layout` + `addr_of_mut!`. The skiplist "tower" lives **inline**, in the same allocation as the key. One RAM fetch pulls the whole node into cache. (Measured: this single change roughly halved per-op latency.)

- **Fixed-size arena allocator + EBR.** No heap fragmentation, no `malloc` on the critical path — memory is pre-allocated and sealed. Safe concurrency comes from **Epoch-Based Reclamation**: retired nodes are freed in bulk only once no thread can still be reading them. No garbage collector, no stop-the-world.

- **Shared-nothing mesh.** Concurrent databases suffer **cache-line bouncing** from cross-core atomic coordination (MESI). `zerokv`'s `MeshService` pins worker threads to physical cores; the client hashes the key up front (**client-side routing**) and fires the request straight into the owning core's lock-free MPSC queue. Each core works in isolation on its private L1/L2. (A/B measured: write speedup scales 1.2× → 1.6× → 2.0× at 4/8/16 cores.)

- **Group-commit WAL, native-I/O ready.** The write-ahead log coalesces hundreds of concurrent transactions into a single **vectored write (`writev`)**, with a seam ready to wire natively to **Direct I/O (`O_DIRECT`) and `io_uring`**, bypassing the OS page cache.

## 🛠️ The intended impact on LLM servers (llama.cpp / ds4 / vLLM)

> These are the **design targets** the architecture is built to hit, not yet end-to-end
> benchmarks inside an inference server. The numbers above are the engine's measured
> primitives; integration results will follow.

- **Faster Time-To-First-Token (TTFT) on context resume.** When an agent resumes a 64k-token chat, retrieving and zero-copy-mapping the stored KV-Cache through the index is a software-cheap operation, letting the SSD feed the PCIe channel to the GPU instead of the CPU stalling on serialization/compaction.

- **No multi-user "stuttering."** In multi-tenant serving, where the server constantly swaps KV cache out to disk under VRAM pressure, the tamed **746 ns p99.9** means I/O never blocks the compute threads — token-generation cadence stays flat.

## ⚖️ License & commercial strategy (AGPLv3)

`zerokv` tackles a live, critical, multi-million-dollar problem in modern AI infrastructure. It is released under the **GNU Affero General Public License v3 (AGPLv3)**.

- **Open source / research:** free to use, modify, and integrate into academic or open-source projects (e.g. custom FFI bridges for llama.cpp) — **provided the entire stack that interacts with `zerokv` is released under the AGPLv3.**

- **Commercial / closed-source:** if you are a cloud provider, an AI startup, or a quantitative trading fund and want `zerokv`'s performance inside proprietary infrastructure **without** releasing your source, you are legally required to obtain a **Commercial License**.

For dual-licensing, dedicated C-ABI bridges, or enterprise support: **gerardo.mancini@gmail.com**

---

Repo: https://github.com/cokila/zerokv · Built in Rust, `std` + `core_affinity` only. 16 integration tests, clippy-clean.

# zerokv

A **sub-millisecond, garbage-collector-free, lock-free** key-value storage
engine written in Rust, built as a deep exploration of the language's most
advanced systems-programming features. The design goal is *predictable* latency
under concurrent load: no GC pauses, no blocking locks on the data path, and the
minimum number of memory barriers required for correctness.

```
cargo test            # 9 integration tests (incl. concurrent EBR stress)
cargo bench           # throughput / latency probe (custom harness)
cargo clippy --all-targets   # clean
```

## Why these choices

| Requirement | Where | Technique |
|---|---|---|
| **Custom allocator** (no heap on reads) | `arena.rs` | Wait-free bump allocator over `std::alloc::Layout` + `NonNull<u8>`; O(1) reset, atomic cursor, no per-record `malloc`. |
| **Zero-copy reads via GATs** | `zerocopy.rs` | `trait ZeroCopyStorage { type Ref<'a>; }` — a Generic Associated Type ties returned `&'a [u8]` borrows to the store's lifetime, so the borrow checker statically forbids dangling reads. |
| **Self-referential durable log** | `mmap_log.rs` | A struct that owns a page-aligned (`O_DIRECT`-ready) buffer **and** stores raw pointers into it. Made sound with `PhantomPinned` + `Pin<Box<Self>>` and a manual `Drop` (single dealloc, no leak/double-free). |
| **Lock-free ordered index** | `skiplist.rs` | Herlihy–Shavit skiplist using **pointer-tagged `AtomicPtr`** ("marked references": the free low bit flags logical deletion, so one CAS atomically updates link + delete-flag). No `Mutex`/`RwLock`. |
| **Granular memory ordering** | `skiplist.rs` | `Acquire` loads / `Release` stores / `AcqRel` CAS; `Relaxed` only for counters. Documented per operation. |
| **Safe concurrent reclamation** | `ebr.rs` | **Epoch-Based Reclamation** replaces the GC: retired nodes are freed only after the global epoch advances two steps past any pinned reader — defeating use-after-free / stale reads. Itself lock-free (atomic slots + Treiber-stack garbage bags). |
| **Backoff** | `backoff.rs` | Exponential backoff + **random jitter** (de-synchronizes herds) with `core::hint::spin_loop()` (`PAUSE`/`YIELD`) to cut coherence-bus traffic and pipeline flushes. |
| **Adaptive height** | `skiplist.rs` | Tower height adapts to expected N (`ceil(log₄ N)`), and the level distribution uses `p = 1/4` — fewer levels ⇒ fewer CAS per insert ⇒ smaller writer collision surface. |
| **Sharding / partitioning** | `shard.rs` | `K` independent skiplists routed by `hash(key) & (K-1)`; cuts writer-vs-writer collisions ~`K×` while reads pay only a predicted hash. |
| **Custom async runtime** | `executor.rs` | `block_on` built on a **hand-written `RawWaker` + `RawWakerVTable`** (park/unpark); a multi-task `Executor` via a manual `Wake` impl; a `DirectIoRead` future with a **structurally pinned** aligned I/O buffer and a hand-written `Future::poll`. |
| **Compile-time serialization** | `zerokv-derive/` | `#[derive(ZeroCopy)]` generates zero-copy encode/decode/`view` and emits `const` assertions that **reject non-POD fields and any padding at compile time** (a padded struct fails to build). |

## `Send` / `Sync`, covariance & invariance

- `Arena`, `SkipList`, `ShardedIndex` are `Send + Sync`: all shared mutation is
  atomic and frees are gated by EBR. The `unsafe impl`s assert what the compiler
  cannot infer (raw pointers default a type to `!Send`/`!Sync`).
- `ebr::Guard` is deliberately **`!Send`** — it owns a thread-bound epoch slot.
- Borrows from `get<'g>(…, &'g Guard)` are **covariant** in `'g` (`&'g [u8]`):
  a longer pin may be shortened but never extended past the guard, so a
  zero-copy value reference can never escape the epoch that keeps it alive.

## Where `unsafe` lives

Every `unsafe` block carries a `// SAFETY:` note. All of them reduce to four
owners of truth:

1. **Arena** owns its bytes for its whole lifetime (disjoint, non-overlapping
   reservations).
2. **EBR** delays every free past any live `Guard`.
3. **`Pod`** guarantees byte-validity for the zero-copy casts (enforced at
   compile time by the derive).
4. **`Pin`** keeps the WAL buffer and async I/O buffers fixed in memory.

## Layout

```
crates/
  zerokv/          # the engine (std + core_affinity)
    src/
      arena.rs        # bump arena allocator
      zerocopy.rs     # Pod / ZeroCopy / GAT ZeroCopyStorage
      skiplist.rs     # lock-free skiplist (SKO + inline tower + prefetch)
      ebr.rs          # epoch-based reclamation (no GC)
      backoff.rs      # spin_loop + jittered exponential backoff
      shard.rs        # hash-partitioned index
      mmap_log.rs     # self-referential, pinned WAL buffer
      regbuf.rs       # registered fixed buffers (io_uring/RIO-ready) + FixedIoBackend
      group_commit.rs # group-commit WAL batcher (vectored writes)
      spsc.rs         # lock-free SPSC ring (cache-line padded)
      mesh.rs         # N×N SPSC shared-nothing mesh + MeshService client API
      executor.rs     # custom async runtime (RawWaker/Wake/Future/Pin)
      store.rs        # KvStore façade
    tests/integration.rs   # 16 tests
    benches/throughput.rs
  zerokv-derive/   # the #[derive(ZeroCopy)] procedural macro (syn/quote)
```

> Note: the WAL / Direct-I/O layer uses a portable aligned-buffer abstraction in
> place of raw platform `io_uring`/`O_DIRECT`/RIO syscalls so the engine builds
> and is testable on any OS; the alignment, pinning, registered-buffer and
> self-referential ownership model are exactly what the real syscall path
> requires (marked with `// NATIVE …:` seams).

## License

Copyright © 2026 gerardo.

`zerokv` is free software, licensed under the **GNU Affero General Public License
v3.0 or later (AGPL-3.0-or-later)**. See [LICENSE](LICENSE). In particular, if you
run a modified version to provide a network service, the AGPL requires you to
offer the corresponding source to its users.

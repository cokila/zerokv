// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright © 2026 gerardo. Part of zerokv; see LICENSE.

//! # zerokv — a sub-millisecond, GC-free, lock-free key-value storage engine
//!
//! `zerokv` is a study in pushing Rust's most advanced features toward a single
//! goal: predictable sub-millisecond latency under concurrent load, with **no
//! garbage collector**, **minimal memory barriers**, and **no blocking locks on
//! the data path**.
//!
//! ## Architecture map
//!
//! | Concern                         | Module            | Key techniques |
//! |---------------------------------|-------------------|----------------|
//! | Custom allocation               | [`arena`]         | `Layout`, `NonNull<u8>`, wait-free bump |
//! | Zero-copy reads                 | [`zerocopy`]      | **GAT** `ZeroCopyStorage::Ref<'a>`, `Pod` |
//! | Self-referential durable log    | [`mmap_log`]      | self-ref pointers, `Pin`, manual `Drop` |
//! | Lock-free ordered index         | [`skiplist`]      | tagged `AtomicPtr`, granular `Ordering` |
//! | Safe concurrent reclamation     | [`ebr`]           | epoch-based reclamation (no GC) |
//! | Contention control              | [`backoff`]       | `spin_loop()` + exponential jitter |
//! | Horizontal write scaling        | [`shard`]         | hash-partitioned skiplists |
//! | Async storage I/O               | [`executor`]      | manual `RawWaker`/`Wake`/`Future`, `Pin` |
//! | Registered fixed I/O buffers    | [`regbuf`]        | `IORING_REGISTER_BUFFERS` / RIO readiness, pinned slab |
//! | Public façade                   | [`store`]         | `KvStore` (index + WAL) |
//! | Compile-time serialization      | `#[derive(ZeroCopy)]` | proc-macro, const bound checks |
//!
//! ## Send / Sync stance (covariance & invariance)
//!
//! * [`arena::Arena`], [`skiplist::SkipList`], [`shard::ShardedIndex`] are
//!   `Send + Sync`: all shared mutation is atomic and reclamation is governed by
//!   [`ebr`]. The unsafe `impl`s assert this where the compiler cannot infer it
//!   (raw pointers make a type `!Send`/`!Sync` by default).
//! * [`ebr::Guard`] is intentionally **`!Send`** (it holds a thread-bound epoch
//!   slot); its lifetime parameter on borrows is **invariant**, which is what
//!   forbids a returned `&'g [u8]` from being smuggled past the pin.
//! * The borrows returned by [`zerocopy::ZeroCopyStorage::get`] are **covariant**
//!   in `'a` (they are `&'a [u8]`), so a longer arena lifetime may be shortened
//!   safely, but never extended past the arena — the GAT bound `Ref<'a>: 'a`
//!   plus the `&'a self` receiver enforce this with zero runtime cost.
//!
//! ## Where `unsafe` lives and why it is sound
//! Every `unsafe` block is paired with a `// SAFETY:` note stating the invariant
//! it relies on. The invariants reduce to four owners of truth: the arena owns
//! its bytes for its lifetime; EBR delays frees past any live guard; `Pod`
//! guarantees byte-validity for zero-copy casts; and pinning keeps I/O buffers
//! and the WAL fixed in memory.

pub mod arena;
pub mod backoff;
pub mod ebr;
pub mod executor;
pub mod group_commit;
pub mod mesh;
pub mod mmap_log;
pub mod regbuf;
pub mod shard;
pub mod skiplist;
pub mod spsc;
pub mod store;
pub mod zerocopy;

// Re-exports forming the crate's public surface.
pub use store::KvStore;
pub use shard::ShardedIndex;
pub use zerocopy::{Pod, ZeroCopy, ZeroCopyStorage};

// The derive macro. `#[derive(ZeroCopy)]` expands to code that names
// `::zerokv::Pod` and `::zerokv::ZeroCopy`, so both must be in scope at the
// crate root (above).
pub use zerokv_derive::ZeroCopy;

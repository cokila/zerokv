//! **Lock-free SkipList** index (Herlihy–Shavit style) with pointer-tagged
//! "marked references", epoch-based reclamation, and contention-aware backoff.
//!
//! Why a skiplist and not a B-tree? A skiplist's structural modifications are
//! *local single-pointer CAS operations*, which map cleanly onto lock-free
//! progress without the node-splitting/merging cascades a concurrent B-tree
//! requires. Reads are wait-free in the common case and touch O(log N) cache
//! lines.
//!
//! ## No `Mutex`, no `RwLock`
//! Every mutation is a `compare_exchange` on an `AtomicPtr`. Logical deletion is
//! signaled by tagging the *low bit* of the forward pointer (`Node` is 8-byte
//! aligned, so bits 0..3 are free) — the classic "marked reference" that lets a
//! single CAS atomically change the link AND its deleted-flag, defeating the
//! lost-unlink race.
//!
//! ## Memory ordering
//! * `get` (load) → `Acquire`: observe the publishing `Release` store of the
//!   inserter, so the node's key/value are visible once its pointer is.
//! * link/unlink CAS → `AcqRel` on success: publish the new node to readers and
//!   acquire the predecessor's prior writes.
//!
//! Relaxed is used only for counters where no data is published.
//!
//! ## Reclamation
//! Unlinked nodes are *retired* to the EBR domain, never freed inline — a reader
//! holding a [`Guard`] can finish dereferencing a pointer it already loaded.

use crate::backoff::{random_geometric_level, Backoff};
use crate::ebr::Guard;
use std::alloc::{self, Layout};
use std::cell::Cell;
use std::cmp::Ordering as CmpOrdering;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Hard ceiling on tower height. The *effective* cap is adaptive (see
/// [`SkipList::with_capacity_hint`]): fewer levels ⇒ fewer CAS per insert ⇒ a
/// smaller writer-vs-writer collision surface.
pub const MAX_HEIGHT: usize = 16;

/// Software prefetch hint: pull `*p`'s cache line toward the core *now*, while
/// we still have useful work (comparisons) to overlap with the RAM latency. On
/// x86-64 this is `PREFETCHT0` (all cache levels); a no-op elsewhere. Uses the
/// **stable** `core::arch` intrinsic (`core::intrinsics::prefetch_*` is
/// nightly-only).
#[inline(always)]
fn prefetch<T>(p: *const T) {
    if p.is_null() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` cannot cause UB for any address — at worst the hint
    // is ignored. `p` is non-null and points at a (possibly soon-freed) node;
    // prefetching a stale line is harmless.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(p as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = p; // portable no-op
    }
}

/// Inline byte capacity for the **Small Key Optimization (SKO)**. Keys up to
/// `INLINE_CAP` bytes are stored *inside the node* (same allocation, same cache
/// line as the node header), so loading a node to read its forward pointers
/// already pulls in its key — eliminating the second, dependent cache miss that
/// a `Box<[u8]>` key would incur on every comparison.
const INLINE_CAP: usize = 22;

/// A key that is either inlined into the node (the common, fast case) or, for
/// rare oversized keys, spilled to the heap — exactly the SSO/`std::string`
/// small-buffer trick applied to skiplist keys.
enum KeyRepr {
    /// `len` bytes valid in `buf`; `len <= INLINE_CAP`. No indirection.
    Inline { len: u8, buf: [u8; INLINE_CAP] },
    /// Fallback for keys larger than `INLINE_CAP`.
    Boxed(Box<[u8]>),
}

impl KeyRepr {
    #[inline]
    fn new(key: &[u8]) -> Self {
        if key.len() <= INLINE_CAP {
            let mut buf = [0u8; INLINE_CAP];
            buf[..key.len()].copy_from_slice(key);
            KeyRepr::Inline {
                len: key.len() as u8,
                buf,
            }
        } else {
            KeyRepr::Boxed(key.to_vec().into_boxed_slice())
        }
    }

    #[inline(always)]
    fn as_bytes(&self) -> &[u8] {
        match self {
            // No pointer chase: the bytes live in the node header itself.
            KeyRepr::Inline { len, buf } => &buf[..*len as usize],
            KeyRepr::Boxed(b) => b,
        }
    }
}

/// `head < every key`, `tail > every key`. Modeled explicitly so the search
/// loop needs no special-casing for the sentinels.
enum BoundKey {
    NegInf,
    Fin(KeyRepr),
    PosInf,
}

/// Heap value cell. Swapped atomically on update; the displaced cell is retired
/// through EBR so concurrent readers keep a valid view.
struct ValueRecord {
    bytes: Box<[u8]>,
}

/// A tagged atomic forward pointer ("AtomicMarkableReference").
///
/// Layout trick: `Node` has alignment ≥ 8, so the low bit of a node pointer is
/// always 0 and free to carry the "this node is logically deleted" mark.
struct Link {
    p: AtomicPtr<Node>,
}

impl Link {
    #[inline]
    fn null() -> Self {
        Link {
            p: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    #[inline]
    fn pack(ptr: *mut Node, mark: bool) -> *mut Node {
        ((ptr as usize) | (mark as usize)) as *mut Node
    }
    #[inline]
    fn ptr_of(v: *mut Node) -> *mut Node {
        ((v as usize) & !1usize) as *mut Node
    }
    #[inline]
    fn mark_of(v: *mut Node) -> bool {
        (v as usize) & 1 == 1
    }

    /// Returns `(reference, mark)`.
    #[inline]
    fn get(&self) -> (*mut Node, bool) {
        let v = self.p.load(Ordering::Acquire);
        (Self::ptr_of(v), Self::mark_of(v))
    }
    /// Just the reference (mark stripped).
    #[inline]
    fn reference(&self) -> *mut Node {
        Self::ptr_of(self.p.load(Ordering::Acquire))
    }
    /// Unconditional store (used to initialize a not-yet-published node).
    #[inline]
    fn store(&self, ptr: *mut Node, mark: bool) {
        self.p.store(Self::pack(ptr, mark), Ordering::Release);
    }
    /// CAS expecting `(exp_ref, exp_mark)` → `(new_ref, new_mark)`.
    #[inline]
    fn cas(&self, exp_ref: *mut Node, new_ref: *mut Node, exp_mark: bool, new_mark: bool) -> bool {
        self.p
            .compare_exchange(
                Self::pack(exp_ref, exp_mark),
                Self::pack(new_ref, new_mark),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
    /// Set the mark while keeping the reference (logical delete of one level).
    #[inline]
    fn attempt_mark(&self, exp_ref: *mut Node, mark: bool) -> bool {
        self.p
            .compare_exchange(
                Self::pack(exp_ref, false),
                Self::pack(exp_ref, mark),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

/// A skiplist node.
///
/// **Inline tower layout.** The `top_level + 1` forward [`Link`]s are *not* a
/// separate `Box<[Link]>` allocation; they live in the *same* allocation,
/// immediately after this header (a hand-rolled flexible-array member). One node
/// = one allocation = one (or two) cache lines holding header + inline key
/// (SKO) + forward pointers. A traversal step that loads the node to read a
/// forward pointer therefore also pulls in its key and the rest of the tower —
/// no second pointer chase to a links array, no third to a boxed key.
///
/// Because Rust has no native flexible array member, the links are reached via
/// pointer arithmetic at [`Node::LINK_OFFSET`] and the node is allocated/freed
/// with an explicit [`Layout`]; it is never a `Box<Node>`.
struct Node {
    key: BoundKey,
    value: AtomicPtr<ValueRecord>,
    top_level: usize,
    // Followed in the same allocation by `[Link; top_level + 1]`.
}

impl Node {
    /// Byte offset of the trailing `[Link]` array. `size_of::<Node>()` is a
    /// multiple of `align_of::<Node>() (= align_of::<Link>() = 8)`, so the array
    /// begins exactly at the end of the header with no extra padding.
    const LINK_OFFSET: usize = std::mem::size_of::<Node>();

    /// Combined layout for `header + [Link; height]`.
    ///
    /// NOTE — *measured* decision: forcing 64-byte (cache-line) alignment to stop
    /// a node from straddling two lines was tried and **regressed this workload
    /// ~40%**. The padding inflates a short node (~48 B → 64 B), enlarging an
    /// already memory-capacity-bound working set; the extra RAM traffic costs
    /// more than the rare straddle saves. We keep the natural 8-byte alignment.
    /// (Cache-line alignment would pay off only under heavy *false sharing*
    /// between adjacent nodes' atomics — not the case here.)
    #[inline]
    fn layout_for(height: usize) -> Layout {
        let (layout, off) = Layout::new::<Node>()
            .extend(Layout::array::<Link>(height).expect("link array layout"))
            .expect("node layout");
        debug_assert_eq!(off, Self::LINK_OFFSET);
        layout.pad_to_align()
    }

    /// Allocate one node with an inline tower of `top_level + 1` links, all
    /// initialized to null. Returns a raw pointer; the node is owned manually
    /// and must be released with [`free_node`].
    fn new(key: BoundKey, value: *mut ValueRecord, top_level: usize) -> *mut Node {
        let height = top_level + 1;
        let layout = Self::layout_for(height);
        // SAFETY: layout has non-zero size (Node is non-ZST).
        let raw = unsafe { alloc::alloc(layout) };
        if raw.is_null() {
            alloc::handle_alloc_error(layout);
        }
        let node = raw as *mut Node;
        // SAFETY: `raw` is a fresh, suitably-aligned allocation for `Node`
        // followed by `height` `Link`s. We initialize every field exactly once
        // with `write` (no drop of uninitialized memory), using `addr_of_mut!`
        // so we never form a reference to uninitialized data.
        unsafe {
            std::ptr::write(std::ptr::addr_of_mut!((*node).key), key);
            std::ptr::write(std::ptr::addr_of_mut!((*node).value), AtomicPtr::new(value));
            std::ptr::write(std::ptr::addr_of_mut!((*node).top_level), top_level);
            let links = raw.add(Self::LINK_OFFSET) as *mut Link;
            for i in 0..height {
                std::ptr::write(links.add(i), Link::null());
            }
        }
        node
    }

    /// Borrow forward link at `level` (`level <= top_level`). No indirection
    /// beyond this node's own allocation.
    #[inline(always)]
    fn link(&self, level: usize) -> &Link {
        debug_assert!(level <= self.top_level);
        // SAFETY: the allocation contains `top_level + 1` `Link`s starting at
        // `LINK_OFFSET`; `level` is in range.
        unsafe {
            let base = (self as *const Node as *const u8).add(Self::LINK_OFFSET) as *const Link;
            &*base.add(level)
        }
    }
}

/// Drop a node's owned resources and free its (header + tower) allocation.
///
/// # Safety
/// `node` must be a live pointer returned by [`Node::new`], uniquely owned at
/// this point (already unlinked, and — under concurrency — only reachable here
/// via EBR after the grace period).
unsafe fn free_node(node: *mut Node) {
    // SAFETY: caller guarantees unique ownership of a valid node.
    unsafe {
        let top = (*node).top_level;
        // Drop the key in place (frees a `Boxed` key; `Inline` is trivial).
        std::ptr::drop_in_place(std::ptr::addr_of_mut!((*node).key));
        // Free the owned value cell, if any.
        let v = (*node).value.load(Ordering::Relaxed);
        if !v.is_null() {
            drop(Box::from_raw(v));
        }
        // `Link`s are trivially destructible (atomics over raw ptrs); just free
        // the whole combined allocation.
        alloc::dealloc(node as *mut u8, Node::layout_for(top + 1));
    }
}

/// EBR drop-glue: drop the node's resources and free its inline-tower allocation.
unsafe fn drop_node(p: *mut u8) {
    // SAFETY: `p` is a `*mut Node` from `Node::new`, retired exactly once and
    // freed only after the epoch grace period.
    unsafe { free_node(p as *mut Node) };
}
/// EBR drop-glue for a displaced value cell.
unsafe fn drop_value(p: *mut u8) {
    // SAFETY: as above, for a `ValueRecord`.
    unsafe { drop(Box::from_raw(p as *mut ValueRecord)) };
}

/// The lock-free ordered map.
pub struct SkipList {
    head: *mut Node,
    max_level: usize,
    len: AtomicUsize,
}

// SAFETY: all shared mutation is through atomics with explicit ordering; `Node`
// ownership is governed by EBR. No interior `!Send` state is exposed.
unsafe impl Send for SkipList {}
unsafe impl Sync for SkipList {}

impl SkipList {
    /// Build a skiplist whose effective height is adapted to `expected_items`:
    /// `ceil(log4(N))`, clamped to `[1, MAX_HEIGHT]`. Capping height for small N
    /// is the single biggest lever on writer contention — every saved level is
    /// one fewer CAS in the `insert` fast path.
    pub fn with_capacity_hint(expected_items: usize) -> Self {
        let n = expected_items.max(2) as f64;
        // log base 4 because we use p = 1/4 for the geometric level distribution.
        let lvl = (n.ln() / 4f64.ln()).ceil() as usize;
        let max_level = lvl.clamp(1, MAX_HEIGHT) - 1; // 0-based top index
        Self::with_max_level(max_level)
    }

    fn with_max_level(max_level: usize) -> Self {
        let head = Node::new(BoundKey::NegInf, std::ptr::null_mut(), MAX_HEIGHT - 1);
        let tail = Node::new(BoundKey::PosInf, std::ptr::null_mut(), MAX_HEIGHT - 1);
        // SAFETY: both just allocated; wire every head level straight to tail.
        unsafe {
            for level in 0..MAX_HEIGHT {
                (*head).link(level).store(tail, false);
            }
        }
        SkipList {
            head,
            max_level,
            len: AtomicUsize::new(0),
        }
    }

    /// Number of live entries (relaxed; advisory).
    #[inline]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn key_lt(node_key: &BoundKey, target: &[u8]) -> bool {
        match node_key {
            BoundKey::NegInf => true,
            BoundKey::PosInf => false,
            BoundKey::Fin(k) => k.as_bytes().cmp(target) == CmpOrdering::Less,
        }
    }
    #[inline]
    fn key_eq(node_key: &BoundKey, target: &[u8]) -> bool {
        match node_key {
            BoundKey::Fin(k) => k.as_bytes() == target,
            _ => false,
        }
    }

    /// Per-thread geometric level with p = 1/4. Returns a 0-based top index in
    /// `[0, max_level]`.
    fn random_level(&self) -> usize {
        thread_local! {
            // Seed per thread from a monotonically increasing counter mixed
            // with a constant; avoids the unstable `ThreadId::as_u64`.
            static RNG: Cell<u64> = {
                static SEED: AtomicUsize = AtomicUsize::new(1);
                let s = SEED.fetch_add(1, Ordering::Relaxed) as u64;
                Cell::new(0x243F_6A88_85A3_08D3 ^ s.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            };
        }
        RNG.with(|r| random_geometric_level(r, self.max_level + 1) - 1)
    }

    /// Core search: fills `preds`/`succs` for every level and returns whether an
    /// exact match exists at the bottom level. Helps unlink marked nodes it
    /// passes (cooperative cleanup), so the structure self-heals.
    ///
    /// # Safety
    /// `self.head`/`tail` and all reachable nodes are valid; caller must hold a
    /// live `Guard` so no traversed node is reclaimed mid-search.
    unsafe fn find(
        &self,
        key: &[u8],
        preds: &mut [*mut Node; MAX_HEIGHT],
        succs: &mut [*mut Node; MAX_HEIGHT],
    ) -> bool {
        let backoff = Backoff::new();
        'retry: loop {
            let mut pred = self.head;
            for level in (0..=self.max_level).rev() {
                // SAFETY: `pred` always has this level (head has all; any node
                // reached at `level` was linked at `level`).
                let mut curr = unsafe { (*pred).link(level).reference() };
                loop {
                    let (mut succ, mut marked) = unsafe { (*curr).link(level).get() };
                    // Start pulling the forward node's cache line in now; we'll
                    // likely hop to it, and the latency hides behind the compare.
                    prefetch(succ);
                    // Splice out a run of logically-deleted successors.
                    while marked {
                        if !unsafe { (*pred).link(level).cas(curr, succ, false, false) } {
                            backoff.spin();
                            continue 'retry; // structure changed under us
                        }
                        curr = succ;
                        let g = unsafe { (*curr).link(level).get() };
                        succ = g.0;
                        marked = g.1;
                    }
                    if Self::key_lt(unsafe { &(*curr).key }, key) {
                        pred = curr;
                        curr = succ;
                    } else {
                        break;
                    }
                }
                preds[level] = pred;
                succs[level] = curr;
                // Distance-2 (down-level) prefetch: we're about to drop to
                // `level-1` and resume from `pred`'s link there. Fetch that node
                // now so its line is in flight during this iteration's
                // bookkeeping — measured a small but consistent read win.
                if level > 0 {
                    prefetch(unsafe { (*pred).link(level - 1).reference() });
                }
            }
            // SAFETY: succs[0] is a real node or `tail`; key_eq handles PosInf.
            return Self::key_eq(unsafe { &(*succs[0]).key }, key);
        }
    }

    /// Insert or update `key`. Returns `true` if a new node was created, `false`
    /// if an existing value was overwritten. The caller-supplied `guard` keeps
    /// any displaced value cell alive for concurrent readers.
    pub fn insert(&self, key: &[u8], val: &[u8], guard: &Guard) -> bool {
        let top_level = self.random_level();
        let mut preds = [std::ptr::null_mut(); MAX_HEIGHT];
        let mut succs = [std::ptr::null_mut(); MAX_HEIGHT];
        let backoff = Backoff::new();

        loop {
            let found = unsafe { self.find(key, &mut preds, &mut succs) };
            if found {
                // Update path: swap the value cell atomically, retire the old.
                let node = succs[0];
                let newv = Box::into_raw(Box::new(ValueRecord {
                    bytes: val.to_vec().into_boxed_slice(),
                }));
                let old = unsafe { (*node).value.swap(newv, Ordering::AcqRel) };
                if !old.is_null() {
                    // SAFETY: `old` just unlinked from `node.value`; defer free.
                    unsafe { guard.retire(old as *mut u8, drop_value) };
                }
                return false;
            }

            // Fresh node, levels 0..=top_level pre-pointed at the found succs.
            let valrec = Box::into_raw(Box::new(ValueRecord {
                bytes: val.to_vec().into_boxed_slice(),
            }));
            let node = Node::new(BoundKey::Fin(KeyRepr::new(key)), valrec, top_level);
            #[allow(clippy::needless_range_loop)] // index addresses node + succs
            for level in 0..=top_level {
                unsafe { (*node).link(level).store(succs[level], false) };
            }

            // Publish at the bottom level first; this is the linearization point.
            let pred = preds[0];
            let succ = succs[0];
            if !unsafe { (*pred).link(0).cas(succ, node, false, false) } {
                // Lost the race: free our node (and its value) and retry.
                // SAFETY: `node` was never published; we own it solely.
                unsafe { free_node(node) };
                backoff.spin();
                continue;
            }

            // Link the upper levels, re-finding on every contended CAS. The
            // index `level` addresses three parallel arrays (preds/succs/node
            // links), so a range loop is the clearest form here.
            #[allow(clippy::needless_range_loop)]
            for level in 1..=top_level {
                loop {
                    let pred = preds[level];
                    let succ = succs[level];
                    unsafe { (*node).link(level).store(succ, false) };
                    if unsafe { (*pred).link(level).cas(succ, node, false, false) } {
                        break;
                    }
                    backoff.spin();
                    unsafe { self.find(key, &mut preds, &mut succs) };
                }
            }

            self.len.fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }

    /// Point lookup. Returns a borrow into the value cell whose lifetime is tied
    /// to the `guard` — the zero-copy read. EBR guarantees the cell is not freed
    /// while the guard is pinned, so the borrow can never dangle.
    pub fn get<'g>(&self, key: &[u8], guard: &'g Guard) -> Option<&'g [u8]> {
        let _ = guard; // borrow proves a pin is held for the whole read
        let mut pred = self.head;
        unsafe {
            for level in (0..=self.max_level).rev() {
                let mut curr = (*pred).link(level).reference();
                loop {
                    let (mut succ, mut marked) = (*curr).link(level).get();
                    // Prefetch the forward node while we compare the current one.
                    prefetch(succ);
                    while marked {
                        curr = succ;
                        let g = (*curr).link(level).get();
                        succ = g.0;
                        marked = g.1;
                    }
                    if Self::key_lt(&(*curr).key, key) {
                        pred = curr;
                        curr = succ;
                    } else {
                        break;
                    }
                }
                // Distance-2 (down-level) prefetch — see `find`.
                if level > 0 {
                    prefetch((*pred).link(level - 1).reference());
                }
                // `curr` is the level-`level` candidate; descend.
                if level == 0 {
                    if Self::key_eq(&(*curr).key, key) {
                        let (_, marked) = (*curr).link(0).get();
                        if marked {
                            return None; // logically deleted
                        }
                        let vp = (*curr).value.load(Ordering::Acquire);
                        if vp.is_null() {
                            return None;
                        }
                        // Lifetime widened to `'g`: sound because `guard` keeps
                        // the cell alive (EBR) for at least as long as `'g`.
                        let bytes: &'g [u8] = &(*vp).bytes;
                        return Some(bytes);
                    }
                    return None;
                }
            }
        }
        None
    }

    /// Remove `key`. Returns `true` if this call performed the removal.
    pub fn remove(&self, key: &[u8], guard: &Guard) -> bool {
        let mut preds = [std::ptr::null_mut(); MAX_HEIGHT];
        let mut succs = [std::ptr::null_mut(); MAX_HEIGHT];

        let found = unsafe { self.find(key, &mut preds, &mut succs) };
        if !found {
            return false;
        }
        let node = succs[0];

        unsafe {
            // Mark all upper levels deleted (top-down) so searchers stop routing
            // through this node. `attempt_mark` retries are handled inline.
            for level in (1..=(*node).top_level).rev() {
                let (mut succ, mut marked) = (*node).link(level).get();
                while !marked {
                    (*node).link(level).attempt_mark(succ, true);
                    let g = (*node).link(level).get();
                    succ = g.0;
                    marked = g.1;
                }
            }

            // Bottom level: the CAS that flips the mark is the removal's
            // linearization point. Exactly one remover succeeds. We spin only on
            // the bottom-level mark CAS; the upper levels are already marked.
            let (mut succ, _) = (*node).link(0).get();
            loop {
                let i_marked = (*node).link(0).cas(succ, succ, false, true);
                let g = (*node).link(0).get();
                let marked = g.1;
                if i_marked {
                    // We won. Trigger physical unlink (find self-heals) and
                    // retire the node for epoch-safe reclamation.
                    self.find(key, &mut preds, &mut succs);
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    guard.retire(node as *mut u8, drop_node);
                    return true;
                } else if marked {
                    return false; // someone else removed it first
                }
                succ = g.0; // CAS failed because succ moved; retry the mark
            }
        }
    }
}

impl Drop for SkipList {
    fn drop(&mut self) {
        // Single-threaded teardown: walk level 0 freeing every node, including
        // the sentinels. Marks are irrelevant here.
        let mut cur = self.head;
        while !cur.is_null() {
            // SAFETY: exclusive ownership at drop; each node freed once.
            let next = unsafe { Link::ptr_of((*cur).link(0).p.load(Ordering::Relaxed)) };
            unsafe { free_node(cur) };
            cur = next;
        }
    }
}

//! **Self-referential write-ahead log** over a pre-allocated, page-aligned
//! buffer, with manual `Drop` and `Pin` to model an `O_DIRECT` I/O region.
//!
//! Durable stores need a sequential log. We pre-allocate one aligned buffer up
//! front (no per-append allocation, no GC) and append records into it. The
//! struct is *self-referential*: alongside the owning buffer it stores raw
//! pointers (`*const u8`) that point **into that same buffer** — the record
//! index. Rust's borrow checker forbids a safe `&self`-into-`self` field, so we:
//!
//!   * keep the index as raw pointers, materializing `&[u8]` on demand bound to
//!     `&self` (so a record borrow can never outlive the log);
//!   * mark the type `!Unpin` via [`PhantomPinned`] and only ever hand it out as
//!     `Pin<Box<Self>>`, so it cannot be moved — modeling an `O_DIRECT` buffer
//!     whose address the kernel DMA engine has latched onto;
//!   * implement `Drop` manually to `dealloc` the buffer exactly once, with no
//!     leak and no double-free.
//!
//! The page-aligned (4096) allocation is exactly what `O_DIRECT` / unbuffered
//! writes require (buffer base and length must be sector/page aligned).

use std::alloc::{self, Layout};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomPinned;
use std::path::Path;
use std::pin::Pin;
use std::ptr::NonNull;

/// Alignment required for `O_DIRECT`-style unbuffered I/O on typical systems.
const PAGE: usize = 4096;

/// A pinned, pre-allocated, append-only log buffer with a self-referential
/// record index.
pub struct MappedLog {
    /// Owning, page-aligned backing allocation of `cap` bytes.
    base: NonNull<u8>,
    cap: usize,
    /// Append cursor (bytes written).
    len: usize,
    /// Self-referential index: each entry is `(ptr_into_base, len)`. These raw
    /// pointers alias `base`; this is the crux of the self-reference.
    spans: Vec<(*const u8, usize)>,
    /// Backing file handle kept alive for the buffer's lifetime.
    file: File,
    /// Opt out of `Unpin`: the struct must not move once constructed.
    _pin: PhantomPinned,
}

// SAFETY: the raw `*const u8` spans only address bytes inside `base`, which this
// struct uniquely owns until `Drop`. There is no shared mutability, so sending
// the (pinned) log to another thread is sound.
unsafe impl Send for MappedLog {}

impl MappedLog {
    /// Open (or create) `path` and map a `cap`-byte (rounded up to a page)
    /// pre-allocated region, loading any existing file contents into it.
    pub fn open<P: AsRef<Path>>(path: P, cap: usize) -> std::io::Result<Pin<Box<Self>>> {
        let cap = (cap + PAGE - 1) & !(PAGE - 1);
        let layout = Layout::from_size_align(cap, PAGE).expect("layout");
        // SAFETY: cap > 0 and PAGE is a power of two.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let base = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false) // a WAL is append/replay; never clobber on open
            .open(path)?;

        // Load existing contents (replay material for the WAL).
        let mut existing = Vec::new();
        file.read_to_end(&mut existing)?;
        let len = existing.len().min(cap);
        if len > 0 {
            // SAFETY: `base` has `cap >= len` bytes; regions are disjoint.
            unsafe { std::ptr::copy_nonoverlapping(existing.as_ptr(), base.as_ptr(), len) };
        }
        file.seek(SeekFrom::End(0))?;

        // Box::pin freezes the address; internal pointers we create below remain
        // valid for the lifetime of the box.
        Ok(Box::pin(MappedLog {
            base,
            cap,
            len,
            spans: Vec::new(),
            file,
            _pin: PhantomPinned,
        }))
    }

    /// Append `bytes` to the log, recording a self-referential span. Returns the
    /// record's index, or `None` if the pre-allocated region is full.
    ///
    /// Takes `Pin<&mut Self>`: we can mutate through the pin (the data does not
    /// move), but the pin guarantees the buffer's address is stable for the DMA
    /// engine / our own stored pointers.
    pub fn append(self: Pin<&mut Self>, bytes: &[u8]) -> Option<usize> {
        // SAFETY: we never move out of `this`; we only mutate fields in place,
        // which preserves the pinning invariant.
        let this = unsafe { self.get_unchecked_mut() };
        if this.len + bytes.len() > this.cap {
            return None;
        }
        // SAFETY: dst is within `[base, base+cap)`, disjoint from src.
        let dst = unsafe { this.base.as_ptr().add(this.len) };
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
        this.spans.push((dst as *const u8, bytes.len()));
        this.len += bytes.len();
        Some(this.spans.len() - 1)
    }

    /// Borrow a previously appended record. The returned slice borrows `&self`,
    /// so it can never outlive the log — the self-reference is made safe by
    /// tying its lifetime to the owner.
    pub fn record(&self, index: usize) -> Option<&[u8]> {
        let &(ptr, len) = self.spans.get(index)?;
        // SAFETY: `ptr` points into `base` (we put it there in `append`) and
        // remains valid because the buffer is pinned and owned by `self`.
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Persist the populated prefix to the backing file (analogous to `msync`).
    pub fn flush(self: Pin<&mut Self>) -> std::io::Result<()> {
        // SAFETY: in-place mutation only.
        let this = unsafe { self.get_unchecked_mut() };
        // SAFETY: `base[..len]` is initialized (zeroed + appended).
        let data = unsafe { std::slice::from_raw_parts(this.base.as_ptr(), this.len) };
        this.file.seek(SeekFrom::Start(0))?;
        this.file.write_all(data)?;
        this.file.flush()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for MappedLog {
    fn drop(&mut self) {
        // Manual, single deallocation of the page-aligned buffer — no leak, no
        // double free. The `spans` (raw pointers into `base`) are simply
        // forgotten; they never owned anything.
        let layout = Layout::from_size_align(self.cap, PAGE).expect("layout");
        // SAFETY: `base`/`cap`/`PAGE` reproduce the exact allocation layout, and
        // we hold unique ownership at drop time.
        unsafe { alloc::dealloc(self.base.as_ptr(), layout) };
    }
}

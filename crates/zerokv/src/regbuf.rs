//! **Registered fixed-buffer pool** — readiness for `io_uring`
//! `IORING_REGISTER_BUFFERS` (Linux) and Windows RIO `RIORegisterBuffer`.
//!
//! Modern zero-syscall-overhead I/O works by *registering* a fixed set of
//! page-aligned buffers with the kernel **once**. The kernel pins the pages
//! (`get_user_pages`), records their physical mapping, and returns an index per
//! buffer. Subsequent `READ_FIXED`/`WRITE_FIXED` (io_uring) or
//! `RIOReceive`/`RIOSend` (Windows RIO) reference the buffer *by index*, so the
//! per-I/O path skips:
//!   * the page-table walk to translate the user address, and
//!   * the page refcount/pin dance that an unregistered `read`/`pread` pays.
//!
//! The registration contract is exactly the invariant the rest of `zerokv`
//! already upholds: **a registered buffer must never move or be freed while it
//! stays registered**. That is `PhantomPinned` + `Pin<Box<Self>>` here, mirroring
//! the fixed arena (`reset` instead of per-record `free`) and the pinned async
//! I/O buffers in [`crate::executor`].
//!
//! This module keeps a *portable* backend so the engine builds and is testable
//! on any OS; the `// NATIVE io_uring:` / `// NATIVE Windows RIO:` comments mark
//! the exact spots where the real submission calls drop in unchanged, because
//! the buffer ownership/lifetime model already matches.

use std::alloc::{self, Layout};
use std::fs::File;
use std::io;
use std::marker::PhantomPinned;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::Mutex;

/// Page alignment: required for `O_DIRECT` and the natural granularity for
/// registered buffers.
const PAGE: usize = 4096;

/// A pool of pre-registered, page-aligned fixed buffers carved from a single
/// aligned slab — the analogue of one `IORING_REGISTER_BUFFERS` call over an
/// `iovec` array, or one RIO buffer registration.
///
/// The whole slab is allocated once and pinned; individual buffers are handed
/// out as [`FixedBuf`] handles that carry their `buf_index`.
pub struct RegisteredBuffers {
    /// One contiguous, page-aligned, registered slab.
    base: NonNull<u8>,
    /// Size of each individual buffer (page-rounded).
    buf_len: usize,
    /// Number of buffers in the slab.
    count: usize,
    /// Free indices (cold path; acquisition/release is not on the I/O hot loop).
    free: Mutex<Vec<u16>>,
    /// The slab address is registered with the kernel: it must not move.
    _pin: PhantomPinned,
}

// SAFETY: the slab is uniquely owned; buffer hand-out is serialized via the
// free-list mutex; the raw `base` only addresses bytes inside the owned slab.
unsafe impl Send for RegisteredBuffers {}
unsafe impl Sync for RegisteredBuffers {}

impl RegisteredBuffers {
    /// Allocate and "register" `count` buffers of `buf_len` (rounded up to a
    /// page) each. Returned pinned so the slab address — the thing the kernel
    /// latches onto — is immovable.
    ///
    /// `count` must fit in a `u16` (io_uring's `buf_index` is 16-bit).
    pub fn register(count: usize, buf_len: usize) -> Pin<Box<Self>> {
        assert!(count > 0 && count <= u16::MAX as usize, "count out of range");
        let buf_len = (buf_len.max(1) + PAGE - 1) & !(PAGE - 1);
        let total = count.checked_mul(buf_len).expect("slab size overflow");
        let layout = Layout::from_size_align(total, PAGE).expect("layout");
        // Zeroed so `as_slice` on an untouched buffer is well-defined.
        // SAFETY: total > 0, PAGE is a power of two.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let base = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));

        // NATIVE io_uring: after this allocation, build one `iovec` per buffer
        // (base + i*buf_len, buf_len) and call
        //   io_uring_register_buffers(&ring, iovecs, count)
        // NATIVE Windows RIO: RIORegisterBuffer(base, total) once, then derive
        // per-buffer RIO_BUF { BufferId, Offset = i*buf_len, Length = buf_len }.

        Box::pin(RegisteredBuffers {
            base,
            buf_len,
            count,
            free: Mutex::new((0..count as u16).collect()),
            _pin: PhantomPinned,
        })
    }

    /// Borrow a free fixed buffer, or `None` if all are in flight. The returned
    /// handle's `index()` is the kernel `buf_index`.
    pub fn acquire(&self) -> Option<FixedBuf<'_>> {
        let index = self.free.lock().unwrap().pop()?;
        // SAFETY: index < count, so `base + index*buf_len` is within the slab and
        // disjoint from every other live `FixedBuf` (an index is handed out at
        // most once until released).
        let ptr = unsafe { NonNull::new_unchecked(self.base.as_ptr().add(index as usize * self.buf_len)) };
        Some(FixedBuf {
            pool: self,
            index,
            ptr,
            len: self.buf_len,
        })
    }

    fn release(&self, index: u16) {
        self.free.lock().unwrap().push(index);
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }
    #[inline]
    pub fn buf_len(&self) -> usize {
        self.buf_len
    }
    /// Number of buffers currently available (advisory).
    pub fn available(&self) -> usize {
        self.free.lock().unwrap().len()
    }
}

impl Drop for RegisteredBuffers {
    fn drop(&mut self) {
        // NATIVE io_uring: io_uring_unregister_buffers(&ring) BEFORE this free,
        // so the kernel releases its page pins first.
        let layout = Layout::from_size_align(self.count * self.buf_len, PAGE).expect("layout");
        // SAFETY: exact layout, unique ownership at drop.
        unsafe { alloc::dealloc(self.base.as_ptr(), layout) };
    }
}

/// A handle to one registered buffer. Carries the kernel `buf_index`. While this
/// handle lives the buffer is "checked out": its bytes are stable and it will
/// not be re-handed-out, matching the "kernel owns the buffer until completion"
/// rule. Dropping it returns the slot to the pool.
pub struct FixedBuf<'a> {
    pool: &'a RegisteredBuffers,
    index: u16,
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: a `FixedBuf` is an exclusive checkout of a disjoint slab region.
unsafe impl Send for FixedBuf<'_> {}

impl FixedBuf<'_> {
    /// The `buf_index` to pass to `io_uring_prep_read_fixed` / `..._write_fixed`.
    #[inline]
    pub fn index(&self) -> u16 {
        self.index
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: exclusive, initialized (zeroed) region of `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: exclusive borrow of our own slab region.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for FixedBuf<'_> {
    fn drop(&mut self) {
        self.pool.release(self.index);
    }
}

/// Pluggable submission backend. The *seam* where a native io_uring/RIO
/// submitter replaces the portable fallback without touching callers.
pub trait FixedIoBackend {
    /// Read `len` bytes from `file` at `offset` into the registered buffer.
    /// Returns the number of bytes read.
    fn read_fixed(
        &self,
        file: &File,
        offset: u64,
        buf: &mut FixedBuf<'_>,
        len: usize,
    ) -> io::Result<usize>;

    /// **Vectored** append of many buffers in one submission — the Group-Commit
    /// fast path. The portable impl emulates `writev`; the native path issues a
    /// single `io_uring_prep_writev`, consolidating hundreds of transactions
    /// into one SSD write. Returns total bytes written.
    fn write_vectored(&self, file: &File, offset: u64, bufs: &[io::IoSlice<'_>]) -> io::Result<usize>;
}

/// Portable backend using positional reads (`pread`/`seek_read`). Correct
/// everywhere; pays the page-table walk that the native fixed path elides.
pub struct PortableBackend;

impl FixedIoBackend for PortableBackend {
    fn read_fixed(
        &self,
        file: &File,
        offset: u64,
        buf: &mut FixedBuf<'_>,
        len: usize,
    ) -> io::Result<usize> {
        let len = len.min(buf.len);
        let dst = &mut buf.as_mut_slice()[..len];

        // NATIVE io_uring (replaces the positional read below):
        //   let sqe = ring.get_sqe();
        //   io_uring_prep_read_fixed(sqe, fd, dst.as_mut_ptr(), len, offset,
        //                            buf.index() as i32);
        //   ring.submit(); // then await the CQE — the Future's Pending state
        // NATIVE Windows RIO:
        //   RIO_BUF rb { BufferId, Offset: index*buf_len, Length: len };
        //   RIOReceiveEx(rq, &rb, 1, ...); // completion via IOCP

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            file.read_at(dst, offset)
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            file.seek_read(dst, offset)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (file, offset);
            Ok(0)
        }
    }

    fn write_vectored(&self, file: &File, offset: u64, bufs: &[io::IoSlice<'_>]) -> io::Result<usize> {
        // NATIVE io_uring (replaces the loop below):
        //   let sqe = ring.get_sqe();
        //   io_uring_prep_writev(sqe, fd, iovecs.as_ptr(), iovecs.len(), offset);
        //   ring.submit_and_wait(1); // one syscall for the whole batch
        //
        // Portable emulation: positional writes, advancing the offset. Still a
        // single logical group commit from the callers' perspective — they all
        // wake only after this returns.
        let mut pos = offset;
        let mut total = 0usize;
        for slice in bufs {
            let n = write_all_at(file, slice, pos)?;
            pos += n as u64;
            total += n;
        }
        Ok(total)
    }
}

/// Positional `write_all` (portable over the platform `*_at` APIs).
fn write_all_at(file: &File, mut data: &[u8], mut offset: u64) -> io::Result<usize> {
    let total = data.len();
    while !data.is_empty() {
        #[cfg(unix)]
        let n = {
            use std::os::unix::fs::FileExt;
            file.write_at(data, offset)?
        };
        #[cfg(windows)]
        let n = {
            use std::os::windows::fs::FileExt;
            file.seek_write(data, offset)?
        };
        #[cfg(not(any(unix, windows)))]
        let n = {
            let _ = (file, offset);
            data.len()
        };
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "write_all_at: zero-length write"));
        }
        data = &data[n..];
        offset += n as u64;
    }
    Ok(total)
}

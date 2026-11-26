//! Read-only memory mapping of a GGUF file.
//!
//! This avoids copying multi-gigabyte tensor data into the heap: the OS maps
//! the file pages on demand and can evict them under memory pressure.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result};

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

const MAP_FAILED: usize = usize::MAX;

/// A read-only mapping of an entire file.
pub struct Mmap {
    ptr: *mut u8,
    len: usize,
}

impl Mmap {
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(Error::InvalidGguf("empty file".into()));
        }
        // SAFETY: read-only private mapping; the pointer stays valid for the
        // lifetime of this struct because the mapping is never truncated.
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
        if ptr as usize == MAP_FAILED {
            return Err(Error::InvalidGguf("mmap failed".into()));
        }
        Ok(Arc::new(Self { ptr, len }))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn slice(&self, offset: usize, len: usize) -> Result<&[u8]> {
        if offset.checked_add(len).map_or(true, |end| end > self.len) {
            return Err(Error::InvalidGguf("mmap slice out of range".into()));
        }
        // SAFETY: bounds checked above; mapping is read-only and immutable.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) })
    }
}

// SAFETY: the mapping is read-only and never mutated; sharing it across
// threads is safe.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr, self.len);
        }
    }
}

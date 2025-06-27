#![cfg_attr(deny_warnings, deny(warnings))]
#![cfg_attr(deny_warnings, deny(missing_docs))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

//! Rust bindings for blosc - a blocking, shuffling and lossless compression library.
//!
//! Provide a safe interface to the [blosc](https://github.com/Blosc/c-blosc) library.
//!
//! # Getting Started
//!
//! To use this library, add the following to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! blosc-rs = "0.1"
//!
//! # Or alternatively, rename the crate to `blosc`
//! blosc = { package = "blosc-rs", version = "0.1" }
//! ```
//!
//! In the following example we compress a vector of integers and then decompress it back:
//! ```rust
//! use blosc_rs::{CLevel, CompressAlgo, Shuffle, compress, decompress};
//!
//! let data: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];
//! let data_bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<i32>()) };
//! let numinternalthreads = 4;
//! let compressed = compress(
//!     CLevel::L5,
//!     Shuffle::Byte,
//!     std::mem::size_of::<i32>(), // itemsize
//!     data_bytes,
//!     &CompressAlgo::Blosclz,
//!     None, // automatic block size
//!     numinternalthreads,
//! ).unwrap();
//! let decompressed = decompress(
//!     &compressed,
//!     numinternalthreads,
//! ).unwrap();
//! // SAFETY: we know the data is of type i32
//! let decompressed: &[i32] = unsafe { std::slice::from_raw_parts(decompressed.as_ptr() as *const i32, decompressed.len() / std::mem::size_of::<i32>()) };
//! assert_eq!(data, *decompressed);
//! ```

use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;

/// Compress a block of data in the `src` buffer and returns the compressed data.
///
/// Note that this function allocates a new `Vec<u8>` for the compressed data, with the maximum possible size required
/// for the compressed data, which may be larger than the actual compressed data size. If this function is used in a
/// critical performance path, consider using `compress_into` instead, which allows you to provide a pre-allocated
/// buffer, which can be used repeatedly without the overhead of allocations.
///
/// # Arguments
///
/// * `clevel`: The desired compression level.
/// * `shuffle`: Specifies which (if any) shuffle compression filters should be applied.
/// * `typesize`: The number of bytes for the atomic type in the binary `src` buffer. This is mainly useful for the
///   shuffle filters. For implementation reasons, only a `typesize` in the range 1 < `typesize` < 256 will allow the
///   shuffle filter to work. When `typesize` is not in this range, shuffle will be silently disabled.
/// * `src`: The source data to compress.
/// * `compressor`: The compression algorithm to use.
/// * `blocksize`: Optional block size for compression. If `None`, an automatic block size will be used.
/// * `numinternalthreads`: The number of threads to use internally.
///
/// # Returns
///
/// A `Result` containing the compressed data as a `Vec<u8>`, or a `CompressError` if an error occurs.
pub fn compress(
    clevel: CLevel,
    shuffle: Shuffle,
    typesize: usize,
    src: &[u8],
    compressor: &CompressAlgo,
    blocksize: Option<NonZeroUsize>,
    numinternalthreads: u32,
) -> Result<Vec<u8>, CompressError> {
    let dst_max_len = src.len() + blosc_rs_sys::BLOSC_MAX_OVERHEAD as usize;
    let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(dst_max_len);
    unsafe { dst.set_len(dst_max_len) };

    let len = compress_into(
        clevel,
        shuffle,
        typesize,
        src,
        dst.as_mut_slice(),
        compressor,
        blocksize,
        numinternalthreads,
    )?;
    assert!(len <= dst_max_len);
    unsafe { dst.set_len(len) };
    // SAFETY: every element from 0 to len was initialized
    let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
    Ok(vec)
}

/// Compress a block of data in the `src` buffer into the `dst` buffer.
///
/// # Arguments
///
/// * `clevel`: The desired compression level.
/// * `shuffle`: Specifies which (if any) shuffle compression filters should be applied.
/// * `typesize`: The number of bytes for the atomic type in the binary `src` buffer. This is mainly useful for the
///   shuffle filters. For implementation reasons, only a `typesize` in the range 1 < `typesize` < 256 will allow the
///   shuffle filter to work. When `typesize` is not in this range, shuffle will be silently disabled.
/// * `src`: The source data to compress.
/// * `dst`: The destination buffer where the compressed data will be written.
/// * `compressor`: The compression algorithm to use.
/// * `blocksize`: Optional block size for compression. If `None`, an automatic block size will be used.
/// * `numinternalthreads`: The number of threads to use internally.
///
/// # Returns
///
/// A `Result` containing the number of bytes written to the `dst` buffer, or a `CompressError` if an error occurs.
#[allow(clippy::too_many_arguments)]
pub fn compress_into(
    clevel: CLevel,
    shuffle: Shuffle,
    typesize: usize,
    src: &[u8],
    dst: &mut [MaybeUninit<u8>],
    compressor: &CompressAlgo,
    blocksize: Option<NonZeroUsize>,
    numinternalthreads: u32,
) -> Result<usize, CompressError> {
    let status = unsafe {
        blosc_rs_sys::blosc_compress_ctx(
            clevel as i32 as std::ffi::c_int,
            shuffle as u32 as std::ffi::c_int,
            typesize,
            src.len(),
            src.as_ptr() as *const std::ffi::c_void,
            dst.as_mut_ptr() as *mut std::ffi::c_void,
            dst.len(),
            compressor.as_ref().as_ptr(),
            blocksize.map(|b| b.get()).unwrap_or(0),
            numinternalthreads as std::ffi::c_int,
        )
    };
    match status {
        len if len > 0 => {
            assert!(len as usize <= dst.len());
            Ok(len as usize)
        }
        0 => Err(CompressError::DestinationBufferTooSmall),
        _ => {
            debug_assert!(status < 0);
            Err(CompressError::InternalError(status))
        }
    }
}

/// Error that can occur during compression.
#[derive(thiserror::Error, Debug)]
pub enum CompressError {
    /// Error indicating that the destination buffer is too small to hold the compressed data.
    #[error("destination buffer is too small")]
    DestinationBufferTooSmall,
    /// blosc internal error.
    #[error("blosc internal error: {0}")]
    InternalError(i32),
}

/// Represents the compression levels used by Blosc.
///
/// The levels range from 0 to 9, where 0 is no compression and 9 is maximum compression.
#[allow(missing_docs)]
#[repr(i32)]
pub enum CLevel {
    L0 = 0,
    L1 = 1,
    L2 = 2,
    L3 = 3,
    L4 = 4,
    L5 = 5,
    L6 = 6,
    L7 = 7,
    L8 = 8,
    L9 = 9,
}
impl TryFrom<i32> for CLevel {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(CLevel::L0),
            1 => Ok(CLevel::L1),
            2 => Ok(CLevel::L2),
            3 => Ok(CLevel::L3),
            4 => Ok(CLevel::L4),
            5 => Ok(CLevel::L5),
            6 => Ok(CLevel::L6),
            7 => Ok(CLevel::L7),
            8 => Ok(CLevel::L8),
            9 => Ok(CLevel::L9),
            _ => Err(()),
        }
    }
}

/// Represents the shuffle filters used by Blosc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Shuffle {
    /// no shuffle
    None = blosc_rs_sys::BLOSC_NOSHUFFLE,
    /// byte-wise shuffle
    Byte = blosc_rs_sys::BLOSC_SHUFFLE,
    /// bit-wise shuffle
    Bit = blosc_rs_sys::BLOSC_BITSHUFFLE,
}

/// Represents the compression algorithms supported by Blosc.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum CompressAlgo {
    Blosclz,
    Lz4,
    Lz4hc,
    // Snappy,
    Zlib,
    Zstd,
    Other(CString),
}
impl AsRef<CStr> for CompressAlgo {
    fn as_ref(&self) -> &CStr {
        match self {
            CompressAlgo::Blosclz => c"blosclz",
            CompressAlgo::Lz4 => c"lz4",
            CompressAlgo::Lz4hc => c"lz4hc",
            // CompressAlgo::Snappy => c"snappy",
            CompressAlgo::Zlib => c"zlib",
            CompressAlgo::Zstd => c"zstd",
            CompressAlgo::Other(c) => c.as_ref(),
        }
    }
}

/// Decompress a block of compressed data in `src` and returns the decompressed data.
///
/// # Arguments
///
/// * `src`: The compressed data to decompress.
/// * `numinternalthreads`: The number of threads to use internally.
///
/// # Returns
///
/// A `Result` containing the decompressed data as a `Vec<u8>`, or a `DecompressError` if an error occurs.
pub fn decompress(src: &[u8], numinternalthreads: u32) -> Result<Vec<u8>, DecompressError> {
    let dst_len = validate_compressed_slice_and_get_uncompressed_len(src)
        .ok_or(DecompressError::DecompressingError)?;
    let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(dst_len);
    unsafe { dst.set_len(dst_len) };

    let len = unsafe { decompress_into_unchecked(src, dst.as_mut_slice(), numinternalthreads)? };
    assert!(len <= dst_len);
    unsafe { dst.set_len(len) };
    // SAFETY: every element from 0 to len was initialized
    let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
    Ok(vec)
}

/// Decompress a block of compressed data in `src` into the `dst` buffer.
///
/// # Arguments
///
/// * `src`: The compressed data to decompress.
/// * `dst`: The destination buffer where the decompressed data will be written.
/// * `numinternalthreads`: The number of threads to use internally.
///
/// # Returns
///
/// A `Result` containing the number of bytes written to the `dst` buffer, or a `DecompressError` if an error occurs.
pub fn decompress_into(
    src: &[u8],
    dst: &mut [MaybeUninit<u8>],
    numinternalthreads: u32,
) -> Result<usize, DecompressError> {
    let dst_len = validate_compressed_slice_and_get_uncompressed_len(src)
        .ok_or(DecompressError::DecompressingError)?;
    if dst.len() < dst_len {
        return Err(DecompressError::DestinationBufferTooSmall);
    }
    let len = unsafe { decompress_into_unchecked(src, dst, numinternalthreads)? };
    assert!(len <= dst_len);
    Ok(len)
}

unsafe fn decompress_into_unchecked(
    src: &[u8],
    dst: &mut [MaybeUninit<u8>],
    numinternalthreads: u32,
) -> Result<usize, DecompressError> {
    let status = unsafe {
        blosc_rs_sys::blosc_decompress_ctx(
            src.as_ptr() as *const std::ffi::c_void,
            dst.as_mut_ptr() as *mut std::ffi::c_void,
            dst.len(),
            numinternalthreads as std::ffi::c_int,
        )
    };
    match status {
        len if len >= 0 => Ok(len as usize),
        _ => Err(DecompressError::InternalError(status)),
    }
}

/// Error that can occur during decompression.
#[derive(thiserror::Error, Debug)]
pub enum DecompressError {
    /// Error indicating that the destination buffer is too small to hold the decompressed data.
    #[error("destination buffer is too small")]
    DestinationBufferTooSmall,
    /// Error indicating that the data could not be decompressed.
    #[error("failed to decompress the data")]
    DecompressingError,
    /// blosc internal error.
    #[error("blosc internal error: {0}")]
    InternalError(i32),
}

fn validate_compressed_slice_and_get_uncompressed_len(src: &[u8]) -> Option<usize> {
    let mut dst_len = 0;
    let status = unsafe {
        blosc_rs_sys::blosc_cbuffer_validate(
            src.as_ptr() as *const std::ffi::c_void,
            src.len(),
            &mut dst_len,
        )
    };
    if status < 0 {
        None
    } else {
        Some(dst_len)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::{CLevel, CompressAlgo, Shuffle};

    #[test]
    fn round_trip() {
        let mut rand = StdRng::seed_from_u64(0xb1ba0c326dc4dbba);

        for _ in 0..100 {
            let src_len = {
                let max_lens = [0x1, 0x10, 0x100, 0x1000, 0x10000, 0x100000];
                let max_len = max_lens[rand.random_range(0..max_lens.len())];
                rand.random_range(0..=max_len)
            };
            let src = (0..rand.random_range(0..=src_len))
                .map(|_| rand.random_range(0..=255) as u8)
                .collect::<Vec<u8>>();
            let clevel: CLevel = rand.random_range(0..=9).try_into().unwrap();
            let shuffle = {
                let shuffles = [Shuffle::None, Shuffle::Byte, Shuffle::Bit];
                shuffles[rand.random_range(0..shuffles.len())]
            };
            let typesize = (1..=8)
                .map(|i| rand.random_range(1..=(1 << (8 - i))))
                .find(|&ts| src.len() % ts == 0)
                .unwrap();
            let compressor = {
                let compressors = [
                    CompressAlgo::Blosclz,
                    CompressAlgo::Lz4,
                    CompressAlgo::Lz4hc,
                    // CompressAlgo::Snappy,
                    CompressAlgo::Zlib,
                    CompressAlgo::Zstd,
                ];
                compressors[rand.random_range(0..compressors.len())].clone()
            };
            let blocksize = {
                let blocksizes = [
                    Option::<NonZeroUsize>::None,
                    Some(1.try_into().unwrap()),
                    Some(64.try_into().unwrap()),
                    Some(4096.try_into().unwrap()),
                    Some(262144.try_into().unwrap()),
                    Some(rand.random_range(1..4096).try_into().unwrap()),
                ];
                blocksizes[rand.random_range(0..blocksizes.len())]
            };
            let numinternalthreads = rand.random_range(1..=16);

            let compressed = crate::compress(
                clevel,
                shuffle,
                typesize,
                &src,
                &compressor,
                blocksize,
                numinternalthreads,
            )
            .unwrap();

            let decompressed = crate::decompress(&compressed, numinternalthreads).unwrap();

            assert_eq!(src, decompressed);
        }
    }
}

#![cfg_attr(deny_warnings, deny(warnings))]
#![cfg_attr(deny_warnings, deny(missing_docs))]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

//! Rust bindings for blosc - a blocking, shuffling and lossless compression library.
//!
//! Provide a safe interface to the [blosc](https://github.com/Blosc/c-blosc) library.
//! The crate has zero runtime dependencies.
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
//! use blosc_rs::{CompressAlgo, Encoder, Decoder};
//!
//! let data: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];
//!
//! let data_bytes = unsafe {
//!     std::slice::from_raw_parts(
//!         data.as_ptr() as *const u8,
//!         data.len() * std::mem::size_of::<i32>(),
//!     )
//! };
//! let numinternalthreads = 4;
//!
//! let compressed = Encoder::default()
//!     .typesize(std::mem::size_of::<i32>())
//!     .numinternalthreads(numinternalthreads)
//!     .compress(&data_bytes)
//!     .expect("failed to compress");
//!
//! let decoder = Decoder::new(&compressed).expect("invalid buffer");
//!
//! // Access some items using random random access, without decompressing the entire buffer
//! assert_eq!(&data_bytes[0..4], decoder.item(0).expect("failed to get the 0-th item"));
//! assert_eq!(&data_bytes[12..16], decoder.item(3).expect("failed to get the 3-th item"));
//! assert_eq!(&data_bytes[20..24], decoder.item(5).expect("failed to get the 5-th item"));
//!
//! // Decompress the entire buffer
//! let decompressed = decoder.decompress(numinternalthreads).expect("failed to decompress");
//! // SAFETY: we know the data is of type i32
//! let decompressed: &[i32] = unsafe {
//!     std::slice::from_raw_parts(
//!         decompressed.as_ptr() as *const i32,
//!         decompressed.len() / std::mem::size_of::<i32>(),
//!     )
//! };
//!
//! assert_eq!(data, *decompressed);
//! ```

use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::io::Read;
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;

/// The version of the underlying C-blosc library used by this crate.
pub const BLOSC_C_VERSION: &str = {
    let version = match CStr::from_bytes_until_nul(blosc_sys::BLOSC_VERSION_STRING) {
        Ok(v) => v,
        Err(_) => unreachable!(),
    };
    match version.to_str() {
        Ok(s) => s,
        Err(_) => unreachable!(),
    }
};

/// Encoder for Blosc compression.
///
/// This struct is not the usual stream-like encoder, but rather a configuration builder for the Blosc compression.
/// This is because blosc is not a streaming compression library, and it operate on the entire data buffer at once.
pub struct Encoder {
    clevel: CLevel,
    shuffle: Shuffle,
    typesize: usize,
    compressor: CompressAlgo,
    blocksize: Option<NonZeroUsize>,
    numinternalthreads: u32,
}
impl Default for Encoder {
    fn default() -> Self {
        Self::new(CLevel::L9)
    }
}
impl Encoder {
    /// Create a new encoder with the specified compression level.
    pub fn new(level: CLevel) -> Self {
        Self {
            clevel: level,
            shuffle: Shuffle::Byte,
            typesize: 1,
            compressor: CompressAlgo::Blosclz,
            blocksize: None,
            numinternalthreads: 1,
        }
    }

    /// Sets the compression level for the encoder.
    pub fn level(&mut self, level: CLevel) -> &mut Self {
        self.clevel = level;
        self
    }

    /// Sets which (if any) shuffle compression filters should be applied.
    ///
    /// By default, the shuffle filter is set to `Shuffle::Byte`.
    pub fn shuffle(&mut self, shuffle: Shuffle) -> &mut Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets the typesize for the encoder.
    ///
    /// This is the number of bytes for the atomic type in the binary `src` buffer.
    /// For implementation reasons, only a `typesize` in the range 1 < `typesize` < 256 will allow the
    /// shuffle filter to work. When `typesize` is not in this range, shuffle will be silently disabled.
    ///
    /// By default, the typesize is set to 1.
    pub fn typesize(&mut self, typesize: usize) -> &mut Self {
        self.typesize = typesize;
        self
    }

    /// Sets the compression algorithm to use.
    ///
    /// By default, the compression algorithm is set to `CompressAlgo::Blosclz`.
    pub fn compressor(&mut self, compressor: CompressAlgo) -> &mut Self {
        self.compressor = compressor;
        self
    }

    /// Sets the block size for compression.
    ///
    /// If `None`, an automatic block size will be used.
    /// By default, the block size is set to `None`.
    pub fn blocksize(&mut self, blocksize: Option<NonZeroUsize>) -> &mut Self {
        self.blocksize = blocksize;
        self
    }

    /// Sets the number of threads to use internally for compression.
    ///
    /// By default, the number of internal threads is set to 1.
    pub fn numinternalthreads(&mut self, numinternalthreads: u32) -> &mut Self {
        self.numinternalthreads = numinternalthreads;
        self
    }

    /// Compress a block of data in the `src` buffer and returns the compressed data.
    ///
    /// Note that this function allocates a new `Vec<u8>` for the compressed data with the maximum possible size
    /// required for it (uncompressed size + 16), which may be larger than whats actually needed. If this function is
    /// used in a critical performance path, consider using `compress_into` instead, allowing you to provide a
    /// pre-allocated buffer which can be used repeatedly without the overhead of allocations.
    pub fn compress(&self, src: &[u8]) -> Result<Vec<u8>, CompressError> {
        let dst_max_len = src.len() + blosc_sys::BLOSC_MAX_OVERHEAD as usize;
        let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(dst_max_len);
        unsafe { dst.set_len(dst_max_len) };

        let len = self.compress_into(src, dst.as_mut_slice())?;
        assert!(len <= dst_max_len);
        unsafe { dst.set_len(len) };
        // SAFETY: every element from 0 to len was initialized
        let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
        Ok(vec)
    }

    /// Compress a block of data in the `src` buffer into the `dst` buffer.
    pub fn compress_into(
        &self,
        src: &[u8],
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, CompressError> {
        let status = unsafe {
            blosc_sys::blosc_compress_ctx(
                self.clevel as i32 as std::ffi::c_int,
                self.shuffle as u32 as std::ffi::c_int,
                self.typesize,
                src.len(),
                src.as_ptr() as *const std::ffi::c_void,
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                dst.len(),
                self.compressor.as_ref().as_ptr(),
                self.blocksize.map(|b| b.get()).unwrap_or(0),
                self.numinternalthreads as std::ffi::c_int,
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
}

/// Error that can occur during compression.
#[derive(Debug)]
pub enum CompressError {
    /// Error indicating that the destination buffer is too small to hold the compressed data.
    DestinationBufferTooSmall,
    /// blosc internal error.
    InternalError(i32),
}
impl std::fmt::Display for CompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressError::DestinationBufferTooSmall => {
                f.write_str("destination buffer is too small")
            }
            CompressError::InternalError(status) => write!(f, "blosc internal error: {status}"),
        }
    }
}
impl std::error::Error for CompressError {}

/// Represents the compression levels used by Blosc.
///
/// The levels range from 0 to 9, where 0 is no compression and 9 is maximum compression.
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Shuffle {
    /// no shuffle
    None = blosc_sys::BLOSC_NOSHUFFLE,
    /// byte-wise shuffle
    Byte = blosc_sys::BLOSC_SHUFFLE,
    /// bit-wise shuffle
    Bit = blosc_sys::BLOSC_BITSHUFFLE,
}

/// Represents the compression algorithms supported by Blosc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

/// A decoder for Blosc compressed data.
///
/// This struct is not the usual stream-like decoder, but rather an array-like object that allows random access to
/// elements in the compressed data or decoding the entire buffer. This is because blosc is not a streaming library,
/// and it operates on the entire data buffer at once.
///
/// The compressed data is held in memory within the decoder, and decoding is done either by decompressing the entire
/// buffer, or by accessing individual items or ranges of items. In both cases, the decoder remains unchanged and only
/// the compressed data is held by it.
pub struct Decoder<'a> {
    src: Cow<'a, [u8]>,
    typesize: usize,
    dst_len: usize,
}
impl<'a> Decoder<'a> {
    /// Create a new decoder from a reader that contains Blosc compressed data.
    ///
    /// First a header of a fixed size is read from the reader, which contains metadata about the length of the
    /// compressed data. Then the rest of the compressed data is read into an in-memory internal buffer held by the
    /// decoder, and the reader is not used after this point.
    pub fn from_reader(reader: &mut impl Read) -> Result<Self, DecompressError> {
        // Read the header
        let mut header =
            [const { MaybeUninit::<u8>::uninit() }; blosc_sys::BLOSC_MIN_HEADER_LENGTH as usize];
        reader.read_exact(unsafe {
            std::mem::transmute::<&mut [MaybeUninit<u8>], &mut [u8]>(&mut header)
        })?;

        // Get the number of compressed bytes
        let mut nbytes = MaybeUninit::uninit();
        let mut cbytes = MaybeUninit::uninit();
        let mut blocksize = MaybeUninit::uninit();
        unsafe {
            blosc_sys::blosc_cbuffer_sizes(
                header.as_ptr() as *const std::ffi::c_void,
                nbytes.as_mut_ptr(),
                cbytes.as_mut_ptr(),
                blocksize.as_mut_ptr(),
            )
        };
        // let nbytes = unsafe { nbytes.assume_init() };
        // let blocksize = unsafe { blocksize.assume_init() };
        let cbytes = unsafe { cbytes.assume_init() };
        if cbytes == 0 {
            return Err(DecompressError::DecompressingError);
        }

        // Create a new buffer with all of the compressed data
        let mut src = Vec::<MaybeUninit<u8>>::with_capacity(cbytes);
        unsafe { src.set_len(cbytes) };
        // Copy the header to the new buffer
        src[..blosc_sys::BLOSC_MIN_HEADER_LENGTH as usize]
            .copy_from_slice(&header[..blosc_sys::BLOSC_MIN_HEADER_LENGTH as usize]);
        // Read the rest of the compressed data
        reader.read_exact(unsafe {
            std::mem::transmute::<&mut [MaybeUninit<u8>], &mut [u8]>(
                &mut src[blosc_sys::BLOSC_MIN_HEADER_LENGTH as usize..],
            )
        })?;
        let src = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(src) };

        Self::new(src)
    }

    /// Create a new decoder from a slice of Blosc compressed data.
    ///
    /// No decompression is performed at this point. Decompression is done on demand either by decompressing the
    /// entire buffer or by accessing individual items or ranges of items.
    pub fn new(src: impl Into<Cow<'a, [u8]>>) -> Result<Self, DecompressError> {
        let src: Cow<'a, [u8]> = src.into();

        // Validate
        let mut dst_len = 0;
        let status = unsafe {
            blosc_sys::blosc_cbuffer_validate(
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
                &mut dst_len,
            )
        };
        if status < 0 {
            return Err(DecompressError::DecompressingError);
        }

        let mut typesize = MaybeUninit::<usize>::uninit();
        let mut flags = MaybeUninit::<std::ffi::c_int>::uninit();
        unsafe {
            blosc_sys::blosc_cbuffer_metainfo(
                src.as_ptr() as *const std::ffi::c_void,
                typesize.as_mut_ptr(),
                flags.as_mut_ptr(),
            )
        };
        let typesize = unsafe { typesize.assume_init() };

        Ok(Self {
            src,
            typesize,
            dst_len,
        })
    }

    /// Decompress the entire buffer and return the decompressed data as a `Vec<u8>`.
    ///
    /// # Arguments
    ///
    /// * `numinternalthreads`: The number of threads to use internally.
    ///
    /// # Returns
    ///
    /// A `Result` containing the decompressed data as a `Vec<u8>`, or a `DecompressError` if an error occurs.
    pub fn decompress(&self, numinternalthreads: u32) -> Result<Vec<u8>, DecompressError> {
        let mut dst = Vec::<MaybeUninit<u8>>::with_capacity(self.dst_len);
        unsafe { dst.set_len(self.dst_len) };

        let len = self.decompress_into(dst.as_mut_slice(), numinternalthreads)?;

        assert!(len <= self.dst_len);
        unsafe { dst.set_len(len) };
        // SAFETY: every element from 0 to len was initialized
        let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
        Ok(vec)
    }

    /// Decompress the entire buffer and write the decompressed data into the provided destination buffer.
    ///
    /// # Arguments
    ///
    /// * `dst`: The destination buffer where the decompressed data will be written.
    /// * `numinternalthreads`: The number of threads to use internally.
    ///
    /// # Returns
    ///
    /// A `Result` containing the number of bytes written to the `dst` buffer, or a `DecompressError` if an error occurs.
    pub fn decompress_into(
        &self,
        dst: &mut [MaybeUninit<u8>],
        numinternalthreads: u32,
    ) -> Result<usize, DecompressError> {
        if dst.len() < self.dst_len {
            return Err(DecompressError::DestinationBufferTooSmall);
        }

        let status = unsafe {
            blosc_sys::blosc_decompress_ctx(
                self.src.as_ptr() as *const std::ffi::c_void,
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                dst.len(),
                numinternalthreads as std::ffi::c_int,
            )
        };
        match status {
            len if len >= 0 => {
                assert!(len as usize <= self.dst_len);
                Ok(len as usize)
            }
            _ => Err(DecompressError::InternalError(status)),
        }
    }

    /// Get the inner compressed data buffer.
    pub fn into_buf(self) -> Cow<'a, [u8]> {
        self.src
    }

    /// Get an element at the specified index.
    pub fn item(&self, idx: usize) -> Result<Vec<u8>, DecompressError> {
        self.items(idx..idx + 1)
    }

    /// Get an element at the specified index and copy it into the provided destination buffer.
    pub fn item_into(
        &self,
        idx: usize,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecompressError> {
        self.items_into(idx..idx + 1, dst)
    }

    /// Get a range of elements specified by the index range.
    pub fn items(&self, idx: std::ops::Range<usize>) -> Result<Vec<u8>, DecompressError> {
        let mut dst = vec![MaybeUninit::<u8>::uninit(); self.typesize * idx.len()];
        self.items_into(idx, &mut dst)?;
        // SAFETY: every element in dst is initialized
        Ok(unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) })
    }

    /// Get a range of elements specified by the index range and copy them into the provided destination buffer.
    pub fn items_into(
        &self,
        idx: std::ops::Range<usize>,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecompressError> {
        let required_len = self.typesize * idx.len();
        if dst.len() < required_len {
            return Err(DecompressError::DestinationBufferTooSmall);
        }
        let status = unsafe {
            blosc_sys::blosc_getitem(
                self.src.as_ptr() as *const std::ffi::c_void,
                idx.start as std::ffi::c_int,
                idx.len() as std::ffi::c_int,
                dst.as_mut_ptr() as *mut std::ffi::c_void,
            )
        };
        let dst_len = if status < 0 {
            return Err(DecompressError::DecompressingError);
        } else {
            status as usize
        };
        if dst_len != required_len {
            // Unexpected
            return Err(DecompressError::DecompressingError);
        }
        Ok(dst_len)
    }
}

/// Error that can occur during decompression.
#[derive(Debug)]
pub enum DecompressError {
    /// Error indicating that the destination buffer is too small to hold the decompressed data.
    DestinationBufferTooSmall,
    /// Error indicating that the data could not be decompressed.
    DecompressingError,
    /// blosc internal error.
    InternalError(i32),
    /// An I/O error occurred while reading the compressed data.
    IoError(std::io::Error),
}
impl From<std::io::Error> for DecompressError {
    fn from(err: std::io::Error) -> Self {
        DecompressError::IoError(err)
    }
}
impl std::fmt::Display for DecompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecompressError::DestinationBufferTooSmall => {
                f.write_str("destination buffer is too small")
            }
            DecompressError::DecompressingError => f.write_str("failed to decompress the data"),
            DecompressError::InternalError(status) => write!(f, "blosc internal error: {status}"),
            DecompressError::IoError(err) => write!(f, "I/O error: {}", err),
        }
    }
}
impl std::error::Error for DecompressError {}

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

            let compressed = crate::Encoder::new(clevel)
                .shuffle(shuffle)
                .typesize(typesize)
                .compressor(compressor)
                .blocksize(blocksize)
                .numinternalthreads(numinternalthreads)
                .compress(&src)
                .unwrap();

            let decoder = crate::Decoder::new(&compressed).unwrap();
            let items_num = src.len() / typesize;
            if items_num > 0 {
                for _ in 0..10 {
                    let idx = rand.random_range(0..items_num);
                    let item = decoder.item(idx).unwrap();
                    assert_eq!(item, src[idx * typesize..(idx + 1) * typesize]);
                }
                for _ in 0..10 {
                    let start = rand.random_range(0..items_num);
                    let end = rand.random_range(start..items_num);
                    let items = decoder.items(start..end).unwrap();
                    assert_eq!(items, src[start * typesize..end * typesize]);
                }
            }

            let decompressed = decoder.decompress(numinternalthreads).unwrap();
            assert_eq!(src, decompressed);
        }
    }
}

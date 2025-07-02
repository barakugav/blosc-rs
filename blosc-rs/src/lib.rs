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
//! blosc-rs = "0.2"
//!
//! # Or alternatively, rename the crate to `blosc`
//! blosc = { package = "blosc-rs", version = "0.2" }
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
//! // Read some items using random access, without decompressing the entire buffer
//! assert_eq!(&data_bytes[0..4], decoder.item(0).expect("failed to get the 0-th item"));
//! assert_eq!(&data_bytes[12..16], decoder.item(3).expect("failed to get the 3-th item"));
//! assert_eq!(&data_bytes[4..20], decoder.items(1..5).expect("failed to get items 1 to 4"));
//!
//! // Decompress the entire buffer
//! let decompressed = decoder.decompress(numinternalthreads).expect("failed to decompress");
//! assert_eq!(data_bytes, decompressed);
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
/// This struct is not the usual stream-like encoder that commonly exists in Rust compression libraries, but rather a
/// configuration builder for the Blosc compression.
/// This is because blosc is not a streaming compression library, and it operate on the entire data buffer at once.
pub struct Encoder {
    level: Level,
    shuffle: Shuffle,
    typesize: usize,
    compressor: CompressAlgo,
    blocksize: Option<NonZeroUsize>,
    numinternalthreads: u32,
}
impl Default for Encoder {
    fn default() -> Self {
        Self::new(Level::new(9).unwrap())
    }
}
impl Encoder {
    /// Create a new encoder with the specified compression level.
    pub fn new(level: Level) -> Self {
        Self {
            level,
            shuffle: Shuffle::Byte,
            typesize: 1,
            compressor: CompressAlgo::Blosclz,
            blocksize: None,
            numinternalthreads: 1,
        }
    }

    /// Sets the compression level for the encoder.
    pub fn level(&mut self, level: Level) -> &mut Self {
        self.level = level;
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
    /// For implementation reasons, only a `typesize` in the range `1 < typesize < 256` will allow the
    /// shuffle filter to work. When `typesize` is not in this range, shuffle will be silently disabled.
    ///
    /// The `typesize` is also used to split the input bytes into logical items, and a `Decoder` can access these items
    /// by their index without decompressing the entire buffer. See [`Decoder::item`] and [`Decoder::items`].
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
                self.level.0 as i32 as std::ffi::c_int,
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

/// A compression level used by Blosc.
///
/// The levels range from 0 to 9, where 0 is no compression and 9 is maximum compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Level(u32);
impl Level {
    /// Creates a new compression level.
    ///
    /// # Arguments
    ///
    /// * `level`: The compression level, must be in the range 0 to 9.
    ///
    /// # Returns
    ///
    /// The created `Level` if the input is valid, otherwise `None`.
    pub fn new(level: u32) -> Option<Self> {
        (0..=9).contains(&level).then_some(Self(level))
    }
}
impl TryFrom<u32> for Level {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::new(value).ok_or(())
    }
}
impl From<Level> for u32 {
    fn from(level: Level) -> Self {
        level.0
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
/// This struct is not the usual stream-like decoder that commonly exists in Rust compression libraries, but rather
/// an array-like object that allows random access to elements in the compressed data or decoding the entire buffer.
/// This is because blosc is not a streaming library, and it operates on the entire data buffer at once.
///
/// The compressed data is held in memory within the decoder, and decoding is done either by decompressing the entire
/// buffer, or by accessing individual items or ranges of items. In both cases, the decoder remains unchanged and only
/// the compressed data is held by it.
pub struct Decoder<'a> {
    src: Cow<'a, [u8]>,
    typesize: usize,
    dst_len: usize,
    decompression_alignment: Alignment,
}
impl<'a> Decoder<'a> {
    /// Create a new decoder from a reader that contains Blosc compressed data.
    ///
    /// First a header of a fixed size is read from the reader, which contains metadata about the length of the
    /// compressed data. Then the rest of the compressed data is read into an in-memory internal buffer held by the
    /// decoder, and the reader is not used after this point.
    pub fn from_reader(reader: &mut impl Read) -> Result<Self, DecompressError> {
        // Read the header
        let mut header = [MaybeUninit::<u8>::uninit(); blosc_sys::BLOSC_MIN_HEADER_LENGTH as usize];
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
            decompression_alignment: Alignment::new(1).unwrap(),
        })
    }

    /// Set the alignment used when allocating vectors for decompression.
    ///
    /// The alignment argument will be used for all vectors returned by `decompress`, `item` and `items` methods.
    /// This is useful for transmuting the decompressed data into original type. Consider the following
    /// use case for example:
    /// ```rust
    /// use blosc_rs::{CompressAlgo, Encoder, Decoder};
    ///
    /// let data: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];
    /// let data_bytes = unsafe {
    ///     std::slice::from_raw_parts(
    ///         data.as_ptr() as *const u8,
    ///         data.len() * std::mem::size_of::<i32>(),
    ///     )
    /// };
    ///
    /// let compressed = Encoder::default()
    ///     .typesize(std::mem::size_of::<i32>())
    ///     .compress(&data_bytes)
    ///     .expect("failed to compress");
    /// let mut decoder = Decoder::new(&compressed).expect("invalid buffer");
    ///
    /// // Decompress the data without setting the alignment
    /// let decompressed_unaligned = decoder.decompress(1).expect("failed to decompress");
    /// // !! NOT SAFE !!
    /// let _not_safe_decompressed: &[i32] = unsafe {
    ///     std::slice::from_raw_parts(
    ///         // there is no guaruntee the slice pointer is aligned to i32's alignment
    ///         decompressed_unaligned.as_ptr() as *const i32,
    ///         decompressed_unaligned.len() / std::mem::size_of::<i32>(),
    ///     )
    /// };
    ///
    /// let decompressed = decoder
    ///     .set_decompression_alignment(std::mem::align_of::<i32>())
    ///     .unwrap()
    ///     .decompress(1)
    ///     .expect("failed to decompress");
    /// assert!(decompressed.as_ptr() as usize % std::mem::align_of::<i32>() == 0);
    /// // SAFETY: we know the data is of type i32 and the slice pointer is aligned to i32's alignment
    /// let decompressed: &[i32] = unsafe {
    ///     std::slice::from_raw_parts(
    ///         decompressed.as_ptr() as *const i32,
    ///         decompressed.len() / std::mem::size_of::<i32>(),
    ///     )
    /// };
    /// assert_eq!(decompressed, &data);
    /// let item = decoder.item(3).expect("failed to get an item");
    /// assert!(item.as_ptr() as usize % std::mem::align_of::<i32>() == 0);
    /// // SAFETY: we know the item is of type i32 and the slice pointer is aligned to i32's alignment
    /// let item = unsafe { *(item.as_ptr() as *const i32) };
    /// assert_eq!(item, 4);
    /// ```
    ///
    /// By default, the alignment is set to 1.
    ///
    /// # Arguments
    ///
    /// * `alignment`: The alignment to use, must be a power of two.
    ///
    /// # Returns
    ///
    /// Ok or Err if the alignment is not a power of two.
    pub fn set_decompression_alignment(
        &mut self,
        alignment: usize,
    ) -> Result<&mut Self, AlignmentError> {
        self.decompression_alignment = Alignment::new(alignment)?;
        Ok(self)
    }

    /// Decompress the entire buffer and return the decompressed data as a `Vec<u8>`.
    ///
    /// Note that the returned vector may not be aligned to the original data type's alignment, and the caller should
    /// ensure that the alignment is correct before transmuting it to original type. If the alignment does not match
    /// the original data type, the bytes should be copied to a new aligned allocation before transmuting, otherwise
    /// undefined behavior may occur. To enforce a specific alignment, use [`Self::set_decompression_alignment`]
    /// affecting this, [`Self::item`] and [`Self::items`] methods.
    ///
    /// # Arguments
    ///
    /// * `numinternalthreads`: The number of threads to use internally.
    ///
    /// # Returns
    ///
    /// A `Result` containing the decompressed data as a `Vec<u8>`, or a `DecompressError` if an error occurs.
    pub fn decompress(&self, numinternalthreads: u32) -> Result<Vec<u8>, DecompressError> {
        let mut dst = new_vec_aligned(self.dst_len, self.decompression_alignment);
        let len = self.decompress_into(dst.as_mut_slice(), numinternalthreads)?;

        assert!(len <= self.dst_len);
        unsafe { dst.set_len(len) };
        // SAFETY: every element from 0 to len was initialized
        let vec = unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) };
        Ok(vec)
    }

    /// Decompress the entire buffer and write the decompressed data into the provided destination buffer.
    ///
    /// Note that if the destination buffer is not aligned to the original data type's alignment, the caller should
    /// not transmute the decompressed data to the original type, as this may lead to undefined behavior.
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

    /// Get a reference to the inner compressed data buffer.
    pub fn as_buf(&self) -> &[u8] {
        &self.src
    }

    /// Get the inner compressed data buffer.
    pub fn into_buf(self) -> Cow<'a, [u8]> {
        self.src
    }

    /// Get an element at the specified index.
    ///
    /// Each item is `typesize` (as provided during encoding) bytes long, and the index is zero-based.
    ///
    /// Note that the returned vector may not be aligned to the original data type's alignment, and the caller should
    /// ensure that the alignment is correct before transmuting it to original type. If the alignment does not match
    /// the original data type, the bytes should be copied to a new aligned allocation before transmuting, otherwise
    /// undefined behavior may occur. To enforce a specific alignment, use [`Self::set_decompression_alignment`]
    /// affecting this, [`Self::items`] and [`Self::decompress`] methods.
    ///
    pub fn item(&self, idx: usize) -> Result<Vec<u8>, DecompressError> {
        self.items(idx..idx + 1)
    }

    /// Get an element at the specified index and copy it into the provided destination buffer.
    ///
    /// Each item is `typesize` (as provided during encoding) bytes long, and the index is zero-based.
    ///
    /// Note that if the destination buffer is not aligned to the original data type's alignment, the caller should
    /// not transmute the decompressed data to original type, as this may lead to undefined behavior.
    pub fn item_into(
        &self,
        idx: usize,
        dst: &mut [MaybeUninit<u8>],
    ) -> Result<usize, DecompressError> {
        self.items_into(idx..idx + 1, dst)
    }

    /// Get a range of elements specified by the index range.
    ///
    /// Each item is `typesize` (as provided during encoding) bytes long, and the index is zero-based.
    ///
    /// Note that the returned vector may not be aligned to the original data type's alignment, and the caller should
    /// ensure that the alignment is correct before transmuting it to original type. If the alignment does not match
    /// the original data type, the bytes should be copied to a new aligned allocation before transmuting, otherwise
    /// undefined behavior may occur. To enforce a specific alignment, use [`Self::set_decompression_alignment`]
    /// affecting this, [`Self::item`] and [`Self::decompress`] methods.
    pub fn items(&self, idx: std::ops::Range<usize>) -> Result<Vec<u8>, DecompressError> {
        let mut dst = new_vec_aligned(self.typesize * idx.len(), self.decompression_alignment);
        self.items_into(idx, &mut dst)?;
        // SAFETY: every element in dst is initialized
        Ok(unsafe { std::mem::transmute::<Vec<MaybeUninit<u8>>, Vec<u8>>(dst) })
    }

    /// Get a range of elements specified by the index range and copy them into the provided destination buffer.
    ///
    /// Each item is `typesize` (as provided during encoding) bytes long, and the index is zero-based.
    ///
    /// Note that if the destination buffer is not aligned to the original data type's alignment, the caller should
    /// not transmute the decompressed data to original type, as this may lead to undefined behavior.
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
            DecompressError::IoError(err) => write!(f, "I/O error: {err}"),
        }
    }
}
impl std::error::Error for DecompressError {}

#[derive(Clone, Copy)]
#[repr(usize)]
enum Alignment {
    A0 = 1,
    A1 = 2,
    A2 = 4,
    A3 = 8,
    A4 = 16,
    A5 = 32,
    A6 = 64,
    A7 = 128,
    A8 = 256,
    A9 = 512,
    A10 = 1024,
    A11 = 2048,
    A12 = 4096,
    A13 = 8192,
    A14 = 16384,
    A15 = 32768,
    A16 = 65536,
    A17 = 131072,
    A18 = 262144,
    A19 = 524288,
    A20 = 1048576,
    A21 = 2097152,
    A22 = 4194304,
    A23 = 8388608,
    A24 = 16777216,
    A25 = 33554432,
    A26 = 67108864,
    A27 = 134217728,
    A28 = 268435456,
    A29 = 536870912,
}
impl Alignment {
    fn new(align: usize) -> Result<Self, AlignmentError> {
        Ok(match align {
            0 | 1 => Self::A0,
            2 => Self::A1,
            4 => Self::A2,
            8 => Self::A3,
            16 => Self::A4,
            32 => Self::A5,
            64 => Self::A6,
            128 => Self::A7,
            256 => Self::A8,
            512 => Self::A9,
            1024 => Self::A10,
            2048 => Self::A11,
            4096 => Self::A12,
            8192 => Self::A13,
            16384 => Self::A14,
            32768 => Self::A15,
            65536 => Self::A16,
            131072 => Self::A17,
            262144 => Self::A18,
            524288 => Self::A19,
            1048576 => Self::A20,
            2097152 => Self::A21,
            4194304 => Self::A22,
            8388608 => Self::A23,
            16777216 => Self::A24,
            33554432 => Self::A25,
            67108864 => Self::A26,
            134217728 => Self::A27,
            268435456 => Self::A28,
            536870912 => Self::A29,
            _ => return Err(AlignmentError),
        })
    }
}

/// Error representing an invalid alignment value.
#[derive(Debug)]
#[non_exhaustive]
pub struct AlignmentError;
impl std::fmt::Display for AlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("alignment must be a power of two, smaller or equal to 2^30")
    }
}
impl std::error::Error for AlignmentError {}

fn new_vec_aligned(size: usize, alignment: Alignment) -> Vec<MaybeUninit<u8>> {
    unsafe fn new_vec_aligned_impl<DUMMY>() -> Vec<u8> {
        let alignment = std::mem::align_of::<DUMMY>();
        assert_eq!(alignment, std::mem::size_of::<DUMMY>());
        assert!(alignment.is_power_of_two());

        let raw_vec = Vec::<DUMMY>::new();
        let ptr = raw_vec.as_ptr() as *mut u8;
        let capacity = raw_vec.capacity() * std::mem::size_of::<DUMMY>();
        std::mem::forget(raw_vec);
        unsafe { Vec::from_raw_parts(ptr, 0, capacity) }
    }

    macro_rules! new_vec_aligned_impl {
        ($alignment:expr) => {{
            #[repr(align($alignment))]
            struct AlignedDummy(#[allow(dead_code)] [u8; $alignment]);
            unsafe { new_vec_aligned_impl::<AlignedDummy>() }
        }};
    }

    let vec = match alignment {
        Alignment::A0 => new_vec_aligned_impl!(1),
        Alignment::A1 => new_vec_aligned_impl!(2),
        Alignment::A2 => new_vec_aligned_impl!(4),
        Alignment::A3 => new_vec_aligned_impl!(8),
        Alignment::A4 => new_vec_aligned_impl!(16),
        Alignment::A5 => new_vec_aligned_impl!(32),
        Alignment::A6 => new_vec_aligned_impl!(64),
        Alignment::A7 => new_vec_aligned_impl!(128),
        Alignment::A8 => new_vec_aligned_impl!(256),
        Alignment::A9 => new_vec_aligned_impl!(512),
        Alignment::A10 => new_vec_aligned_impl!(1024),
        Alignment::A11 => new_vec_aligned_impl!(2048),
        Alignment::A12 => new_vec_aligned_impl!(4096),
        Alignment::A13 => new_vec_aligned_impl!(8192),
        Alignment::A14 => new_vec_aligned_impl!(16384),
        Alignment::A15 => new_vec_aligned_impl!(32768),
        Alignment::A16 => new_vec_aligned_impl!(65536),
        Alignment::A17 => new_vec_aligned_impl!(131072),
        Alignment::A18 => new_vec_aligned_impl!(262144),
        Alignment::A19 => new_vec_aligned_impl!(524288),
        Alignment::A20 => new_vec_aligned_impl!(1048576),
        Alignment::A21 => new_vec_aligned_impl!(2097152),
        Alignment::A22 => new_vec_aligned_impl!(4194304),
        Alignment::A23 => new_vec_aligned_impl!(8388608),
        Alignment::A24 => new_vec_aligned_impl!(16777216),
        Alignment::A25 => new_vec_aligned_impl!(33554432),
        Alignment::A26 => new_vec_aligned_impl!(67108864),
        Alignment::A27 => new_vec_aligned_impl!(134217728),
        Alignment::A28 => new_vec_aligned_impl!(268435456),
        Alignment::A29 => new_vec_aligned_impl!(536870912),
    };

    assert_eq!(0, vec.as_ptr() as usize % alignment as usize);
    assert_eq!(0, vec.len());

    let mut vec = unsafe { std::mem::transmute::<Vec<u8>, Vec<MaybeUninit<u8>>>(vec) };
    vec.reserve(size);
    unsafe { vec.set_len(size) };

    vec
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::{CompressAlgo, Level, Shuffle};

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
            let clevel: Level = rand.random_range(0..=9).try_into().unwrap();
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

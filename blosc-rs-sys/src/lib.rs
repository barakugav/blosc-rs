//! Unsafe Rust bindings for blosc - a blocking, shuffling and lossless compression library.

mod c_bridge {
    #![allow(dead_code)]
    #![allow(unused_imports)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::missing_safety_doc)]
    #![allow(rustdoc::invalid_html_tags)]
    #![allow(rustdoc::broken_intra_doc_links)]
    #![allow(missing_docs)]
    #![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
pub use c_bridge::*;

#[cfg(test)]
mod tests {
    #[test]
    fn check_linking() {
        let data: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut dest = [0u8; 100];
        unsafe {
            crate::blosc_compress_ctx(
                0,
                crate::BLOSC_NOSHUFFLE as i32,
                1,
                data.len(),
                data.as_ptr() as *const core::ffi::c_void,
                dest.as_mut_ptr() as *mut core::ffi::c_void,
                dest.len(),
                c"zstd".as_ptr(),
                0,
                1,
            );
        }
    }
}

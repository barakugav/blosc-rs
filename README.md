# blosc-rs

[![Crates.io](https://img.shields.io/crates/v/blosc-rs.svg)](https://crates.io/crates/blosc-rs/)
[![Documentation](https://docs.rs/blosc-rs/badge.svg)](https://docs.rs/blosc-rs/)
![License](https://img.shields.io/crates/l/blosc-rs)

Rust bindings for blosc - a blocking, shuffling and lossless compression library.

Provide a safe interface to the [blosc](https://github.com/Blosc/c-blosc) library.
The crate has zero runtime dependencies.

### Getting Started

To use this library, add the following to your `Cargo.toml`:
```toml
[dependencies]
blosc-rs = "0.3"

# Or alternatively, rename the crate to `blosc`
blosc = { package = "blosc-rs", version = "0.3" }
```

In the following example we compress a vector of integers and then decompress it back:
```rust
use blosc_rs::{CompressAlgo, Encoder, Decoder};

let data: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];

let data_bytes = unsafe {
    std::slice::from_raw_parts(
        data.as_ptr() as *const u8,
        data.len() * std::mem::size_of::<i32>(),
    )
};
let numinternalthreads = 4;

let compressed = Encoder::default()
    .typesize(std::mem::size_of::<i32>().try_into().unwrap())
    .numinternalthreads(numinternalthreads)
    .compress(&data_bytes)
    .expect("failed to compress");

let decoder = Decoder::new(&compressed).expect("invalid buffer");

// Read some items using random access, without decompressing the entire buffer
assert_eq!(&data_bytes[0..4], decoder.item(0).expect("failed to get the 0-th item"));
assert_eq!(&data_bytes[12..16], decoder.item(3).expect("failed to get the 3-th item"));
assert_eq!(&data_bytes[4..20], decoder.items(1..5).expect("failed to get items 1 to 4"));

// Decompress the entire buffer
let decompressed = decoder.decompress(numinternalthreads).expect("failed to decompress");
assert_eq!(data_bytes, decompressed);
```

# blosc-rs
Rust bindings for blosc - a blocking, shuffling and lossless compression library.

Provide a safe interface to the [blosc](https://github.com/Blosc/c-blosc) library.

### Getting Started

To use this library, add the following to your `Cargo.toml`:
```toml
[dependencies]
blosc-rs = "0.1"

# Or alternatively, rename the crate to `blosc`
blosc = { package = "blosc-rs", version = "0.1" }
```

In the following example we compress a vector of integers and then decompress it back:
```rust
use blosc_rs::{CLevel, CompressAlgo, Shuffle, compress, decompress};

let data: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];
let data_bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * ::mem::size_of::<i32>()) };
let numinternalthreads = 4;
let compressed = compress(
    CLevel::L5,
    Shuffle::Byte,
    std::mem::size_of::<i32>(), // itemsize
    data_bytes,
    &CompressAlgo::Blosclz,
    None, // automatic block size
    numinternalthreads,
).unwrap();
let decompressed = decompress(
    &compressed,
    numinternalthreads,
).unwrap();
// SAFETY: we know the data is of type i32
let decompressed: &[i32] = unsafe { std::slice::from_raw_parts(decompressed.as_ptr() as *const i32, .len() / td::mem::size_of::<i32>()) };
assert_eq!(data, *decompressed);
```

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    generate_bindings();
    build_c_lib();
    link_c_lib();
}

fn generate_bindings() {
    let c_lib = Path::new(&env!("CARGO_MANIFEST_DIR")).join("c");
    let builder = bindgen::Builder::default()
        .clang_arg(format!("-I{}", c_lib.to_str().unwrap()))
        .use_core()
        .header(c_lib.join("bindings.h").to_str().unwrap())
        .allowlist_file(format!("{}/blosc/blosc.h", c_lib.to_str().unwrap()))
        // .default_enum_style(bindgen::EnumVariation::Rust {
        //     non_exhaustive: false,
        // })
        // .no_copy(".*")
        // .manually_drop_union(".*")
        // .opaque_type("EValueStorage")
        // .opaque_type("TensorStorage")
        // .opaque_type("TensorImpl")
        // .opaque_type("Program")
        // .opaque_type("TensorInfo")
        // .opaque_type("MethodMeta")
        // .opaque_type("Method")
        // .opaque_type("BufferDataLoader")
        // .opaque_type("FileDataLoader")
        // .opaque_type("MmapDataLoader")
        // .opaque_type("MemoryAllocator")
        // .opaque_type("HierarchicalAllocator")
        // .opaque_type("MemoryManager")
        // .opaque_type("OptionalTensorStorage")
        // .opaque_type("ETDumpGen")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    let bindings = builder.generate().expect("Failed to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn build_c_lib() {
    let blosc_dir = blosc_dir();
    if !blosc_dir.exists() {
        std::fs::create_dir_all(&blosc_dir).expect("Failed to create c-blosc directory");
        Command::new("git")
            .args(&[
                "clone",
                "--depth",
                "1",
                "--branch",
                "v1.21.6",
                "https://github.com/Blosc/c-blosc.git",
                ".",
            ])
            .current_dir(&blosc_dir)
            .status()
            .map_err(|e| {
                let _ = std::fs::remove_dir_all(&blosc_dir);
                e
            })
            .expect("Failed to clone c-blosc repository");
    }

    let blosc_build_dir = blosc_build_dir();
    std::fs::create_dir_all(&blosc_build_dir).expect("Failed to create c-blosc build directory");
    Command::new("cmake")
        .args(&[
            "-S",
            ".",
            "-B",
            blosc_build_dir.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_STATIC=ON",
            "-DBUILD_SHARED=OFF",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_BENCHMARKS=OFF",
        ])
        .current_dir(&blosc_dir)
        .status()
        .expect("Failed to configure c-blosc build");
    Command::new("cmake")
        .args(&["--build", "."])
        .current_dir(&blosc_build_dir)
        .status()
        .expect("Failed to build c-blosc");
}

fn link_c_lib() {
    let libs_dir = blosc_build_dir().join("blosc");
    println!(
        "cargo::rustc-link-search=native={}",
        libs_dir.to_str().unwrap()
    );
    println!("cargo::rustc-link-lib=static=blosc");
}

fn blosc_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("c-blosc")
}
fn blosc_build_dir() -> PathBuf {
    blosc_dir().join("build")
}

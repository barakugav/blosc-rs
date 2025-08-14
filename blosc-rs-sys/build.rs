use std::path::PathBuf;

fn main() {
    generate_bindings();

    // Build and link
    let lib_name = build_c_lib();
    println!("cargo::rustc-link-lib=static={lib_name}");
}

fn generate_bindings() {
    let builder = bindgen::Builder::default()
        .use_core()
        .generate_cstr(true)
        .header("c/bindings.h")
        .allowlist_file(".*/blosc.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    let bindings = builder.generate().expect("Failed to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn build_c_lib() -> String {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

    let blosc_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("third-party")
        .join("c-blosc");
    println!("cargo::rerun-if-changed={}", blosc_dir.display());

    let mut build = cmake::Config::new(&blosc_dir);
    let bool2opt = |b: bool| if b { "ON" } else { "OFF" };
    build
        .define("BUILD_STATIC", "ON")
        .define("BUILD_SHARED", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_FUZZERS", "OFF")
        .define("BUILD_BENCHMARKS", "OFF")
        .define("DEACTIVATE_LZ4", bool2opt(!cfg!(feature = "lz4")))
        // .define("DEACTIVATE_SNAPPY", bool2opt(!cfg!(feature = "snappy")))
        .define("DEACTIVATE_ZLIB", bool2opt(!cfg!(feature = "zlib")))
        .define("DEACTIVATE_ZSTD", bool2opt(!cfg!(feature = "zstd")))
        .out_dir(out_dir.join("c-blosc-build"));
    let profile = build.get_profile().to_string();
    let blosc_build_dir = build.build();
    let blosc_build_dir = blosc_build_dir.join("build");

    let target = std::env::var("TARGET").unwrap();
    let (lib_dir, libname) = if target.contains("windows") && target.contains("msvc") {
        (blosc_build_dir.join("blosc").join(profile), "libblosc")
    } else {
        (blosc_build_dir.join("blosc"), "blosc")
    };

    println!(
        "cargo::rustc-link-search=native={}",
        lib_dir.to_str().unwrap()
    );
    libname.to_string()
}

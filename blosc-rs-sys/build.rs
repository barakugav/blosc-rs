use std::path::PathBuf;
use std::process::Command;

fn main() {
    generate_bindings();

    // Build and link
    if std::env::var("DOCS_RS").is_err() {
        build_c_lib();

        let lib_name = if cfg!(target_os = "windows") {
            "libblosc"
        } else {
            "blosc"
        };
        println!("cargo::rustc-link-lib=static={lib_name}");
    }
}

fn generate_bindings() {
    let builder = bindgen::Builder::default()
        .use_core()
        .header("c/bindings.h")
        .allowlist_file(".*/blosc.h")
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
            .args([
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
            .inspect_err(|_| {
                let _ = std::fs::remove_dir_all(&blosc_dir);
            })
            .expect("Failed to clone c-blosc repository");
    }

    let blosc_build_dir = blosc_build_dir();
    std::fs::create_dir_all(&blosc_build_dir).expect("Failed to create c-blosc build directory");
    Command::new("cmake")
        .args([
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
        .args(["--build", ".", "--config", "Release"])
        .current_dir(&blosc_build_dir)
        .status()
        .expect("Failed to build c-blosc");

    let libs_dir = if cfg!(target_os = "windows") {
        blosc_build_dir.join("blosc").join("Release")
    } else {
        blosc_build_dir.join("blosc")
    };

    println!(
        "cargo::rustc-link-search=native={}",
        libs_dir.to_str().unwrap()
    );
}

fn blosc_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("c-blosc")
}
fn blosc_build_dir() -> PathBuf {
    blosc_dir().join("build")
}

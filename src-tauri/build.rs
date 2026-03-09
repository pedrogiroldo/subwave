fn main() {
    tauri_build::build();

    // Link libvosk from src-tauri/bin for Linux builds.
    if cfg!(target_os = "linux") {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        println!("cargo:rustc-link-search=native={}/bin", manifest_dir);
        println!("cargo:rustc-link-lib=vosk");
        // subwave binary is at src-tauri/target/debug; libvosk.so is at src-tauri/bin
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../bin");
        println!("cargo:rerun-if-changed=bin/libvosk.so");
    }
}

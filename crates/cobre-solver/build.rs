//! Build script for cobre-solver: builds the vendored `HiGHS` (`highs` feature,
//! default) and/or `CLP` (`clp` feature) solver libraries via `cmake`, then
//! compiles their thin C wrappers via `cc`.

// Build scripts routinely use expect/panic for unrecoverable configuration
// errors. Allow these lints here since there is no caller to propagate errors to.
// `too_many_lines` is allowed because `main` drives two optional vendored
// solver builds (HiGHS + CLP) sequentially; splitting it would scatter the
// shared target-env setup without improving clarity.
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::manual_assert,
    clippy::too_many_lines
)]

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // Exactly one LP backend may be enabled; enabling both `highs` and `clp` is
    // rejected at compile time (see the `compile_error!` in `lib.rs`).
    if env::var("CARGO_FEATURE_HIGHS").is_ok() {
        println!("cargo:rerun-if-changed=csrc/highs_wrapper.c");
        println!("cargo:rerun-if-changed=csrc/highs_wrapper.h");
        println!("cargo:rerun-if-changed=csrc/highs_wrapper_cpp.cpp");

        let highs_src = manifest_dir.join("vendor/HiGHS");

        if !highs_src.join("CMakeLists.txt").exists() {
            panic!(
                "HiGHS source not found at crates/cobre-solver/vendor/HiGHS/. \
                 Run: git submodule update --init --recursive"
            );
        }

        eprintln!("cobre-solver: building HiGHS from {}", highs_src.display());

        // Always build HiGHS in Release mode regardless of the Rust profile.
        // HiGHS is a solver library — an unoptimized build is ~10x slower and
        // would produce misleading results even during development.
        let mut cmake_config = cmake::Config::new(&highs_src);
        cmake_config
            .define("CMAKE_BUILD_TYPE", "Release")
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("HIGHS_NO_DEFAULT_THREADS", "ON")
            .define("BUILD_TESTING", "OFF")
            .define("BUILD_EXAMPLES", "OFF")
            // Must stay 32-bit to match the i32 types in the FFI bindings;
            // highs_wrapper.c's _Static_assert also catches a mismatch, but this
            // flag prevents the cmake build from ever producing one.
            .define("HIGHSINT64", "OFF")
            // HiGHS uses zlib only for Highs_readModel() (.mps.gz/.lp.gz); Cobre
            // builds every LP programmatically via the C API and never reads model
            // files, so disabling it drops a system dependency that otherwise
            // breaks cross-compilation in the Python wheel CI.
            .define("CMAKE_DISABLE_FIND_PACKAGE_ZLIB", "ON");

        // On MSVC, use static CRT to avoid requiring vcruntime140.dll in the wheel.
        if target_env == "msvc" {
            cmake_config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
            cmake_config.cflag("/MT");
            cmake_config.cxxflag("/MT");
        }

        let highs_dst = cmake_config.build();

        eprintln!(
            "cobre-solver: HiGHS cmake output at {}",
            highs_dst.display()
        );

        println!(
            "cargo:rustc-link-search=native={}",
            highs_dst.join("lib").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            highs_dst.join("lib64").display()
        );

        // MSVC cmake may place libraries in a configuration subdirectory.
        if target_env == "msvc" {
            println!(
                "cargo:rustc-link-search=native={}",
                highs_dst.join("lib/Release").display()
            );
        }

        println!("cargo:rustc-link-lib=static=highs");

        // MSVC links the C++ runtime automatically; no explicit directive needed.
        if target_env != "msvc" {
            if target_os == "macos" {
                println!("cargo:rustc-link-lib=c++");
            } else {
                println!("cargo:rustc-link-lib=stdc++");
            }
        }

        let highs_include = highs_dst.join("include");
        let highs_include_highs = highs_dst.join("include/highs");

        eprintln!(
            "cobre-solver: compiling C wrapper with include paths: {}, {}",
            highs_include.display(),
            highs_include_highs.display()
        );

        let mut build = cc::Build::new();
        build
            .file("csrc/highs_wrapper.c")
            .include("csrc")
            .warnings(true)
            .extra_warnings(true);

        // Treat HiGHS headers as system includes so their warnings don't surface
        // while we keep full warning coverage on our own wrappers. MSVC lacks
        // -isystem; fall back to -I and accept the vendor noise there.
        add_system_or_include(&mut build, target_env == "msvc", &highs_include);
        add_system_or_include(&mut build, target_env == "msvc", &highs_include_highs);

        // MSVC: link against static CRT (`/MT`) to match the static-CRT HiGHS
        // build above — otherwise the cc-compiled wrappers default to dynamic
        // CRT (`/MD`) and the linker rejects the mismatch (`LNK2038`).
        // cargo-dist sets `+crt-static` via `RUSTFLAGS` (which `cc` honours),
        // but the PyO3/maturin wheel build does not, so this override is
        // required there.
        if target_env == "msvc" {
            build.static_crt(true);
        }

        if target_env != "msvc" {
            build.flag("-Wno-unused-function");
        }

        build.compile("highs_wrapper");

        // C++ shim: implements cobre_highs_set_basis_non_alien which constructs a
        // HighsBasis (C++ type) with alien = false and calls
        // Highs::setBasis(const HighsBasis&) directly, bypassing the alien-path LU
        // factorisation that the HiGHS C API always triggers.  Compiled as a
        // separate object with C++17 so that the plain-C wrapper above is
        // unaffected.
        let mut build_cpp = cc::Build::new();
        build_cpp
            .file("csrc/highs_wrapper_cpp.cpp")
            .cpp(true)
            .include("csrc")
            .warnings(true)
            .extra_warnings(true);

        add_system_or_include(&mut build_cpp, target_env == "msvc", &highs_include);
        add_system_or_include(&mut build_cpp, target_env == "msvc", &highs_include_highs);

        build_cpp.flag_if_supported("-std=c++17");

        // Mirror the MSVC CRT setting used for the HiGHS cmake build above:
        // static CRT (`/MT`) so that the linker accepts the C++ wrapper
        // alongside HiGHS's C++ objects (which were also compiled `/MT`).
        if target_env == "msvc" {
            build_cpp.flag("/std:c++17");
            build_cpp.static_crt(true);
        } else {
            build_cpp.flag("-Wno-unused-function");
        }

        build_cpp.compile("highs_wrapper_cpp");
    }

    // Cargo exposes feature activation to build scripts via the
    // CARGO_FEATURE_<NAME> environment variable, so this entire block is
    // skipped — and the default build artifact stays byte-identical — unless
    // `--features clp` is active.
    if env::var("CARGO_FEATURE_CLP").is_ok() {
        println!("cargo:rerun-if-changed=csrc/clp_wrapper.c");
        println!("cargo:rerun-if-changed=csrc/clp_wrapper.h");
        println!("cargo:rerun-if-changed=csrc/clp_wrapper_cpp.cpp");
        println!("cargo:rerun-if-changed=vendor/coin-build/CMakeLists.txt");
        // The vendored config headers drive the superbuild's compilation; edits
        // to them must retrigger cmake or the build tree goes stale.
        println!("cargo:rerun-if-changed=vendor/coin-build/include/ClpConfig.h");
        println!("cargo:rerun-if-changed=vendor/coin-build/include/CoinUtilsConfig.h");
        println!("cargo:rerun-if-changed=vendor/coin-build/include/config_clp.h");
        println!("cargo:rerun-if-changed=vendor/coin-build/include/config_coinutils.h");

        // The COIN-OR submodules use a nested directory layout, so the real
        // source roots are one level deeper than the submodule root. Probe a
        // representative header from each to confirm the submodules are
        // initialized before handing off to cmake.
        let clp_header = manifest_dir.join("vendor/Clp/Clp/src/Clp_C_Interface.h");
        let coinutils_header =
            manifest_dir.join("vendor/CoinUtils/CoinUtils/src/CoinFactorization.hpp");
        if !clp_header.exists() || !coinutils_header.exists() {
            panic!(
                "CLP/CoinUtils source not found under crates/cobre-solver/vendor/Clp/ and \
                 crates/cobre-solver/vendor/CoinUtils/. \
                 Run: git submodule update --init --recursive"
            );
        }

        let coin_build_src = manifest_dir.join("vendor/coin-build");

        eprintln!(
            "cobre-solver: building CLP superbuild from {}",
            coin_build_src.display()
        );

        // Always build CLP in Release mode regardless of the Rust profile —
        // same rationale as the HiGHS build: an unoptimized solver is far
        // slower and would mislead even during development.
        let mut clp_config = cmake::Config::new(&coin_build_src);
        clp_config
            .define("CMAKE_BUILD_TYPE", "Release")
            .define("BUILD_SHARED_LIBS", "OFF");

        // On MSVC, use static CRT to match the HiGHS build above and avoid
        // requiring vcruntime140.dll in the wheel.
        if target_env == "msvc" {
            clp_config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
            clp_config.cflag("/MT");
            clp_config.cxxflag("/MT");
        }

        let clp_dst = clp_config.build();

        eprintln!("cobre-solver: CLP cmake output at {}", clp_dst.display());

        println!(
            "cargo:rustc-link-search=native={}",
            clp_dst.join("lib").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            clp_dst.join("lib64").display()
        );

        // Same MSVC subdirectory quirk as the HiGHS build above.
        if target_env == "msvc" {
            println!(
                "cargo:rustc-link-search=native={}",
                clp_dst.join("lib/Release").display()
            );
        }

        // Link order matters: Clp depends on CoinUtils, so Clp must precede
        // CoinUtils on the linker command line (GNU ld resolves left-to-right).
        println!("cargo:rustc-link-lib=static=Clp");
        println!("cargo:rustc-link-lib=static=CoinUtils");

        // Mirror the HiGHS target-OS/env logic: MSVC links it automatically.
        if target_env != "msvc" {
            if target_os == "macos" {
                println!("cargo:rustc-link-lib=c++");
            } else {
                println!("cargo:rustc-link-lib=stdc++");
            }
        }

        // Compile the thin C wrapper (csrc/clp_wrapper.c). It includes
        // <Clp_C_Interface.h>, which in turn includes "Coin_C_defines.h".
        // The cmake superbuild installs only Clp_C_Interface.h into the
        // out-dir include/, so the transitive Coin_C_defines.h is NOT present
        // there. Point the compiler at the two nested COIN-OR source roots
        // (where both headers live) so the include resolves. The out-dir
        // include/ is added as well for completeness.
        let clp_include = clp_dst.join("include");
        let clp_src_include = manifest_dir.join("vendor/Clp/Clp/src");
        let coinutils_src_include = manifest_dir.join("vendor/CoinUtils/CoinUtils/src");

        eprintln!(
            "cobre-solver: compiling CLP wrapper with include paths: {}, {}, {}",
            clp_src_include.display(),
            coinutils_src_include.display(),
            clp_include.display()
        );

        let mut clp_build = cc::Build::new();
        clp_build
            .file("csrc/clp_wrapper.c")
            .include("csrc")
            .warnings(true)
            .extra_warnings(true);

        // Treat the COIN-OR headers as system includes (same rationale as the
        // HiGHS wrappers): suppress third-party header warnings while keeping
        // full warning coverage on our own wrapper. The nested source dirs
        // supply Clp_C_Interface.h and its transitive Coin_C_defines.h.
        add_system_or_include(&mut clp_build, target_env == "msvc", &clp_src_include);
        add_system_or_include(&mut clp_build, target_env == "msvc", &coinutils_src_include);
        add_system_or_include(&mut clp_build, target_env == "msvc", &clp_include);

        // MSVC: link against the static CRT (`/MT`) to match the static-CRT
        // CLP cmake build above, mirroring the HiGHS wrapper handling.
        if target_env == "msvc" {
            clp_build.static_crt(true);
        }

        clp_build.compile("clp_wrapper");

        // C++ shim: implements the CLP class-only knobs (dual-steepest-edge
        // pricing, factorization frequency, hot-start snapshot/restore) that
        // live solely on the C++ ClpSimplex class and are absent from
        // <Clp_C_Interface.h>. It casts the opaque model handle to `Clp_Simplex*`
        // (the concrete C-interface wrapper struct) and reads `->model_` to reach
        // the live C++ `ClpSimplex`, then calls the class API. Compiled as a
        // separate C++17 object so the plain-C wrapper above is unaffected. The same
        // -isystem set as the C wrapper supplies the vendored C++ class headers
        // (ClpSimplex.hpp / ClpDualRowSteepest.hpp live in the vendored source
        // dirs, NOT the cmake-installed include/).
        let mut clp_build_cpp = cc::Build::new();
        clp_build_cpp
            .file("csrc/clp_wrapper_cpp.cpp")
            .cpp(true)
            .include("csrc")
            .warnings(true)
            .extra_warnings(true);

        add_system_or_include(&mut clp_build_cpp, target_env == "msvc", &clp_src_include);
        add_system_or_include(
            &mut clp_build_cpp,
            target_env == "msvc",
            &coinutils_src_include,
        );
        add_system_or_include(&mut clp_build_cpp, target_env == "msvc", &clp_include);

        clp_build_cpp.flag_if_supported("-std=c++17");

        // Mirror the MSVC CRT setting used for the CLP cmake build above:
        // static CRT (`/MT`) so the linker accepts the C++ wrapper alongside
        // CLP's C++ objects (which were also compiled `/MT`).
        if target_env == "msvc" {
            clp_build_cpp.flag("/std:c++17");
            clp_build_cpp.static_crt(true);
        }

        clp_build_cpp.compile("clp_wrapper_cpp");
    }
}

/// Add an include path, preferring `-isystem` on GCC/Clang so warnings from
/// third-party headers are suppressed while warnings on our own wrappers stay
/// active.  MSVC does not accept `-isystem`, so fall back to a regular
/// `.include()` there.
fn add_system_or_include(build: &mut cc::Build, is_msvc: bool, path: &std::path::Path) {
    if is_msvc {
        build.include(path);
    } else {
        build.flag("-isystem");
        build.flag(path.to_str().expect("HiGHS include path must be UTF-8"));
    }
}

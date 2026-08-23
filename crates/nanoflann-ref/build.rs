//! Builds `cpp/wrapper.cpp` (the extern-C surface over the vendored,
//! unmodified `cpp/nanoflann.hpp`) into a static library and links it into
//! this crate. This crate is the correctness oracle / benchmark baseline, so
//! two flags here are load-bearing and must never be dropped:
//!
//! - `opt_level(3)`: the oracle must always be the fast, `-O3` baseline —
//!   even when the Rust side is built in `cargo test`'s default debug
//!   profile, the C++ reference must not be. Passed unconditionally (not
//!   gated on Rust's profile) so debug test runs still compare against an
//!   optimized C++ tree.
//! - `-ffp-contract=off`: at `-O3`, both gcc and clang will by default fuse
//!   `a*b+c` into a single fused-multiply-add (FMA) instruction, which
//!   rounds once instead of twice and can produce a different bit pattern
//!   than the un-fused expression. rustc does NOT perform this contraction
//!   by default. Without this flag, the C++ oracle's distance computations
//!   could silently diverge from the Rust implementation's bit-for-bit
//!   output on the same inputs, breaking the whole point of cross-validating
//!   against it.
fn main() {
    println!("cargo:rerun-if-changed=cpp/wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp/nanoflann.hpp");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/wrapper.cpp")
        .include("cpp")
        .opt_level(3) // load-bearing: always -O3, regardless of Rust profile (see module docs above).
        .flag_if_supported("-march=native")
        .flag_if_supported("-ffp-contract=off") // load-bearing: no FMA contraction, bit-parity with rustc (see module docs above).
        .compile("nanoflann_ref");
}

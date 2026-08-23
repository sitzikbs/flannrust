//! Native-codegen bit-parity canary. `#[ignore]`d -- run explicitly, and
//! specifically WITH `RUSTFLAGS="-C target-cpu=native"`, since that is
//! exactly how the criterion benches (`benches/*.rs`) and the perf gate
//! (`tests/perf_gate.rs`) run, and the crate's whole bit-parity comparison
//! strategy (`xval::assert_knn_equal_*`, used everywhere else in this crate
//! under the DEFAULT target-cpu) is otherwise never validated under that
//! flag at all. `-C target-cpu=native` can enable wider vector ISAs (AVX2/
//! AVX-512/FMA units) that change which instructions LLVM selects for a
//! given floating-point expression; combined with the C++ oracle's own
//! `-march=native` (always on, unconditionally -- see
//! `crates/nanoflann-ref/build.rs`) and its `-ffp-contract=off` (matches
//! rustc's no-FMA-contraction default), the two sides SHOULD still land on
//! bit-identical distance computations. This test is the only place that
//! assumption is actually exercised, and it also guards against a future
//! `mul_add` edit on the Rust side silently reintroducing contraction.
//!
//! Run: `RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release --
//! --ignored native_parity --nocapture`

use nanoflann_ref::{Metric, RefIndexF64};
use xval::{build_rust_f64, cfg_seed, queries, uniform, BuildThreads, XMetric};

#[test]
#[ignore]
fn native_parity_build_knn_dim8_f64() {
    const N: usize = 50_000;
    const DIM: usize = 8;
    const LEAF: usize = 10;
    const K: usize = 10;
    const N_QUERIES: usize = 100;

    let data = uniform(cfg_seed("native_parity", &[N, DIM]), N, DIM);
    let q = queries(cfg_seed("native_parity_q", &[N, DIM]), &data, DIM, N_QUERIES);

    let rust_idx = build_rust_f64(&data, DIM, XMetric::L2, LEAF, BuildThreads::Sequential);
    let cpp_idx = RefIndexF64::build(&data, DIM, Metric::L2, LEAF, 1);

    for i in 0..N_QUERIES {
        let query = &q[i * DIM..(i + 1) * DIM];
        let (r_idx, r_dist) = rust_idx.knn(query, K, 0.0);
        let (c_idx, c_dist) = cpp_idx.knn(query, K, 0.0);
        assert_eq!(r_idx.len(), c_idx.len(), "query {i}: result count mismatch: rust={} cpp={}", r_idx.len(), c_idx.len());
        for rank in 0..r_idx.len() {
            assert_eq!(
                r_idx[rank], c_idx[rank],
                "query {i} rank {rank}: index mismatch rust={} cpp={}",
                r_idx[rank], c_idx[rank]
            );
            // The literal `to_bits()` equality the brief asks for -- NOT
            // `assert_knn_equal_f64`'s ulp_diff comparator, which
            // deliberately treats +0.0/-0.0 as equal (0 ULP apart) for the
            // rest of this crate's cross-validation suite; this canary
            // wants the raw bit pattern, unconditionally.
            assert_eq!(
                r_dist[rank].to_bits(), c_dist[rank].to_bits(),
                "query {i} rank {rank}: distance to_bits() mismatch rust={:?} (bits={:#x}) cpp={:?} (bits={:#x})",
                r_dist[rank], r_dist[rank].to_bits(), c_dist[rank], c_dist[rank].to_bits()
            );
        }
    }

    println!(
        "native_parity_build_knn_dim8_f64: PASSED -- {N_QUERIES} queries, k={K}, dim={DIM}, f64, \
         bit-exact (to_bits()) under whatever RUSTFLAGS this run used"
    );
}

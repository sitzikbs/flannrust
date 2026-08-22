//! Cross-validation: BUILD parity between the Rust `KdTree` and the C++
//! nanoflann 1.12.1 oracle -- the strongest check, since it validates the
//! build-time permutation (`vind` / nanoflann's `vAcc_`) directly, rather
//! than through the lens of a particular query.

use nanoflann_ref::{RefIndexF32, RefIndexF64};
use xval::{
    all_identical, build_rust_f32, build_rust_f64, cfg_seed, clustered, exponential_spacing,
    to_f32, uniform, with_duplicates, BuildThreads, XMetric,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetKind {
    Uniform,
    Clustered,
    WithDuplicates,
    AllIdentical,
}

const DATASET_KINDS: [DatasetKind; 4] = [
    DatasetKind::Uniform,
    DatasetKind::Clustered,
    DatasetKind::WithDuplicates,
    DatasetKind::AllIdentical,
];

impl DatasetKind {
    fn generate(self, seed: u64, n: usize, dim: usize) -> Vec<f64> {
        match self {
            DatasetKind::Uniform => uniform(seed, n, dim),
            DatasetKind::Clustered => clustered(seed, n, dim, 5),
            DatasetKind::WithDuplicates => with_duplicates(seed, n, dim, 0.3),
            DatasetKind::AllIdentical => all_identical(n, dim),
        }
    }

    fn tag(self) -> usize {
        match self {
            DatasetKind::Uniform => 0,
            DatasetKind::Clustered => 1,
            DatasetKind::WithDuplicates => 2,
            DatasetKind::AllIdentical => 3,
        }
    }
}

/// Dumps the first 20 differing positions on mismatch (length mismatch is
/// reported directly, without a position dump, since there's nothing
/// position-aligned to show).
fn assert_vind_eq(rust: &[u32], cpp: &[u32], ctx: &str) {
    assert_eq!(
        rust.len(),
        cpp.len(),
        "{ctx}\nvind LENGTH mismatch: rust={} cpp={}",
        rust.len(),
        cpp.len()
    );
    let mut diffs: Vec<(usize, u32, u32)> = Vec::new();
    for i in 0..rust.len() {
        if rust[i] != cpp[i] {
            diffs.push((i, rust[i], cpp[i]));
            if diffs.len() >= 20 {
                break;
            }
        }
    }
    if !diffs.is_empty() {
        panic!(
            "{ctx}\nvind MISMATCH -- first {} differing (position, rust_val, cpp_val): {:?}",
            diffs.len(),
            diffs
        );
    }
}

// ---------------------------------------------------------------------
// Main matrix: dims x {uniform, clustered, with_duplicates, all_identical}
// x leaf, n=1000, both scalars.
// ---------------------------------------------------------------------

macro_rules! build_matrix_test {
    ($fn_name:ident, $t:ty, $build_rust:ident, $RefIndex:ident, $cast:expr) => {
        #[test]
        fn $fn_name() {
            let dims = [2usize, 3, 8, 16, 32];
            let leaves = [1usize, 10, 64];
            let n = 1000usize;

            for &dim in &dims {
                for &dsk in &DATASET_KINDS {
                    for &leaf in &leaves {
                        let seed = cfg_seed("build_matrix", &[dim, dsk.tag(), leaf]);
                        let data64 = dsk.generate(seed, n, dim);
                        let data: Vec<$t> = $cast(&data64);

                        let rust_idx = $build_rust(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
                        let cpp_idx = $RefIndex::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);

                        let ctx = format!(
                            "build_matrix config: dim={dim} dataset={:?} leaf={leaf} n={n} seed={seed}",
                            dsk
                        );
                        assert_vind_eq(&rust_idx.vind(), &cpp_idx.vind(), &ctx);

                        assert!(rust_idx.used_memory_bytes() > 0, "{ctx}\nrust used_memory_bytes() must be > 0");
                        assert!(cpp_idx.used_memory() > 0, "{ctx}\ncpp used_memory() must be > 0");
                    }
                }
            }
        }
    };
}

build_matrix_test!(build_matrix_f64, f64, build_rust_f64, RefIndexF64, |v: &Vec<f64>| v.clone());
build_matrix_test!(build_matrix_f32, f32, build_rust_f32, RefIndexF32, |v: &Vec<f64>| to_f32(v));

// ---------------------------------------------------------------------
// exponential_spacing (dim 1) -- deep degenerate tree parity. Run on a
// 32 MiB stack thread: the C++ oracle's `divideTree` is UNBOUNDED native
// recursion (unmodified nanoflann.hpp), and this spine construction is
// EXACTLY the one nanoflann-rs's own `heavy_exponential_build_1m` test
// (crates/nanoflann-rs/src/build.rs) measured reaching depth ~2115 at
// n=1M -- at n=1000/10_000 depth is capped near n-1 but still far beyond
// what a default ~2 MiB test thread stack safely holds for a
// template-heavy recursive C++ function. The ITERATIVE Rust builder
// (`SubtreeBuilder`, explicit stack) has no such limit and would be safe on
// the default stack; the C++ side is what needs the larger stack.
// ---------------------------------------------------------------------

/// f32-NATIVE analog of `exponential_spacing`: same shrinking-halves spine
/// construction, but starting near `f32::MAX` (`2^127`) and computed
/// ENTIRELY in f32 arithmetic -- deliberately NOT `to_f32(&exponential_spacing(n))`.
///
/// INVESTIGATION (see task-11-report.md for the full writeup): the naive
/// `to_f32(&exponential_spacing(n))` composition was tried first and found
/// to hang/OOM. Root cause: `exponential_spacing`'s spine starts at `2^1023`
/// (f64::MAX-ish) to stress f64's dynamic range, per the formula proven
/// degenerate in nanoflann-rs's `heavy_exponential_build_1m` (an f64-ONLY
/// test -- no f32 analog exists there). `2^1023` is ~900 orders of magnitude
/// past f32::MAX (~2^128), so casting saturates: 896 of the first 1000 spine
/// values become the EXACT SAME `f32::INFINITY`. Building a KD-tree over
/// mostly-identical-infinity points is outside both implementations' domain
/// -- confirmed by isolating EACH side with `ulimit -v`-capped/timeout-capped
/// debug runs: the Rust side OOMs (unbounded arena growth) and the
/// UNMODIFIED C++ oracle SIGSEGVs (stack overflow) on the exact same data,
/// independently. The likely shared mechanism: an all-`+inf` bbox has
/// `span = inf - inf = NaN`, and NaN fails every `<` comparison in the
/// split-axis/cutval selection, so partitioning can never reduce the
/// candidate set -- recursion (or, on the Rust side, iterative arena growth)
/// never terminates. This is a property of feeding IEEE infinity as point
/// coordinates to a spatial index -- not specific to either implementation
/// (both the untouched C++ oracle and the Rust port break on it) -- so it is
/// a bug in this xval file's ORIGINAL data composition, not a finding about
/// earlier tasks. Confirmed fixed: this native-f32-range generator builds
/// cleanly (finite throughout) and produces a BIT-IDENTICAL `vind` on both
/// sides.
fn exponential_spacing_f32_native(n: usize) -> Vec<f32> {
    let mut spine: Vec<f32> = Vec::new();
    let mut v = 2f32.powi(127);
    while v > 0.0 && spine.len() < n.saturating_sub(1) {
        spine.push(v);
        v /= 2.0;
    }
    spine.resize(n, 0.0);
    spine
}

fn run_on_big_stack<F: FnOnce() + Send>(f: F) {
    std::thread::scope(|s| {
        let handle = std::thread::Builder::new()
            .stack_size(32 << 20)
            .spawn_scoped(s, f)
            .expect("failed to spawn 32MiB-stack worker thread");
        handle
            .join()
            .expect("exponential_spacing worker thread panicked (see panic message above)");
    });
}

#[test]
fn build_exponential_spacing_n1000_both_scalars() {
    run_on_big_stack(|| {
        let leaves = [1usize, 10, 64];
        let n = 1000usize;
        let dim = 1usize;

        for &leaf in &leaves {
            let data64 = exponential_spacing(n);

            let rust_idx64 = build_rust_f64(&data64, dim, XMetric::L2, leaf, BuildThreads::Sequential);
            let cpp_idx64 = RefIndexF64::build(&data64, dim, XMetric::L2.to_ref(), leaf, 1);
            let ctx64 = format!("exponential_spacing f64 config: leaf={leaf} n={n}");
            assert_vind_eq(&rust_idx64.vind(), &cpp_idx64.vind(), &ctx64);
            assert!(rust_idx64.used_memory_bytes() > 0, "{ctx64}\nrust used_memory_bytes() must be > 0");
            assert!(cpp_idx64.used_memory() > 0, "{ctx64}\ncpp used_memory() must be > 0");

            let data32 = exponential_spacing_f32_native(n);
            let rust_idx32 = build_rust_f32(&data32, dim, XMetric::L2, leaf, BuildThreads::Sequential);
            let cpp_idx32 = RefIndexF32::build(&data32, dim, XMetric::L2.to_ref(), leaf, 1);
            let ctx32 = format!("exponential_spacing f32 config: leaf={leaf} n={n}");
            assert_vind_eq(&rust_idx32.vind(), &cpp_idx32.vind(), &ctx32);
            assert!(rust_idx32.used_memory_bytes() > 0, "{ctx32}\nrust used_memory_bytes() must be > 0");
            assert!(cpp_idx32.used_memory() > 0, "{ctx32}\ncpp used_memory() must be > 0");
        }
    });
}

// ---------------------------------------------------------------------
// Parallel build cross-language parity (task 12): Rust `BuildThreads::Auto`
// vs C++ `n_thread_build=4` -- C++'s `divideTreeConcurrent` also partitions
// the whole range BEFORE spawning worker threads (confirmed in
// `crates/nanoflann-rs/src/build.rs`'s `SubtreeBuilder::build` doc comment,
// which cites having read `divideTreeConcurrent`, nanoflann.hpp ~1422-1479),
// so its `vAcc_` is identical to what its OWN sequential build produces --
// meaning this closes the loop cross-language: Rust's `Auto` build must
// match C++'s single-threaded `vind`, which in turn is what `n_thread_build=4`
// must ALSO produce.
// ---------------------------------------------------------------------

#[test]
fn build_parallel_auto_matches_cpp_n_thread_build_4_uniform_n5000_dim3_f64() {
    let n = 5000usize;
    let dim = 3usize;
    let leaf = 10usize;
    let seed = cfg_seed("build_parallel_auto_vs_cpp_threads", &[n, dim, leaf]);
    let data = uniform(seed, n, dim);

    let rust_idx = build_rust_f64(&data, dim, XMetric::L2, leaf, BuildThreads::Auto);
    let cpp_idx = RefIndexF64::build(&data, dim, XMetric::L2.to_ref(), leaf, 4);

    let ctx = format!("build_parallel_auto_vs_cpp config: n={n} dim={dim} leaf={leaf} seed={seed}");
    assert_vind_eq(&rust_idx.vind(), &cpp_idx.vind(), &ctx);
    assert!(rust_idx.used_memory_bytes() > 0, "{ctx}\nrust used_memory_bytes() must be > 0");
    assert!(cpp_idx.used_memory() > 0, "{ctx}\ncpp used_memory() must be > 0");
}

#[test]
fn build_exponential_spacing_n10000_leaf10_f64() {
    run_on_big_stack(|| {
        let n = 10_000usize;
        let leaf = 10usize;
        let dim = 1usize;
        let data = exponential_spacing(n);

        let rust_idx = build_rust_f64(&data, dim, XMetric::L2, leaf, BuildThreads::Sequential);
        let cpp_idx = RefIndexF64::build(&data, dim, XMetric::L2.to_ref(), leaf, 1);
        let ctx = format!("exponential_spacing deep config: n={n} leaf={leaf}");
        assert_vind_eq(&rust_idx.vind(), &cpp_idx.vind(), &ctx);
        assert!(rust_idx.used_memory_bytes() > 0, "{ctx}\nrust used_memory_bytes() must be > 0");
        assert!(cpp_idx.used_memory() > 0, "{ctx}\ncpp used_memory() must be > 0");
    });
}


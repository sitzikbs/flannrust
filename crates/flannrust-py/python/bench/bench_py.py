"""flannrust vs scipy.spatial.cKDTree vs pynanoflann -- Python-binding
speed benchmark. Emits ONE JSON document on stdout: `{meta: {...}, workloads:
[{name, flannrust_ms, ckdtree_ms, pynanoflann_ms, ratio_ckdtree,
ratio_pynanoflann}, ...]}`. `crates/xval/examples/render_report.rs` reads
this JSON (as its optional 2nd argv) and renders a "Python bindings"
section alongside the Rust-vs-C++ scorecard.

Re-run with (from the repo root):

    cd crates/flannrust-py
    RUSTFLAGS="-C target-cpu=native" .venv/bin/maturin develop --release
    .venv/bin/python python/bench/bench_py.py > /tmp/report_py.json

`RUSTFLAGS="-C target-cpu=native"` is REQUIRED before running this script --
without it the flannrust wheel is not built with the same codegen flags as
the rest of the repo's Rust-vs-C++ numbers, and the comparison here would
not be apples-to-apples with that methodology (see docs/EXPERIMENTS.md
Section 1). `meta.rustflags` records what was actually set in THIS
process's environment at run time (best-effort signal, not a guarantee --
see `report_data.rs`'s `target_cpu_native()` for the same caveat).

Methodology (mirrors `crates/xval/src/lib.rs`'s `timed_median_ms`/
`median_of` and `report_data.rs`'s workload structure, adapted for Python):
- `time.perf_counter()`, median of 7 repeats per cell.
- Within each repeat, every engine being compared in that cell is timed
  once, back-to-back (interleaved A/B/C/A/B/C/...), rather than all of A's
  7 repeats then all of B's -- this spreads any thermal/scheduler drift
  evenly across engines instead of biasing whichever one runs first or
  last (WSL2 noise -- docs/EXPERIMENTS.md Section 1).
- `gc.disable()` for the entire timed block (see `timed_median_interleaved`).
- Every dataset is seeded (numpy `default_rng`, CRC32-derived integer seeds
  via `_seed()` -- deterministic across runs/processes, unlike Python's
  salted `hash()`).

Cross-check before timing: for every distinct (n, dim, dtype) dataset used
below, `cross_check()` runs a 10-query tie-free ("uniform") sample through
both `flannrust.KDTree.query` and `pynanoflann.KDTree.kneighbors` BEFORE
any timing happens, and asserts (a) identical returned indices and (b)
sqrt-mapped distances agree to within 1e-6 relative tolerance (NOT
bit-exact -- pynanoflann vendors nanoflann 1.5.5, which has a documented
1-ULP summation-order gap vs this repo's own 1.12.1 C++ reference; see
`python/tests/test_parity_pynanoflann.py`'s module docstring for the full
root-cause writeup). A cross-check failure aborts the whole bench with a
clear error -- never silently benchmarks against a wrong-answer engine.

Distances: flannrust/nanoflann's l2 metric is SQUARED; cKDTree's `query`/
`query_ball_point` and pynanoflann's `kneighbors`/`radius_neighbors` all
take/return EUCLIDEAN (unsquared) distances -- every cross-engine radius
argument and cross-check comparison below maps explicitly via sqrt.

Build-thread comparison caveat: neither cKDTree nor pynanoflann exposes a
parallel-build option (both build single-threaded only). The
"...threadsNone_parallel_build" workload rows therefore compare flannrust's
`threads=None` (parallel) build against the SAME single-thread cKDTree/
pynanoflann build numbers already captured in the sibling "...threads1" row
-- an intentionally-labelled apples-to-oranges comparison (see each such
row's `note` field), not a parity claim.
"""

import gc
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy.spatial import cKDTree

import pynanoflann

import flannrust

# Reuse the pytest suite's own seeded-dataset generator rather than
# reimplementing it -- see python/tests/conftest.py.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")))
from conftest import make_points  # noqa: E402

RUNS = 7
LEAF = 10


def progress(msg):
    print(f"[bench_py] {msg}", file=sys.stderr, flush=True)


def _seed(*parts):
    """Deterministic seed (Python's `hash()` on str is salted per-process by
    default) -- mirrors test_parity_pynanoflann.py's `_seed`."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


# ============================================================================
# meta
# ============================================================================


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_meta():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pynanoflann": _pkg_version("pynanoflann"),
        "cpu": _cpu_model(),
        "threads": os.cpu_count() or 1,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "rustflags": os.environ.get("RUSTFLAGS", ""),
        "wheel_profile": (
            "release (.venv/bin/maturin develop --release); "
            "RUSTFLAGS=\"-C target-cpu=native\" REQUIRED at build time for parity "
            "with the repo's Rust-vs-C++ xval methodology -- see meta.rustflags "
            "for what THIS run's process environment actually had set"
        ),
    }


# ============================================================================
# timing
# ============================================================================


def timed_median_interleaved(fns, runs=RUNS):
    """`fns`: dict[name -> zero-arg callable]. Each repeat runs every
    callable once, interleaved in dict order (A, B, C, A, B, C, ...) rather
    than clustered (AAAAAAA, BBBBBBB, ...) -- see module docstring.
    `gc` is disabled for the whole timed block. Returns dict[name -> median
    elapsed ms across `runs`]."""
    times = {name: [] for name in fns}
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(runs):
            for name, fn in fns.items():
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
                times[name].append((t1 - t0) * 1000.0)
    finally:
        if was_enabled:
            gc.enable()
    return {name: statistics.median(v) for name, v in times.items()}


def make_row(name, flannrust_ms, ckdtree_ms, pynanoflann_ms, note=None):
    row = {
        "name": name,
        "flannrust_ms": flannrust_ms,
        "ckdtree_ms": ckdtree_ms,
        "pynanoflann_ms": pynanoflann_ms,
        "ratio_ckdtree": flannrust_ms / ckdtree_ms,
        "ratio_pynanoflann": flannrust_ms / pynanoflann_ms,
    }
    if note:
        row["note"] = note
    return row


# ============================================================================
# cross-check (the correctness "test" for this bench -- see module docstring)
# ============================================================================


def cross_check(ftree, pnf, dim, dtype, metric, seed, label):
    q = make_points(10, dim, dtype, seed=seed, kind="uniform")
    d_ours, i_ours = ftree.query(q, k=10)
    d_pnf, i_pnf = pnf.kneighbors(q, 10)

    if not np.array_equal(np.asarray(i_ours), np.asarray(i_pnf)):
        raise SystemExit(
            f"bench_py: cross-check FAILED ({label}): flannrust/pynanoflann knn "
            f"indices differ on a 10-query tie-free sample -- refusing to time a "
            f"benchmark against a wrong-answer oracle.\nours=\n{i_ours}\npnf=\n{i_pnf}"
        )

    d_ours_mapped = (np.sqrt(d_ours) if metric == "l2" else d_ours).astype(np.float64)
    d_pnf64 = np.asarray(d_pnf, dtype=np.float64)
    rel = np.abs(d_ours_mapped - d_pnf64) / np.maximum(np.abs(d_pnf64), 1e-12)
    if not np.all(rel < 1e-6):
        raise SystemExit(
            f"bench_py: cross-check FAILED ({label}): sqrt-mapped knn distances "
            f"differ from pynanoflann by more than 1e-6 relative tolerance "
            f"(max observed: {float(rel.max()):.3e})"
        )
    progress(f"cross-check OK ({label}): 10/10 index match, max rel dist err {float(rel.max()):.2e}")


# ============================================================================
# workloads
# ============================================================================


def build_workload(n, name_prefix):
    pts = make_points(n, 3, "float32", seed=_seed("build", n), kind="uniform")

    ftree_cc = flannrust.KDTree(pts, leaf_size=LEAF, metric="l2", threads=1)
    pnf_cc = pynanoflann.KDTree(n_neighbors=10, leaf_size=LEAF, metric="l2")
    pnf_cc.fit(pts)
    cross_check(ftree_cc, pnf_cc, 3, "float32", "l2", _seed("build_cc", n), f"{name_prefix} dataset")

    def build_ckdtree():
        cKDTree(pts, leafsize=LEAF, balanced_tree=False, compact_nodes=False)

    def build_pynanoflann():
        pnf = pynanoflann.KDTree(n_neighbors=10, leaf_size=LEAF, metric="l2")
        pnf.fit(pts)

    threads1 = timed_median_interleaved(
        {
            "flannrust": lambda: flannrust.KDTree(pts, leaf_size=LEAF, metric="l2", threads=1),
            "ckdtree": build_ckdtree,
            "pynanoflann": build_pynanoflann,
        }
    )
    threads_none = timed_median_interleaved(
        {"flannrust": lambda: flannrust.KDTree(pts, leaf_size=LEAF, metric="l2", threads=None)}
    )

    return [
        make_row(f"{name_prefix}_threads1", threads1["flannrust"], threads1["ckdtree"], threads1["pynanoflann"]),
        make_row(
            f"{name_prefix}_threadsNone_parallel_build",
            threads_none["flannrust"],
            threads1["ckdtree"],
            threads1["pynanoflann"],
            note=(
                "flannrust threads=None (parallel build) vs cKDTree/pynanoflann single-thread "
                "build -- neither offers a parallel-build option, so this row reuses the "
                "sibling threads1 row's cKDTree/pynanoflann numbers. Labelled honestly, not a "
                "same-config parity claim."
            ),
        ),
    ]


def _pnf_jobs(workers):
    """pynanoflann's `n_jobs` (unlike scipy's `workers`/cKDTree's `workers`)
    is a literal C++ thread count -- it does NOT special-case -1 as "all
    cores" (verified: `KDTree.kneighbors_multithreaded` raises a pybind11
    TypeError on a negative count). Map flannrust/cKDTree's `-1` convention
    to the actual core count for pynanoflann's call only."""
    return os.cpu_count() or 1 if workers == -1 else workers


def batched_knn_dim3_workload():
    n, dim, dtype, n_queries, k = 1_000_000, 3, "float32", 200_000, 10
    pts = make_points(n, dim, dtype, seed=_seed("knn_batch", n, dim), kind="uniform")
    q = make_points(n_queries, dim, dtype, seed=_seed("knn_batch_q", n, dim), kind="uniform")

    progress("knn_batched_dim3_f32: building trees (untimed)")
    ftree = flannrust.KDTree(pts, leaf_size=LEAF, metric="l2", threads=None)
    ctree = cKDTree(pts, leafsize=LEAF, balanced_tree=False, compact_nodes=False)
    pnf = pynanoflann.KDTree(n_neighbors=k, leaf_size=LEAF, metric="l2")
    pnf.fit(pts)

    cross_check(ftree, pnf, dim, dtype, "l2", _seed("knn_batch_cc", n), "knn_batched_dim3_f32 dataset")

    rows = []
    for workers, label in [(1, "workers1"), (-1, "workersNeg1")]:
        timings = timed_median_interleaved(
            {
                "flannrust": lambda w=workers: ftree.query(q, k=k, workers=w),
                "ckdtree": lambda w=workers: ctree.query(q, k=k, workers=w),
                "pynanoflann": lambda w=workers: pnf.kneighbors(q, k, n_jobs=_pnf_jobs(w)),
            }
        )
        rows.append(
            make_row(
                f"knn_batched_dim3_f32_k{k}_q{n_queries // 1000}k_{label}",
                timings["flannrust"],
                timings["ckdtree"],
                timings["pynanoflann"],
            )
        )
    return rows


def knn_dim_workload(dim, dtype, k=10):
    n, n_queries = 100_000, 20_000
    pts = make_points(n, dim, dtype, seed=_seed("knn_dim", dim, dtype), kind="uniform")
    q = make_points(n_queries, dim, dtype, seed=_seed("knn_dim_q", dim, dtype), kind="uniform")

    ftree = flannrust.KDTree(pts, leaf_size=LEAF, metric="l2", threads=None)
    ctree = cKDTree(pts, leafsize=LEAF, balanced_tree=False, compact_nodes=False)
    pnf = pynanoflann.KDTree(n_neighbors=k, leaf_size=LEAF, metric="l2")
    pnf.fit(pts)

    cross_check(ftree, pnf, dim, dtype, "l2", _seed("knn_dim_cc", dim, dtype), f"knn_dim{dim}_{dtype} dataset")

    timings = timed_median_interleaved(
        {
            "flannrust": lambda: ftree.query(q, k=k, workers=1),
            "ckdtree": lambda: ctree.query(q, k=k, workers=1),
            "pynanoflann": lambda: pnf.kneighbors(q, k, n_jobs=1),
        }
    )
    return make_row(
        f"knn_dim{dim}_{dtype}_k{k}_workers1", timings["flannrust"], timings["ckdtree"], timings["pynanoflann"]
    )


def radius_workload():
    n, dim, dtype, n_queries = 100_000, 3, "float32", 2_000
    pts = make_points(n, dim, dtype, seed=_seed("radius", n), kind="uniform")
    q = make_points(n_queries, dim, dtype, seed=_seed("radius_q", n), kind="uniform")

    ftree = flannrust.KDTree(pts, leaf_size=LEAF, metric="l2", threads=None)
    ctree = cKDTree(pts, leafsize=LEAF, balanced_tree=False, compact_nodes=False)
    pnf = pynanoflann.KDTree(n_neighbors=10, leaf_size=LEAF, metric="l2")
    pnf.fit(pts)

    cross_check(ftree, pnf, dim, dtype, "l2", _seed("radius_cc", n), "radius_dim3_f32 dataset")

    rows = []
    for target_k, label in [(10, "sel10"), (1000, "sel1000")]:
        # Calibrate a squared radius, at q[0], whose neighbor count is
        # approximately `target_k` -- same calibration idea as
        # report_data.rs's radius_dim3_f32 workload.
        d_sel, _ = ftree.query(q[0].reshape(1, -1), k=target_k)
        r_sq = float(d_sel[0, -1])
        r_euclid = math.sqrt(r_sq)

        timings = timed_median_interleaved(
            {
                "flannrust": lambda rs=r_sq: ftree.query_radius(q, rs, workers=1),
                "ckdtree": lambda re=r_euclid: ctree.query_ball_point(q, re, workers=1),
                "pynanoflann": lambda re=r_euclid: pnf.radius_neighbors(q, radius=re, n_jobs=1),
            }
        )
        rows.append(
            make_row(f"radius_dim3_f32_{label}", timings["flannrust"], timings["ckdtree"], timings["pynanoflann"])
        )
    return rows


def single_query_loop_workload():
    n, dim, dtype, n_calls, k = 100_000, 3, "float32", 1_000, 10
    pts = make_points(n, dim, dtype, seed=_seed("single_q", n), kind="uniform")
    q = make_points(n_calls, dim, dtype, seed=_seed("single_q_q", n), kind="uniform")

    ftree = flannrust.KDTree(pts, leaf_size=LEAF, metric="l2", threads=None)
    ctree = cKDTree(pts, leafsize=LEAF, balanced_tree=False, compact_nodes=False)
    pnf = pynanoflann.KDTree(n_neighbors=k, leaf_size=LEAF, metric="l2")
    pnf.fit(pts)

    cross_check(ftree, pnf, dim, dtype, "l2", _seed("single_q_cc", n), "single_query_loop dataset")

    def loop_flannrust():
        for row in q:
            ftree.query(row, k=k, workers=1)

    def loop_ckdtree():
        for row in q:
            ctree.query(row, k=k, workers=1)

    def loop_pynanoflann():
        for row in q:
            pnf.kneighbors(row.reshape(1, -1), k, n_jobs=1)

    totals = timed_median_interleaved(
        {"flannrust": loop_flannrust, "ckdtree": loop_ckdtree, "pynanoflann": loop_pynanoflann}
    )
    per_call = {name: v / n_calls for name, v in totals.items()}
    return make_row(
        f"single_query_loop_dim3_f32_k{k}_percall_ms",
        per_call["flannrust"],
        per_call["ckdtree"],
        per_call["pynanoflann"],
        note=(
            "overhead-bound (see docs/EXPERIMENTS.md): per-call Python/pybind11 binding "
            "overhead dominates at n_calls=1 scale, not tree-traversal cost. ms values here "
            f"are PER-CALL (loop's own median-of-{RUNS} total time / {n_calls}), not total loop time."
        ),
    )


# ============================================================================
# main
# ============================================================================


def main():
    progress("starting")
    meta = build_meta()

    workloads = []
    progress("build_100k_dim3_f32")
    workloads += build_workload(100_000, "build_100k_dim3_f32")
    progress("build_1M_dim3_f32")
    workloads += build_workload(1_000_000, "build_1M_dim3_f32")
    progress("knn_batched_dim3_f32")
    workloads += batched_knn_dim3_workload()
    progress("knn_dim8_f64")
    workloads.append(knn_dim_workload(8, "float64"))
    progress("knn_dim32_f32")
    workloads.append(knn_dim_workload(32, "float32"))
    progress("radius_dim3_f32")
    workloads += radius_workload()
    progress("single_query_loop")
    workloads.append(single_query_loop_workload())

    doc = {"meta": meta, "workloads": workloads}
    print(json.dumps(doc, indent=2))
    progress("done")


if __name__ == "__main__":
    main()

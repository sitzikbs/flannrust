"""flannrust vs scipy.spatial.cKDTree vs pynanoflann -- Python-binding
speed benchmark. Emits ONE JSON document on stdout: `{schema_version: 2,
meta: {...}, workloads: [{name, flannrust_stats, ckdtree_stats,
pynanoflann_stats, ratio_ckdtree, ratio_pynanoflann}, ...]}`.
`crates/xval/examples/render_report.rs` (via `xval::render_python`) reads
this JSON (as its optional 2nd argv) and renders a "Python bindings"
section alongside the Rust-vs-C++ scorecard -- and REFUSES a document
whose `schema_version` isn't `2` (a pre-M2.6-task-3 JSON, `flannrust_ms`
flat fields instead of `flannrust_stats` objects, is rejected loudly
rather than silently mis-rendered).

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
`meta.loadavg_start`/`meta.loadavg_end` (M2.6 task 6) record `/proc/loadavg`'s
raw line at process start and just before the JSON is printed -- the
Python-side half of this repo's idle-host measurement-conditions protocol
(docs/EXPERIMENTS.md Section 1); `None` off-Linux or on any read failure.
The Rust-side perf gate/report-chain sessions have no equivalent automatic
capture and record loadavg manually per that same protocol.

Methodology (M2.6 task 3 -- mirrors `crates/xval/src/lib.rs`'s
`TimingStats`/`rep_count`/`measure_pair` adaptive-repetition policy,
generalized from 2 sides to N engines):
- `time.perf_counter()`. `WARMUP_REPS` (2) untimed-but-clocked warmup
  reps, then `rep_count(BUDGET_S, t_est_ms)` timed reps -- `n =
  clamp(10, 100, floor(budget_s*1000 / t_est_ms))`, where `t_est_ms` is
  the SLOWEST engine's own 2-warmup-rep mean (keeps every engine within
  `BUDGET_S` at the shared `n`, same as `xval::measure_pair`). Warmup
  reps are excluded from the published `TimingStats`-shaped sample
  (`mean_ms`/`std_ms` [sample, n-1] `/median_ms`/`min_ms`/`max_ms`/`n`)
  -- see `stats_from_samples`.
- Within each repeat (warmup AND timed), every engine being compared in
  that cell is timed once, back-to-back (interleaved A/B/C/A/B/C/...),
  rather than all of A's reps then all of B's -- this spreads any
  thermal/scheduler drift evenly across engines instead of biasing
  whichever one runs first or last (WSL2 noise -- docs/EXPERIMENTS.md
  Section 1).
- `gc.disable()` for the entire timed block (warmup + timed reps -- see
  `timed_stats_interleaved`).
- Every dataset is seeded (numpy `default_rng`, CRC32-derived integer seeds
  via `_seed()` -- deterministic across runs/processes, unlike Python's
  salted `hash()`).
- Ratios (`ratio_ckdtree`/`ratio_pynanoflann`) are computed from MEDIANS
  (the gate/doc decision statistic, same as `xval`'s report chain) --
  mean/std/n are published alongside for statistical honesty, not used
  for the ratio.

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

Runtime: the adaptive `n` (up to 100 reps on cheap cells, floored at 10 on
expensive ones -- e.g. `knn_dim32_..._workers1`, whose ~40ms-per-call
pynanoflann side alone blows past the 30s/side budget at `n=1`) makes a
full run take on the order of 20-30 minutes, vs. the old fixed-n=7 run's
~7 minutes -- `progress()` prefixes every line with elapsed wall time
since process start, and `timed_stats_interleaved` prints each cell's
chosen `n`/`t_est_ms` as it starts, so the run's shape is visible live.
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

SCHEMA_VERSION = 2
BUDGET_S = 30.0
WARMUP_REPS = 2
LEAF = 10

_START = time.perf_counter()


def progress(msg):
    elapsed = time.perf_counter() - _START
    print(f"[bench_py +{elapsed:7.1f}s] {msg}", file=sys.stderr, flush=True)


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


def _loadavg():
    """Best-effort `/proc/loadavg` read (Linux-only, matches this repo's
    idle-host measurement-conditions protocol -- docs/EXPERIMENTS.md
    Section 1's "Measurement-conditions protocol"). `None` off-Linux or on
    any read failure, never raises -- meta capture must not abort a bench
    run over an unrelated OS quirk."""
    try:
        with open("/proc/loadavg") as f:
            return f.read().strip()
    except OSError:
        return None


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
        "loadavg_start": _loadavg(),
        "rustflags": os.environ.get("RUSTFLAGS", ""),
        "wheel_profile": (
            "release (.venv/bin/maturin develop --release); "
            "RUSTFLAGS=\"-C target-cpu=native\" REQUIRED at build time for parity "
            "with the repo's Rust-vs-C++ xval methodology -- see meta.rustflags "
            "for what THIS run's process environment actually had set"
        ),
        "budget_s": BUDGET_S,
        "warmup_reps": WARMUP_REPS,
        "rep_policy": (
            "n = clamp(10, 100, floor(budget_s*1000 / t_est_ms)); t_est_ms = mean of "
            "WARMUP_REPS (2) untimed warmup reps, slowest engine in the interleaved "
            "group -- mirrors xval::measure_pair (M2.6 Task 1), generalized to N "
            "engines. Ratios are median-based; mean/std(sample, n-1)/min/max/n "
            "published per engine alongside."
        ),
    }


# ============================================================================
# timing -- M2.6 task 3: adaptive-repetition, mean/std/median/min/max/n,
# generalizing xval::measure_pair (2 sides) to N interleaved engines.
# ============================================================================


def rep_count(budget_s, t_est_ms):
    """Mirrors `xval::rep_count`: `clamp(10, 100, floor(budget_s*1000 /
    t_est_ms))`. `t_est_ms <= 0.0` (a per-rep cost too fast for
    `perf_counter`'s resolution to distinguish from zero) maps to the
    upper bound (100) rather than dividing by zero/a negative."""
    if t_est_ms <= 0.0:
        return 100
    raw = math.floor(budget_s * 1000.0 / t_est_ms)
    return max(10, min(100, raw))


def stats_from_samples(samples_ms):
    """Mirrors `xval::TimingStats::from_samples`: mean/median/min/max plus
    SAMPLE standard deviation (divisor n-1, Bessel-corrected; 0.0 for a
    single-element sample -- zero degrees of freedom, not a 0/0 error)."""
    n = len(samples_ms)
    return {
        "mean_ms": statistics.mean(samples_ms),
        "std_ms": statistics.stdev(samples_ms) if n > 1 else 0.0,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "n": n,
    }


def scale_stats(stats, factor):
    """Rescales a stats dict by a positive scalar (used by
    `single_query_loop_workload` to turn a per-rep TOTAL-loop-time sample
    into a per-call one) -- mean/std/median/min/max all scale linearly
    with a positive factor; `n` (rep count) is unchanged."""
    return {
        "mean_ms": stats["mean_ms"] * factor,
        "std_ms": stats["std_ms"] * factor,
        "median_ms": stats["median_ms"] * factor,
        "min_ms": stats["min_ms"] * factor,
        "max_ms": stats["max_ms"] * factor,
        "n": stats["n"],
    }


def timed_stats_interleaved(fns, budget_s=BUDGET_S, label=None):
    """`fns`: dict[name -> zero-arg callable]. Generalizes the pre-M2.6
    `timed_median_interleaved` (fixed n=7, median-only) to an adaptive `n`
    and full `TimingStats`-shaped output, mirroring `xval::measure_pair`'s
    policy extended from 2 sides to however many callables `fns` holds:

    - `WARMUP_REPS` (2) untimed-but-clocked warmup reps, each repeat
      running every callable once, interleaved in dict order (A, B, C, A,
      B, C, ...) -- same interleave discipline as the timed reps below,
      for the same thermal/scheduler-drift reason.
    - `t_est_ms` = the SLOWEST engine's own mean of its 2 warmup reps --
      using the slowest side keeps every engine's total measured time
      within `budget_s` at the shared `n` (the faster engines, by
      definition, use less than their own budget at that same `n`).
    - `n = rep_count(budget_s, t_est_ms)` timed reps, same interleave
      order, warmup excluded from the published sample.
    - `gc` is disabled for the whole call (warmup + timed).

    Returns dict[name -> stats dict] (see `stats_from_samples`), all
    engines sharing the same `n`. Prints the chosen `n`/`t_est_ms` via
    `progress()` before the timed reps start.
    """
    names = list(fns.keys())
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        warm_ms = {name: [] for name in names}
        for _ in range(WARMUP_REPS):
            for name in names:
                t0 = time.perf_counter()
                fns[name]()
                t1 = time.perf_counter()
                warm_ms[name].append((t1 - t0) * 1000.0)
        t_est_ms = max(statistics.mean(v) for v in warm_ms.values())
        n = rep_count(budget_s, t_est_ms)
        prefix = f"{label}: " if label else ""
        progress(f"  {prefix}n={n} (t_est={t_est_ms:.3f}ms/rep, budget={budget_s:.0f}s/side)")

        samples_ms = {name: [] for name in names}
        for _ in range(n):
            for name in names:
                t0 = time.perf_counter()
                fns[name]()
                t1 = time.perf_counter()
                samples_ms[name].append((t1 - t0) * 1000.0)
    finally:
        if was_enabled:
            gc.enable()
    return {name: stats_from_samples(v) for name, v in samples_ms.items()}


def make_row(name, flannrust_stats, ckdtree_stats, pynanoflann_stats, note=None):
    row = {
        "name": name,
        "flannrust_stats": flannrust_stats,
        "ckdtree_stats": ckdtree_stats,
        "pynanoflann_stats": pynanoflann_stats,
        "ratio_ckdtree": flannrust_stats["median_ms"] / ckdtree_stats["median_ms"],
        "ratio_pynanoflann": flannrust_stats["median_ms"] / pynanoflann_stats["median_ms"],
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


def radius_cross_check(ftree, ctree, pnf, dim, dtype, r_sq, seed, label):
    """Pre-timing correctness gate for the radius codepath -- `cross_check()`
    above only exercises knn (`query`/`kneighbors`); `query_radius`/
    `query_ball_point`/`radius_neighbors` are different code with their own
    unit mapping (ours squared, theirs euclidean via sqrt), boundary
    inclusivity, and selection-count logic, all otherwise unguarded. Compares
    INDEX SETS (not order -- cKDTree/pynanoflann radius results are
    unsorted) across all three engines on a 10-query tie-free sample at the
    exact radius a timed cell is about to use. `seed` must be independent of
    whatever query point `r_sq` was calibrated from (radius_workload's `r_sq`
    sits exactly on that point's k-th-neighbor distance -- a genuine
    engine-boundary-convention tie there, not a bug; a fresh uniform sample
    at a different seed has ~0 probability of landing on that exact same
    boundary, since distances over continuous uniform data are essentially
    never exactly equal). Aborts with a clear error on any mismatch."""
    q = make_points(10, dim, dtype, seed=seed, kind="uniform")
    r_euclid = math.sqrt(r_sq)

    f_idxs, _ = ftree.query_radius(q, r_sq, workers=1)
    c_idxs = ctree.query_ball_point(q, r_euclid, workers=1)
    _, p_idxs = pnf.radius_neighbors(q, radius=r_euclid, n_jobs=1)

    for i in range(len(q)):
        f_set = {int(x) for x in np.asarray(f_idxs[i])}
        c_set = {int(x) for x in c_idxs[i]}
        p_set = {int(x) for x in np.asarray(p_idxs[i])}
        if not (f_set == c_set == p_set):
            raise SystemExit(
                f"bench_py: radius cross-check FAILED ({label}, row {i}): index sets differ -- "
                f"flannrust={sorted(f_set)} ckdtree={sorted(c_set)} pynanoflann={sorted(p_set)}"
            )
    progress(f"radius cross-check OK ({label}): 10/10 rows index-set match (flannrust/cKDTree/pynanoflann)")


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

    threads1 = timed_stats_interleaved(
        {
            "flannrust": lambda: flannrust.KDTree(pts, leaf_size=LEAF, metric="l2", threads=1),
            "ckdtree": build_ckdtree,
            "pynanoflann": build_pynanoflann,
        },
        label=f"{name_prefix}_threads1",
    )
    threads_none = timed_stats_interleaved(
        {"flannrust": lambda: flannrust.KDTree(pts, leaf_size=LEAF, metric="l2", threads=None)},
        label=f"{name_prefix}_threadsNone_parallel_build",
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
        n_jobs = _pnf_jobs(workers)
        name = f"knn_batched_dim3_f32_k{k}_q{n_queries // 1000}k_{label}"
        stats = timed_stats_interleaved(
            {
                "flannrust": lambda w=workers: ftree.query(q, k=k, workers=w),
                "ckdtree": lambda w=workers: ctree.query(q, k=k, workers=w),
                "pynanoflann": lambda j=n_jobs: pnf.kneighbors(q, k, n_jobs=j),
            },
            label=name,
        )
        rows.append(make_row(name, stats["flannrust"], stats["ckdtree"], stats["pynanoflann"]))
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

    name = f"knn_dim{dim}_{dtype}_k{k}_workers1"
    stats = timed_stats_interleaved(
        {
            "flannrust": lambda: ftree.query(q, k=k, workers=1),
            "ckdtree": lambda: ctree.query(q, k=k, workers=1),
            "pynanoflann": lambda: pnf.kneighbors(q, k, n_jobs=1),
        },
        label=name,
    )
    return make_row(name, stats["flannrust"], stats["ckdtree"], stats["pynanoflann"])


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

        radius_cross_check(
            ftree, ctree, pnf, dim, dtype, r_sq, _seed("radius_cc2", n, target_k), f"radius_dim3_f32_{label}"
        )

        name = f"radius_dim3_f32_{label}"
        stats = timed_stats_interleaved(
            {
                "flannrust": lambda rs=r_sq: ftree.query_radius(q, rs, workers=1),
                "ckdtree": lambda re=r_euclid: ctree.query_ball_point(q, re, workers=1),
                "pynanoflann": lambda re=r_euclid: pnf.radius_neighbors(q, radius=re, n_jobs=1),
            },
            label=name,
        )
        rows.append(make_row(name, stats["flannrust"], stats["ckdtree"], stats["pynanoflann"]))
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

    name = f"single_query_loop_dim3_f32_k{k}_percall_ms"
    totals = timed_stats_interleaved(
        {"flannrust": loop_flannrust, "ckdtree": loop_ckdtree, "pynanoflann": loop_pynanoflann},
        label=name,
    )
    per_call = {name_: scale_stats(v, 1.0 / n_calls) for name_, v in totals.items()}
    return make_row(
        name,
        per_call["flannrust"],
        per_call["ckdtree"],
        per_call["pynanoflann"],
        note=(
            "overhead-bound (see docs/EXPERIMENTS.md): per-call Python/pybind11 binding "
            "overhead dominates at n_calls=1 scale, not tree-traversal cost. Each stats "
            f"object here is PER-CALL (the loop's own {WARMUP_REPS}-warmup/adaptive-n "
            f"TOTAL loop time per rep, divided by n_calls={n_calls}), not total loop time."
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

    meta["loadavg_end"] = _loadavg()
    doc = {"schema_version": SCHEMA_VERSION, "meta": meta, "workloads": workloads}
    print(json.dumps(doc, indent=2))
    progress("done")


if __name__ == "__main__":
    main()

# flannrust

[![CI](https://github.com/sitzikbs/flannrust/actions/workflows/ci.yml/badge.svg)](https://github.com/sitzikbs/flannrust/actions/workflows/ci.yml)
[![License: BSD-2-Clause](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/flannrust.svg)](https://crates.io/crates/flannrust)
[![docs.rs](https://docs.rs/flannrust/badge.svg)](https://docs.rs/flannrust)
[![PyPI](https://img.shields.io/pypi/v/flannrust.svg)](https://pypi.org/project/flannrust/)

A Rust port of [nanoflann](https://github.com/jlblancoc/nanoflann) (the C++
kd-tree library), targeting bit-exact result parity with the C++ reference
and equal-or-better speed. Static and dynamic indexes, with Python bindings.

## Features

- **Bit-exact parity with nanoflann 1.12.1** — every result index, distance,
  and internal tree permutation is cross-validated in-process against the
  vendored C++ reference on every commit ([docs/testing.md](docs/testing.md))
- **Static kd-tree** (`KdTree`) and **dynamic** Bentley–Saxe forest
  (`DynamicKdTree`) with point add/remove after construction
- **Matches or beats the C++** on most benchmarked workloads (table below)
- L1 / L2 / L2-simple / SO2 / SO3 metrics; `f32`/`f64`; compile-time
  (`ConstDim`) or runtime (`DynDim`) dimension; `u32`/`u64`/`usize` indices
- Optional parallel build via rayon (default feature `parallel`)
- Python bindings: `flannrust.KDTree` / `DynamicKDTree`, NumPy in/out,
  GIL released during build and query
- Exactly two `unsafe` blocks, both miri-verified in CI

## Installation

Rust (from [crates.io](https://crates.io/crates/flannrust)):

```toml
[dependencies]
flannrust = "0.1"
```

Python (from [PyPI](https://pypi.org/project/flannrust/)):

```bash
pip install flannrust
```

To build from source (e.g. with `target-cpu=native` for SIMD):

```bash
pip install maturin numpy
RUSTFLAGS="-C target-cpu=native" maturin develop -m crates/flannrust-py/Cargo.toml --release
```

MSRV: Rust 1.98.0 (pinned in `rust-toolchain.toml`).

## Quick start

### Rust

```rust
use flannrust::{ConstDim, KdTreeBuilder};

let pts: &[[f64; 3]] = &[
    [0.0, 0.0, 0.0],
    [10.0, 10.0, 10.0],
    [1.0, 1.0, 1.0],
];
let tree = KdTreeBuilder::new(ConstDim::<3>, pts).build();

let mut indices = [0u32; 2];
let mut dists = [0.0f64; 2];
let found = tree.knn_search(&[0.1, 0.1, 0.1], &mut indices, &mut dists);

assert_eq!(found, 2);
assert_eq!(indices[0], 0); // nearest point is [0.0, 0.0, 0.0]
```

### Python

```python
import numpy as np
import flannrust

pts = np.random.default_rng(0).uniform(-10, 10, size=(100_000, 3)).astype(np.float32)
tree = flannrust.KDTree(pts, leaf_size=10, metric="l2", threads=None)

q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
dists, idxs = tree.query(q, k=5)               # dists are SQUARED l2
r_idxs, r_dists = tree.query_radius(q, r=4.0)  # r is SQUARED too, strict `<`

dyn = flannrust.DynamicKDTree(dim=3, dtype="float32")
dyn.add_points(pts)
dyn.remove_point(0)                            # lazy tombstone
dists, idxs = dyn.query(q, k=5)
```

> **Distances and radii are SQUARED** for `l2`/`l2_simple` — unlike
> `scipy.spatial.cKDTree`. Square your radius before calling; expect squared
> values back (`l1` is an unsquared sum of absolute differences). This is
> the single most common mistake porting code from `cKDTree`.

## Performance

Six gated workloads vs. the vendored C++ oracle, ratio = Rust time / C++
time (lower is better for Rust). Latest idle-host re-measurement (M2.6,
3 sessions, statistical harness):

| Workload | Ratio (Rust / C++) |
|---|---|
| build 100k, dim 3, f32, sequential | 0.99–1.01 |
| knn, fixed dim 3, f32, k=10 | 1.01–1.04 |
| knn, runtime dim 8, f64, k=10 | 0.93–0.94 |
| radius, dim 3, f32 | 0.83–0.87 |
| dynamic add 20k, dim 3, f32 | 1.03–1.04 |
| dynamic knn after churn, dim 3, f32 | 0.93–0.96 |

Measured on WSL2, AMD Ryzen 7 9800X3D, `rustc 1.98.0`, `-C target-cpu=native`
vs. C++ `-O3 -march=native -ffp-contract=off`.

### Python bindings vs. SciPy / pynanoflann / scikit-learn

| Workload | flannrust | scipy cKDTree | pynanoflann | sklearn KDTree |
|---|---|---|---|---|
| build 100k (1 thread) | **9.07 ms** | 11.0 ms | 17.8 ms | 32.9 ms |
| build 1M (parallel) | **28.0 ms** | 134 ms | 242 ms | 458 ms |
| batched knn 200k (all cores) | **42.1 ms** | 66.8 ms | 56.6 ms | 626 ms |
| knn dim 8, f64 (1 worker) | **15.8 µs** | 25.4 µs | 15.7 µs | 56.6 µs |
| radius ~1000 hits | **32.0 µs** | 63.4 µs | 118 µs | 41.5 µs |

100 repetitions per workload, interleaved, idle host. Benchmark script:
[`crates/flannrust-py/python/bench/bench_py.py`](crates/flannrust-py/python/bench/bench_py.py).

Full methodology, history, honest residuals, and a portable repro kit:
[docs/benchmarks.md](docs/benchmarks.md),
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md),
[docs/benchkit.md](docs/benchkit.md).

## Documentation

- API docs: `cargo doc -p flannrust --open` (docs.rs after publish);
  Python docstrings on every class/method
- [docs/semantics.md](docs/semantics.md) — exact behavioral contracts,
  deliberate deviations from C++, input domain, feature flags, dynamic
  adaptor and Python API details
- [docs/testing.md](docs/testing.md) — how bit-exact parity is verified
  (cross-validation matrix, dynamic op-sequence suite, miri, canaries)
- [docs/benchmarks.md](docs/benchmarks.md) / [docs/benchkit.md](docs/benchkit.md)
  — the numbers and how to reproduce them on your hardware
- [docs/ROADMAP.md](docs/ROADMAP.md) — what's next (M3 incremental adaptor,
  M4 multithreaded wrapper, serialization)
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup; note that
  cross-validation against the C++ oracle needs a C++17 compiler
  (`cargo test --workspace`)

## How this was built

Every line of Rust, C++ FFI, and Python-binding code here was written by an
AI agent (Claude Code), directed and reviewed by Itzik Ben-Shabat.
Correctness does not rest on human code review — it rests on the bit-exact
cross-validation suite run against the real C++ library on every change,
and every performance figure traces to a pasted, reproducible run. The full
process record — plans, specs, and per-task reports:
[docs/agentic-development/](docs/agentic-development/).

## License

BSD-2-Clause — see [LICENSE](LICENSE). flannrust is a derivative work of
[nanoflann](https://github.com/jlblancoc/nanoflann) by Jose Luis
Blanco-Claraco et al., which builds on FLANN by Marius Muja and David G.
Lowe; the upstream copyright notices are retained in LICENSE. The vendored
`nanoflann.hpp` (used only as a test/benchmark oracle, not part of the Rust
library) keeps its original license header verbatim.

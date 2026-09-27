# Changelog

All notable changes to flannrust (the Rust crate and the Python package —
they version in lockstep) are documented here. Format:
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning:
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-29

Initial release.

- Static kd-tree (`KdTree`): bit-exact port of nanoflann 1.12.1's
  `KDTreeSingleIndexAdaptor` — knn / rknn / radius / box search.
- Dynamic Bentley–Saxe forest (`DynamicKdTree`): port of
  `KDTreeSingleIndexDynamicAdaptor` — point add/remove after construction.
- L1 / L2 / L2-simple / SO2 / SO3 metrics; `f32`/`f64`; compile-time or
  runtime dimension; `u32`/`u64`/`usize` indices; optional rayon-parallel
  build (feature `parallel`, on by default).
- Python bindings (PyPI package `flannrust`): `KDTree` / `DynamicKDTree`,
  NumPy in/out, GIL released during build/query. Distances and radii are
  SQUARED for l2 metrics (see README).

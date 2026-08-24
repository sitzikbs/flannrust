# Final review fix wave — report

Branch `m-py`, base HEAD `d3756ab`.

## Finding 1 (Important): "zero-copy NumPy build input" claim is false — shipped design is copy-in

Grepped `README.md` and `docs/*.md` for "zero-copy" / "zero copy": exactly two hits, both about the
Python *input* path (no "zero-copy OUT" claim existed to fix).

- `README.md:658` (M-py "Delivered" bullet, in the M2.5/roadmap section) — was:
  `zero-copy NumPy build input, GIL released during build/query,`
  Now:
  `copy-in NumPy build input (one copy at construction; zero-copy input deferred), GIL released during build/query,`
- `docs/ROADMAP.md:100` (M-py "Delivered" line) — same substitution:
  `zero-copy NumPy build input` → `copy-in NumPy build input (one copy at construction; zero-copy input deferred)`

No other "zero-copy" claims found anywhere in README.md/docs/. `crates/flannrust-py/src/convert.rs`'s
own doc comment already correctly describes `as_rows_2d` as copying element-by-element into a plain
`Vec` — consistent with the corrected wording.

## Finding 2 (Important): FFI panics — `threads=2**32` and `workers=2**40`

### (a) `parse_threads` truncation panic

`crates/flannrust-py/src/static_tree.rs:84-93`. Old code:
```rust
Some(n) if n > 1 => Ok(BuildThreads::Threads(NonZeroU32::new(n as u32).unwrap())),
```
`threads=2**32` (`i64`) truncates to `0u32` via `as u32`, then `NonZeroU32::new(0).unwrap()` panics
inside the PyO3 call — an unrecoverable Rust panic crossing the FFI boundary instead of a Python
exception.

Fix: reject `n > u32::MAX` before the cast, with a clean `ValueError`:
```rust
fn parse_threads(threads: Option<i64>) -> PyResult<BuildThreads> {
    match threads {
        None => Ok(BuildThreads::Auto),
        Some(1) => Ok(BuildThreads::Sequential),
        Some(n) if n > 1 && n <= i64::from(u32::MAX) => Ok(BuildThreads::Threads(NonZeroU32::new(n as u32).unwrap())),
        Some(n) => Err(PyValueError::new_err(format!(
            "threads must be None or a positive integer <= {} (got {n})",
            u32::MAX
        ))),
    }
}
```
`DynamicKDTree::new` has no `threads=` parameter at all (`crates/flannrust-py/src/dynamic_tree.rs:130-131`),
so this fix applies only to `static_tree.rs`.

### (b) `workers` unbounded thread-pool build

`workers=2**40` passes validation (`workers >= 1`), then hits
`rayon::ThreadPoolBuilder::new().num_threads(workers as usize).build().expect("thread pool build")` —
rayon spends minutes spawning OS threads before eventually panicking (`.expect`) across the FFI
boundary. This pattern was duplicated at all four call sites:
- `crates/flannrust-py/src/static_tree.rs:325` (`do_query`)
- `crates/flannrust-py/src/static_tree.rs:385` (`do_query_radius`)
- `crates/flannrust-py/src/dynamic_tree.rs:336` (`do_query`)
- `crates/flannrust-py/src/dynamic_tree.rs:394` (`do_query_radius`)

Root-cause fix: one shared helper in `crates/flannrust-py/src/convert.rs` (imported by both
`static_tree.rs` and `dynamic_tree.rs`, so it's the natural common module — no new file needed):
```rust
/// Caps a validated `workers > 1` count at the number of available CPUs
/// before it's handed to `rayon::ThreadPoolBuilder`. ...
pub fn capped_workers(workers: i64) -> usize {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    (workers as usize).min(cpus)
}
```
All four sites now call `rayon::ThreadPoolBuilder::new().num_threads(capped_workers(workers)).build().expect("thread pool build")`
— the cap runs before pool construction, so `.expect` is now building a pool sized `<= available_parallelism()`,
which never has to spawn an absurd thread count and will not hang/panic on that input.

### Tests added

`crates/flannrust-py/python/tests/test_behavior.py` (next to the existing `threads=`/`workers=`
determinism tests):
- `test_threads_overflow_raises_valueerror` — `flannrust.KDTree(pts, threads=2**32)` raises `ValueError`.
- `test_workers_huge_is_capped_and_matches_sequential` (parametrized `dtype`) — `tree.query(q, k=5,
  workers=2**40)` returns results identical to `workers=1`.

Isolated run, both new cases plus their dtype parametrizations:
```
python/tests/test_behavior.py::test_threads_overflow_raises_valueerror PASSED
python/tests/test_behavior.py::test_workers_huge_is_capped_and_matches_sequential[float32] PASSED
python/tests/test_behavior.py::test_workers_huge_is_capped_and_matches_sequential[float64] PASSED
3 passed, 39 deselected in 0.02s
```
0.02s confirms `workers=2**40` no longer hangs spawning threads.

## Finding 3 (Minor): xfail node dims misstated (radius vs tie-multiset)

Verified against `_RADIUS_XFAIL_NODES` (dim ∈ {8,32} only) and `_TIE_MULTISET_XFAIL_NODES` (dim ∈
{3,8,32}, including two `("clustered", 3, "float32", {1,64}, "l1")` nodes) in
`crates/flannrust-py/python/tests/test_parity_pynanoflann.py:203-225`.

- `README.md:597` — was "radius search and near-tie/duplicate data at dim ∈ {8,32}/float32
  specifically"; now "radius search at dim ∈ {8,32}/float32, and near-tie/duplicate data at
  dim ∈ {3,8,32}/float32".
- `docs/EXPERIMENTS.md:1137` — was "(10 in radius-index parity, 6 in the tie-multiset comparison,
  both dim ∈ {8,32}/float32)"; now "(10 in radius-index parity at dim ∈ {8,32}/float32, 6 in the
  tie-multiset comparison at dim ∈ {3,8,32}/float32)".
- `docs/benchmarks.md:654` — was "16 parametrized nodes (radius-index parity and tie-multiset
  comparison, both dim ∈ {8,32}/float32 ...)"; now "16 parametrized nodes (radius-index parity at
  dim ∈ {8,32}/float32, and tie-multiset comparison at dim ∈ {3,8,32}/float32 ...)".

## Verification (paste)

```
$ .venv/bin/maturin develop --release
   Compiling flannrust v0.1.0 (/home/sitzikbs/dev/flannrust/crates/flannrust)
   Compiling flannrust-py v0.1.0 (/home/sitzikbs/dev/flannrust/crates/flannrust-py)
    Finished `release` profile [optimized] target(s) in 9.08s
🛠 Installed flannrust-0.1.0

$ .venv/bin/pytest python/tests -q
........................................................................ [ 18%]
........................................................................ [ 37%]
........................................................................ [ 56%]
.........x...x...............................xxxxx.......x.x.x.......... [ 75%]
......x.x....x.x.....x.x.........................................xxxXxxx [ 94%]
xxxxx..............                                                      [100%]
351 passed, 27 xfailed, 1 xpassed in 3.94s
```
351 = baseline 348 + 3 new (1 unparametrized + 2 dtype-parametrized). 0 failed, 27 xfailed,
1 xpassed — matches baseline exactly.

```
$ cargo clippy --workspace --all-targets -- -D warnings
    Checking flannrust-py v0.1.0 (/home/sitzikbs/dev/flannrust/crates/flannrust-py)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.21s
```
No warnings.

```
$ cargo test --workspace
```
All crates: 0 failed across every test binary (flannrust unit tests 198 passed/4 ignored, xval_dynamic
11 passed/2 ignored, xval_knn 13 passed/1 ignored, xval_radius_box 12 passed/1 ignored, doc-tests 2
passed, plus the smaller flannrust-py/nanoflann_ref/xval unit suites — all "0 failed").

`git diff | grep unsafe` → no `unsafe` added anywhere.

## Commit

`fix: final-review fixes — copy-in claim, FFI-safe threads/workers validation, xfail dim docs`

Files touched: `README.md`, `docs/ROADMAP.md`, `docs/EXPERIMENTS.md`, `docs/benchmarks.md`,
`crates/flannrust-py/src/convert.rs`, `crates/flannrust-py/src/static_tree.rs`,
`crates/flannrust-py/src/dynamic_tree.rs`, `crates/flannrust-py/python/tests/test_behavior.py`.

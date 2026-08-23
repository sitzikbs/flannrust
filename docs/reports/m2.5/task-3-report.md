# M2.5 Task 3 — Dim-32 fix: bounds-check-free chunked row walk in the L2/L1 kernel

Branch `m2p5-perf`, on top of T1 (`9b2e85f chore(xval): M2.5 diagnostic probes`).
Productionizes T1's top-ranked candidate: `DataSource::point_row` + a
bounds-check-free chunked (`as_chunks::<4>()`) row walk in `L2::eval`, with
the identical lever also applied to `L1::eval` after an A/B confirmed it wins
there too.

Environment: WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2), AMD Ryzen 7
9800X3D (Zen 5, AVX-512), rustc 1.98.0, g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2).
`RUSTFLAGS="-C target-cpu=native"` + `--release` for every measurement below
(workspace profile `codegen-units=1`, `lto="thin"`), matching T1's
methodology. Every A/B is **interleaved** (before, after, before, after, ...)
using two prebuilt binaries (a "before" build from the stashed baseline and
an "after" build from the working tree) so no measurement pair straddles a
recompile, and every cited number is **median of 7 timed runs × ≥3 repeats**.

---

## 1. Implementation

### `data_source.rs` — `DataSource::point_row`

Added a defaulted trait method:

```rust
fn point_row(&self, _idx: usize) -> Option<&[T]> {
    None
}
```

Contract (documented on the trait): when `Some(row)` is returned, `row.len()
>= dim` and `row[d] == self.point_component(idx, d)` for all `d < dim`,
where `dim` is whatever dimensionality the caller is currently evaluating
with — the row is not required to be exactly `dim` long, only long enough.

Overridden for the two built-in sources:
- `FlatSlice::point_row`: `self.data.get(idx*self.dim .. idx*self.dim +
  self.dim)` — `None` if `idx` is out of range (defensive; `.get` never
  panics).
- `&[[T; N]]::point_row`: `Some(&self[idx][..])`.

Every custom `DataSource` implementation (including `FallbackOnlySource` used
by the tests) keeps the default `None` and is therefore byte-for-byte
unaffected — the M2 dynamic path's `GrowableFlat`/other adaptors that wrap a
`FlatSlice` internally get the fast path automatically wherever they forward
to it; adaptors that don't wrap contiguous storage keep working exactly as
before through `point_component`.

### `metric.rs` — the chunked row walk

Added `l2_eval_row<T: Scalar>(query, row, dim) -> T` and
`l1_eval_row<T: Scalar>(query, row, dim) -> T`, both generic (not
macro-duplicated per f32/f64 — monomorphized per call site, `#[inline]`).
Each computes the bit-identical summation to its per-component fallback:

- `l2_eval_row`: `query[..multof4]`/`row[..multof4]` walked via
  `as_chunks::<4>()`, accumulating `(d0*d0 + d1*d1) + (d2*d2 + d3*d3)` per
  chunk (T1's exact grouping), then the descending remainder `d+2, d+1, d+0`.
- `l1_eval_row`: identical structure, `abs_ternary!`'s ternary reproduced as
  a local `abs_t` helper (`if v < zero { zero - v } else { v }`; `zero - v`
  stands in for unary negation since `Scalar` has no `Neg` bound — bit-
  identical to `-v` here because that arm only fires when `v` is strictly
  negative, so there is no `+0.0`/`-0.0` sign ambiguity).

Both `L1::eval` and `L2::eval` gained one branch at the top, before the
existing loop, which is otherwise **completely untouched**:

```rust
let dim = dim.dim();
if let Some(row) = ds.point_row(idx) {
    if row.len() >= dim && query.len() >= dim {
        return l2_eval_row(query, row, dim);   // or l1_eval_row
    }
}
let mut result: $t = 0.0;
// ... unmodified fallback loop ...
```

`L2Simple` and `SO2`/`SO3` are untouched, per the brief (`L2Simple` mirrors
C++'s own plain loop; SO2/SO3 aren't part of this lever).

### L1 decision: measured in

The brief made L1 conditional ("if its A/B also wins — measure"). No gate or
diagnostic tool covers L1's kernel in isolation, so I wrote a temporary,
uncommitted example (`crates/xval/examples/t3_l1_probe.rs`, deleted after
use) that measures `L1.eval` through a `FlatSlice` (row path, live in the
current tree) against the same `FallbackOnlySource` wrapper the bit-equality
tests use (fallback path) — same binary, interleaved, no recompile between
arms, n=3 medians-of-7 each:

```
dim=32 row_ms=[0.546,0.536,0.537] -> med=0.537  fallback_ms=[0.930,0.924,0.925] -> med=0.925  speedup=1.721x
dim=64 row_ms=[1.095,1.080,1.073] -> med=1.080  fallback_ms=[1.857,1.853,1.850] -> med=1.853  speedup=1.715x
```

Decisive (~1.72×), same order of magnitude as L2's kernel-level win, so L1
was wired in permanently. This is a pure kernel-isolation number (no gate
exists to re-measure it end-to-end), so it's reported as a decision input,
not a landed gate.

---

## 2. TDD: RED/GREEN evidence

### Stage 1 — bit-equality test written first, trivially green

Added `FallbackOnlySource` (wraps a `FlatSlice`, forwards
`point_component`, never overrides `point_row` — forced through the
fallback loop) and four tests in `metric.rs`:
`l2_eval_row_path_bit_equals_fallback_path_all_dims_{f32,f64}`,
`l1_eval_row_path_bit_equals_fallback_path_all_dims_{f32,f64}`, each sweeping
dims `{1..=8, 15, 16, 17, 31, 32, 33, 64}` (all four 4-wide remainder
classes, both sides of the dim-16/32 boundaries) on deterministic
non-integer "random-ish" data (`sin`/`cos` of irrational-ish multipliers —
chosen so summation-order differences would show up as rounding
differences, unlike small exact integers). Written and run **before** the
`point_row` branch existed in `eval` — at that point `FlatSlice` already
overrode `point_row`, but `L2::eval`/`L1::eval` didn't call it yet, so both
arms of every test ran through the same single code path:

```
running 24 tests
test metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f32 ... ok
test metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f64 ... ok
test metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f32 ... ok
test metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f64 ... ok
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 159 filtered out; finished in 0.00s
```
Trivially green, as required — not yet a discriminator.

### Stage 2 — chunked walk lands, test becomes a real discriminator

After wiring the `if let Some(row) = ds.point_row(idx)` branch into
`L2::eval` (then `L1::eval`), the same four tests now exercise two genuinely
different code paths (chunked `as_chunks` row walk vs. the untouched
per-component loop) and still pass bit-identically at every dim:

```
running 24 tests
test metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f32 ... ok
test metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f64 ... ok
test metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f32 ... ok
test metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f64 ... ok
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 159 filtered out; finished in 0.00s
```

### Stage 3 — mandatory genuine RED capture

Temporarily broke `l2_eval_row`'s remainder order (descending `d+2, d+1,
d+0` → ascending `d, d+1, d+2`) — a single self-contained edit, fallback
loop untouched:

```rust
// RED-CAPTURE SABOTAGE (temporary)
if rem >= 1 { let diff = query[d] - row[d]; result = result + diff * diff; }
if rem >= 2 { let diff = query[d + 1] - row[d + 1]; result = result + diff * diff; }
if rem >= 3 { let diff = query[d + 2] - row[d + 2]; result = result + diff * diff; }
```

`cargo test -p nanoflann-rs metric::` — **RED**, both bit-equality tests fail
with 1-ULP mismatches at exactly the dims whose remainder class is 3 (the
only class where reordering three sequential adds changes rounding on
non-exact data):

```
---- metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f32 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:601:13:
assertion `left == right` failed: dim=31 row=1418074 fallback=1418074.1
  left: 1236081360
 right: 1236081361

---- metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f64 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:617:13:
assertion `left == right` failed: dim=3 row=175078.06636661338 fallback=175078.0663666134
  left: 4685255649392277520
 right: 4685255649392277521

test result: FAILED. 22 passed; 2 failed; 0 ignored; 0 measured; 159 filtered out; finished in 0.00s
```
(The unrelated `l2_matches_reference_various_dims_*` tests stayed green — they
use exact small integers, which have no rounding error to reorder, so this
was expected and not a discrepancy.)

Restored the descending order. Re-ran — **GREEN**:
```
test metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f32 ... ok
test metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f64 ... ok
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 159 filtered out; finished in 0.00s
```

### `point_row` contract tests

Added to `data_source.rs`: length/value checks for both `FlatSlice` and
`&[[T; N]]`, boundary-idx checks (last valid `idx`), an out-of-range check
(`FlatSlice::point_row` returns `None`, doesn't panic), and a check that a
hand-rolled `DataSource` which doesn't override `point_row` keeps returning
`None` (i.e. the default is really the fallback-forcing default).

---

## 3. Correctness verification

- `cargo test --workspace` (debug): **355 passed, 0 failed** (was 345 at the
  end of T1; +10 new tests: 4 bit-equality + 6 `point_row` contract). Full
  static + dynamic xval suites (bit-exact-vs-C++-reference comparisons)
  unchanged and green — this is the load-bearing "digest" evidence: every
  `xval_knn`/`xval_radius_box`/`xval_dynamic` test compares Rust output
  against the C++ oracle exactly, and all pass with the new kernel live.
- `RUSTFLAGS="-C target-cpu=native" cargo test --workspace --release --
  --ignored`: **all green**, including the 1M-point heavy build/xval tests
  (109.5s) and both mutation canaries (`mutation_canary_doctored_slot_order_
  breaks_per_slot_comparison`, `mutation_canary_skipped_remove_breaks_
  structure_parity`, `mutation_canary_tie_rule`, `mutation_canary_distance_
  perturbation` — all still correctly detect their injected mutations, i.e.
  the parity harness itself wasn't accidentally weakened).
- `cargo clippy --workspace --all-targets`: clean (fixed two self-inflicted
  lints along the way: `approx_constant` on my test data's salt/multiplier
  literals, `needless_range_loop` in the `point_row` contract tests).
- `RUSTDOCFLAGS="-D warnings" cargo doc -p nanoflann-rs --no-deps`: zero
  warnings.
- `cargo build -p nanoflann-rs --no-default-features`: builds clean.

---

## 4. A/B methodology and results

Two binaries built from the **same source tree state**, each **once**:
`before` = `git stash` (T3 diff removed, tree at T1's `9b2e85f`) then build;
`after` = `git stash pop` (T3 diff restored) then build. Runs alternate
`before, after, before, after, before, after` (3 full interleaved passes),
so no run pair straddles a recompile and drift affects both arms equally.
Every number below is the test/probe's own median-of-7.

### 4a. The six PERF_GATE gates (margin: ratio ≤ 1.25)

`PERF_GATE=1 <binary> --ignored --test-threads=1 --nocapture`, n=3 each side:

| Gate | before (3 runs) | before median | after (3 runs) | after median |
|---|---|---|---|---|
| `build_100k_dim3_f32_seq` | 0.989, 0.998, 0.987 | **0.989** | 0.974, 0.971, 0.971 | **0.971** |
| `dyn_add_20k_dim3_f32` | 1.210, 1.120, 1.122 | **1.122** | 1.137, 1.139, 1.141 | **1.139** |
| `dyn_knn_after_churn_dim3_f32` | 0.956, 0.965, 0.965 | **0.965** | 0.969, 0.971, 0.968 | **0.969** |
| `knn_dim3_f32_k10` (fixed3) | 1.030, 1.045, 1.039 | **1.039** | 1.065, 1.049, 1.054 | **1.054** |
| `knn_dyn_dim8_f64_k10` | 0.949, 0.944, 0.945 | **0.945** | 0.864, 0.868, 0.867 | **0.867** |
| `radius_dim3_f32` | 0.784, 0.779, 0.778 | **0.779** | 0.776, 0.784, 0.777 | **0.777** |

All six pass on both sides (`ratio <= 1.25`) both before and after. Per T1's
documented noise bound (`knn_fixed3` ranged 0.956–1.192 over 8 runs, median
1.064 — recorded in T1's report §5), the fixed-3 gate's move (1.039→1.054)
and the dyn-churn gate's move (0.965→0.969) are both **inside the noise
floor**, i.e. no regression. The dim-3 array-slice path (`arr3.as_slice()`
in the fixed3 gate) also now takes the row walk (`&[[T;N]]` overrides
`point_row`), so this is a real "did the new kernel regress dim-3" check,
not a no-op arm — and it didn't. `knn_dyn_dim8_f64_k10` improved decisively
(0.945→0.867), consistent with T1's prototype prediction (0.947→0.893).

### 4b. Dim-32/dim-64, gate-style real tree (`m25_diag knn`, leaf=10, n=100k, 200 queries, k=10)

n=3 each side, interleaved:

| dim | scalar | before (3 runs) | before median | after (3 runs) | after median |
|---|---|---|---|---|---|
| 32 | f32 | 1.423, 1.420, 1.428 | **1.423** | 0.966, 0.947, 0.972 | **0.966** |
| 32 | f64 | 1.315, 1.315, 1.317 | **1.315** | 1.119, 1.105, 1.113 | **1.113** |
| 64 | f32 | 1.917, 1.923, 1.918 | **1.918** | 1.228, 1.234, 1.228 | **1.228** |
| 64 | f64 | 1.220, 1.213, 1.208 | **1.213** | 0.877, 0.874, 0.874 | **0.874** |

**Dim-32 f32: 1.423 → 0.966** (matches T1's prototype prediction of
~1.48→~0.99 to within the two prototypes' different random seeds/session
noise). Dim-64 f32 also improved substantially (1.918→1.228) though it
doesn't cross 1.0 — consistent with T1's diagnosis that dim-64's residual
gap is larger to start with (T1 measured 1.944→1.142 for its local
prototype; this run's 1.918→1.228 is the same shape, decisively improved
but not fully closed, matching T1's honest framing of dim-64 as "improved,
not eliminated").

### 4c. Dim-32/dim-64, pure-kernel isolation (`m25_diag leafn`, single-leaf tree, n=100k, 200 queries, k=10)

n=3 each side, interleaved (corroborating measurement, not gated):

| dim | scalar | before median | after median |
|---|---|---|---|
| 32 | f32 | **1.973** | **1.128** |
| 32 | f64 | **1.551** | **0.952** |
| 64 | f32 | **2.272** | **1.257** |
| 64 | f64 | **1.620** | **1.017** |

Same shape as T1's diagnosis: with traversal removed entirely (single leaf =
pure linear kernel sweep), the Rust deficit before the fix was *larger* than
the end-to-end dim-32 gap (1.973 vs 1.423) — confirming the kernel is the
entire gap — and after the fix it lands close to parity (1.128, 0.952 f64).

---

## 5. Self-review

- **Fallback path byte-for-byte unchanged**: confirmed by diff — the only
  edit to the existing per-component loops in both `L1::eval` and
  `L2::eval` is the new `if let Some(row) = ...` block inserted *before*
  them; not a single line inside the old loops changed.
- **Public API surface**: `DataSource::point_row` is a new defaulted trait
  method — source-compatible for every existing implementor (including any
  external/custom `DataSource`s, which silently keep the `None` default and
  are therefore functionally and performance-identical to before this
  change).
- **Generic helpers vs macro duplication**: `l1_eval_row`/`l2_eval_row` are
  written once each, generic over `T: Scalar`, rather than duplicated inside
  the `impl_l1!`/`impl_l2!` per-type macros — monomorphization still
  produces a concrete f32/f64 body per call site (verified indirectly: the
  measured wins match T1's asm-level analysis of why per-type specialization
  matters, and the bit-equality tests exercise both f32 and f64 through this
  same generic code).
- **`L2Simple`/`SO2`/`SO3` untouched**: confirmed by diff — zero lines
  changed in those three impls.
- **L1's decision was measured, not assumed**: the brief made it
  conditional; I did not skip the measurement or default to "leave it out"
  out of caution — a same-binary, no-recompile-needed A/B (row path via live
  `FlatSlice` vs. fallback path via the same `FallbackOnlySource` the unit
  tests use) gave a decisive, low-noise 1.72× kernel speedup at both dim 32
  and dim 64, so it was wired in permanently and covered by the same
  bit-equality test family as L2.
- **Concern**: dim-64 remains above the gate margin's comfort zone in
  isolation (1.228 gate-style, 1.257 kernel-only) though nothing in the
  current 6 gates tests dim-64 directly, so it isn't a blocker for this
  task. This matches T1's own honest framing — the row walk closes dim-32
  to parity and *substantially* shrinks dim-64 without fully closing it;
  further dim-64-specific work (if ever prioritized) is out of scope for T3.
  I did not chase it further given the brief's explicit target was dim-32.
  Documented here, not silently dropped.
- **Noise caveat**: every side-measurement in this report used n≥3 (never
  n=1), per the T1-review rigor note this brief explicitly called out.

---

## 6. Files changed

- `/home/sitzikbs/dev/flannrust/crates/nanoflann-rs/src/data_source.rs` —
  `DataSource::point_row` default method + doc; `FlatSlice`/`&[[T;N]]`
  overrides; 6 new contract tests.
- `/home/sitzikbs/dev/flannrust/crates/nanoflann-rs/src/metric.rs` —
  `l1_eval_row`/`l2_eval_row` generic helpers; one new branch each at the
  top of `L1::eval`/`L2::eval`; `FallbackOnlySource` test helper; 4 new
  bit-equality tests.

No other files changed. `crates/xval/examples/t3_l1_probe.rs` was created
temporarily to decide the L1 question and deleted before commit (not part of
the diff) — its numbers are quoted in §1 above for provenance.

## 7. Commit

`perf: bounds-check-free chunked row walk in L2/L1 kernel (dim-32: 1.42x -> 0.97x)`

---

## 8. Fix round 1 (review response)

Review verified the kernel itself line-by-line (summation order, guards,
`abs` semantics, reassociation-freedom) and found it correct, but found the
test guard critically weakened by a post-RED-capture edit, plus two
coverage gaps. Addressed below, in the order raised.

### 8.1 Critical: RED evidence invalid + guard insensitive

**What happened:** the original RED capture in §2 was genuine *at the time
it was run*, but the clippy `approx_constant` fix (changing the test data's
salt literals) landed **after** that capture and **before** the commit — so
the pasted RED transcript describes deleted data, not the committed data.
Independently, review reimplemented the generators and proved that against
the actually-committed single salt, `l2_..._f64` had **zero** sensitivity to
the ascending-remainder mutation, and `L1` f64 was blind to left-associative
chunk-grouping, at every swept dim. A single fixed salt is not a reliable
discriminator — it can land on values where a given reassociation happens
not to move the rounded bit pattern, and apparently did.

**Fix:** every bit-equality test now sweeps **every dim x every one of 6
salts** (`SALTS_F32`/`SALTS_F64` in `metric.rs`), for both `L1` and `L2`,
both scalar types. Verified by running **both** sabotages independently and
confirming each of the four tests fails **on its own** for **each**
mutation (8 checks total: 4 tests x 2 mutations) — genuine RED captured
against the exact code now committed:

**Sabotage 1 — ascending remainder order** (`d, d+1, d+2` instead of the
required descending `d+2, d+1, d+0`), applied to both `l1_eval_row` and
`l2_eval_row`'s remainder tails:

```
failures:

---- metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f32 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:730:17:
assertion `left == right` failed: dim=3 salt=5.7731 row=426.05698 fallback=426.05695
  left: 1138034507
 right: 1138034506

---- metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f64 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:748:17:
assertion `left == right` failed: dim=3 salt=-4.2214 row=402.50832636550115 fallback=402.5083263655011
  left: 4645788617553459290
 right: 4645788617553459289

---- metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f32 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:694:17:
assertion `left == right` failed: dim=3 salt=-4.2214 row=54666.434 fallback=54666.438
  left: 1196788335
 right: 1196788336

---- metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f64 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:712:17:
assertion `left == right` failed: dim=3 salt=0.4173 row=186546.66906305557 fallback=186546.6690630556
  left: 4685649707580373813
 right: 4685649707580373814

---- metric::tests::l2_eval_row_path_bit_equals_fallback_path_constdim3_array_f32 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:771:13:
assertion `left == right` failed: salt=-4.2214 row=54666.434 fallback=54666.438

---- metric::tests::l2_eval_row_path_bit_equals_fallback_path_constdim3_array_f64 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:788:13:
assertion `left == right` failed: salt=0.4173 row=186546.66906305557 fallback=186546.6690630556

failures:
    metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f32
    metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f64
    metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f32
    metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f64
    metric::tests::l2_eval_row_path_bit_equals_fallback_path_constdim3_array_f32
    metric::tests::l2_eval_row_path_bit_equals_fallback_path_constdim3_array_f64

test result: FAILED. 21 passed; 6 failed; 0 ignored; 0 measured; 159 filtered out; finished in 0.00s
```
All 4 core tests fail independently (plus both new ConstDim3-array tests,
which also exercise the remainder path at dim 3).

Reverted, confirmed GREEN (27/27), then applied **Sabotage 2 — fully
left-associative chunk grouping** (`((d0[²]+d1[²])+d2[²])+d3[²]` instead of
the required `(d0[²]+d1[²])+(d2[²]+d3[²])`), applied to both `l1_eval_row`
and `l2_eval_row`'s main chunked loop:

```
failures:

---- metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f32 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:730:17:
assertion `left == right` failed: dim=4 salt=-1.9021 row=935.83527 fallback=935.8352
  left: 1147794805
 right: 1147794804

---- metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f64 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:748:17:
assertion `left == right` failed: dim=4 salt=0.4173 row=942.7740646341315 fallback=942.7740646341316
  left: 4651503944190428334
 right: 4651503944190428335

---- metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f32 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:694:17:
assertion `left == right` failed: dim=4 salt=3.5588 row=150733.8 fallback=150733.78
  left: 1209217907
 right: 1209217906

---- metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f64 stdout ----
thread '...' panicked at crates/nanoflann-rs/src/metric.rs:712:17:
assertion `left == right` failed: dim=4 salt=3.5588 row=150733.77824485858 fallback=150733.77824485855
  left: 4684419186021658815
 right: 4684419186021658814

failures:
    metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f32
    metric::tests::l1_eval_row_path_bit_equals_fallback_path_all_dims_f64
    metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f32
    metric::tests::l2_eval_row_path_bit_equals_fallback_path_all_dims_f64

test result: FAILED. 23 passed; 4 failed; 0 ignored; 0 measured; 159 filtered out; finished in 0.00s
```
All 4 core tests fail independently again (the ConstDim3-array tests pass
under this mutation, correctly — dim 3 never enters the chunked main loop,
only the remainder tail, so they were never expected to catch a
chunk-grouping bug; they're covered by Sabotage 1 instead).

Reverted, confirmed GREEN (27/27; `git diff` shows no `SABOTAGE` code
remaining, only the explanatory doc comment). Both mutations now make all
four load-bearing tests fail independently, against the exact salts/dims
that are committed.

### 8.2 Important: dynamic path had zero row-path coverage

Confirmed the review's finding by reading `crates/xval/src/lib.rs`:
`GrowableFlat` does **not** wrap `FlatSlice` (it holds its own `data: &'a
[T]` + `dim` directly) and its `DataSource<T> for &GrowableFlat<'a, T>` impl
never overrode `point_row` — so every dynamic-path test (`xval_dynamic.rs`,
the mutation canaries) and both `PERF_GATE` dyn gates were exercising only
the fallback loop, and my original report's claim that "the M2 dynamic path
... benefits automatically" was **wrong** for `GrowableFlat` specifically
(true only insofar as `GrowableFlat` forwards to nothing that already had
the fast path — it forwards to nothing at all; it's a leaf implementation).
Retracted; corrected here.

**Fix:** added `point_row` to `&GrowableFlat`'s impl, mirroring
`FlatSlice`'s exactly (`self.data.get(idx*self.dim..idx*self.dim+self.dim)`)
— deliberately **not** bounds-checked against `current_n`, for the same
reason `point_component` isn't: consistency. `point_component` already
indexes `self.data` directly regardless of logical size, so a row for `idx
>= current_n()` behaves exactly as inconsistently/consistently (i.e.
identically) as a component read at the same `idx` already did — no new
behavior was introduced at the `current_n` boundary, only a new, faster
path to the same values. `cargo test -p xval` confirmed the full dynamic
suite (including both mutation canaries) stays green with the row path live.

**Dyn gates re-measured** (they are now a real arm — before this fix the
two dyn gates ran byte-identical code before/after the original T3 diff, so
their originally-reported movement was provably noise, per review). n=3
interleaved, "before" = `fecbdde` (this task's original commit, fallback
`GrowableFlat`), "after" = this fix round (row-path `GrowableFlat`):

| Gate | before (3 runs) | before median | after (3 runs) | after median |
|---|---|---|---|---|
| `dyn_add_20k_dim3_f32` | 1.146, 1.142, 1.150 | **1.146** | 1.123, 1.112, 1.131 | **1.123** |
| `dyn_knn_after_churn_dim3_f32` | 0.980, 0.974, 0.989 | **0.980** | 0.969, 0.980, 0.974 | **0.974** |

Both pass comfortably (`<=1.25`) on both sides. The movement is small (as
expected at dim 3 — T1's noise floor for dim-3 gates is documented at
0.956–1.192) but now attributable to a genuine code-path change rather than
being provable noise from an unchanged binary.

### 8.3 Important: trust-boundary `debug_assert`

Added `debug_check_point_row_contract` (new private fn in `metric.rs`),
called at both row-path branch sites (`L1::eval`, `L2::eval`) right before
the fast-path return: an O(1) spot check of `row[0]` and `row[dim-1]`
against `ds.point_component(idx, 0)`/`ds.point_component(idx, dim-1)`,
`debug_assert!`-gated (compiles to nothing in `--release`). Confirmed it
fires zero false positives across the full workspace test suite (debug
build, where `debug_assert!` is active) — 355→391 passing tests including
the now-row-path-covered dynamic suite, all green, so every in-tree
`point_row` implementation (`FlatSlice`, `&[[T;N]]`, `&GrowableFlat`) is
contract-conformant by this spot check.

### 8.4 Minors

(a) **`docs/benchmarks.md`** — added two "Update (M2.5-T3)" paragraphs (near
the M1-T14 `point_row`-reverted claim at the former lines 158-162, and near
the dim-32 "plausibly needs SIMD" claim at the former line 140) reconciling
both with what actually shipped: the lever is the bounds-check-free
*chunked* walk (not a bare row pointer, which really was measured as a net
loss for dim 3/8 — that part of the M1 record stands), and dim-32 closed to
~0.97x with zero SIMD, using this session's measured numbers (1.423x →
0.966x).

(b) **`&[[T;3]]` + `ConstDim<3>` bit-equality test** — added
`l2_eval_row_path_bit_equals_fallback_path_constdim3_array_{f32,f64}`
(§8.1 above), covering the exact gate combination
(`perf_gate_knn_dim3_f32_k10` builds over `arr3.as_slice(): &[[f32;3]]`
with `ConstDim<3>`), independent from the `DynDim`/`FlatSlice` sweep.

(c) **OOB-behavior doc on `point_row`** — added a paragraph to the trait
doc in `data_source.rs` stating that behavior for `idx >= point_count()` is
deliberately unspecified (an implementation may return `None`, as
`FlatSlice` does via `slice::get`, or `Some` of whatever the backing buffer
holds, as `&GrowableFlat` does) — either is conforming, since callers are
never expected to pass an out-of-range `idx` in the first place (same
precondition `point_component` already has).

(d) **`-0.0` abs test** — added
`l1_accum_dist_negative_zero_diff_stays_negative_zero`: `L1.accum_dist(-0.0,
0.0, 0)` must return `-0.0` (bit-exact), locking in `abs_ternary!`'s
documented ternary-vs-`.abs()` distinction (the ternary's `v < 0.0` is
`false` for `-0.0`, so it takes the `else` branch and returns `v`
unchanged — `.abs()` would clear the sign bit instead).

(e) **`Scalar::Default`-must-be-zero doc** — strengthened the existing
(informational-only) doc line in `scalar.rs` into an explicit contract
statement, naming exactly where it's relied upon
(`l1_eval_row`/`l2_eval_row`'s `T::default()` zero-accumulator).

### 8.5 Re-verification after all fixes

- `cargo test --workspace`: **358 passed, 0 failed** (was 355 before this
  round; +3 new test *functions* — 2 ConstDim3-array tests + 1 `-0.0` test.
  The 4 core bit-equality tests gained a x6-salt inner loop each, which
  multiplies assertions run, not test-function count, so it doesn't show up
  in this delta).
- `RUSTFLAGS="-C target-cpu=native" cargo test --workspace --release --
  --ignored`: green, including all four mutation canaries and the 1M-point
  heavy build/xval tests (108.8s).
- `cargo clippy --workspace --all-targets`: clean.
- `RUSTDOCFLAGS="-D warnings" cargo doc -p nanoflann-rs --no-deps`: zero
  warnings.
- `cargo build -p nanoflann-rs --no-default-features`: builds clean.

### 8.6 Files changed (fix round 1)

- `crates/nanoflann-rs/src/metric.rs` — `debug_check_point_row_contract`;
  `debug_check_point_row_contract` calls at both row-path branches;
  `FallbackOnlySource` → generic `FallbackOnly<DS>`; multi-salt sweep
  (`SALTS_F32`/`SALTS_F64`) in all four bit-equality tests; 2 new
  ConstDim3-array tests; 1 new `-0.0` test.
- `crates/nanoflann-rs/src/data_source.rs` — OOB-behavior paragraph on
  `point_row`'s doc.
- `crates/nanoflann-rs/src/scalar.rs` — strengthened `Default`-is-zero doc
  into an explicit contract statement.
- `crates/xval/src/lib.rs` — `point_row` override on `&GrowableFlat`.
- `docs/benchmarks.md` — two reconciling "Update (M2.5-T3)" paragraphs.

### 8.7 Commit

`fix: multi-salt kernel guards, dynamic row-path coverage, contract debug-assert`

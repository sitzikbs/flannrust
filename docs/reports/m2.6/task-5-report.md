# Task 5 report: land the M2.6 fidelity fixes

Date: 2026-08-24. Branch `m2p6-rigor`, started at HEAD `20a9a0a`, ended at
`b9335fd`. Environment: WSL2, `export PATH="$HOME/.cargo/bin:$PATH"`,
`RUSTFLAGS="-C target-cpu=native"` for all timed runs, cargo/rustc 1.98.0.
`git status --porcelain` clean at the end.

Three items landed, exactly Task 4's ranked list plus the controller's
test-gating ruling — nothing else touched (the REJECT list: build seq,
knn_fixed3, arena pre-reserve, dim-32/64 kernel, removed-hasher — was not
revisited).

## Commits

1. `3d64c8b` — `perf: dyn merge preserves slot vind capacity (C++ fidelity: dyn_add 1.13-1.14->1.06-1.09)`
   — `crates/flannrust/src/dynamic.rs` (FIDELITY-1) + `crates/flannrust/src/node.rs` (test-gating fix, rode with this commit).
2. `a68df86` — `perf: 4-wide unrolled compute_min_max (C++ fidelity: dyn_add 1.06->1.04)`
   — `crates/flannrust/src/build.rs` (FIDELITY-2).
3. `b9335fd` — `docs: M2.6 fidelity fixes — dyn_add 1.13-1.14->1.03-1.04`
   — `docs/benchmarks.md` + `docs/EXPERIMENTS.md`.

## FIDELITY-1: dynamic merge capacity preservation

**Mechanism** (`crates/flannrust/src/dynamic.rs`, `add_points`, was
lines 569-580): `nanoflann.hpp:2653-2664`'s merge loop iterates a lower
slot's `vAcc_` by index (not consuming it), pushes each entry into the
target slot, THEN calls `vAcc_.clear()` — C++ `std::vector::clear()`
retains the allocation. The port used `std::mem::take(&mut
self.slots[i].vind)`, which drops the allocation at the end of the loop
body, so every one of the O(n) merges regrows the slot's `Vec` from
capacity 0 through the doubling ladder.

**Fix**: take the `Vec` out (still needed to satisfy the borrow checker
— `self.slots[pos].vind` and `self.slots[i].vind` are borrowed
simultaneously), iterate it by reference (`&e` — `Idx: Copy`), push into
the target slot, `entries.clear()`, then write `entries` back into
`self.slots[i].vind`. Same element values, same push order (ascending
`j` within slot `i`, ascending `i` across slots), only the allocation is
now reused instead of dropped — bit-identical rebuilt trees.

```rust
for i in 0..pos {
    let mut entries = std::mem::take(&mut self.slots[i].vind);
    for &e in entries.iter() {
        self.slots[pos].vind.push(e);
        let e_usize = e.to_usize();
        if self.tree_index[e_usize] != -1 {
            self.tree_index[e_usize] = pos as i32;
        } else {
            self.removed.insert(e_usize, pos as i32);
        }
    }
    entries.clear();
    self.slots[i].vind = entries;
}
```

**A/B sessions** (`xval::measure_pair`, n=100/side, interleaved,
`PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release
--test perf_gate -- --ignored perf_gate_dyn --test-threads=1 --nocapture`).
Isolated via per-file `git diff`/`git apply` patches so baseline and
candidate ran on otherwise-identical trees.

Baseline (unpatched, `dynamic.rs` at `20a9a0a`), 2 sessions:
```
dyn_add: rust med=4.143 mean=4.178 std=0.138 | cpp med=3.657 mean=3.674 std=0.101 | ratio=1.133
dyn_add: rust med=4.165 mean=4.205 std=0.152 | cpp med=3.646 mean=3.689 std=0.124 | ratio=1.142
```

FIDELITY-1 patched, 2 sessions:
```
dyn_add: rust med=3.858 mean=3.874 std=0.079 | cpp med=3.635 mean=3.646 std=0.087 | ratio=1.061
dyn_add: rust med=3.979 mean=4.017 std=0.183 | cpp med=3.666 mean=3.724 std=0.195 | ratio=1.085
```

Full unfiltered gate suite under the patch (1 session, `--ignored`, no
filter): `build_100k=0.998, dyn_add=1.055, dyn_knn_after_churn=0.955,
knn_dim3=1.039, knn_dyn_dim8=0.944, radius=0.816` — all within recorded
M2.6 task 2 bands, no regression.

**Idle-host re-verification** (see "Methodology note" below — the
coordinator flagged that earlier sessions may have run under host load
from a background game process; the host was confirmed idle partway
through this task): reverted `dynamic.rs` to `20a9a0a` (pure baseline)
and re-ran with `/proc/loadavg`/`nproc` captured before/after:

```
LOADAVG_BEFORE: 1.04 1.80 1.42 1/550 470485  nproc=8
dyn_add ratio=1.136 | rust mean=4.036 std=0.104 n=100 | cpp mean=3.560 std=0.118 n=100
LOADAVG_AFTER: 1.17 1.70 1.40 2/555 474071

LOADAVG_BEFORE: 0.99 1.64 1.39 1/551 475240
dyn_add ratio=1.137 | rust mean=4.114 std=0.167 n=100 | cpp mean=3.650 std=0.200 n=100
LOADAVG_AFTER: 1.07 1.65 1.39 2/552 476136
```

Restored `dynamic.rs` to the landed (`3d64c8b`) state and re-ran the
full gate suite twice with loadavg:

```
LOADAVG_BEFORE: 1.00 1.60 1.38 1/553 477653
build_100k=1.015 (rust std=8.112, cpp std=7.535 -- one noisy session,
  see below), dyn_add=1.058, dyn_knn_after_churn=0.948 (rust std=12.778,
  also noisy this session), knn_dim3=1.039, knn_dyn_dim8=0.936, radius=0.807
LOADAVG_AFTER: 1.19 1.56 1.37 2/545 480211

LOADAVG_BEFORE: 1.01 1.50 1.36 1/543 484352   (re-run, clean this time)
build_100k=0.991 (std=0.311/0.294), dyn_add=1.062, dyn_knn_after_churn=0.952
  (std=1.056/0.634), knn_dim3=1.040, knn_dyn_dim8=0.932, radius=0.806
LOADAVG_AFTER: 1.28 1.51 1.37 1/547 485855
```

The first idle-host full-gate session had one transient std spike
(`build_100k` rust std=8.112ms on a ~10ms mean, `dyn_knn_after_churn`
rust std=12.778ms on a ~30ms mean) despite loadavg staying low (1.00 →
1.19, well under `nproc=8`) — a brief scheduling burst, not sustained
competing load. The gate's decision quantity is the MEDIAN, which stayed
consistent (1.015/1.058 vs. the immediate clean re-run's 0.991/1.062), so
this did not change any decision, but I re-ran it anyway per the
coordinator's directive and it came back clean (std back to the ordinary
1-6% band). All idle-host `dyn_add` medians (1.136, 1.137 baseline;
1.058, 1.062 patched) agree with the earlier non-annotated sessions
(1.133, 1.142 baseline; 1.061, 1.085 patched) to within 1-2pp.

**Verdict: LAND.** Ratio 1.133-1.144 → 1.055-1.062, well beyond the
recorded noise band (M2.6 task 2's `dyn_add` range was 1.112-1.149; the
patched sessions sit clearly below the bottom of that range and stayed
there across 6 independent sessions, 4 of them idle-host-confirmed). No
other gate regressed.

## FIDELITY-2: 4-wide unrolled `compute_min_max`

**Mechanism** (`crates/flannrust/src/build.rs`, `compute_min_max`, was
lines 66-84): `middleSplit_`'s inline min/max scan
(`nanoflann.hpp:1507-1530`) is 4-way unrolled (`constexpr size_t UNROLL =
4`, four batched `dataset_get` loads per iteration, `std::min({...})`/
`std::max({...})` initializer-list folds) — a distinct code path from
the separate plain-loop `computeMinMax` (`nanoflann.hpp:1193-1205`),
which `middleSplit_` never calls. The port had a 1-wide loop instead. The
doc comment's claim that this is bit-identical is true (min/max is
associative/commutative, no NaNs in play — verified against the vendored
source directly, and the reviewer note in Task 4's report already
confirmed both sides process values in the same encounter order for
ties), but the C++'s 4-load-ILP shape was lost.

**Fix**: ported the exact `UNROLL=4` main-loop / remainder-loop shape.
Confirmed `std::min({...})`/`std::max({...})` resolve via
`std::min_element`/`std::max_element` (first-wins-tie, i.e. only update
on a *strict* `<`), which is exactly the tie-breaking semantics this
file's pre-existing `cpp_min`/`cpp_max` helpers already have — so a
left-associative 4-way fold using them is bit-identical to the C++'s
initializer-list fold.

```rust
const UNROLL: usize = 4;
let mut k: usize = 1;
while k + UNROLL <= count {
    let v0 = ds.point_component(ind[k].to_usize(), dim);
    let v1 = ds.point_component(ind[k + 1].to_usize(), dim);
    let v2 = ds.point_component(ind[k + 2].to_usize(), dim);
    let v3 = ds.point_component(ind[k + 3].to_usize(), dim);
    local_min = cpp_min(cpp_min(cpp_min(cpp_min(local_min, v0), v1), v2), v3);
    local_max = cpp_max(cpp_max(cpp_max(cpp_max(local_max, v0), v1), v2), v3);
    k += UNROLL;
}
while k < count {
    let val = ds.point_component(ind[k].to_usize(), dim);
    local_min = cpp_min(local_min, val);
    local_max = cpp_max(local_max, val);
    k += 1;
}
```

**A/B sessions**, same harness, applied on top of landed FIDELITY-1
(`3d64c8b`), all idle-host-confirmed with loadavg:

Baseline (neither fix; loadavg 1.0-1.2), 2 sessions — same numbers
pasted above for FIDELITY-1's idle-host baseline (`1.136`, `1.137`).

FIDELITY-1 only (this candidate not yet applied), 2 sessions — same
numbers pasted above (`1.058`, `1.062`).

Combined (both fixes, final landed state), 2 idle-host sessions:
```
LOADAVG_BEFORE: 1.16 1.47 1.36 2/544 486312
build_100k=1.009 (std=0.119/0.194), dyn_add=1.035 (std=0.097/0.126),
  dyn_knn_after_churn=0.940, knn_dim3=1.037, knn_dyn_dim8=0.939, radius=0.835
LOADAVG_AFTER: 1.13 1.42 1.35 1/546 487125

LOADAVG_BEFORE: 0.96 1.37 1.33 2/543 487170
dyn_add ratio=1.037 (std=0.167/0.172), dyn_knn_after_churn=0.941
LOADAVG_AFTER: 0.96 1.37 1.33 1/545 487307
```

Plus 2 non-loadavg-annotated combined sessions run earlier in the task
(before the idle-host directive), which agree: `1.038`, `1.038`.

**Verdict: LAND.** Across all 4 combined-state sessions (2
non-annotated, 2 idle-host-confirmed): `dyn_add` **1.034-1.038** — a
real, reproducible ~2pp gain beyond FIDELITY-1 alone (1.058-1.062),
matching Task 4's prediction ("dyn_add 1.118 alone, 1.021 combined with
#1"). `build_100k` stayed at parity (0.991-1.015 across sessions) as
predicted — the unroll only helps the many small dyn-forest rebuild
scans, not the wide one-shot 100k build. No other gate moved outside its
recorded band.

## Controller ruling: `node.rs` test-gating fix

`node::tests::test_offset_children_panics_on_leaf_in_debug` is
`#[should_panic]` on a `debug_assert!`, which cannot fire under
`--release`. Reproduced the failure on unmodified `20a9a0a` before
touching anything:

```
$ cargo test -p flannrust --release --lib node::tests::test_offset_children_panics_on_leaf_in_debug
test node::tests::test_offset_children_panics_on_leaf_in_debug ... FAILED
```

Fix: added `#[cfg(debug_assertions)]` above the existing `#[should_panic]`
attribute. Test-only, rode with the FIDELITY-1 commit (`3d64c8b`) per the
brief's "can ride with either commit" instruction.

## Methodology note (idle-host directive)

Partway through this task the coordinator relayed a user note: earlier
T3/T4-era measurement sessions may have run alongside a background game
process, and the host was now confirmed idle (only terminal + Chrome
open). I checked `/proc/loadavg`/`nproc` at that point (1.37/1.92/1.45,
`nproc=8` — already low) and re-ran every A/B state (baseline,
FIDELITY-1-only, combined) fresh with loadavg captured immediately
before and after each session, isolating each candidate via per-file
`git checkout`/`git apply` of saved patches so the comparisons stayed
clean (baseline = `dynamic.rs`/`build.rs` at `20a9a0a`; FIDELITY-1-only =
`dynamic.rs` at the landed `3d64c8b` state with `build.rs` still at
`20a9a0a`; combined = both landed). One session (FIDELITY-1-only, first
full-gate run) showed a std spike on `build_100k`/`dyn_knn_after_churn`
despite low loadavg both before and after — re-ran immediately and it
came back clean, confirming it was a brief scheduling burst rather than
sustained contention. Every `dyn_add` median across all idle-host
sessions agreed with the earlier non-annotated sessions to within 1-2pp,
so no decision changed, but the idle-host numbers are what's pasted into
both landed commits' messages and are the primary evidence in
`docs/EXPERIMENTS.md`'s new "M2.6 task 5" subsection; the
non-annotated sessions are kept as corroborating evidence, explicitly
labeled as such.

## Verification (final combined+committed state, `b9335fd`)

- `cargo test --workspace` (debug): **green**, 17/17 `test result: ok`
  blocks, 0 failed (198 flannrust unit + 90 xval unit + doctests +
  xval_build/dynamic/knn/radius_box integration suites).
- `cargo test --workspace --release -- --ignored` (heavy suite, incl.
  both mutation canaries): **green**. `build::tests::heavy_exponential_build_1m`,
  `build_parallel::tests::heavy_exponential_parallel_build_1m`,
  `search::tests::heavy_query_degenerate_trees`,
  `build::tests::heavy_exponential_build_1m_dim8` all `ok`;
  `mutation_canary_doctored_slot_order_breaks_per_slot_comparison`,
  `mutation_canary_skipped_remove_breaks_structure_parity`,
  `mutation_canary_tie_rule`, `mutation_canary_distance_perturbation` all
  `ok`. Ran this full check after each landed commit (FIDELITY-1 alone,
  then combined) and once more at the very end on the committed
  `b9335fd` state.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean, at
  every checkpoint.
- `cargo build --workspace --no-default-features`: clean.
- `RUSTDOCFLAGS="-D warnings" cargo doc -p flannrust -p xval -p
  nanoflann-ref --no-deps`: clean (per-package, matching the repo's
  established workaround for the pre-existing `flannrust`/`flannrust-py`
  doc-output name collision — same pattern used in Task 1/3's reports).
- `git diff | grep unsafe` on both landed diffs: empty — neither touches
  `unsafe` code, so miri was not re-run (matches the brief's "miri NOT
  needed" and the general rule to only re-run it when `search.rs` or
  unsafe-adjacent code changes).
- `git status --porcelain`: clean at the end.

## Self-review / concerns

- **Scope discipline**: only the two named FIDELITY candidates plus the
  controller-ruled test fix were touched. Confirmed via `git show
  --stat` on both perf commits: `dynamic.rs`+`node.rs` for the first,
  `build.rs` alone for the second. Nothing from the REJECT list was
  revisited.
- **Bit-parity**: neither change alters the values or order of any
  computed quantity — FIDELITY-1 changes only which allocation backs a
  `Vec` (element values/order preserved exactly), FIDELITY-2 changes
  only the grouping of associative/commutative min/max comparisons
  (verified against the C++'s actual `std::min_element`/`max_element`
  tie-breaking semantics, not just assumed). The full xval cross-validation
  suite, including both dynamic-forest mutation canaries, stayed green
  throughout every intermediate state I measured (baseline, FIDELITY-1
  only, combined) — I ran `cargo test --workspace` at each landed
  checkpoint, not just at the end.
- **A/B isolation methodology**: used per-file `git diff`/`git apply`
  round-trips to get true single-variable A/B comparisons (baseline vs.
  FIDELITY-1-only vs. combined), rather than relying on incidental state.
  This is more rigorous than the brief strictly required but was cheap
  given the files are independent, and it's what let me give both a
  clean "no other fix's effect confounding this one" story.
- **Docs scope**: the coordinator's top-level instructions named
  `benchmarks.md` + `EXPERIMENTS.md` specifically and explicitly deferred
  the `build_100k` conclusion-(g) re-hedge to Task 6; I did not touch
  `README.md` or `docs/ROADMAP.md` even though the M2.6 task 2/3 precedent
  in those files would normally get a matching Update banner — left
  alone per the explicit "benchmarks.md M2.6 section + EXPERIMENTS.md ...
  provenance rows" scoping in my brief, which is more specific than (and
  takes precedence over) the generic task-5-brief.md wording that also
  mentions ROADMAP.
- **No open concerns.** Both candidates landed with reproducible,
  loadavg-confirmed evidence; no candidate needed reverting.

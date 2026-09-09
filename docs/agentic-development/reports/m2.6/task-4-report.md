# Task 4 report: M2.6 fidelity audit of the losing workloads (diagnosis only)

Date: 2026-08-24. Branch `m2p6-rigor`, HEAD `20a9a0a`. No library code changed;
no probe code committed (every measurement used the existing harness —
`tests/perf_gate.rs`, `examples/report_data.rs`, `examples/m25_diag.rs`,
`examples/m25_asm.rs` — plus uncommitted local A/B patches, each measured and
reverted, plus standalone C++ benches in the session scratchpad). `git status
--porcelain` clean at the end. Environment: WSL2, `export
PATH="$HOME/.cargo/bin:$PATH"`, `RUSTFLAGS="-C target-cpu=native"`, cargo/rustc
1.98.0, g++ 13.3.0. All gate numbers are `xval::measure_pair` medians (2
discarded warmups, n=100 interleaved reps/side unless a `std` note says the
session was contaminated).

User directive honored throughout: **"no extra tricks, follow flann exactly but
see if we missed something in the rust implementation of it."** Every ranked
candidate below is a divergence from what the vendored
`crates/nanoflann-ref/cpp/nanoflann.hpp` (1.12.1) actually does; anything else
found is in the "not eligible" list.

---

## 1. Fresh baseline (this session)

```
PERF_GATE=1 RUSTFLAGS="-C target-cpu=native" cargo test -p xval --release --test perf_gate -- --ignored --test-threads=1 --nocapture perf_gate_build_100k perf_gate_knn_dim3 perf_gate_dyn_add
PERF_GATE perf_gate_build_100k_dim3_f32_seq: rust=10.003ms cpp=10.347ms ratio=0.967 | rust mean=10.139 std=0.377 n=100 | cpp mean=10.461 std=0.524 n=100
PERF_GATE perf_gate_dyn_add_20k_dim3_f32:    rust=4.277ms  cpp=3.740ms  ratio=1.144 | (means outlier-inflated, medians robust)
PERF_GATE perf_gate_knn_dim3_f32_k10:        rust=7.965ms  cpp=7.700ms  ratio=1.034
```

Reproduces T2's gate cluster exactly (0.967 / 1.141-1.149 / 1.030-1.032).

---

## 2. The crossing experiment (build_100k) — answer to conclusion (g)

Four cells, all n=100/side. Arms B/C are one-line uncommitted seed-tag patches
(`"perf_gate_build"` ↔ `"report_build_100k"`), reverted after each run.

| cell (harness + dataset seed) | ratio | rust med | cpp med |
|---|---|---|---|
| gate harness + gate seed (baseline above) | **0.967** | 10.003 | 10.347 |
| gate harness + report seed (patched `perf_gate.rs`) | **0.976** | 9.904 | 10.144 |
| report harness + gate seed (patched `report_data.rs`) | **0.984** | 9.823 | 9.981 |
| report harness + report seed (unpatched control, today) | **0.9825** | 10.013 | 10.191 |

Plus one more same-session unpatched gate control captured during the A/B work:
`ratio=0.944 (rust=9.727 cpp=10.300, std 0.230/0.198)`.

**Findings:**

1. **Dataset seed is NOT the cause** — swapping datasets moves the ratio by
   ≤0.017, both directions, all cells stay in the "Rust wins" cluster.
2. **The T2 "report cluster" (1.037/1.039) does not reproduce at all today**:
   the unpatched report_data control — identical code, identical seeds — gives
   0.9825/0.984 (two runs). The T2 split (conclusion (g), §3 "M2.6 task 2" of
   `docs/EXPERIMENTS.md`) is therefore **CONTRADICTED as a configuration
   property**: it was a property of that day's session/binary state (T2's
   binaries were built from an `a30a819-dirty` tree; code layout and session
   conditions differ), not of harness or dataset. See §7.
3. **build_100k seq is NOT losing.** Today's five unpatched measurements:
   0.944, 0.967, 0.976, 0.9825, 0.984 — Rust won every cell of the crossing
   matrix. Deliverable (4) answer: **no real gap exists; Task 5 should skip
   build_100k.**

`build_1M_dim3_f32_seq` (from today's two full report_data runs, n=100):
ratio **1.0124** (131.701/130.082) and **1.0056** (129.867/129.148) — 0.6-1.2%,
inside the same session envelope that moves build_100k by ±4pp. No real 1M gap
either; the T2 1.0446/1.0469 values look like the same session-state effect.

---

## 3. knn_fixed3 residual — is there a real gap? (deliverable 4)

Today: gate harness **1.034**; report harness **1.0569 / 1.062** (two runs).
Combined with T2's four sessions (1.030/1.030/1.061/1.067), every n=100 session
lands in 1.03-1.07 — consistently above 1.0, but entirely inside the canonical
noise envelope 0.94-1.16, and the gate-vs-report-harness sub-split (~1.03 vs
~1.06) again tracks process context, not code.

**Line-by-line audit result: no unported C++ behavior remains.**
`search_level` (search.rs:440-577) matches `searchLevel` (nanoflann.hpp:
1244-1307) branch-for-branch: same `(diff1+diff2) < 0` child choice, same
live-in-loop `worst_dist()` read, same `mindist + cut_dist - dst` update, same
restore semantics on all three exit paths. The only structural difference is
the M2.5 explicit-stack conversion itself (`FrameStack`, 32-byte frames, spill
check per push) — a **recorded, accepted deviation** (C++ uses native
recursion, which the port cannot, by the stack-safety design decision), whose
cost was already measured in M2.5 (1.052→1.021 on conversion; radius/dim8-f64
give-backs documented). Post-M2.5 residue checked: node layout (20B packed vs
C++ 48B alignas(16)) favors Rust; leaf loop, prune gate, eps widening all
identical.

**Verdict: no fidelity divergence exists to fix; the 3-7% residual is
process-context + iterative-stack overhead, sub-noise. REJECT for Task 5.**

---

## 4. dyn_add_20k — two real fidelity divergences found (the audit's win)

This is the one target with a consistent, real gap (today 1.130-1.144
unpatched; T2 1.112-1.149). Line-by-line comparison of
`DynamicKdTree::add_points` (dynamic.rs:529-609) against `addPoints`
(nanoflann.hpp:2629-2675) + the slot rebuild path found two places where C++
does X and we do Y:

### 4a. Merge loop drops slot `vind` allocations; C++ keeps them (FIDELITY, top candidate)

C++ (nanoflann.hpp:2653-2665): merging slot `i` into slot `pos` iterates
`index_[i].vAcc_` then calls `index_[i].vAcc_.clear()` — **capacity is
retained**, so in steady state every merge pushes into already-sized buffers.
Rust (dynamic.rs:570): `let entries = std::mem::take(&mut self.slots[i].vind)`
— the allocation is dropped at the end of the iteration, so every slot regrows
from capacity 0 through the full doubling ladder on every one of the O(n)
merges.

A/B (uncommitted patch: iterate `&entries`, then `entries.clear(); 
self.slots[i].vind = entries;` — value-and-order identical, only the allocation
is reused, exactly C++'s `.clear()` semantics):

```
unpatched controls:  ratio=1.144 (4.277/3.740), ratio=1.130 (4.054/3.586)
capacity-preserving: ratio=1.055 (rust=3.877 cpp=3.675, std 0.153/0.179)
```

**Predicted gain: dyn_add ~1.13-1.14 → ~1.05-1.06 (≈8pp). Parity risk: none**
(identical values, identical push order, bit-identical slot vind → bit-identical
rebuilt trees; Task 5 must still run the full xval_dynamic suite as proof).

### 4b. `middleSplit_`'s 4-wide unrolled min/max scan was flattened in the port (FIDELITY)

Vendored 1.12.1 `middleSplit_` (nanoflann.hpp:1510-1530) scans candidate
dimensions with an explicit `UNROLL=4` loop (`std::min({local_min,v0,v1,v2,v3})`
over 4 batched `dataset_get` loads). Our `compute_min_max` (build.rs:66-84)
deliberately flattened it to a 1-wide loop (the doc comment argues bit-identical
results — true — but the C++ shape has 4-way load ILP the flat loop lacks).
Notably, nanoflann 1.5.5 still had the plain loop; upstream added the unroll as
a perf change, and the port undid it.

A/B (uncommitted patch: exact 1.12.1 shape, `cpp_min`/`cpp_max` left-fold =
`std::min` initializer-list fold, bit-identical min/max):

```
unrolled alone: dyn_add ratio=1.118 (4.127/3.692);  build_100k ratio=0.966 (unchanged)
```

**Predicted gain: dyn_add ~1-2pp (rebuild-dominated workload, many small
`middle_split` scans); build_100k/1M ≈ 0 (dim-3 scan too narrow to matter).
The dyn_add gain is near the workload's session spread — flagged as such.
Parity risk: none** (min/max is order-insensitive on non-NaN data; outputs
bit-identical; `cargo test -p flannrust` under the patch: all pass except
`node::tests::test_offset_children_panics_on_leaf_in_debug`, which fails in
`--release` **unpatched too** — a pre-existing debug_assert/should_panic
release-profile quirk, unrelated; in the default debug profile the workspace is
fully green).

### Combined A/B (4a + 4b)

```
dyn_add ratio=1.021 (rust=4.061 cpp=3.976); build_100k ratio=1.001 (noisy session, std 5.8/4.5)
```

**Combined prediction for Task 5: dyn_add lands ≈1.02-1.06 — from a
1.13-1.14 baseline, closing most of the only real gap this audit confirmed.**

### dyn_add items checked and cleared (we already match or already beat C++)

- **Slot node arena**: C++ `freeIndex` → `pool_.free_all()` frees every block
  back to malloc on every rebuild; our `slot.nodes.clear()` retains capacity —
  the Rust side is already *better* than C++ here. Match-or-better; no action.
- **`removed` HashMap (SipHash vs `std::unordered_map` identity hash)**: the
  gate workload has zero removals; std's hashbrown `get` short-circuits an
  empty table before hashing, so it costs nothing here. Only relevant to
  tombstone-heavy mixes, and the tombstone-heavy gate
  (`dyn_knn_after_churn`) is a Rust win (0.937-0.940 today). REJECT for now.
- `tree_index` resize-on-demand, `first0bit`, reactivation short-circuit,
  merge append order, rebuild schedule `0..=max_index`: verbatim matches.

---

## 5. Allocation-pattern audit (build) — brief's questions answered

- **C++ PooledAllocator**: BLOCKSIZE is **8192 bytes** (+16B link word),
  nanoflann.hpp:904 — not the 260KB the task brief states; nodes are 48-byte
  alignas(16) bump-allocated, never individually freed, never moved.
- **Our arena**: `Vec<Node<T>>` (20B/node f32), doubling growth ≈13-17
  reallocs-with-memcpy per 100k build. Different pattern, but measurably not a
  loss: build_100k is a Rust win in every unpatched cell (§2).
- **A/B: pre-reserving the arena** (`Vec::with_capacity(4*n/leaf+4)` in
  `build_sequential_core`) was measured twice and was **worse both times**
  (patched 1.036 and 1.025 vs paired unpatched 0.944-0.984; both patched
  sessions were noise-contaminated — std 5.8-10.2 — but the rust-side median
  moved the same direction both times, +3-6%). A single large cold
  `with_capacity` (fresh pages) loses to doubling growth over warm allocator
  blocks, and it is *not* what C++ does anyway (8KB pooled blocks). **REJECT.**
- **`vAcc_`/`vind` width + iota**: both sides u32 (wrapper instantiates
  `index_t = uint32_t`); both sides allocate + iota-fill per build
  (`init_vind`, nanoflann.hpp:2174-2180 = build.rs:27-33). Match.
- **Build-loop dataset access**: per-component on BOTH sides (`dataset_get`
  = `point_component`); build was never row-pointered in C++ either. The
  min/max scan cannot autovectorize on either side (gather through `vind`);
  the C++ advantage is 4-load ILP, which is exactly candidate 4b. Match
  otherwise (computeBoundingBox, planeSplit — swap-for-swap identical, leaf
  tight-bbox loops same k-outer/d-inner shape).

---

## 6. dim-32 f64 (~1.05-1.10) and dim-64 — compiler codegen, not fidelity

- **Kernel isolation** (`m25_diag kernel`, fresh session): the library kernel
  (`lib_dyndim`, row path) is the fastest or near-fastest Rust access-path
  variant at every dim/scalar — e.g. dim32 f64 seq 6.47 ns/pt vs
  hand_component 8.77, hand_row 8.10; dim64 f64 perm 46.3 vs hand variants
  57.3/59.2 (only `hand_row_chunks` at 40.2 edges it). The M2.5 row-path fix
  is intact; no plumbing overhead remains worth chasing.
- **Single-leaf kernel sweep** (`m25_diag leafn`, fresh): dim32 f64 1.053,
  dim64 f64 1.036 — the same few-percent band as the knn-level numbers, i.e.
  the residual is pure kernel throughput, not traversal.
- **Asm** (`m25_asm` --emit asm vs `g++ -O3 -march=native -ffp-contract=off -S`
  on the vendored header, scratchpad `cpp_kernel.cpp`): the Rust
  `probe_lib_dyndim32_f64` body uses **ymm-width AVX2 only** (6×vmulpd,
  5×vhaddpd, no zmm), while g++ 13 emits **AVX-512 zmm** code for the same
  `L2_Adaptor<double>::evalMetric` (30×vpermt2pd, 8×vmulpd over zmm). Same
  arithmetic (bit-parity holds — parity suites green), different vector width:
  LLVM's conservative prefer-256 default on AVX-512 hardware vs GCC's 512-bit
  choice.

**Verdict: REJECT as a Rust-code fidelity item — the kernel already mirrors
1.12.1 exactly; the residual is compiler codegen.** (A `RUSTFLAGS`-level
512-bit experiment is listed under "not eligible" — it is a toolchain knob,
not a port-fidelity change.)

---

## 7. Python dim8-f64 vs pynanoflann (T3's 1.233-1.271) — classified

pynanoflann vendors nanoflann **1.5.5** (`NANOFLANN_VERSION 0x155`,
`docs/agentic-development/reports/m-py/task-3-report.md:86`). Two checks:

1. **Source diff** (fetched v1.5.5 upstream): 1.5.5's `L2_Adaptor::evalMetric`
   uses pointer-walk + *sequential-chain* summation
   (`result += d0²+d1²+d2²+d3²`) and carries a `worst_dist` early-exit
   parameter — but 1.5.5's `searchLevel` **never passes it** (verified: the
   call is `evalMetric(vec, accessor, dim)`), so no early-abandon advantage
   exists. 1.5.5's `middleSplit_` min/max scan is the *plain* loop (the 4-wide
   unroll is a 1.12.1-era addition — corroborating candidate 4b).
2. **Head-to-head measurement** (scratchpad `bench_ver.cpp`, same g++, same
   LCG data, dim8 f64 n=100k, 10k queries, k=10, leaf=10, median of 15,
   identical result digests):

```
v1.12.1 -ffp-contract=off: knn10k median = 204.685 ms
v1.12.1 default (FMA on):  knn10k median = 196.904 ms
v1.5.5  -ffp-contract=off: knn10k median = 216.767 ms
v1.5.5  default (FMA on):  knn10k median = 208.027 ms   (all digest=80071)
```

**1.5.5 is ~5-6% SLOWER than 1.12.1 on this exact workload** — the "1.5.5
kernel autovectorizes faster" hypothesis is refuted. Since the same workload
vs our own 1.12.1 oracle is a Rust *win* (0.9291-0.9612 across T2 + today's
runs), the Python-side miss vs pynanoflann cannot be a Rust kernel/library
defect: it lives in the binding layer (marshalling/batch-loop overhead in
flannrust-py vs pynanoflann's pybind11 path) — an M-py investigation item,
out of this milestone's library scope. Side observation: FMA contraction buys
C++ ~4% here; our oracle correctly forgoes it for bit-parity (`-ffp-contract=off`
is load-bearing), so the *shipped* comparison baseline is fair to Rust.

---

## 8. Ranked candidate list for Task 5

1. **FIDELITY — dynamic merge capacity preservation** (dynamic.rs
   `add_points` merge loop: iterate borrowed entries, `clear()` + restore the
   allocation, = C++ `vAcc_.clear()` semantics, nanoflann.hpp:2664).
   Measured A/B: dyn_add **1.13-1.14 → 1.055**. Predicted gain ≈8pp on the
   only real losing gate. Parity risk: **none** (bit-identical vind order).
2. **FIDELITY — 4-wide unrolled `compute_min_max`** (build.rs, = vendored
   `middleSplit_` UNROLL=4 scan, nanoflann.hpp:1510-1530). Measured A/B:
   dyn_add 1.118 alone, **1.021 combined with #1**; build ≈ no change.
   Predicted gain ≈1-3pp on dyn_add (inside that workload's 1.11-1.15 session
   band — honest caveat), ≈0 elsewhere. Parity risk: **none** (min/max fold,
   bit-identical; unit tests green under patch).
3. **REJECT — build_100k / build_1M seq**: no real gap (Rust won all five
   unpatched cells today, 0.944-0.984; 1M 1.006-1.012). T2's split was
   session state (§2, §9).
4. **REJECT — knn_fixed3 residual**: no unported behavior; residual 1.03-1.07
   is process-context + the recorded M2.5 iterative-stack trade-off, inside
   the 0.94-1.16 envelope.
5. **REJECT — arena pre-reserve** (measured worse twice) and any other
   allocation-pattern change to the static build.
6. **REJECT — dim-32 f64 / dim-64 kernel**: exact C++ arithmetic already;
   residual is LLVM-vs-GCC vector-width codegen (asm evidence §6).
7. **REJECT — `removed` map hasher**: unmeasurable on the gate workloads
   (empty map short-circuits; tombstone gate already a Rust win).

### Not eligible (would deviate from the C++ 1.12.1 algorithm, or is not a library change)

- Forcing 512-bit vectors via RUSTFLAGS/`-C llvm-args` (toolchain knob;
  controller may choose to experiment at the harness level, but it is not a
  port-fidelity code change).
- Adopting 1.5.5's kernel shape or a `worst_dist` early-abandon (deviates
  from vendored 1.12.1 AND breaks bit-parity).
- Allowing FMA contraction on either side (breaks the bit-parity contract;
  `-ffp-contract=off` is load-bearing by design).
- Shrinking `search_level`'s 32-byte frames / hoisting spill checks (no C++
  counterpart exists — C++ recurses natively; the explicit stack is itself the
  recorded M2.5 deviation, already accepted).

---

## 9. Contradictions with recorded conclusions (deliverable 5)

1. **T2 conclusion (g)** (`docs/EXPERIMENTS.md` §3 "M2.6 task 2"; also the
   "M2.6 build_100k configuration-split finding" row of the §7 claims table):
   "reproducibly split by configuration (0.967 gate vs 1.037-1.039 report)" —
   **contradicted**. The crossing experiment it explicitly requested was run:
   dataset seed moves the ratio ≤0.017; the report-harness cluster itself does
   not reproduce (0.9825/0.984 today, unpatched, identical code+seeds). The
   split was a property of that day's session/binary state, not of
   configuration. Docs should re-hedge that row when next touched (doc changes
   are outside this task's file scope).
2. **build_1M seq ~1.045 (T2)**: today 1.006/1.012 — the recorded value is
   the top of a session-dependent band, not a stable gap.
3. No other recorded conclusion was contradicted; dyn_add's 1.11-1.15 band,
   knn_fixed3's 1.03-1.07, dim-32/64 and the radius/dim8-f64 wins all
   reproduced within their recorded ranges.

---

## 10. Verification

- `cargo test --workspace` (default debug profile): **green** — all suites
  `0 failed` (198+90+23+19+13+12+12+11+5+3+2+2 passed, heavy/gate tests
  ignored as designed).
- Known pre-existing quirk (NOT introduced here, fails on unpatched code too,
  release profile only): `node::tests::test_offset_children_panics_on_leaf_in_debug`
  fails under `cargo test -p flannrust --release --lib` because its
  `debug_assert!` cannot fire in release. Flagged for the controller; out of
  this task's scope to fix.
- `git status --porcelain`: clean after every A/B (verified after each revert
  and at end). No commits made by this task.

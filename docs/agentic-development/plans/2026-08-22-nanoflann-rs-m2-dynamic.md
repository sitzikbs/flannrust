# nanoflann-rs — Milestone 2: Dynamic Adaptor (Bentley–Saxe forest)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Same pipeline as M1: fresh implementer per task, review gate per task, fix loops, final whole-branch review.

**Goal:** Port `KDTreeSingleIndexDynamicAdaptor` (nanoflann 1.12.1) — a logarithmic forest of static kd-trees supporting `add_points` / `remove_point` — with bit-exact cross-validation against the C++ oracle, reusing M1's builder and search core unchanged.

**Spec authority:** vendored `crates/nanoflann-ref/cpp/nanoflann.hpp` (forest: 2521-2718; sub-tree `_` class: 2248-2519) + `docs/nanoflann-notes.md` §"Dynamic adaptor (M2) essentials" + the M1 plan's Corrections items 4/6/7 (forest quirks). Source wins over this plan's prose.

**Success criteria (same shape as M1 §5b):**
1. Parity: xval dynamic suite green bit-exact — after every batch of a seeded add/remove op sequence, knn/radius results and per-sub-tree `vAcc_` permutations equal the C++ forest's.
2. Speed: perf gates (≤1.25×) for add-heavy, churn (add+remove), and query-after-churn workloads; goal ≤1.0.
3. Robustness: op-sequence property tests incl. re-add of removed points, remove-then-merge tombstone migration, full-drain-and-refill.
4. Hygiene: zero warnings, `--no-default-features` builds, doctests.

## Verified upstream mechanics (from M1's source verification — implementers re-verify against the header)

- Ctor: `maximumPointCount` default `1000000000U`; `treeCount_ = log2(max)+1`; auto `addPoints(0, n-1)` on pre-existing data. Forest slots via `First0Bit(pointCount_)` (2565-2574).
- `addPoints` merge into slot `pos`: for each moved point `e`: `if (treeIndex_[e] != -1) treeIndex_[e] = pos; else removedPoints_[e] = pos;` (tombstones migrate in the ELSE branch — 2654-2668); re-adds short-circuit to reactivate tombstones in place (2640-2650); after moving, `freeIndex` + rebuild each affected slot (2671-2675).
- `removePoint`: `removedPoints_[idx] = treeIndex_[idx]; treeIndex_[idx] = -1` (2678-2685).
- Deleted-point skip: CRTP `isActive(idx)` = `treeIndex_[idx] != -1` (2288), consulted in `searchLevel` — maps to M1's `PointFilter` seam at zero static-path cost.
- The forest has NO `knnSearch`/`radiusSearch` — only `findNeighbors` (loops all sub-trees into ONE result set) + `addPoints`/`removePoint`/`getAllIndices`. It performs NO emptiness checks: radius/box on an empty forest return `true` (M1 Corrections #4).
- Sub-trees are the STATIC algorithm over index subsets → M1's `SubtreeBuilder`/`SearchCtx` are reused as-is; a sub-tree's `vAcc_` is its slot's point list, so per-slot vind equality is the build-parity check.

## Design

- `DynamicKdTree<T, D, DS, M = L2, Idx = u32, TB = KeepInsertionOrder>` in new `crates/nanoflann-rs/src/dynamic.rs`:
  - fields: `slots: Vec<Slot>` (each: `vind: Vec<Idx>`, `nodes: Vec<Node<T>>`, `root_bbox`, non-empty flag), `tree_index: Vec<i32>` (-1 = removed/absent), `removed: HashMap<usize, i32>`, `point_count`, `tree_count`, shared `dim/metric/leaf_max_size/dataset`.
  - `TombstoneFilter<'a>` implementing `PointFilter` over `tree_index` — plugs into the M1 search core.
  - API: `DynamicKdTreeBuilder` (mirrors M1 builder + `maximum_point_count`), `add_points(start, end_inclusive)` (C++ signature parity; an additive range-exclusive helper allowed if documented), `remove_point(idx) -> bool`, `find_neighbors<R>` (parity, incl. the empty-forest `full()` quirk — DOCUMENTED, matched exactly), plus ADDITIVE ergonomic `knn_search`/`radius_search` wrappers (marked as extensions; forest-looped internally; absent upstream).
  - `point_indices_of_slot(i)` for xval vAcc parity; `active_count()`, `size()` mapping documented.
- C++ oracle: extend wrapper with `nfrd_*` handle (build with max_point_count, add_points, remove_point, knn via KNNResultSet+findNeighbors loop — mirroring the C++ class's own usage, radius same, per-slot vAcc export, treeIndex export for tombstone cross-checks).
- Parallel build of slots: NOT in M2 scope (C++ rebuilds slots sequentially inside addPoints; keep parity; note as M2.5 candidate).

## Tasks

- **T0** Branch/setup: branch `m2-dynamic` (from M1 head or post-merge main), commit this plan, extend ledger workspace.
- **T1** Oracle extension (`nfrd_*` + safe `RefDynIndexF32/F64` + unit tests incl. hand-computed add/remove/query sequence).
- **T2** `dynamic.rs` core: slots, First0Bit, add_points with merge/tombstone/reactivation rules, remove_point; unit tests for every bookkeeping rule (tombstone migrates on merge; re-add reactivates in place; treeIndex bookkeeping; slot rebuild determinism = M1 sequential builder ⇒ per-slot vind reproducible).
- **T3** Search: TombstoneFilter + find_neighbors forest loop (+ additive wrappers); unit tests: removed points never returned; results equal a filtered brute force; empty-forest quirk matched + documented.
- **T4** xval dynamic suite: seeded op-sequence generator (interleaved adds/removes/re-adds, incl. adversarial: remove-all-then-refill, duplicate coords); after each batch compare vs C++: knn/radius bit-exact positional (ties per M1 comparator), per-slot vind equality, treeIndex/tombstone map equality; mutation canary (skip one remove on the Rust side → suite must fail).
- **T5** Benches + gates: `dyn_add_100k` (batched adds from empty), `dyn_churn` (add/remove mix), `dyn_knn_after_churn`; gate ≤1.25; report_data extension rows (accuracy vs brute force over live points at eps 0/0.1; speed medians).
- **T5b** Report automation (user directive — the suite generates the report): `crates/xval/examples/render_report.rs` (or a checked-in script) that renders the scorecard HTML from `report_data.json` — same visual design as the M1 scorecard, committed as a template; one command chain (`report_data | render_report`) reproduces the report end-to-end. `report_data` meta extended to capture CPU model (/proc/cpuinfo), kernel (`uname -sr`), C++ compiler version, and git SHA automatically.
- **T6** Docs: README dynamic section (API, the empty-forest quirk, size semantics, extension-vs-parity table), notes-file M2 outcome.
- **T7** `docs/EXPERIMENTS.md` (user directive — publication-grade experimental setup): hardware/OS/toolchain/flags/seeds/run-counts for every number in the report, the exact reproduction commands, the WSL2 noise caveat and the bare-metal re-run requirement before any public claim (see docs/ROADMAP.md M-pub).

Each task: TDD with captured RED, brief+report files in the M2 SDD workspace, review gate, fix loops, ledger — exactly as M1.

# nanoflann-rs — Milestone 2.5: Performance deep-dive

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Same pipeline as M1/M2.

**Goal:** Close the two known performance gaps — fixed-dim-3 kNN (~4–6% behind C++, asm-attributed to recursive search-call overhead) and dim-32 kNN (~1.3–1.45×, root cause NOT yet isolated) — without giving up an inch of the bit-exact parity story on the default build.

**Binding constraints (from M1/M2, non-negotiable):**
1. The DEFAULT build's results stay bit-exact vs the C++ oracle: the full xval suite (static + dynamic) must pass unchanged after every landed change. Any optimization that reorders floating-point arithmetic goes behind a NON-default cargo feature with its own accuracy characterization vs brute-force GT (documented error profile), and the parity suites always run against the default path.
2. Measure-first discipline: no code change without a baseline, an A/B (perf-gate methodology, median×7, PERF_GATE=1, --test-threads=1), and a revert-with-numbers if it loses. WSL2 noise ±3–4%; decisions need margins beyond that or repeated runs.
3. All existing perf gates keep passing; the suite-generated report + benchmarks.md/EXPERIMENTS.md are updated through the established provenance rules (every number traces to a pasted run).

**Success criteria:**
1. `knn_fixed3` ratio ≤ 1.00 (or a deeper evidence-backed analysis of why the residual is irreducible without breaking parity).
2. dim-32 knn ratio ≤ 1.10 on the default build (or root cause isolated + documented, with the remaining gap's fix identified even if feature-gated).
3. No regression: all M1+M2 gates pass; headline workloads within noise of their recorded ranges.
4. Hygiene: workspace tests green, clippy clean, zero rustdoc warnings, docs updated via the suite.

## Tasks

- **T1 — Diagnosis (no code changes).** Isolate both gaps with evidence before touching anything:
  - dim-32: microbench the L2 eval kernel in isolation (Rust vs C++ through the FFI at dim 32 over the same buffer) to split "kernel is slower" from "traversal/layout is slower"; inspect asm of the monomorphized DynDim dim-32 eval (vectorization? bounds checks? `point_component` per-element indexing vs C++'s restrict pointer walk); measure leaf-visit counts (identical trees ⇒ identical visits — confirm, then it's pure per-visit cost); test the hypothesis that per-component `FlatSlice` indexing defeats vectorization while C++'s flat pointer autovectorizes.
  - fixed-3: quantify the recursion overhead (count search_level calls per query; estimate call cost from the asm evidence in benchmarks.md; measure an inline-always experiment as a probe).
  - Deliverable: written diagnosis in the task report with ranked, evidence-backed candidate fixes and predicted gains. Perf tooling: `perf` may be unavailable under WSL2 — use targeted microbenches + asm + counting instrumentation instead.
- **T2 — Fixed-dim-3 residual.** Per T1's ranking; leading candidate: convert `search_level` to an explicit-stack iterative form that preserves EXACT traversal order and dists save/restore semantics (bit-parity constraint — the xval suite is the judge), eliminating call/ABI overhead. A/B on knn_fixed3 + all knn gates; land only if it wins without regressing dim-8/radius.
- **T3 — Dim-32 gap.** Per T1's diagnosis. Leading hypotheses: (a) row-pointer access path for the eval kernel (get the whole point row once, walk a slice — LLVM can then autovectorize like C++'s restrict walk; SAME summation order, so bit-parity holds — note the M1-T14 `point_row` attempt lost at dim 3/8 but was never evaluated at dim 32 where the per-component cost dominates); (b) prefetching the next leaf point row; (c) if and only if diagnosis shows the unroll itself can't vectorize under order constraints: a `fast-math` NON-default feature with reordered kernels + documented accuracy profile (never default, never used in parity suites).
- **T4 — Regression sweep + docs.** Re-run ALL gates (M1 + dynamic) ×2 with pasted outputs; update benchmarks.md (new M2.5 section: before/after per workload, honest ranges), EXPERIMENTS.md provenance rows, regenerate the report chain; ROADMAP updated.

Stretch (only if T2+T3 land early and cleanly): parallel slot rebuilds for the dynamic forest (documented deviation; determinism per-slot preserved).

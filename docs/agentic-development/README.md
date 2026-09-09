# Agentic development record

This codebase was implemented by an AI agent (Claude Code), directed and
reviewed by Itzik Ben-Shabat, who does not write Rust. Every line of Rust,
C++ FFI, and Python-binding code here is agent-written; the author's role
was specifying requirements, reviewing the generated code and each task's
report, and directing dedicated rigor/fidelity-audit passes (the plans and
reports below). Because the author cannot independently vet Rust idioms or
catch subtle logic errors by reading the code, correctness here does not
rest on the author's Rust expertise — it rests on the bit-exact
cross-validation suite run against the vendored nanoflann 1.12.1 C++
reference implementation (see [../testing.md](../testing.md)), which checks
every result index, distance, and internal tree permutation against the
real C++ library, in-process, on every change. Every performance figure
quoted in the README and [../benchmarks.md](../benchmarks.md) traces to a
reproducible, pasted command and run recorded in
[../EXPERIMENTS.md](../EXPERIMENTS.md) or in the commit-tagged report data
cited alongside the figure; see
[reports/m-pub/claims-audit.md](reports/m-pub/claims-audit.md) for the
audit that verified that claim.

## Layout

- [`plans/`](plans/) — per-milestone implementation plans (M1 static tree,
  M2 dynamic forest, M2.5 perf, M2.6 rigor, M-py bindings, M-pub, M-ship).
- [`specs/`](specs/) — design/requirement specs the plans implement.
- [`reports/`](reports/) — per-task worker/reviewer reports and audits
  (m2.5, m2.6, m-py, m-pub), the primary evidence chain behind the
  numbers in [../benchmarks.md](../benchmarks.md).

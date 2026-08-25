# Task 1 report — OwnedRows + dataset accessors

## Status: DONE

Commit: `6539e8d` — `feat: OwnedRows row-major DataSource with point_row fast path; dataset accessors`

## RED (captured before implementation)

Test code was added first (OwnedRows usages, `dataset()`/`dataset_mut()` calls) with zero
implementation. `cargo test -p flannrust --lib` failed to compile with 15 errors, e.g.:

```
error[E0432]: unresolved import `crate::data_source::OwnedRows`
   --> crates/flannrust/src/metric.rs:426:41
error[E0432]: unresolved import `crate::data_source::OwnedRows`
    --> crates/flannrust/src/dynamic.rs:1758:45
error[E0425]: cannot find type `OwnedRows` in this scope
   --> crates/flannrust/src/data_source.rs:289:19
error[E0433]: cannot find type `OwnedRows` in this scope
   --> crates/flannrust/src/data_source.rs:268:20
... (15 total, all E0425/E0432/E0433 "OwnedRows" not found)
error: could not compile `flannrust` (lib test) due to 15 previous errors
```

(`dataset()`/`dataset_mut()` calls in the round-trip test were part of the same failing
compilation unit, so no separate RED capture was needed for those — the whole test file
failed to build until both pieces existed.)

## What was built

- `crates/flannrust/src/data_source.rs`: `OwnedRows<T>` — owned row-major buffer (`Vec<T>` +
  `dim`), mirroring `FlatSlice`'s contract but owning its storage instead of borrowing.
  `new` (panics on `dim == 0` or non-multiple length), `with_capacity` (panics on `dim == 0`,
  pre-reserves `dim * n_points`), `push_rows` (panics on ragged input), `dim`/`len`/
  `is_empty`/`as_slice`. `DataSource<T>` implemented for both `OwnedRows<T>` and
  `&OwnedRows<T>` (the latter delegates via `(**self)`), both overriding `point_row` (the
  M2.5 fast-path precondition) with the same `data.get(idx*dim..idx*dim+dim)` bounds-checked
  slice `FlatSlice` uses.
- `crates/flannrust/src/tree.rs`: `KdTree::dataset(&self) -> &DS`.
- `crates/flannrust/src/dynamic.rs`: `DynamicKdTree::dataset(&self) -> &DS` and
  `dataset_mut(&mut self) -> &mut DS`, the latter documented with an explicit append-only
  contract (never shrink below any index already passed to `add_points`).
- `crates/flannrust/src/lib.rs`: re-exported `OwnedRows` alongside `FlatSlice`.

## Tests added (13 new, all passing)

`data_source.rs` (9): count/component, dim==0 panics, ragged-length panics,
`with_capacity` starts empty, `push_rows` grows len/values, `push_rows` ragged-input panics,
`point_row` length+values across every valid idx (never `None` for valid idx), `point_row`
boundary idx, `DataSource` impl for `&OwnedRows<T>`.

`dynamic.rs` (1): `dynamic_over_owned_rows_matches_fresh_static_tree_over_live_rows` — builds
a `DynamicKdTree` over `OwnedRows::with_capacity`, grows it via
`dataset_mut().push_rows(..)` + `add_points` in 3 batches (sizes 3/2/2, indices 0..=6), then
`remove_point(6)` (the last-appended point, so the live rows are exactly the buffer's first
6*dim elements in original index order — no reindexing needed for comparison). Compares
`knn_search` results (index set + distances, order-independent) against a fresh static
`KdTree` built with `FlatSlice` over those same live rows.

`metric.rs` (4): extended the row-vs-fallback bit-equality invariant (Test 11) to `OwnedRows`
at dims {3, 8, 32}, f32/f64, reusing the existing multi-salt `SALTS_F32`/`SALTS_F64` sweep and
`FallbackOnly` wrapper (wrapping `&OwnedRows<T>` since `OwnedRows` isn't `Copy`) — confirms the
M2.5 row kernel engages identically off `OwnedRows`, not just `FlatSlice`.

## Verification commands + decisive output

```
$ cargo test -p flannrust --lib
test result: ok. 197 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out

$ cargo test --workspace
(every crate/suite) test result: ok. ... 0 failed ...
(flannrust lib 197, xval/nanoflann_ref suites 23+12+80+2+..., xval_knn 13, xval_radius_box 12,
 doc-tests flannrust 2 — all 0 failed)

$ cargo clippy --workspace --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.51s   (clean, no warnings)

$ RUSTDOCFLAGS="-D warnings" cargo doc -p flannrust --no-deps
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.28s   (clean, no warnings)

$ cargo build -p flannrust --no-default-features
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.15s
```

New OwnedRows-related tests confirmed present in the run:
```
data_source::tests::test_owned_rows_data_source_impl_for_reference ... ok
data_source::tests::test_owned_rows_new_dim_zero_panics - should panic ... ok
data_source::tests::test_owned_rows_new_non_multiple_length_panics - should panic ... ok
data_source::tests::test_owned_rows_new_point_count_and_component ... ok
data_source::tests::test_owned_rows_point_row_boundary_idx ... ok
data_source::tests::test_owned_rows_point_row_length_and_values ... ok
data_source::tests::test_owned_rows_push_rows_grows_len_and_values ... ok
data_source::tests::test_owned_rows_push_rows_ragged_input_panics - should panic ... ok
data_source::tests::test_owned_rows_with_capacity_starts_empty ... ok
dynamic::tests::dynamic_over_owned_rows_matches_fresh_static_tree_over_live_rows ... ok
metric::tests::l1_eval_row_path_bit_equals_fallback_path_owned_rows_f32 ... ok
metric::tests::l1_eval_row_path_bit_equals_fallback_path_owned_rows_f64 ... ok
metric::tests::l2_eval_row_path_bit_equals_fallback_path_owned_rows_f32 ... ok
metric::tests::l2_eval_row_path_bit_equals_fallback_path_owned_rows_f64 ... ok
```

## Self-review

- Signatures match the brief verbatim (`OwnedRows::new/with_capacity/push_rows/dim/len/
  is_empty/as_slice`, `KdTree::dataset`, `DynamicKdTree::dataset`/`dataset_mut`).
- `DataSource` implemented for both `OwnedRows<T>` and `&OwnedRows<T>` as required.
- `point_row` overridden on both impls (fast-path precondition satisfied) and covered by the
  metric.rs bit-equality extension at the requested dims/dtypes.
- No new `unsafe` introduced. No changes to any existing behavior/signature — purely additive
  (new type, new re-export, new accessor methods on existing structs' existing impl blocks).
- Full workspace suite green unchanged (default-build bit-exactness preserved — nothing in
  the hot search/build/metric code paths was touched, only additive accessors and a new
  `DataSource` impl).
- One commit, as required.

## Concerns

- None outstanding. The `dataset_mut` append-only contract is documentation-only (not
  enforced at runtime) — consistent with this crate's existing style of documenting caller
  contracts rather than runtime-checking every one (e.g. `add_points`'s own doc explicitly
  contrasts its *enforced* per-index panic against contracts elsewhere left as caller
  responsibility). Flagging for awareness, not as a defect.

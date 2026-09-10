#!/usr/bin/env bash
set -euo pipefail

PGO_DIR="${PGO_DIR:-/tmp/flannrust-pgo}"
LLVM_PROFDATA="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | awk '/host:/{print $2}')/bin/llvm-profdata"

if [[ ! -x "$LLVM_PROFDATA" ]]; then
    echo "llvm-profdata not found. Install: rustup component add llvm-tools" >&2
    exit 1
fi

rm -rf "$PGO_DIR"
mkdir -p "$PGO_DIR"

echo "=== Step 1: Instrumented build ==="
RUSTFLAGS="-Cprofile-generate=$PGO_DIR -C target-cpu=native" \
    cargo bench -p xval --bench bench_knn --no-run

echo "=== Step 2: Training run (all knn benchmarks) ==="
RUSTFLAGS="-Cprofile-generate=$PGO_DIR -C target-cpu=native" \
    cargo bench -p xval --bench bench_knn -- --profile-time 2

echo "=== Step 3: Merge profiles ==="
"$LLVM_PROFDATA" merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR"

echo "=== Step 4: PGO-optimized build ==="
RUSTFLAGS="-Cprofile-use=$PGO_DIR/merged.profdata -C target-cpu=native" \
    cargo build -p flannrust --release

echo "=== Step 5: PGO-optimized bench ==="
RUSTFLAGS="-Cprofile-use=$PGO_DIR/merged.profdata -C target-cpu=native" \
    cargo bench -p xval --bench bench_knn

echo "=== Done. Profile data at $PGO_DIR/merged.profdata ==="

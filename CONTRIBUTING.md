# Contributing to flannrust

Thank you for your interest in contributing to flannrust! This document outlines the development workflow and key principles for contributing to this project.

## Development Setup

### Rust Toolchain

The project uses Rust 1.98.0, pinned in `rust-toolchain.toml`. When you check out the repo and run `cargo` commands, `rustup` will automatically switch to the pinned version.

### Python Environment (for Python bindings)

If you're working on the Python bindings in `crates/flannrust-py`:

```bash
# Create a Python virtual environment (Python 3.9+)
python3 -m venv venv
source venv/bin/activate

# Install development dependencies (maturin and other build tools)
pip install maturin
```

### Building Python Bindings

To build the Python bindings in development mode:

```bash
cd crates/flannrust-py
maturin develop
```

## Testing

### Running Rust Tests

Test the entire workspace:

```bash
cargo test --workspace
```

Test a specific crate:

```bash
cargo test -p flannrust
cargo test -p xval
```

### Running Python Tests

If you have the Python bindings installed:

```bash
cd crates/flannrust-py
pytest
```

## Parity Principle

flannrust targets **bit-exact result parity** with the vendored nanoflann 1.12.1 reference implementation (`crates/nanoflann-ref/cpp/nanoflann.hpp`). This is the core guarantee of the port.

**Any changes to the behavior of search results or tree construction must keep the xval (cross-validation) test suite passing.** The xval suite is the judge of parity — it compares Rust results byte-for-byte against the C++ oracle.

If you modify core algorithm logic:

1. Run `cargo test --workspace` to ensure all tests pass
2. Verify that xval results still match the C++ implementation

## Benchmarking

For reproducible performance measurements and benchmarking protocol:

- See `docs/EXPERIMENTS.md` for the full measurement-conditions specification
- See `docs/benchmarks.md` for published results and baseline comparisons

When adding new performance-critical code, consider measuring it against the existing baseline to ensure no regression.

## Code Style

The project follows Rust conventions enforced by `rustfmt`. Before committing:

```bash
cargo fmt --all
cargo clippy --workspace
```

## Licensing

- The Rust library and bindings are licensed under BSD-2-Clause (see `LICENSE`)
- The vendored nanoflann header (`crates/nanoflann-ref/cpp/nanoflann.hpp`) retains its original BSD license header and remains unmodified
- All new contributions must be compatible with BSD-2-Clause

## Getting Help

- Check `docs/nanoflann-notes.md` for detailed notes on nanoflann's design and implementation
- Review the design documents in `docs/superpowers/plans/` for architectural context
- See `README.md` for an overview of the project and its capabilities

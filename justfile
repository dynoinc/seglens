# Format code
fmt:
    cargo fmt

# Run lints and checks
lint: fmt
    cargo clippy -- -D warnings
    cargo check

# Run tests (depends on lint passing)
test: lint
    cargo test

# Build release binary
build:
    cargo build --release

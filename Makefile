
all: lint test


test:
	cargo test

lint:
	cargo clean
	cargo clippy

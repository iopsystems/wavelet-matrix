.PHONY: wasm
wasm:
	wasm-pack build --target web --no-typescript --reference-types --dev

.PHONY: wasm-profiling
wasm-profiling:
	wasm-pack build --target web --no-typescript --reference-types --profiling

.PHONY: wasm-release
wasm-release:
	wasm-pack build --target web --no-typescript --reference-types

.PHONY: test
test:
# 	RUST_BACKTRACE=full cargo test -- --nocapture --test-threads=1
	RUST_BACKTRACE=full cargo test -- --test-threads=1
# 	cargo test 

.PHONY: run
run:
#	cargo run --release
	cargo run
	
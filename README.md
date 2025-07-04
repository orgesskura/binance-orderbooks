# High-Performance Binance Orderbook Implementation

A production-ready, high-performance Rust implementation of a Binance orderbook that processes Individual Symbol Book Ticker and Partial Book Depth streams with microsecond-level latency optimization and comprehensive overflow protection. Check **description.md** for more.

## Overview

This implementation demonstrates high proficiency Rust performance engineering for financial systems, featuring:

- **Direct JSON deserialization** to typed values (no intermediate string allocation)
- **Fixed-point arithmetic** for exact precision and overflow protection
- **Stack-allocated data structures** (ArrayVec) for zero heap allocation
- **Cache-optimized memory layout** for high-frequency trading scenarios
- **Real-time WebSocket integration** with configurable binance stream
- **Comprehensive input validation** with graceful error handling

Built specifically for cryptocurrency market making and high-frequency trading applications requiring microsecond-level performance with bulletproof safety guarantees.

## Getting Started

### Build Dependencies
```bash
cargo build
```
Downloads and compiles all required dependencies:
- `serde` - JSON serialization/deserialization
- `serde_json` - High-performance JSON parsing
- `arrayvec` - Fixed-capacity stack arrays
- `tokio` - Async runtime for WebSocket connections
- `tokio-tungstenite` - WebSocket client implementation
- `futures-util` - Stream utilities for async processing

### Run Tests
```bash
cargo test
```
Executes comprehensive test suite including:
- Unit tests for all orderbook components
- JSON deserialization with direct type conversion
- Overflow protection and input validation
- Realistic trading scenarios with spread analysis
- Performance tests with bulk update processing
- Error handling for malicious/invalid inputs

Expected output:
```
running 18 tests
test orderbooks::tests::test_book_ticker_deserialization ... ok
test orderbooks::tests::test_book_ticker_update ... ok
test orderbooks::tests::test_depth_update_deserialization ... ok
test orderbooks::tests::test_edge_values ... ok
test orderbooks::tests::test_infinity_and_nan ... ok
test orderbooks::tests::test_invalid_price_error ... ok
test orderbooks::tests::test_invalid_symbol_error ... ok
test orderbooks::tests::test_malicious_input_rejection ... ok
test orderbooks::tests::test_negative_prices ... ok
test orderbooks::tests::test_orderbook_creation ... ok
test orderbooks::tests::test_orderbook_formatting ... ok
test orderbooks::tests::test_sorted_order_maintained ... ok
test orderbooks::tests::test_realistic_orderbook_scenario ... ok
test orderbooks::tests::test_twenty_level_capacity ... ok
test orderbooks::tests::test_update_depth_2 ... ok
test orderbooks::tests::test_zero_quantity_removal ... ok
test orderbooks::tests::test_update_depth ... ok
test orderbooks::tests::test_bulk_updates ... ok

test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

### Run BTCUSDT Live Stream
```bash
cargo run
```
Connects to **Binance WebSocket API** and runs a live **BTCUSDT partial  or full book depth stream** that:
- **Connects to real Binance WebSocket** (`wss://stream.binance.com:9443`)
- **Prints top 20 best bids/asks every 10 seconds** in formatted table
- **Shows real-time market data** with live price movements
- **Demonstrates production-ready WebSocket handling** with heartbeat and error recovery
- **Is capable of processing thousands of updates per second** ( laptop measured latency on average 4 - 5 microseconds )
- **Run with release flag to optimize performance**
- **To run partial depth stream, pass the argument partial to cargo run:  cargo run --release -- partial. For full depth: cargo run --release --full**

Expected output:
```
Connecting to BTCUSDT book depth stream...
Connecting to: wss://stream.binance.com:9443/ws/btcusdt@depth@100ms
Stream: btcusdt@depth20@100ms
Connected! Listening for messages...
Average depth update took 0.14851485 microseconds
[ 1] [ 0.15420 ] 43250.540 | 43251.110 [ 0.23450 ]
[ 2] [ 0.28350 ] 43250.250 | 43251.250 [ 0.34560 ]
[ 3] [ 0.45720 ] 43250.000 | 43251.500 [ 0.12890 ]
...
[20] [ 0.41230 ] 43240.250 | 43261.250 [ 0.18920 ]
-------------------------------------------------------------
```

The live stream demonstrates real-world performance handling thousands of Binance market data updates with zero heap allocation and sub micro-second processing latency.

---

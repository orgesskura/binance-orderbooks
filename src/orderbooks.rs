use arrayvec::ArrayVec;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

// I am using [inline] on functions that are called a lot. That is to improve performance by eliminating function call overhead.

// Type aliases for clarity
type Price = f64;
type Quantity = f64;

// Custom error types for better error handling
#[derive(Debug, Clone)]
pub enum OrderBookError {
    InvalidPrice(String),
    InvalidQuantity(String),
    InvalidSymbol(String),
    SerializationError(String),
}

impl fmt::Display for OrderBookError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OrderBookError::InvalidPrice(msg) => write!(f, "Invalid price: {}", msg),
            OrderBookError::InvalidQuantity(msg) => write!(f, "Invalid quantity: {}", msg),
            OrderBookError::InvalidSymbol(msg) => write!(f, "Invalid symbol: {}", msg),
            OrderBookError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for OrderBookError {}

// Force Inline. This function is called a lot.
#[inline(always)]
fn validate_price(price: f64) -> Result<f64, OrderBookError> {
    if !price.is_finite() || price <= 0.0 {
        return Err(OrderBookError::InvalidPrice(format!(
            "Invalid price: {}",
            price
        )));
    }
    Ok(price)
}

// Force Inline. This function is called a lot.
#[inline(always)]
fn validate_quantity(quantity: f64) -> Result<f64, OrderBookError> {
    if !quantity.is_finite() || quantity < 0.0 {
        return Err(OrderBookError::InvalidQuantity(format!(
            "Invalid quantity: {}",
            quantity
        )));
    }
    Ok(quantity)
}

#[derive(Debug, Copy, Clone)]
pub struct PriceLevel {
    pub price: Price,
    pub quantity: Quantity,
}

// Custom deserializer for string-to-f64 conversion
fn deserialize_string_to_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BookTickerUpdate {
    #[serde(rename = "u")]
    pub update_id: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b", deserialize_with = "deserialize_string_to_f64")]
    pub bid_price: f64,
    #[serde(rename = "B", deserialize_with = "deserialize_string_to_f64")]
    pub bid_qty: f64,
    #[serde(rename = "a", deserialize_with = "deserialize_string_to_f64")]
    pub ask_price: f64,
    #[serde(rename = "A", deserialize_with = "deserialize_string_to_f64")]
    pub ask_qty: f64,
}

impl BookTickerUpdate {
    pub fn from_json(json_str: &str) -> Result<Self, OrderBookError> {
        serde_json::from_str(json_str)
            .map_err(|e| OrderBookError::SerializationError(e.to_string()))
    }

    // Safe parsing with validation
    #[inline]
    pub fn parse_best_bid(&self) -> Result<(Price, Quantity), OrderBookError> {
        let price = validate_price(self.bid_price)?;
        let quantity = validate_quantity(self.bid_qty)?;

        if quantity <= 0.0 {
            return Err(OrderBookError::InvalidQuantity(
                "Bid quantity must be positive".to_string(),
            ));
        }

        Ok((price, quantity))
    }

    #[inline]
    pub fn parse_best_ask(&self) -> Result<(Price, Quantity), OrderBookError> {
        let price = validate_price(self.ask_price)?;
        let quantity = validate_quantity(self.ask_qty)?;

        if quantity <= 0.0 {
            return Err(OrderBookError::InvalidQuantity(
                "Ask quantity must be positive".to_string(),
            ));
        }

        Ok((price, quantity))
    }
}

// Custom deserializer for string array to f64 array conversion
fn deserialize_string_array_to_f64<'de, D>(
    deserializer: D,
) -> Result<ArrayVec<[f64; 2], 20>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let string_arrays: Vec<[String; 2]> = Vec::deserialize(deserializer)?;
    let mut result = ArrayVec::new();

    for [price_str, qty_str] in string_arrays {
        if result.is_full() {
            break; // Enforce 20-level limit
        }
        let price = price_str.parse().map_err(serde::de::Error::custom)?;
        let qty = qty_str.parse().map_err(serde::de::Error::custom)?;
        result.push([price, qty]);
    }

    Ok(result)
}

// Custom deserializer for string array to f64 vec conversion
fn deserialize_vec_to_f64<'de, D>(deserializer: D) -> Result<Vec<[f64; 2]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let string_arrays: Vec<[String; 2]> = Vec::deserialize(deserializer)?;
    let mut result = Vec::new();

    for [price_str, qty_str] in string_arrays {
        let price = price_str.parse().map_err(serde::de::Error::custom)?;
        let qty = qty_str.parse().map_err(serde::de::Error::custom)?;
        result.push([price, qty]);
    }

    Ok(result)
}

// Depth Update with maximum 20 levels - No heap allocations
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PartialDepthUpdate {
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: u64,
    #[serde(deserialize_with = "deserialize_string_array_to_f64")]
    pub bids: ArrayVec<[f64; 2], 20>,
    #[serde(deserialize_with = "deserialize_string_array_to_f64")]
    pub asks: ArrayVec<[f64; 2], 20>,
}

impl PartialDepthUpdate {
    pub fn from_json(json_str: &str) -> Result<Self, OrderBookError> {
        serde_json::from_str(json_str)
            .map_err(|e| OrderBookError::SerializationError(e.to_string()))
    }

    // Batch parsing with validation
    #[inline]
    pub fn parse_bids(&self) -> Result<ArrayVec<(Price, Quantity), 20>, OrderBookError> {
        let mut results = ArrayVec::new();

        for level in &self.bids {
            let price = validate_price(level[0])?;
            let quantity = validate_quantity(level[1])?;
            results.push((price, quantity));
        }

        Ok(results)
    }

    // Batch parsing with validation
    #[inline]
    pub fn parse_asks(&self) -> Result<ArrayVec<(Price, Quantity), 20>, OrderBookError> {
        let mut results = ArrayVec::new();

        for level in &self.asks {
            let price = validate_price(level[0])?;
            let quantity = validate_quantity(level[1])?;
            results.push((price, quantity));
        }

        Ok(results)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DepthUpdate {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "u")]
    pub last_update_id: u64,
    #[serde(rename = "b", deserialize_with = "deserialize_vec_to_f64")]
    pub bids: Vec<[f64; 2]>,
    #[serde(rename = "a", deserialize_with = "deserialize_vec_to_f64")]
    pub asks: Vec<[f64; 2]>,
}

impl DepthUpdate {
    pub fn from_json(json_str: &str) -> Result<Self, OrderBookError> {
        serde_json::from_str(json_str)
            .map_err(|e| OrderBookError::SerializationError(e.to_string()))
    }

    // Batch parsing with validation
    #[inline]
    pub fn parse_bids(&self) -> Result<Vec<(Price, Quantity)>, OrderBookError> {
        let mut results = Vec::new();

        for level in &self.bids {
            let price = validate_price(level[0])?;
            let quantity = validate_quantity(level[1])?;
            results.push((price, quantity));
        }

        Ok(results)
    }

    // Batch parsing with validation
    #[inline]
    pub fn parse_asks(&self) -> Result<Vec<(Price, Quantity)>, OrderBookError> {
        let mut results = Vec::new();

        for level in &self.asks {
            let price = validate_price(level[0])?;
            let quantity = validate_quantity(level[1])?;
            results.push((price, quantity));
        }

        Ok(results)
    }
}

#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: String,
    // Stack-allocated sorted arrays (20 levels max) . Firstly I was going to use BTreeMap to store prices and quantities maps.
    // However, we are guaranteed to have 20 levels and we are only interested in top 20 levels.
    // Also, we parse f64 for price and quantities. f64 does not implement Ord so
    // I would need to convert f64 -> u64 each time in order to use BTreeMap keys effectively. That and the fact I would need to allocate on the
    // heap made ArrayVec a better option for the problem's specifications ( With BTreeMap higher cache misses , more allocations on the heap. arrayvec is contigous on stack and can fit in the cache for 20 levels).
    bids: ArrayVec<PriceLevel, 20>, // Sorted descending (best first)
    asks: ArrayVec<PriceLevel, 20>, // Sorted ascending (best first)
    last_update_id: u64,
}

impl OrderBook {
    pub fn new(symbol: String) -> Result<OrderBook, OrderBookError> {
        if symbol.is_empty() {
            return Err(OrderBookError::InvalidSymbol(
                "Symbol cannot be empty".to_string(),
            ));
        }

        Ok(OrderBook {
            symbol,
            bids: ArrayVec::new(),
            asks: ArrayVec::new(),
            last_update_id: 0,
        })
    }

    #[inline]
    pub fn update_book_ticker(&mut self, data: &BookTickerUpdate) -> Result<(), OrderBookError> {
        if data.symbol != self.symbol {
            return Err(OrderBookError::InvalidSymbol(format!(
                "Symbol mismatch: expected {}, got {}",
                self.symbol, data.symbol
            )));
        }

        let bid_price = validate_price(data.bid_price)?;
        let ask_price = validate_price(data.ask_price)?;

        if data.bid_qty <= 0.0 || data.ask_qty <= 0.0 {
            return Err(OrderBookError::InvalidQuantity(
                "Quantities must be positive".to_string(),
            ));
        }

        // Update best bid
        self.update_or_insert_bid(bid_price, data.bid_qty);

        // Update best ask
        self.update_or_insert_ask(ask_price, data.ask_qty);

        self.last_update_id = data.update_id;
        Ok(())
    }

    #[inline]
    pub fn update_partial_depth(
        &mut self,
        data: &PartialDepthUpdate,
    ) -> Result<(), OrderBookError> {
        // Very efficient way of cleaning up memory
        std::mem::take(&mut self.bids);
        std::mem::take(&mut self.asks);
        // Process bid updates
        for bid_data in &data.bids {
            let price = validate_price(bid_data[0])?;
            let qty = bid_data[1];

            if qty > 0.0 {
                self.update_or_insert_bid(price, qty);
            } else {
                return Err(OrderBookError::InvalidQuantity(
                    "Quantity needs to be greater than 0".to_string(),
                ));
            }
        }

        // Process ask updates
        for ask_data in &data.asks {
            let price = validate_price(ask_data[0])?;
            let qty = ask_data[1];

            if qty > 0.0 {
                self.update_or_insert_ask(price, qty);
            } else {
                return Err(OrderBookError::InvalidQuantity(
                    "Quantity needs to be greater than 0".to_string(),
                ));
            }
        }

        self.last_update_id = data.last_update_id;
        Ok(())
    }

    // O(1) best bid/ask access (first elements in sorted arrays) - Force inline
    #[inline(always)]
    pub fn get_best_bid_ask(&self) -> Option<((f64, f64), (f64, f64))> {
        match (self.bids.first(), self.asks.first()) {
            (Some(best_bid), Some(best_ask)) => Some((
                (best_bid.price, best_bid.quantity),
                (best_ask.price, best_ask.quantity),
            )),
            _ => None,
        }
    }

    // Optimized bid insertion maintaining descending order
    #[inline]
    fn update_or_insert_bid(&mut self, price: Price, quantity: Quantity) {
        // Find existing level or insertion point (reverse order for bids)
        match self
            .bids
            .binary_search_by(|level| level.price.partial_cmp(&price).unwrap().reverse())
        {
            Ok(index) => {
                // Update existing level
                self.bids[index].quantity = quantity;
            }
            Err(index) => {
                // Insert new level at correct position
                let new_level = PriceLevel { price, quantity };

                if self.bids.len() < 20 {
                    self.bids.insert(index, new_level);
                } else if index < 20 {
                    // Remove last element and insert new one
                    self.bids.pop();
                    self.bids.insert(index, new_level);
                }
                // If index >= 20, ignore (outside our 20-level window)
            }
        }
    }

    // Optimized ask insertion maintaining ascending order
    #[inline]
    fn update_or_insert_ask(&mut self, price: Price, quantity: Quantity) {
        match self
            .asks
            .binary_search_by(|level| level.price.partial_cmp(&price).unwrap())
        {
            Ok(index) => {
                // Update existing level
                self.asks[index].quantity = quantity;
            }
            Err(index) => {
                // Insert new level at correct position
                let new_level = PriceLevel { price, quantity };

                if self.asks.len() < 20 {
                    self.asks.insert(index, new_level);
                } else if index < 20 {
                    // Remove last element and insert new one
                    self.asks.pop();
                    self.asks.insert(index, new_level);
                }
                // If index >= 20, ignore (outside our 20-level window)
            }
        }
    }

    #[inline]
    fn remove_bid(&mut self, price: Price) {
        if let Ok(index) = self
            .bids
            .binary_search_by(|level| level.price.partial_cmp(&price).unwrap().reverse())
        {
            self.bids.remove(index);
        }
    }

    #[inline]
    fn remove_ask(&mut self, price: Price) {
        if let Ok(index) = self
            .asks
            .binary_search_by(|level| level.price.partial_cmp(&price).unwrap())
        {
            self.asks.remove(index);
        }
    }

    // Fast level extraction with stack-allocated arrays
    #[inline]
    pub fn get_levels(&self, depth: usize) -> (ArrayVec<(f64, f64), 20>, ArrayVec<(f64, f64), 20>) {
        let bid_depth = depth.min(self.bids.len()).min(20);
        let ask_depth = depth.min(self.asks.len()).min(20);

        let mut bids = ArrayVec::new();
        let mut asks = ArrayVec::new();

        // Fill bids array (no heap allocation)
        for level in self.bids.iter().take(bid_depth) {
            bids.push((level.price, level.quantity));
        }

        // Fill asks array (no heap allocation)
        for level in self.asks.iter().take(ask_depth) {
            asks.push((level.price, level.quantity));
        }

        (bids, asks)
    }

    // Optimized string formatting with minimal allocation
    pub fn to_string(&self) -> String {
        let (bids, asks) = self.get_levels(20);

        // Pre-calculate string capacity to avoid reallocation
        let estimated_size = bids.len().max(asks.len()) * 64;
        let mut result = String::with_capacity(estimated_size);

        let max_levels = bids.len().max(asks.len());

        for i in 0..max_levels {
            result.push_str(&format!("[{:2}] ", i + 1));

            if i < bids.len() {
                result.push_str(&format!("[ {:.5} ] {:>9.3}", bids[i].1, bids[i].0));
            } else {
                result.push_str(&format!("[ {:>7} ] {:>9}", "", ""));
            }

            result.push_str(" | ");

            if i < asks.len() {
                result.push_str(&format!("{:>9.3} [ {:.5} ]", asks[i].0, asks[i].1));
            } else {
                result.push_str(&format!("{:>9} [ {:>7} ]", "", ""));
            }

            result.push('\n');
        }

        result
    }

    // Performance monitoring helpers
    #[inline]
    pub fn bid_count(&self) -> usize {
        self.bids.len()
    }

    #[inline]
    pub fn ask_count(&self) -> usize {
        self.asks.len()
    }

    #[inline]
    pub fn last_update_id(&self) -> u64 {
        self.last_update_id
    }
}

#[derive(Debug, Clone)]
pub struct FullDepthOrderBook {
    pub symbol: String,
    pub bids: BTreeMap<OrderedFloat<Price>, Quantity>,
    pub asks: BTreeMap<OrderedFloat<Price>, Quantity>,
    last_update_id: u64,
}

impl FullDepthOrderBook {
    pub fn new(symbol: String) -> Result<FullDepthOrderBook, OrderBookError> {
        if symbol.is_empty() {
            return Err(OrderBookError::InvalidSymbol(
                "Symbol cannot be empty".to_string(),
            ));
        }

        Ok(FullDepthOrderBook {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
        })
    }

    #[inline]
    pub fn update_depth(&mut self, data: &DepthUpdate) -> Result<(), OrderBookError> {
        if data.last_update_id < self.last_update_id || self.symbol != data.symbol {
            return Ok(());
        }
        // Process bid updates
        for bid_data in &data.bids {
            let price = OrderedFloat(validate_price(bid_data[0])?);
            let qty = bid_data[1];

            if qty == 0.0 {
                self.bids.remove(&price);
            } else if qty > 0.0 {
                self.bids.insert(price, qty);
            } else {
                return Err(OrderBookError::InvalidQuantity(
                    "Quantity cannot be negative".to_string(),
                ));
            }
        }

        // Process ask updates
        for ask_data in &data.asks {
            let price = OrderedFloat(validate_price(ask_data[0])?);
            let qty = ask_data[1];

            if qty == 0.0 {
                self.asks.remove(&price);
            } else if qty > 0.0 {
                self.asks.insert(price, qty);
            } else {
                return Err(OrderBookError::InvalidQuantity(
                    "Quantity cannot be negative".to_string(),
                ));
            }
        }

        self.last_update_id = data.last_update_id;
        Ok(())
    }

    #[inline]
    pub fn get_levels(
        &self,
        depth: usize,
    ) -> (
        ArrayVec<(&OrderedFloat<Price>, &f64), 20>,
        ArrayVec<(&OrderedFloat<Price>, &f64), 20>,
    ) {
        let bid_depth = depth.min(self.bids.len()).min(20);
        let ask_depth = depth.min(self.asks.len()).min(20);

        let mut bids = ArrayVec::new();
        let mut asks = ArrayVec::new();

        // Fill bids array (no heap allocation)
        for level in self.bids.iter().rev().take(bid_depth) {
            bids.push((level.0, level.1));
        }

        // Fill asks array (no heap allocation)
        for level in self.asks.iter().take(ask_depth) {
            asks.push((level.0, level.1));
        }

        (bids, asks)
    }

    // Optimized string formatting with minimal allocation
    pub fn to_string(&self) -> String {
        let (bids, asks) = self.get_levels(20);

        // Pre-calculate string capacity to avoid reallocation
        let estimated_size = bids.len().max(asks.len()) * 64;
        let mut result = String::with_capacity(estimated_size);

        let max_levels = bids.len().max(asks.len());

        for i in 0..max_levels {
            result.push_str(&format!("[{:2}] ", i + 1));

            if i < bids.len() {
                result.push_str(&format!("[ {:.5} ] {:>9.3}", bids[i].1, bids[i].0));
            } else {
                result.push_str(&format!("[ {:>7} ] {:>9}", "", ""));
            }

            result.push_str(" | ");

            if i < asks.len() {
                result.push_str(&format!("{:>9.3} [ {:.5} ]", asks[i].0, asks[i].1));
            } else {
                result.push_str(&format!("{:>9} [ {:>7} ]", "", ""));
            }

            result.push('\n');
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_book_ticker_deserialization() {
        let json = r#"
        {
            "u":400900217,
            "s":"BNBUSDT",
            "b":"25.35190000",
            "B":"31.21000000",
            "a":"25.36520000",
            "A":"40.66000000"
        }"#;

        let update = BookTickerUpdate::from_json(json).unwrap();
        assert_eq!(update.update_id, 400900217);
        assert_eq!(update.symbol, "BNBUSDT");

        let (bid_price, bid_qty) = update.parse_best_bid().unwrap();
        let (ask_price, ask_qty) = update.parse_best_ask().unwrap();

        assert_eq!(bid_price, 25.3519);
        assert_eq!(bid_qty, 31.21);
        assert_eq!(ask_price, 25.3652);
        assert_eq!(ask_qty, 40.66);
    }

    #[test]
    fn test_depth_update_deserialization() {
        let json = r#"
        {
            "lastUpdateId": 160,
            "bids": [
                ["0.0024", "10"]
            ],
            "asks": [
                ["0.0026", "100"]
            ]
        }"#;

        let update = PartialDepthUpdate::from_json(json).unwrap();
        assert_eq!(update.last_update_id, 160);

        let bids = update.parse_bids().unwrap();
        let asks = update.parse_asks().unwrap();

        assert_eq!(bids[0].0, 0.0024);
        assert_eq!(bids[0].1, 10.0);
        assert_eq!(asks[0].0, 0.0026);
        assert_eq!(asks[0].1, 100.0);
    }

    #[test]
    fn test_orderbook_creation() {
        let book = OrderBook::new("BTCUSDT".to_string()).unwrap();
        assert_eq!(book.symbol, "BTCUSDT");
        assert_eq!(book.get_best_bid_ask(), None);
    }

    #[test]
    fn test_update_depth() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();
        let mut bids = ArrayVec::new();
        bids.push([50000.0, 1.0]);
        bids.push([49999.0, 2.0]);

        let mut asks = ArrayVec::new();
        asks.push([50001.0, 1.5]);
        asks.push([50002.0, 2.5]);

        let update = PartialDepthUpdate {
            last_update_id: 1,
            bids,
            asks,
        };

        book.update_partial_depth(&update).unwrap();

        let best = book.get_best_bid_ask().unwrap();
        assert_eq!(best.0, (50000.0, 1.0)); // Best bid
        assert_eq!(best.1, (50001.0, 1.5)); // Best ask
    }

    #[test]
    fn test_sorted_order_maintained() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();

        // Create update with ArrayVec in random order
        let mut bids = ArrayVec::new();
        bids.push([49995.0, 1.0]);
        bids.push([50000.0, 2.0]);
        bids.push([49999.0, 1.5]);

        let mut asks = ArrayVec::new();
        asks.push([50005.0, 1.0]);
        asks.push([50001.0, 2.0]);
        asks.push([50003.0, 1.5]);

        let update = PartialDepthUpdate {
            last_update_id: 1,
            bids,
            asks,
        };

        book.update_partial_depth(&update).unwrap();

        let (bids, asks) = book.get_levels(3);

        // Bids should be descending (highest first)
        assert_eq!(bids[0].0, 50000.0);
        assert_eq!(bids[1].0, 49999.0);
        assert_eq!(bids[2].0, 49995.0);

        // Asks should be ascending (lowest first)
        assert_eq!(asks[0].0, 50001.0);
        assert_eq!(asks[1].0, 50003.0);
        assert_eq!(asks[2].0, 50005.0);
    }

    #[test]
    fn test_book_ticker_update() {
        let mut book = OrderBook::new("BNBUSDT".to_string()).unwrap();

        let update = BookTickerUpdate {
            update_id: 400900217,
            symbol: "BNBUSDT".to_string(),
            bid_price: 25.35190000,
            bid_qty: 31.21000000,
            ask_price: 25.36520000,
            ask_qty: 40.66000000,
        };

        book.update_book_ticker(&update).unwrap();

        let best = book.get_best_bid_ask().unwrap();
        assert!((best.0 .0 - 25.3519).abs() < 1e-6);
        assert!((best.0 .1 - 31.21).abs() < 1e-6);
        assert!((best.1 .0 - 25.3652).abs() < 1e-6);
        assert!((best.1 .1 - 40.66).abs() < 1e-6);
    }

    #[test]
    fn test_update_depth_2() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();

        let mut bids = ArrayVec::new();
        bids.push([50000.0, 1.0]);
        bids.push([49999.0, 2.0]);

        let mut asks = ArrayVec::new();
        asks.push([50001.0, 1.5]);
        asks.push([50002.0, 2.5]);

        let update = PartialDepthUpdate {
            last_update_id: 160,
            bids,
            asks,
        };

        book.update_partial_depth(&update).unwrap();

        let best = book.get_best_bid_ask().unwrap();
        assert_eq!(best.0, (50000.0, 1.0));
        assert_eq!(best.1, (50001.0, 1.5));
        assert_eq!(book.bid_count(), 2);
        assert_eq!(book.ask_count(), 2);
    }

    #[test]
    fn test_invalid_symbol_error() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();

        let update = BookTickerUpdate {
            update_id: 1,
            symbol: "ETHUSDT".to_string(), // Wrong symbol
            bid_price: 25.35190000,
            bid_qty: 31.21000000,
            ask_price: 25.36520000,
            ask_qty: 40.66000000,
        };

        let result = book.update_book_ticker(&update);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OrderBookError::InvalidSymbol(_)
        ));
    }

    #[test]
    fn test_invalid_price_error() {
        let update = BookTickerUpdate {
            update_id: 400900217,
            symbol: "BNBUSDT".to_string(),
            bid_price: -1.0, // Invalid negative price
            bid_qty: 31.21000000,
            ask_price: 25.36520000,
            ask_qty: 40.66000000,
        };

        let result = update.parse_best_bid();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OrderBookError::InvalidPrice(_)
        ));
    }

    #[test]
    fn test_orderbook_formatting() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();

        let mut bids = ArrayVec::new();
        bids.push([50000.0, 1.0]);
        bids.push([49999.0, 2.0]);

        let mut asks = ArrayVec::new();
        asks.push([50001.0, 1.5]);
        asks.push([50002.0, 2.5]);

        let update = PartialDepthUpdate {
            last_update_id: 1,
            bids,
            asks,
        };

        book.update_partial_depth(&update).unwrap();
        let formatted = book.to_string();

        // Check that the output contains expected elements
        assert!(formatted.contains("50000.000"));
        assert!(formatted.contains("50001.000"));
        assert!(formatted.contains("[ 1]"));
        assert!(formatted.contains("|"));
    }

    #[test]
    fn test_edge_values() {
        // Test smallest representable value
        assert!(validate_price(0.00000001).is_ok());

        // Test zero (should fail for price)
        assert!(validate_price(0.0).is_err());

        // Test large values
        assert!(validate_price(1_000_000.0).is_ok());
        assert!(validate_price(10_000_000.0).is_ok());
    }

    #[test]
    fn test_negative_prices() {
        assert!(validate_price(-1.0).is_err());
        assert!(validate_price(-0.1).is_err());
    }

    #[test]
    fn test_infinity_and_nan() {
        assert!(validate_price(f64::INFINITY).is_err());
        assert!(validate_price(f64::NEG_INFINITY).is_err());
        assert!(validate_price(f64::NAN).is_err());
    }

    #[test]
    fn test_realistic_orderbook_scenario() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();

        // Initial depth snapshot with clear bid-ask spread
        let depth_json = r#"
        {
            "lastUpdateId": 1000,
            "bids": [
                ["49998.00", "1.5"],
                ["49997.50", "2.0"],
                ["49997.00", "0.8"]
            ],
            "asks": [
                ["50002.00", "1.2"],
                ["50002.50", "2.5"],
                ["50003.00", "0.9"]
            ]
        }"#;

        let depth_update = PartialDepthUpdate::from_json(depth_json).unwrap();
        book.update_partial_depth(&depth_update).unwrap();

        // Verify initial state - clear $4 spread
        let best = book.get_best_bid_ask().unwrap();
        assert_eq!(best.0, (49998.0, 1.5)); // Best bid: $49,998
        assert_eq!(best.1, (50002.0, 1.2)); // Best ask: $50,002
        let initial_spread = best.1 .0 - best.0 .0;
        assert_eq!(initial_spread, 4.0); // $4 spread

        // Update with book ticker - tightening the spread
        let ticker_update = BookTickerUpdate {
            update_id: 1001,
            symbol: "BTCUSDT".to_string(),
            bid_price: 49999.0,
            bid_qty: 3.0,
            ask_price: 50001.0,
            ask_qty: 2.0,
        };

        book.update_book_ticker(&ticker_update).unwrap();

        // Verify updated state - spread tightened to $2
        let best_after = book.get_best_bid_ask().unwrap();
        assert_eq!(best_after.0, (49999.0, 3.0)); // New best bid: $49,999 (higher)
        assert_eq!(best_after.1, (50001.0, 2.0)); // New best ask: $50,001 (lower)
        let new_spread = best_after.1 .0 - best_after.0 .0;
        assert_eq!(new_spread, 2.0); // Tighter $2 spread

        // Verify the spread tightened (more liquid market)
        assert!(new_spread < initial_spread);

        // Verify orderbook integrity: bid < ask (no crossing)
        assert!(best_after.0 .0 < best_after.1 .0);
    }

    #[test]
    fn test_twenty_level_capacity() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();

        // Create 20 bid levels (at capacity)
        let mut bids = ArrayVec::new();
        for i in 0..20 {
            bids.push([50000.0 - i as f64, 1.0]);
        }

        let update = PartialDepthUpdate {
            last_update_id: 1,
            bids,
            asks: ArrayVec::new(),
        };

        book.update_partial_depth(&update).unwrap();

        // Should have exactly 20 levels
        assert_eq!(book.bid_count(), 20);

        // Best bid should be the highest price
        let best = book.get_best_bid_ask();
        if let Some(((best_bid_price, _), _)) = best {
            assert_eq!(best_bid_price, 50000.0);
        }
    }

    #[test]
    fn test_bulk_updates() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();

        // Simulate high-frequency updates
        for i in 0..1000 {
            let mut bids = ArrayVec::new();
            bids.push([50000.0 - i as f64, 1.0]);
            bids.push([49999.5 - i as f64, 2.0]);

            let mut asks = ArrayVec::new();
            asks.push([50001.0 + i as f64, 1.5]);
            asks.push([50001.5 + i as f64, 2.5]);

            let update = PartialDepthUpdate {
                last_update_id: i,
                bids,
                asks,
            };

            book.update_partial_depth(&update).unwrap();

            // Verify we can still access best levels efficiently
            assert!(book.get_best_bid_ask().is_some());
        }

        // Verify final state
        assert!(book.bid_count() <= 20);
        assert!(book.ask_count() <= 20);
    }

    #[test]
    fn test_malicious_input_rejection() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();

        // Test various malicious inputs
        let mut malicious_cases = Vec::new();

        // Case 1: Infinity
        let mut bids1 = ArrayVec::new();
        bids1.push([f64::INFINITY, 1.0]);
        malicious_cases.push(PartialDepthUpdate {
            last_update_id: 1,
            bids: bids1,
            asks: ArrayVec::new(),
        });

        // Case 2: NaN
        let mut bids2 = ArrayVec::new();
        bids2.push([f64::NAN, 1.0]);
        malicious_cases.push(PartialDepthUpdate {
            last_update_id: 2,
            bids: bids2,
            asks: ArrayVec::new(),
        });

        // Case 3: Negative
        let mut bids3 = ArrayVec::new();
        bids3.push([-1.0, 1.0]);
        malicious_cases.push(PartialDepthUpdate {
            last_update_id: 3,
            bids: bids3,
            asks: ArrayVec::new(),
        });

        for malicious_update in malicious_cases {
            let result = book.update_partial_depth(&malicious_update);

            // Should fail gracefully without corrupting the orderbook
            assert!(result.is_err());

            // Orderbook should still be functional
            assert_eq!(book.bid_count(), 0);
            assert_eq!(book.ask_count(), 0);
        }
    }
}

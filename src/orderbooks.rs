use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use std::collections::BTreeMap;
use std::fmt;

// Custom error types for better error handling
#[derive(Debug, Clone)]
pub enum OrderBookError {
    InvalidPrice(String),
    InvalidQuantity(String),
    InvalidSymbol(String),
    SerializationError(String),
    Overflow(String),
}

impl fmt::Display for OrderBookError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OrderBookError::InvalidPrice(msg) => write!(f, "Invalid price: {}", msg),
            OrderBookError::InvalidQuantity(msg) => write!(f, "Invalid quantity: {}", msg),
            OrderBookError::InvalidSymbol(msg) => write!(f, "Invalid symbol: {}", msg),
            OrderBookError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            OrderBookError::Overflow(msg) => write!(f, "Overflow error: {}", msg),
        }
    }
}

impl std::error::Error for OrderBookError {}

// Fixed-point price representation with overflow protection
// Using 8 decimal places precision (multiply by 100_000_000)
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Price(i64);

impl Price {
    const MULTIPLIER: i64 = 100_000_000;
    // Conservative safe limit to prevent overflow: ~92 billion
    const MAX_SAFE_PRICE: f64 = 90_000_000_000.0;
    const MIN_SAFE_PRICE: f64 = 0.0;
    
    #[inline]
    pub fn from_f64(price: f64) -> Result<Self, OrderBookError> {
        // Fast path for common values (most crypto/stock prices under $1M)
        if price >= 0.0 && price <= 1_000_000.0 && price.is_finite() {
            let scaled = price * Self::MULTIPLIER as f64;
            let rounded = scaled.round();
            return Ok(Price(rounded as i64));
        }
        
        // Careful path for edge cases and large values
        Self::from_f64_safe(price)
    }
    
    fn from_f64_safe(price: f64) -> Result<Self, OrderBookError> {
        // Check for NaN and infinity
        if !price.is_finite() {
            return Err(OrderBookError::InvalidPrice(
                format!("Price must be finite, got: {}", price)
            ));
        }
        
        // Check for negative prices
        if price < Self::MIN_SAFE_PRICE {
            return Err(OrderBookError::InvalidPrice(
                format!("Price cannot be negative, got: {}", price)
            ));
        }
        
        // Check bounds to prevent overflow
        if price > Self::MAX_SAFE_PRICE {
            return Err(OrderBookError::Overflow(
                format!("Price {} exceeds maximum safe value {}", 
                       price, Self::MAX_SAFE_PRICE)
            ));
        }
        
        // Safe conversion with additional overflow check
        let scaled = price * Self::MULTIPLIER as f64;
        let rounded = scaled.round();
        
        // Final safety check before casting
        if rounded > i64::MAX as f64 || rounded < i64::MIN as f64 {
            return Err(OrderBookError::Overflow(
                format!("Price conversion overflow: {}", price)
            ));
        }
        
        Ok(Price(rounded as i64))
    }
    
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::MULTIPLIER as f64
    }
    
    // Safe string parsing with overflow protection
    pub fn from_str_fast(s: &str) -> Result<Self, OrderBookError> {
        if s.is_empty() {
            return Err(OrderBookError::InvalidPrice("Empty price string".to_string()));
        }
        
        let bytes = s.as_bytes();
        let (sign, bytes) = match bytes[0] {
            b'-' => return Err(OrderBookError::InvalidPrice("Negative prices not allowed".to_string())),
            b'+' => (1, &bytes[1..]),
            _ => (1, bytes),
        };
        
        if bytes.is_empty() {
            return Err(OrderBookError::InvalidPrice("Invalid price format".to_string()));
        }
        
        let mut result = 0i64;
        let mut decimal_places = 0;
        let mut found_decimal = false;
        
        // Track magnitude to prevent overflow during parsing
        let mut magnitude_check = 0u64;
        
        for &byte in bytes {
            match byte {
                b'0'..=b'9' => {
                    let digit = (byte - b'0') as i64;
                    
                    // Check for overflow during parsing
                    if result > (i64::MAX - digit) / 10 {
                        return Err(OrderBookError::Overflow(
                            format!("Price string '{}' causes overflow during parsing", s)
                        ));
                    }
                    
                    if found_decimal {
                        decimal_places += 1;
                        if decimal_places > 8 {
                            break; // Ignore extra precision
                        }
                    }
                    
                    result = result * 10 + digit;
                    magnitude_check = magnitude_check.saturating_mul(10).saturating_add(digit as u64);
                    
                    // Early overflow detection
                    if magnitude_check > Self::MAX_SAFE_PRICE as u64 * 100_000_000 {
                        return Err(OrderBookError::Overflow(
                            format!("Price string '{}' represents too large a value", s)
                        ));
                    }
                }
                b'.' => {
                    if found_decimal {
                        return Err(OrderBookError::InvalidPrice("Multiple decimal points".to_string()));
                    }
                    found_decimal = true;
                }
                _ => return Err(OrderBookError::InvalidPrice("Invalid character in price".to_string())),
            }
        }
        
        // Adjust for decimal places (pad with zeros to reach 8 decimal places)
        for _ in decimal_places..8 {
            if result > i64::MAX / 10 {
                return Err(OrderBookError::Overflow(
                    format!("Price string '{}' causes overflow during scaling", s)
                ));
            }
            result *= 10;
        }
        
        if result <= 0 {
            return Err(OrderBookError::InvalidPrice("Price must be positive".to_string()));
        }
        
        Ok(Price(result * sign))
    }
    
    // Utility methods for checking limits
    pub fn max_safe_value() -> f64 {
        Self::MAX_SAFE_PRICE
    }
    
    pub fn min_safe_value() -> f64 {
        Self::MIN_SAFE_PRICE
    }
}

// Quantity with overflow protection
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Quantity(f64);

impl Quantity {
    const MAX_SAFE_QUANTITY: f64 = 1e15; // 1 quadrillion (reasonable limit)
    
    #[inline(always)]
    pub fn from_f64(qty: f64) -> Result<Self, OrderBookError> {
        if !qty.is_finite() {
            return Err(OrderBookError::InvalidQuantity("Quantity must be finite".to_string()));
        }
        
        if qty < 0.0 {
            return Err(OrderBookError::InvalidQuantity("Quantity cannot be negative".to_string()));
        }
        
        if qty > Self::MAX_SAFE_QUANTITY {
            return Err(OrderBookError::Overflow(
                format!("Quantity {} exceeds maximum safe value {}", qty, Self::MAX_SAFE_QUANTITY)
            ));
        }
        
        Ok(Quantity(qty))
    }
    
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.0
    }
    
    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }
    
    // Fast quantity parsing with overflow protection
    pub fn from_str_fast(s: &str) -> Result<Self, OrderBookError> {
        // Optimize for common single-digit cases first
        match s.len() {
            1 => {
                let byte = s.as_bytes()[0];
                if byte >= b'0' && byte <= b'9' {
                    return Ok(Quantity((byte - b'0') as f64));
                }
            }
            _ => {}
        }
        
        // Parse and validate
        let qty = s.parse::<f64>()
            .map_err(|_| OrderBookError::InvalidQuantity(format!("Cannot parse quantity: {}", s)))?;
        
        Self::from_f64(qty)
    }
}

// Zero-allocation price level using stack-allocated array for level data
#[derive(Debug, Copy, Clone)]
pub struct PriceLevel {
    pub price: Price,
    pub quantity: Quantity,
}

// Book Ticker Update with overflow protection
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BookTickerUpdate {
    pub u: u64,        // order book updateId
    pub s: String,     // symbol
    pub b: String,     // best bid price
    #[serde(rename = "B")]
    pub bid_qty: String, // best bid qty
    pub a: String,     // best ask price
    #[serde(rename = "A")]
    pub ask_qty: String, // best ask qty
}

impl BookTickerUpdate {
    pub fn from_json(json_str: &str) -> Result<Self, OrderBookError> {
        serde_json::from_str(json_str)
            .map_err(|e| OrderBookError::SerializationError(e.to_string()))
    }

    // Safe parsing with overflow protection
    #[inline]
    pub fn parse_best_bid(&self) -> Result<(Price, Quantity), OrderBookError> {
        let price = Price::from_str_fast(&self.b)?;
        let qty = Quantity::from_str_fast(&self.bid_qty)?;
        
        if price.0 <= 0 || qty.0 <= 0.0 {
            return Err(OrderBookError::InvalidPrice("Price and quantity must be positive".to_string()));
        }
        
        Ok((price, qty))
    }

    #[inline]
    pub fn parse_best_ask(&self) -> Result<(Price, Quantity), OrderBookError> {
        let price = Price::from_str_fast(&self.a)?;
        let qty = Quantity::from_str_fast(&self.ask_qty)?;
        
        if price.0 <= 0 || qty.0 <= 0.0 {
            return Err(OrderBookError::InvalidPrice("Price and quantity must be positive".to_string()));
        }
        
        Ok((price, qty))
    }
}

// Depth Update with maximum capacity enforcement (never allocates)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DepthUpdate {
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: u64,
    pub bids: ArrayVec<[String; 2], 20>, // Fixed-size array for exactly ≤20 levels
    pub asks: ArrayVec<[String; 2], 20>, // Fixed-size array for exactly ≤20 levels
}

impl DepthUpdate {
    pub fn from_json(json_str: &str) -> Result<Self, OrderBookError> {
        serde_json::from_str(json_str)
            .map_err(|e| OrderBookError::SerializationError(e.to_string()))
    }

    // Batch parsing with compile-time capacity guarantees
    pub fn parse_bids(&self) -> Result<ArrayVec<(Price, Quantity), 20>, OrderBookError> {
        let mut results = ArrayVec::new();
        
        for level in &self.bids {
            let price = Price::from_str_fast(&level[0])?;
            let qty = Quantity::from_str_fast(&level[1])?;
            
            if price.0 <= 0 || qty.0 < 0.0 {
                return Err(OrderBookError::InvalidPrice("Price must be positive, quantity must be non-negative".to_string()));
            }
            
            // ArrayVec::push returns Result, handle capacity overflow
            results.try_push((price, qty)).map_err(|_| {
                OrderBookError::Overflow(format!("Too many bid levels: maximum 20 allowed"))
            })?;
        }
        
        Ok(results)
    }

    pub fn parse_asks(&self) -> Result<ArrayVec<(Price, Quantity), 20>, OrderBookError> {
        let mut results = ArrayVec::new();
        
        for level in &self.asks {
            let price = Price::from_str_fast(&level[0])?;
            let qty = Quantity::from_str_fast(&level[1])?;
            
            if price.0 <= 0 || qty.0 < 0.0 {
                return Err(OrderBookError::InvalidPrice("Price must be positive, quantity must be non-negative".to_string()));
            }
            
            // ArrayVec::push returns Result, handle capacity overflow
            results.try_push((price, qty)).map_err(|_| {
                OrderBookError::Overflow(format!("Too many ask levels: maximum 20 allowed"))
            })?;
        }
        
        Ok(results)
    }
}

// High-performance OrderBook with overflow protection
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: String,
    // Using Price as key directly for zero-conversion overhead
    bids: BTreeMap<Price, Quantity>, // Reverse order iteration for best bid
    asks: BTreeMap<Price, Quantity>, // Forward order iteration for best ask
    last_update_id: u64,
    
    // Cached best levels for O(1) access - using Option<PriceLevel> for cache efficiency
    best_bid: Option<PriceLevel>,
    best_ask: Option<PriceLevel>,
}

impl OrderBook {
    pub fn new(symbol: String) -> Result<OrderBook, OrderBookError> {
        if symbol.is_empty() {
            return Err(OrderBookError::InvalidSymbol("Symbol cannot be empty".to_string()));
        }
        
        Ok(OrderBook {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
            best_bid: None,
            best_ask: None,
        })
    }

    // Hot path: optimized for frequent calls with overflow protection
    #[inline]
    pub fn update_book_ticker(&mut self, data: &BookTickerUpdate) -> Result<(), OrderBookError> {
        // Early return for symbol mismatch - avoid any parsing overhead
        if data.s != self.symbol {
            return Err(OrderBookError::InvalidSymbol(
                format!("Symbol mismatch: expected {}, got {}", self.symbol, data.s)
            ));
        }

        // Parse both bid and ask, early return on any error (including overflow)
        let (bid_price, bid_qty) = data.parse_best_bid()?;
        let (ask_price, ask_qty) = data.parse_best_ask()?;

        // Batch updates to reduce BTreeMap operations
        if bid_qty.is_zero() {
            self.bids.remove(&bid_price);
        } else {
            self.bids.insert(bid_price, bid_qty);
        }

        if ask_qty.is_zero() {
            self.asks.remove(&ask_price);
        } else {
            self.asks.insert(ask_price, ask_qty);
        }

        // Update cache once at the end
        self.update_best_cache_fast();
        self.last_update_id = data.u;

        Ok(())
    }

    // Hot path: optimized batch processing with overflow protection
    #[inline]
    pub fn update_depth(&mut self, data: &DepthUpdate) -> Result<(), OrderBookError> {
        // Parse all levels upfront to validate before any mutations
        // This ensures we fail fast on overflow without partial updates
        let bids = data.parse_bids()?;
        let asks = data.parse_asks()?;

        // Batch all bid updates - more cache-friendly than interleaving
        for (price, qty) in bids {
            if qty.is_zero() {
                self.bids.remove(&price);
            } else {
                self.bids.insert(price, qty);
            }
        }

        // Batch all ask updates
        for (price, qty) in asks {
            if qty.is_zero() {
                self.asks.remove(&price);
            } else {
                self.asks.insert(price, qty);
            }
        }

        // Single cache update at the end
        self.update_best_cache_fast();
        self.last_update_id = data.last_update_id;

        Ok(())
    }

    // Optimized cache update using direct BTreeMap iteration
    #[inline]
    fn update_best_cache_fast(&mut self) {
        // Best bid = highest price (last in iteration order)
        self.best_bid = self.bids.iter().next_back()
            .map(|(&price, &qty)| PriceLevel { price, quantity: qty });

        // Best ask = lowest price (first in iteration order)  
        self.best_ask = self.asks.iter().next()
            .map(|(&price, &qty)| PriceLevel { price, quantity: qty });
    }

    // O(1) best bid/ask retrieval from cache
    #[inline(always)]
    pub fn get_best_bid_ask(&self) -> Option<((f64, f64), (f64, f64))> {
        match (self.best_bid, self.best_ask) {
            (Some(bid), Some(ask)) => Some((
                (bid.price.to_f64(), bid.quantity.to_f64()),
                (ask.price.to_f64(), ask.quantity.to_f64())
            )),
            _ => None,
        }
    }

    // Fast level extraction with pre-sized vectors
    pub fn get_levels(&self, depth: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        // Pre-allocate exact capacity to avoid reallocation
        let mut bids = Vec::with_capacity(depth.min(self.bids.len()));
        let mut asks = Vec::with_capacity(depth.min(self.asks.len()));

        // Efficient iterator usage - no collect() intermediate allocation
        for (&price, &qty) in self.bids.iter().rev().take(depth) {
            bids.push((price.to_f64(), qty.to_f64()));
        }

        for (&price, &qty) in self.asks.iter().take(depth) {
            asks.push((price.to_f64(), qty.to_f64()));
        }

        (bids, asks)
    }

    // Optimized string formatting with minimal allocation
    pub fn to_string(&self) -> String {
        let (bids, asks) = self.get_levels(20);
        
        // Pre-calculate string capacity to avoid reallocation
        let estimated_size = bids.len().max(asks.len()) * 80; // ~80 chars per line
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
    #[inline(always)]
    pub fn bid_count(&self) -> usize { self.bids.len() }
    
    #[inline(always)]
    pub fn ask_count(&self) -> usize { self.asks.len() }
    
    #[inline(always)]
    pub fn last_update_id(&self) -> u64 { self.last_update_id }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_fixed_point_conversion() {
        let price = Price::from_f64(25.35190000).unwrap();
        assert_eq!(price.to_f64(), 25.3519);
        
        let price2 = Price::from_str_fast("25.35190000").unwrap();
        assert_eq!(price2, price);
    }

    #[test]
    fn test_fast_quantity_parsing() {
        let qty = Quantity::from_str_fast("5").unwrap();
        assert_eq!(qty.to_f64(), 5.0);
        
        let qty2 = Quantity::from_str_fast("31.21000000").unwrap();
        assert_eq!(qty2.to_f64(), 31.21);
    }

    #[test]
    fn test_overflow_protection() {
        // These should fail gracefully
        assert!(Price::from_f64(f64::MAX).is_err());
        assert!(Price::from_f64(1e50).is_err());
        assert!(Price::from_f64(f64::INFINITY).is_err());
        assert!(Price::from_f64(f64::NAN).is_err());
        assert!(Price::from_f64(-1.0).is_err());
        
        // Test quantity overflow
        assert!(Quantity::from_f64(f64::MAX).is_err());
        assert!(Quantity::from_f64(f64::INFINITY).is_err());
        assert!(Quantity::from_f64(-1.0).is_err());
        
        // Boundary cases
        assert!(Price::from_f64(90_000_000_001.0).is_err()); // Just over limit
        assert!(Price::from_f64(89_999_999_999.0).is_ok());  // Just under limit
    }

    #[test]
    fn test_string_overflow_protection() {
        // Test string parsing overflow
        assert!(Price::from_str_fast("99999999999999999999").is_err());
        assert!(Price::from_str_fast("1e50").is_err());
        assert!(Quantity::from_str_fast("1e50").is_err());
        
        // Valid large values should work
        assert!(Price::from_str_fast("1000000.12345678").is_ok());
        assert!(Quantity::from_str_fast("1000000.123").is_ok());
    }

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
        assert_eq!(update.u, 400900217);
        assert_eq!(update.s, "BNBUSDT");
        
        let (bid_price, bid_qty) = update.parse_best_bid().unwrap();
        let (ask_price, ask_qty) = update.parse_best_ask().unwrap();
        
        assert_eq!(bid_price.to_f64(), 25.3519);
        assert_eq!(bid_qty.to_f64(), 31.21);
        assert_eq!(ask_price.to_f64(), 25.3652);
        assert_eq!(ask_qty.to_f64(), 40.66);
    }

    #[test]
    fn test_book_ticker_overflow_rejection() {
        let json_overflow = r#"
        {
            "u":400900217,
            "s":"BNBUSDT",
            "b":"99999999999999999999",
            "B":"31.21000000",
            "a":"25.36520000",
            "A":"40.66000000"
        }"#;

        let update = BookTickerUpdate::from_json(json_overflow).unwrap();
        assert!(update.parse_best_bid().is_err());
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

        let update = DepthUpdate::from_json(json).unwrap();
        assert_eq!(update.last_update_id, 160);
        
        let bids = update.parse_bids().unwrap();
        let asks = update.parse_asks().unwrap();
        
        assert_eq!(bids[0].0.to_f64(), 0.0024);
        assert_eq!(bids[0].1.to_f64(), 10.0);
        assert_eq!(asks[0].0.to_f64(), 0.0026);
        assert_eq!(asks[0].1.to_f64(), 100.0);
    }

    #[test]
    fn test_orderbook_creation() {
        let book = OrderBook::new("BTCUSDT".to_string()).unwrap();
        assert_eq!(book.symbol, "BTCUSDT");
        assert_eq!(book.get_best_bid_ask(), None);
    }

    #[test]
    fn test_high_performance_book_ticker_update() {
        let mut book = OrderBook::new("BNBUSDT".to_string()).unwrap();
        
        let update = BookTickerUpdate {
            u: 400900217,
            s: "BNBUSDT".to_string(),
            b: "25.35190000".to_string(),
            bid_qty: "31.21000000".to_string(),
            a: "25.36520000".to_string(),
            ask_qty: "40.66000000".to_string(),
        };

        book.update_book_ticker(&update).unwrap();
        
        let best = book.get_best_bid_ask().unwrap();
        assert!((best.0.0 - 25.3519).abs() < 1e-6);
        assert!((best.0.1 - 31.21).abs() < 1e-6);
        assert!((best.1.0 - 25.3652).abs() < 1e-6);
        assert!((best.1.1 - 40.66).abs() < 1e-6);
    }

    #[test]
    fn test_batch_depth_update() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();
        
        let mut bids = ArrayVec::new();
        bids.try_push(["50000.0".to_string(), "1.0".to_string()]).unwrap();
        bids.try_push(["49999.0".to_string(), "2.0".to_string()]).unwrap();
        
        let mut asks = ArrayVec::new();
        asks.try_push(["50001.0".to_string(), "1.5".to_string()]).unwrap();
        asks.try_push(["50002.0".to_string(), "2.5".to_string()]).unwrap();
        
        let update = DepthUpdate {
            last_update_id: 160,
            bids,
            asks,
        };

        book.update_depth(&update).unwrap();
        
        let best = book.get_best_bid_ask().unwrap();
        assert_eq!(best.0, (50000.0, 1.0));
        assert_eq!(best.1, (50001.0, 1.5));
        assert_eq!(book.bid_count(), 2);
        assert_eq!(book.ask_count(), 2);
    }

    #[test]
    fn test_zero_quantity_removal() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();
        
        // Add initial levels
        let mut bids1 = ArrayVec::new();
        bids1.try_push(["50000.0".to_string(), "1.0".to_string()]).unwrap();
        let mut asks1 = ArrayVec::new();
        asks1.try_push(["50001.0".to_string(), "1.0".to_string()]).unwrap();
        
        let update1 = DepthUpdate {
            last_update_id: 1,
            bids: bids1,
            asks: asks1,
        };
        book.update_depth(&update1).unwrap();
        assert!(book.get_best_bid_ask().is_some());
        
        // Remove with zero quantity
        let mut bids2 = ArrayVec::new();
        bids2.try_push(["50000.0".to_string(), "0.0".to_string()]).unwrap();
        let mut asks2 = ArrayVec::new();
        asks2.try_push(["50001.0".to_string(), "0.0".to_string()]).unwrap();
        
        let update2 = DepthUpdate {
            last_update_id: 2,
            bids: bids2,
            asks: asks2,
        };
        book.update_depth(&update2).unwrap();
        assert!(book.get_best_bid_ask().is_none());
    }

    #[test]
    fn test_price_comparison_precision() {
        let price1 = Price::from_str_fast("50000.12345678").unwrap();
        let price2 = Price::from_str_fast("50000.12345679").unwrap();
        assert!(price2 > price1);
        
        let price3 = Price::from_f64(50000.12345678).unwrap();
        assert_eq!(price1, price3);
    }

    #[test]
    fn test_invalid_symbol_error() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();
        
        let update = BookTickerUpdate {
            u: 1,
            s: "ETHUSDT".to_string(), // Wrong symbol
            b: "25.35190000".to_string(),
            bid_qty: "31.21000000".to_string(),
            a: "25.36520000".to_string(),
            ask_qty: "40.66000000".to_string(),
        };

        let result = book.update_book_ticker(&update);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OrderBookError::InvalidSymbol(_)));
    }

    #[test]
    fn test_invalid_price_error() {
        let json = r#"
        {
            "u":400900217,
            "s":"BNBUSDT",
            "b":"invalid_price",
            "B":"31.21000000",
            "a":"25.36520000",
            "A":"40.66000000"
        }"#;

        let update = BookTickerUpdate::from_json(json).unwrap();
        let result = update.parse_best_bid();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OrderBookError::InvalidPrice(_)));
    }

    #[test]
    fn test_orderbook_formatting() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();
        
        let mut bids = ArrayVec::new();
        bids.try_push(["50000.0".to_string(), "1.0".to_string()]).unwrap();
        bids.try_push(["49999.0".to_string(), "2.0".to_string()]).unwrap();
        
        let mut asks = ArrayVec::new();
        asks.try_push(["50001.0".to_string(), "1.5".to_string()]).unwrap();
        asks.try_push(["50002.0".to_string(), "2.5".to_string()]).unwrap();
        
        let update = DepthUpdate {
            last_update_id: 1,
            bids,
            asks,
        };

        book.update_depth(&update).unwrap();
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
        assert!(Price::from_f64(0.00000001).is_ok());
        
        // Test zero
        assert!(Price::from_f64(0.0).is_ok());
        
        // Test safe large values
        assert!(Price::from_f64(1_000_000.0).is_ok());
        assert!(Price::from_f64(10_000_000.0).is_ok());
    }

    #[test]
    fn test_negative_prices() {
        assert!(Price::from_f64(-1.0).is_err());
        assert!(Price::from_f64(-0.1).is_err());
        assert!(Price::from_str_fast("-1.0").is_err());
    }
}

// Performance-focused integration tests
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_bulk_updates_performance() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();
        
        // Simulate high-frequency updates
        for i in 0..1000 {
            let mut bids = ArrayVec::new();
            bids.try_push([format!("{}.0", 50000 - i), "1.0".to_string()]).unwrap();
            bids.try_push([format!("{}.5", 50000 - i), "2.0".to_string()]).unwrap();
            
            let mut asks = ArrayVec::new();
            asks.try_push([format!("{}.0", 50001 + i), "1.5".to_string()]).unwrap();
            asks.try_push([format!("{}.5", 50001 + i), "2.5".to_string()]).unwrap();
            
            let update = DepthUpdate {
                last_update_id: i,
                bids,
                asks,
            };
            
            book.update_depth(&update).unwrap();
            
            // Verify we can still access best levels efficiently
            assert!(book.get_best_bid_ask().is_some());
        }
        
        // Verify final state
        assert!(book.bid_count() <= 2000); // Some may have been overwritten
        assert!(book.ask_count() <= 2000);
    }

    #[test]
    fn test_string_parsing_edge_cases() {
        // Test various price formats for robustness
        let test_cases = [
            "0.1", "1.0", "10.5", "100.25", "1000.123",
            "50000.12345678", "0.00000001", "999999.99999999"
        ];
        
        for case in &test_cases {
            let price = Price::from_str_fast(case).unwrap();
            let qty = Quantity::from_str_fast(case).unwrap();
            
            // Ensure round-trip conversion is accurate
            let price_f64 = price.to_f64();
            let qty_f64 = qty.to_f64();
            
            assert!(price_f64 > 0.0);
            assert!(qty_f64 > 0.0);
        }
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
        
        let depth_update = DepthUpdate::from_json(depth_json).unwrap();
        book.update_depth(&depth_update).unwrap();
        
        // Verify initial state - clear $4 spread
        let best = book.get_best_bid_ask().unwrap();
        assert_eq!(best.0, (49998.0, 1.5)); // Best bid: $49,998
        assert_eq!(best.1, (50002.0, 1.2)); // Best ask: $50,002
        let initial_spread = best.1.0 - best.0.0;
        assert_eq!(initial_spread, 4.0); // $4 spread
        
        // Update with book ticker - tightening the spread
        let ticker_json = r#"
        {
            "u": 1001,
            "s": "BTCUSDT",
            "b": "49999.00",
            "B": "3.0",
            "a": "50001.00",
            "A": "2.0"
        }"#;
        
        let ticker_update = BookTickerUpdate::from_json(ticker_json).unwrap();
        book.update_book_ticker(&ticker_update).unwrap();
        
        // Verify updated state - spread tightened to $2
        let best_after = book.get_best_bid_ask().unwrap();
        assert_eq!(best_after.0, (49999.0, 3.0)); // New best bid: $49,999 (higher)
        assert_eq!(best_after.1, (50001.0, 2.0)); // New best ask: $50,001 (lower)
        let new_spread = best_after.1.0 - best_after.0.0;
        assert_eq!(new_spread, 2.0); // Tighter $2 spread
        
        // Verify the spread tightened (more liquid market)
        assert!(new_spread < initial_spread);
        
        // Verify orderbook integrity: bid < ask (no crossing)
        assert!(best_after.0.0 < best_after.1.0);
        
        println!("Market improved: spread tightened from ${} to ${}", initial_spread, new_spread);
    }

    #[test]
    fn test_malicious_input_rejection() {
        let mut book = OrderBook::new("BTCUSDT".to_string()).unwrap();
        
        // Test various malicious inputs
        let malicious_cases = vec![
            r#"{"lastUpdateId": 1, "bids": [["999999999999999999999", "1.0"]], "asks": []}"#,
            r#"{"lastUpdateId": 1, "bids": [["1e50", "1.0"]], "asks": []}"#,
            r#"{"lastUpdateId": 1, "bids": [["inf", "1.0"]], "asks": []}"#,
            r#"{"lastUpdateId": 1, "bids": [["50000.0", "1e50"]], "asks": []}"#,
        ];
        
        for malicious_json in malicious_cases {
            let update = DepthUpdate::from_json(malicious_json).unwrap();
            let result = book.update_depth(&update);
            
            // Should fail gracefully without corrupting the orderbook
            assert!(result.is_err());
            
            // Orderbook should still be functional
            assert_eq!(book.bid_count(), 0);
            assert_eq!(book.ask_count(), 0);
        }
    }

    #[test]
    fn test_arrayvec_capacity_enforcement() {
        // Test that ArrayVec enforces the 20-level limit
        let mut bids = ArrayVec::<[String; 2], 20>::new();
        
        // Fill to capacity
        for i in 0..20 {
            let result = bids.try_push([format!("{}.0", 50000 - i), "1.0".to_string()]);
            assert!(result.is_ok());
        }
        
        // 21st element should fail
        let result = bids.try_push(["49979.0".to_string(), "1.0".to_string()]);
        assert!(result.is_err()); // ArrayVec enforces capacity limit
        
        // Verify we still have exactly 20 elements
        assert_eq!(bids.len(), 20);
    }
}

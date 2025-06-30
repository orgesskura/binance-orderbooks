mod orderbooks;
use crate::orderbooks::OrderBook;

use std::time::Duration;
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use std::time::Instant;

#[derive(Debug)]
pub enum BinanceError {
    ConnectionError(String),
    UrlError(String),
}

impl std::fmt::Display for BinanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinanceError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            BinanceError::UrlError(msg) => write!(f, "URL error: {}", msg),
        }
    }
}

impl std::error::Error for BinanceError {}

#[derive(Debug, Clone)]
pub enum DepthLevel {
    Five = 5,
    Ten = 10,
    Twenty = 20,
}

#[derive(Debug, Clone)]
pub enum UpdateSpeed {
    Standard,  // 1000ms
    Fast,      // 100ms
}

pub struct BinanceSingleStreamClient {
    base_url: String,
}

impl BinanceSingleStreamClient {
    pub fn new() -> Self {
        Self {
            base_url: "wss://stream.binance.com:9443".to_string(),
        }
    }

    pub fn with_data_stream_url(mut self) -> Self {
        self.base_url = "wss://data-stream.binance.vision".to_string();
        self
    }

    fn build_stream_name(&self, symbol: &str, level: DepthLevel, speed: UpdateSpeed) -> String {
        let level_str = match level {
            DepthLevel::Five => "5",
            DepthLevel::Ten => "10",
            DepthLevel::Twenty => "20",
        };

        let speed_suffix = match speed {
            UpdateSpeed::Standard => "@1000ms",
            UpdateSpeed::Fast => "@100ms",
        };

        format!("{}@depth{}{}", symbol.to_lowercase(), level_str, speed_suffix)
    }

    fn build_url(&self, stream_name: &str) -> Result<String, BinanceError> {
        let url = format!("{}/ws/{}", self.base_url, stream_name);

        // Validate URL
        url::Url::parse(&url).map_err(|e| BinanceError::UrlError(e.to_string()))?;

        Ok(url)
    }

    pub async fn connect<F>(
        &self,
        symbol: &str,
        level: DepthLevel,
        speed: UpdateSpeed,
        mut message_handler: F,
    ) -> Result<(), BinanceError>
    where
        F: FnMut(String) + Send + 'static,
    {
        // Build stream name and URL
        let stream_name = self.build_stream_name(symbol, level, speed);
        let url = self.build_url(&stream_name)?;

        println!("Connecting to: {}", url);
        println!("Stream: {}", stream_name);

        // Connect to WebSocket
        let (ws_stream, _) = connect_async(&url).await
            .map_err(|e| BinanceError::ConnectionError(e.to_string()))?;

        let (mut write, mut read) = ws_stream.split();

        // Set up ping interval for heartbeat
        let mut ping_interval = interval(Duration::from_secs(20));

        println!("Connected! Listening for messages...");

        // Main message loop
        loop {
            tokio::select! {
                // Handle incoming messages
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            message_handler(text);
                        },
                        Some(Ok(Message::Ping(payload))) => {
                            // Respond to server ping with pong
                            if let Err(e) = write.send(Message::Pong(payload)).await {
                                eprintln!("Failed to send pong: {}", e);
                                break;
                            }
                        },
                        Some(Ok(Message::Close(_))) => {
                            println!("WebSocket connection closed by server");
                            break;
                        },
                        Some(Err(e)) => {
                            eprintln!("WebSocket error: {}", e);
                            return Err(BinanceError::ConnectionError(e.to_string()));
                        },
                        None => {
                            println!("WebSocket stream ended");
                            break;
                        },
                        _ => {} // Handle other message types
                    }
                },
                // Send periodic ping
                _ = ping_interval.tick() => {
                    if let Err(e) = write.send(Message::Ping(vec![])).await {
                        eprintln!("Failed to send ping: {}", e);
                        break;
                    }
                }
            }
        }

        Ok(())
    }
}

// Example usage
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Connecting to BTCUSDT depth stream...");
    let mut last_print = Instant::now();


    let client = BinanceSingleStreamClient::new();
    let orderbook = OrderBook::new("BTCUSDT".to_string());

    client.connect(
        "BTCUSDT",
        DepthLevel::Twenty,     // 20 levels
        UpdateSpeed::Fast,    // 100ms updates
        |message| {

            if last_print.elapsed().as_secs() >= 10 {
            println!("10 seconds passed. Processed {} messages", message_count);
            last_print = Instant::now();
            message_count = 0; // Reset counter
        }
        }
    ).await?;

    Ok(())
}

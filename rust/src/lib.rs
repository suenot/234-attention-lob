use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;

// ─── LOB Data Structures ───────────────────────────────────────────

/// A single snapshot of the limit order book.
#[derive(Debug, Clone)]
pub struct LobSnapshot {
    pub bid_prices: Vec<f64>,
    pub bid_volumes: Vec<f64>,
    pub ask_prices: Vec<f64>,
    pub ask_volumes: Vec<f64>,
    pub timestamp: u64,
}

impl LobSnapshot {
    /// Compute the mid-price.
    pub fn mid_price(&self) -> f64 {
        if self.bid_prices.is_empty() || self.ask_prices.is_empty() {
            return 0.0;
        }
        (self.bid_prices[0] + self.ask_prices[0]) / 2.0
    }

    /// Compute the bid-ask spread.
    pub fn spread(&self) -> f64 {
        if self.bid_prices.is_empty() || self.ask_prices.is_empty() {
            return 0.0;
        }
        self.ask_prices[0] - self.bid_prices[0]
    }

    /// Extract feature matrix: rows = levels (bids then asks), cols = features.
    /// Features per level: [price, volume, distance_from_mid, cumulative_volume, side_indicator]
    pub fn to_feature_matrix(&self) -> Array2<f64> {
        let mid = self.mid_price();
        let n_bid = self.bid_prices.len();
        let n_ask = self.ask_prices.len();
        let total_levels = n_bid + n_ask;
        let n_features = 5;

        let mut matrix = Array2::zeros((total_levels, n_features));

        // Bid side
        let mut cum_vol = 0.0;
        for i in 0..n_bid {
            cum_vol += self.bid_volumes[i];
            matrix[[i, 0]] = self.bid_prices[i];
            matrix[[i, 1]] = self.bid_volumes[i];
            matrix[[i, 2]] = (self.bid_prices[i] - mid).abs();
            matrix[[i, 3]] = cum_vol;
            matrix[[i, 4]] = 1.0; // bid indicator
        }

        // Ask side
        cum_vol = 0.0;
        for i in 0..n_ask {
            cum_vol += self.ask_volumes[i];
            matrix[[n_bid + i, 0]] = self.ask_prices[i];
            matrix[[n_bid + i, 1]] = self.ask_volumes[i];
            matrix[[n_bid + i, 2]] = (self.ask_prices[i] - mid).abs();
            matrix[[n_bid + i, 3]] = cum_vol;
            matrix[[n_bid + i, 4]] = -1.0; // ask indicator
        }

        matrix
    }

    /// Compute order flow imbalance (volume imbalance at top levels).
    pub fn volume_imbalance(&self, levels: usize) -> f64 {
        let bid_vol: f64 = self.bid_volumes.iter().take(levels).sum();
        let ask_vol: f64 = self.ask_volumes.iter().take(levels).sum();
        if bid_vol + ask_vol == 0.0 {
            return 0.0;
        }
        (bid_vol - ask_vol) / (bid_vol + ask_vol)
    }
}

// ─── Attention Mechanisms ──────────────────────────────────────────

/// Softmax over a 1D array.
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum = exp_x.sum();
    if sum == 0.0 {
        Array1::ones(x.len()) / x.len() as f64
    } else {
        exp_x / sum
    }
}

/// Softmax over rows of a 2D array (each row independently).
fn softmax_rows(x: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros(x.raw_dim());
    for (i, row) in x.axis_iter(Axis(0)).enumerate() {
        let s = softmax(&row.to_owned());
        for (j, val) in s.iter().enumerate() {
            result[[i, j]] = *val;
        }
    }
    result
}

/// Initialize a random weight matrix with Xavier initialization.
fn xavier_init(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
}

/// Multi-head self-attention over price levels.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub w_o: Array2<f64>,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention module.
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;
        Self {
            num_heads,
            d_model,
            d_k,
            w_q: xavier_init(d_model, d_model),
            w_k: xavier_init(d_model, d_model),
            w_v: xavier_init(d_model, d_model),
            w_o: xavier_init(d_model, d_model),
        }
    }

    /// Compute scaled dot-product attention for a single head.
    /// Returns (output, attention_weights).
    pub fn scaled_dot_product_attention(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let d_k = q.ncols() as f64;
        let scores = q.dot(&k.t()) / d_k.sqrt();
        let weights = softmax_rows(&scores);
        let output = weights.dot(v);
        (output, weights)
    }

    /// Forward pass: apply multi-head self-attention.
    /// Input shape: (seq_len, d_model)
    /// Returns (output, attention_weights_per_head).
    pub fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, Vec<Array2<f64>>) {
        let seq_len = x.nrows();
        let q_full = x.dot(&self.w_q);
        let k_full = x.dot(&self.w_k);
        let v_full = x.dot(&self.w_v);

        let mut head_outputs = Vec::new();
        let mut all_weights = Vec::new();

        for h in 0..self.num_heads {
            let start = h * self.d_k;
            let end = start + self.d_k;

            let q_h = q_full.slice(ndarray::s![.., start..end]).to_owned();
            let k_h = k_full.slice(ndarray::s![.., start..end]).to_owned();
            let v_h = v_full.slice(ndarray::s![.., start..end]).to_owned();

            let (out, weights) = self.scaled_dot_product_attention(&q_h, &k_h, &v_h);
            head_outputs.push(out);
            all_weights.push(weights);
        }

        // Concatenate heads
        let mut concat = Array2::zeros((seq_len, self.d_model));
        for (h, head_out) in head_outputs.iter().enumerate() {
            let start = h * self.d_k;
            for i in 0..seq_len {
                for j in 0..self.d_k {
                    concat[[i, start + j]] = head_out[[i, j]];
                }
            }
        }

        let output = concat.dot(&self.w_o);
        (output, all_weights)
    }
}

/// Temporal attention over a sequence of LOB snapshot embeddings.
#[derive(Debug, Clone)]
pub struct TemporalAttention {
    pub d_model: usize,
    pub w_h: Array2<f64>,
    pub v: Array1<f64>,
    pub bias: Array1<f64>,
}

impl TemporalAttention {
    /// Create a new temporal attention module (Bahdanau-style).
    pub fn new(d_model: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / d_model as f64).sqrt();
        Self {
            d_model,
            w_h: xavier_init(d_model, d_model),
            v: Array1::from_shape_fn(d_model, |_| rng.gen_range(-scale..scale)),
            bias: Array1::zeros(d_model),
        }
    }

    /// Forward pass: compute weighted sum of sequence.
    /// Input: list of embeddings, each of shape (d_model,).
    /// Returns (context_vector, attention_weights).
    pub fn forward(&self, sequence: &[Array1<f64>]) -> (Array1<f64>, Array1<f64>) {
        let t = sequence.len();
        let mut energies = Array1::zeros(t);

        for (i, h) in sequence.iter().enumerate() {
            let projected = self.w_h.dot(h) + &self.bias;
            let activated = projected.mapv(|x| x.tanh());
            energies[i] = self.v.dot(&activated);
        }

        let weights = softmax(&energies);

        let mut context = Array1::zeros(self.d_model);
        for (i, h) in sequence.iter().enumerate() {
            context = context + &(h * weights[i]);
        }

        (context, weights)
    }
}

/// Cross-attention between bid and ask sides.
#[derive(Debug, Clone)]
pub struct CrossAttention {
    pub d_model: usize,
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
}

impl CrossAttention {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            w_q: xavier_init(d_model, d_model),
            w_k: xavier_init(d_model, d_model),
            w_v: xavier_init(d_model, d_model),
        }
    }

    /// Forward pass: query attends to key-value pairs.
    /// query_side: (n_levels, d_model), kv_side: (n_levels, d_model)
    /// Returns (output, attention_weights).
    pub fn forward(
        &self,
        query_side: &Array2<f64>,
        kv_side: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let q = query_side.dot(&self.w_q);
        let k = kv_side.dot(&self.w_k);
        let v = kv_side.dot(&self.w_v);

        let d_k = self.d_model as f64;
        let scores = q.dot(&k.t()) / d_k.sqrt();
        let weights = softmax_rows(&scores);
        let output = weights.dot(&v);

        (output, weights)
    }
}

// ─── Prediction Head ───────────────────────────────────────────────

/// Mid-price direction prediction: Up, Down, Stationary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PriceDirection {
    Up,
    Down,
    Stationary,
}

/// Linear prediction head for mid-price direction.
#[derive(Debug, Clone)]
pub struct PredictionHead {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl PredictionHead {
    pub fn new(input_dim: usize, n_classes: usize) -> Self {
        Self {
            weights: xavier_init(input_dim, n_classes),
            bias: Array1::zeros(n_classes),
        }
    }

    /// Predict class probabilities from a feature vector.
    pub fn predict(&self, features: &Array1<f64>) -> (PriceDirection, Array1<f64>) {
        let logits = features.dot(&self.weights) + &self.bias;
        let probs = softmax(&logits);

        let direction = if probs[0] > probs[1] && probs[0] > probs[2] {
            PriceDirection::Up
        } else if probs[1] > probs[0] && probs[1] > probs[2] {
            PriceDirection::Down
        } else {
            PriceDirection::Stationary
        };

        (direction, probs)
    }
}

// ─── Full Attention LOB Model ──────────────────────────────────────

/// Complete attention-based LOB model combining all components.
#[derive(Debug)]
pub struct AttentionLobModel {
    pub input_projection: Array2<f64>,
    pub level_attention: MultiHeadAttention,
    pub cross_attention: CrossAttention,
    pub temporal_attention: TemporalAttention,
    pub prediction_head: PredictionHead,
    pub d_model: usize,
    pub num_levels: usize,
}

impl AttentionLobModel {
    /// Create a new model.
    /// `num_levels`: number of levels per side (e.g., 10 for 10-level LOB).
    /// `n_features`: number of raw features per level (e.g., 5).
    /// `d_model`: hidden dimension.
    /// `num_heads`: number of attention heads.
    pub fn new(num_levels: usize, n_features: usize, d_model: usize, num_heads: usize) -> Self {
        Self {
            input_projection: xavier_init(n_features, d_model),
            level_attention: MultiHeadAttention::new(d_model, num_heads),
            cross_attention: CrossAttention::new(d_model),
            temporal_attention: TemporalAttention::new(d_model),
            prediction_head: PredictionHead::new(d_model, 3),
            d_model,
            num_levels,
        }
    }

    /// Process a single LOB snapshot through level attention and cross-attention.
    /// Returns a fixed-size embedding for the snapshot.
    pub fn encode_snapshot(&self, snapshot: &LobSnapshot) -> (Array1<f64>, Vec<Array2<f64>>) {
        let features = snapshot.to_feature_matrix();
        let projected = features.dot(&self.input_projection);

        // Self-attention over all levels
        let (attended, level_weights) = self.level_attention.forward(&projected);

        // Split into bid and ask for cross-attention
        let n_bid = snapshot.bid_prices.len();
        let bid_side = attended.slice(ndarray::s![..n_bid, ..]).to_owned();
        let ask_side = attended.slice(ndarray::s![n_bid.., ..]).to_owned();

        // Cross-attention: bids attend to asks
        let (cross_out, _cross_weights) = self.cross_attention.forward(&bid_side, &ask_side);

        // Pool: mean over levels to get a single vector
        let mut embedding = Array1::zeros(self.d_model);
        for row in cross_out.axis_iter(Axis(0)) {
            embedding = embedding + &row.to_owned();
        }
        let n = cross_out.nrows() as f64;
        embedding = embedding / n;

        (embedding, level_weights)
    }

    /// Predict mid-price direction from a sequence of LOB snapshots.
    pub fn predict(
        &self,
        snapshots: &[LobSnapshot],
    ) -> (PriceDirection, Array1<f64>, Array1<f64>) {
        // Encode each snapshot
        let mut embeddings = Vec::new();
        for snap in snapshots {
            let (emb, _) = self.encode_snapshot(snap);
            embeddings.push(emb);
        }

        // Temporal attention over the sequence
        let (context, temporal_weights) = self.temporal_attention.forward(&embeddings);

        // Prediction
        let (direction, probs) = self.prediction_head.predict(&context);

        (direction, probs, temporal_weights)
    }
}

// ─── Bybit Integration ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct BybitOrderbookResponse {
    result: BybitOrderbookResult,
}

#[derive(Debug, Deserialize)]
struct BybitOrderbookResult {
    #[serde(rename = "b")]
    bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    asks: Vec<[String; 2]>,
    #[serde(rename = "ts")]
    timestamp: u64,
}

/// Fetch orderbook from Bybit REST API.
pub async fn fetch_bybit_orderbook(symbol: &str, limit: usize) -> Result<LobSnapshot> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={}&limit={}",
        symbol, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitOrderbookResponse = client.get(&url).send().await?.json().await?;

    let mut bid_prices = Vec::new();
    let mut bid_volumes = Vec::new();
    for entry in &resp.result.bids {
        bid_prices.push(entry[0].parse::<f64>()?);
        bid_volumes.push(entry[1].parse::<f64>()?);
    }

    let mut ask_prices = Vec::new();
    let mut ask_volumes = Vec::new();
    for entry in &resp.result.asks {
        ask_prices.push(entry[0].parse::<f64>()?);
        ask_volumes.push(entry[1].parse::<f64>()?);
    }

    Ok(LobSnapshot {
        bid_prices,
        bid_volumes,
        ask_prices,
        ask_volumes,
        timestamp: resp.result.timestamp,
    })
}

/// Fetch orderbook using blocking client (for non-async contexts).
pub fn fetch_bybit_orderbook_blocking(symbol: &str, limit: usize) -> Result<LobSnapshot> {
    let url = format!(
        "https://api.bybit.com/v5/market/orderbook?category=spot&symbol={}&limit={}",
        symbol, limit
    );

    let resp: BybitOrderbookResponse = reqwest::blocking::get(&url)?.json()?;

    let mut bid_prices = Vec::new();
    let mut bid_volumes = Vec::new();
    for entry in &resp.result.bids {
        bid_prices.push(entry[0].parse::<f64>()?);
        bid_volumes.push(entry[1].parse::<f64>()?);
    }

    let mut ask_prices = Vec::new();
    let mut ask_volumes = Vec::new();
    for entry in &resp.result.asks {
        ask_prices.push(entry[0].parse::<f64>()?);
        ask_volumes.push(entry[1].parse::<f64>()?);
    }

    Ok(LobSnapshot {
        bid_prices,
        bid_volumes,
        ask_prices,
        ask_volumes,
        timestamp: resp.result.timestamp,
    })
}

/// Create a synthetic LOB snapshot for testing.
pub fn create_synthetic_lob(levels: usize, base_price: f64, tick_size: f64) -> LobSnapshot {
    let mut rng = rand::thread_rng();
    let mut bid_prices = Vec::with_capacity(levels);
    let mut bid_volumes = Vec::with_capacity(levels);
    let mut ask_prices = Vec::with_capacity(levels);
    let mut ask_volumes = Vec::with_capacity(levels);

    for i in 0..levels {
        bid_prices.push(base_price - (i as f64 + 0.5) * tick_size);
        bid_volumes.push(rng.gen_range(0.1..10.0));
        ask_prices.push(base_price + (i as f64 + 0.5) * tick_size);
        ask_volumes.push(rng.gen_range(0.1..10.0));
    }

    LobSnapshot {
        bid_prices,
        bid_volumes,
        ask_prices,
        ask_volumes,
        timestamp: 0,
    }
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> LobSnapshot {
        LobSnapshot {
            bid_prices: vec![100.0, 99.5, 99.0, 98.5, 98.0],
            bid_volumes: vec![1.0, 2.0, 3.0, 1.5, 0.5],
            ask_prices: vec![100.5, 101.0, 101.5, 102.0, 102.5],
            ask_volumes: vec![1.2, 1.8, 2.5, 0.8, 0.3],
            timestamp: 1000,
        }
    }

    #[test]
    fn test_mid_price() {
        let snap = sample_snapshot();
        let mid = snap.mid_price();
        assert!((mid - 100.25).abs() < 1e-10);
    }

    #[test]
    fn test_spread() {
        let snap = sample_snapshot();
        let spread = snap.spread();
        assert!((spread - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_feature_matrix_shape() {
        let snap = sample_snapshot();
        let mat = snap.to_feature_matrix();
        assert_eq!(mat.nrows(), 10); // 5 bids + 5 asks
        assert_eq!(mat.ncols(), 5); // 5 features
    }

    #[test]
    fn test_volume_imbalance() {
        let snap = sample_snapshot();
        let imb = snap.volume_imbalance(3);
        let bid_vol: f64 = vec![1.0, 2.0, 3.0].iter().sum();
        let ask_vol: f64 = vec![1.2, 1.8, 2.5].iter().sum();
        let expected = (bid_vol - ask_vol) / (bid_vol + ask_vol);
        assert!((imb - expected).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let s = softmax(&x);
        assert!((s.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_head_attention_output_shape() {
        let d_model = 8;
        let num_heads = 2;
        let seq_len = 10;
        let mha = MultiHeadAttention::new(d_model, num_heads);
        let input = Array2::from_shape_fn((seq_len, d_model), |_| rand::random::<f64>());
        let (output, weights) = mha.forward(&input);
        assert_eq!(output.shape(), &[seq_len, d_model]);
        assert_eq!(weights.len(), num_heads);
        for w in &weights {
            assert_eq!(w.shape(), &[seq_len, seq_len]);
        }
    }

    #[test]
    fn test_temporal_attention_output() {
        let d_model = 8;
        let ta = TemporalAttention::new(d_model);
        let sequence: Vec<Array1<f64>> = (0..5)
            .map(|_| Array1::from_shape_fn(d_model, |_| rand::random::<f64>()))
            .collect();
        let (context, weights) = ta.forward(&sequence);
        assert_eq!(context.len(), d_model);
        assert_eq!(weights.len(), 5);
        assert!((weights.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_attention_output_shape() {
        let d_model = 8;
        let n_levels = 5;
        let ca = CrossAttention::new(d_model);
        let bids = Array2::from_shape_fn((n_levels, d_model), |_| rand::random::<f64>());
        let asks = Array2::from_shape_fn((n_levels, d_model), |_| rand::random::<f64>());
        let (output, weights) = ca.forward(&bids, &asks);
        assert_eq!(output.shape(), &[n_levels, d_model]);
        assert_eq!(weights.shape(), &[n_levels, n_levels]);
    }

    #[test]
    fn test_full_model_prediction() {
        let model = AttentionLobModel::new(5, 5, 10, 2);
        let snapshots: Vec<LobSnapshot> = (0..3)
            .map(|_| create_synthetic_lob(5, 100.0, 0.5))
            .collect();
        let (direction, probs, temporal_weights) = model.predict(&snapshots);
        assert!(matches!(
            direction,
            PriceDirection::Up | PriceDirection::Down | PriceDirection::Stationary
        ));
        assert_eq!(probs.len(), 3);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert_eq!(temporal_weights.len(), 3);
        assert!((temporal_weights.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_synthetic_lob_structure() {
        let lob = create_synthetic_lob(10, 50000.0, 0.5);
        assert_eq!(lob.bid_prices.len(), 10);
        assert_eq!(lob.ask_prices.len(), 10);
        // Bids should be descending
        for i in 1..lob.bid_prices.len() {
            assert!(lob.bid_prices[i] < lob.bid_prices[i - 1]);
        }
        // Asks should be ascending
        for i in 1..lob.ask_prices.len() {
            assert!(lob.ask_prices[i] > lob.ask_prices[i - 1]);
        }
        // Best bid < best ask
        assert!(lob.bid_prices[0] < lob.ask_prices[0]);
    }
}

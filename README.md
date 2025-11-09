# Chapter 276: Attention LOB (Limit Order Book) Trading

## Introduction

The Limit Order Book (LOB) is the fundamental data structure of modern electronic markets, recording all outstanding buy and sell orders at various price levels. Traditional approaches to LOB modeling treat price levels uniformly or rely on hand-crafted features such as order flow imbalance or weighted mid-price. However, not all price levels carry equal informational content at any given moment. A sudden surge of volume at the third bid level, or an unusually thin ask at the best price, may be far more predictive than the aggregate statistics suggest.

Attention mechanisms, originally developed for natural language processing in the Transformer architecture, provide an elegant solution to this problem. By allowing the model to dynamically weight the importance of different price levels and different time steps, attention-based LOB models can learn which parts of the order book matter most for a given prediction task.

This chapter explores three forms of attention applied to LOB data:

1. **Cross-level attention** (self-attention over price levels): The model learns which price levels to attend to when making predictions. For example, the best bid might attend strongly to the second-best ask if there is a pattern of quote stuffing.

2. **Temporal attention** (attention over LOB snapshots across time): Instead of treating all historical snapshots equally (as an LSTM or simple moving average would), the model learns which past states are most relevant. A snapshot from 50ms ago showing a large order cancellation might be more informative than the snapshot from 10ms ago.

3. **Cross-attention between bid and ask sides**: The bid side attends to the ask side and vice versa, capturing the interplay between supply and demand at different levels.

## Mathematical Foundations

### LOB Feature Matrix

At each timestamp $t$, we represent the LOB as a feature matrix $\mathbf{X}_t \in \mathbb{R}^{L \times F}$, where $L$ is the number of price levels and $F$ is the number of features per level. Typical features include:

- Price $p_i$
- Volume $v_i$
- Number of orders $n_i$
- Cumulative volume $\sum_{j=1}^{i} v_j$
- Price distance from mid: $\delta_i = |p_i - p_{mid}|$

For a 10-level order book with both bid and ask sides, we might have $L = 20$ (10 bid + 10 ask levels) and $F = 5$ features per level.

### Self-Attention Over Price Levels

Given the LOB feature matrix $\mathbf{X}_t \in \mathbb{R}^{L \times F}$, we first project it into query, key, and value spaces:

$$\mathbf{Q} = \mathbf{X}_t \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}_t \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}_t \mathbf{W}_V$$

where $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{F \times d_k}$ and $\mathbf{W}_V \in \mathbb{R}^{F \times d_v}$.

The attention weights are computed as:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

The attention matrix $\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{L \times L}$ has a clear interpretation: $A_{ij}$ represents how much level $i$ attends to level $j$. This allows us to discover cross-level dependencies, such as the best bid attending to deep ask levels during aggressive selling.

### Multi-Head Attention

Multi-head attention runs $H$ parallel attention functions with different learned projections:

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}_O$$

where $\text{head}_h = \text{Attention}(\mathbf{X}\mathbf{W}_Q^h, \mathbf{X}\mathbf{W}_K^h, \mathbf{X}\mathbf{W}_V^h)$.

Each head can specialize in capturing different types of LOB patterns: one head might focus on the spread dynamics, another on volume imbalance, and a third on deep-book pressure.

### Temporal Attention Over LOB Snapshots

Given a sequence of LOB representations $\{\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T\}$ (where each $\mathbf{h}_t$ is the output of the level-wise attention layer), we apply temporal attention:

$$\alpha_t = \frac{\exp(e_t)}{\sum_{s=1}^{T} \exp(e_s)}, \quad e_t = \mathbf{v}^T \tanh(\mathbf{W}_h \mathbf{h}_t + \mathbf{b})$$

The context vector is then:

$$\mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t$$

This form of attention (Bahdanau-style) allows the model to focus on the most informative snapshots. During volatile periods, recent snapshots receive higher weights; during calm periods, the model might attend to a broader temporal window.

### Cross-Attention Between Bid and Ask

Let $\mathbf{B} \in \mathbb{R}^{L/2 \times d}$ represent bid-side embeddings and $\mathbf{A} \in \mathbb{R}^{L/2 \times d}$ represent ask-side embeddings. Cross-attention computes:

$$\text{CrossAttn}(\mathbf{B}, \mathbf{A}) = \text{softmax}\left(\frac{\mathbf{B}\mathbf{W}_Q (\mathbf{A}\mathbf{W}_K)^T}{\sqrt{d_k}}\right) \mathbf{A}\mathbf{W}_V$$

This captures the interaction between supply and demand. For instance, a large bid at level 3 attending strongly to the best ask suggests potential aggressive buying pressure.

## Applications

### Mid-Price Prediction

The primary application is predicting the direction of mid-price movement over a short horizon (e.g., 1-10 seconds):

$$p_{mid}(t) = \frac{p_{ask,1}(t) + p_{bid,1}(t)}{2}$$

The model predicts whether $p_{mid}(t+\Delta) - p_{mid}(t)$ will be positive (up), negative (down), or approximately zero (stationary). The attention output is passed through a classification head:

$$\hat{y} = \text{softmax}(\mathbf{W}_c \mathbf{c} + \mathbf{b}_c)$$

where $\mathbf{c}$ is the final context vector combining level-wise and temporal attention.

### Spread Prediction

Predicting the bid-ask spread is crucial for market-making strategies:

$$s(t) = p_{ask,1}(t) - p_{bid,1}(t)$$

The attention model excels here because spread dynamics depend on complex interactions between multiple LOB levels. A thinning of volume at the best levels (captured by cross-level attention) combined with historical spread patterns (captured by temporal attention) provide strong predictive signals.

### Order Flow Imbalance

Order flow imbalance (OFI) measures the pressure difference between buy and sell sides:

$$\text{OFI}(t) = \sum_{i=1}^{K} \left( v_{bid,i}(t) - v_{bid,i}(t-1) \right) - \sum_{i=1}^{K} \left( v_{ask,i}(t) - v_{ask,i}(t-1) \right)$$

While traditional OFI uses fixed weights across levels, the attention mechanism learns adaptive weights, producing a more informative imbalance measure. The model can learn that changes at deeper levels (e.g., level 5-10) carry different significance during different market regimes.

## Rust Implementation

The Rust implementation in `rust/src/lib.rs` provides a complete attention-based LOB trading system. Key components include:

### LOB Feature Extraction

The `LobSnapshot` struct captures the state of the order book at a given time:

```rust
pub struct LobSnapshot {
    pub bid_prices: Vec<f64>,
    pub bid_volumes: Vec<f64>,
    pub ask_prices: Vec<f64>,
    pub ask_volumes: Vec<f64>,
    pub timestamp: u64,
}
```

Features are extracted into a matrix where each row represents a price level and columns represent different features (price, volume, distance from mid, cumulative volume, order imbalance ratio).

### Multi-Head Self-Attention

The `MultiHeadAttention` struct implements scaled dot-product attention with configurable number of heads:

```rust
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub w_o: Array2<f64>,
}
```

### Temporal Attention

The `TemporalAttention` module applies Bahdanau-style attention over a sequence of LOB snapshots, learning which historical states are most relevant for the current prediction.

### Bybit Integration

The `fetch_bybit_orderbook` function retrieves live orderbook data from the Bybit API:

```rust
pub async fn fetch_bybit_orderbook(symbol: &str, limit: usize)
    -> Result<LobSnapshot>
```

This allows real-time application of the attention model to cryptocurrency markets.

## Bybit Data Integration

The implementation connects to Bybit's public REST API to fetch orderbook snapshots:

1. **Endpoint**: `https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit={limit}`
2. **Data format**: Returns bid and ask arrays with `[price, size]` pairs
3. **Rate limits**: The public endpoint allows frequent polling suitable for research

The typical workflow is:
1. Poll the orderbook at regular intervals (e.g., every 100ms)
2. Construct `LobSnapshot` objects from each response
3. Maintain a sliding window of recent snapshots
4. Run the attention model on the window to generate predictions

For backtesting, historical orderbook data can be loaded from files in the same format.

## Key Takeaways

1. **Attention mechanisms naturally fit LOB data**: Price levels are analogous to tokens in NLP, and attention can discover which levels are most informative for a given prediction task.

2. **Multi-head attention captures diverse patterns**: Different heads specialize in different LOB dynamics - spread, volume imbalance, depth pressure, and cross-side interactions.

3. **Temporal attention outperforms fixed windows**: Instead of equally weighting all historical snapshots, the model learns to focus on the most informative past states, adapting to market regime changes.

4. **Cross-attention between bid and ask sides** captures supply-demand dynamics that are invisible to models treating each side independently.

5. **Interpretability is a key advantage**: Attention weights provide a natural explanation for model predictions. Traders can inspect which levels and time steps the model focused on.

6. **Rust implementation enables low-latency deployment**: The performance characteristics of Rust make it suitable for real-time LOB processing where microseconds matter.

7. **Integration with exchange APIs** (such as Bybit) allows seamless transition from research to live trading, using the same data structures and model code.

8. **Feature engineering remains important**: While attention reduces the need for hand-crafted features, the choice of per-level features (price, volume, cumulative volume, distance from mid) significantly impacts model performance.

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
- Zhang, Z., Zohren, S., & Roberts, S. (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." IEEE Transactions on Signal Processing.
- Tran, D. T., Iosifidis, A., Kanniainen, J., & Gabbouj, M. (2019). "Temporal Attention-Augmented Bilinear Network for Financial Time-Series Data Analysis." IEEE Transactions on Neural Networks and Learning Systems.
- Sirignano, J. A. (2019). "Deep Learning for Limit Order Books." Quantitative Finance.
- Tsantekidis, A., et al. (2017). "Forecasting Stock Prices from the Limit Order Book Using Convolutional Neural Networks." IEEE Conference on Business Informatics.

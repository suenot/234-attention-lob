use attention_lob::*;

fn main() {
    println!("=== Attention LOB Trading Example ===\n");

    // --- Step 1: Create synthetic LOB snapshots (simulating Bybit data) ---
    println!("Step 1: Generating synthetic LOB snapshots...");
    let num_snapshots = 10;
    let num_levels = 10;
    let base_price = 50000.0; // BTCUSDT-like price
    let tick_size = 0.5;

    let snapshots: Vec<LobSnapshot> = (0..num_snapshots)
        .map(|i| {
            let price_drift = (i as f64 - 5.0) * 0.1;
            create_synthetic_lob(num_levels, base_price + price_drift, tick_size)
        })
        .collect();

    for (i, snap) in snapshots.iter().enumerate() {
        println!(
            "  Snapshot {}: mid={:.2}, spread={:.4}, imbalance={:.4}",
            i,
            snap.mid_price(),
            snap.spread(),
            snap.volume_imbalance(5)
        );
    }

    // --- Step 2: Inspect feature matrix for first snapshot ---
    println!("\nStep 2: Feature matrix for first snapshot:");
    let features = snapshots[0].to_feature_matrix();
    println!("  Shape: {} levels x {} features", features.nrows(), features.ncols());
    println!("  Columns: [price, volume, dist_from_mid, cum_volume, side]");
    for i in 0..features.nrows().min(4) {
        let side = if features[[i, 4]] > 0.0 { "BID" } else { "ASK" };
        println!(
            "  Level {:2} ({}): price={:.2}, vol={:.4}, dist={:.4}, cum_vol={:.4}",
            i, side,
            features[[i, 0]],
            features[[i, 1]],
            features[[i, 2]],
            features[[i, 3]]
        );
    }
    println!("  ... ({} more levels)", features.nrows() - 4);

    // --- Step 3: Initialize the attention model ---
    println!("\nStep 3: Initializing Attention LOB Model...");
    let d_model = 16;
    let num_heads = 4;
    let n_features = 5;
    let model = AttentionLobModel::new(num_levels, n_features, d_model, num_heads);
    println!(
        "  Model: d_model={}, num_heads={}, d_k={}",
        d_model, num_heads, d_model / num_heads
    );

    // --- Step 4: Run attention on a single snapshot ---
    println!("\nStep 4: Encoding single snapshot with level attention...");
    let (embedding, level_weights) = model.encode_snapshot(&snapshots[0]);
    println!("  Embedding dimension: {}", embedding.len());
    println!("  Number of attention heads: {}", level_weights.len());

    // Show attention weights for first head
    println!("  Attention weights (head 0, first 5x5 block):");
    let w = &level_weights[0];
    for i in 0..5.min(w.nrows()) {
        print!("    ");
        for j in 0..5.min(w.ncols()) {
            print!("{:.3} ", w[[i, j]]);
        }
        println!();
    }

    // --- Step 5: Run full model with temporal attention ---
    println!("\nStep 5: Running full model on snapshot sequence...");
    let window_size = 5;
    let window = &snapshots[snapshots.len() - window_size..];
    let (direction, probs, temporal_weights) = model.predict(window);

    println!("  Prediction: {:?}", direction);
    println!(
        "  Probabilities: Up={:.4}, Down={:.4}, Stationary={:.4}",
        probs[0], probs[1], probs[2]
    );
    println!("  Temporal attention weights:");
    for (i, w) in temporal_weights.iter().enumerate() {
        let bar_len = (w * 40.0) as usize;
        let bar: String = "#".repeat(bar_len);
        println!("    t-{}: {:.4} {}", window_size - 1 - i, w, bar);
    }

    // --- Step 6: Sliding window predictions ---
    println!("\nStep 6: Sliding window predictions:");
    println!("  {:>6} {:>10} {:>12} {:>10}", "Window", "Direction", "Confidence", "Imbalance");
    for start in 0..=(snapshots.len() - window_size) {
        let w = &snapshots[start..start + window_size];
        let (dir, probs, _) = model.predict(w);
        let confidence = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let last_imb = w.last().unwrap().volume_imbalance(5);
        println!(
            "  {:>3}-{:>2} {:>10?} {:>12.4} {:>10.4}",
            start,
            start + window_size - 1,
            dir,
            confidence,
            last_imb
        );
    }

    // --- Step 7: Note about Bybit integration ---
    println!("\nStep 7: Bybit Integration");
    println!("  To fetch live BTCUSDT orderbook data, use:");
    println!("    let snapshot = fetch_bybit_orderbook(\"BTCUSDT\", 50).await?;");
    println!("  This requires network access and an async runtime.");
    println!("  The returned LobSnapshot can be fed directly into the model.");

    println!("\n=== Example Complete ===");
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: sync tokenizer.encode() on tokio runtime vs spawn_blocking offload
// Matches the exact Arc<dyn Tokenizer> pattern from preprocessor.rs
//
// Usage:
//   TOKENIZER_PATH=/path/to/tokenizer.json VOCAB_SIZE=151643 \
//     cargo bench --bench spawn_blocking_tokenizer
//
// Output: benchmarks/tokenizer_spawn_blocking_results.jsonl

use std::sync::Arc;
use std::time::Instant;

use dynamo_llm::tokenizers::hf::HuggingFaceTokenizer;
use dynamo_llm::tokenizers::traits::{Encoder, Tokenizer};
use rand::Rng;

const DEFAULT_TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
);

const ISLS: &[usize] = &[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
const CONCURRENCIES: &[usize] = &[1, 4, 8, 16, 32, 64];
const NUM_SAMPLES: usize = 1000;

/// Generate random text by sampling token IDs and decoding
fn generate_samples(
    tokenizer: &Arc<dyn Tokenizer>,
    vocab_size: usize,
    isl: usize,
    n: usize,
) -> Vec<String> {
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let token_ids: Vec<u32> = (0..isl)
            .map(|_| rng.gen_range(100..vocab_size as u32))
            .collect();
        let text = tokenizer.decode(&token_ids, true).unwrap_or_default();
        samples.push(text);
    }
    samples
}

/// BEFORE pattern from preprocessor.rs — sync encode blocks the tokio runtime thread:
///   let encoding = self.tokenizer.encode(&prompt)?;
async fn bench_sync_on_runtime(
    tokenizer: Arc<dyn Tokenizer>,
    texts: &[String],
    concurrency: usize,
) -> Vec<f64> {
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut handles = Vec::with_capacity(texts.len());

    for text in texts {
        let tok = tokenizer.clone();
        let t = text.clone();
        let permit = sem.clone().acquire_owned().await.unwrap();
        handles.push(tokio::spawn(async move {
            let start = Instant::now();
            let _encoding = tok.encode(&t).unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            drop(permit);
            elapsed
        }));
    }

    let mut latencies = Vec::with_capacity(handles.len());
    for h in handles {
        latencies.push(h.await.unwrap());
    }
    latencies
}

/// AFTER pattern from preprocessor.rs — offload to blocking thread pool:
///   let tokenizer = self.tokenizer.clone();
///   let prompt_owned = prompt.clone();
///   let encoding = tokio::task::spawn_blocking(move || {
///       tokenizer.encode(&prompt_owned)
///   }).await??;
async fn bench_spawn_blocking(
    tokenizer: Arc<dyn Tokenizer>,
    texts: &[String],
    concurrency: usize,
) -> Vec<f64> {
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut handles = Vec::with_capacity(texts.len());

    for text in texts {
        let tok = tokenizer.clone();
        let t = text.clone();
        let permit = sem.clone().acquire_owned().await.unwrap();
        handles.push(tokio::spawn(async move {
            let start = Instant::now();
            let _encoding = tokio::task::spawn_blocking(move || tok.encode(&t))
                .await
                .unwrap()
                .unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            drop(permit);
            elapsed
        }));
    }

    let mut latencies = Vec::with_capacity(handles.len());
    for h in handles {
        latencies.push(h.await.unwrap());
    }
    latencies
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[tokio::main]
async fn main() {
    let tokenizer_path =
        std::env::var("TOKENIZER_PATH").unwrap_or_else(|_| DEFAULT_TOKENIZER.to_string());

    eprintln!("Loading tokenizer from: {tokenizer_path}");
    let hf_tok =
        HuggingFaceTokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(hf_tok);

    let vocab_size: usize = std::env::var("VOCAB_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32000);

    let output_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benchmarks")
        .join("tokenizer_spawn_blocking_results.jsonl");

    eprintln!(
        "ISLs: {:?}\nConcurrencies: {:?}\nSamples per config: {}",
        ISLS, CONCURRENCIES, NUM_SAMPLES
    );

    let mut jsonl_lines = Vec::new();

    // Pre-generate samples for each ISL (reuse across concurrency levels)
    let mut all_samples = Vec::new();
    for &isl in ISLS {
        eprint!("Generating ISL {isl:>6}... ");
        let samples = generate_samples(&tokenizer, vocab_size, isl, NUM_SAMPLES);
        let avg_chars: f64 =
            samples.iter().map(|s| s.len() as f64).sum::<f64>() / samples.len() as f64;
        eprintln!("avg_chars={avg_chars:.0}");
        all_samples.push((isl, avg_chars, samples));
    }

    for &concurrency in CONCURRENCIES {
        eprintln!("\n=== concurrency={concurrency} ===");

        for (isl, avg_chars, samples) in &all_samples {
            // Warmup
            for s in samples.iter().take(10) {
                let _ = tokenizer.encode(s);
            }

            // Bench sync (before)
            let mut sync_lats =
                bench_sync_on_runtime(tokenizer.clone(), samples, concurrency).await;
            sync_lats.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Bench spawn_blocking (after)
            let mut sb_lats =
                bench_spawn_blocking(tokenizer.clone(), samples, concurrency).await;
            sb_lats.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let sync_mean: f64 = sync_lats.iter().sum::<f64>() / sync_lats.len() as f64;
            let sb_mean: f64 = sb_lats.iter().sum::<f64>() / sb_lats.len() as f64;

            eprintln!(
                "  ISL {isl:>6}: sync p50={:.2} p99={:.2} | sb p50={:.2} p99={:.2} ms",
                percentile(&sync_lats, 50.0),
                percentile(&sync_lats, 99.0),
                percentile(&sb_lats, 50.0),
                percentile(&sb_lats, 99.0),
            );

            let row = serde_json::json!({
                "isl": isl,
                "concurrency": concurrency,
                "num_samples": NUM_SAMPLES,
                "avg_chars": (*avg_chars).round() as usize,
                "sync_p50_ms": (percentile(&sync_lats, 50.0) * 1000.0).round() / 1000.0,
                "sync_p99_ms": (percentile(&sync_lats, 99.0) * 1000.0).round() / 1000.0,
                "sync_mean_ms": (sync_mean * 1000.0).round() / 1000.0,
                "spawn_blocking_p50_ms": (percentile(&sb_lats, 50.0) * 1000.0).round() / 1000.0,
                "spawn_blocking_p99_ms": (percentile(&sb_lats, 99.0) * 1000.0).round() / 1000.0,
                "spawn_blocking_mean_ms": (sb_mean * 1000.0).round() / 1000.0,
            });
            jsonl_lines.push(row.to_string());
        }
    }

    // Write JSONL
    std::fs::create_dir_all(output_path.parent().unwrap()).ok();
    std::fs::write(&output_path, jsonl_lines.join("\n") + "\n").expect("Failed to write JSONL");
    eprintln!("\nResults written to {}", output_path.display());
}

//! Wikipedia indexing example.
//!
//! Downloads the English Wikipedia from HuggingFace (wikimedia/wikipedia dataset)
//! using the hf-hub crate and builds a search index in a streaming fashion.
//! Uses random embeddings for demonstration (replace with a real model in production).
//!
//! Usage:
//!   cargo run --release --example wikipedia -- [--limit N] [--dim D]
//!
//! Options:
//!   --limit N   Limit number of documents to index (default: all ~6.7M articles)
//!   --dim D     Embedding dimension (default: 128)
//!
//! The example streams documents directly into the IndexBuilder to avoid loading
//! the full corpus into memory.

use arrow_array::StringArray;
use bytes::Bytes;
use futures::stream::{self, StreamExt};
use hf_hub::api::tokio::Api;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::Rng;
use seglens::{Document, IndexBuilder, IndexReader};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

const DOWNLOAD_CONCURRENCY: usize = 4;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut limit: Option<usize> = None;
    let mut dim: usize = 128;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                i += 1;
                limit = args.get(i).and_then(|s| s.parse().ok());
            }
            "--dim" => {
                i += 1;
                dim = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(128);
            }
            _ => {}
        }
        i += 1;
    }

    println!("Wikipedia Search Index Example");
    println!("==============================");
    if let Some(n) = limit {
        println!("Document limit: {}", n);
    } else {
        println!("Document limit: none (full dataset)");
    }
    println!("Embedding dimension: {}", dim);
    println!();

    // Set up storage
    let index_dir = std::env::temp_dir().join("seglens-wikipedia-index");
    let temp_dir = std::env::temp_dir().join("seglens-wikipedia-temp");
    let _ = std::fs::remove_dir_all(&index_dir);
    let _ = std::fs::remove_dir_all(&temp_dir);
    let store = Arc::new(seglens::local(&index_dir)?);

    // Streaming index build: download + parse + add docs directly.
    let (doc_count, num_clusters) = download_and_feed(&temp_dir, dim, limit).await?;

    println!("\nBuilding index for {} articles...", doc_count);
    println!("Using {} clusters for IVF", num_clusters);

    // Pull builder out of TLS, finish writing index
    let mut builder = INDEX_BUILDER
        .with(|b| b.borrow_mut().take())
        .expect("builder should exist");
    builder.set_num_clusters(num_clusters.max(2));
    println!("Writing index to disk...");
    builder.build(store.as_ref(), "wiki").await?;
    println!("Index built successfully!");

    // Open and query the index
    println!("\nOpening index for queries...");
    let reader = IndexReader::open(store, "wiki").await?;
    println!("Index contains {} documents", reader.doc_count());

    // Demo queries
    println!("\n--- Lexical Search Demo ---");
    let queries = [
        "machine learning",
        "world war",
        "united states",
        "quantum physics",
    ];

    for query in queries {
        println!("\nSearching for: \"{}\"", query);
        let results = reader.search_lexical(query, 3).await?;

        if results.is_empty() {
            println!("  No results found");
        } else {
            for (i, result) in results.iter().enumerate() {
                let title = result
                    .attributes
                    .get("title")
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                let preview: String = result.text.chars().take(100).collect();
                println!("  {}. [score: {:.4}] {}", i + 1, result.score, title);
                println!("     {}...", preview);
            }
        }
    }

    // Vector search with random query
    println!("\n--- Vector Search Demo ---");
    println!("Searching with random query embedding...");
    let query_embedding = random_embedding(dim);
    let results = reader.search_vector(&query_embedding, 3, 2).await?;

    for (i, result) in results.iter().enumerate() {
        let title = result
            .attributes
            .get("title")
            .map(|s| s.as_str())
            .unwrap_or("Unknown");
        println!("  {}. [score: {:.4}] {}", i + 1, result.score, title);
    }

    // Hybrid search
    println!("\n--- Hybrid Search Demo ---");
    println!("Searching for \"artificial intelligence\" with random embedding...");
    let results = reader
        .search_hybrid("artificial intelligence", &random_embedding(dim), 3, 2)
        .await?;

    for (i, result) in results.iter().enumerate() {
        let title = result
            .attributes
            .get("title")
            .map(|s| s.as_str())
            .unwrap_or("Unknown");
        println!("  {}. [score: {:.4}] {}", i + 1, result.score, title);
    }

    println!("\nIndex location: {}", index_dir.display());
    println!("Done!");

    Ok(())
}

thread_local! {
    static INDEX_BUILDER: RefCell<Option<IndexBuilder>> = RefCell::new(None);
}

async fn download_and_feed(
    temp_dir: &PathBuf,
    dim: usize,
    limit: Option<usize>,
) -> Result<(usize, u32), Box<dyn std::error::Error>> {
    println!("Downloading English Wikipedia from HuggingFace (wikimedia/wikipedia)...");
    println!("Using hf-hub crate for efficient caching and downloads");
    println!();

    // Initialize builder with streaming temp dir; store in TLS so we can mutate inside loops.
    INDEX_BUILDER.with(|b| {
        *b.borrow_mut() = Some(
            IndexBuilder::new(dim as u32, 0)
                .with_temp_dir(temp_dir)
                .expect("temp dir"),
        )
    });

    let api = Api::new()?;
    let repo = Arc::new(api.dataset("wikimedia/wikipedia".to_string()));

    println!("Fetching file list from HuggingFace...");
    let info = repo.info().await?;

    let mut parquet_files: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|f| f.starts_with("20231101.en/") && f.ends_with(".parquet"))
        .collect();

    parquet_files.sort();
    let total_files = parquet_files.len();
    println!(
        "Found {} parquet files for English Wikipedia (~{}GB total)",
        total_files,
        total_files * 400 / 1024
    );
    println!();

    let doc_limit = limit.unwrap_or(usize::MAX);
    let mut doc_id: u32 = 0;

    let file_pb = ProgressBar::new(total_files as u64);
    file_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] file {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut stream = stream::iter(parquet_files.into_iter().map(|filename| {
        let repo = repo.clone();
        async move {
            let res = repo.get(&filename).await;
            (filename, res)
        }
    }))
    .buffer_unordered(DOWNLOAD_CONCURRENCY);

    while let Some((filename, path_res)) = stream.next().await {
        if (doc_id as usize) >= doc_limit {
            break;
        }

        println!("Processing file: {}", filename);

        let path = match path_res {
            Ok(p) => p,
            Err(e) => {
                println!("  Warning: Failed to download: {}", e);
                file_pb.inc(1);
                continue;
            }
        };

        println!("  Cached at: {}", path.display());
        let file_bytes = Bytes::from(std::fs::read(&path)?);
        println!("  Size: {} MB", file_bytes.len() / (1024 * 1024));

        let builder = ParquetRecordBatchReaderBuilder::try_new(file_bytes)?;
        let reader = builder.with_batch_size(4096).build()?;

        let doc_pb = ProgressBar::new_spinner();
        doc_pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        doc_pb.enable_steady_tick(Duration::from_millis(100));

        for batch_result in reader {
            if (doc_id as usize) >= doc_limit {
                break;
            }
            let batch = batch_result?;

            let id_col = batch
                .column_by_name("id")
                .ok_or("Missing 'id' column")?
                .as_any()
                .downcast_ref::<StringArray>();
            let title_col = batch
                .column_by_name("title")
                .ok_or("Missing 'title' column")?
                .as_any()
                .downcast_ref::<StringArray>();
            let text_col = batch
                .column_by_name("text")
                .ok_or("Missing 'text' column")?
                .as_any()
                .downcast_ref::<StringArray>();
            let url_col = batch
                .column_by_name("url")
                .ok_or("Missing 'url' column")?
                .as_any()
                .downcast_ref::<StringArray>();

            if let (Some(ids), Some(titles), Some(texts), Some(urls)) =
                (id_col, title_col, text_col, url_col)
            {
                for i in 0..batch.num_rows() {
                    if (doc_id as usize) >= doc_limit {
                        break;
                    }

                    let text = texts.value(i);
                    if text.len() < 100 {
                        continue;
                    }

                    let doc = Document {
                        id: doc_id,
                        text: text.to_string(),
                        embedding: random_embedding(dim),
                        attributes: HashMap::from([
                            ("title".to_string(), titles.value(i).to_string()),
                            ("url".to_string(), urls.value(i).to_string()),
                            ("source_id".to_string(), ids.value(i).to_string()),
                        ]),
                    };

                    INDEX_BUILDER.with(|b| {
                        let mut builder = b.borrow_mut();
                        let inner = builder.as_mut().expect("builder set");
                        inner.add(doc).expect("add doc");
                    });

                    doc_id += 1;
                    if doc_id % 10_000 == 0 {
                        doc_pb.set_message(format!("Docs indexed: {}", doc_id));
                    }
                }
            }
        }

        doc_pb.finish_with_message(format!("Docs indexed so far: {}", doc_id));
        file_pb.inc(1);
    }

    file_pb.finish_with_message(format!("Completed files, total docs: {}", doc_id));

    let num_clusters = (doc_id as f64).sqrt().ceil() as u32;
    Ok((doc_id as usize, num_clusters))
}

/// Generate a random embedding vector.
fn random_embedding(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut embedding: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

    // Normalize to unit length
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    embedding
}

//! Wikipedia indexing binary.
//!
//! Downloads the English Wikipedia dataset via HuggingFace and builds a
//! seglens search index in a streaming fashion. Random embeddings are used for
//! demonstration purposes; swap them with a real embedding model for production.
//!
//! Usage (common options):
//! ```bash
//! cargo run --release --bin build-wikipedia-index -- \
//!   --output ./wiki-index --temp ./wiki-temp --limit 50_000 --dim 128
//! ```
//!
//! This writes the index to `./wiki-index/wiki`.

use arrow_array::StringArray;
use futures::stream::{self, StreamExt};
use hf_hub::api::tokio::Api;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::Rng;
use seglens::{Document, IndexBuilder, IndexReader};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

const DOWNLOAD_CONCURRENCY: usize = 4;

#[derive(Debug, Clone)]
struct Config {
    limit: Option<usize>,
    dim: usize,
    output: PathBuf,
    temp: PathBuf,
}

impl Config {
    fn from_args() -> Self {
        let mut limit: Option<usize> = None;
        let mut dim: usize = 128;
        let mut output: Option<PathBuf> = None;
        let mut temp: Option<PathBuf> = None;

        let args: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--limit" => {
                    i += 1;
                    limit = args.get(i).and_then(|s| s.replace('_', "").parse().ok());
                }
                "--dim" => {
                    i += 1;
                    dim = args
                        .get(i)
                        .and_then(|s| s.replace('_', "").parse().ok())
                        .unwrap_or(128);
                }
                "--output" => {
                    i += 1;
                    output = args.get(i).map(PathBuf::from);
                }
                "--temp" => {
                    i += 1;
                    temp = args.get(i).map(PathBuf::from);
                }
                _ => {}
            }
            i += 1;
        }

        let output_dir = output.unwrap_or_else(|| PathBuf::from("./wiki-index"));
        let temp_dir = temp.unwrap_or_else(|| PathBuf::from("./wiki-temp"));

        Self {
            limit,
            dim,
            output: output_dir,
            temp: temp_dir,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_args();

    println!("Wikipedia Search Index Builder");
    println!("===============================");
    println!("Output directory: {}", config.output.display());
    println!("Temp directory: {}", config.temp.display());
    if let Some(n) = config.limit {
        println!("Document limit: {}", n);
    } else {
        println!("Document limit: none (full dataset)");
    }
    println!("Embedding dimension: {}", config.dim);
    println!();

    // Set up storage
    let _ = std::fs::remove_dir_all(&config.output);
    let _ = std::fs::remove_dir_all(&config.temp);
    let store = Arc::new(seglens::local(&config.output)?);

    // Streaming index build: download + parse + add docs directly.
    let (doc_count, num_clusters) =
        download_and_feed(&config.temp, config.dim, config.limit).await?;

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
    let query_embedding = random_embedding(config.dim);
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
        .search_hybrid(
            "artificial intelligence",
            &random_embedding(config.dim),
            3,
            2,
        )
        .await?;

    for (i, result) in results.iter().enumerate() {
        let title = result
            .attributes
            .get("title")
            .map(|s| s.as_str())
            .unwrap_or("Unknown");
        println!("  {}. [score: {:.4}] {}", i + 1, result.score, title);
    }

    println!("\nIndex location: {}", config.output.join("wiki").display());
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

    if let Some(limit) = limit {
        println!("Limiting to first {} documents", limit);
    }

    let pb = ProgressBar::new(parquet_files.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} left) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let total_docs = Arc::new(tokio::sync::Mutex::new(0usize));
    let total_clusters = Arc::new(tokio::sync::Mutex::new(0u32));

    stream::iter(parquet_files)
        .map(|path| {
            let repo = repo.clone();
            let pb = pb.clone();
            let total_docs = total_docs.clone();
            let total_clusters = total_clusters.clone();
            async move {
                pb.set_message(path.clone());
                let parquet_path = repo.get(&path).await.expect("download parquet");
                tokio::time::sleep(Duration::from_millis(200)).await; // allow progress bars to render

                let remaining_limit = if let Some(limit) = limit {
                    let guard = total_docs.lock().await;
                    if *guard >= limit {
                        pb.inc(1);
                        return Ok::<(), Box<dyn std::error::Error>>(());
                    }
                    let remaining = limit - *guard;
                    drop(guard);
                    Some(remaining)
                } else {
                    None
                };

                let docs = tokio::task::spawn_blocking({
                    let parquet_path = parquet_path.clone();
                    move || parse_parquet(&parquet_path, dim, remaining_limit)
                })
                .await
                .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?
                .map_err(|e| -> Box<dyn std::error::Error> { e })?;

                let doc_len = docs.len();
                if doc_len == 0 {
                    pb.inc(1);
                    return Ok(());
                }

                // Estimate cluster count proportional to docs
                let clusters = ((doc_len as f32 / 10_000.0).ceil() as u32).max(2);

                INDEX_BUILDER.with(|builder| {
                    let mut builder = builder.borrow_mut();
                    let builder = builder.as_mut().expect("builder set");
                    let mut total_docs = total_docs.blocking_lock();
                    let base_id = *total_docs as u32;
                    *total_docs += doc_len;

                    for (offset, mut doc) in docs.into_iter().enumerate() {
                        doc.id = base_id + offset as u32;
                        builder.add(doc).expect("add doc");
                    }

                    let mut total_clusters = total_clusters.blocking_lock();
                    *total_clusters += clusters;
                });

                pb.inc(1);
                Ok::<(), Box<dyn std::error::Error>>(())
            }
        })
        .buffer_unordered(DOWNLOAD_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;

    pb.finish_with_message("Download + ingestion complete");

    let doc_count = *total_docs.lock().await;
    let avg_clusters = (*total_clusters.lock().await).max(2);
    Ok((doc_count, avg_clusters))
}

fn parse_parquet(
    parquet_path: &PathBuf,
    dim: usize,
    limit: Option<usize>,
) -> Result<Vec<Document>, Box<dyn std::error::Error + Send + Sync>> {
    let file = File::open(parquet_path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut documents = Vec::new();
    for batch in reader {
        let batch = batch?;
        let title_array = batch
            .column_by_name("title")
            .expect("title column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("title string array");
        let text_array = batch
            .column_by_name("text")
            .expect("text column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("text string array");

        for i in 0..batch.num_rows() {
            if let Some(limit) = limit {
                if documents.len() >= limit {
                    return Ok(documents);
                }
            }

            let title = title_array.value(i).to_string();
            let text = text_array.value(i).to_string();
            let embedding = random_embedding(dim);

            let mut attributes = HashMap::new();
            attributes.insert("title".to_string(), title.clone());

            documents.push(Document {
                id: documents.len() as u32,
                text,
                embedding,
                attributes,
            });
        }
    }

    Ok(documents)
}

fn random_embedding(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(0.0..1.0)).collect()
}

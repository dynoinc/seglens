//! Wikipedia indexing example.
//!
//! Downloads English Wikipedia articles from HuggingFace (wikimedia/wikipedia dataset)
//! using the hf-hub crate and builds a search index.
//! Uses random embeddings for demonstration (in production, use a real embedding model).
//!
//! Usage:
//!   cargo run --release --example wikipedia -- [--count N] [--dim D]
//!
//! Options:
//!   --count N   Number of documents to index (default: 10000)
//!   --dim D     Embedding dimension (default: 128)
//!
//! Note: The English Wikipedia dataset is large (~450MB per parquet file).
//! Use --release for better performance when indexing many documents.

use arrow_array::StringArray;
use bytes::Bytes;
use hf_hub::api::tokio::Api;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::Rng;
use seglens::{Document, IndexBuilder, IndexReader};
use std::collections::HashMap;
use std::sync::Arc;

/// Wikipedia article parsed from parquet.
#[derive(Debug)]
struct WikiArticle {
    id: String,
    url: String,
    title: String,
    text: String,
}

/// Download and parse Wikipedia articles from HuggingFace using hf-hub.
/// Downloads parquet files sequentially until we have enough articles.
async fn download_wikipedia(count: usize) -> Result<Vec<WikiArticle>, Box<dyn std::error::Error>> {
    println!("Downloading English Wikipedia from HuggingFace (wikimedia/wikipedia)...");
    println!("Using hf-hub crate for efficient caching and downloads");
    println!("Target: {} articles", count);
    println!();

    // Initialize HuggingFace Hub API
    let api = Api::new()?;
    let repo = api.dataset("wikimedia/wikipedia".to_string());

    // Get list of parquet files from the repository
    println!("Fetching file list from HuggingFace...");
    let info = repo.info().await?;

    let mut parquet_files: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|f| f.starts_with("20231101.en/") && f.ends_with(".parquet"))
        .collect();

    parquet_files.sort();
    println!(
        "Found {} parquet files for English Wikipedia",
        parquet_files.len()
    );
    println!();

    let mut all_articles = Vec::new();
    let mut file_index = 0;
    let total_files = parquet_files.len();

    while all_articles.len() < count && file_index < total_files {
        let filename = &parquet_files[file_index];

        println!(
            "Downloading file {}/{} (have {}/{} articles)...",
            file_index + 1,
            total_files,
            all_articles.len(),
            count
        );
        println!("  File: {}", filename);

        match repo.get(filename).await {
            Ok(path) => {
                println!("  Cached at: {}", path.display());

                // Read the file
                let file_bytes = std::fs::read(&path)?;
                let bytes = Bytes::from(file_bytes);
                println!("  Size: {} MB", bytes.len() / (1024 * 1024));

                let remaining = count - all_articles.len();
                match parse_parquet(bytes, remaining) {
                    Ok(articles) => {
                        println!("  Parsed {} articles from this file", articles.len());
                        all_articles.extend(articles);
                    }
                    Err(e) => {
                        println!("  Warning: Failed to parse parquet: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("  Warning: Failed to download: {}", e);
            }
        }

        file_index += 1;
    }

    println!("\nTotal articles downloaded: {}", all_articles.len());
    Ok(all_articles)
}

/// Parse Wikipedia articles from parquet bytes.
fn parse_parquet(
    data: Bytes,
    count: usize,
) -> Result<Vec<WikiArticle>, Box<dyn std::error::Error>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(data)?;

    // Configure batch size for efficient reading
    let reader = builder.with_batch_size(1024).build()?;

    let mut articles = Vec::new();

    for batch_result in reader {
        let batch = batch_result?;

        // Get columns by name - wikimedia/wikipedia schema
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

        // Extract values
        if let (Some(ids), Some(titles), Some(texts), Some(urls)) =
            (id_col, title_col, text_col, url_col)
        {
            for i in 0..batch.num_rows() {
                if articles.len() >= count {
                    break;
                }

                let id = ids.value(i).to_string();
                let title = titles.value(i).to_string();
                let text = texts.value(i).to_string();
                let url = urls.value(i).to_string();

                // Skip very short articles
                if text.len() >= 100 {
                    articles.push(WikiArticle {
                        id,
                        url,
                        title,
                        text,
                    });
                }
            }
        }

        if articles.len() >= count {
            break;
        }
    }

    Ok(articles)
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut count: usize = 10000;
    let mut dim: usize = 128;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--count" => {
                i += 1;
                count = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(10000);
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
    println!("Documents: {}", count);
    println!("Embedding dimension: {}", dim);
    println!();

    // Download articles
    let articles = download_wikipedia(count).await?;

    if articles.is_empty() {
        return Err("No articles downloaded".into());
    }

    // Set up storage
    let index_dir = std::env::temp_dir().join("seglens-wikipedia-index");
    let _ = std::fs::remove_dir_all(&index_dir);
    let store = Arc::new(seglens::local(&index_dir)?);

    // Build the index
    println!("\nBuilding index...");
    let num_clusters = (articles.len() as f64).sqrt().ceil() as u32;
    let mut builder = IndexBuilder::new(dim as u32, num_clusters.max(2));

    let pb = ProgressBar::new(articles.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    for (i, article) in articles.iter().enumerate() {
        let doc = Document {
            id: i as u32,
            text: article.text.clone(),
            embedding: random_embedding(dim),
            attributes: HashMap::from([
                ("title".to_string(), article.title.clone()),
                ("url".to_string(), article.url.clone()),
                ("source_id".to_string(), article.id.clone()),
            ]),
        };
        builder.add(doc)?;
        pb.inc(1);
    }
    pb.finish_with_message("Documents added");

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

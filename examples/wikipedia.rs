//! Wikipedia indexing example.
//!
//! Downloads Simple English Wikipedia articles from HuggingFace (parquet format)
//! and builds a search index.
//! Uses random embeddings for demonstration (in production, use a real embedding model).
//!
//! Usage:
//!   cargo run --example wikipedia -- [--count N] [--dim D]
//!
//! Options:
//!   --count N   Number of documents to index (default: 1000)
//!   --dim D     Embedding dimension (default: 128)

use arrow_array::StringArray;
use bytes::Bytes;
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

/// Download and parse Wikipedia articles from HuggingFace parquet file.
async fn download_wikipedia(count: usize) -> Result<Vec<WikiArticle>, Box<dyn std::error::Error>> {
    println!("Downloading Simple English Wikipedia from HuggingFace...");
    println!("(This may take a moment - the parquet file is ~80MB)");

    // Simple English Wikipedia parquet file from HuggingFace
    let url = "https://huggingface.co/datasets/wikipedia/resolve/main/data/20220301.simple/train-00000-of-00001.parquet";

    let response = reqwest::get(url).await?;

    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
    }

    let bytes = response.bytes().await?;
    println!(
        "Downloaded {} MB, parsing parquet...",
        bytes.len() / (1024 * 1024)
    );

    parse_parquet(bytes, count)
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

        // Get columns by name
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

    println!("Parsed {} articles from parquet", articles.len());
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
    let mut count: usize = 1000;
    let mut dim: usize = 128;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--count" => {
                i += 1;
                count = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1000);
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
    let queries = ["history", "science", "united states", "music"];

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
    println!("Searching for \"england\" with random embedding...");
    let results = reader
        .search_hybrid("england", &random_embedding(dim), 3, 2)
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

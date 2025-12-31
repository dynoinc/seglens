//! Parquet indexing binary.
//!
//! Builds a seglens search index from a directory of parquet files that contain
//! three columns:
//! - `id` (string)
//! - `text` (string)
//! - `embeddings` (list or fixed-size list of floats)
//!
//! Usage (common options):
//! ```bash
//! cargo run --release --bin build-parquet-index -- \
//!   --input ./data/wiki-parquet --output ./wiki-index --temp ./wiki-temp --index-name wiki --limit 50_000
//! ```
//!
//! This writes the index to `./wiki-index/wiki`.

use arrow_array::{Array, Float32Array, Float64Array, LargeListArray, ListArray, StringArray};
use arrow_array::{FixedSizeListArray, RecordBatch};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use seglens::{Document, IndexBuilder, IndexReader};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Config {
    input: PathBuf,
    output: PathBuf,
    temp: PathBuf,
    index_name: String,
    limit: Option<usize>,
}

impl Config {
    fn from_args() -> Result<Self, String> {
        let mut input: Option<PathBuf> = None;
        let mut output: Option<PathBuf> = None;
        let mut temp: Option<PathBuf> = None;
        let mut index_name: Option<String> = None;
        let mut limit: Option<usize> = None;

        let args: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--input" => {
                    i += 1;
                    input = args.get(i).map(PathBuf::from);
                }
                "--output" => {
                    i += 1;
                    output = args.get(i).map(PathBuf::from);
                }
                "--temp" => {
                    i += 1;
                    temp = args.get(i).map(PathBuf::from);
                }
                "--index-name" => {
                    i += 1;
                    index_name = args.get(i).cloned();
                }
                "--limit" => {
                    i += 1;
                    limit = args.get(i).and_then(|s| s.replace('_', "").parse().ok());
                }
                flag => {
                    return Err(format!("Unknown flag: {}", flag));
                }
            }
            i += 1;
        }

        let input_dir = input.ok_or("--input is required")?;
        if !input_dir.is_dir() {
            return Err(format!(
                "--input {} is not a directory",
                input_dir.display()
            ));
        }

        Ok(Self {
            input: input_dir,
            output: output.unwrap_or_else(|| PathBuf::from("./parquet-index")),
            temp: temp.unwrap_or_else(|| PathBuf::from("./parquet-temp")),
            index_name: index_name.unwrap_or_else(|| "dataset".to_string()),
            limit,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_args().map_err(|e| format!("invalid arguments: {}", e))?;

    println!("Parquet Search Index Builder");
    println!("============================");
    println!("Input directory: {}", config.input.display());
    println!("Output directory: {}", config.output.display());
    println!("Temp directory: {}", config.temp.display());
    println!("Index name: {}", config.index_name);
    if let Some(limit) = config.limit {
        println!("Document limit: {}", limit);
    }
    println!();

    // Set up storage
    let _ = std::fs::remove_dir_all(&config.output);
    let _ = std::fs::remove_dir_all(&config.temp);
    let store = Arc::new(seglens::local(&config.output)?);

    let parquet_files = collect_parquet_files(&config.input)?;
    if parquet_files.is_empty() {
        return Err("no parquet files found in input directory".into());
    }

    let pb = ProgressBar::new(parquet_files.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} left) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut builder: Option<IndexBuilder> = None;
    let mut doc_id: u32 = 0;
    let mut sample_embedding: Option<Vec<f32>> = None;
    let mut embedding_dim: Option<usize> = None;

    for file in parquet_files {
        pb.set_message(
            file.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        );
        let remaining = config
            .limit
            .map(|limit| limit.saturating_sub(doc_id as usize));
        if let Some(0) = remaining {
            break;
        }

        let ParseResult {
            docs,
            embedding_dim: parsed_dim,
            sample,
        } = parse_parquet(&file, doc_id, remaining, embedding_dim)?;

        if docs.is_empty() {
            pb.inc(1);
            continue;
        }

        if builder.is_none() {
            embedding_dim = Some(parsed_dim);
            builder = Some(
                IndexBuilder::new(parsed_dim as u32, 0)
                    .with_temp_dir(&config.temp)
                    .expect("temp dir should be writable"),
            );
        }

        if sample_embedding.is_none() {
            sample_embedding = sample;
        }

        if let Some(builder) = builder.as_mut() {
            for doc in docs {
                builder.add(doc)?;
            }
        }

        doc_id = builder.as_ref().map(|b| b.doc_count()).unwrap_or(doc_id);
        pb.inc(1);
    }

    pb.finish_with_message("Ingestion complete");

    let mut builder = builder.ok_or("no documents were indexed")?;
    let doc_count = builder.doc_count();
    let num_clusters = ((doc_count as f32 / 10_000.0).ceil() as u32).max(2);
    builder.set_num_clusters(num_clusters);

    println!(
        "\nBuilding index for {} documents ({} clusters)...",
        doc_count, num_clusters
    );
    builder.build(store.as_ref(), &config.index_name).await?;
    println!(
        "Index built successfully at {}/{}",
        config.output.display(),
        config.index_name
    );

    let reader = IndexReader::open(store, &config.index_name).await?;
    println!("Opened index with {} documents", reader.doc_count());

    println!("\n--- Sample lexical search ---");
    let results = reader.search_lexical("machine learning", 3).await?;
    for (i, result) in results.iter().enumerate() {
        println!(
            "{}. {:.4} {}",
            i + 1,
            result.score,
            result
                .attributes
                .get("id")
                .unwrap_or(&"<missing>".to_string())
        );
        println!("   {}...", result.text.chars().take(96).collect::<String>());
    }

    if let Some(embedding) = sample_embedding {
        println!("\n--- Sample vector search ---");
        let results = reader.search_vector(&embedding, 3, 2).await?;
        for (i, result) in results.iter().enumerate() {
            println!(
                "{}. {:.4} {}",
                i + 1,
                result.score,
                result
                    .attributes
                    .get("id")
                    .unwrap_or(&"<missing>".to_string())
            );
        }
    }

    println!("\nDone!");
    Ok(())
}

#[derive(Debug)]
struct ParseResult {
    docs: Vec<Document>,
    embedding_dim: usize,
    sample: Option<Vec<f32>>,
}

fn collect_parquet_files(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("parquet"))
        .collect();
    files.sort();
    Ok(files)
}

fn parse_parquet(
    parquet_path: &Path,
    starting_doc_id: u32,
    limit: Option<usize>,
    expected_dim: Option<usize>,
) -> Result<ParseResult, Box<dyn std::error::Error>> {
    let file = File::open(parquet_path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut documents = Vec::new();
    let mut sample: Option<Vec<f32>> = None;
    let mut embedding_dim: Option<usize> = None;
    let mut next_id = starting_doc_id;

    for batch in reader {
        let batch = batch?;
        let ids = as_string_array(&batch, "id")?;
        let texts = as_string_array(&batch, "text")?;
        let embeddings = batch
            .column_by_name("embeddings")
            .ok_or("missing embeddings column")?;

        for row in 0..batch.num_rows() {
            if let Some(limit) = limit {
                if documents.len() >= limit {
                    break;
                }
            }

            let embedding = extract_embedding(embeddings.as_ref(), row)?;
            let dim = embedding.len();
            if dim == 0 {
                return Err("embedding must not be empty".into());
            }

            if embedding_dim.is_none() {
                embedding_dim = Some(dim);
            }

            if let Some(expected) = expected_dim {
                if dim != expected {
                    return Err(format!(
                        "embedding dimension mismatch: expected {}, found {} in {}",
                        expected,
                        dim,
                        parquet_path.display()
                    )
                    .into());
                }
            }

            if sample.is_none() {
                sample = Some(embedding.clone());
            }

            let mut attributes = HashMap::new();
            attributes.insert("id".to_string(), ids.value(row).to_string());

            documents.push(Document {
                id: next_id,
                text: texts.value(row).to_string(),
                embedding,
                attributes,
            });
            next_id += 1;
        }

        if let Some(limit) = limit {
            if documents.len() >= limit {
                break;
            }
        }
    }

    let embedding_dim = embedding_dim.ok_or("no embeddings found in file")?;

    Ok(ParseResult {
        docs: documents,
        embedding_dim,
        sample,
    })
}

fn as_string_array<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a StringArray, Box<dyn std::error::Error>> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| format!("missing {} column", name))
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    column
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| format!("{} column must be a string array", name))
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })
}

fn extract_embedding(
    array: &dyn Array,
    row: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if let Some(fixed) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let values = fixed.values();
        let dim = fixed.value_length() as usize;
        let start = row * dim;
        let end = start + dim;
        return float_values(&values, start, end);
    }

    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        let offsets = list.value_offsets();
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        return float_values(list.values(), start, end);
    }

    if let Some(list) = array.as_any().downcast_ref::<LargeListArray>() {
        let offsets = list.value_offsets();
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        return float_values(list.values(), start, end);
    }

    Err("embeddings column must be a (fixed-size) list of floats".into())
}

fn float_values(
    values: &dyn Array,
    start: usize,
    end: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if let Some(arr) = values.as_any().downcast_ref::<Float32Array>() {
        return Ok(arr.values()[start..end].to_vec());
    }

    if let Some(arr) = values.as_any().downcast_ref::<Float64Array>() {
        return Ok(arr.values()[start..end].iter().map(|v| *v as f32).collect());
    }

    Err("embedding list must contain f32 or f64 values".into())
}

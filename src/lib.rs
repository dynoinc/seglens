//! # seglens
//!
//! A Rust library for building and querying hybrid (semantic + lexical) search indexes
//! optimized for object storage.
//!
//! ## Features
//!
//! - **Lexical search**: BM25-based text search using inverted indexes
//! - **Vector search**: IVF (Inverted File) based approximate nearest neighbor search
//! - **Hybrid search**: RRF (Reciprocal Rank Fusion) to combine lexical and vector results
//! - **Object storage optimized**: Stores index in few large segments (~64MB) with on-demand fetching
//! - **Zero-copy deserialization**: Uses rkyv for efficient data loading
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use seglens::{Document, IndexBuilder, IndexReader, local};
//! use std::collections::HashMap;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a local storage backend
//!     let store = Arc::new(local("/tmp/seglens-index")?);
//!
//!     // Build an index
//!     let mut builder = IndexBuilder::new(4, 2); // 4-dim embeddings, 2 clusters
//!     builder.add(Document {
//!         id: 0,
//!         text: "hello world".to_string(),
//!         embedding: vec![1.0, 0.0, 0.0, 0.0],
//!         attributes: HashMap::new(),
//!     })?;
//!     builder.build(store.as_ref(), "v1").await?;
//!
//!     // Query the index
//!     let reader = IndexReader::open(store, "v1").await?;
//!     let results = reader.search_lexical("hello", 10).await?;
//!     let results = reader.search_vector(&[1.0, 0.0, 0.0, 0.0], 10, 2).await?;
//!     let results = reader.search_hybrid("hello", &[1.0, 0.0, 0.0, 0.0], 10, 2).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The index is stored in object storage with the following structure:
//!
//! ```text
//! {prefix}/
//! ├── manifest.bin     # Config, doc count, doc pointers (~KB)
//! ├── lexical.idx      # Vocab, posting refs, BM25 stats (~MB)
//! ├── vector.idx       # IVF centroids, cluster refs (~MB)
//! └── segments/
//!     ├── 0000.dat     # Posting lists, cluster vectors, doc data
//!     ├── 0001.dat     # Target ~64MB per segment
//!     └── ...
//! ```
//!
//! At startup, only the metadata files are loaded. Segment data is fetched
//! on-demand via range requests, making this ideal for object storage.

pub mod builder;
pub mod error;
pub mod hybrid;
pub mod lexical;
pub mod object_store;
pub mod reader;
pub mod storage;
pub mod types;
pub mod vector;

// Re-export commonly used types
pub use builder::IndexBuilder;
pub use error::{Error, IndexError, SearchError};
pub use hybrid::{fuse_hybrid, rrf_fuse, rrf_score};
pub use lexical::{bm25_score, tokenize, LexicalIndex, LexicalIndexBuilder, PostingList};
pub use reader::IndexReader;
pub use storage::{BlobStore, StorageError, StorageResult};
pub use types::{DocId, Document, SearchResult, SegmentPtr, StoredDocument, TermId};
pub use vector::{ClusterData, VectorIndex, VectorIndexBuilder};

// Re-export convenience functions
pub use object_store::local;

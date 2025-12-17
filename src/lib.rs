//! seglens - A Rust library for building and querying hybrid search indexes.
//!
//! This library provides tools for building and querying search indexes that combine:
//! - **Lexical search**: BM25-based text search using inverted indexes
//! - **Vector search**: IVF (Inverted File) based approximate nearest neighbor search
//! - **Hybrid search**: RRF (Reciprocal Rank Fusion) to combine lexical and vector results
//!
//! The index is designed to be stored in object storage (S3, GCS, local filesystem)
//! with minimal metadata loaded at startup and on-demand segment fetching via range requests.

pub mod builder;
pub mod hybrid;
pub mod lexical;
pub mod object_store;
pub mod reader;
pub mod storage;
pub mod types;
pub mod vector;

// Re-export commonly used types
pub use builder::IndexBuilder;
pub use hybrid::{fuse_hybrid, rrf_fuse, rrf_score};
pub use lexical::{bm25_score, tokenize, LexicalIndex, LexicalIndexBuilder, PostingList};
pub use reader::IndexReader;
pub use storage::{BlobStore, StorageError, StorageResult};
pub use types::{DocId, Document, SearchResult, SegmentPtr, StoredDocument, TermId};
pub use vector::{ClusterData, VectorIndex, VectorIndexBuilder};

// Re-export convenience functions
pub use object_store::local;

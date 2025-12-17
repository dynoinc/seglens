//! Error types for the seglens library.

use thiserror::Error;

/// Top-level error type for seglens operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Storage-related errors.
    #[error("storage error: {0}")]
    Storage(#[from] crate::storage::StorageError),

    /// Index building errors.
    #[error("index error: {0}")]
    Index(#[from] IndexError),

    /// Search errors.
    #[error("search error: {0}")]
    Search(#[from] SearchError),

    /// I/O errors.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors that occur during index building.
#[derive(Error, Debug)]
pub enum IndexError {
    /// No documents were provided for indexing.
    #[error("no documents to index")]
    EmptyIndex,

    /// Invalid embedding dimension.
    #[error("invalid embedding dimension: expected {expected}, got {got}")]
    DimensionMismatch { expected: u32, got: u32 },

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Document with duplicate ID.
    #[error("duplicate document ID: {0}")]
    DuplicateId(u32),
}

/// Errors that occur during search operations.
#[derive(Error, Debug)]
pub enum SearchError {
    /// Invalid query.
    #[error("invalid query: {0}")]
    InvalidQuery(String),

    /// Deserialization error when loading index data.
    #[error("deserialization error: {0}")]
    Deserialization(String),

    /// Segment not found.
    #[error("segment not found: {0}")]
    SegmentNotFound(String),
}

/// Result type for seglens operations.
pub type Result<T> = std::result::Result<T, Error>;

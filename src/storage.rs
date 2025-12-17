//! Storage abstraction for the search index.

use async_trait::async_trait;
use bytes::Bytes;
use std::ops::Range;
use thiserror::Error;

/// Storage-related errors.
#[derive(Error, Debug)]
pub enum StorageError {
    /// Object not found in storage.
    #[error("object not found: {0}")]
    NotFound(String),

    /// I/O error during storage operation.
    #[error("storage I/O error: {0}")]
    Io(String),

    /// Error from the underlying storage backend.
    #[error("backend error: {0}")]
    Backend(String),
}

/// Result type for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;

/// Abstraction over blob storage backends (S3, GCS, local filesystem).
#[async_trait]
pub trait BlobStore: Send + Sync {
    /// Read a byte range from an object.
    async fn get_range(&self, path: &str, range: Range<u64>) -> StorageResult<Bytes>;

    /// Read an entire object.
    async fn get(&self, path: &str) -> StorageResult<Bytes>;

    /// Write data to an object (overwrites if exists).
    async fn put(&self, path: &str, data: Bytes) -> StorageResult<()>;

    /// List objects with a given prefix.
    async fn list(&self, prefix: &str) -> StorageResult<Vec<String>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_error_display() {
        let err = StorageError::NotFound("test.bin".to_string());
        assert!(err.to_string().contains("test.bin"));

        let err = StorageError::Io("connection refused".to_string());
        assert!(err.to_string().contains("connection refused"));

        let err = StorageError::Backend("S3 error".to_string());
        assert!(err.to_string().contains("S3 error"));
    }
}

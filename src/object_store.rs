//! object_store adapter implementing the BlobStore trait.

use crate::storage::{BlobStore, StorageError, StorageResult};
use async_trait::async_trait;
use bytes::Bytes;
use object_store::{local::LocalFileSystem, ObjectStore};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

/// BlobStore implementation backed by the object_store crate.
pub struct ObjectStoreBackend {
    store: Arc<dyn ObjectStore>,
}

impl ObjectStoreBackend {
    /// Create a new backend from any object_store implementation.
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }

    /// Create a backend for local filesystem storage.
    pub fn local(path: impl Into<PathBuf>) -> StorageResult<Self> {
        let path = path.into();
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&path)
            .map_err(|e| StorageError::Io(format!("failed to create directory: {}", e)))?;

        let store = LocalFileSystem::new_with_prefix(&path)
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        Ok(Self {
            store: Arc::new(store),
        })
    }
}

#[async_trait]
impl BlobStore for ObjectStoreBackend {
    async fn get_range(&self, path: &str, range: Range<u64>) -> StorageResult<Bytes> {
        let location = object_store::path::Path::from(path);
        let opts = object_store::GetOptions {
            range: Some(object_store::GetRange::Bounded(
                range.start as usize..range.end as usize,
            )),
            ..Default::default()
        };

        let result = self
            .store
            .get_opts(&location, opts)
            .await
            .map_err(|e| match e {
                object_store::Error::NotFound { .. } => StorageError::NotFound(path.to_string()),
                _ => StorageError::Backend(e.to_string()),
            })?;

        result
            .bytes()
            .await
            .map_err(|e| StorageError::Io(e.to_string()))
    }

    async fn get(&self, path: &str) -> StorageResult<Bytes> {
        let location = object_store::path::Path::from(path);

        let result = self.store.get(&location).await.map_err(|e| match e {
            object_store::Error::NotFound { .. } => StorageError::NotFound(path.to_string()),
            _ => StorageError::Backend(e.to_string()),
        })?;

        result
            .bytes()
            .await
            .map_err(|e| StorageError::Io(e.to_string()))
    }

    async fn put(&self, path: &str, data: Bytes) -> StorageResult<()> {
        let location = object_store::path::Path::from(path);
        let payload = object_store::PutPayload::from_bytes(data);

        self.store
            .put(&location, payload)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        Ok(())
    }

    async fn list(&self, prefix: &str) -> StorageResult<Vec<String>> {
        use futures::TryStreamExt;

        let prefix_path = if prefix.is_empty() {
            None
        } else {
            Some(object_store::path::Path::from(prefix))
        };

        let stream = self.store.list(prefix_path.as_ref());
        let results: Vec<_> = stream
            .try_collect()
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|meta| meta.location.to_string())
            .collect())
    }
}

/// Convenience function to create a local filesystem backend.
pub fn local(path: impl Into<PathBuf>) -> StorageResult<ObjectStoreBackend> {
    ObjectStoreBackend::local(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_local_put_get() {
        let tmp = TempDir::new().unwrap();
        let store = local(tmp.path()).unwrap();

        let data = Bytes::from("hello world");
        store.put("test.txt", data.clone()).await.unwrap();

        let retrieved = store.get("test.txt").await.unwrap();
        assert_eq!(retrieved, data);
    }

    #[tokio::test]
    async fn test_local_get_range() {
        let tmp = TempDir::new().unwrap();
        let store = local(tmp.path()).unwrap();

        let data = Bytes::from("hello world");
        store.put("test.txt", data).await.unwrap();

        let range = store.get_range("test.txt", 0..5).await.unwrap();
        assert_eq!(&range[..], b"hello");
    }

    #[tokio::test]
    async fn test_local_not_found() {
        let tmp = TempDir::new().unwrap();
        let store = local(tmp.path()).unwrap();

        let result = store.get("nonexistent.txt").await;
        assert!(matches!(result, Err(StorageError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_local_list() {
        let tmp = TempDir::new().unwrap();
        let store = local(tmp.path()).unwrap();

        store.put("prefix/a.txt", Bytes::from("a")).await.unwrap();
        store.put("prefix/b.txt", Bytes::from("b")).await.unwrap();
        store.put("other/c.txt", Bytes::from("c")).await.unwrap();

        let mut files = store.list("prefix").await.unwrap();
        files.sort();
        assert_eq!(files.len(), 2);
        assert!(files[0].contains("a.txt"));
        assert!(files[1].contains("b.txt"));
    }
}

//! Index reader: opens and queries the search index.

use crate::builder::Manifest;
use crate::cache::{CacheResult, SegmentCache};
use crate::hybrid::fuse_hybrid;
use crate::lexical::{bm25_score, tokenize, LexicalIndex, PostingList};
use crate::storage::{BlobStore, StorageError, StorageResult};
use crate::types::{DocId, SearchResult, SegmentPtr, StoredDocument};
use crate::vector::{ClusterData, VectorIndex};
use rkyv::rancor::Error as RkyvError;
use std::collections::HashMap;
use std::sync::Arc;

/// Index reader for querying the search index.
pub struct IndexReader {
    /// Storage backend.
    store: Arc<dyn BlobStore>,
    /// Index prefix in storage.
    prefix: String,
    /// Manifest with document pointers.
    manifest: Manifest,
    /// Lexical index metadata.
    lexical_index: LexicalIndex,
    /// Vector index metadata.
    vector_index: VectorIndex,
    /// Vocabulary lookup map (term -> (term_id, entry)).
    vocab_map: HashMap<String, (u32, crate::lexical::VocabEntry)>,
    /// Segment cache for reducing repeated reads.
    cache: SegmentCache,
}

impl IndexReader {
    /// Open an index from storage.
    pub async fn open(store: Arc<dyn BlobStore>, prefix: &str) -> StorageResult<Self> {
        // Load manifest
        let manifest_bytes = store.get(&format!("{}/manifest.bin", prefix)).await?;
        let manifest = deserialize_manifest(&manifest_bytes)?;

        // Load lexical index
        let lexical_bytes = store.get(&format!("{}/lexical.idx", prefix)).await?;
        let lexical_index = deserialize_lexical_index(&lexical_bytes)?;

        // Load vector index
        let vector_bytes = store.get(&format!("{}/vector.idx", prefix)).await?;
        let vector_index = deserialize_vector_index(&vector_bytes)?;

        // Build vocab map for fast lookups
        let vocab_map = lexical_index.vocab_map();

        Ok(Self {
            store,
            prefix: prefix.to_string(),
            manifest,
            lexical_index,
            vector_index,
            vocab_map,
            cache: SegmentCache::new(),
        })
    }

    /// Get the number of indexed documents.
    pub fn doc_count(&self) -> u32 {
        self.manifest.doc_count
    }

    /// Fetch segment data for a given pointer, using cache with 2MB aligned reads.
    async fn fetch_segment(&self, ptr: &SegmentPtr) -> StorageResult<bytes::Bytes> {
        let start = ptr.offset;
        let end = ptr.offset + ptr.length as u64;

        // Check cache first
        match self.cache.get(ptr.segment_id, start, end) {
            CacheResult::Hit(data) => Ok(data),
            CacheResult::Miss(missing_ranges) => {
                // Fetch missing chunks from storage
                let path = format!("{}/segments/{:04}.dat", self.prefix, ptr.segment_id);

                for (chunk_start, chunk_end) in missing_ranges {
                    let chunk_data = self.store.get_range(&path, chunk_start..chunk_end).await?;
                    self.cache.put(ptr.segment_id, chunk_start, chunk_data);
                }

                // Now get from cache (should hit)
                match self.cache.get(ptr.segment_id, start, end) {
                    CacheResult::Hit(data) => Ok(data),
                    CacheResult::Miss(_) => {
                        // Fallback to direct fetch (shouldn't happen)
                        self.store.get_range(&path, start..end).await
                    }
                }
            }
        }
    }

    /// Fetch and deserialize a posting list.
    async fn fetch_posting_list(&self, ptr: &SegmentPtr) -> StorageResult<PostingList> {
        let bytes = self.fetch_segment(ptr).await?;
        deserialize_posting_list(&bytes)
    }

    /// Fetch and deserialize cluster data.
    async fn fetch_cluster_data(&self, ptr: &SegmentPtr) -> StorageResult<ClusterData> {
        let bytes = self.fetch_segment(ptr).await?;
        deserialize_cluster_data(&bytes)
    }

    /// Fetch and deserialize a stored document.
    async fn fetch_document(&self, ptr: &SegmentPtr) -> StorageResult<StoredDocument> {
        let bytes = self.fetch_segment(ptr).await?;
        deserialize_stored_document(&bytes)
    }

    /// Perform lexical (BM25) search.
    ///
    /// # Arguments
    /// * `query` - Text query to search for
    /// * `top_k` - Maximum number of results to return
    pub async fn search_lexical(
        &self,
        query: &str,
        top_k: usize,
    ) -> StorageResult<Vec<SearchResult>> {
        let results = self.search_lexical_raw(query, top_k).await?;
        self.fetch_results(&results).await
    }

    /// Perform vector (IVF) search.
    ///
    /// # Arguments
    /// * `query` - Query embedding vector
    /// * `top_k` - Maximum number of results to return
    /// * `n_probe` - Number of clusters to search
    pub async fn search_vector(
        &self,
        query: &[f32],
        top_k: usize,
        n_probe: usize,
    ) -> StorageResult<Vec<SearchResult>> {
        let results = self.search_vector_raw(query, top_k, n_probe).await?;
        self.fetch_results(&results).await
    }

    /// Perform hybrid (lexical + vector) search with RRF fusion.
    ///
    /// # Arguments
    /// * `query` - Text query for lexical search
    /// * `embedding` - Query embedding for vector search
    /// * `top_k` - Maximum number of results to return
    /// * `n_probe` - Number of clusters to search for vector
    pub async fn search_hybrid(
        &self,
        query: &str,
        embedding: &[f32],
        top_k: usize,
        n_probe: usize,
    ) -> StorageResult<Vec<SearchResult>> {
        // Run both searches
        let lexical_results = self.search_lexical_raw(query, top_k * 2).await?;
        let vector_results = self
            .search_vector_raw(embedding, top_k * 2, n_probe)
            .await?;

        // Fuse with RRF
        let fused = fuse_hybrid(&lexical_results, &vector_results, top_k);

        // Fetch document metadata
        self.fetch_results(&fused).await
    }

    /// Internal: lexical search returning (doc_id, score) pairs.
    async fn search_lexical_raw(
        &self,
        query: &str,
        top_k: usize,
    ) -> StorageResult<Vec<(DocId, f32)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let mut doc_scores: HashMap<DocId, f32> = HashMap::new();

        for token in &tokens {
            if let Some((_, entry)) = self.vocab_map.get(token) {
                let posting_list = self.fetch_posting_list(&entry.posting_ptr).await?;

                for posting in &posting_list.postings {
                    let doc_len = self
                        .lexical_index
                        .doc_lengths
                        .get(posting.doc_id as usize)
                        .copied()
                        .unwrap_or(0);
                    let score = bm25_score(
                        posting.tf,
                        doc_len,
                        self.lexical_index.avg_doc_len,
                        self.lexical_index.doc_count,
                        entry.doc_freq,
                    );
                    *doc_scores.entry(posting.doc_id).or_insert(0.0) += score;
                }
            }
        }

        let mut results: Vec<(DocId, f32)> = doc_scores.into_iter().collect();
        top_k_descending(&mut results, top_k);

        Ok(results)
    }

    /// Internal: vector search returning (doc_id, score) pairs.
    async fn search_vector_raw(
        &self,
        query: &[f32],
        top_k: usize,
        n_probe: usize,
    ) -> StorageResult<Vec<(DocId, f32)>> {
        if self.vector_index.num_clusters == 0 || top_k == 0 {
            return Ok(Vec::new());
        }

        let cluster_indices = self.vector_index.find_clusters(query, n_probe);
        let mut all_results: Vec<(DocId, f32)> = Vec::new();

        for cluster_idx in cluster_indices {
            if cluster_idx < self.vector_index.cluster_ptrs.len() {
                let ptr = &self.vector_index.cluster_ptrs[cluster_idx];
                let cluster_data = self.fetch_cluster_data(ptr).await?;
                let results = cluster_data.search(query, top_k);
                all_results.extend(results);
            }
        }

        top_k_descending(&mut all_results, top_k);

        Ok(all_results)
    }

    /// Fetch full SearchResult objects for a list of (doc_id, score) pairs.
    async fn fetch_results(&self, results: &[(DocId, f32)]) -> StorageResult<Vec<SearchResult>> {
        let mut search_results = Vec::with_capacity(results.len());

        for &(doc_id, score) in results {
            if let Some(ptr) = self.manifest.doc_ptrs.get(doc_id as usize) {
                let stored_doc = self.fetch_document(ptr).await?;
                let attributes = stored_doc.attributes_map();
                search_results.push(SearchResult {
                    doc_id,
                    score,
                    text: stored_doc.text,
                    attributes,
                });
            }
        }

        Ok(search_results)
    }
}

/// Retain the top_k entries by score (descending) using partial sorting.
fn top_k_descending(results: &mut Vec<(DocId, f32)>, top_k: usize) {
    if results.is_empty() || top_k == 0 {
        results.clear();
        return;
    }

    if results.len() > top_k {
        let pivot = top_k.saturating_sub(1);
        results.select_nth_unstable_by(pivot, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
    }

    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
}

// Deserialization helpers using rkyv 0.8 API

fn deserialize_manifest(bytes: &[u8]) -> StorageResult<Manifest> {
    rkyv::from_bytes::<Manifest, RkyvError>(bytes)
        .map_err(|e| StorageError::Io(format!("manifest deserialization error: {}", e)))
}

fn deserialize_lexical_index(bytes: &[u8]) -> StorageResult<LexicalIndex> {
    rkyv::from_bytes::<LexicalIndex, RkyvError>(bytes)
        .map_err(|e| StorageError::Io(format!("lexical index deserialization error: {}", e)))
}

fn deserialize_vector_index(bytes: &[u8]) -> StorageResult<VectorIndex> {
    rkyv::from_bytes::<VectorIndex, RkyvError>(bytes)
        .map_err(|e| StorageError::Io(format!("vector index deserialization error: {}", e)))
}

fn deserialize_posting_list(bytes: &[u8]) -> StorageResult<PostingList> {
    rkyv::from_bytes::<PostingList, RkyvError>(bytes)
        .map_err(|e| StorageError::Io(format!("posting list deserialization error: {}", e)))
}

fn deserialize_cluster_data(bytes: &[u8]) -> StorageResult<ClusterData> {
    rkyv::from_bytes::<ClusterData, RkyvError>(bytes)
        .map_err(|e| StorageError::Io(format!("cluster data deserialization error: {}", e)))
}

fn deserialize_stored_document(bytes: &[u8]) -> StorageResult<StoredDocument> {
    rkyv::from_bytes::<StoredDocument, RkyvError>(bytes)
        .map_err(|e| StorageError::Io(format!("stored document deserialization error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::IndexBuilder;
    use crate::object_store::local;
    use crate::types::Document;
    use tempfile::TempDir;

    async fn build_test_index(store: &dyn BlobStore) {
        let mut builder = IndexBuilder::new(4, 2);
        builder
            .add(Document {
                id: 0,
                text: "hello world".to_string(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                attributes: HashMap::from([("type".to_string(), "greeting".to_string())]),
            })
            .unwrap();
        builder
            .add(Document {
                id: 1,
                text: "goodbye world".to_string(),
                embedding: vec![0.0, 1.0, 0.0, 0.0],
                attributes: HashMap::from([("type".to_string(), "farewell".to_string())]),
            })
            .unwrap();
        builder
            .add(Document {
                id: 2,
                text: "hello rust programming".to_string(),
                embedding: vec![0.9, 0.1, 0.0, 0.0],
                attributes: HashMap::from([("type".to_string(), "tech".to_string())]),
            })
            .unwrap();
        builder.build(store, "test").await.unwrap();
    }

    #[tokio::test]
    async fn test_reader_open() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(local(tmp.path()).unwrap());

        build_test_index(store.as_ref()).await;

        let reader = IndexReader::open(store, "test").await.unwrap();
        assert_eq!(reader.doc_count(), 3);
    }

    #[tokio::test]
    async fn test_search_lexical() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(local(tmp.path()).unwrap());

        build_test_index(store.as_ref()).await;

        let reader = IndexReader::open(store, "test").await.unwrap();

        // Search for "hello" - should match docs 0 and 2
        let results = reader.search_lexical("hello", 10).await.unwrap();
        assert!(!results.is_empty());

        let doc_ids: Vec<DocId> = results.iter().map(|r| r.doc_id).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&2));

        // Search for "goodbye" - should match doc 1
        let results = reader.search_lexical("goodbye", 10).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, 1);
    }

    #[tokio::test]
    async fn test_search_vector() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(local(tmp.path()).unwrap());

        build_test_index(store.as_ref()).await;

        let reader = IndexReader::open(store, "test").await.unwrap();

        // Search with embedding similar to doc 0
        let results = reader
            .search_vector(&[1.0, 0.0, 0.0, 0.0], 10, 2)
            .await
            .unwrap();
        assert!(!results.is_empty());

        // Doc 0 should be the closest match
        assert_eq!(results[0].doc_id, 0);
    }

    #[tokio::test]
    async fn test_search_hybrid() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(local(tmp.path()).unwrap());

        build_test_index(store.as_ref()).await;

        let reader = IndexReader::open(store, "test").await.unwrap();

        // Hybrid search for "hello" with embedding similar to doc 0
        let results = reader
            .search_hybrid("hello", &[1.0, 0.0, 0.0, 0.0], 10, 2)
            .await
            .unwrap();

        assert!(!results.is_empty());
        // Doc 0 should rank highly (matches both lexical "hello" and vector similarity)
        assert_eq!(results[0].doc_id, 0);
    }

    #[tokio::test]
    async fn test_search_result_attributes() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(local(tmp.path()).unwrap());

        build_test_index(store.as_ref()).await;

        let reader = IndexReader::open(store, "test").await.unwrap();

        let results = reader.search_lexical("hello", 10).await.unwrap();
        assert!(!results.is_empty());

        // Check that attributes are returned
        let result = results.iter().find(|r| r.doc_id == 0).unwrap();
        assert_eq!(result.attributes.get("type"), Some(&"greeting".to_string()));
    }

    #[tokio::test]
    async fn test_search_empty_query() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(local(tmp.path()).unwrap());

        build_test_index(store.as_ref()).await;

        let reader = IndexReader::open(store, "test").await.unwrap();

        // Empty query should return empty results
        let results = reader.search_lexical("", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_no_matches() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(local(tmp.path()).unwrap());

        build_test_index(store.as_ref()).await;

        let reader = IndexReader::open(store, "test").await.unwrap();

        // Query with no matches
        let results = reader.search_lexical("nonexistent", 10).await.unwrap();
        assert!(results.is_empty());
    }
}

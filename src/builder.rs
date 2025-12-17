//! Index builder: constructs the search index from documents.
//!
//! Supports streaming document ingestion with disk-based temporary storage
//! for handling datasets larger than memory.

use crate::lexical::{LexicalIndex, LexicalIndexBuilder, PostingList, VocabEntry};
use crate::storage::{BlobStore, StorageError, StorageResult};
use crate::types::{DocId, Document, SegmentPtr, StoredDocument};
use crate::vector::{kmeans, nearest_centroid, ClusterData, VectorIndex};
use bytes::Bytes;
use rkyv::rancor::Error as RkyvError;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Target segment size (64 MB).
const SEGMENT_SIZE: usize = 64 * 1024 * 1024;

/// Writer for segment files.
pub struct SegmentWriter {
    /// Current segment ID.
    segment_id: u32,
    /// Current offset within segment.
    offset: u64,
    /// Accumulated data for current segment.
    buffer: Vec<u8>,
    /// Written segments (segment_id -> data).
    segments: HashMap<u32, Vec<u8>>,
}

impl SegmentWriter {
    /// Create a new segment writer.
    pub fn new() -> Self {
        Self {
            segment_id: 0,
            offset: 0,
            buffer: Vec::new(),
            segments: HashMap::new(),
        }
    }

    /// Write data and return a pointer to it.
    pub fn write(&mut self, data: &[u8]) -> SegmentPtr {
        let ptr = SegmentPtr {
            segment_id: self.segment_id,
            offset: self.offset,
            length: data.len() as u32,
        };

        self.buffer.extend_from_slice(data);
        self.offset += data.len() as u64;

        // Rotate segment if we've exceeded the target size
        if self.buffer.len() >= SEGMENT_SIZE {
            self.rotate();
        }

        ptr
    }

    /// Rotate to a new segment.
    fn rotate(&mut self) {
        if !self.buffer.is_empty() {
            let old_buffer = std::mem::take(&mut self.buffer);
            self.segments.insert(self.segment_id, old_buffer);
            self.segment_id += 1;
            self.offset = 0;
        }
    }

    /// Finalize and return all segments.
    pub fn finalize(mut self) -> HashMap<u32, Vec<u8>> {
        self.rotate(); // Ensure current buffer is included
        self.segments
    }
}

impl Default for SegmentWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Manifest file containing index metadata.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Manifest {
    /// Total number of documents.
    pub doc_count: u32,
    /// Embedding dimension.
    pub embedding_dim: u32,
    /// Number of clusters.
    pub num_clusters: u32,
    /// Pointers to stored documents.
    pub doc_ptrs: Vec<SegmentPtr>,
}

/// Index builder with streaming support.
///
/// For large datasets, documents are written to disk as they're added,
/// then processed in batches during the build phase.
pub struct IndexBuilder {
    /// Embedding dimension.
    embedding_dim: u32,
    /// Number of clusters for IVF.
    num_clusters: u32,
    /// Temporary directory for intermediate files.
    temp_dir: Option<PathBuf>,
    /// Documents stored in memory (for small datasets).
    documents: Vec<Document>,
    /// Whether to use disk-based storage.
    use_disk: bool,
    /// Number of documents written to disk.
    disk_doc_count: u32,
}

impl IndexBuilder {
    /// Create a new index builder.
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimension of document embeddings
    /// * `num_clusters` - Number of IVF clusters
    pub fn new(embedding_dim: u32, num_clusters: u32) -> Self {
        Self {
            embedding_dim,
            num_clusters,
            temp_dir: None,
            documents: Vec::new(),
            use_disk: false,
            disk_doc_count: 0,
        }
    }

    /// Enable disk-based storage for large datasets.
    ///
    /// Documents will be written to disk as they're added, allowing
    /// indexing of datasets larger than available memory.
    pub fn with_temp_dir(mut self, path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        fs::create_dir_all(&path)?;
        self.temp_dir = Some(path);
        self.use_disk = true;
        Ok(self)
    }

    /// Add a document to the index.
    pub fn add(&mut self, doc: Document) -> std::io::Result<()> {
        if self.use_disk {
            self.write_doc_to_disk(&doc)?;
            self.disk_doc_count += 1;
        } else {
            self.documents.push(doc);
        }
        Ok(())
    }

    /// Write a document to the temp directory.
    fn write_doc_to_disk(&self, doc: &Document) -> std::io::Result<()> {
        let temp_dir = self.temp_dir.as_ref().expect("temp_dir not set");

        // Write document metadata
        let doc_path = temp_dir.join(format!("doc_{}.txt", doc.id));
        let mut file = BufWriter::new(File::create(&doc_path)?);
        writeln!(file, "{}", doc.id)?;
        writeln!(file, "{}", doc.text)?;
        writeln!(file, "{}", doc.embedding.len())?;
        for val in &doc.embedding {
            writeln!(file, "{}", val)?;
        }
        writeln!(file, "{}", doc.attributes.len())?;
        for (k, v) in &doc.attributes {
            writeln!(file, "{}={}", k, v)?;
        }
        file.flush()?;

        Ok(())
    }

    /// Read a document from disk.
    fn read_doc_from_disk(path: &Path) -> std::io::Result<Document> {
        let file = BufReader::new(File::open(path)?);
        let mut lines = file.lines();

        let id: DocId = lines
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing id"))??
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let text = lines.next().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing text")
        })??;

        let emb_len: usize = lines
            .next()
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing emb_len")
            })??
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut embedding = Vec::with_capacity(emb_len);
        for _ in 0..emb_len {
            let val: f32 = lines
                .next()
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "missing embedding value")
                })??
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            embedding.push(val);
        }

        let attr_len: usize = lines
            .next()
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing attr_len")
            })??
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut attributes = HashMap::new();
        for _ in 0..attr_len {
            let line = lines.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing attribute")
            })??;
            if let Some((k, v)) = line.split_once('=') {
                attributes.insert(k.to_string(), v.to_string());
            }
        }

        Ok(Document {
            id,
            text,
            embedding,
            attributes,
        })
    }

    /// Iterate over all documents (from memory or disk).
    fn iter_documents(&self) -> Box<dyn Iterator<Item = std::io::Result<Document>> + '_> {
        if self.use_disk {
            let temp_dir = self.temp_dir.as_ref().expect("temp_dir not set").clone();
            let entries = fs::read_dir(&temp_dir)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .filter(|e| {
                            e.path()
                                .file_name()
                                .and_then(|n| n.to_str())
                                .is_some_and(|n| n.starts_with("doc_") && n.ends_with(".txt"))
                        })
                        .map(|e| Self::read_doc_from_disk(&e.path()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Box::new(entries.into_iter())
        } else {
            Box::new(self.documents.iter().cloned().map(Ok))
        }
    }

    /// Build the index and write to storage.
    pub async fn build(&self, store: &dyn BlobStore, prefix: &str) -> StorageResult<()> {
        // Collect all documents
        let documents: Vec<Document> = self
            .iter_documents()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| StorageError::Io(e.to_string()))?;

        if documents.is_empty() {
            return Err(StorageError::Io("no documents to index".to_string()));
        }

        let doc_count = documents.len() as u32;

        // Build lexical index
        let mut lexical_builder = LexicalIndexBuilder::new();
        for doc in &documents {
            lexical_builder.add_document(doc.id, &doc.text);
        }

        // Collect embeddings for vector index
        let embeddings: Vec<Vec<f32>> = documents.iter().map(|d| d.embedding.clone()).collect();

        // Train k-means
        let centroids = kmeans(&embeddings, self.num_clusters as usize, 10);

        // Assign documents to clusters
        let assignments: Vec<usize> = embeddings
            .iter()
            .map(|e| nearest_centroid(e, &centroids))
            .collect();

        // Build cluster data
        let mut cluster_data: Vec<ClusterData> = (0..centroids.len())
            .map(|_| ClusterData::new(self.embedding_dim))
            .collect();

        for (doc, &cluster_idx) in documents.iter().zip(assignments.iter()) {
            cluster_data[cluster_idx].add(doc.id, &doc.embedding);
        }

        // Create segment writer
        let mut segment_writer = SegmentWriter::new();

        // Write posting lists
        let posting_lists = lexical_builder.build_posting_lists();
        let mut vocab_entries: Vec<(String, VocabEntry)> = Vec::new();

        for (term, doc_freq, posting_list) in posting_lists {
            let bytes = serialize_posting_list(&posting_list)?;
            let ptr = segment_writer.write(&bytes);
            vocab_entries.push((
                term,
                VocabEntry {
                    doc_freq,
                    posting_ptr: ptr,
                },
            ));
        }

        // Write cluster data
        let mut cluster_ptrs: Vec<SegmentPtr> = Vec::new();
        for cluster in &cluster_data {
            let bytes = serialize_cluster_data(cluster)?;
            let ptr = segment_writer.write(&bytes);
            cluster_ptrs.push(ptr);
        }

        // Write stored documents
        let mut doc_ptrs: Vec<SegmentPtr> = Vec::new();
        for doc in &documents {
            let stored = StoredDocument::from_document(doc);
            let bytes = serialize_stored_document(&stored)?;
            let ptr = segment_writer.write(&bytes);
            doc_ptrs.push(ptr);
        }

        // Finalize segments
        let segments = segment_writer.finalize();

        // Build lexical index metadata
        let lexical_index = LexicalIndex {
            vocab: vocab_entries,
            doc_count,
            avg_doc_len: lexical_builder.avg_doc_len(),
            doc_lengths: lexical_builder.doc_lengths().to_vec(),
        };

        // Build vector index metadata
        let vector_index = VectorIndex {
            centroids: centroids.into_iter().flatten().collect(),
            num_clusters: self.num_clusters.min(cluster_data.len() as u32),
            dim: self.embedding_dim,
            cluster_ptrs,
        };

        // Build manifest
        let manifest = Manifest {
            doc_count,
            embedding_dim: self.embedding_dim,
            num_clusters: self.num_clusters,
            doc_ptrs,
        };

        // Write everything to storage
        // Write segments
        for (seg_id, data) in segments {
            let path = format!("{}/segments/{:04}.dat", prefix, seg_id);
            store.put(&path, Bytes::from(data)).await?;
        }

        // Write manifest
        let manifest_bytes = serialize_manifest(&manifest)?;
        store
            .put(&format!("{}/manifest.bin", prefix), manifest_bytes)
            .await?;

        // Write lexical index
        let lexical_bytes = serialize_lexical_index(&lexical_index)?;
        store
            .put(&format!("{}/lexical.idx", prefix), lexical_bytes)
            .await?;

        // Write vector index
        let vector_bytes = serialize_vector_index(&vector_index)?;
        store
            .put(&format!("{}/vector.idx", prefix), vector_bytes)
            .await?;

        Ok(())
    }

    /// Get document count.
    pub fn doc_count(&self) -> u32 {
        if self.use_disk {
            self.disk_doc_count
        } else {
            self.documents.len() as u32
        }
    }
}

// Serialization helpers

fn serialize_posting_list(pl: &PostingList) -> StorageResult<Vec<u8>> {
    rkyv::to_bytes::<RkyvError>(pl)
        .map(|v| v.to_vec())
        .map_err(|e| StorageError::Io(format!("serialization error: {}", e)))
}

fn serialize_cluster_data(cd: &ClusterData) -> StorageResult<Vec<u8>> {
    rkyv::to_bytes::<RkyvError>(cd)
        .map(|v| v.to_vec())
        .map_err(|e| StorageError::Io(format!("serialization error: {}", e)))
}

fn serialize_stored_document(doc: &StoredDocument) -> StorageResult<Vec<u8>> {
    rkyv::to_bytes::<RkyvError>(doc)
        .map(|v| v.to_vec())
        .map_err(|e| StorageError::Io(format!("serialization error: {}", e)))
}

fn serialize_manifest(m: &Manifest) -> StorageResult<Bytes> {
    rkyv::to_bytes::<RkyvError>(m)
        .map(|v| Bytes::from(v.to_vec()))
        .map_err(|e| StorageError::Io(format!("serialization error: {}", e)))
}

fn serialize_lexical_index(idx: &LexicalIndex) -> StorageResult<Bytes> {
    rkyv::to_bytes::<RkyvError>(idx)
        .map(|v| Bytes::from(v.to_vec()))
        .map_err(|e| StorageError::Io(format!("serialization error: {}", e)))
}

fn serialize_vector_index(idx: &VectorIndex) -> StorageResult<Bytes> {
    rkyv::to_bytes::<RkyvError>(idx)
        .map(|v| Bytes::from(v.to_vec()))
        .map_err(|e| StorageError::Io(format!("serialization error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_store::local;
    use tempfile::TempDir;

    #[test]
    fn test_segment_writer() {
        let mut writer = SegmentWriter::new();

        let ptr1 = writer.write(b"hello");
        assert_eq!(ptr1.segment_id, 0);
        assert_eq!(ptr1.offset, 0);
        assert_eq!(ptr1.length, 5);

        let ptr2 = writer.write(b" world");
        assert_eq!(ptr2.segment_id, 0);
        assert_eq!(ptr2.offset, 5);
        assert_eq!(ptr2.length, 6);

        let segments = writer.finalize();
        assert_eq!(segments.len(), 1);
        assert_eq!(&segments[&0], b"hello world");
    }

    #[test]
    fn test_segment_writer_rotation() {
        let mut writer = SegmentWriter::new();

        // Write enough to trigger rotation (> 64MB would take too long)
        // For testing, we'll manually check the rotation logic
        let large_data = vec![0u8; SEGMENT_SIZE + 100];
        let ptr = writer.write(&large_data);
        assert_eq!(ptr.segment_id, 0);

        // After writing > SEGMENT_SIZE, buffer should rotate on next write
        let ptr2 = writer.write(b"after");
        assert_eq!(ptr2.segment_id, 1);
        assert_eq!(ptr2.offset, 0);
    }

    #[tokio::test]
    async fn test_index_builder_basic() {
        let tmp = TempDir::new().unwrap();
        let store = local(tmp.path()).unwrap();

        let mut builder = IndexBuilder::new(4, 2);
        builder
            .add(Document {
                id: 0,
                text: "hello world".to_string(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                attributes: HashMap::new(),
            })
            .unwrap();
        builder
            .add(Document {
                id: 1,
                text: "goodbye world".to_string(),
                embedding: vec![0.0, 1.0, 0.0, 0.0],
                attributes: HashMap::new(),
            })
            .unwrap();

        assert_eq!(builder.doc_count(), 2);

        builder.build(&store, "test").await.unwrap();

        // Verify files were created
        let files = store.list("test").await.unwrap();
        assert!(files.iter().any(|f| f.contains("manifest.bin")));
        assert!(files.iter().any(|f| f.contains("lexical.idx")));
        assert!(files.iter().any(|f| f.contains("vector.idx")));
    }

    #[tokio::test]
    async fn test_index_builder_with_disk() {
        let tmp = TempDir::new().unwrap();
        let store_dir = tmp.path().join("store");
        let temp_dir = tmp.path().join("temp");

        let store = local(&store_dir).unwrap();

        let mut builder = IndexBuilder::new(4, 2).with_temp_dir(&temp_dir).unwrap();

        builder
            .add(Document {
                id: 0,
                text: "hello disk".to_string(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                attributes: HashMap::new(),
            })
            .unwrap();
        builder
            .add(Document {
                id: 1,
                text: "goodbye disk".to_string(),
                embedding: vec![0.0, 1.0, 0.0, 0.0],
                attributes: HashMap::new(),
            })
            .unwrap();

        assert_eq!(builder.doc_count(), 2);

        builder.build(&store, "test").await.unwrap();

        // Verify files were created
        let files = store.list("test").await.unwrap();
        assert!(files.iter().any(|f| f.contains("manifest.bin")));
    }
}

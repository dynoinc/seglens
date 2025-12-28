//! Index builder: constructs the search index from documents.
//!
//! Supports streaming document ingestion with disk-based temporary storage
//! for handling datasets larger than memory. Uses batched writes to minimize
//! I/O overhead (~8MB per syscall instead of one file per document).

use crate::lexical::{
    LexicalIndex, LexicalIndexBuilder, PostingList, StreamingLexicalBuilder, VocabEntry,
};
use crate::storage::{BlobStore, StorageError, StorageResult};
use crate::types::{DocId, Document, SegmentPtr, StoredDocument};
use crate::vector::{kmeans, nearest_centroid, ClusterData, StreamingVectorBuilder, VectorIndex};
use bytes::Bytes;
use futures::{stream, StreamExt, TryStreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rayon::prelude::*;
use rkyv::rancor::Error as RkyvError;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::Duration;

/// Target segment size (64 MB).
const SEGMENT_SIZE: usize = 64 * 1024 * 1024;

/// Write buffer size for batched I/O (8 MB).
const WRITE_BUFFER_SIZE: usize = 8 * 1024 * 1024;
/// Number of concurrent segment writes.
const WRITE_CONCURRENCY: usize = 8;
/// Chunk size for parallel stats processing.
const STATS_CHUNK_SIZE: usize = 10_000;

// ============================================================================
// Document Batch I/O
// ============================================================================

/// Serializable document for batch storage.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct BatchDocument {
    id: DocId,
    text: String,
    embedding: Vec<f32>,
    attributes: Vec<(String, String)>,
}

impl From<&Document> for BatchDocument {
    fn from(doc: &Document) -> Self {
        Self {
            id: doc.id,
            text: doc.text.clone(),
            embedding: doc.embedding.clone(),
            attributes: doc
                .attributes
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }
}

impl From<BatchDocument> for Document {
    fn from(bd: BatchDocument) -> Self {
        Self {
            id: bd.id,
            text: bd.text,
            embedding: bd.embedding,
            attributes: bd.attributes.into_iter().collect(),
        }
    }
}

/// Writer for batched document storage.
///
/// Buffers documents in memory and flushes to disk when buffer exceeds threshold.
/// Batch file format:
/// ```text
/// [doc_0_bytes][doc_1_bytes]...[doc_n_bytes]
/// [offset_0: u64][offset_1: u64]...[offset_n: u64][count: u32]
/// ```
pub struct DocBatchWriter {
    /// Directory for batch files.
    dir: PathBuf,
    /// Current buffer of serialized documents.
    buffer: Vec<u8>,
    /// Offsets of each document within current buffer.
    offsets: Vec<u64>,
    /// Current batch file ID.
    batch_id: u32,
    /// Total documents written.
    doc_count: u32,
}

impl DocBatchWriter {
    /// Create a new batch writer.
    pub fn new(dir: PathBuf) -> std::io::Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            buffer: Vec::with_capacity(WRITE_BUFFER_SIZE),
            offsets: Vec::new(),
            batch_id: 0,
            doc_count: 0,
        })
    }

    /// Add a document to the batch.
    pub fn add(&mut self, doc: &Document) -> std::io::Result<()> {
        let batch_doc = BatchDocument::from(doc);
        let serialized = rkyv::to_bytes::<RkyvError>(&batch_doc)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        self.offsets.push(self.buffer.len() as u64);
        self.buffer.extend_from_slice(&serialized);
        self.doc_count += 1;

        if self.buffer.len() >= WRITE_BUFFER_SIZE {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush current buffer to disk.
    fn flush(&mut self) -> std::io::Result<()> {
        if self.offsets.is_empty() {
            return Ok(());
        }

        let path = self.dir.join(format!("batch_{:06}.bin", self.batch_id));
        let mut writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, File::create(&path)?);

        // Write document data
        writer.write_all(&self.buffer)?;

        // Write offset index
        for offset in &self.offsets {
            writer.write_all(&offset.to_le_bytes())?;
        }

        // Write document count
        let count = self.offsets.len() as u32;
        writer.write_all(&count.to_le_bytes())?;

        writer.flush()?;

        // Reset for next batch
        self.buffer.clear();
        self.offsets.clear();
        self.batch_id += 1;

        Ok(())
    }

    /// Finalize writing and return total document count.
    pub fn finalize(mut self) -> std::io::Result<u32> {
        self.flush()?;
        Ok(self.doc_count)
    }

    /// Get the directory path.
    pub fn dir(&self) -> &PathBuf {
        &self.dir
    }

    /// Get total documents written so far.
    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }
}

/// Reader for batched document storage.
///
/// Streams documents from batch files without loading all into memory.
pub struct DocBatchReader {
    /// Sorted list of batch file paths.
    batch_files: Vec<PathBuf>,
}

impl DocBatchReader {
    /// Create a reader for the given directory.
    pub fn new(dir: PathBuf) -> std::io::Result<Self> {
        let mut batch_files: Vec<PathBuf> = fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("batch_") && n.ends_with(".bin"))
            })
            .collect();

        batch_files.sort();

        Ok(Self { batch_files })
    }

    /// Iterate over all documents.
    pub fn iter(&self) -> DocBatchIter {
        DocBatchIter {
            batch_files: self.batch_files.clone(),
            current_batch_idx: 0,
            current_docs: Vec::new(),
            current_doc_idx: 0,
        }
    }
}

/// Iterator over documents in batch files.
pub struct DocBatchIter {
    batch_files: Vec<PathBuf>,
    current_batch_idx: usize,
    current_docs: Vec<Document>,
    current_doc_idx: usize,
}

impl DocBatchIter {
    /// Load documents from a batch file.
    fn load_batch(&mut self, path: &PathBuf) -> std::io::Result<()> {
        let mut file = BufReader::new(File::open(path)?);
        let file_len = file.seek(SeekFrom::End(0))?;

        // Read document count (last 4 bytes)
        file.seek(SeekFrom::End(-4))?;
        let mut count_bytes = [0u8; 4];
        file.read_exact(&mut count_bytes)?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        if count == 0 {
            self.current_docs.clear();
            return Ok(());
        }

        // Read offset index
        let index_size = count * 8 + 4; // offsets + count
        let data_end = file_len - index_size as u64;

        file.seek(SeekFrom::Start(data_end))?;
        let mut offsets = Vec::with_capacity(count);
        for _ in 0..count {
            let mut offset_bytes = [0u8; 8];
            file.read_exact(&mut offset_bytes)?;
            offsets.push(u64::from_le_bytes(offset_bytes));
        }

        // Read document data
        file.seek(SeekFrom::Start(0))?;
        let mut data = vec![0u8; data_end as usize];
        file.read_exact(&mut data)?;

        // Deserialize documents
        self.current_docs.clear();
        for i in 0..count {
            let start = offsets[i] as usize;
            let end = if i + 1 < count {
                offsets[i + 1] as usize
            } else {
                data_end as usize
            };

            let slice = &data[start..end];
            let batch_doc: BatchDocument = rkyv::from_bytes::<BatchDocument, RkyvError>(slice)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

            self.current_docs.push(Document::from(batch_doc));
        }

        self.current_doc_idx = 0;
        Ok(())
    }
}

impl Iterator for DocBatchIter {
    type Item = std::io::Result<Document>;

    fn next(&mut self) -> Option<Self::Item> {
        // Return next doc from current batch if available
        if self.current_doc_idx < self.current_docs.len() {
            let doc = self.current_docs[self.current_doc_idx].clone();
            self.current_doc_idx += 1;
            return Some(Ok(doc));
        }

        // Try to load next batch
        while self.current_batch_idx < self.batch_files.len() {
            let path = self.batch_files[self.current_batch_idx].clone();
            self.current_batch_idx += 1;

            if let Err(e) = self.load_batch(&path) {
                return Some(Err(e));
            }

            if !self.current_docs.is_empty() {
                let doc = self.current_docs[0].clone();
                self.current_doc_idx = 1;
                return Some(Ok(doc));
            }
        }

        None
    }
}

// ============================================================================
// Segment Writer
// ============================================================================

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

/// Statistics collected from first pass over documents.
///
/// Used for memory-efficient index building without loading all documents.
#[derive(Debug)]
struct BuildStats {
    /// Total number of documents.
    doc_count: u32,
    /// Sum of all document lengths (in tokens).
    total_doc_len: u64,
    /// Random sample of embeddings for k-means training.
    embedding_samples: Vec<Vec<f32>>,
    /// Document lengths for BM25 normalization.
    doc_lengths: Vec<u32>,
}

/// Index builder with streaming support.
///
/// For large datasets, documents are written to disk as they're added
/// using batched I/O (~8MB per syscall), then processed in multiple passes
/// during the build phase to maintain bounded memory usage.
pub struct IndexBuilder {
    /// Embedding dimension.
    embedding_dim: u32,
    /// Number of clusters for IVF.
    num_clusters: u32,
    /// Temporary directory for intermediate files.
    temp_dir: Option<PathBuf>,
    /// Documents stored in memory (for small datasets).
    documents: Vec<Document>,
    /// Batch writer for disk-based storage.
    batch_writer: Option<DocBatchWriter>,
    /// Document count (tracked separately for after writer is finalized).
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
            batch_writer: None,
            disk_doc_count: 0,
        }
    }

    /// Enable disk-based storage for large datasets.
    ///
    /// Documents will be written to disk in batches (~8MB each) as they're added,
    /// allowing indexing of datasets larger than available memory.
    pub fn with_temp_dir(mut self, path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let docs_dir = path.join("docs");
        self.temp_dir = Some(path);
        self.batch_writer = Some(DocBatchWriter::new(docs_dir)?);
        Ok(self)
    }

    /// Add a document to the index.
    pub fn add(&mut self, doc: Document) -> std::io::Result<()> {
        if let Some(ref mut writer) = self.batch_writer {
            writer.add(&doc)?;
            self.disk_doc_count += 1;
        } else {
            self.documents.push(doc);
        }
        Ok(())
    }

    /// Iterate over all documents (from memory or disk).
    fn iter_documents(&self) -> Box<dyn Iterator<Item = std::io::Result<Document>> + '_> {
        if let Some(ref temp_dir) = self.temp_dir {
            // Read from batch files
            let docs_dir = temp_dir.join("docs");
            match DocBatchReader::new(docs_dir) {
                Ok(reader) => Box::new(reader.iter()),
                Err(e) => Box::new(std::iter::once(Err(e))),
            }
        } else {
            // Read from memory
            Box::new(self.documents.iter().cloned().map(Ok))
        }
    }

    /// Collect statistics from first pass over documents.
    ///
    /// This pass collects:
    /// - Document count
    /// - Total document length (for avg_doc_len)
    /// - Random sample of embeddings for k-means training
    ///
    /// Memory usage: O(sample_size * embedding_dim), not O(total_docs).
    fn collect_stats(
        &self,
        sample_size: usize,
        progress: Option<&ProgressBar>,
    ) -> std::io::Result<BuildStats> {
        let mut stats = BuildStats {
            doc_count: 0,
            total_doc_len: 0,
            embedding_samples: Vec::with_capacity(sample_size),
            doc_lengths: Vec::new(),
        };

        let mut rng = rand::thread_rng();

        let mut chunk: Vec<Document> = Vec::with_capacity(STATS_CHUNK_SIZE);

        for doc_result in self.iter_documents() {
            chunk.push(doc_result?);

            if chunk.len() >= STATS_CHUNK_SIZE {
                process_stats_chunk(&mut stats, &mut rng, sample_size, &chunk, progress);
                chunk.clear();
            }
        }

        if !chunk.is_empty() {
            process_stats_chunk(&mut stats, &mut rng, sample_size, &chunk, progress);
        }

        Ok(stats)
    }

    /// Build the index and write to storage.
    ///
    /// Uses multi-pass streaming for disk-based storage, or single-pass in-memory
    /// for small datasets.
    pub async fn build(&mut self, store: &dyn BlobStore, prefix: &str) -> StorageResult<()> {
        // Finalize batch writer to flush remaining documents
        if let Some(writer) = self.batch_writer.take() {
            writer
                .finalize()
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }

        // Choose streaming or in-memory build based on storage mode
        if self.temp_dir.is_some() {
            self.build_streaming(store, prefix).await
        } else {
            self.build_in_memory(store, prefix).await
        }
    }

    /// Build index using multi-pass streaming for large datasets.
    ///
    /// Memory usage: ~1-2GB regardless of dataset size.
    async fn build_streaming(&self, store: &dyn BlobStore, prefix: &str) -> StorageResult<()> {
        let temp_dir = self.temp_dir.as_ref().unwrap();

        // === Pass 1: Collect statistics ===
        const SAMPLE_SIZE: usize = 10_000;
        let total_docs_hint = self.doc_count().max(1) as u64;
        let pb_stats = progress_bar(total_docs_hint, "Pass 1/4: collecting stats");
        let stats = self
            .collect_stats(SAMPLE_SIZE, Some(&pb_stats))
            .map_err(|e| StorageError::Io(e.to_string()))?;
        pb_stats.finish_with_message(format!("Pass 1/4 done: {} docs", stats.doc_count));

        if stats.doc_count == 0 {
            return Err(StorageError::Io("no documents to index".to_string()));
        }

        // Train k-means on sampled embeddings
        let centroids = if stats.embedding_samples.is_empty() {
            Vec::new()
        } else {
            kmeans(&stats.embedding_samples, self.num_clusters as usize, 10)
        };

        // === Pass 2: Build lexical index (streaming) ===
        let lexical_dir = temp_dir.join("lexical");
        let mut lexical_builder = StreamingLexicalBuilder::new(lexical_dir)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        let pb_lex = progress_bar(stats.doc_count as u64, "Pass 2/4: lexical index");
        for doc_result in self.iter_documents() {
            let doc = doc_result.map_err(|e| StorageError::Io(e.to_string()))?;
            lexical_builder
                .add_document(doc.id, &doc.text)
                .map_err(|e| StorageError::Io(e.to_string()))?;
            pb_lex.inc(1);
        }
        pb_lex.finish_with_message("Pass 2/4 done");

        let (posting_lists, doc_lengths) =
            tokio::task::spawn_blocking(move || lexical_builder.build())
                .await
                .map_err(|e| StorageError::Io(format!("task join error: {}", e)))?
                .map_err(|e| StorageError::Io(e.to_string()))?;

        // === Pass 3: Build vector index (streaming) ===
        let vector_dir = temp_dir.join("vector");
        let mut vector_builder = StreamingVectorBuilder::new(centroids.clone(), vector_dir)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        let pb_vec = progress_bar(stats.doc_count as u64, "Pass 3/4: vector index");
        for doc_result in self.iter_documents() {
            let doc = doc_result.map_err(|e| StorageError::Io(e.to_string()))?;
            vector_builder
                .add(doc.id, &doc.embedding)
                .map_err(|e| StorageError::Io(e.to_string()))?;
            pb_vec.inc(1);
        }
        pb_vec.finish_with_message("Pass 3/4 done");

        let (centroids, cluster_data) = vector_builder
            .build()
            .map_err(|e| StorageError::Io(e.to_string()))?;

        // === Pass 4: Write stored documents and build segments ===
        let mut segment_writer = SegmentWriter::new();

        // Write posting lists
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

        // Write stored documents (streaming with buffer)
        let mut doc_ptrs: Vec<SegmentPtr> = Vec::with_capacity(stats.doc_count as usize);
        let mut doc_buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);

        let pb_docs = progress_bar(stats.doc_count as u64, "Pass 4/4: writing docs");
        for doc_result in self.iter_documents() {
            let doc = doc_result.map_err(|e| StorageError::Io(e.to_string()))?;
            let stored = StoredDocument::from_document(&doc);
            let bytes = serialize_stored_document(&stored)?;

            let ptr = SegmentPtr {
                segment_id: segment_writer.segment_id,
                offset: segment_writer.offset + doc_buffer.len() as u64,
                length: bytes.len() as u32,
            };
            doc_ptrs.push(ptr);
            doc_buffer.extend_from_slice(&bytes);
            pb_docs.inc(1);

            // Flush buffer when full
            if doc_buffer.len() >= WRITE_BUFFER_SIZE {
                segment_writer.write(&doc_buffer);
                doc_buffer.clear();
            }
        }
        pb_docs.finish_with_message("Pass 4/4 done");

        // Flush remaining buffer
        if !doc_buffer.is_empty() {
            segment_writer.write(&doc_buffer);
        }

        // Finalize and write to storage
        self.write_index_to_storage(
            store,
            prefix,
            segment_writer,
            vocab_entries,
            cluster_ptrs,
            doc_ptrs,
            centroids,
            stats.doc_count,
            &doc_lengths,
        )
        .await
    }

    /// Build index in memory for small datasets.
    async fn build_in_memory(&self, store: &dyn BlobStore, prefix: &str) -> StorageResult<()> {
        // Collect all documents into memory
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

        // Finalize and write to storage
        self.write_index_to_storage(
            store,
            prefix,
            segment_writer,
            vocab_entries,
            cluster_ptrs,
            doc_ptrs,
            centroids,
            doc_count,
            lexical_builder.doc_lengths(),
        )
        .await
    }

    /// Write the built index to storage.
    #[allow(clippy::too_many_arguments)]
    async fn write_index_to_storage(
        &self,
        store: &dyn BlobStore,
        prefix: &str,
        segment_writer: SegmentWriter,
        vocab_entries: Vec<(String, VocabEntry)>,
        cluster_ptrs: Vec<SegmentPtr>,
        doc_ptrs: Vec<SegmentPtr>,
        centroids: Vec<Vec<f32>>,
        doc_count: u32,
        doc_lengths: &[u32],
    ) -> StorageResult<()> {
        // Finalize segments
        let segments = segment_writer.finalize();
        let write_pb = progress_bar(
            (segments.len() as u64) + 3,
            "Writing index to disk (segments + metadata)",
        );

        // Calculate avg_doc_len
        let avg_doc_len = if doc_lengths.is_empty() {
            0.0
        } else {
            let total: u32 = doc_lengths.iter().sum();
            total as f32 / doc_lengths.len() as f32
        };

        // Build lexical index metadata
        let lexical_index = LexicalIndex {
            vocab: vocab_entries,
            doc_count,
            avg_doc_len,
            doc_lengths: doc_lengths.to_vec(),
        };

        // Build vector index metadata
        let vector_index = VectorIndex {
            centroids: centroids.into_iter().flatten().collect(),
            num_clusters: self.num_clusters.min(cluster_ptrs.len() as u32),
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

        // Write segments concurrently
        stream::iter(segments.into_iter())
            .map(|(seg_id, data)| {
                let path = format!("{}/segments/{:04}.dat", prefix, seg_id);
                let store = store;
                let pb = write_pb.clone();
                async move {
                    store.put(&path, Bytes::from(data)).await?;
                    pb.inc(1);
                    StorageResult::Ok(())
                }
            })
            .buffer_unordered(WRITE_CONCURRENCY)
            .try_collect::<()>()
            .await?;

        // Write manifest
        let manifest_bytes = serialize_manifest(&manifest)?;
        store
            .put(&format!("{}/manifest.bin", prefix), manifest_bytes)
            .await?;
        write_pb.inc(1);

        // Write lexical index
        let lexical_bytes = serialize_lexical_index(&lexical_index)?;
        store
            .put(&format!("{}/lexical.idx", prefix), lexical_bytes)
            .await?;
        write_pb.inc(1);

        // Write vector index
        let vector_bytes = serialize_vector_index(&vector_index)?;
        store
            .put(&format!("{}/vector.idx", prefix), vector_bytes)
            .await?;
        write_pb.inc(1);
        write_pb.finish_with_message("Writing index to disk: done");

        Ok(())
    }

    /// Get document count.
    pub fn doc_count(&self) -> u32 {
        if self.temp_dir.is_some() {
            self.disk_doc_count
        } else {
            self.documents.len() as u32
        }
    }

    /// Override the number of clusters (useful when doc count is known later).
    pub fn set_num_clusters(&mut self, num_clusters: u32) {
        self.num_clusters = num_clusters;
    }
}

fn process_stats_chunk(
    stats: &mut BuildStats,
    rng: &mut rand::rngs::ThreadRng,
    sample_size: usize,
    chunk: &[Document],
    progress: Option<&ProgressBar>,
) {
    // Compute document lengths in parallel to utilize multiple CPUs.
    let doc_lens: Vec<u32> = chunk
        .par_iter()
        .map(|doc| crate::lexical::tokenize(&doc.text).len() as u32)
        .collect();

    let chunk_total: u64 = doc_lens.iter().map(|&l| l as u64).sum();

    // Merge results sequentially to maintain sampling correctness and doc_lengths indexing.
    for (doc, &doc_len) in chunk.iter().zip(doc_lens.iter()) {
        let doc_id = doc.id as usize;
        if stats.doc_lengths.len() <= doc_id {
            stats.doc_lengths.resize(doc_id + 1, 0);
        }
        stats.doc_lengths[doc_id] = doc_len;
    }

    stats.doc_count += chunk.len() as u32;
    stats.total_doc_len += chunk_total;

    for doc in chunk {
        if stats.embedding_samples.len() < sample_size {
            stats.embedding_samples.push(doc.embedding.clone());
        } else if sample_size > 0 {
            let idx = rng.gen_range(0..stats.doc_count as usize);
            if idx < sample_size {
                stats.embedding_samples[idx] = doc.embedding.clone();
            }
        }
    }

    if let Some(pb) = progress {
        pb.set_position(stats.doc_count as u64);
        pb.set_message(format!("Pass 1/4: docs {}", stats.doc_count));
    }
}

fn progress_bar(len: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb
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

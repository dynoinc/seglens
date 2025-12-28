//! Lexical search components: tokenizer, BM25 scoring, and inverted index.

use crate::types::{DocId, SegmentPtr, TermId};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

/// BM25 parameter k1 (term frequency saturation).
const BM25_K1: f32 = 1.2;

/// BM25 parameter b (length normalization).
const BM25_B: f32 = 0.75;

/// Tokenize text into terms.
///
/// Applies: lowercase, split on non-alphanumeric, filter tokens with length > 1.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 1)
        .map(|s| s.to_string())
        .collect()
}

/// Calculate BM25 score for a term in a document.
///
/// # Arguments
/// * `tf` - Term frequency in the document
/// * `doc_len` - Number of terms in the document
/// * `avg_doc_len` - Average document length across corpus
/// * `doc_count` - Total number of documents
/// * `doc_freq` - Number of documents containing the term
pub fn bm25_score(tf: f32, doc_len: u32, avg_doc_len: f32, doc_count: u32, doc_freq: u32) -> f32 {
    if doc_freq == 0 || doc_count == 0 {
        return 0.0;
    }

    // IDF component: log((N - n + 0.5) / (n + 0.5) + 1)
    let n = doc_freq as f32;
    let big_n = doc_count as f32;
    let idf = ((big_n - n + 0.5) / (n + 0.5) + 1.0).ln();

    // TF component with length normalization
    let dl = doc_len as f32;
    let tf_component =
        (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avg_doc_len));

    idf * tf_component
}

/// A posting entry: document ID and term frequency.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Posting {
    /// Document ID.
    pub doc_id: DocId,
    /// Term frequency in this document.
    pub tf: f32,
}

/// A posting list for a term.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct PostingList {
    /// Postings sorted by doc_id.
    pub postings: Vec<Posting>,
}

impl PostingList {
    /// Create a new empty posting list.
    pub fn new() -> Self {
        Self {
            postings: Vec::new(),
        }
    }

    /// Add a posting for a document.
    pub fn add(&mut self, doc_id: DocId, tf: f32) {
        self.postings.push(Posting { doc_id, tf });
    }

    /// Sort postings by doc_id.
    pub fn sort(&mut self) {
        self.postings.sort_by_key(|p| p.doc_id);
    }
}

impl Default for PostingList {
    fn default() -> Self {
        Self::new()
    }
}

/// Vocabulary entry with term statistics.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct VocabEntry {
    /// Number of documents containing this term.
    pub doc_freq: u32,
    /// Pointer to the posting list in segment storage.
    pub posting_ptr: SegmentPtr,
}

/// Lexical index metadata (loaded at startup).
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct LexicalIndex {
    /// Vocabulary: term -> (term_id, vocab_entry).
    pub vocab: Vec<(String, VocabEntry)>,
    /// Total document count.
    pub doc_count: u32,
    /// Average document length (in terms).
    pub avg_doc_len: f32,
    /// Document lengths for normalization.
    pub doc_lengths: Vec<u32>,
}

impl LexicalIndex {
    /// Create a new empty lexical index.
    pub fn new() -> Self {
        Self {
            vocab: Vec::new(),
            doc_count: 0,
            avg_doc_len: 0.0,
            doc_lengths: Vec::new(),
        }
    }

    /// Get vocabulary as a HashMap for fast lookups.
    pub fn vocab_map(&self) -> HashMap<String, (TermId, VocabEntry)> {
        self.vocab
            .iter()
            .enumerate()
            .map(|(i, (term, entry))| (term.clone(), (i as TermId, entry.clone())))
            .collect()
    }
}

impl Default for LexicalIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing the lexical index.
pub struct LexicalIndexBuilder {
    /// Term -> (doc_id -> term_frequency).
    term_docs: HashMap<String, HashMap<DocId, f32>>,
    /// Document lengths.
    doc_lengths: Vec<u32>,
    /// Next document to process.
    next_doc: DocId,
}

impl LexicalIndexBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            term_docs: HashMap::new(),
            doc_lengths: Vec::new(),
            next_doc: 0,
        }
    }

    /// Add a document's text to the index.
    pub fn add_document(&mut self, doc_id: DocId, text: &str) {
        let tokens = tokenize(text);
        let doc_len = tokens.len() as u32;

        // Ensure doc_lengths has enough capacity
        while self.doc_lengths.len() <= doc_id as usize {
            self.doc_lengths.push(0);
        }
        self.doc_lengths[doc_id as usize] = doc_len;

        // Count term frequencies
        let mut term_freqs: HashMap<String, f32> = HashMap::new();
        for token in tokens {
            *term_freqs.entry(token).or_insert(0.0) += 1.0;
        }

        // Add to term_docs
        for (term, tf) in term_freqs {
            self.term_docs.entry(term).or_default().insert(doc_id, tf);
        }

        self.next_doc = self.next_doc.max(doc_id + 1);
    }

    /// Build posting lists for all terms.
    ///
    /// Returns: (term, doc_freq, posting_list) for each term.
    pub fn build_posting_lists(&self) -> Vec<(String, u32, PostingList)> {
        let mut result = Vec::new();

        for (term, docs) in &self.term_docs {
            let mut posting_list = PostingList::new();
            for (&doc_id, &tf) in docs {
                posting_list.add(doc_id, tf);
            }
            posting_list.sort();

            result.push((term.clone(), docs.len() as u32, posting_list));
        }

        // Sort by term for consistent ordering
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Get document count.
    pub fn doc_count(&self) -> u32 {
        self.next_doc
    }

    /// Get average document length.
    pub fn avg_doc_len(&self) -> f32 {
        if self.doc_lengths.is_empty() {
            return 0.0;
        }
        let total: u32 = self.doc_lengths.iter().sum();
        total as f32 / self.doc_lengths.len() as f32
    }

    /// Get document lengths.
    pub fn doc_lengths(&self) -> &[u32] {
        &self.doc_lengths
    }
}

impl Default for LexicalIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Streaming Lexical Builder (for large datasets)
// ============================================================================

use crossbeam_channel::{bounded, Receiver, TryRecvError};
use indicatif::{ProgressBar, ProgressStyle};
use rkyv::rancor::Error as RkyvError;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

/// Write buffer size for batched I/O (8 MB).
const WRITE_BUFFER_SIZE: usize = 8 * 1024 * 1024;

/// Default batch size (number of documents per batch).
const DEFAULT_BATCH_SIZE: usize = 200_000;

/// A batch entry for external merge sort.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct BatchEntry {
    term: String,
    doc_id: DocId,
    tf: f32,
}

/// Streaming lexical index builder for large datasets.
///
/// Processes documents in batches, flushing sorted posting entries to disk
/// when memory threshold is reached. Uses external merge sort to combine
/// batches into the final posting lists.
pub struct StreamingLexicalBuilder {
    /// Temporary directory for batch files.
    temp_dir: PathBuf,
    /// Current batch of posting entries.
    current_batch: Vec<BatchEntry>,
    /// Number of batch files written.
    batch_count: u32,
    /// Batch size threshold (number of documents).
    batch_size: usize,
    /// Documents processed in current batch.
    docs_in_batch: usize,
    /// Total document count.
    doc_count: u32,
    /// Document lengths for BM25 normalization.
    doc_lengths: Vec<u32>,
}

impl StreamingLexicalBuilder {
    /// Create a new streaming builder.
    pub fn new(temp_dir: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&temp_dir)?;
        Ok(Self {
            temp_dir,
            current_batch: Vec::new(),
            batch_count: 0,
            batch_size: DEFAULT_BATCH_SIZE,
            docs_in_batch: 0,
            doc_count: 0,
            doc_lengths: Vec::new(),
        })
    }

    /// Set the batch size (number of documents per batch).
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Add a document's text to the index.
    pub fn add_document(&mut self, doc_id: DocId, text: &str) -> std::io::Result<()> {
        let tokens = tokenize(text);
        let doc_len = tokens.len() as u32;

        // Track document length
        while self.doc_lengths.len() <= doc_id as usize {
            self.doc_lengths.push(0);
        }
        self.doc_lengths[doc_id as usize] = doc_len;
        self.doc_count = self.doc_count.max(doc_id + 1);

        // Count term frequencies
        let mut term_freqs: HashMap<String, f32> = HashMap::new();
        for token in tokens {
            *term_freqs.entry(token).or_insert(0.0) += 1.0;
        }

        // Add entries to current batch
        for (term, tf) in term_freqs {
            self.current_batch.push(BatchEntry { term, doc_id, tf });
        }

        self.docs_in_batch += 1;

        // Flush if batch is full
        if self.docs_in_batch >= self.batch_size {
            self.flush_batch()?;
        }

        Ok(())
    }

    /// Flush current batch to disk.
    fn flush_batch(&mut self) -> std::io::Result<()> {
        if self.current_batch.is_empty() {
            return Ok(());
        }

        // Sort by term, then by doc_id
        self.current_batch
            .sort_by(|a, b| a.term.cmp(&b.term).then(a.doc_id.cmp(&b.doc_id)));

        // Write to batch file
        let path = self
            .temp_dir
            .join(format!("batch_{:06}.bin", self.batch_count));
        let mut writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, File::create(&path)?);

        // Write entry count
        let count = self.current_batch.len() as u32;
        writer.write_all(&count.to_le_bytes())?;

        // Serialize entries individually with a length prefix so we can stream them back.
        for entry in &self.current_batch {
            let serialized = rkyv::to_bytes::<RkyvError>(entry)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            let len = serialized.len() as u32;
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&serialized)?;
        }

        writer.flush()?;

        // Reset for next batch
        self.current_batch.clear();
        self.batch_count += 1;
        self.docs_in_batch = 0;

        Ok(())
    }

    /// Build the final posting lists by merging all batches.
    #[allow(clippy::type_complexity)]
    pub fn build(mut self) -> std::io::Result<(Vec<(String, u32, PostingList)>, Vec<u32>)> {
        // Flush any remaining entries
        self.flush_batch()?;

        if self.batch_count == 0 {
            return Ok((Vec::new(), self.doc_lengths));
        }

        // Pre-read entry counts for accurate progress.
        let mut entry_counts: Vec<u64> = Vec::new();
        let mut batch_paths: Vec<PathBuf> = Vec::new();
        for i in 0..self.batch_count {
            let path = self.temp_dir.join(format!("batch_{:06}.bin", i));
            entry_counts.push(BatchFileIter::entry_count(&path)? as u64);
            batch_paths.push(path);
        }

        // Stream batches with a k-way merge backed by a small thread pool prefetcher.
        let mut receivers: Vec<Receiver<BatchEntry>> = Vec::new();
        let mut senders = Vec::new();
        for _ in 0..self.batch_count {
            let (tx, rx) = bounded(BatchFileIter::CHANNEL_CAP);
            senders.push(tx);
            receivers.push(rx);
        }

        let (tasks_tx, tasks_rx) = bounded::<usize>(self.batch_count as usize);
        for idx in 0..self.batch_count as usize {
            tasks_tx.send(idx).ok();
        }
        drop(tasks_tx); // Signal no more tasks - prevents worker deadlock

        let mut handles = Vec::new();
        // Worker pool size: min(num_batches, 8)
        let workers = self.batch_count.min(8);
        for _ in 0..workers {
            let tasks_rx = tasks_rx.clone();
            let paths = batch_paths.clone();
            let senders = senders.clone();

            let handle = thread::spawn(move || -> std::io::Result<()> {
                while let Ok(batch_idx) = tasks_rx.recv() {
                    let path = &paths[batch_idx];
                    let mut iter = BatchFileIter::open(path)?;
                    while let Some(entry) = iter.next_entry()? {
                        if senders[batch_idx].send(entry).is_err() {
                            break;
                        }
                    }
                }
                Ok(())
            });

            handles.push(handle);
        }

        let total_entries: u64 = entry_counts.iter().copied().sum();
        let merge_pb = merge_progress_bar(total_entries.max(1));

        // Min-heap keyed by (term, doc_id) pulling the head entry from each batch.
        let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
        for (idx, rx) in receivers.iter().enumerate() {
            if let Ok(entry) = rx.recv() {
                heap.push(Reverse(HeapItem {
                    entry,
                    batch_idx: idx,
                }));
            }
        }

        let mut result: Vec<(String, u32, PostingList)> = Vec::new();
        let mut current_term: Option<String> = None;
        let mut current_postings = PostingList::new();

        while let Some(Reverse(item)) = heap.pop() {
            merge_pb.inc(1);

            if current_term.as_ref() != Some(&item.entry.term) {
                if let Some(term) = current_term.take() {
                    let doc_freq = current_postings.postings.len() as u32;
                    result.push((term, doc_freq, current_postings));
                    current_postings = PostingList::new();
                }
                current_term = Some(item.entry.term.clone());
            }

            current_postings.add(item.entry.doc_id, item.entry.tf);

            match receivers[item.batch_idx].try_recv() {
                Ok(next) => heap.push(Reverse(HeapItem {
                    entry: next,
                    batch_idx: item.batch_idx,
                })),
                Err(TryRecvError::Empty) => {
                    // Allow a short wait to give the producer time to fill the channel.
                    if let Ok(next) =
                        receivers[item.batch_idx].recv_timeout(Duration::from_millis(5))
                    {
                        heap.push(Reverse(HeapItem {
                            entry: next,
                            batch_idx: item.batch_idx,
                        }));
                    }
                }
                Err(TryRecvError::Disconnected) => {}
            }
        }

        if let Some(term) = current_term {
            let doc_freq = current_postings.postings.len() as u32;
            result.push((term, doc_freq, current_postings));
        }

        merge_pb.finish_with_message("Merged lexical batches");

        // Ensure producer threads completed.
        for handle in handles {
            if let Err(e) = handle.join() {
                eprintln!("Batch reader thread panicked: {:?}", e);
            }
        }

        // Clean up batch files
        for i in 0..self.batch_count {
            let path = self.temp_dir.join(format!("batch_{:06}.bin", i));
            let _ = std::fs::remove_file(path);
        }

        Ok((result, self.doc_lengths))
    }

    /// Get document count.
    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    /// Get average document length.
    pub fn avg_doc_len(&self) -> f32 {
        if self.doc_lengths.is_empty() {
            return 0.0;
        }
        let total: u32 = self.doc_lengths.iter().sum();
        total as f32 / self.doc_lengths.len() as f32
    }
}

/// Iterator over a serialized batch file (length-prefixed entries).
struct BatchFileIter {
    reader: BufReader<File>,
    remaining: u32,
    buffer: Vec<BatchEntry>,
    buffer_idx: usize,
}

impl BatchFileIter {
    /// Channel capacity for batch prefetchers.
    const CHANNEL_CAP: usize = 50_000;
    /// Number of entries to prefetch into memory at once.
    const PREFETCH: usize = 32_768;

    fn open(path: &Path) -> std::io::Result<Self> {
        let mut reader = BufReader::with_capacity(WRITE_BUFFER_SIZE, File::open(path)?);
        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        let remaining = u32::from_le_bytes(count_bytes);

        Ok(Self {
            reader,
            remaining,
            buffer: Vec::new(),
            buffer_idx: 0,
        })
    }

    fn entry_count(path: &Path) -> std::io::Result<u32> {
        let mut reader = BufReader::with_capacity(16, File::open(path)?);
        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        Ok(u32::from_le_bytes(count_bytes))
    }

    fn next_entry(&mut self) -> std::io::Result<Option<BatchEntry>> {
        if self.buffer_idx >= self.buffer.len() {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.refill_buffer()?;
        }

        if self.buffer_idx >= self.buffer.len() {
            return Ok(None);
        }

        let entry = self.buffer[self.buffer_idx].clone();
        self.buffer_idx += 1;
        Ok(Some(entry))
    }

    /// Prefetch a chunk of entries into memory to smooth out disk waits.
    fn refill_buffer(&mut self) -> std::io::Result<()> {
        self.buffer.clear();
        self.buffer_idx = 0;

        let to_read = self.remaining.min(Self::PREFETCH as u32).max(1) as usize;

        for _ in 0..to_read {
            if self.remaining == 0 {
                break;
            }

            let mut len_bytes = [0u8; 4];
            self.reader.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as usize;

            let mut buf = vec![0u8; len];
            self.reader.read_exact(&mut buf)?;

            let entry: BatchEntry = rkyv::from_bytes::<BatchEntry, RkyvError>(&buf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

            self.buffer.push(entry);
            self.remaining -= 1;
        }

        Ok(())
    }
}

fn merge_progress_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap(),
    );
    pb.set_message("Merging lexical batches".to_string());
    pb
}

/// Heap item for k-way merge (min-heap by term then doc_id).
#[derive(Clone)]
struct HeapItem {
    entry: BatchEntry,
    batch_idx: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.entry.term == other.entry.term && self.entry.doc_id == other.entry.doc_id
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.entry
            .term
            .cmp(&other.entry.term)
            .then(self.entry.doc_id.cmp(&other.entry.doc_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_filters_short() {
        let tokens = tokenize("I am a cat");
        // "i", "a" are filtered out (len <= 1)
        assert_eq!(tokens, vec!["am", "cat"]);
    }

    #[test]
    fn test_tokenize_numbers() {
        let tokens = tokenize("test123 456test");
        assert_eq!(tokens, vec!["test123", "456test"]);
    }

    #[test]
    fn test_bm25_score_basic() {
        let score = bm25_score(1.0, 10, 10.0, 100, 10);
        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25_score_rare_term_higher() {
        // Rare term (doc_freq=1) should score higher than common term (doc_freq=50)
        let rare_score = bm25_score(1.0, 10, 10.0, 100, 1);
        let common_score = bm25_score(1.0, 10, 10.0, 100, 50);
        assert!(rare_score > common_score);
    }

    #[test]
    fn test_bm25_score_higher_tf_higher() {
        // Higher term frequency should score higher
        let low_tf = bm25_score(1.0, 10, 10.0, 100, 10);
        let high_tf = bm25_score(5.0, 10, 10.0, 100, 10);
        assert!(high_tf > low_tf);
    }

    #[test]
    fn test_bm25_score_zero_doc_freq() {
        let score = bm25_score(1.0, 10, 10.0, 100, 0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_posting_list() {
        let mut pl = PostingList::new();
        pl.add(2, 1.0);
        pl.add(0, 2.0);
        pl.add(1, 1.5);
        pl.sort();

        assert_eq!(pl.postings.len(), 3);
        assert_eq!(pl.postings[0].doc_id, 0);
        assert_eq!(pl.postings[1].doc_id, 1);
        assert_eq!(pl.postings[2].doc_id, 2);
    }

    #[test]
    fn test_lexical_index_builder() {
        let mut builder = LexicalIndexBuilder::new();
        builder.add_document(0, "hello world");
        builder.add_document(1, "hello rust");
        builder.add_document(2, "goodbye world");

        assert_eq!(builder.doc_count(), 3);
        assert!(builder.avg_doc_len() > 0.0);

        let posting_lists = builder.build_posting_lists();

        // Find "hello" posting list
        let hello_pl = posting_lists.iter().find(|(t, _, _)| t == "hello").unwrap();
        assert_eq!(hello_pl.1, 2); // doc_freq = 2 (docs 0 and 1)

        // Find "world" posting list
        let world_pl = posting_lists.iter().find(|(t, _, _)| t == "world").unwrap();
        assert_eq!(world_pl.1, 2); // doc_freq = 2 (docs 0 and 2)

        // Find "rust" posting list
        let rust_pl = posting_lists.iter().find(|(t, _, _)| t == "rust").unwrap();
        assert_eq!(rust_pl.1, 1); // doc_freq = 1 (doc 1 only)
    }

    #[test]
    fn test_vocab_map() {
        let index = LexicalIndex {
            vocab: vec![
                (
                    "hello".to_string(),
                    VocabEntry {
                        doc_freq: 2,
                        posting_ptr: SegmentPtr {
                            segment_id: 0,
                            offset: 0,
                            length: 10,
                        },
                    },
                ),
                (
                    "world".to_string(),
                    VocabEntry {
                        doc_freq: 1,
                        posting_ptr: SegmentPtr {
                            segment_id: 0,
                            offset: 10,
                            length: 5,
                        },
                    },
                ),
            ],
            doc_count: 2,
            avg_doc_len: 5.0,
            doc_lengths: vec![5, 5],
        };

        let vocab_map = index.vocab_map();
        assert_eq!(vocab_map.len(), 2);

        let (term_id, entry) = vocab_map.get("hello").unwrap();
        assert_eq!(*term_id, 0);
        assert_eq!(entry.doc_freq, 2);
    }

    #[test]
    fn test_streaming_lexical_builder() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut builder = StreamingLexicalBuilder::new(tmp.path().to_path_buf())
            .unwrap()
            .with_batch_size(2); // Small batch size to test flushing

        builder.add_document(0, "hello world").unwrap();
        builder.add_document(1, "hello rust").unwrap();
        builder.add_document(2, "goodbye world").unwrap();

        assert_eq!(builder.doc_count(), 3);

        let (posting_lists, doc_lengths) = builder.build().unwrap();

        // Find "hello" posting list
        let hello_pl = posting_lists.iter().find(|(t, _, _)| t == "hello").unwrap();
        assert_eq!(hello_pl.1, 2); // doc_freq = 2

        // Find "world" posting list
        let world_pl = posting_lists.iter().find(|(t, _, _)| t == "world").unwrap();
        assert_eq!(world_pl.1, 2); // doc_freq = 2

        // Check doc_lengths
        assert_eq!(doc_lengths.len(), 3);
    }
}

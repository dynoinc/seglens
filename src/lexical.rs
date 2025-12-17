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
}

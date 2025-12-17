//! Core types for the seglens search index.

use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

/// Document identifier type.
pub type DocId = u32;

/// Term identifier type for the inverted index.
pub type TermId = u32;

/// A document to be indexed.
#[derive(Debug, Clone, Default)]
pub struct Document {
    /// Unique document identifier.
    pub id: DocId,
    /// Text content for lexical search.
    pub text: String,
    /// Vector embedding for semantic search.
    pub embedding: Vec<f32>,
    /// Additional metadata attributes.
    pub attributes: HashMap<String, String>,
}

/// Pointer to data within a segment file.
#[derive(Debug, Clone, Copy, Archive, Serialize, Deserialize)]
pub struct SegmentPtr {
    /// Which segment file contains the data.
    pub segment_id: u32,
    /// Byte offset within the segment.
    pub offset: u64,
    /// Length in bytes.
    pub length: u32,
}

/// A search result returned from queries.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The document ID.
    pub doc_id: DocId,
    /// Relevance score (higher is better).
    pub score: f32,
    /// The document text.
    pub text: String,
    /// Document attributes.
    pub attributes: HashMap<String, String>,
}

/// Stored document data (serialized in segments).
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct StoredDocument {
    /// Document ID.
    pub id: DocId,
    /// Text content.
    pub text: String,
    /// Attributes as key-value pairs.
    pub attributes: Vec<(String, String)>,
}

impl StoredDocument {
    /// Create from a Document.
    pub fn from_document(doc: &Document) -> Self {
        Self {
            id: doc.id,
            text: doc.text.clone(),
            attributes: doc
                .attributes
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }

    /// Convert attributes to HashMap.
    pub fn attributes_map(&self) -> HashMap<String, String> {
        self.attributes.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_default() {
        let doc = Document::default();
        assert_eq!(doc.id, 0);
        assert!(doc.text.is_empty());
        assert!(doc.embedding.is_empty());
        assert!(doc.attributes.is_empty());
    }

    #[test]
    fn test_segment_ptr() {
        let ptr = SegmentPtr {
            segment_id: 1,
            offset: 1024,
            length: 256,
        };
        assert_eq!(ptr.segment_id, 1);
        assert_eq!(ptr.offset, 1024);
        assert_eq!(ptr.length, 256);
    }

    #[test]
    fn test_stored_document_roundtrip() {
        let mut attrs = HashMap::new();
        attrs.insert("key".to_string(), "value".to_string());

        let doc = Document {
            id: 42,
            text: "hello world".to_string(),
            embedding: vec![1.0, 2.0, 3.0],
            attributes: attrs,
        };

        let stored = StoredDocument::from_document(&doc);
        assert_eq!(stored.id, 42);
        assert_eq!(stored.text, "hello world");

        let attrs_map = stored.attributes_map();
        assert_eq!(attrs_map.get("key"), Some(&"value".to_string()));
    }
}

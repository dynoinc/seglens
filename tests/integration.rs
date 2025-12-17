//! Integration tests for the seglens library.
//!
//! These tests verify the full roundtrip: building an index and querying it.

use seglens::{Document, IndexBuilder, IndexReader};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

/// Create test documents for indexing.
fn create_test_documents() -> Vec<Document> {
    vec![
        Document {
            id: 0,
            text: "The quick brown fox jumps over the lazy dog".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            attributes: HashMap::from([
                ("category".to_string(), "animals".to_string()),
                ("source".to_string(), "pangram".to_string()),
            ]),
        },
        Document {
            id: 1,
            text: "A journey of a thousand miles begins with a single step".to_string(),
            embedding: vec![0.0, 1.0, 0.0, 0.0],
            attributes: HashMap::from([
                ("category".to_string(), "proverbs".to_string()),
                ("source".to_string(), "laozi".to_string()),
            ]),
        },
        Document {
            id: 2,
            text: "To be or not to be that is the question".to_string(),
            embedding: vec![0.0, 0.0, 1.0, 0.0],
            attributes: HashMap::from([
                ("category".to_string(), "literature".to_string()),
                ("source".to_string(), "shakespeare".to_string()),
            ]),
        },
        Document {
            id: 3,
            text: "The only thing we have to fear is fear itself".to_string(),
            embedding: vec![0.0, 0.0, 0.0, 1.0],
            attributes: HashMap::from([
                ("category".to_string(), "politics".to_string()),
                ("source".to_string(), "fdr".to_string()),
            ]),
        },
        Document {
            id: 4,
            text: "I think therefore I am".to_string(),
            embedding: vec![0.5, 0.5, 0.0, 0.0],
            attributes: HashMap::from([
                ("category".to_string(), "philosophy".to_string()),
                ("source".to_string(), "descartes".to_string()),
            ]),
        },
    ]
}

#[tokio::test]
async fn test_full_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(seglens::local(tmp.path()).unwrap());

    // Build the index
    let mut builder = IndexBuilder::new(4, 2);
    for doc in create_test_documents() {
        builder.add(doc).unwrap();
    }
    builder.build(store.as_ref(), "index").await.unwrap();

    // Open and query the index
    let reader = IndexReader::open(store, "index").await.unwrap();
    assert_eq!(reader.doc_count(), 5);

    // Test lexical search
    let results = reader.search_lexical("fox", 10).await.unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, 0);
    assert!(results[0].text.contains("fox"));

    // Test vector search
    let results = reader
        .search_vector(&[1.0, 0.0, 0.0, 0.0], 10, 2)
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, 0);

    // Test hybrid search
    let results = reader
        .search_hybrid("fox", &[1.0, 0.0, 0.0, 0.0], 10, 2)
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, 0);
}

#[tokio::test]
async fn test_lexical_search_ranking() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(seglens::local(tmp.path()).unwrap());

    let mut builder = IndexBuilder::new(4, 2);
    for doc in create_test_documents() {
        builder.add(doc).unwrap();
    }
    builder.build(store.as_ref(), "index").await.unwrap();

    let reader = IndexReader::open(store, "index").await.unwrap();

    // Search for "fear" - should match doc 3 which has "fear" twice
    let results = reader.search_lexical("fear", 10).await.unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, 3);

    // Search for common word "the" - should match multiple docs
    let results = reader.search_lexical("the", 10).await.unwrap();
    assert!(results.len() >= 2);
}

#[tokio::test]
async fn test_vector_search_similarity() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(seglens::local(tmp.path()).unwrap());

    let mut builder = IndexBuilder::new(4, 2);
    for doc in create_test_documents() {
        builder.add(doc).unwrap();
    }
    builder.build(store.as_ref(), "index").await.unwrap();

    let reader = IndexReader::open(store, "index").await.unwrap();

    // Search with embedding [0.5, 0.5, 0, 0] - should match doc 4 exactly
    let results = reader
        .search_vector(&[0.5, 0.5, 0.0, 0.0], 10, 2)
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, 4);
}

#[tokio::test]
async fn test_hybrid_search_combines_signals() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(seglens::local(tmp.path()).unwrap());

    let mut builder = IndexBuilder::new(4, 2);
    for doc in create_test_documents() {
        builder.add(doc).unwrap();
    }
    builder.build(store.as_ref(), "index").await.unwrap();

    let reader = IndexReader::open(store, "index").await.unwrap();

    // Hybrid search where lexical and vector signals agree
    let results = reader
        .search_hybrid("journey", &[0.0, 1.0, 0.0, 0.0], 10, 2)
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, 1);
}

#[tokio::test]
async fn test_attributes_preserved() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(seglens::local(tmp.path()).unwrap());

    let mut builder = IndexBuilder::new(4, 2);
    for doc in create_test_documents() {
        builder.add(doc).unwrap();
    }
    builder.build(store.as_ref(), "index").await.unwrap();

    let reader = IndexReader::open(store, "index").await.unwrap();

    // Search for "question" which appears in doc 2's text
    let results = reader.search_lexical("question", 10).await.unwrap();
    assert!(!results.is_empty());

    let result = &results[0];
    assert_eq!(result.doc_id, 2);
    assert_eq!(
        result.attributes.get("category"),
        Some(&"literature".to_string())
    );
    assert_eq!(
        result.attributes.get("source"),
        Some(&"shakespeare".to_string())
    );
}

#[tokio::test]
async fn test_empty_results() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(seglens::local(tmp.path()).unwrap());

    let mut builder = IndexBuilder::new(4, 2);
    for doc in create_test_documents() {
        builder.add(doc).unwrap();
    }
    builder.build(store.as_ref(), "index").await.unwrap();

    let reader = IndexReader::open(store, "index").await.unwrap();

    // Search for a term that doesn't exist
    let results = reader.search_lexical("nonexistentterm", 10).await.unwrap();
    assert!(results.is_empty());

    // Empty query
    let results = reader.search_lexical("", 10).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_top_k_limit() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(seglens::local(tmp.path()).unwrap());

    let mut builder = IndexBuilder::new(4, 2);
    for doc in create_test_documents() {
        builder.add(doc).unwrap();
    }
    builder.build(store.as_ref(), "index").await.unwrap();

    let reader = IndexReader::open(store, "index").await.unwrap();

    // Request only top 2 results
    let results = reader.search_lexical("the", 2).await.unwrap();
    assert!(results.len() <= 2);
}

#[tokio::test]
async fn test_disk_based_indexing() {
    let tmp = TempDir::new().unwrap();
    let store_dir = tmp.path().join("store");
    let temp_dir = tmp.path().join("temp");

    let store = Arc::new(seglens::local(&store_dir).unwrap());

    // Build with disk-based storage
    let mut builder = IndexBuilder::new(4, 2).with_temp_dir(&temp_dir).unwrap();
    for doc in create_test_documents() {
        builder.add(doc).unwrap();
    }
    builder.build(store.as_ref(), "index").await.unwrap();

    // Verify index works correctly
    let reader = IndexReader::open(store, "index").await.unwrap();
    assert_eq!(reader.doc_count(), 5);

    let results = reader.search_lexical("fox", 10).await.unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, 0);
}

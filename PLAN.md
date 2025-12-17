# seglens

A Rust library for building and querying hybrid (semantic + lexical) search indexes optimized for object storage.

## Goal

Build a search index that:
1. Stores index in few large segment files + small metadata files
2. Loads only metadata at startup
3. Fetches segments on-demand via range requests
4. Supports BM25 lexical search, IVF vector search, and hybrid (RRF fusion)

## Dependencies

```toml
[dependencies]
object_store = "0.11"          # S3/GCS/local with range reads
rkyv = { version = "0.8", features = ["validation"] }  # zero-copy serde
bytes = "1"
thiserror = "2"
tokio = { version = "1", features = ["rt-multi-thread"] }
```

## File Layout

```
{prefix}/
├── manifest.bin     # Config, doc count, doc pointers (~KB)
├── lexical.idx      # Vocab, posting refs, BM25 stats (~MB)  
├── vector.idx       # IVF centroids, cluster refs (~MB)
└── segments/
    ├── 0000.dat     # Posting lists, cluster vectors, doc data
    ├── 0001.dat     # Target ~64MB per segment
    └── ...
```

## Module Structure

```
seglens/
├── src/
│   ├── lib.rs           # Public API, core types
│   ├── types.rs         # DocId, SegmentPtr, Document, etc.
│   ├── builder.rs       # IndexBuilder
│   ├── reader.rs        # IndexReader + search methods
│   ├── lexical.rs       # Tokenizer, BM25 scoring
│   ├── vector.rs        # IVF: kmeans, nearest centroid, distance fns
│   ├── storage.rs       # BlobStore trait
│   └── object_store.rs  # object_store adapter
├── tests/
│   └── integration.rs   # Build + query roundtrip
└── Cargo.toml
```

## Core Types

```rust
pub type DocId = u32;
pub type TermId = u32;

pub struct Document {
    pub id: DocId,
    pub text: String,
    pub embedding: Vec<f32>,
    pub attributes: HashMap<String, String>,
}

pub struct SegmentPtr {
    pub segment_id: u32,
    pub offset: u64,
    pub length: u32,
}

pub struct SearchResult {
    pub doc_id: DocId,
    pub score: f32,
    pub text: String,
    pub attributes: HashMap<String, String>,
}
```

## Public API

```rust
// Builder
let mut builder = IndexBuilder::new(embedding_dim, num_clusters);
builder.add(doc);
builder.build(&store, "index/v1").await?;

// Reader
let reader = IndexReader::open(store, "index/v1").await?;
let results = reader.search_lexical("query", top_k).await?;
let results = reader.search_vector(&embedding, top_k, n_probe).await?;
let results = reader.search_hybrid("query", &embedding, top_k, n_probe).await?;
```

## Implementation Tasks

### Phase 1: Core Types & Storage
1. [ ] Define all types in `types.rs` with rkyv derives
2. [ ] Implement `BlobStore` trait in `storage.rs`
3. [ ] Implement `object_store` adapter with GCS/S3/local constructors
4. [ ] Write unit tests for storage layer

### Phase 2: Lexical Index
1. [ ] Implement simple tokenizer (lowercase, split on non-alphanumeric, filter len > 1)
2. [ ] Implement BM25 scoring function (k1=1.2, b=0.75)
3. [ ] Build inverted index: vocab HashMap, posting lists
4. [ ] Serialize posting lists to segments, store SegmentPtrs
5. [ ] Implement `search_lexical`: fetch postings, score, return top-k

### Phase 3: Vector Index (IVF)
1. [ ] Implement L2 squared distance, dot product
2. [ ] Implement simple k-means (random init, 10 iterations Lloyd's)
3. [ ] Assign docs to nearest centroid
4. [ ] Serialize cluster data (doc_ids + vectors) to segments
5. [ ] Implement `search_vector`: find nearest centroids, fetch clusters, brute-force

### Phase 4: Hybrid Search
1. [ ] Implement RRF fusion (k=60)
2. [ ] Implement `search_hybrid`: run both searches, fuse, fetch doc metadata

### Phase 5: Builder
1. [ ] Implement SegmentWriter (append buffer, rotate at 64MB)
2. [ ] Wire up lexical + vector + doc writing
3. [ ] Write manifest, lexical.idx, vector.idx files

### Phase 6: Reader
1. [ ] Implement `open()`: load 3 metadata files
2. [ ] Implement segment fetching with SegmentPtr
3. [ ] Wire up all search methods

### Phase 7: Polish
1. [ ] Integration test: build small index, run all query types
2. [ ] Add `thiserror` error types
3. [ ] Concurrent segment fetches (tokio::join! for multiple clusters)
4. [ ] Optional: LRU cache for hot segments
5. [ ] Docs and examples

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector ANN | IVF | Natural segment boundaries, no graph traversal needed |
| Hybrid fusion | RRF | Simple, no tuning parameters |
| Serialization | rkyv | Zero-copy deserialization |
| Segment size | 64MB | Good for GCS range reads |
| Storage abstraction | Trait | Pluggable backends |

## Non-Goals (v1)

- Attribute filtering (just return attributes)
- Index updates (rebuild only)
- Sharding / distributed queries
- Compression

## Testing

```rust
#[tokio::test]
async fn test_roundtrip() {
    let store = local("/tmp/seglens-test");
    
    let mut builder = IndexBuilder::new(4, 2);
    builder.add(Document { id: 0, text: "hello world".into(), embedding: vec![1.,0.,0.,0.], .. });
    builder.add(Document { id: 1, text: "goodbye world".into(), embedding: vec![0.,1.,0.,0.], .. });
    builder.build(&store, "test").await.unwrap();
    
    let reader = IndexReader::open(store, "test").await.unwrap();
    
    let lexical = reader.search_lexical("hello", 10).await.unwrap();
    assert_eq!(lexical[0].0, 0);
    
    let vector = reader.search_vector(&[1.,0.,0.,0.], 10, 2).await.unwrap();
    assert_eq!(vector[0].0, 0);
}
```

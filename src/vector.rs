//! Vector search components: distance functions, k-means clustering, and IVF index.

use crate::types::{DocId, SegmentPtr};
use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

/// Compute squared L2 (Euclidean) distance between two vectors.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Compute dot product between two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normalize a vector to unit length (for cosine similarity via dot product).
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Find the index of the nearest centroid to a given vector.
pub fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_squared(vector, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Find the k nearest centroids to a given vector.
pub fn nearest_centroids(vector: &[f32], centroids: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_squared(vector, c)))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.into_iter().take(k).map(|(i, _)| i).collect()
}

/// K-means clustering with Lloyd's algorithm.
///
/// # Arguments
/// * `vectors` - Data points to cluster
/// * `k` - Number of clusters
/// * `max_iterations` - Maximum number of iterations (default 10)
///
/// Returns cluster centroids.
pub fn kmeans(vectors: &[Vec<f32>], k: usize, max_iterations: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() || k == 0 {
        return Vec::new();
    }

    let dim = vectors[0].len();
    let k = k.min(vectors.len()); // Can't have more clusters than points

    // Initialize centroids by taking first k vectors (simple initialization)
    // A better approach would be k-means++ but this is simpler for v1
    let mut centroids: Vec<Vec<f32>> = vectors.iter().take(k).cloned().collect();

    for _ in 0..max_iterations {
        // Assign points to nearest centroid (parallel)
        let assignments: Vec<usize> = vectors
            .par_iter()
            .map(|v| nearest_centroid(v, &centroids))
            .collect();

        // Prepare sums and counts
        let mut counts = vec![0usize; k];
        let mut sums = vec![vec![0.0f32; dim]; k];

        for (vec_idx, &cluster_idx) in assignments.iter().enumerate() {
            counts[cluster_idx] += 1;
            for (j, val) in vectors[vec_idx].iter().enumerate() {
                sums[cluster_idx][j] += val;
            }
        }

        // Update centroids
        let new_centroids: Vec<Vec<f32>> = (0..k)
            .into_par_iter()
            .map(|cluster_idx| {
                if counts[cluster_idx] == 0 {
                    centroids[cluster_idx].clone()
                } else {
                    let n = counts[cluster_idx] as f32;
                    sums[cluster_idx]
                        .iter()
                        .map(|&v| v / n)
                        .collect::<Vec<f32>>()
                }
            })
            .collect();

        // Check for convergence (centroids didn't change much)
        let mut converged = true;
        for (old, new) in centroids.iter().zip(new_centroids.iter()) {
            if l2_squared(old, new) > 1e-6 {
                converged = false;
                break;
            }
        }

        centroids = new_centroids;

        if converged {
            break;
        }
    }

    centroids
}

/// Assign documents to their nearest cluster.
///
/// Returns a vector where index is doc index and value is cluster index.
pub fn assign_clusters(vectors: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    vectors
        .iter()
        .map(|v| nearest_centroid(v, centroids))
        .collect()
}

/// Data for a single cluster (stored in segments).
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct ClusterData {
    /// Document IDs in this cluster.
    pub doc_ids: Vec<DocId>,
    /// Embeddings for documents in this cluster (flattened).
    pub embeddings: Vec<f32>,
    /// Embedding dimension.
    pub dim: u32,
}

impl ClusterData {
    /// Create a new cluster data.
    pub fn new(dim: u32) -> Self {
        Self {
            doc_ids: Vec::new(),
            embeddings: Vec::new(),
            dim,
        }
    }

    /// Add a document to this cluster.
    pub fn add(&mut self, doc_id: DocId, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.dim as usize);
        self.doc_ids.push(doc_id);
        self.embeddings.extend_from_slice(embedding);
    }

    /// Get the number of documents in this cluster.
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Check if the cluster is empty.
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Get embedding for a document at index.
    pub fn get_embedding(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim as usize;
        let end = start + self.dim as usize;
        &self.embeddings[start..end]
    }

    /// Find k nearest documents to a query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(DocId, f32)> {
        if k == 0 || self.doc_ids.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<(DocId, f32)> = self
            .doc_ids
            .iter()
            .enumerate()
            .map(|(i, &doc_id)| {
                let emb = self.get_embedding(i);
                let dist = l2_squared(query, emb);
                (doc_id, dist)
            })
            .collect();

        // Keep only the closest k using partial sorting.
        let keep = k.min(results.len());
        if results.len() > keep {
            let pivot = keep.saturating_sub(1);
            results.select_nth_unstable_by(pivot, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(keep);
        }

        // Sort the retained slice for deterministic ordering.
        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Convert distance to score (lower distance = higher score)
        // Use 1 / (1 + dist) for score normalization
        results
            .into_iter()
            .map(|(doc_id, dist)| (doc_id, 1.0 / (1.0 + dist)))
            .collect()
    }
}

/// Vector index metadata (loaded at startup).
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct VectorIndex {
    /// Cluster centroids (flattened).
    pub centroids: Vec<f32>,
    /// Number of clusters.
    pub num_clusters: u32,
    /// Embedding dimension.
    pub dim: u32,
    /// Pointers to cluster data in segments.
    pub cluster_ptrs: Vec<SegmentPtr>,
}

impl VectorIndex {
    /// Create a new empty vector index.
    pub fn new(dim: u32) -> Self {
        Self {
            centroids: Vec::new(),
            num_clusters: 0,
            dim,
            cluster_ptrs: Vec::new(),
        }
    }

    /// Get centroid for a cluster.
    pub fn get_centroid(&self, cluster_idx: usize) -> &[f32] {
        let start = cluster_idx * self.dim as usize;
        let end = start + self.dim as usize;
        &self.centroids[start..end]
    }

    /// Get all centroids as vectors.
    pub fn get_centroids(&self) -> Vec<Vec<f32>> {
        (0..self.num_clusters as usize)
            .map(|i| self.get_centroid(i).to_vec())
            .collect()
    }

    /// Find the n_probe nearest clusters to a query.
    pub fn find_clusters(&self, query: &[f32], n_probe: usize) -> Vec<usize> {
        let centroids = self.get_centroids();
        nearest_centroids(query, &centroids, n_probe)
    }
}

/// Builder for constructing the vector index.
pub struct VectorIndexBuilder {
    /// Collected embeddings (doc_id, embedding).
    embeddings: Vec<(DocId, Vec<f32>)>,
    /// Embedding dimension.
    dim: u32,
    /// Number of clusters.
    num_clusters: u32,
}

impl VectorIndexBuilder {
    /// Create a new builder.
    pub fn new(dim: u32, num_clusters: u32) -> Self {
        Self {
            embeddings: Vec::new(),
            dim,
            num_clusters,
        }
    }

    /// Add a document embedding.
    pub fn add(&mut self, doc_id: DocId, embedding: Vec<f32>) {
        debug_assert_eq!(embedding.len(), self.dim as usize);
        self.embeddings.push((doc_id, embedding));
    }

    /// Train centroids and assign documents to clusters.
    ///
    /// Returns (centroids, cluster_data) where cluster_data[i] contains
    /// documents assigned to cluster i.
    pub fn build(&self) -> (Vec<Vec<f32>>, Vec<ClusterData>) {
        if self.embeddings.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let vectors: Vec<Vec<f32>> = self.embeddings.iter().map(|(_, e)| e.clone()).collect();

        // Train k-means
        let centroids = kmeans(&vectors, self.num_clusters as usize, 10);

        // Assign documents to clusters
        let assignments = assign_clusters(&vectors, &centroids);

        // Build cluster data
        let mut cluster_data: Vec<ClusterData> = (0..centroids.len())
            .map(|_| ClusterData::new(self.dim))
            .collect();

        for (i, (doc_id, embedding)) in self.embeddings.iter().enumerate() {
            let cluster_idx = assignments[i];
            cluster_data[cluster_idx].add(*doc_id, embedding);
        }

        (centroids, cluster_data)
    }
}

// ============================================================================
// Streaming Vector Builder (for large datasets)
// ============================================================================

use rkyv::rancor::Error as RkyvError;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

/// Write buffer size per cluster (8 MB).
const CLUSTER_BUFFER_SIZE: usize = 2 * 1024 * 1024;
/// Total in-memory budget across clusters before forcing flushes (512 MB).
const MAX_BUFFER_BYTES: usize = 512 * 1024 * 1024;

/// Streaming vector index builder for large datasets.
///
/// Takes pre-trained centroids and streams documents through, buffering
/// cluster entries and flushing to disk when memory threshold is reached.
pub struct StreamingVectorBuilder {
    /// Embedding dimension.
    dim: u32,
    /// Pre-trained centroids.
    centroids: Vec<Vec<f32>>,
    /// Temporary directory for cluster files.
    temp_dir: PathBuf,
    /// Per-cluster buffers (doc_ids, embeddings).
    cluster_buffers: Vec<(Vec<DocId>, Vec<f32>)>,
    /// Per-cluster buffer sizes (in bytes).
    cluster_sizes: Vec<usize>,
    /// Total buffered bytes across all clusters.
    total_buffered_bytes: usize,
    /// Number of flushes per cluster.
    flush_counts: Vec<u32>,
}

impl StreamingVectorBuilder {
    /// Create a new streaming builder with pre-trained centroids.
    pub fn new(centroids: Vec<Vec<f32>>, temp_dir: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&temp_dir)?;

        let num_clusters = centroids.len();
        let dim = centroids.first().map(|c| c.len() as u32).unwrap_or(0);

        Ok(Self {
            dim,
            centroids,
            temp_dir,
            cluster_buffers: vec![(Vec::new(), Vec::new()); num_clusters],
            cluster_sizes: vec![0; num_clusters],
            total_buffered_bytes: 0,
            flush_counts: vec![0; num_clusters],
        })
    }

    /// Add a document to the appropriate cluster.
    pub fn add(&mut self, doc_id: DocId, embedding: &[f32]) -> std::io::Result<()> {
        let cluster_idx = nearest_centroid(embedding, &self.centroids);

        // Add to buffer
        let (doc_ids, embeddings) = &mut self.cluster_buffers[cluster_idx];
        doc_ids.push(doc_id);
        embeddings.extend_from_slice(embedding);

        // Update size estimate (4 bytes per doc_id + 4 bytes per f32)
        let added = 4 + embedding.len() * 4;
        self.cluster_sizes[cluster_idx] += added;
        self.total_buffered_bytes += added;

        // Flush if buffer is full or we exceed global budget.
        if self.cluster_sizes[cluster_idx] >= CLUSTER_BUFFER_SIZE
            || self.total_buffered_bytes >= MAX_BUFFER_BYTES
        {
            let target_idx = if self.total_buffered_bytes >= MAX_BUFFER_BYTES {
                self.largest_cluster_index().unwrap_or(cluster_idx)
            } else {
                cluster_idx
            };
            self.flush_cluster(target_idx)?;
        }

        Ok(())
    }

    /// Flush a single cluster's buffer to disk.
    fn flush_cluster(&mut self, cluster_idx: usize) -> std::io::Result<()> {
        let (doc_ids, embeddings) = &mut self.cluster_buffers[cluster_idx];

        if doc_ids.is_empty() {
            return Ok(());
        }

        let flush_id = self.flush_counts[cluster_idx];
        let path = self
            .temp_dir
            .join(format!("cluster_{:04}_{:04}.bin", cluster_idx, flush_id));

        let cluster_data = ClusterData {
            doc_ids: doc_ids.clone(),
            embeddings: embeddings.clone(),
            dim: self.dim,
        };

        let serialized = rkyv::to_bytes::<RkyvError>(&cluster_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let mut writer = BufWriter::with_capacity(CLUSTER_BUFFER_SIZE, File::create(&path)?);
        writer.write_all(&serialized)?;
        writer.flush()?;

        // Clear buffer and update accounting
        let flushed = self.cluster_sizes[cluster_idx];
        doc_ids.clear();
        embeddings.clear();
        self.cluster_sizes[cluster_idx] = 0;
        self.total_buffered_bytes = self.total_buffered_bytes.saturating_sub(flushed);
        self.flush_counts[cluster_idx] += 1;

        Ok(())
    }

    /// Build the final cluster data by merging all flushed files.
    pub fn build(mut self) -> std::io::Result<(Vec<Vec<f32>>, Vec<ClusterData>)> {
        // Flush remaining buffers
        for i in 0..self.centroids.len() {
            self.flush_cluster(i)?;
        }

        // Merge cluster files
        let mut cluster_data: Vec<ClusterData> = Vec::with_capacity(self.centroids.len());

        for cluster_idx in 0..self.centroids.len() {
            let mut merged = ClusterData::new(self.dim);

            // Load all flush files for this cluster
            for flush_id in 0..self.flush_counts[cluster_idx] {
                let path = self
                    .temp_dir
                    .join(format!("cluster_{:04}_{:04}.bin", cluster_idx, flush_id));

                if path.exists() {
                    let mut file = BufReader::new(File::open(&path)?);
                    let mut data = Vec::new();
                    file.read_to_end(&mut data)?;

                    let chunk: ClusterData = rkyv::from_bytes::<ClusterData, RkyvError>(&data)
                        .map_err(|e| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                        })?;

                    // Merge into result
                    merged.doc_ids.extend(chunk.doc_ids);
                    merged.embeddings.extend(chunk.embeddings);

                    // Clean up
                    let _ = std::fs::remove_file(path);
                }
            }

            cluster_data.push(merged);
        }

        Ok((self.centroids, cluster_data))
    }

    /// Get the number of clusters.
    pub fn num_clusters(&self) -> usize {
        self.centroids.len()
    }

    /// Get the embedding dimension.
    pub fn dim(&self) -> u32 {
        self.dim
    }

    fn largest_cluster_index(&self) -> Option<usize> {
        self.cluster_sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, sz)| *sz)
            .map(|(idx, _)| idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_squared() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((l2_squared(&a, &b) - 2.0).abs() < 1e-6);

        let c = vec![1.0, 1.0, 1.0];
        let d = vec![1.0, 1.0, 1.0];
        assert!((l2_squared(&c, &d) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);

        // Check it's unit length
        let len: f32 = n.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nearest_centroid() {
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0], vec![20.0, 20.0]];

        let query1 = vec![1.0, 1.0];
        assert_eq!(nearest_centroid(&query1, &centroids), 0);

        let query2 = vec![9.0, 11.0];
        assert_eq!(nearest_centroid(&query2, &centroids), 1);

        let query3 = vec![19.0, 21.0];
        assert_eq!(nearest_centroid(&query3, &centroids), 2);
    }

    #[test]
    fn test_nearest_centroids() {
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0], vec![20.0, 20.0]];

        let query = vec![8.0, 8.0];
        let nearest = nearest_centroids(&query, &centroids, 2);

        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0], 1); // Closest
        assert!(nearest.contains(&0) || nearest.contains(&2)); // Second closest
    }

    #[test]
    fn test_kmeans_basic() {
        // Two clear clusters
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 10.0],
            vec![10.0, 11.0],
        ];

        let centroids = kmeans(&vectors, 2, 10);
        assert_eq!(centroids.len(), 2);

        // One centroid should be near origin, one near (10, 10)
        let near_origin = centroids.iter().any(|c| l2_squared(c, &[0.33, 0.33]) < 2.0);
        let near_ten = centroids
            .iter()
            .any(|c| l2_squared(c, &[10.33, 10.33]) < 2.0);

        assert!(near_origin);
        assert!(near_ten);
    }

    #[test]
    fn test_cluster_data() {
        let mut cluster = ClusterData::new(3);
        cluster.add(0, &[1.0, 0.0, 0.0]);
        cluster.add(1, &[0.0, 1.0, 0.0]);
        cluster.add(2, &[0.0, 0.0, 1.0]);

        assert_eq!(cluster.len(), 3);
        assert!(!cluster.is_empty());

        let emb = cluster.get_embedding(1);
        assert_eq!(emb, &[0.0, 1.0, 0.0]);

        // Search for nearest to [1, 0, 0]
        let results = cluster.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // doc_id 0 is closest
    }

    #[test]
    fn test_vector_index_builder() {
        let mut builder = VectorIndexBuilder::new(4, 2);
        builder.add(0, vec![1.0, 0.0, 0.0, 0.0]);
        builder.add(1, vec![1.0, 0.1, 0.0, 0.0]);
        builder.add(2, vec![0.0, 0.0, 1.0, 0.0]);
        builder.add(3, vec![0.0, 0.0, 1.0, 0.1]);

        let (centroids, cluster_data) = builder.build();

        assert_eq!(centroids.len(), 2);
        assert_eq!(cluster_data.len(), 2);

        // All documents should be assigned
        let total_docs: usize = cluster_data.iter().map(|c| c.len()).sum();
        assert_eq!(total_docs, 4);
    }

    #[test]
    fn test_vector_index() {
        let index = VectorIndex {
            centroids: vec![0.0, 0.0, 10.0, 10.0],
            num_clusters: 2,
            dim: 2,
            cluster_ptrs: vec![
                SegmentPtr {
                    segment_id: 0,
                    offset: 0,
                    length: 100,
                },
                SegmentPtr {
                    segment_id: 0,
                    offset: 100,
                    length: 100,
                },
            ],
        };

        assert_eq!(index.get_centroid(0), &[0.0, 0.0]);
        assert_eq!(index.get_centroid(1), &[10.0, 10.0]);

        let centroids = index.get_centroids();
        assert_eq!(centroids.len(), 2);

        let clusters = index.find_clusters(&[1.0, 1.0], 1);
        assert_eq!(clusters, vec![0]);
    }

    #[test]
    fn test_streaming_vector_builder() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Pre-compute centroids for 2 clusters
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        let mut builder = StreamingVectorBuilder::new(centroids, tmp.path().to_path_buf()).unwrap();

        // Add documents - should be assigned to appropriate clusters
        builder.add(0, &[0.0, 1.0]).unwrap(); // Near cluster 0
        builder.add(1, &[1.0, 0.0]).unwrap(); // Near cluster 0
        builder.add(2, &[9.0, 10.0]).unwrap(); // Near cluster 1
        builder.add(3, &[10.0, 9.0]).unwrap(); // Near cluster 1

        let (centroids, cluster_data) = builder.build().unwrap();

        assert_eq!(centroids.len(), 2);
        assert_eq!(cluster_data.len(), 2);

        // Total documents should be 4
        let total_docs: usize = cluster_data.iter().map(|c| c.len()).sum();
        assert_eq!(total_docs, 4);

        // Cluster 0 should have docs 0 and 1
        let cluster0_docs: Vec<_> = cluster_data[0].doc_ids.clone();
        assert!(cluster0_docs.contains(&0) && cluster0_docs.contains(&1));

        // Cluster 1 should have docs 2 and 3
        let cluster1_docs: Vec<_> = cluster_data[1].doc_ids.clone();
        assert!(cluster1_docs.contains(&2) && cluster1_docs.contains(&3));
    }
}

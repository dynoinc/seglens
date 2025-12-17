//! Segment cache for efficient repeated reads.
//!
//! Caches downloaded segment data to disk with 2MB aligned chunks.
//! Uses memory-mapped files for zero-copy access.

use bytes::Bytes;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

/// Cache chunk size (2MB).
const CHUNK_SIZE: u64 = 2 * 1024 * 1024;

/// A memory-mapped chunk.
struct MappedChunk {
    /// The memory-mapped file.
    mmap: Mmap,
    /// Start offset of this chunk.
    start: u64,
}

/// Track which chunks exist on disk and their mmaps.
#[derive(Default)]
struct ChunkTracker {
    /// Memory-mapped chunks keyed by (segment_id, chunk_index).
    mmaps: HashMap<(u32, u64), MappedChunk>,
}

/// File-based segment cache with memory-mapped access.
pub struct SegmentCache {
    /// Cache directory path.
    cache_dir: PathBuf,
    /// Track chunks and their mmaps.
    tracker: Mutex<ChunkTracker>,
}

impl SegmentCache {
    /// Create a new file-based cache in /tmp with a unique subdirectory.
    pub fn new() -> Self {
        // Use a unique directory per instance to avoid conflicts
        let unique_id = std::process::id();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let cache_dir = std::env::temp_dir()
            .join("seglens-cache")
            .join(format!("{}-{}", unique_id, timestamp));
        Self::with_dir(cache_dir)
    }

    /// Create a new cache in a specific directory.
    pub fn with_dir(cache_dir: PathBuf) -> Self {
        // Create cache directory if it doesn't exist
        let _ = fs::create_dir_all(&cache_dir);

        Self {
            cache_dir,
            tracker: Mutex::new(ChunkTracker::default()),
        }
    }

    /// Get chunk size for aligned reads.
    pub fn chunk_size() -> u64 {
        CHUNK_SIZE
    }

    /// Align an offset down to chunk boundary.
    pub fn align_down(offset: u64) -> u64 {
        (offset / CHUNK_SIZE) * CHUNK_SIZE
    }

    /// Align an offset up to chunk boundary.
    pub fn align_up(offset: u64) -> u64 {
        offset.div_ceil(CHUNK_SIZE) * CHUNK_SIZE
    }

    /// Get the chunk index for an offset.
    pub fn chunk_index(offset: u64) -> u64 {
        offset / CHUNK_SIZE
    }

    /// Get file path for a chunk.
    fn chunk_path(&self, segment_id: u32, chunk_idx: u64) -> PathBuf {
        self.cache_dir
            .join(format!("seg{:04}_chunk{:08}.bin", segment_id, chunk_idx))
    }

    /// Calculate which chunks are needed to cover a range.
    pub fn chunks_for_range(start: u64, end: u64) -> Vec<(u64, u64)> {
        let first_chunk = Self::chunk_index(start);
        let last_chunk = Self::chunk_index(end.saturating_sub(1));

        (first_chunk..=last_chunk)
            .map(|idx| {
                let chunk_start = idx * CHUNK_SIZE;
                let chunk_end = chunk_start + CHUNK_SIZE;
                (chunk_start, chunk_end)
            })
            .collect()
    }

    /// Try to memory-map a chunk if it exists on disk.
    fn try_mmap_chunk(&self, segment_id: u32, chunk_idx: u64) -> Option<()> {
        let path = self.chunk_path(segment_id, chunk_idx);

        // Check if already mmapped
        {
            let tracker = self.tracker.lock().unwrap();
            if tracker.mmaps.contains_key(&(segment_id, chunk_idx)) {
                return Some(());
            }
        }

        // Try to open and mmap the file
        let file = File::open(&path).ok()?;
        let mmap = unsafe { Mmap::map(&file).ok()? };

        let mut tracker = self.tracker.lock().unwrap();
        tracker.mmaps.insert(
            (segment_id, chunk_idx),
            MappedChunk {
                mmap,
                start: chunk_idx * CHUNK_SIZE,
            },
        );

        Some(())
    }

    /// Get a slice of data from a mmapped chunk.
    fn get_chunk_slice(
        &self,
        segment_id: u32,
        chunk_idx: u64,
        start: u64,
        end: u64,
    ) -> Option<Bytes> {
        let tracker = self.tracker.lock().unwrap();
        let chunk = tracker.mmaps.get(&(segment_id, chunk_idx))?;

        let chunk_start = chunk.start;
        let chunk_end = chunk_start + chunk.mmap.len() as u64;

        // Calculate overlap with requested range
        let overlap_start = start.max(chunk_start);
        let overlap_end = end.min(chunk_end);

        if overlap_start >= overlap_end {
            return None;
        }

        let local_start = (overlap_start - chunk_start) as usize;
        let local_end = (overlap_end - chunk_start) as usize;

        Some(Bytes::copy_from_slice(&chunk.mmap[local_start..local_end]))
    }

    /// Try to get data from cache for a range.
    ///
    /// Returns Hit if all chunks covering the range are cached, Miss otherwise.
    pub fn get(&self, segment_id: u32, start: u64, end: u64) -> CacheResult {
        let needed_chunks = Self::chunks_for_range(start, end);
        let mut missing_ranges: Vec<(u64, u64)> = Vec::new();
        let mut all_cached = true;

        // First pass: check which chunks exist and try to mmap them
        for (chunk_start, chunk_end) in &needed_chunks {
            let chunk_idx = Self::chunk_index(*chunk_start);
            if self.try_mmap_chunk(segment_id, chunk_idx).is_none() {
                missing_ranges.push((*chunk_start, *chunk_end));
                all_cached = false;
            }
        }

        if !all_cached {
            return CacheResult::Miss(missing_ranges);
        }

        // All chunks cached, assemble the data
        let mut result = Vec::with_capacity((end - start) as usize);

        for (chunk_start, _) in needed_chunks {
            let chunk_idx = Self::chunk_index(chunk_start);
            if let Some(slice) = self.get_chunk_slice(segment_id, chunk_idx, start, end) {
                result.extend_from_slice(&slice);
            }
        }

        CacheResult::Hit(Bytes::from(result))
    }

    /// Store a chunk in the cache (writes to disk).
    pub fn put(&self, segment_id: u32, chunk_start: u64, data: Bytes) {
        let chunk_idx = Self::chunk_index(chunk_start);
        let path = self.chunk_path(segment_id, chunk_idx);

        // Write to disk
        if let Ok(mut file) = File::create(&path) {
            let _ = file.write_all(&data);
        }

        // Memory-map the file
        let _ = self.try_mmap_chunk(segment_id, chunk_idx);
    }

    /// Clear all cached data.
    pub fn clear(&self) {
        let mut tracker = self.tracker.lock().unwrap();
        tracker.mmaps.clear();

        // Remove all files in cache directory
        if let Ok(entries) = fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
    }

    /// Get the cache directory path.
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }
}

impl Default for SegmentCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a cache lookup.
pub enum CacheResult {
    /// All data was found in cache.
    Hit(Bytes),
    /// Some or all data was missing. Contains list of missing (start, end) ranges.
    Miss(Vec<(u64, u64)>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_align_down() {
        assert_eq!(SegmentCache::align_down(0), 0);
        assert_eq!(SegmentCache::align_down(1000), 0);
        assert_eq!(SegmentCache::align_down(CHUNK_SIZE), CHUNK_SIZE);
        assert_eq!(SegmentCache::align_down(CHUNK_SIZE + 1000), CHUNK_SIZE);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(SegmentCache::align_up(0), 0);
        assert_eq!(SegmentCache::align_up(1), CHUNK_SIZE);
        assert_eq!(SegmentCache::align_up(CHUNK_SIZE), CHUNK_SIZE);
        assert_eq!(SegmentCache::align_up(CHUNK_SIZE + 1), CHUNK_SIZE * 2);
    }

    #[test]
    fn test_chunk_index() {
        assert_eq!(SegmentCache::chunk_index(0), 0);
        assert_eq!(SegmentCache::chunk_index(CHUNK_SIZE - 1), 0);
        assert_eq!(SegmentCache::chunk_index(CHUNK_SIZE), 1);
        assert_eq!(SegmentCache::chunk_index(CHUNK_SIZE * 2 + 1000), 2);
    }

    #[test]
    fn test_chunks_for_range() {
        // Within single chunk
        let chunks = SegmentCache::chunks_for_range(100, 1000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], (0, CHUNK_SIZE));

        // Spanning two chunks
        let chunks = SegmentCache::chunks_for_range(CHUNK_SIZE - 100, CHUNK_SIZE + 100);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], (0, CHUNK_SIZE));
        assert_eq!(chunks[1], (CHUNK_SIZE, CHUNK_SIZE * 2));
    }

    #[test]
    fn test_cache_put_get() {
        let tmp = TempDir::new().unwrap();
        let cache = SegmentCache::with_dir(tmp.path().to_path_buf());

        // Store a chunk
        let data = Bytes::from(vec![42u8; CHUNK_SIZE as usize]);
        cache.put(0, 0, data);

        // Try to get a range within that chunk
        match cache.get(0, 100, 200) {
            CacheResult::Hit(bytes) => {
                assert_eq!(bytes.len(), 100);
                assert!(bytes.iter().all(|&b| b == 42));
            }
            CacheResult::Miss(_) => panic!("expected cache hit"),
        }
    }

    #[test]
    fn test_cache_miss() {
        let tmp = TempDir::new().unwrap();
        let cache = SegmentCache::with_dir(tmp.path().to_path_buf());

        match cache.get(0, 100, 200) {
            CacheResult::Hit(_) => panic!("expected cache miss"),
            CacheResult::Miss(ranges) => {
                assert_eq!(ranges.len(), 1);
                assert_eq!(ranges[0], (0, CHUNK_SIZE));
            }
        }
    }

    #[test]
    fn test_cache_persistence() {
        let tmp = TempDir::new().unwrap();
        let cache_dir = tmp.path().to_path_buf();

        // Write data with one cache instance
        {
            let cache = SegmentCache::with_dir(cache_dir.clone());
            let data = Bytes::from(vec![99u8; CHUNK_SIZE as usize]);
            cache.put(0, 0, data);
        }

        // Read data with a new cache instance (tests mmap from existing file)
        {
            let cache = SegmentCache::with_dir(cache_dir);
            match cache.get(0, 0, 100) {
                CacheResult::Hit(bytes) => {
                    assert_eq!(bytes.len(), 100);
                    assert!(bytes.iter().all(|&b| b == 99));
                }
                CacheResult::Miss(_) => panic!("expected cache hit after persistence"),
            }
        }
    }

    #[test]
    fn test_cache_clear() {
        let tmp = TempDir::new().unwrap();
        let cache = SegmentCache::with_dir(tmp.path().to_path_buf());

        cache.put(0, 0, Bytes::from(vec![0u8; 100]));
        cache.clear();

        match cache.get(0, 0, 100) {
            CacheResult::Hit(_) => panic!("expected cache miss after clear"),
            CacheResult::Miss(_) => {}
        }
    }
}

# seglens

[![test](https://github.com/dynoinc/seglens/actions/workflows/test.yml/badge.svg)](https://github.com/dynoinc/seglens/actions/workflows/test.yml)

Search index in a box

## Getting started

### Build the Wikipedia index binary
1. Build and run the indexer to stream the English Wikipedia dump into a local directory:
   ```bash
   cargo run --release --bin build-wikipedia-index -- --output ./wiki-index --temp ./wiki-temp --limit 50_000 --dim 128
   ```
   The index will be stored under `./wiki-index/wiki`.

### Query from Python via PyO3
1. Install the Python extension (requires `maturin` in your environment):
   ```bash
   uv run --with maturin maturin develop --release --features python
   ```
2. Open an interactive session and query the index:
   ```bash
   uv run ipython
   >>> import seglens
   >>> index = seglens.PyIndex("./wiki-index", "wiki")
   >>> index.doc_count
   50000
   >>> index.search_lexical("machine learning", 3)[0].text[:120]
   "Machine learning is a field of study that gives computers the ability to learn..."
   ```

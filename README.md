# seglens

[![test](https://github.com/dynoinc/seglens/actions/workflows/test.yml/badge.svg)](https://github.com/dynoinc/seglens/actions/workflows/test.yml)

Search index in a box

## Getting started

### 1) Prepare a parquet dataset with embeddings
Use the Hugging Face English Wikipedia dataset and `sentence-transformers` to materialize a
small parquet shard that matches the expected schema (`id`, `text`, `embeddings`):

```bash
uv run --with datasets --with sentence-transformers python - <<'PY'
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

out_dir = Path("data/wiki-parquet")
out_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:2000]")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(batch):
    batch["embeddings"] = model.encode(batch["text"], convert_to_numpy=True).astype("float32")
    batch["id"] = [str(i) for i in batch["id"]]
    return batch

ds = ds.map(embed, batched=True, batch_size=32)
ds.select_columns(["id", "text", "embeddings"]).to_parquet(out_dir / "wiki.parquet")
print("Wrote", out_dir / "wiki.parquet")
PY
```

### 2) Build an index from the parquet files
Run the generic parquet indexer against the directory that contains the parquet shards:

```bash
cargo run --release --bin build-parquet-index -- --input ./data/wiki-parquet --output ./wiki-index --temp ./wiki-temp --index-name wiki --limit 2_000
```

### 3) Query from Python via PyO3
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
   2000
   >>> index.search_lexical("machine learning", 3)[0].attributes["id"]
   '12345'
   ```

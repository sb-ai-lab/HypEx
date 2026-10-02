# `hypex.config` — Centralized Configuration

Centralized configuration module for the HypEx library. Contains all tunable
parameters for dataset display, Spark interactions, and FAISS-based matching in
a single, easily modifiable location.

## Role in the architecture

This module provides **runtime configuration** that controls various aspects of
HypEx behavior without requiring code changes. It serves as a single source of
truth for default values and limits used across different components.

```
User Code                    HypEx Components                  This Module
─────────────────────────────────────────────────────────────────────────────
sets parameters  ──────────────►  reads defaults ──────────────►  DatasetConfig
                                        │                          MatchingConfig
                                        └──────────────────────────────────┘
```

## File map

| File | Contents |
|---|---|
| `config.py` | `DatasetConfig`, `MatchingConfig` — configuration classes for dataset and matching operations. |
| `__init__.py` | Public exports of configuration classes. |

## Key classes

### `DatasetConfig`

Configuration constants for dataset display and Spark operations.

**Display Controls:**
- `DISPLAY_ROWS: int = 5` — Maximum number of rows to display in Dataset string/HTML representation
- `DISPLAY_COLS: int = 10` — Maximum number of columns to display in Dataset representation

**Spark-Pandas Interoperability:**
- `SPARK_PANDAS_CONVERSION_LIMIT: int = 100_000` — Maximum rows allowed when converting Spark → Pandas
- `SPARK_MAX_ROWS_FOR_DOT: int = 1_000` — Maximum rows for Spark `dot()` operation right-hand operand
- `SPARK_INDEX_COL: str = "index"` — Default index column name for Spark operations
- `BACKEND_CONVERSION_INDEX_COL: str = "__hypex_temp_index__"` — Temporary index column for backend conversions

**Usage example:**
```python
from hypex.config import DatasetConfig

# Change display limits for debugging
DatasetConfig.DISPLAY_ROWS = 20
DatasetConfig.DISPLAY_COLS = 15

# Increase Spark-Pandas conversion limit for large datasets
DatasetConfig.SPARK_PANDAS_CONVERSION_LIMIT = 500_000
```

### `MatchingConfig` (dataclass)

Configuration constants for FAISS-based distributed matching pipeline.

**Storage and Persistence:**
- `FAISS_PERSIST_POLITIC: StorageLevel = StorageLevel.MEMORY_AND_DISK` — Spark storage level for intermediate RDDs
- `FAISS_PERSIST_SEARCH: bool = True` — Whether to persist search results
- `FAISS_UNPERSIST_AFTER_SEARCH: bool = True` — Whether to unpersist after search

**Sampling and Training:**
- `FAISS_SAMPLE_TARGET: int = 5_000_000` — Target rows for IVF quantizer training in sample mode
- `FAISS_DRIVER_INDEX_LIMIT: int = 1_000_000` — Max vectors per batch during prefit phase

**Index Configuration:**
- `FAISS_NLIST: int = 100` — Number of clusters for IVF index
- `FAISS_NPROBE: int = 10` — Number of clusters to search during query
- `FAISS_METRIC_TYPE: str = "L2"` — Distance metric for FAISS (L2, IP, COSINE)

**Batch Processing:**
- `FAISS_BATCH_SIZE: int = 100_000` — Number of vectors processed per batch
- `FAISS_MAX_BATCH_SIZE: int = 1_000_000` — Maximum batch size

**Resource Management:**
- `FAISS_GPU_MODE: bool = False` — Whether to use GPU-accelerated FAISS
- `FAISS_FAST_MODE_THRESHOLD: int = 1_000_000` — Dataset size threshold for fast mode

**Usage example:**
```python
from hypex.config import MatchingConfig

# Use GPU-accelerated matching for large datasets
MatchingConfig.FAISS_GPU_MODE = True

# Increase batch size for better throughput
MatchingConfig.FAISS_BATCH_SIZE = 200_000

# Use cosine similarity instead of L2 distance
MatchingConfig.FAISS_METRIC_TYPE = "COSINE"
```

## How to work with it

### Changing configuration at runtime

All configuration parameters can be modified directly:

```python
import hypex
from hypex.config import DatasetConfig, MatchingConfig

# Increase display limits for debugging
DatasetConfig.DISPLAY_ROWS = 50
DatasetConfig.DISPLAY_COLS = 20

# Optimize matching for large datasets
MatchingConfig.FAISS_BATCH_SIZE = 500_000
MatchingConfig.FAISS_NLIST = 200
```

### Using configuration in your code

```python
from hypex.config import DatasetConfig, MatchingConfig

def my_function(data):
    # Respect the configured display limits
    if len(data) > DatasetConfig.DISPLAY_ROWS:
        data = data.head(DatasetConfig.DISPLAY_ROWS)
    
    # Use configured batch size
    batch_size = MatchingConfig.FAISS_BATCH_SIZE
    # Process data in batches...
```

## How to add new configuration

1. Add the configuration parameter to the appropriate class (`DatasetConfig` or `MatchingConfig`)
2. Document it in this README.md
3. Use the parameter consistently throughout the codebase

```python
# In config.py
class MyNewConfig:
    MY_PARAMETER: ClassVar[int] = 100  # Default value

# In __init__.py
from .config import MyNewConfig
__all__ = [..., "MyNewConfig"]
```

## Gotchas

* **Class variables vs instances** — Configuration parameters are class variables, not instance variables. Change them directly on the class.
* **Performance impact** — Some parameters (like `FAISS_BATCH_SIZE`) can significantly impact performance. Test changes thoroughly.
* **Memory limits** — Parameters like `SPARK_PANDAS_CONVERSION_LIMIT` exist to prevent out-of-memory errors. Increasing them may cause crashes.
* **Thread safety** — Configuration changes affect all instances. Be cautious in multi-threaded environments.

## Configuration Reference

### Dataset Configuration

| Parameter | Default | Purpose | Impact |
|---|---|---|---|
| `DISPLAY_ROWS` | 5 | Max rows in Dataset display | Affects notebook/terminal output |
| `DISPLAY_COLS` | 10 | Max columns in Dataset display | Affects notebook/terminal output |
| `SPARK_PANDAS_CONVERSION_LIMIT` | 100,000 | Max rows for Spark→Pandas conversion | Prevents OOM errors |
| `SPARK_MAX_ROWS_FOR_DOT` | 1,000 | Max rows for Spark dot() operation | Prevents memory issues |

### Matching Configuration

| Parameter | Default | Purpose | Impact |
|---|---|---|---|
| `FAISS_PERSIST_POLITIC` | MEMORY_AND_DISK | Storage level for FAISS RDDs | Memory/performance tradeoff |
| `FAISS_SAMPLE_TARGET` | 5,000,000 | Target rows for IVF training | Training accuracy vs memory |
| `FAISS_DRIVER_INDEX_LIMIT` | 1,000,000 | Max vectors per batch in prefit | Driver memory usage |
| `FAISS_NLIST` | 100 | Number of IVF clusters | Accuracy/performance tradeoff |
| `FAISS_NPROBE` | 10 | Clusters to search per query | Accuracy/performance tradeoff |
| `FAISS_METRIC_TYPE` | "L2" | Distance metric | Matching behavior |
| `FAISS_BATCH_SIZE` | 100,000 | Vectors per processing batch | Throughput/memory tradeoff |
| `FAISS_GPU_MODE` | False | Use GPU acceleration | Hardware requirements |

## Related modules

`../dataset/README.md` (uses DatasetConfig for display limits) ·
`../ml/README.md` (uses MatchingConfig for FAISS operations) ·
`../dataset/backends/README.md` (backend-specific configuration impacts).

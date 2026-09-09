"""
Configuration module for HypEx library.

This module contains centralized configuration constants used across
the HypEx pipeline. All tunable parameters for dataset display,
Spark interactions, and FAISS-based matching are defined here to
provide a single point of modification without touching core logic.

Classes:
    DatasetConfig: Configuration parameters for dataset representation,
        display limits, and Spark-Pandas interoperability.
    MatchingConfig: Configuration parameters for the distributed FAISS
        matching pipeline, including persistence policies, sampling
        targets, and batch sizes.
"""
from typing import Literal

from pyspark import StorageLevel


class DatasetConfig:
    """
    Configuration constants for dataset display and Spark operations.

    These parameters control how datasets are rendered in notebooks/terminals,
    the thresholds for converting between Spark and Pandas representations,
    and limits on operations that may be memory-intensive on distributed data.

    Attributes:
        DISPLAY_ROWS (int): Maximum number of rows to display in the
            string/HTML representation of a Dataset. Rows beyond this
            limit are omitted with an ellipsis indicator.
            Defaults to 5.

        DISPLAY_COLS (int): Maximum number of columns to display in the
            string/HTML representation of a Dataset. Columns beyond this
            limit are truncated.
            Defaults to 10.

        SPARK_PANDAS_CONVERSION_LIMIT (int): Maximum number of rows
            allowed when converting a Spark DataFrame to a Pandas
            DataFrame (e.g., via ``toPandas()``). Operations exceeding
            this limit should remain in the distributed Spark domain.
            Defaults to 100_000.

        SPARK_MAX_ROWS_FOR_DOT (int): Maximum number of rows permitted
            in the right-hand operand of a ``dot()`` operation when
            using the Spark backend. This prevents excessive memory
            usage on the driver during matrix multiplication.
            Defaults to 1000.

        SPARK_INDEX_COL (str): Default column name used as the index
            when converting between Spark DataFrames and the HypEx
            Dataset abstraction. This column preserves row identity
            across distributed operations.
            Defaults to "index".
    """
    DISPLAY_ROWS: int = 5
    DISPLAY_COLS: int = 10

    SPARK_PANDAS_CONVERSION_LIMIT: int = 100_000
    SPARK_MAX_ROWS_FOR_DOT: int = 1000
    SPARK_INDEX_COL: str = "index"

class MatchingConfig:
    """
    Configuration constants for the FAISS-based distributed matching pipeline.

    These parameters govern the behavior of the ``SparkFaissExtension`` and
    related components during the fit and predict phases of nearest-neighbor
    matching. Adjusting these values allows tuning the trade-off between
    memory consumption, training accuracy, and prediction throughput.

    Attributes:
        FAISS_PERSIST_POLITIC (StorageLevel): Spark storage level used
            for persisting intermediate RDDs (e.g., the sharded index
            RDD produced during the distributed fit phase). Using
            ``MEMORY_AND_DISK`` balances speed and memory pressure.
            Defaults to ``StorageLevel.MEMORY_AND_DISK``.

        FAISS_SAMPLE_TARGET (int): Target number of rows to sample from
            the dataset when training the IVF quantizer in "sample" mode.
            A larger sample improves cluster quality but increases driver
            memory usage. If the dataset is smaller than this value, the
            entire dataset is used.
            Defaults to 5_000_000.

        FAISS_DRIVER_INDEX_LIMIT (int): Maximum number of feature vectors
            loaded onto the driver per batch during the iterative prefit
            phase ("full" training mode). This limits peak driver memory
            consumption while training clustering models (MiniBatchKMeans
            or BIRCH) incrementally.
            Defaults to 5_000_000.

        FAISS_CHUNK_SIZE (int): Number of query rows processed per batch
            on each executor during the distributed predict phase. Larger
            chunks improve throughput but increase per-executor memory
            usage. Smaller chunks reduce memory pressure at the cost of
            additional iteration overhead.
            Defaults to 4096.

        FAISS_FIT_MODE (Literal["full", "sample"]): Strategy for training
            the IVF quantizer during the distributed fit phase.
            - ``"sample"``: trains the quantizer on a random subset of
              the data (up to ``FAISS_SAMPLE_TARGET`` rows). Faster but
              may produce less accurate clusters for highly non-uniform
              distributions.
            - ``"cluster"``: trains the quantizer on the entire dataset
              using iterative mini-batch clustering (MiniBatchKMeans or
              BIRCH) via ``_prefit``. Slower but yields higher-quality
              clusters.
            - ``"full"``: exact search; each partition gets its own flat
              ``IndexFlatL2`` index (no shared quantizer). Slower than
              `"sample"`` but it has the highest accuracy according
              to the `pandas` realization.
            Defaults to ``"sample"``.

        CACHING_INDEX_MAX_SIZE (int): Maximum number of serialized FAISS
            partition indexes that can be simultaneously held in the
            per-executor LRU cache (``CachingIndex``) during the
            distributed predict phase. When the cache is full and a new
            index is requested, the least-recently-used entry is evicted
            and its memory is released. A larger value reduces redundant
            deserialization at the cost of higher per-executor memory
            consumption; a smaller value limits memory pressure but may
            increase I/O and deserialization overhead. This value is
            consumed by ``get_executor_cache()`` in
            ``hypex/extensions/faiss.py`` and passed to the
            ``CachingIndex`` constructor in ``hypex/utils/index_utils.py``.
            Defaults to 5.
    """
    FAISS_PERSIST_POLITIC: StorageLevel = StorageLevel.MEMORY_AND_DISK
    FAISS_SAMPLE_TARGET: int = 5_000_000
    FAISS_DRIVER_INDEX_LIMIT: int = 5_000_000
    FAISS_CHUNK_SIZE: int = 4096
    FAISS_FIT_MODE: Literal["sample", "cluster", "full"] = "sample"

    CACHING_INDEX_MAX_SIZE: int = 5

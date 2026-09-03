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
from __future__ import annotations

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

        FAISS_SEARCH_MODE (str): Strategy used by ``SparkFaissExtension``
            for the fit/predict phases.

            - ``"copartitioned"``: control rows are grouped by their nearest
              IVF centroid and queries are routed to the same groups, so a
              task holds exactly one cluster index and its own queries.
              Executor memory does not depend on the dataset size.
            - ``"legacy"``: one FAISS index per data partition, collected on
              the driver and distributed to every executor via ``SparkFiles``.
              Executor memory grows with the whole dataset; kept only for
              reproducing earlier results.

            Defaults to ``"copartitioned"``.

        FAISS_N_PROBES (int): Number of nearest centroids each query is
            routed to in ``"copartitioned"`` mode. Higher values raise
            recall and the shuffled volume proportionally (the query
            payload is duplicated once per probe). ``2`` reproduces the
            recall of the legacy path; ``8`` gives ~0.99 recall@1 and
            ``16`` ~0.9998 on synthetic data.
            Defaults to 8.

        FAISS_MAX_GROUP_ROWS (int): Upper bound on the number of rows of a
            single side of one co-grouped cluster. Clusters exceeding it
            are split into salted sub-groups (control) or buckets (queries),
            with the opposite side duplicated across them. Guards against
            skewed clusters producing tasks that do not fit in memory.
            Defaults to 250_000.

        FAISS_CHECKPOINT_RESULT (bool): Whether to truncate the Spark lineage
            of the matched-indexes result before handing it downstream. The
            search plan is long (cluster assignment, co-group, aggregation);
            without truncation every later action in ``Bias`` and
            ``MatchingMetrics`` re-analyzes and replays it, which dominates the
            runtime on small datasets. Requires ``sc.setCheckpointDir(...)`` to
            be fault-tolerant — otherwise ``SparkDataset.checkpoint`` falls back
            to ``local_checkpoint``, which is lost together with an executor.
            Defaults to True.

        FAISS_INDEX_CACHE_LIMIT (int): Maximum number of FAISS indexes held
            simultaneously in the executor-side ``CachingIndex`` in
            ``"legacy"`` mode. ``None`` means unbounded, which lets a worker
            accumulate every partition index in RSS.
            Defaults to 2.
    """
    FAISS_PERSIST_POLITIC: StorageLevel = StorageLevel.MEMORY_AND_DISK
    FAISS_SAMPLE_TARGET: int = 5_000_000
    FAISS_DRIVER_INDEX_LIMIT: int = 5_000_000
    FAISS_CHUNK_SIZE: int = 4096
    FAISS_SEARCH_MODE: str = "copartitioned"
    FAISS_N_PROBES: int = 8
    FAISS_MAX_GROUP_ROWS: int = 250_000
    FAISS_CHECKPOINT_RESULT: bool = True
    FAISS_INDEX_CACHE_LIMIT: int | None = 2

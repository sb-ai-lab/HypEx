from __future__ import annotations

import builtins
import gc
import math
from abc import abstractmethod
from typing import (
    Callable,
    ClassVar,
    Generator,
    Iterable,
    Literal,
)

import faiss
import numpy as np
import pandas as pd

# Spark imports
import pyspark.sql as spark
import pyspark.sql.functions as F
from pyspark import RDD, Broadcast
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import ArrayType, FloatType, LongType, StructField, StructType
from sklearn.cluster import Birch, MiniBatchKMeans

from ..config import MatchingConfig
from ..dataset import AdditionalMatchingRole, Dataset
from ..dataset.backends import PandasDataset, SparkDataset
from ..utils.errors import AbstractMethodError
from ..utils.index_utils import CachingIndex, FaissIndexStorage

# TODO: Logger
from ..utils.logger import logger
from ..utils.registry import backend_factory
from .abstract import MLExtension


class FaissExtension(MLExtension):
    """
    Master-abstract and master-backend class for FAISS-based nearest neighbor matching.

    This class provides the high-level interface for performing k-nearest neighbors
    (k-NN) search using the FAISS library within the HypEx matching pipeline.
    It defines the abstract contract that backend-specific implementations
    (e.g., Pandas, Spark) must fulfill.

    The FAISS index is built on the control (baseline) group and then queried
    with the treatment (test) group to find the closest matches. An optional
    Mahalanobis transformation matrix can be applied before indexing to account
    for feature correlations.

    Inherits from:
        MLExtension: The base class for machine learning extensions in the HypEx library.

    Attributes:
        n_neighbors (int): Number of nearest neighbors to find for each query.
        faiss_mode (Literal["base", "fast", "auto"]): Execution mode controlling
            the trade-off between accuracy and speed. "auto" selects the best
            index type based on dataset size.
        mahalanobis (Dataset | None): Optional Mahalanobis transformation matrix
            applied to features before indexing.
        index: The underlying FAISS index object (set after ``fit``).

    See Also:
        PandasFaissExtension: In-memory implementation for Pandas-backed datasets.
        SparkFaissExtension: Distributed implementation for Spark-backed datasets.
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        faiss_mode: Literal["base", "fast", "auto"] = "auto",
        mahalanobis: Dataset = None,
    ):
        """
        Initialize the FAISS extension.

        Args:
            n_neighbors (int, optional): Number of nearest neighbors to retrieve
                for each query observation. Defaults to 1.
            faiss_mode (Literal["base", "fast", "auto"], optional): Execution mode.
                - "base": Uses a standard flat index (exact search).
                - "fast": Forces an optimized approximate index (IVF).
                - "auto": Automatically selects the best index type based on
                  dataset size. Defaults to "auto".
            mahalanobis (Dataset, optional): A pre-computed Mahalanobis
                transformation matrix. If provided, features are projected into
                a decorrelated space before building the FAISS index.
                Defaults to None.
        """
        self.n_neighbors = n_neighbors
        self.faiss_mode = faiss_mode
        self.mahalanobis = mahalanobis
        self.index = None

        super().__init__()

    @staticmethod
    def _mahalanobis_transform(data: Dataset, mahalanobis: Dataset | None) -> Dataset:
        """
        Apply the Mahalanobis transformation to the input dataset.

        Projects the features into a decorrelated space using the provided
        transformation matrix. If no matrix is provided, returns the data
        unchanged.

        Args:
            data (Dataset): The input dataset to transform.
            mahalanobis (Dataset | None): The transformation matrix. If None,
                no transformation is applied.

        Returns:
            Dataset: The transformed dataset, or the original if ``mahalanobis``
                is None.
        """
        if mahalanobis is None:
            return data
        else:
            mahalanobis_index = list(mahalanobis.index)
            valid_cols = [col for col in data.columns if col in mahalanobis_index]
            if valid_cols and len(valid_cols) < len(data.columns):
                data = data[valid_cols]
            return data.dot(mahalanobis.raw_data)

    @abstractmethod
    def calc(
        self,
        data: Dataset,
        test_data: Dataset | None = None,
        mode: Literal["auto", "fit", "predict"] | None = None,
        **kwargs,
    ):
        """
        Execute the FAISS matching pipeline (fit and/or predict).

        Args:
            data (Dataset): The baseline (control) dataset used to build the index.
            test_data (Dataset | None, optional): The query (treatment) dataset
                to search against the index. Defaults to None.
            mode (Literal["auto", "fit", "predict"] | None, optional): Operation mode.
                - "auto": Fit the index and then predict.
                - "fit": Build the FAISS index only.
                - "predict": Search the index only (requires prior ``fit``).
                Defaults to None (treated as "auto").
            **kwargs: Additional keyword arguments passed to backend-specific
                implementations.

        Raises:
            AbstractMethodError: This method must be implemented by subclasses.
        """
        raise AbstractMethodError

    def fit(self, X: Dataset, Y: Dataset | None = None, **kwargs):
        """
        Build the FAISS index from the provided dataset.

        Args:
            X (Dataset): The dataset to build the index from (typically the
                control group).
            Y (Dataset | None, optional): Optional target dataset. Not typically
                used for FAISS indexing. Defaults to None.

        Returns:
            FaissExtension: The fitted extension instance with a populated index.
        """
        return super().calc(X, target_data=Y, mode="fit", **kwargs)

    def predict(self, X: Dataset, **kwargs) -> Dataset:
        """
        Search the FAISS index for the nearest neighbors of the given dataset.

        Args:
            X (Dataset): The query dataset (typically the treatment group).
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset: A dataset containing the indices of the nearest neighbors
                for each observation in ``X``, wrapped with
                ``AdditionalMatchingRole``.
        """
        return self.result_to_dataset(
            super().calc(X, mode="predict", **kwargs), AdditionalMatchingRole()
        )


@backend_factory.register(FaissExtension, PandasDataset)
class PandasFaissExtension(FaissExtension):
    """
    Pandas backend implementation for FAISS-based nearest neighbor matching.

    Performs in-memory k-NN search using FAISS flat or IVF indexes. Suitable
    for datasets that fit entirely in the driver's memory.

    Inherits from:
        FaissExtension: The master-abstract FAISS extension class.

    Note:
        This implementation loads the entire dataset into memory, so it should
        only be used when the dataset size is manageable on a single machine.
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        faiss_mode: Literal["base", "fast", "auto"] = "auto",
        mahalanobis: Dataset = None,
    ):
        """
        Initialize the Pandas FAISS extension.

        Args:
            n_neighbors (int, optional): Number of nearest neighbors to retrieve.
                Defaults to 1.
            faiss_mode (Literal["base", "fast", "auto"], optional): Execution mode.
                - "base": Uses a standard flat index (exact search).
                - "fast": Forces an IVF index for approximate search.
                - "auto": Automatically selects based on dataset size.
                Defaults to "auto".
            mahalanobis (Dataset, optional): Mahalanobis transformation matrix.
                Defaults to None.
        """
        super().__init__(n_neighbors, faiss_mode, mahalanobis)

    @staticmethod
    def _prepare_indexes(index: np.ndarray, dist: np.ndarray, k: int) -> np.ndarray:
        """
        Prepare and deduplicate nearest neighbor indices based on distances.

        For each query, sorts the candidate neighbors by distance and selects
        the top-k unique indices. This handles the case where multiple candidates
        have the same distance.

        Args:
            index (np.ndarray): Array of candidate neighbor indices, shape (n_queries, n_candidates).
            dist (np.ndarray): Array of corresponding distances, same shape as ``index``.
            k (int): Number of unique neighbors to select per query.

        Returns:
            np.ndarray: Array of shape (n_queries, k) containing the deduplicated
                neighbor indices.
        """
        new = np.vstack(
            [
                np.concatenate(
                    [val[np.where(dist[i] == d)[0]] for d in sorted(set(dist[i]))[:k]]
                )
                for i, val in enumerate(index)
            ]
        )
        return new

    def _predict(
        self,
        data: Dataset,
        test_data: Dataset,
        X: np.ndarray,
    ) -> Dataset:
        """
        Perform the FAISS search on the query vectors.

        Searches the built FAISS index for the ``n_neighbors`` closest matches
        to each query vector in ``X``. Handles the special case of ``n_neighbors=1``
        by resolving ties among equidistant candidates.

        Args:
            data (Dataset): The baseline dataset (used for index resolution).
            test_data (Dataset): The query dataset.
            X (np.ndarray): Query vectors of shape (n_queries, n_features).

        Returns:
            Dataset: A dataset containing the matched indices,
                wrapped as a Dataset via ``result_to_dataset``.
        """
        dist, indexes = self.index.search(X, k=self.n_neighbors)
        if self.n_neighbors == 1:
            equal_dist = list(map(lambda x: np.where(x == x[0])[0], dist))
            indexes = [
                (
                    int(index[dist][0])
                    if abs(index[dist][0]) <= len(data) + len(test_data)
                    else -1
                )
                for index, dist in zip(indexes, equal_dist)
            ]
        else:
            indexes = self._prepare_indexes(indexes, dist, self.n_neighbors)
        result = self.result_to_dataset(result=indexes, roles={}).set_index(
            test_data.index, drop=False
        )
        result.index.name = None

        return result

    def _fit(
        self,
        data: Dataset,
        test_data: Dataset,
    ) -> None:
        """
        Build the FAISS index from the baseline dataset.

        Extracts feature vectors from ``data``, optionally applies the Mahalanobis
        transformation, and builds either a flat or IVF index depending on the
        dataset size and ``faiss_mode``.

        For large datasets (>1M rows) in "auto" mode, or when "fast" mode is
        explicitly requested, an IVF (Inverted File) index is trained and used.
        Otherwise, a flat L2 index is used for exact search.

        Args:
            data (Dataset): The baseline dataset to index.
            test_data (Dataset): The query dataset (used for size heuristics).
        """
        X = self._mahalanobis_transform(data, self.mahalanobis).raw_data.values
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(X.shape[1]))
        if (
            (
                (len(X) > 1_000_000 and self.faiss_mode == "auto")
                or self.faiss_mode == "fast"
            )
            and len(X) > 1_000
            and len(test_data) > 1_000
        ):
            m = 4  # heuristic
            n_clusters = int(np.sqrt(len(X) / m))
            nlist = min(n_clusters, max(1, X.shape[0] // 39))
            quantizer = faiss.IndexFlatL2(X.shape[1])
            _index = faiss.IndexIVFFlat(quantizer, X.shape[1], nlist)
            _index.train(X)
            self.index = faiss.IndexIDMap(_index)
        self.index.add_with_ids(X, np.array(data.index.tolist(), dtype=np.int64))

    def calc(
        self,
        data: Dataset,
        test_data: Dataset | None = None,
        mode: Literal["auto", "fit", "predict"] | None = None,
        **kwargs,
    ):
        """
        Execute the FAISS matching pipeline for Pandas-backed datasets.

        Orchestrates the fit and predict phases based on the ``mode`` argument.

        Args:
            data (Dataset): The baseline (control) dataset.
            test_data (Dataset | None, optional): The query (treatment) dataset.
                Required for "predict" and "auto" modes. Defaults to None.
            mode (Literal["auto", "fit", "predict"] | None, optional): Operation mode.
                - "auto": Fit the index and then predict.
                - "fit": Build the FAISS index only.
                - "predict": Search the index only (requires prior ``fit``).
                Defaults to None (treated as "auto").
            **kwargs: Additional keyword arguments.

        Returns:
            PandasFaissExtension or Dataset: The fitted extension (for "fit" mode)
                or the matched indices (for "predict"/"auto" modes).

        Raises:
            ValueError: If ``test_data`` is None when prediction is required,
                or if the index has not been built before prediction.
        """
        mode = mode or "auto"
        if mode in ["auto", "fit"]:
            self._fit(data, test_data)
        if mode in ["auto", "predict"]:
            if test_data is None:
                raise ValueError("test_data is needed for evaluation")
            if self.index is None:
                raise ValueError(
                    "index is not created yet. Call 'fit' before 'predict'."
                )

            X = (
                self._mahalanobis_transform(test_data, self.mahalanobis).raw_data.values
                if mode == "auto"
                else self._mahalanobis_transform(data, self.mahalanobis).raw_data.values
            )
            return self._predict(data, test_data, X)
        return self


# ===========================================================================
# Global functions for PySpark partition logic
# ===========================================================================

# ---------------------------------------------------------------------------
# FUNCTIONS IN FIT METHOD
# ---------------------------------------------------------------------------


def _spark_partition_fit(
    iterator: Iterable,
    bc_index: Broadcast,
    bc_storage: Broadcast,
) -> Generator[bytes, None, None]:
    """
    Build a local FAISS index on each Spark partition.

    Receives a pre-trained IVF quantizer via broadcast, adds the partition's
    vectors to a local ``IndexIDMap`` wrapper, and yields the serialized
    index. Each partition produces one serialized index file that is later
    used during the distributed predict phase.

    Args:
        iterator (Iterable): Iterator over partition rows. Each row must
            contain ``index`` (long) and ``_features`` (vector) columns.
        bc_index (Broadcast): Broadcasted pre-trained FAISS index (quantizer).
        bc_storage (Broadcast): Broadcasted FaissIndexStorage instance for
            index serialization.

    Yields:
        bytes: Serialized FAISS index for the partition, produced by
            ``faiss.serialize_index``.
    """
    import faiss
    import numpy as np

    index = bc_index.value
    storage = bc_storage.value
    ids, vectors = [], []
    for row in iterator:
        ids.append(row["index"])
        vectors.append(list(row["_features"]))

    if not ids:
        return  # for empty partition

    ids = np.array(ids, dtype=np.int64)
    vectors = np.array(vectors, dtype=np.float32)

    index_copy = faiss.clone_index(index)
    index_with_ids = faiss.IndexIDMap(index_copy)
    index_with_ids.add_with_ids(vectors, ids)

    yield storage.save_index(index_with_ids)


def _spark_full_partition_fit(
    iterator: Iterable,
    bc_storage: Broadcast,
) -> Generator[bytes, None, None]:
    """
    Build a local flat FAISS index on each Spark partition.

    Creates a standalone ``IndexFlatL2`` for each partition without a shared
    quantizer. Used in "full" mode for exact search across partitions.

    Args:
        iterator (Iterable): Iterator over partition rows. Each row must
            contain ``index`` (long) and ``_features`` (vector) columns.
        bc_storage (Broadcast): Broadcasted FaissIndexStorage instance for
            index serialization.

    Yields:
        bytes: Serialized FAISS index for the partition.
    """
    import faiss
    import numpy as np

    storage = bc_storage.value
    ids, vectors = [], []
    for row in iterator:
        ids.append(row["index"])
        vectors.append(list(row["_features"]))

    if not ids:
        return  # for empty partition

    ids = np.array(ids, dtype=np.int64)
    vectors = np.array(vectors, dtype=np.float32)

    d = vectors.shape[1]

    quantizer = faiss.IndexFlatL2(d)
    index_with_ids = faiss.IndexIDMap(quantizer)
    index_with_ids.add_with_ids(vectors, ids)

    yield storage.save_index(index_with_ids)


# ---------------------------------------------------------------------------
# FUNCTIONS IN PREDICT METHODS
# ---------------------------------------------------------------------------


def _per_partition_predict(
    shard_iter: Iterable,
    bc_n_neighbors: Broadcast,
    bc_references: Broadcast,
    bc_chunk_size: Broadcast,
    bc_storage: Broadcast,
) -> Generator[tuple, None, None]:
    """
    Perform distributed nearest-neighbor search on each Spark partition.

    For each chunk of query vectors in the partition, iteratively loads
    serialized FAISS indexes from the driver-distributed files, searches
    for the top-k nearest neighbors, and aggregates candidates across all
    partition indexes. The final top-``n_neighbors`` results are yielded
    as ``(query_id, [neighbor_ids])`` tuples.

    Args:
        shard_iter (Iterable): Iterator over partition rows. Each row must
            contain ``index`` (long) and ``_features`` (vector) columns.
        bc_n_neighbors (Broadcast): Number of nearest neighbors to return.
        bc_references (Broadcast): List of serialized index file names
            distributed via ``SparkFiles``.
        bc_chunk_size (Broadcast): Number of query rows to process per batch.
        bc_storage (Broadcast): Broadcasted FaissIndexStorage instance.

    Yields:
        tuple: ``(int(query_id), list[int(neighbor_ids)])`` for each query
            vector in the partition.
    """
    import numpy as np

    cache = get_executor_cache()

    real_n = bc_n_neighbors.value
    references = bc_references.value
    chunk_size = bc_chunk_size.value
    storage = bc_storage.value

    def iter_chunk(it: Iterable, chunk_size: int) -> Generator[list, None, None]:
        """Helper generator to yield chunks of rows from iterator."""
        chunk = []
        amount = 0
        for row in it:
            chunk.append(row)
            amount += 1

            if amount >= chunk_size:
                amount = 0
                yield chunk
                chunk = []

        if chunk:
            yield chunk

    for chunk in iter_chunk(shard_iter, chunk_size):
        if not chunk:
            return
        query_ids = np.array([r["index"] for r in chunk], dtype=np.int64)
        batch = np.array(
            [r["_features"].toArray() for r in chunk], dtype=np.float32
        )  # (Q, d)
        del chunk
        # gc.collect() # TODO: detect time decr when gc.collect disabled

        candidates = [[] for _ in range(len(query_ids))]
        for ref in references:
            tmp_index = cache.get(ref, storage)
            k = min(real_n, tmp_index.ntotal)
            dists, nids = tmp_index.search(batch, k)  # (Q, k)
            del tmp_index
            gc.collect()  # TODO: detect time decr when gc.collect disabled

            for q_idx in range(len(query_ids)):
                for rank in range(k):
                    nid = int(nids[q_idx, rank])
                    if nid >= 0:
                        candidates[q_idx].append((float(dists[q_idx, rank]), nid))

        for q_idx, qid in enumerate(query_ids):
            top = sorted(candidates[q_idx], key=lambda x: x[0])[:real_n]
            output = [int(nid) for _, nid in top]
            yield (int(qid), output)


def _crossover_search(
    control_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bc_n_neighbors: Broadcast,
) -> pd.DataFrame:
    """
    Perform exact nearest neighbor search between control and test dataframes.

    Used in "shuffle" mode for cluster pair matching. Builds a flat FAISS index
    on the control data and searches for nearest neighbors from the test data.

    Args:
        control_df (pd.DataFrame): DataFrame containing control group vectors
            with ``_features`` column.
        test_df (pd.DataFrame): DataFrame containing test group vectors
            with ``_features`` column.
        bc_n_neighbors (Broadcast): Number of nearest neighbors to find.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - ``index``: Original test group index (repeated for each neighbor).
            - ``dists``: Distance to each neighbor.
            - ``nids``: Index of each neighbor in the control group.
    """
    import faiss
    import numpy as np

    n_neighbors = bc_n_neighbors.value
    control_data = np.ascontiguousarray(
        np.vstack(control_df["_features"].to_numpy())
    ).astype(np.float32)
    test_data = np.ascontiguousarray(np.vstack(test_df["_features"].to_numpy())).astype(
        np.float32
    )

    quantizer = faiss.IndexFlatL2(control_data.shape[1])
    quantizer.add(control_data)

    k = min(n_neighbors, control_data.shape[0])
    dists, nids = quantizer.search(test_data, k)

    return pd.DataFrame(
        {
            "index": np.repeat(test_df["index"].to_numpy(), repeats=k),
            "dists": dists.ravel(),
            "nids": control_df["index"].to_numpy()[nids.ravel()],
        }
    )


@logger.log_methods(log_args=False, log_result=False, private=True)
@backend_factory.register(FaissExtension, SparkDataset)
class SparkFaissExtension(FaissExtension):
    """
    Spark backend implementation for distributed FAISS-based nearest neighbor matching.

    This class implements a fully distributed FAISS pipeline for datasets that
    exceed single-machine memory. The pipeline consists of three phases:

    1. **Vectorization**: Feature columns are assembled into a single vector
       column using Spark's ``VectorAssembler``.
    2. **Distributed Fit**: The dataset is partitioned, and each partition
       builds a local FAISS ``IndexIDMap`` on top of a shared IVF quantizer.
       The quantizer is trained either on a random sample ("sample" mode),
       via iterative mini-batch clustering on the full dataset ("cluster" mode),
       or each partition gets its own flat index ("full" mode).
    3. **Distributed Predict**: Serialized partition indexes are distributed
       to all executors via ``SparkFiles``. Each executor loads the indexes
       into a local LRU cache and performs batched nearest-neighbor searches.

    Inherits from:
        FaissExtension: The master-abstract FAISS extension class.

    Attributes:
        PREDICT_SCHEMA (StructType): Schema for prediction results DataFrame.
        CLUSTERING_METHODS_MAPPER (ClassVar[dict]): Mapping of clustering
            method names to their corresponding model classes and parameters.
        _FIT_DISPATCH (ClassVar[dict]): Mapping of fit mode names to their
            corresponding handler method names.

    See Also:
        CachingIndex: Executor-side LRU cache for FAISS indexes.
        FaissIndexStorage: Storage utility for FAISS indexes.
    """

    # Storage level for intermediate RDDs
    PREDICT_SCHEMA = StructType(
        [
            StructField("index", LongType(), False),
            StructField("index_list", ArrayType(LongType()), False),
        ]
    )

    CLUSTERING_METHODS_MAPPER: ClassVar[dict] = {
        "k-means": {
            "model": MiniBatchKMeans,
            "params": {
                "random_state": 21,
                "max_no_improvement": None,
                "batch_size": 5_000_000,
                "n_init": 5,
            },
        },
        "birch": {
            "model": Birch,
            "params": {"n_clusters": None},
        },
    }

    _FIT_DISPATCH: ClassVar[dict[str, str]] = {
        "sample": "_fit_sample",
        "cluster": "_fit_cluster",
        "full": "_fit_full",
        "shuffle": "_fit_shuffle",
    }

    def __init__(
        self,
        n_neighbors: int = 1,
        faiss_mode: Literal["base", "fast", "auto"] = "auto",
        mahalanobis: Dataset = None,
    ):
        """
        Initialize the Spark FAISS extension.

        Args:
            n_neighbors (int, optional): Number of nearest neighbors to retrieve.
                Defaults to 1.
            faiss_mode (Literal["base", "fast", "auto"], optional): Execution mode.
                - "base": Uses a standard flat index (exact search).
                - "fast": Forces an IVF index for approximate search.
                - "auto": Automatically selects based on dataset size.
                Defaults to "auto".
            mahalanobis (Dataset, optional): Mahalanobis transformation matrix.
                Defaults to None.
        """
        super().__init__(n_neighbors, faiss_mode, mahalanobis)
        self.seed: int = 21
        self.storage: FaissIndexStorage | None = None
        self._data_size: int | None = None

        self._fitted: bool = False
        self.new_execution_flag: bool = False
        self._sharded_rdd: RDD | None = None
        self._clustered_control: spark.DataFrame | None = None
        self._clustered_test: spark.DataFrame | None = None
        self._centroids: np.ndarray | None = None
        self._control_clusters_dict: dict[int, int] = {}
        self._test_clusters_dict: dict[int, int] = {}

    def _vectorize_data(
        self,
        data: spark.DataFrame,
    ) -> spark.DataFrame:
        """
        Assemble feature columns into a single vector column for FAISS.

        Uses Spark's ``VectorAssembler`` to combine all numeric feature columns
        (everything except the ``index`` column) into a single ``_features``
        vector column.

        Args:
            data (spark.DataFrame): Input Spark DataFrame containing numeric
                features and an ``index`` column.

        Returns:
            spark.DataFrame: DataFrame with an additional ``_features`` column
                of type ``pyspark.ml.linalg.Vector``.

        Raises:
            TypeError: If any feature column has a string/categorical type.
                All categorical features must be encoded before calling this method.
        """
        self.feature_cols = list(set(data.columns) - {"index"})
        if (
            len(
                set(map(lambda x: x[1], data.dtypes)).intersection(
                    ["varchar", "string"]
                )
            )
            > 0
        ):
            raise TypeError("Unencoded categorical features are not allowed!")

        if MatchingConfig.FAISS_FIT_MODE == "shuffle":
            return data.select(
                F.col("index"),
                F.array(
                    *[F.col(col).cast(FloatType()) for col in self.feature_cols]
                ).alias("_features"),
            )
        else:
            vecAssembler = VectorAssembler(
                inputCols=self.feature_cols,
                outputCol="_features",
                handleInvalid="keep",
            )

            return vecAssembler.transform(data)

    # ================================================================
    # FIT PHASE
    # ================================================================

    def _fit(
        self,
        vectorized_data: spark.DataFrame,
        model_name: str | None,
    ) -> SparkFaissExtension:
        """
        Build distributed FAISS indexes across Spark partitions.

        Supported training modes:
        - **"sample"**: Trains the IVF quantizer on a random sample of the data
            (up to ``FAISS_SAMPLE_TARGET`` rows). Faster but may produce less
            accurate clusters for non-uniform distributions.
        - **"cluster"**: Trains the IVF quantizer on the entire dataset using
            iterative mini-batch clustering via ``_prefit``. Slower but more
            accurate.
        - **"full"**: Exact search; each partition gets its own flat
            ``IndexFlatL2`` index (no shared quantizer).
        - **"shuffle"**: Uses cluster-based partitioning for efficient
            nearest neighbor search.

        After training, each partition builds a local ``IndexIDMap`` on top of
        the shared quantizer, and the serialized indexes are persisted as an RDD.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the
                ``_features`` vector column.
            model_name (str | None): Clustering model name for "cluster" and
                "shuffle" modes (e.g., "k-means", "birch"). Ignored in "sample"
                and "full" modes. Defaults to None.

        Returns:
            SparkFaissExtension: Self, for method chaining.

        Raises:
            ValueError: If the fit mode in MatchingConfig is not supported.
        """
        mode = MatchingConfig.FAISS_FIT_MODE
        handler_name = self._FIT_DISPATCH.get(mode)
        if handler_name is None:
            raise ValueError(f"Incorrect faiss fit mode: '{mode}'")

        session = vectorized_data.sparkSession
        self.storage = FaissIndexStorage(session)
        self._data_size = self._data_size or vectorized_data.count()
        self._compute_cluster_params()

        handler = getattr(self, handler_name)
        handler(vectorized_data, model_name)

        self._fitted = True
        return self

    # Clusterization parameters

    def _compute_cluster_params(self) -> None:
        """
        Compute clustering parameters for IVF index.

        Calculates the number of clusters (k) and the number of probes (nprobe)
        based on the dataset size. Uses heuristic values for the factor and
        bounds for nprobe.

        Note:
            This method should be called after ``_data_size`` is set.
        """
        # TODO: insert into config `m` and 10 and 50 params
        # factor = MatchingConfig.FAISS_CLUSTER_FACTOR
        factor = 4  # heuristic
        self.k = int(np.sqrt(self._data_size / factor))
        self._nprobe = max(
            self.n_neighbors,
            min(
                # max(self.k // 10, MatchingConfig.FAISS_MIN_NPROBE),
                # MatchingConfig.FAISS_MAX_NPROBE,
                max(self.k // 10, 10),
                50,
            ),
        )

    # Formatting clusters for "cluster" mode

    def _prefit(self, vectorized_data: spark.DataFrame, model_name: str) -> None:
        """
        Train the IVF quantizer on the full dataset via iterative partition upload.

        Loads feature vectors from Spark partitions in batches of
        ``FAISS_DRIVER_INDEX_LIMIT`` rows and incrementally fits a clustering
        model (MiniBatchKMeans or BIRCH) on the driver. The resulting cluster
        centers are used to initialize the FAISS IVF quantizer.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the
                ``_features`` vector column.
            model_name (str): Name of the clustering algorithm to use.
                Must be a key in ``CLUSTERING_METHODS_MAPPER`` (e.g., "k-means", "birch").
        """

        def _partition_load(
            partition_iter: Iterable, batch_size: int
        ) -> Generator[list, None, None]:
            """
            Load batches of feature vectors from a Spark partition iterator.

            Reads rows from the partition in chunks of ``batch_size`` and yields
            each batch as a list of feature vectors. Used during the iterative
            prefit phase to train clustering models on the driver without loading
            the entire dataset at once.

            Args:
                partition_iter (Iterable): Iterator over partition rows. Each row
                    is expected to have a ``_features`` column containing the
                    feature vector.
                batch_size (int): Number of rows to accumulate per batch.

            Yields:
                list: A batch of feature vectors (each element is a list of floats).
            """
            batch = []
            for row in partition_iter:
                batch.append(list(row["_features"]))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        model_dict = self.CLUSTERING_METHODS_MAPPER[model_name]
        model_cls, model_params = model_dict["model"], model_dict["params"]
        model_params["n_clusters"] = self.k
        model = model_cls(**model_params)

        batch_size = MatchingConfig.FAISS_DRIVER_INDEX_LIMIT
        np_batch = None

        for batch in (
            vectorized_data.select("_features")
            .rdd.mapPartitions(lambda it: _partition_load(it, batch_size))
            .toLocalIterator()
        ):
            np_batch = np.array(batch, dtype=np.float32)
            model.partial_fit(np_batch)

        if np_batch is not None:
            del np_batch
            gc.collect()

        centroids = (
            model.cluster_centers_
            if model_name == "k-means"
            else model.subcluster_centers_
        )
        centroids = centroids.astype(np.float32)
        index_shape = centroids.shape[1]
        nlist = len(centroids)

        quantizer = faiss.IndexFlatL2(index_shape)
        quantizer.add(centroids)
        self.index = faiss.IndexIVFFlat(quantizer, index_shape, nlist)
        self.index.is_trained = True

        self._centroids = centroids

        self._clustering_model = model

    # Assign train data to cluster in "shuffle" mode

    def _shuffle_fit(self, data: spark.DataFrame, model_name: str) -> None:
        """
        Perform clustering and prepare data for shuffle mode.

        Uses the pre-trained clustering model to assign each control group
        observation to a cluster, then computes the cluster size distribution
        for later use in the predict phase.

        Args:
            data (spark.DataFrame): Input DataFrame with ``index`` and ``_features``
                columns.
            model_name (str): Name of the clustering algorithm used for training.
        """
        self._prefit(vectorized_data=data, model_name=model_name)
        session = data.sparkSession
        bc_clusters = session.sparkContext.broadcast(self._centroids)

        def _set_clusters(
            it: Iterable[pd.DataFrame],
        ) -> Generator[pd.DataFrame, None, None]:
            import faiss
            import numpy as np

            clusters = np.array(bc_clusters.value, dtype=np.float32)
            cluster_index = faiss.IndexFlatL2(clusters.shape[1])
            cluster_index.add(clusters)

            for pdf in it:
                # Ensure features are contiguous in memory
                # vstack does not guarantee this
                features = np.ascontiguousarray(np.vstack(pdf["_features"].to_numpy()))
                _, row_cluster = cluster_index.search(features, 1)

                output = pdf.loc[:, ["index", "_features"]].copy()
                output["_cluster"] = row_cluster[:, 0].astype(np.int32)

                yield output

        schema = StructType(
            [
                StructField("index", LongType(), False),
                StructField("_features", ArrayType(FloatType()), False),
                StructField("_cluster", LongType(), False),
            ]
        )

        self._clustered_control = data.mapInPandas(
            _set_clusters, schema=schema
        ).persist(MatchingConfig.FAISS_PERSIST_POLITIC)
        clusters_info = self._clustered_control.groupBy("_cluster").count().collect()
        self._control_clusters_dict = {
            c["_cluster"]: max(math.ceil(c["count"] / MatchingConfig.BUCKET_SIZE), 1)
            for c in clusters_info
        }

    # Fit in "sample" mode

    def _fit_sample(
        self,
        vectorized_data: spark.DataFrame,
        model_name: str | None,
    ) -> None:
        """
        Build indexes using sample-based IVF training.

        Trains the IVF quantizer on a random sample of the data, then builds
        partition indexes using the trained quantizer.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the
                ``_features`` vector column.
            model_name (str | None): Not used in sample mode. Defaults to None.
        """
        X = self._collect_training_sample(vectorized_data)
        self._train_ivf_on_array(X)
        self._build_and_persist_sharded_rdd(
            vectorized_data,
            partition_func=_spark_partition_fit,
            broadcast_index=True,
        )
        self.new_execution_flag = False

    def _collect_training_sample(self, vectorized_data: spark.DataFrame) -> np.ndarray:
        """
        Collect a random sample of training data for IVF quantizer training.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the
                ``_features`` vector column.

        Returns:
            np.ndarray: Array of feature vectors with shape (n_samples, n_features)
                and dtype float32.
        """
        frac = min(MatchingConfig.FAISS_SAMPLE_TARGET / max(self._data_size, 1), 1.0)
        sample_rows = (
            vectorized_data.sample(fraction=frac, seed=self.seed)
            .select("_features")
            .collect()
        )
        return np.array(
            [list(row["_features"]) for row in sample_rows],
            dtype=np.float32,
        )

    def _train_ivf_on_array(self, X: np.ndarray) -> None:
        """
        Train an IVF index on the provided array of feature vectors.

        Args:
            X (np.ndarray): Training data with shape (n_samples, n_features).
        """
        d = X.shape[1]
        # IVF Faiss supports up to 39 * (training points) per cluster
        nlist = min(self.k, max(1, X.shape[0] // 39))

        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
        self.index.train(X)
        self.index.nprobe = self._nprobe

    # Fit in "cluster" mode

    def _fit_cluster(
        self,
        vectorized_data: spark.DataFrame,
        model_name: str | None,
    ) -> None:
        """
        Build indexes using full-dataset clustering for IVF training.

        Uses iterative mini-batch clustering on the entire dataset to train
        the IVF quantizer, then builds partition indexes.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the
                ``_features`` vector column.
            model_name (str | None): Clustering model name (e.g., "k-means", "birch").
                Defaults to None (uses "k-means").
        """
        self._prefit(vectorized_data=vectorized_data, model_name=model_name)
        self.index.nprobe = self._nprobe
        self._build_and_persist_sharded_rdd(
            vectorized_data,
            partition_func=_spark_partition_fit,
            broadcast_index=True,
        )
        self.new_execution_flag = False

    # Fit in "full" mode

    def _fit_full(
        self,
        vectorized_data: spark.DataFrame,
        model_name: str | None,
    ) -> None:
        """
        Build flat indexes for each partition (exact search).

        Each partition gets its own ``IndexFlatL2`` without a shared quantizer.
        This provides exact search results but may be slower for large datasets.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the
                ``_features`` vector column.
            model_name (str | None): Not used in full mode. Defaults to None.
        """
        self.index = None
        self._build_and_persist_sharded_rdd(
            vectorized_data,
            partition_func=_spark_full_partition_fit,
            broadcast_index=False,
        )
        self.new_execution_flag = False

    # Fit in "shuffle" mode

    def _fit_shuffle(
        self,
        vectorized_data: spark.DataFrame,
        model_name: str | None,
    ) -> None:
        """
        Build indexes using cluster-based partitioning.

        Uses the shuffle-based approach for efficient nearest neighbor search
        by partitioning data based on cluster assignments.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the
                ``_features`` vector column.
            model_name (str | None): Clustering model name (e.g., "k-means", "birch").
                Defaults to None (uses "k-means").
        """
        features = ["index", "_features"]
        self._shuffle_fit(vectorized_data.select(*features), model_name)
        self.new_execution_flag = True

    # General RDD constructor

    def _build_and_persist_sharded_rdd(
        self,
        vectorized_data: spark.DataFrame,
        partition_func: Callable,
        broadcast_index: bool,
    ) -> None:
        """
        Build and persist an RDD of serialized FAISS partition indexes.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the
                ``_features`` vector column.
            partition_func (Callable): Function to apply to each partition
                to build the index.
            broadcast_index (bool): Whether to broadcast the trained index
                to all partitions.
        """
        session = vectorized_data.sparkSession
        bc_storage = session.sparkContext.broadcast(self.storage)
        features = ["index", "_features"]

        rdd = vectorized_data.select(*features).rdd

        if broadcast_index:
            bc_index = session.sparkContext.broadcast(self.index)
            del self.index
            self.index = None
            gc.collect()

            self._sharded_rdd = rdd.mapPartitions(
                lambda it: partition_func(it, bc_index, bc_storage)
            ).persist(MatchingConfig.FAISS_PERSIST_POLITIC)
        else:
            self._sharded_rdd = rdd.mapPartitions(
                lambda it: partition_func(it, bc_storage)
            ).persist(MatchingConfig.FAISS_PERSIST_POLITIC)
        self._sharded_rdd.count()

    # ==============================================================================
    # PREDICT PHASE
    # ==============================================================================

    def _predict(
        self,
        test_data: spark.DataFrame,
        storage_level: Literal["MEMORY_ONLY", "MEMORY_AND_DISK", "DISK_ONLY"] | None,
    ) -> Dataset:
        """
        Perform distributed nearest-neighbor search across Spark partitions.

        The prediction pipeline consists of the following steps:

        1. Deserialize partition indexes and save them as ``.index`` files.
        2. Distribute the ``.index`` files to all executors via ``SparkFiles``.
        3. On each executor, iteratively load batches of query vectors and
           search against all partition indexes, using the ``CachingIndex``
           to avoid redundant deserialization.
        4. Collect the top-``n_neighbors`` results and wrap them in a
           Spark DataFrame with the ``PREDICT_SCHEMA`` schema.
        5. Clean up temporary files after materialization.

        Args:
            test_data (spark.DataFrame): Input DataFrame with the ``_features``
                vector column containing query vectors.
            storage_level (Literal | None): Storage strategy for cached results.
                Options: "MEMORY_ONLY", "MEMORY_AND_DISK", "DISK_ONLY".
                Defaults to None (uses "MEMORY_AND_DISK").

        Returns:
            Dataset: A Dataset containing the matched neighbor indices, indexed
                by the original row index.
        """
        storage_level = storage_level or "MEMORY_AND_DISK"

        if self.new_execution_flag:
            result, broadcasts = self._predict_shuffle(test_data)
        else:
            result, broadcasts = self._predict_transmission(test_data)

        return self._persist_and_finalize(result, storage_level, broadcasts)

    # Predict finalizer

    def _persist_and_finalize(
        self,
        result: Dataset,
        storage_level: str,
        broadcasts_to_destroy: list[Broadcast] | None = None,
    ) -> Dataset:
        """
        Persist the result dataset and clean up broadcast variables.

        Args:
            result (Dataset): The result dataset to persist.
            storage_level (str): Storage level for persisting the result.
            broadcasts_to_destroy (list[Broadcast] | None): List of broadcast
                variables to destroy after persisting. Defaults to None.

        Returns:
            Dataset: The persisted result dataset.
        """
        result = result.set_index("index")
        result.index.name = None
        result.persist(storage_level=storage_level, action="count")
        result.checkpoint(eager=True)

        if broadcasts_to_destroy:
            for bc in broadcasts_to_destroy:
                try:
                    bc.destroy(blocking=True)
                except Exception:
                    pass  # ignore if broadcast is already cleaned
        return result

    # Predict in "shuffle" mode

    def _predict_shuffle(
        self, test_data: spark.DataFrame
    ) -> tuple[Dataset, list[Broadcast]]:
        """
        Perform prediction using shuffle mode.

        Args:
            test_data (spark.DataFrame): Input DataFrame with query vectors.

        Returns:
            tuple: (result Dataset, list of Broadcast variables to clean up)
        """
        features = ["index", "_features"]
        data = test_data.select(*features)
        self._test_group_clustering(data)

        session = test_data.sparkSession
        control, test = self._build_shuffled_frames(session)
        neighbors_pairs, bc_n_neighbors = self._search_cluster_pairs(control, test)
        result = self._aggregate_neighbors(neighbors_pairs)
        return result, [bc_n_neighbors]

    def _build_shuffled_frames(
        self,
        session: spark.SparkSession,
    ) -> tuple[spark.DataFrame, spark.DataFrame]:
        """
        Build shuffled control and test DataFrames for cluster pair matching.

        Creates DataFrames where each row is assigned to multiple bucket
        combinations to ensure comprehensive cluster pair comparisons.

        Args:
            session (spark.SparkSession): Active Spark session.

        Returns:
            tuple: (control DataFrame, test DataFrame) with bucket assignments.
        """
        clusters_frame = F.broadcast(
            self._create_clusters_frame(
                self._test_clusters_dict,
                self._control_clusters_dict,
                session,
            )
        )
        control = (
            self._clustered_control.join(clusters_frame, on="_cluster")
            .withColumn(
                "c_buckets", F.pmod(F.hash(F.col("index")), F.col("c_buckets"))
            )  # we use `hash` to guarantee uniform distribution into buckets
            .withColumn(
                "t_buckets",
                F.explode(F.sequence(F.lit(0), F.col("t_buckets") - F.lit(1))),
            )
            .select("_cluster", "c_buckets", "t_buckets", "index", "_features")
        )

        test = (
            self._clustered_test.withColumn(
                "_cluster", F.explode(F.col("_cluster")).alias("_cluster")
            )
            .join(clusters_frame, on="_cluster")
            .withColumn("t_buckets", F.pmod(F.hash(F.col("index")), F.col("t_buckets")))
            .withColumn(
                "c_buckets",
                F.explode(F.sequence(F.lit(0), F.col("c_buckets") - F.lit(1))),
            )
            .select("_cluster", "c_buckets", "t_buckets", "index", "_features")
        )

        return control, test

    @staticmethod
    def _create_clusters_frame(
        test_dict: dict,
        control_dict: dict,
        session: spark.SparkSession,
    ) -> spark.DataFrame:
        """
        Create a DataFrame mapping cluster IDs to bucket counts.

        Args:
            test_dict (dict): Mapping of cluster IDs to test bucket counts.
            control_dict (dict): Mapping of cluster IDs to control bucket counts.
            session (spark.SparkSession): Active Spark session.

        Returns:
            spark.DataFrame: DataFrame with columns ``_cluster``, ``c_buckets``,
                ``t_buckets``.
        """
        columns = ["_cluster", "c_buckets", "t_buckets"]
        rows = [
            (cluster, c_bucket, test_dict.get(cluster, 1))
            for cluster, c_bucket in sorted(control_dict.items())
        ]
        return session.createDataFrame(rows, columns)

    def _test_group_clustering(self, data: spark.DataFrame) -> None:
        """
        Assign test group observations to clusters.

        For each test observation, find the nearest clusters using the
        pre-trained clustering model.

        Args:
            data (spark.DataFrame): Input DataFrame with ``index`` and ``_features``
                columns.
        """
        sc = data.sparkSession.sparkContext
        bc_clusters = sc.broadcast(self._centroids)
        # TODO
        bc_clusters_search = sc.broadcast(
            min(
                (
                    MatchingConfig.FAISS_N_PROBES
                    if MatchingConfig.FAISS_N_PROBES > 0
                    else self.n_neighbors
                ),
                len(self._centroids),
            )
        )

        def _partition_search(
            it: Iterable[pd.DataFrame],
        ) -> Generator[pd.DataFrame, None, None]:
            import faiss
            import numpy as np

            clusters_search = bc_clusters_search.value

            clusters = np.array(bc_clusters.value, dtype=np.float32)
            quantizer = faiss.IndexFlatL2(clusters.shape[1])
            quantizer.add(clusters)

            for pdf in it:
                features = np.ascontiguousarray(np.vstack(pdf["_features"].to_numpy()))
                _, c_ids = quantizer.search(features, clusters_search)

                output = pdf.loc[:, ["index", "_features"]]
                # `list` for correct `spark` response
                output["_cluster"] = c_ids.tolist()

                yield output

        schema = StructType(
            [
                StructField("index", LongType(), False),
                StructField("_features", ArrayType(FloatType()), False),
                StructField("_cluster", ArrayType(LongType()), False),
            ]
        )

        self._clustered_test = data.mapInPandas(
            _partition_search, schema=schema
        ).persist(MatchingConfig.FAISS_PERSIST_POLITIC)
        test_info = (
            self._clustered_test.select(F.explode(F.col("_cluster")).alias("_cluster"))
            .groupBy("_cluster")
            .count()
            .collect()
        )

        self._test_clusters_dict = {
            t["_cluster"]: max(math.ceil(t["count"] / MatchingConfig.BUCKET_SIZE), 1)
            for t in test_info
        }

    def _search_cluster_pairs(
        self,
        control: spark.DataFrame,
        test: spark.DataFrame,
    ) -> tuple[spark.DataFrame, Broadcast]:
        """
        Search for nearest neighbors within cluster pairs.

        Args:
            control (spark.DataFrame): Control group DataFrame with cluster
                and bucket assignments.
            test (spark.DataFrame): Test group DataFrame with cluster and
                bucket assignments.

        Returns:
            tuple: (neighbors_pairs DataFrame, bc_n_neighbors Broadcast variable)
        """
        sc = test.sparkSession.sparkContext
        bc_n_neighbors = sc.broadcast(self.n_neighbors)

        schema = StructType(
            [
                StructField("index", LongType(), False),
                StructField("dists", FloatType(), False),
                StructField("nids", LongType(), False),
            ]
        )
        neighbors_pairs = (
            control.groupBy(*["_cluster", "c_buckets", "t_buckets"])
            .cogroup(test.groupBy(*["_cluster", "c_buckets", "t_buckets"]))
            .applyInPandas(
                lambda c_df, t_df: _crossover_search(c_df, t_df, bc_n_neighbors),
                schema=schema,
            )
        )

        return neighbors_pairs, bc_n_neighbors

    def _aggregate_neighbors(self, neighbors_pairs: spark.DataFrame) -> Dataset:
        """
        Aggregate neighbor pairs into final result format.

        Args:
            neighbors_pairs (spark.DataFrame): DataFrame containing raw neighbor
                pairs with distance and ID information.

        Returns:
            Dataset: Aggregated result dataset with neighbor indices.
        """
        if self.n_neighbors == 1:
            result_df = (
                neighbors_pairs.groupBy("index")
                .agg(F.min(F.struct(F.col("dists"), F.col("nids"))).alias("_1"))
                .select(F.col("index"), F.col("_1").alias("1"))
            )
        else:
            result_df = (
                neighbors_pairs.groupBy("index")
                .agg(
                    F.slice(
                        F.array_sort(
                            F.collect_list(F.struct(F.col("dists"), F.col("nids")))
                        ),
                        1,
                        self.n_neighbors,
                    ).alias("_candidates")
                )
                .select(
                    F.col("index"),
                    *[
                        F.col("_candidates")["nids"][i].alias(f"{i + 1}")
                        for i in range(self.n_neighbors)
                    ],
                )
            )

        return self.result_to_dataset(result=result_df, roles={}, small=False)

    # Predict in other modes

    def _predict_transmission(
        self, test_data: spark.DataFrame
    ) -> tuple[Dataset, list[Broadcast]]:
        """
        Perform prediction using transmission mode (standard distributed search).

        Args:
            test_data (spark.DataFrame): Input DataFrame with query vectors.

        Returns:
            tuple: (result Dataset, list of Broadcast variables to clean up)
        """
        session = test_data.sparkSession
        index_references = self.storage.collect_and_register(self._sharded_rdd)
        self._sharded_rdd.unpersist(blocking=True)
        self._sharded_rdd = None

        broadcasts = self._create_predict_broadcasts(session, index_references)

        result_rdd = test_data.rdd.mapPartitions(
            lambda it: _per_partition_predict(
                it,
                bc_n_neighbors=broadcasts["n_neighbors"],
                bc_references=broadcasts["references"],
                bc_chunk_size=broadcasts["chunk_size"],
                bc_storage=broadcasts["storage"],
            )
        )

        result_df = session.createDataFrame(
            result_rdd, schema=self.PREDICT_SCHEMA
        ).select(
            ["index"]
            + [
                F.expr(f"index_list[{i}]").alias(f"{i + 1}")
                for i in range(self.n_neighbors)
            ]
        )

        result = self.result_to_dataset(result=result_df, roles={}, small=False)

        return result, list(broadcasts.values())

    def _create_predict_broadcasts(
        self,
        session: spark.SparkSession,
        index_references: list[str],
    ) -> dict[str, Broadcast]:
        """
        Create broadcast variables for distributed prediction.

        Args:
            session (spark.SparkSession): Active Spark session.
            index_references (list[str]): List of index file references.

        Returns:
            dict: Dictionary mapping variable names to Broadcast instances:
                - "references": List of index file names.
                - "n_neighbors": Number of neighbors to find.
                - "chunk_size": Number of query rows per batch.
                - "storage": FaissIndexStorage instance.
        """
        return {
            "references": session.sparkContext.broadcast(index_references),
            "n_neighbors": session.sparkContext.broadcast(self.n_neighbors),
            "chunk_size": session.sparkContext.broadcast(
                MatchingConfig.FAISS_CHUNK_SIZE
            ),
            "storage": session.sparkContext.broadcast(self.storage),
        }

    def calc(
        self,
        data: Dataset,
        test_data: Dataset | None = None,
        mode: Literal["auto", "fit", "predict"] | None = None,
        **kwargs,
    ):
        """
        Execute the distributed FAISS matching pipeline for Spark-backed datasets.

        Orchestrates the vectorization, fit, and predict phases based on the
        ``mode`` argument. If a Mahalanobis matrix is provided, features are
        projected before vectorization.

        Args:
            data (Dataset): The baseline (control) dataset.
            test_data (Dataset | None, optional): The query (treatment) dataset.
                Required for "predict" and "auto" modes. Defaults to None.
            mode (Literal["auto", "fit", "predict"] | None, optional): Operation mode.
                - "auto": Vectorize, fit the index, and then predict.
                - "fit": Vectorize data and build the FAISS index only.
                - "predict": Search the index only (requires prior ``fit``).
                Defaults to None (treated as "auto").
            **kwargs: Additional keyword arguments, including:
                - ``model`` (str): Clustering model for "cluster" and "shuffle" modes.
                  Options: "k-means", "birch". Defaults to "k-means".

        Returns:
            SparkFaissExtension or Dataset: The fitted extension (for "fit" mode)
                or the matched indices Dataset (for "predict"/"auto" modes).

        Raises:
            ValueError: If ``test_data`` is None when prediction is required,
                or if the sharded RDD has not been built before prediction.
        """
        mode = mode or "auto"
        operating_data = (
            data._backend_data.data.to_spark(index_col="index")
            if self.mahalanobis is None
            else self._mahalanobis_transform(
                data, self.mahalanobis
            )._backend_data.data.to_spark(index_col="index")
        )

        self._data_size = operating_data.count()
        vectorized_data = self._vectorize_data(operating_data)

        if mode in ["auto", "fit"]:
            model_name = kwargs.get("model", "k-means")
            self._fit(vectorized_data=vectorized_data, model_name=model_name)

        if mode in ["auto", "predict"]:
            if test_data is None:
                raise ValueError("test_data is needed for evaluation")
            if not self._fitted:
                raise ValueError(
                    "Index is not created yet. Call 'fit' before 'predict'."
                )

            test_operating_data = (
                test_data._backend_data.data.to_spark(index_col="index")
                if self.mahalanobis is None
                else self._mahalanobis_transform(
                    test_data, self.mahalanobis
                )._backend_data.data.to_spark(index_col="index")
            )
            vectorized_test = self._vectorize_data(test_operating_data)

            return self._predict(vectorized_test, data.get_storage_level())

        return self

    def unpersist(self) -> None:
        """
        Release Spark resources held by this extension.

        Unpersists the sharded RDD and any clustered data, freeing executor
        memory and disk. Should be called when the extension is no longer needed
        to avoid resource leaks in long-running Spark applications.
        """
        clustered = getattr(self, "_clustered_data", None)
        if clustered is not None:
            clustered.unpersist(blocking=True)
            self._clustered_data = None

        sharded = getattr(self, "_sharded_rdd", None)
        if sharded is not None:
            sharded.unpersist(blocking=True)
            self._sharded_rdd = None

        clustered_control = getattr(self, "_clustered_control", None)
        if clustered_control is not None:
            clustered_control.unpersist(blocking=True)

        clustered_test = getattr(self, "_clustered_test", None)
        if clustered_test is not None:
            clustered_test.unpersist(blocking=True)

    def __enter__(self) -> SparkFaissExtension:
        """
        Context manager entry point.

        Returns:
            SparkFaissExtension: Self, for use in with statements.
        """
        return self

    def __del__(self, *_) -> None:
        """Destructor that ensures resources are cleaned up."""
        self.unpersist()


def get_executor_cache() -> CachingIndex:
    """
    Get or create the executor-side LRU cache for FAISS indexes.

    The cache is stored in the builtins module to persist across function calls
    on the same executor.

    Returns:
        CachingIndex: The LRU cache instance for FAISS indexes.
    """
    if not hasattr(builtins, "_faiss_index_cache"):
        builtins._faiss_index_cache = CachingIndex()
    return builtins._faiss_index_cache

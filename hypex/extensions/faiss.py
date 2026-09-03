from __future__ import annotations

import builtins
import gc
import math
from abc import abstractmethod
from typing import (
    ClassVar,
    Iterable,
    Literal,
)

import faiss
import numpy as np
import pandas as pd

# Spark imports
import pyspark.sql as spark
import pyspark.sql.functions as F
from pyspark import Broadcast
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StructField,
    StructType,
)
from sklearn.cluster import Birch, MiniBatchKMeans

from ..config import MatchingConfig
from ..dataset import AdditionalMatchingRole, Dataset
from ..dataset.backends import PandasDataset, SparkDataset
from ..utils.errors import AbstractMethodError
from ..utils.index_utils import CachingIndex, FaissIndexStorage

#TODO: Logger
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
        mahalonobis (Dataset | None): Optional Mahalanobis transformation matrix
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
        mahalonobis: Dataset = None,
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
            mahalonobis (Dataset, optional): A pre-computed Mahalanobis
                transformation matrix. If provided, features are projected into
                a decorrelated space before building the FAISS index.
                Defaults to None.
        """
        self.n_neighbors = n_neighbors
        self.faiss_mode = faiss_mode
        self.mahalonobis = mahalonobis
        self.index = None

        super().__init__()

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

# TODO: add mahalonobis matrix logic into pandas realization
@backend_factory.register(FaissExtension, PandasDataset)
class PandasFaissExtension(FaissExtension):
    """
    Faiss backend-slave class for faiss pairs matching.

    Pandas backend implementation for FAISS-based nearest neighbor matching.

    Performs in-memory k-NN search using FAISS flat or IVF indexes. Suitable
    for datasets that fit entirely in the driver's memory.

    Inherits from:
        FaissExtension: The master-abstract FAISS extension class.
    """
    def __init__(self, n_neighbors = 1, faiss_mode = "auto", mahalonobis: Dataset = None,):
        """
        Initialize the Pandas FAISS extension.

        Args:
            n_neighbors (int, optional): Number of nearest neighbors. Defaults to 1.
            faiss_mode (str, optional): Execution mode. Defaults to "auto".
            mahalonobis (Dataset, optional): Mahalanobis transformation matrix.
                Defaults to None.
        """
        super().__init__(n_neighbors, faiss_mode, mahalonobis)

    @staticmethod
    def _mahalonobis_transform(data: Dataset, mahalonobis: Dataset | None) -> Dataset:
        """
        Apply the Mahalanobis transformation to the input dataset.

        Projects the features into a decorrelated space using the provided
        transformation matrix. If no matrix is provided, returns the data
        unchanged.

        Args:
            data (Dataset): The input dataset to transform.
            mahalonobis (Dataset | None): The transformation matrix. If None,
                no transformation is applied.

        Returns:
            Dataset: The transformed dataset, or the original if ``mahalonobis``
                is None.
        """
        if mahalonobis is None:
            return data
        else:
            return data.dot(mahalonobis)

    @staticmethod
    def _prepare_indexes(index: np.ndarray, dist: np.ndarray, k: int):
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
            X: np.ndarray
    ) -> pd.Series:
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
            pd.Series: A Series-like result containing the matched indices,
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
        result = self.result_to_dataset(result=indexes, roles={}).set_index(test_data.index, drop=False)
        result.index.name = None

        return result

    def _fit(
            self,
            data: Dataset,
            test_data: Dataset
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
        X = self._mahalonobis_transform(data, self.mahalonobis).data.values
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(X.shape[1]))
        if (
            (
                (len(X) > 1_000_000 and self.faiss_mode == "auto")
                or self.faiss_mode == "fast"
            )
            and len(X) > 1_000
            and len(test_data) > 1_000
        ):
            m = 4 # heuristic
            n_clusters = int(np.sqrt(len(X) / m))
            nlist = min(n_clusters, max(1, X.shape[0] // 39))
            quantizer = faiss.IndexFlatL2(X.shape[1])
            _index = faiss.IndexIVFFlat(quantizer, X.shape[1], nlist)
            _index.train(X)
            self.index = faiss.IndexIDMap(_index)
        self.index.add_with_ids(X, np.array(data.index.tolist(), dtype=np.int64))

    def calc(
            self,
            data,
            test_data = None,
            mode = None,
            **kwargs
    ):
        """
        Execute the FAISS matching pipeline for Pandas-backed datasets.

        Orchestrates the fit and predict phases based on the ``mode`` argument.

        Args:
            data: The baseline (control) dataset.
            test_data: The query (treatment) dataset. Required for "predict"
                and "auto" modes.
            mode (str, optional): Operation mode ("auto", "fit", "predict").
                Defaults to None (treated as "auto").
            **kwargs: Additional keyword arguments.

        Returns:
            FaissExtension or Dataset: The fitted extension (for "fit" mode)
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
                raise ValueError("index is not created yet. Raise 'fit' before 'predict'.")

            X = (
                self._mahalonobis_transform(test_data, self.mahalonobis).data.values if mode == "auto"
                else self._mahalonobis_transform(data, self.mahalonobis).data.values
            )
            return self._predict(data, test_data, X)
        return self


# ---------------------------------------------------------------------------
# Global functions for PySpark partition logic
# ---------------------------------------------------------------------------

@logger.log_methods(log_args=False, log_result=False, private=True)
@backend_factory.register(FaissExtension, SparkDataset)
class SparkFaissExtension(FaissExtension):
    """
    Faiss backend-slave class for faiss pairs matching.

    Spark backend implementation for distributed FAISS-based nearest neighbor
    matching. Two strategies are available, selected by
    ``MatchingConfig.FAISS_SEARCH_MODE``.

    **"copartitioned" (default).** The IVF quantizer is trained once (on a
    sample or on the full dataset), then:

    1. every control row is assigned to its nearest centroid — the same rule
       FAISS uses when adding a vector to an inverted list;
    2. every query is routed to its ``FAISS_N_PROBES`` nearest non-empty
       centroids — the same rule FAISS uses when probing;
    3. the two sides are co-grouped on the cluster key, so a task holds one
       cluster's control vectors and the queries that belong to it, builds a
       flat index over them and searches;
    4. per-query candidates from all probed clusters are reduced to the global
       top-``n_neighbors``.

    Because a task only ever sees one cluster, executor memory depends on the
    cluster size, not on the dataset size. Nothing is collected to the driver
    except the centroids and per-cluster counters, and no index files are
    distributed. Clusters larger than ``FAISS_MAX_GROUP_ROWS`` are split into
    salted sub-groups with the opposite side duplicated across them, which
    bounds the task even on skewed data.

    **"legacy".** One FAISS index per data partition; the serialized indexes
    are pulled to the driver, written to files, distributed with
    ``SparkFiles.addFile`` and every executor searches every index. Executor
    memory grows with the whole dataset. Kept for reproducing earlier results.

    Inherits from:
        FaissExtension: The master-abstract FAISS extension class.

    See Also:
        CachingIndex: Executor-side LRU cache for FAISS indexes (legacy mode).
    """

    PREDICT_SCHEMA = StructType([
        StructField("index",          LongType(),            False),
        StructField("index_list",     ArrayType(LongType()), False)
    ])
    ASSIGN_SCHEMA = StructType([
        StructField("index",    LongType(),              False),
        StructField("_fvec",    ArrayType(FloatType()),  False),
        StructField("_cluster", IntegerType(),           False),
    ])
    ASSIGN_MULTI_SCHEMA = StructType([
        StructField("index",     LongType(),               False),
        StructField("_fvec",     ArrayType(FloatType()),   False),
        StructField("_clusters", ArrayType(IntegerType()), False),
    ])
    GROUP_SEARCH_SCHEMA = StructType([
        StructField("index", LongType(),   False),
        StructField("nid",   LongType(),   False),
        StructField("dist",  DoubleType(), False),
    ])
    CLUSTERING_METHODS_MAPPER: ClassVar = {
        "k-means": {
            "model": MiniBatchKMeans,
            "params": {
                "random_state" : 21,
                "max_no_improvement" : None,
                "batch_size" : 5_000_000,
                "n_init" : 5
            }
        },
        "birch": {
            "model": Birch,
            "params": {
                "n_clusters": None
            }
        }
    }

    def __init__(
            self,
            n_neighbors = 1,
            faiss_mode = "auto",
            mahalonobis: Dataset = None,
    ):
        """
        Initialize the Spark FAISS extension.

        Args:
            n_neighbors (int, optional): Number of nearest neighbors. Defaults to 1.
            faiss_mode (str, optional): Execution mode. Defaults to "auto".
            mahalonobis (Dataset, optional): Mahalanobis transformation matrix.
                Defaults to None.
        """
        super().__init__(n_neighbors, faiss_mode, mahalonobis)
        self.seed: int = 21
        self.storage: FaissIndexStorage | None = None
        self._data_size: int | None = None
        self._sharded_rdd = None
        self.feature_cols: list[str] | None = None
        # copartitioned state
        self._centroids: np.ndarray | None = None
        self._control_df: spark.DataFrame | None = None
        self._cluster_salts: dict[int, int] | None = None

    # ------------------------------------------------------------------
    # data preparation
    # ------------------------------------------------------------------

    def _feature_columns(self, data: spark.DataFrame) -> list[str]:
        """
        Return the feature columns of ``data`` in a stable order.

        The order must be identical between the fit and the predict frames,
        otherwise the two sides would be embedded in different spaces. Column
        order of the frame itself is used rather than a ``set`` difference,
        which is not order-stable.

        Args:
            data (spark.DataFrame): Frame carrying the ``index`` column.

        Returns:
            list[str]: Feature column names, ``index`` excluded.

        Raises:
            TypeError: If any column is of a string/categorical type.
        """
        if len(set(map(lambda x: x[1], data.dtypes)).intersection(['varchar', 'string'])) > 0:
            raise TypeError("Unencoded categorical features are not allowed!")
        return [c for c in data.columns if c != "index"]

    def _vectorize_data(
            self,
            data: spark.DataFrame
    ) -> spark.DataFrame:
        """
        Assemble feature columns into a single vector column for FAISS (legacy mode).

        Args:
            data (spark.DataFrame): Input Spark DataFrame containing numeric
                features and an ``index`` column.

        Returns:
            spark.DataFrame: DataFrame with an additional ``_features`` column
                of type ``pyspark.ml.linalg.Vector``.

        Raises:
            TypeError: If any feature column has a string/categorical type.
        """
        self.feature_cols = self._feature_columns(data)

        vecAssembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="_features",
            handleInvalid="keep"
        )

        return vecAssembler.transform(data)

    def _to_array_frame(self, data: spark.DataFrame) -> spark.DataFrame:
        """
        Project the features into a single ``array<float>`` column.

        FAISS operates on ``float32`` regardless of the input precision, so the
        vectors are narrowed here: every row carried through a shuffle or an
        Arrow batch is half the size of the ``double`` representation, and no
        Python float objects are ever materialized on the executors.

        Args:
            data (spark.DataFrame): Input frame with numeric features and an
                ``index`` column.

        Returns:
            spark.DataFrame: Frame with columns ``index`` (long) and ``_fvec``
                (``array<float>``).

        Raises:
            TypeError: If any feature column has a string/categorical type.
        """
        self.feature_cols = self._feature_columns(data)

        return data.select(
            F.col("index").cast(LongType()).alias("index"),
            F.array(*[F.col(c).cast(FloatType()) for c in self.feature_cols]).alias("_fvec"),
        )

    # ------------------------------------------------------------------
    # quantizer training (shared by both modes)
    # ------------------------------------------------------------------

    def _prefit(self, vectorized_data: spark.DataFrame, model_name: str, feature_col: str = "_fvec") -> None:
        """
        Train the IVF quantizer on the full dataset via iterative partition upload.

        Loads feature vectors from Spark partitions in batches of
        ``FAISS_DRIVER_INDEX_LIMIT`` rows and incrementally fits a clustering
        model (MiniBatchKMeans or BIRCH) on the driver. The resulting cluster
        centers initialize the FAISS IVF quantizer.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the feature
                column.
            model_name (str): Name of the clustering algorithm to use. Must be a
                key in ``CLUSTERING_METHODS_MAPPER`` (e.g., "k-means", "birch").
            feature_col (str, optional): Name of the feature column. Defaults to
                ``"_fvec"``.
        """

        def _partition_load(partition_iter: Iterable, batch_size: int):
            """
            Load batches of feature vectors from a Spark partition iterator.

            Args:
                partition_iter (Iterable): Iterator over partition rows.
                batch_size (int): Number of rows to accumulate per batch.

            Yields:
                list: A batch of feature vectors.
            """
            batch = []
            for row in partition_iter:
                batch.append(list(row[feature_col]))
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
            vectorized_data
            .select(feature_col)
            .rdd
            .mapPartitions(lambda it: _partition_load(it, batch_size))
            .toLocalIterator()
        ):
            np_batch = np.array(batch, dtype=np.float32)
            model.partial_fit(np_batch)

        if np_batch is not None:
            del np_batch
            gc.collect()

        centroids = model.cluster_centers_ if model_name == 'k-means' else model.subcluster_centers_
        centroids = centroids.astype(np.float32)
        index_shape = centroids.shape[1]
        nlist = len(centroids)

        quantizer = faiss.IndexFlatL2(index_shape)
        quantizer.add(centroids)
        self.index = faiss.IndexIVFFlat(quantizer, index_shape, nlist)
        self.index.is_trained = True

        self._clustering_model = model

    def _train_quantizer(
            self,
            prepared_data: spark.DataFrame,
            mode: Literal["sample", "full"],
            model_name: str | None,
            feature_col: str,
    ) -> None:
        """
        Train the coarse quantizer and store its centroids in ``self._centroids``.

        Args:
            prepared_data (spark.DataFrame): Frame carrying ``feature_col``.
            mode (Literal["sample", "full"]): "sample" trains on a random subset
                of up to ``FAISS_SAMPLE_TARGET`` rows; "full" runs iterative
                mini-batch clustering over everything via :meth:`_prefit`.
            model_name (str | None): Clustering model for "full" mode.
            feature_col (str): Name of the feature column.

        Raises:
            ValueError: If ``mode`` is neither "sample" nor "full".
        """
        m = 4  # heuristic
        self.k = max(1, int(np.sqrt(self._data_size / m)))

        if mode == "sample":
            frac = min(MatchingConfig.FAISS_SAMPLE_TARGET / max(self._data_size, 1), 1.0)
            sample_rows = (
                prepared_data
                .sample(fraction=frac, seed=self.seed)
                .select(feature_col)
                .collect()
            )

            X = np.array(
                [list(row[feature_col]) for row in sample_rows],
                dtype=np.float32,
            )

            d = X.shape[1]
            # IVF Faiss подерживает до 39 * (training points) на один кластер
            nlist = min(self.k, max(1, X.shape[0] // 39))

            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self.index.train(X)

        elif mode == "full":
            self._prefit(
                vectorized_data=prepared_data,
                model_name=model_name,
                feature_col=feature_col,
            )

        else:
            raise ValueError(f"Incorrect faiss fit mode: '{mode}'")

        coarse = faiss.downcast_index(self.index.quantizer)
        self._centroids = np.ascontiguousarray(
            coarse.reconstruct_n(0, coarse.ntotal), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # copartitioned mode
    # ------------------------------------------------------------------

    def _fit_copartitioned(self, arr_df: spark.DataFrame) -> None:
        """
        Assign every control row to its nearest centroid and persist the result.

        This replaces the per-partition index build: instead of materializing a
        FAISS index per data partition, the control side is only tagged with a
        cluster id, which is what the co-grouped search needs. Cluster sizes are
        collected (one row per cluster) to decide how many salted sub-groups a
        cluster needs so that no group exceeds ``FAISS_MAX_GROUP_ROWS``.

        Args:
            arr_df (spark.DataFrame): Control frame with ``index`` and ``_fvec``.
        """
        session = arr_df.sparkSession
        bc_centroids = session.sparkContext.broadcast(self._centroids)

        def _assign_one(iterator: Iterable) -> Iterable:
            import faiss
            import numpy as np

            centroids = bc_centroids.value
            cq = faiss.IndexFlatL2(centroids.shape[1])
            cq.add(centroids)

            for pdf in iterator:
                if len(pdf) == 0:
                    continue
                X = np.ascontiguousarray(np.vstack(pdf["_fvec"].to_numpy()), dtype=np.float32)
                _, assigned = cq.search(X, 1)
                out = pdf.loc[:, ["index", "_fvec"]].copy()
                out["_cluster"] = assigned[:, 0].astype(np.int32)
                yield out

        control = (
            arr_df
            .mapInPandas(_assign_one, schema=self.ASSIGN_SCHEMA)
            .persist(MatchingConfig.FAISS_PERSIST_POLITIC)
        )

        # one row per non-empty cluster; also materializes the persist
        sizes = control.groupBy("_cluster").count().collect()
        if not sizes:
            raise ValueError("Faiss fit received an empty control dataset")

        max_rows = max(1, MatchingConfig.FAISS_MAX_GROUP_ROWS)
        self._cluster_salts = {
            int(row["_cluster"]): max(1, math.ceil(int(row["count"]) / max_rows))
            for row in sizes
        }
        self._control_df = control

    def _layout_frame(
            self,
            session: spark.SparkSession,
            n_buckets: dict[int, int],
    ) -> spark.DataFrame:
        """
        Build the tiny (one row per non-empty cluster) broadcastable layout frame.

        Args:
            session (spark.SparkSession): Active session.
            n_buckets (dict[int, int]): Query-side bucket count per cluster.

        Returns:
            spark.DataFrame: Columns ``_cluster``, ``_nsalt``, ``_nbucket``.
        """
        rows = [
            (int(cluster), int(n_salt), int(n_buckets.get(cluster, 1)))
            for cluster, n_salt in sorted(self._cluster_salts.items())
        ]
        schema = StructType([
            StructField("_cluster", IntegerType(), False),
            StructField("_nsalt",   IntegerType(), False),
            StructField("_nbucket", IntegerType(), False),
        ])
        return session.createDataFrame(rows, schema=schema)

    def _predict_copartitioned(
            self,
            arr_df: spark.DataFrame,
            storage_level: str | None,
    ) -> Dataset:
        """
        Route queries to their nearest clusters and search inside each group.

        Steps:

        1. every query is assigned its ``FAISS_N_PROBES`` nearest **non-empty**
           centroids (probing an empty inverted list can only waste a probe, so
           empty clusters are excluded — this also guarantees that every query
           meets at least one control row);
        2. the query counts per cluster decide the bucket split of oversized
           clusters, mirroring the salt split of the control side;
        3. control and queries are co-grouped on ``(cluster, salt, bucket)`` and
           searched with a flat index built from the group itself;
        4. candidates are reduced to the global top-``n_neighbors`` per query.

        Args:
            arr_df (spark.DataFrame): Query frame with ``index`` and ``_fvec``.
            storage_level (str | None): Storage level for the resulting Dataset.

        Returns:
            Dataset: Matched neighbour indices, indexed by the query index.
        """
        session = arr_df.sparkSession
        sc = session.sparkContext

        cluster_ids = np.array(sorted(self._cluster_salts.keys()), dtype=np.int32)
        centroids = np.ascontiguousarray(self._centroids[cluster_ids], dtype=np.float32)
        probes = int(max(1, min(MatchingConfig.FAISS_N_PROBES, len(cluster_ids))))

        bc_centroids = sc.broadcast(centroids)
        bc_cluster_ids = sc.broadcast(cluster_ids)
        bc_probes = sc.broadcast(probes)

        def _assign_topp(iterator: Iterable) -> Iterable:
            import faiss
            import numpy as np

            centroids_local = bc_centroids.value
            ids_local = bc_cluster_ids.value
            p = bc_probes.value

            cq = faiss.IndexFlatL2(centroids_local.shape[1])
            cq.add(centroids_local)

            for pdf in iterator:
                if len(pdf) == 0:
                    continue
                X = np.ascontiguousarray(np.vstack(pdf["_fvec"].to_numpy()), dtype=np.float32)
                _, positions = cq.search(X, p)
                out = pdf.loc[:, ["index", "_fvec"]].copy()
                out["_clusters"] = list(ids_local[positions])
                yield out

        queries = (
            arr_df
            .mapInPandas(_assign_topp, schema=self.ASSIGN_MULTI_SCHEMA)
            .persist(MatchingConfig.FAISS_PERSIST_POLITIC)
        )

        # query volume per cluster -> bucket split for the oversized ones
        max_rows = max(1, MatchingConfig.FAISS_MAX_GROUP_ROWS)
        q_counts = (
            queries
            .select(F.explode(F.col("_clusters")).alias("_cluster"))
            .groupBy("_cluster")
            .count()
            .collect()
        )
        n_buckets = {
            int(row["_cluster"]): max(1, math.ceil(int(row["count"]) / max_rows))
            for row in q_counts
        }

        layout = F.broadcast(self._layout_frame(session, n_buckets))

        control_side = (
            self._control_df
            .join(layout, on="_cluster", how="inner")
            .withColumn("_salt", F.pmod(F.hash(F.col("index")), F.col("_nsalt")))
            .withColumn("_bucket", F.explode(F.sequence(F.lit(0), F.col("_nbucket") - F.lit(1))))
            .select("_cluster", "_salt", "_bucket", "index", "_fvec")
        )

        query_side = (
            queries
            .withColumn("_cluster", F.explode(F.col("_clusters")))
            .join(layout, on="_cluster", how="inner")
            .withColumn("_bucket", F.pmod(F.hash(F.col("index")), F.col("_nbucket")))
            .withColumn("_salt", F.explode(F.sequence(F.lit(0), F.col("_nsalt") - F.lit(1))))
            .select("_cluster", "_salt", "_bucket", "index", "_fvec")
        )

        n_neighbors = int(self.n_neighbors)
        keys = ["_cluster", "_salt", "_bucket"]

        def _search_group(control_pdf: pd.DataFrame, query_pdf: pd.DataFrame) -> pd.DataFrame:
            import faiss
            import numpy as np
            import pandas as pd

            empty = pd.DataFrame({
                "index": np.empty(0, dtype=np.int64),
                "nid": np.empty(0, dtype=np.int64),
                "dist": np.empty(0, dtype=np.float64),
            })
            if len(control_pdf) == 0 or len(query_pdf) == 0:
                return empty

            X = np.ascontiguousarray(np.vstack(control_pdf["_fvec"].to_numpy()), dtype=np.float32)
            control_ids = control_pdf["index"].to_numpy(dtype=np.int64)
            Q = np.ascontiguousarray(np.vstack(query_pdf["_fvec"].to_numpy()), dtype=np.float32)
            query_ids = query_pdf["index"].to_numpy(dtype=np.int64)

            index = faiss.IndexFlatL2(X.shape[1])
            index.add(X)
            k = min(n_neighbors, X.shape[0])
            distances, positions = index.search(Q, k)

            return pd.DataFrame({
                "index": np.repeat(query_ids, k),
                "nid": control_ids[positions.ravel()],
                "dist": distances.ravel().astype(np.float64),
            })

        pairs = (
            control_side.groupBy(*keys)
            .cogroup(query_side.groupBy(*keys))
            .applyInPandas(_search_group, schema=self.GROUP_SEARCH_SCHEMA)
        )

        if n_neighbors == 1:
            result_df = (
                pairs
                .groupBy("index")
                .agg(F.min(F.struct("dist", "nid")).alias("_best"))
                .select(F.col("index"), F.col("_best.nid").alias("1"))
            )
        else:
            result_df = (
                pairs
                .groupBy("index")
                .agg(
                    F.slice(
                        F.array_sort(F.collect_list(F.struct("dist", "nid"))),
                        1, n_neighbors,
                    ).alias("_top")
                )
                .select(
                    F.col("index"),
                    *[F.col("_top")[i]["nid"].alias(f"{i + 1}") for i in range(n_neighbors)],
                )
            )

        result = self.result_to_dataset(result=result_df, roles={}, small=False).set_index('index')
        result.index.name = None

        storage_level = storage_level or "MEMORY_AND_DISK"
        result.persist(storage_level=storage_level, action="count")
        if MatchingConfig.FAISS_CHECKPOINT_RESULT:
            result.checkpoint(eager=True)

        queries.unpersist()

        return result

    # ------------------------------------------------------------------
    # legacy mode
    # ------------------------------------------------------------------

    def _fit_legacy(self, vectorized_data: spark.DataFrame) -> None:
        """
        Build one FAISS index per data partition and persist them as an RDD.

        Args:
            vectorized_data (spark.DataFrame): Frame with the ``_features``
                vector column and a trained ``self.index``.
        """
        def _spark_partition_fit(
            iterator: Iterable,
            bc_index: Broadcast,
            bc_storage: Broadcast
        ):
            """
            Build a local FAISS index on each Spark partition.

            Args:
                iterator (Iterable): Iterator over partition rows.
                bc_index (Broadcast): Broadcasted pre-trained FAISS index.
                bc_storage (Broadcast): Broadcasted index storage.

            Yields:
                bytes: Serialized FAISS index for the partition.
            """
            import faiss
            import numpy as np

            index = bc_index.value
            storage = bc_storage.value
            ids, vectors = [], []
            for row in iterator:
                ids.append(row["index"])
                vectors.append(list(row['_features']))

            if not ids:
                return  # for empty partition

            ids = np.array(ids, dtype=np.int64)
            vectors = np.array(vectors, dtype=np.float32)

            index_copy = faiss.clone_index(index)
            index_with_ids = faiss.IndexIDMap(index_copy)
            index_with_ids.add_with_ids(vectors, ids)

            yield storage.save_index(index_with_ids)

        session = vectorized_data.sparkSession
        self.storage = FaissIndexStorage(session)

        self.index.nprobe = min(self.n_neighbors * 2, self.k)
        bc_index = session.sparkContext.broadcast(self.index)
        bc_storage = session.sparkContext.broadcast(self.storage)
        del self.index
        self.index = None
        gc.collect()

        features = ["index", "_features"]
        self._sharded_rdd = (
            vectorized_data
            .select(*features)
            .rdd
            .mapPartitions(lambda it: _spark_partition_fit(it, bc_index, bc_storage))
            .persist(MatchingConfig.FAISS_PERSIST_POLITIC)
        )
        self._sharded_rdd.count()

    def _predict_legacy(
            self,
            test_data: spark.DataFrame,
            storage_level: str | None,
        ) -> Dataset:
        """
        Search every partition index for every query (legacy strategy).

        The partition indexes are pulled to the driver, written to files and
        distributed to the executors with ``SparkFiles``; each task then walks
        the whole list of indexes. Compared to the original implementation the
        loops are inverted — the queries of a partition are materialized once as
        a ``float32`` matrix and each index is loaded once per task instead of
        once per chunk — and the executor cache is bounded, so a worker holds at
        most ``FAISS_INDEX_CACHE_LIMIT`` indexes rather than all of them. The
        driver round-trip and the file fan-out remain: this path is kept for
        reproducing earlier results, not for large datasets.

        Args:
            test_data (spark.DataFrame): Query frame with ``_features``.
            storage_level (str | None): Storage level for the resulting Dataset.

        Returns:
            Dataset: Matched neighbour indices, indexed by the query index.
        """
        def _per_partition_predict(
            shard_iter: Iterable,
            bc_n_neighbors: Broadcast,
            bc_references: Broadcast,
            bc_chunk_size: Broadcast,
            bc_k: Broadcast,
            bc_storage: Broadcast
        ):
            """
            Search all partition indexes for the queries of one Spark partition.

            Yields:
                tuple: ``(query_id, [neighbor_ids])`` per query vector.
            """
            import numpy as np

            cache = get_executor_cache()

            real_n = bc_n_neighbors.value
            references = bc_references.value
            chunk_size = bc_chunk_size.value
            storage = bc_storage.value

            def iter_chunk(it: Iterable, size: int):
                chunk = []
                for row in it:
                    chunk.append(row)
                    if len(chunk) >= size:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk

            # materialize the partition queries once, as compact arrays
            id_blocks, vec_blocks = [], []
            for chunk in iter_chunk(shard_iter, chunk_size):
                id_blocks.append(np.array([r["index"] for r in chunk], dtype=np.int64))
                vec_blocks.append(np.array([list(r["_features"]) for r in chunk], dtype=np.float32))
                del chunk

            if not id_blocks:
                return

            query_ids = np.concatenate(id_blocks)
            batch = np.vstack(vec_blocks)
            del id_blocks, vec_blocks

            n_queries = query_ids.shape[0]
            best_dist = np.full((n_queries, real_n), np.inf, dtype=np.float32)
            best_id = np.full((n_queries, real_n), -1, dtype=np.int64)

            for ref in references:
                tmp_index = cache.get(ref, storage, nprobe=min(real_n, bc_k.value))
                k = min(real_n, tmp_index.ntotal)
                if k <= 0:
                    continue
                dists, nids = tmp_index.search(batch, k)
                del tmp_index

                merged_dist = np.concatenate([best_dist, dists.astype(np.float32)], axis=1)
                merged_id = np.concatenate([best_id, nids.astype(np.int64)], axis=1)
                merged_dist[merged_id < 0] = np.inf
                order = np.argsort(merged_dist, axis=1, kind="stable")[:, :real_n]
                best_dist = np.take_along_axis(merged_dist, order, axis=1)
                best_id = np.take_along_axis(merged_id, order, axis=1)

            for position in range(n_queries):
                neighbours = [int(v) for v in best_id[position] if v >= 0]
                yield (int(query_ids[position]), neighbours)

        session = test_data.sparkSession
        index_references = self.storage.collect_and_register(self._sharded_rdd)

        self._sharded_rdd.unpersist()
        self._sharded_rdd = None
        bc_index_references = session.sparkContext.broadcast(index_references)
        bc_n_neighbors = session.sparkContext.broadcast(self.n_neighbors)
        bc_chunk_size = session.sparkContext.broadcast(MatchingConfig.FAISS_CHUNK_SIZE)
        bc_k = session.sparkContext.broadcast(self.k)
        bc_storage = session.sparkContext.broadcast(self.storage)

        result_rdd = test_data.rdd.mapPartitions(lambda it:
                                        _per_partition_predict(
                                        it,
                                        bc_n_neighbors=bc_n_neighbors,
                                        bc_references=bc_index_references,
                                        bc_chunk_size=bc_chunk_size,
                                        bc_k=bc_k,
                                        bc_storage=bc_storage
            )
        )

        result_df = (
            session.createDataFrame(result_rdd, schema=self.PREDICT_SCHEMA)
            .select(
                ['index'] +
                [F.expr(f"index_list[{i}]").alias(f"{i + 1}") for i in range(self.n_neighbors)]
            )
        )
        result = self.result_to_dataset(result=result_df, roles={}, small=False).set_index('index')
        result.index.name = None

        storage_level = storage_level or "MEMORY_AND_DISK"
        result.persist(storage_level=storage_level, action="count")
        if MatchingConfig.FAISS_CHECKPOINT_RESULT:
            result.checkpoint(eager=True)

        return result

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    def _fit(
            self,
            vectorized_data: spark.DataFrame,
            mode: Literal["sample", "full"],
            model_name: str | None
    ) -> None:
        """
        Train the quantizer and prepare the search structures.

        Args:
            vectorized_data (spark.DataFrame): Prepared control frame.
            mode (Literal["sample", "full"]): IVF training algorithm.
            model_name (str | None): Clustering model name for "full" mode.

        Raises:
            ValueError: If ``mode`` is not "sample" or "full".
        """
        legacy = MatchingConfig.FAISS_SEARCH_MODE == "legacy"
        feature_col = "_features" if legacy else "_fvec"

        self._train_quantizer(
            prepared_data=vectorized_data,
            mode=mode,
            model_name=model_name,
            feature_col=feature_col,
        )

        if legacy:
            self._fit_legacy(vectorized_data)
        else:
            self._fit_copartitioned(vectorized_data)

    def _predict(
            self,
            test_data: spark.DataFrame,
            storage_level: Literal[
                    "MEMORY_ONLY", "MEMORY_AND_DISK", "DISK_ONLY",
            ] | None
        ) -> Dataset:
        """
        Find the nearest control neighbours of every query row.

        Args:
            test_data (spark.DataFrame): Prepared query frame.
            storage_level (Literal): Storage strategy for the result.

        Returns:
            Dataset: Matched neighbour indices, indexed by the query index.
        """
        if MatchingConfig.FAISS_SEARCH_MODE == "legacy":
            return self._predict_legacy(test_data, storage_level)
        return self._predict_copartitioned(test_data, storage_level)

    def calc(
            self,
            data,
            test_data = None,
            mode = None,
            **kwargs
    ):
        """
        Execute the distributed FAISS matching pipeline for Spark-backed datasets.

        Args:
            data: The baseline (control) dataset.
            test_data: The query (treatment) dataset. Required for "predict"
                and "auto" modes.
            mode (str, optional): Operation mode ("auto", "fit", "predict").
                Defaults to None (treated as "auto").
            **kwargs: Additional keyword arguments, including:
                - ``fit_mode`` (str): "sample" or "full" for IVF training.
                  Defaults to "full".
                - ``model`` (str): Clustering model for "full" mode.
                  Defaults to "k-means".

        Returns:
            SparkFaissExtension or Dataset: The fitted extension (for "fit" mode)
                or the matched indices Dataset (for "predict"/"auto" modes).

        Raises:
            ValueError: If ``test_data`` is None when prediction is required,
                if the search structures were not built before prediction, or
                if ``FAISS_SEARCH_MODE`` is unknown.
        """
        search_mode = MatchingConfig.FAISS_SEARCH_MODE
        if search_mode not in ("copartitioned", "legacy"):
            raise ValueError(
                f"Unknown FAISS_SEARCH_MODE: '{search_mode}'. "
                "Expected 'copartitioned' or 'legacy'."
            )
        legacy = search_mode == "legacy"
        prepare = self._vectorize_data if legacy else self._to_array_frame

        mode = mode or "auto"
        operating_data: spark.DataFrame = (
            data._backend_data.data.to_spark(index_col='index')
            if self.mahalonobis is None
            else data.dot(self.mahalonobis)._backend_data.data.to_spark(index_col='index')
        )
        self._data_size = operating_data.count()
        prepared_data = prepare(operating_data)
        fit_features = self.feature_cols

        if mode in ["auto", "fit"]:
            fit_mode = kwargs.get("fit_mode", "full")
            model_name = kwargs.get("model", "k-means")
            self._fit(
                vectorized_data=prepared_data,
                mode=fit_mode,
                model_name=model_name
            )

        if mode in ["auto", "predict"]:
            if test_data is None:
                raise ValueError("test_data is needed for evaluation")
            if legacy and self._sharded_rdd is None:
                raise ValueError("Index is not created yet. Call 'fit' before 'predict'.")
            if not legacy and self._control_df is None:
                raise ValueError("Index is not created yet. Call 'fit' before 'predict'.")

            test_operating_data = (
                test_data._backend_data.data.to_spark(index_col='index')
                if self.mahalonobis is None
                else test_data.dot(self.mahalonobis)._backend_data.data.to_spark(index_col='index')
            )
            prepared_test = prepare(test_operating_data)
            if fit_features is not None and self.feature_cols != fit_features:
                raise ValueError(
                    "Control and test datasets have different feature columns: "
                    f"{fit_features} vs {self.feature_cols}"
                )

            return self._predict(prepared_test, data.get_storage_level())

    def unpersist(self) -> None:
        """
        Release Spark resources held by this extension.

        Unpersists the cached control assignment and the sharded RDD, freeing
        executor memory and disk. Should be called when the extension is no
        longer needed to avoid resource leaks in long-running Spark applications.
        """
        control = getattr(self, '_control_df', None)
        if control is not None:
            control.unpersist()
            self._control_df = None

        sharded = getattr(self, '_sharded_rdd', None)
        if sharded is not None:
            sharded.unpersist()
            self._sharded_rdd = None

    def __enter__(self) -> SparkFaissExtension:
        return self

    def __exit__(self, *_) -> None:
        self.unpersist()

    def __del__(self, *_) -> None:
        self.unpersist()


def get_executor_cache() -> CachingIndex:
    """
    Return the process-wide FAISS index cache of the current executor.

    The cache is bounded by ``MatchingConfig.FAISS_INDEX_CACHE_LIMIT``; an
    unbounded cache lets a Python worker accumulate every partition index it
    ever touches, which on a large dataset means the whole control side in RSS.

    Returns:
        CachingIndex: The executor-local cache.
    """
    cache = getattr(builtins, "_faiss_index_cache", None)
    limit = MatchingConfig.FAISS_INDEX_CACHE_LIMIT
    if cache is None:
        cache = CachingIndex(limit)
        builtins._faiss_index_cache = cache
    else:
        cache.resize(limit)
    return cache

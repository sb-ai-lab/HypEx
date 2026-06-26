from __future__ import annotations

from typing import (
    Literal,
    Iterable
)
from collections import OrderedDict

from ..dataset import AdditionalMatchingRole, Dataset
from .abstract import MLExtension
from abc import abstractmethod
from ..utils.errors import AbstractMethodError
from ..utils.registry import backend_factory
from ..dataset.backends import PandasDataset, SparkDataset

import numpy as np
import pandas as pd
import faiss
import gc
import os
import builtins
import threading
import uuid
import tempfile
import shutil


# Spark imports
import pyspark.sql as spark
import pyspark.pandas as ps
import pyspark.sql.functions as F
# ps.set_option('compute.ops_on_diff_frames', True)

from pyspark.ml.feature import VectorAssembler
from pyspark import StorageLevel, Broadcast, RDD, SparkFiles

from pyspark.sql.types import (
    StructType, 
    StructField, 
    LongType,
    ArrayType
)
from sklearn.cluster import MiniBatchKMeans, Birch

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
            n_clusters = np.sqrt(len(X) / m)
            _index = faiss.IndexIVFFlat(self.index, X.shape[1], n_clusters)
            _index.train(X)
            self.index = faiss.IndexIDMap(_index)
        self.index.add_with_ids(X, np.asarray(data.index, dtype=np.int64))
        
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

def _partition_load(partition_iter: Iterable, batch_size: int):
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

def _spark_partition_fit(
    iterator: Iterable, 
    bc_index: Broadcast
):
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

    Yields:
        bytes: Serialized FAISS index for the partition, produced by
            ``faiss.serialize_index``.
    """
    import faiss
    import numpy as np

    index = bc_index.value
    ids, vectors = [], []
    for row in iterator:
        ids.append(row["index"])
        vectors.append(list(row['_features']))
    
    if not ids:
        return # for empty partition

    ids = np.array(ids, dtype=np.int64)
    vectors = np.array(vectors, dtype=np.float32)

    index_with_ids = faiss.IndexIDMap(index)
    index_with_ids.add_with_ids(vectors, ids)

    yield faiss.serialize_index(index_with_ids)

def  _per_partition_predict(
    shard_iter: Iterable,
    bc_n_neighbors: Broadcast,
    bc_index_files_list: Broadcast,
    bc_chunk_size: Broadcast,
    bc_k: Broadcast
):
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
        bc_index_files_list (Broadcast): List of serialized index file names
            distributed via ``SparkFiles``.
        bc_chunk_size (Broadcast): Number of query rows to process per batch.
        bc_k (Broadcast): Number of IVF clusters (used to set ``nprobe``).

    Yields:
        tuple: ``(int(query_id), list[int(neighbor_ids)])`` for each query
            vector in the partition.
    """
    import faiss
    import numpy as np
    from pyspark import SparkFiles
    import gc
    import builtins

    cache = get_executor_cache()
        
    real_n = bc_n_neighbors.value
    index_files = bc_index_files_list.value
    chunk_size = bc_chunk_size.value

    def iter_chunk(it: Iterable, chunk_size: int):
        chunk = []
        amount = 0
        for row in it:
            chunk.append(row)
            amount += 1

            if amount >= chunk_size:
                amount = 0
                yield chunk
                chunk =[]
            
        if chunk:
            yield chunk
    
    for chunk in iter_chunk(shard_iter, chunk_size):
        if not chunk:
            return
        query_ids = np.array([r["index"] for r in chunk], dtype=np.int64)
        batch = np.array([list(r["_features"]) for r in chunk], dtype=np.float32)  # (Q, d)
        del chunk
        gc.collect()        

        candidates = [[] for _ in range(len(query_ids))]
        for index_file in index_files:
            tmp_index = cache.get(index_file, nprobe=min(real_n * 2, bc_k.value))
            # tmp_index.nprobe = real_n
            # tmp_index.nprobe = min(real_n * 2, bc_k.value)
            k = min(real_n, tmp_index.ntotal)
            dists, nids = tmp_index.search(batch, k)   # (Q, k)
            del tmp_index
            gc.collect()

            for q_idx in range(len(query_ids)):
                for rank in range(k):
                    nid = int(nids[q_idx, rank])
                    if nid >= 0:
                        candidates[q_idx].append((float(dists[q_idx, rank]), nid))

        for q_idx, qid in enumerate(query_ids):
            top = sorted(candidates[q_idx], key=lambda x: x[0])[:real_n]
            output = [int(nid) for _, nid in top]
            yield (int(qid), output)
            # yield (output,)

@backend_factory.register(FaissExtension, SparkDataset)
class SparkFaissExtension(FaissExtension):
    """
    Faiss backend-slave class for faiss pairs matching.

    Spark backend implementation for distributed FAISS-based nearest neighbor matching.

    This class implements a fully distributed FAISS pipeline for datasets that
    exceed single-machine memory. The pipeline consists of three phases:

    1. **Vectorization**: Feature columns are assembled into a single vector
       column using Spark's ``VectorAssembler``.
    2. **Distributed Fit**: The dataset is partitioned, and each partition
       builds a local FAISS ``IndexIDMap`` on top of a shared IVF quantizer.
       The quantizer is trained either on a random sample ("sample" mode) or
       via iterative mini-batch clustering on the full dataset ("full" mode).
    3. **Distributed Predict**: Serialized partition indexes are distributed
       to all executors via ``SparkFiles``. Each executor loads the indexes
       into a local LRU cache and performs batched nearest-neighbor searches.

    Inherits from:
        FaissExtension: The master-abstract FAISS extension class.

    Attributes:
        PERSIST_POLITIC (StorageLevel): Storage level for intermediate RDDs.
        _SAMPLE_TARGET (int): Target number of rows for sampling during
            IVF training in "sample" mode.
        DRIVER_INDEX_LIMIT (int): Maximum number of vectors to load on the
            driver per batch during iterative prefit.
        CHUNK_SIZE (int): Number of query rows processed per batch during
            distributed prediction.

    See Also:
        CachingIndex: Executor-side LRU cache for FAISS indexes.
    """
    PERSIST_POLITIC = StorageLevel.MEMORY_AND_DISK
    _SAMPLE_TARGET = 5_000_000
    # Лимит на то, сколько локальых индексов может быть одновременно загружено на драйвер
    DRIVER_INDEX_LIMIT = 5_000_000 
    PREDICT_SCHEMA = StructType([
        StructField("index",          LongType(),            False),
        StructField("index_list",     ArrayType(LongType()), False)
    ])
    CHUNK_SIZE = 512
    CLUSTERING_METHODS_MAPPER = {
        "k-means": {
            "model": MiniBatchKMeans,
            "params": {
                # "n_clusters" : 1000,
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

    def _vectorize_data(
            self, 
            data: spark.DataFrame
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
        if len(set(map(lambda x: x[1], data.dtypes)).intersection(['varchar', 'string'])) > 0:
            raise TypeError("Unencoded categorical features are not allowed!")

        vecAssembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="_features",
            handleInvalid="keep"
        )
        
        return (
                    vecAssembler
                    .transform(data)
                )

    def _prefit(self, vectorized_data: spark.DataFrame, model_name: str) -> None:
        """
        Train the IVF quantizer on the full dataset via iterative partition upload.

        Loads feature vectors from Spark partitions in batches of
        ``DRIVER_INDEX_LIMIT`` rows and incrementally fits a clustering model
        (MiniBatchKMeans or BIRCH) on the driver. The resulting cluster centers
        are used to initialize the FAISS IVF quantizer.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the
                ``_features`` vector column.
            model_name (str): Name of the clustering algorithm to use.
                Must be a key in ``CLUSTERING_METHODS_MAPPER`` (e.g., "k-means", "birch").
        """
        model_dict = self.CLUSTERING_METHODS_MAPPER[model_name]
        model_cls, model_params = model_dict["model"], model_dict["params"]
        model_params["n_clusters"] = self.k 
        model = model_cls(**model_params)

        batch_size = self.DRIVER_INDEX_LIMIT
        np_batch = None

        for batch in (
            vectorized_data
            .select("_features")
            .rdd
            .mapPartitions(lambda it: _partition_load(it, batch_size))
            .toLocalIterator()
        ):
            np_batch = np.array(batch, dtype=np.float32)
            model.partial_fit(np_batch)
            # self._index.train(np_batch)

        
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
    
    def _fit(
            self, 
            vectorized_data: spark.DataFrame, 
            mode: Literal["sample", "full"],
            model_name: str | None
    ) -> "SparkFaissExtension":
        """
        Build distributed FAISS indexes across Spark partitions.

        Two training modes are supported:
        - **"sample"**: Trains the IVF quantizer on a random sample of the data
          (up to ``_SAMPLE_TARGET`` rows). Faster but may produce less accurate
          clusters for non-uniform distributions.
        - **"full"**: Trains the IVF quantizer on the entire dataset using
          iterative mini-batch clustering via ``_prefit``. Slower but more
          accurate.

        After training, each partition builds a local ``IndexIDMap`` on top of
        the shared quantizer, and the serialized indexes are persisted as an RDD.

        Args:
            vectorized_data (spark.DataFrame): Input DataFrame with the
                ``_features`` vector column.
            mode (Literal["sample", "full"]): IVF training algorithm.
                Defaults to "sample".
            model_name (str | None): Clustering model name for "full" mode
                (e.g., "k-means", "birch"). Ignored in "sample" mode.

        Returns:
            SparkFaissExtension: Self, for method chaining.

        Raises:
            ValueError: If ``mode`` is not "sample" or "full".
        """
        session = vectorized_data.sparkSession
        m = 4 # heuristic
        self.k = int(np.sqrt(vectorized_data.count() / m))

        if mode =="sample":
            data_size = vectorized_data.count()
            frac = min(self._SAMPLE_TARGET / max(data_size, 1), 1.0)
            sample_rows = (
                            vectorized_data
                            .sample(fraction=frac, seed=self.seed)
                            .select("_features")
                            .collect()
                        )
            
            X = np.array(
                [list(row['_features']) for row in sample_rows],
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
                vectorized_data=vectorized_data,
                model_name=model_name
            )
        
        else:
            raise ValueError(f"Incorrect faiss fit mode: '{mode}'")
        self.index.nprobe = min(self.n_neighbors * 2, self.k)

        bc_index = session.sparkContext.broadcast(self.index)
        del self.index
        self.index = None
        gc.collect()

        features = ["index", "_features"]
        self._sharded_rdd = (
            vectorized_data
            .select(*features)
            .rdd
            .mapPartitions(lambda it: _spark_partition_fit(it, bc_index))
            .persist(self.PERSIST_POLITIC)
        )
        self._sharded_rdd.count()
    
    def _predict(
            self, 
            test_data: spark.DataFrame, 
            storage_level: Literal[
                    "MEMORY_ONLY", "MEMORY_AND_DISK", "DISK_ONLY",
            ] | None
        ):
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

            storage_level (Literal): Storage strategy for cached. Use similar option 
                as input `data`.

        Returns:
            Dataset: A Dataset containing the matched neighbor indices, indexed
                by the original row index.
        """
        session = test_data.sparkSession
        
        # tmp_dir = f"__partition_indexes"
        # os.makedirs(tmp_dir, exist_ok=True) 
        tmp_dir = tempfile.mkdtemp(dir=".")
        result = Dataset.create_empty(session=session)
        try:
            index_files_list = []
            

            for partition_index, shard in enumerate(self._sharded_rdd.toLocalIterator()):
                partition_indexes = faiss.deserialize_index(shard)
                run_id = uuid.uuid1().hex[:8]
                index_file_name = f"__{partition_index}_partition_index_{run_id}.index"
                faiss.write_index(
                    partition_indexes,
                    f"{tmp_dir}/{index_file_name}" 
                )    
                session.sparkContext.addFile(f"{tmp_dir}/{index_file_name}")
                index_files_list.append(index_file_name)

                del partition_indexes   # ← explicit release
                gc.collect()
            
            self._sharded_rdd.unpersist()
            self._sharded_rdd = None
            # session.sparkContext.addPyFile("index_cacher.py")
            bc_index_files_list = session.sparkContext.broadcast(index_files_list)
            bc_n_neighbors = session.sparkContext.broadcast(self.n_neighbors)
            bc_chunk_size = session.sparkContext.broadcast(self.CHUNK_SIZE)
            bc_k = session.sparkContext.broadcast(self.k)

            result_rdd = test_data.rdd.mapPartitions(lambda it:
                                            _per_partition_predict(
                                            it, 
                                            bc_n_neighbors=bc_n_neighbors, 
                                            bc_index_files_list=bc_index_files_list,
                                            bc_chunk_size=bc_chunk_size,
                                            bc_k=bc_k
                )
            )

            result_df = (
                session.createDataFrame(result_rdd, schema=self.PREDICT_SCHEMA)
                .select(
                    ['index'] + 
                    [F.expr(f"index_list[{i}]").alias(f"{i + 1}") for i in range(self.n_neighbors)]
                )
                # .persist(self.PERSIST_POLITIC)
            )
            # result_df.count()
            result = self.result_to_dataset(result=result_df, roles={}, small=False).set_index('index')
            result.index.name = None

            storage_level = storage_level or "MEMORY_AND_DISK"
            result.persist(storage_level=storage_level, action="count")
        finally:
            # Удаляем все созданные промежуточные файлы
            # tmp_files = os.listdir(tmp_dir)
            # for file in tmp_files:
            #     os.remove(f"{tmp_dir}/{file}")
            # os.rmdir(tmp_dir)
            shutil.rmtree(tmp_dir)

        return result

    
    def calc(
            self, 
            data, 
            test_data = None,
            mode = None, 
            **kwargs
    ):
        """
        Execute the distributed FAISS matching pipeline for Spark-backed datasets.

        Orchestrates the vectorization, fit, and predict phases based on the
        ``mode`` argument. If a Mahalanobis matrix is provided, features are
        projected before vectorization.

        Args:
            data: The baseline (control) dataset.
            test_data: The query (treatment) dataset. Required for "predict"
                and "auto" modes.
            mode (str, optional): Operation mode ("auto", "fit", "predict").
                Defaults to None (treated as "auto").
            **kwargs: Additional keyword arguments, including:
                - ``fit_mode`` (str): "sample" or "full" for IVF training.
                  Defaults to "sample".
                - ``model`` (str): Clustering model for "full" mode.
                  Defaults to "k-means".

        Returns:
            SparkFaissExtension or Dataset: The fitted extension (for "fit" mode)
                or the matched indices Dataset (for "predict"/"auto" modes).

        Raises:
            ValueError: If ``test_data`` is None when prediction is required,
                or if the sharded RDD has not been built before prediction.
        """
        mode = mode or "auto"
        operating_data: spark.DataFrame = (
            data._backend_data.data.to_spark(index_col='index')
            if self.mahalonobis is None
            else data.dot(self.mahalonobis)._backend_data.data.to_spark(index_col='index')
        )
        self.k = (operating_data.count())
        vectorized_data = self._vectorize_data(operating_data)

        if mode in ["auto", "fit"]:
            fit_mode = kwargs.get("fit_mode", "sample")
            model_name = kwargs.get("model", "k-means")
            self._fit(
                vectorized_data=vectorized_data,
                mode=fit_mode,
                model_name=model_name
            )

        if mode in ["auto", "predict"]:
            if test_data is None:
                raise ValueError("test_data is needed for evaluation")
            if self._sharded_rdd is None:
                raise ValueError("Index is not created yet. Call 'fit' before 'predict'.")
            
            test_operating_data = (
                test_data._backend_data.data.to_spark(index_col='index')
                if self.mahalonobis is None
                else test_data.dot(self.mahalonobis)._backend_data.data.to_spark(index_col='index')
            )
            vectorized_test = self._vectorize_data(test_operating_data)

            return self._predict(vectorized_test, data.get_storage_level())
    
    def unpersist(self) -> None:
        """
        Release Spark resources held by this extension.

        Unpersists the sharded RDD and any clustered data, freeing executor
        memory and disk. Should be called when the extension is no longer needed
        to avoid resource leaks in long-running Spark applications.
        """
        if self._clustered_data is not None:
            self._clustered_data.unpersist()
            self._clustered_data = None
            
        if self._sharded_rdd is not None:
            self._sharded_rdd.unpersist()
            self._sharded_rdd = None
    
    def __enter__(self) -> "SparkFaissExtension":
        return self
        
    def __del__(self, *_) -> None:
        self.unpersist()

# TODO: Проверить, нужна ли вообще эта фича?
class CachingIndex:
    """
    Класс для кэширования индекса на экзекъюторе.
    Предназначен для того, чтобы в памяти экзекъютора, при одновременном 
    выполнении нескольких партиций не происходило ситуации, один и тот же индекс
    train-data-ы не материализовывался несколько раз
    """

    def __init__(
        self,
        max_index: int=2
    ):
        """
        Args
        ----
            k : `float`
                доля от overhead memory, которую мы позволяем использовать для подгрузки индексов.

            executor_cores : `int`
                количество ядер на экзеъюторе.
            
            overhead_memory : `int`
                оверхед экзекъютора, память вне кучи JVM.
            
            index_bytes : `int`
                Размер индекса одной партиции в мегабайтах, по-умолчанию = 128 мб.

            max_index : `int`
                Колчиество индексов в памяти одновременно, если None, то вычисляется автоматически. 
        """        
        self._max = max_index
        self._cache = OrderedDict()
        self._lock = threading.Lock()
    
    def get(
            self,
            index_file: int,
            nprobe: int
    ):
        """
        Получаем индексы по заданному названию файла. Если такой файл уже обрабатывался,
        то просто выгружаем его из словаря и двигаем в последовательности ключей в конец.
        если такого файла нет в нашем кэше, то очищаем первый элемент и записываем в конец
        новый индекс.

        Args
        ----
            index_file : `str`
                Путь до файла с индексами, который выступает ключем.

        Return
        ------
            Возвращает FAISS индексы для заданного файла.
        """
        with self._lock:
            if index_file in self._cache:
                self._cache.move_to_end(key=index_file)
                return self._cache[index_file]

            if len(self._cache) == self._max:
                _, evicted = self._cache.popitem(last=False)
                del evicted
                import gc; gc.collect()
            
            tmp_index = faiss.read_index(SparkFiles.get(index_file))
            inner = faiss.downcast_index(tmp_index.index)
            inner.nprobe = nprobe  
            self._cache[index_file] = tmp_index
            return tmp_index
        
def get_executor_cache() -> CachingIndex:
    if not hasattr(builtins, '_faiss_index_cache'):
        builtins._faiss_index_cache = CachingIndex()
    return builtins._faiss_index_cache
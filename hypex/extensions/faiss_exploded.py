from __future__ import annotations

from typing import (
    Literal,
    Iterable
)

from ..dataset import AdditionalMatchingRole, Dataset
from .abstract import MLExtension
from abc import abstractmethod
from ..utils.errors import AbstractMethodError
from ..utils.registry import backend_registry
from ..dataset.backends import PandasDataset, SparkDataset

import numpy as np
import pandas as pd
import faiss
import gc
import os

# Spark imports
import pyspark.sql as spark
import pyspark.pandas as ps
ps.set_option('compute.ops_on_diff_frames', True)

from pyspark.ml.feature import VectorAssembler
from pyspark import StorageLevel, Broadcast, RDD

from pyspark.sql.types import (
    StructType, 
    StructField, 
    LongType,
    FloatType
)
from sklearn.cluster import MiniBatchKMeans, Birch

class FaissExtention(MLExtension):

    def __init__(
        self, n_neighbors: int = 1, faiss_mode: Literal["base", "fast", "auto"] = "auto"
    ):
        self.n_neighbors = n_neighbors
        self.faiss_mode = faiss_mode
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
        raise AbstractMethodError

    def fit(self, X: Dataset, Y: Dataset | None = None, **kwargs):
        return super().calc(X, target_data=Y, mode="fit", **kwargs)

    def predict(self, X: Dataset, **kwargs) -> Dataset:
        return self.result_to_dataset(
            super().calc(X, mode="predict", **kwargs), AdditionalMatchingRole()
        )

@backend_registry.registry(FaissExtention, PandasDataset)
class PandasFaissExtension(FaissExtention):

    def __init__(self, n_neighbors = 1, faiss_mode = "auto"):
        super().__init__(n_neighbors, faiss_mode)

    @staticmethod
    def _prepare_indexes(index: np.ndarray, dist: np.ndarray, k: int):
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
        return self.result_to_dataset(result=indexes, roles={})

    def _fit(
            self, 
            X: np.ndarray, 
            test: np.ndarray
    ) -> None:
        """
        """
        self.index = faiss.IndexFlatL2(X.shape[1])
        if (
            (
                (len(X) > 1_000_000 and self.faiss_mode == "auto")
                or self.faiss_mode == "fast"
            )
            and len(X) > 1_000
            and len(test) > 1_000
        ):
            m = 4
            n_clusters = np.sqrt(len(X) / m)
            self.index = faiss.IndexIVFFlat(self.index, X.shape[1], n_clusters)
            self.index.train(X)
        self.index.add(X)
        
    def calc(
            self, 
            data, 
            test_data = None, 
            mode = None, 
            **kwargs
    ):
        mode = mode or "auto"
        X = data.data.values
        test = test_data.data.values
        if mode in ["auto", "fit"]:
            self._fit(X, test)
        if mode in ["auto", "predict"]:
            if test_data is None:
                raise ValueError("test_data is needed for evaluation")
            if self.index is None:
                raise ValueError("index is not created yet. Raise 'fit' before 'predict'.")

            X = test_data.data.values if mode == "auto" else data.data.values
            return self._predict(data, test_data, X)
        return self


# global functions for pyspark partition logic.
def _partition_load(partition_iter: Iterable, batch_size: int):
    """
    Load part batch from partition to driver.

    Args
    ----
        partition_iter: `Iterable`
            iterator over partition.

        batch_size: `int`
            size of uploading batch from partition.
    """
    batch = []
    for row in partition_iter:
        batch.append(list(row["features"]))
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
    Local fit on each partition.

    Args
    ----
        iterator: Iterable
            iterator inside partition. 
        
        bc_index: Broadcast
            broadcasted IVF-index.
    
    Return
    ------
        `Generator`: serialized index from each partition.
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
        Predict локально на каждой партиции.

        Из загруженных `.index` файлов итеративно создаем для каждой строки в датафрейме
        новую колонку, где будут находится ближайшие соседи.

        Args
        ----
            shard_iter: `Iterable`
                Итератор по партиции.

            bc_n_neighbors: `Broadcast`
                Количество соседей для поиска.

            bc_index_files_list: `Broadcast`
                Список из номеров файлов, где хранятся построенные индексы каждой партиции.  

            bc_config_dict: `Broadcast`
                Словарь с конфигом сессии.
        """
        import faiss
        import numpy as np
        from pyspark import SparkFiles
        import gc
        import builtins
        from index_cacher import get_executor_cache

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
            query_ids = np.array([r["_id"] for r in chunk], dtype=np.int64)
            batch = np.array([list(r["features"]) for r in chunk], dtype=np.float32)  # (Q, d)
            del chunk
            gc.collect()        

            candidates = [[] for _ in range(len(query_ids))]
            for index_file in index_files:
                tmp_index = cache.get(index_file)
                # tmp_index.nprobe = real_n
                tmp_index.nprobe = min(bc_k.value * 2, real_n)
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
                for dist, nid in top:
                    yield (int(qid), nid, dist)

@backend_registry.registry(FaissExtention, SparkDataset)
class SparkFaissExtension(FaissExtention):

    PERSIST_POLITIC = StorageLevel.MEMORY_AND_DISK
    _SAMPLE_TARGET = 5_000_000
    # Лимит на то, сколько локальых индексов может быть одновременно загружено на драйвер
    DRIVER_INDEX_LIMIT = 5_000_000 
    PREDICT_SCHEMA = StructType([
        StructField("index",          LongType(),  False),
        StructField("neighbor_index", LongType(),  False),
        StructField("distance",       FloatType(), False),
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
            faiss_mode = "auto"
    ):
        super().__init__(n_neighbors, faiss_mode)

    def _vectorize_data(
            self, 
            data: spark.DataFrame
    ) -> spark.DataFrame:
        """
        Input data preparation: vectorization feature columns and categorical columns check.

        Agrs
        ----------
            data : `SparkDataFrame`
                input dataframe. Should contain:
                    - numeric features;
                    - encoded categorical features;
        
        Returns
        -------
            vectorized_data: `SparkDataFrame`
                simmilar as input data with target features column vector.
        """
        if self.feature_cols is None:
            self.feature_cols = list(set(data.columns) - set('index'))
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
        Train IVF-index on full dataset using iterative partition upload on driver.
        
        Args
        ----
            vectorized_data : `spark.DataFrame`
                input vectorized dataframe.

        """
        model_dict = self.CLUSTERING_METHODS_MAPPER[model_name]
        model_cls, model_params = model_dict["model"], model_dict["params"]
        model_params["n_clusters"] = self.k 
        model = model_cls(**model_params)

        batch_size = self.DRIVER_INDEX_LIMIT
        np_batch = None

        for batch in (
            vectorized_data
            .select("features") #TODO: тут или ранее сделать обработку None
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
        index_shape = centroids.shape[1] #TODO: тут или ранее сделать обработку None 
        # nlist = min(len(centroids), max(1, data.count() // 39)) 
        nlist = len(centroids)

        quantizer = faiss.IndexFlatL2(index_shape)
        quantizer.add(centroids)
        self._index = faiss.IndexIVFFlat(quantizer, index_shape, nlist)
        self._index.is_trained = True

        self._clustering_model = model
    
    def _fit(
            self, 
            vectorized_data: spark.DataFrame, 
            mode: Literal["sample", "full"],
            model_name: str | None
    ) -> "SparkFaissExtension":
        """
        Realization of faiss indexes creation on partitions.
        Two options are implemented:
            - train IVF-index on sample of data;
            - train IVF-index on full data using iterative prefit;
        
        Args
        ----
            vectorized_data: `spark.DataFrame`
                input vectorized data.

            mode: `Literal["sample", "full"]`
                IVF-index training algorithm. Default mode is`sample`.
        Return
        ------
            `SparkFaissExtension`
        """
        session = vectorized_data.sparkSession
        self.k = vectorized_data.count()

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
        
        bc_index = session.sparkContext.broadcast(self._index)
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
    
    def _predict(self, test_data: spark.DataFrame):
        """
        Steps
        -----
        1. Итеративно получаем сериализованные индексы с каждой партиции и
        записывем их в файл с расширегием `.index`, которое умеет обрабатывать faiss.

        2. Отправляем `.index` файлы на все партиции test-data, для которой ищем соседей
        в train-data.

        3. На каждой партиции итеративно загружаем в RAM строку test-data, 
        и для нее итеративно подгружаем файл с индексами, где ищем top_k = 
        n_neighbors, после чего удаляем индекс из оперативной памяти.

        4. Результаты обернуты в `Spark.Dataframe` с схемой 
        (_id `LongType`, neighbor_id `LongType`, distance `FloatType`).

        5. Удаление временных файлов и временной дирекятории после материализации
        результатов в датафрейм.

        Args
        ----
            test_data : `spark.DataFrame`
                Данные для которых мы ищем соседей.

        Return
        ------
            result: `RDD` результирующая таблица с индексами соседей и расстоянием до них
        """

        session = test_data.sparkSession
        tmp_dir = "__partition_indexes"
        os.makedirs(tmp_dir, exist_ok=True) 
        index_files_list = []

        for partition_index, shard in enumerate(self._sharded_rdd.toLocalIterator()):
            partition_indexes = faiss.deserialize_index(shard)
            index_file_name = f"__{partition_index}_partition_index.index"
            faiss.write_index(
                partition_indexes,
                f"./{tmp_dir}/{index_file_name}" 
            )    
            session.sparkContext.addFile(f"./{tmp_dir}/{index_file_name}")
            index_files_list.append(index_file_name)

            del partition_indexes   # ← explicit release
            gc.collect()
        
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
            .persist(self.PERSIST_POLITIC)
        )
        result_df.count()
        
        # Удаляем все созданные промежуточные файлы
        tmp_files = os.listdir(tmp_dir)
        for file in tmp_files:
            os.remove(f"{tmp_dir}/{file}")
        os.rmdir(tmp_dir)

        return result_df 

    
    def calc(
            self, 
            data, 
            test_data = None,
            mode = None, 
            **kwargs
    ):
        mode = mode or "auto"
        operating_data: spark.DataFrame = data._backend_data.data.to_spark(index_col='index')
        self.k = operating_data.count()
        transformed_data = self._vectorize_data(operating_data)

        if mode in ["auto", "fit"]:
            fit_mode = kwargs.get("fit_mode", "sample")
            model_name = kwargs.get("model", "k-means")
        
        if mode in ["auto", "predict"]:
            ...
    
    def unpersist(self) -> None:
        """
        Подчистить ресурсы spark.
        """
        if self._clustered_data is not None:
            self._clustered_data.unpersist()
            self._clustered_data = None
            
        if self._sharded_rdd is not None:
            self._sharded_rdd.unpersist()
            self._sharded_rdd = None
    
    def __enter__(self) -> "SparkFaissExtension":
        return self
        
    def __exit__(self, *_) -> None:
        self.unpersist()
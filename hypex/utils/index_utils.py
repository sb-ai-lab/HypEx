from __future__ import annotations

import gc
import os
import shutil
import threading
import uuid
from collections import OrderedDict
from typing import Optional

import faiss
from pyspark import RDD, SparkFiles
from pyspark.sql import SparkSession


class FaissIndexStorage:
    """
    Faiss file manager.

    Simplified version: only the local file system is used.
    and the Spark file distribution mechanism (SparkFiles.addFile).

    This avoids problems with connecting to distributed FS
    (HDFS/viewfs/WebHDFS) on corporate clusters.
    """
    # DISTRIBUTED_DIRS оставлен для совместимости
    DISTRIBUTED_DIRS = []
    LOCAL_DIRS = []

    def __init__(
        self,
        sp_s: SparkSession,
        base_dir: str | None = None  # сохранён для совместимости, не используется
    ):
        self.sp_s = sp_s
        # для совместимости с внешним кодом
        self._distributed = False
        self._distributed_dir = None

        dir_id = uuid.uuid1().hex[:8]
        self._local_tmp_dir = f"__partition_indexes_{dir_id}"
        os.makedirs(self._local_tmp_dir, exist_ok=True)
        FaissIndexStorage.LOCAL_DIRS.append(self._local_tmp_dir)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "sp_s" in state:
            del state["sp_s"]
        return state

    def save_index(self, index: faiss.Index) -> bytes:
        """
        Serialize index to bytes for sending to executors.
        Raisses on executors.

        Args
        ----
            index: faiss.Index
                partition faiss index.
        Returns
        -------
            serialized index in bytes.
        """
        return faiss.serialize_index(index)

    def collect_and_register(self, rdd: RDD) -> list:
        """
        Collects serialized indexes from executors,
        saves them to temporary files and distributes them via SparkFiles.
        It is called on the driver.

        Args
        ----
            rdd: RDD
                ...

        Return
        ------
            list of index references.
        """
        index_refs = []
        for shard in rdd.toLocalIterator():
            partition_indexes = faiss.deserialize_index(shard)
            run_id = uuid.uuid1().hex[:8]
            index_file_name = f"__partition_index_{run_id}.index"
            faiss.write_index(
                partition_indexes,
                f"{self._local_tmp_dir}/{index_file_name}"
            )
            self.sp_s.sparkContext.addFile(f"{self._local_tmp_dir}/{index_file_name}")
            index_refs.append(index_file_name)
            del partition_indexes  # явное освобождение памяти
            gc.collect()
        return index_refs

    def load_index(self, link: str) -> faiss.Index:
        """
        Loads the index distributed through SparkFiles.
        It is invoked on executors.

        Args
        ----
            link: str
                link to index file.

        Return
        ------
            faiss index loaded from file.
        """
        return faiss.read_index(SparkFiles.get(link))

    @staticmethod
    def cleanup():
        """Deletes all temporary local directories."""
        for dir in FaissIndexStorage.LOCAL_DIRS:
            if os.path.exists(dir):
                try:
                    shutil.rmtree(dir)
                except Exception:
                    pass
        FaissIndexStorage.LOCAL_DIRS = []


class CachingIndex:
    """
    LRU is the cache of FAISS indexes in the executor's memory.
    Prevents repeated loading of the same index
    when processing multiple batches.
    """
    def __init__(self, max_index: Optional[int] = None):
        self._max = max_index
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(
        self,
        reference: str,
        storage: FaissIndexStorage,
        nprobe: int,
    ):
        with self._lock:
            if reference in self._cache:
                self._cache.move_to_end(key=reference)
                return self._cache[reference]
            if self._max and len(self._cache) >= self._max:
                _, evicted = self._cache.popitem(last=False)
                del evicted
                gc.collect()
            tmp_index = storage.load_index(reference)
            inner = faiss.downcast_index(tmp_index)
            if hasattr(inner, "nprobe"):
                inner.nprobe = nprobe
            self._cache[reference] = tmp_index
            return tmp_index

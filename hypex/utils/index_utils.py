from __future__ import annotations

import fsspec
import uuid
import faiss
import os
import gc
import tempfile
import threading
import shutil

from typing import Optional
from collections import OrderedDict

from pyspark.sql import SparkSession
from pyspark import RDD, SparkFiles

class FaissIndexStorage:
    DISTRIBUTED_DIRS = []
    LOACL_DIRS = []

    def __init__(
        self,
        sp_s: SparkSession,
        base_dir: Optional[str] = None 
    ):
        self.sp_s = sp_s
        
        self._distributed = False
        self._distributed_dir = None
        self._local_tmp_dir = None

        default_fs = self.define_fs(sp_s)

        if default_fs.startswith("file"):
            dir_id = uuid.uuid1().hex[:8]
            self._local_tmp_dir = f"__partition_indexes{dir_id}"
            os.makedirs(self._local_tmp_dir, exist_ok=True)

            FaissIndexStorage.LOACL_DIRS.append(self._local_tmp_dir)
        else:
            self._distributed = True
            self._distributed_dir = (
                base_dir  or
                f"{default_fs.rstrip('/')}/tmp/faiss_indexes_{uuid.uuid1().hex[:8]}"
            )
            FaissIndexStorage.DISTRIBUTED_DIRS.append(self._distributed_dir)

    @staticmethod
    def define_fs(session: SparkSession) -> str:
        # First approach
        try:
            hadoop_conf = session.sparkContext._jsc.hadoopConfiguration()
            file_sys = hadoop_conf.get("fs.defaultFS", "file:///")
            if file_sys != "file:///":
                return file_sys
        except:
            pass

        # Second appraoch
        try:
            return session.conf.get(
                "spark.hadoop.fs.defaultFS",
                session.conf.get("fs.defaultFS", "file:///")
            )
        except Exception:
            return "file:///"

    def __getstate__(self):
        state = self.__dict__.copy()
        if "sp_s" in state:
            del state["sp_s"]
        return state

    def save_index(self, index: faiss.Index):
        if self._distributed:
            uri = os.path.join(
                self._distributed_dir, f"index_{uuid.uuid1().hex[:8]}.faiss"
            )
            with fsspec.open(uri, "wb").open() as f:
                faiss.write_index(index, faiss.PyCallbackIOWriter(f.write))
            return uri
        else:
            return faiss.serialize_index(index)

    def collect_and_register(self, rdd: RDD):
        index_refs = []
        if self._distributed:
            index_refs = rdd.collect()
        else:
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
    
                del partition_indexes   # ← explicit release
                gc.collect()
        return index_refs

    def load_index(self, link: str) -> faiss.Index:
        if self._distributed:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".faiss"
            ) as temp:
                load_temp = temp.name
            try:
                with fsspec.open(link, "rb").open() as f_in,\
                     open(load_temp, "wb") as f_out:
                    while True:
                        chunk = f_in.read(64 * 1024 * 1024)  # 64 MB chunks
                        if not chunk:
                            break
                        f_out.write(chunk)
                index = faiss.read_index(load_temp)
            finally:
                if os.path.exists(load_temp):
                    os.remove(load_temp)
            return index
        else:
            return faiss.read_index(SparkFiles.get(link))

    @staticmethod
    def cleanup():
        if FaissIndexStorage.DISTRIBUTED_DIRS:
            for dir in FaissIndexStorage.DISTRIBUTED_DIRS:
                try:
                    fs, _ = fsspec.core.url_to_fs(
                        dir, use_listings_cache=False
                    )
                    fs.rm(dir, recursive=True)
                except Exception:
                    pass
            FaissIndexStorage.DISTRIBUTED_DIRS = []

        if FaissIndexStorage.LOACL_DIRS:
            for dir in FaissIndexStorage.LOACL_DIRS:
                if os.path.exists(dir):
                    shutil.rmtree(dir)
            FaissIndexStorage.LOACL_DIRS = []

class CachingIndex:
    """
    LRU-кэш FAISS-индексов в памяти экзекьютора.
    Предотвращает повторную загрузку одного и того же индекса
    при обработке нескольких партиций.
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
                import gc; gc.collect()

            tmp_index = storage.load_index(reference)
            inner = faiss.downcast_index(tmp_index)
            if hasattr(inner, "nprobe"):
                inner.nprobe = nprobe

            self._cache[reference] = tmp_index
            return tmp_index

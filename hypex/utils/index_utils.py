"""Utilities for FAISS index storage and caching in distributed Spark environments.

This module provides two core components:

- :class:`FaissIndexStorage`: Manages serialization, persistence, and distribution
  of FAISS indexes across Spark executors using the local file system and
  ``SparkFiles``.
- :class:`CachingIndex`: A thread-safe LRU cache that prevents redundant
  deserialization of FAISS indexes when processing multiple query batches
  on the same executor.
"""

from __future__ import annotations

import gc
import os
import shutil
import threading
import uuid
from collections import OrderedDict
from typing import Any, ClassVar

import faiss  # pyright: ignore[reportMissingImports]
from pyspark import RDD, SparkFiles  # pyright: ignore[reportMissingImports]
from pyspark.sql import SparkSession  # pyright: ignore[reportMissingImports]


class FaissIndexStorage:
    """Manages FAISS index files for distributed Spark processing.

    Uses only the local file system combined with Spark's file distribution
    mechanism (``SparkFiles.addFile``) to avoid connectivity issues with
    distributed file systems (HDFS/viewfs/WebHDFS) on corporate clusters.

    Attributes:
        DISTRIBUTED_DIRS: Retained for backward compatibility. Always empty.
        LOCAL_DIRS: List of temporary local directories created by instances
            of this class. Used by :meth:`cleanup` for resource release.

    Args:
        sp_s: Active Spark session used for file distribution.
        base_dir: Retained for backward compatibility. Not used.

    Example:
        >>> storage = FaissIndexStorage(spark_session)
        >>> refs = storage.collect_and_register(sharded_rdd)
        >>> index = storage.load_index(refs[0])  # on executor
    """

    DISTRIBUTED_DIRS: ClassVar[list[str]] = []
    LOCAL_DIRS: ClassVar[list[str]] = []

    def __init__(
        self,
        sp_s: SparkSession,
        base_dir: str | None = None,
    ) -> None:
        self.sp_s: SparkSession = sp_s
        # Retained for backward compatibility with external code.
        self._distributed: bool = False
        self._distributed_dir: str | None = None

        dir_id: str = uuid.uuid1().hex[:8]
        self._local_tmp_dir: str = f"__partition_indexes_{dir_id}"
        os.makedirs(self._local_tmp_dir, exist_ok=True)
        FaissIndexStorage.LOCAL_DIRS.append(self._local_tmp_dir)

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the non-serializable SparkSession before pickling.

        Returns:
            Instance state dictionary without the ``sp_s`` key.
        """
        state: dict[str, Any] = self.__dict__.copy()
        state.pop("sp_s", None)
        return state

    def save_index(self, index: faiss.Index) -> bytes:
        """Serialize a FAISS index to bytes for transmission to executors.

        This method is invoked on executors during the distributed fit phase.

        Args:
            index: The partition-level FAISS index to serialize.

        Returns:
            Serialized index as a bytes object.
        """
        return faiss.serialize_index(index)

    def collect_and_register(self, rdd: RDD) -> list[str]:
        """Collect serialized indexes from executors and distribute via SparkFiles.

        Iterates over the RDD produced by the distributed fit phase,
        deserializes each partition index, writes it to a temporary local
        file, and registers the file with ``SparkFiles`` so that executors
        can retrieve it during the predict phase.

        This method must be called on the driver.

        Args:
            rdd: RDD containing serialized FAISS indexes (one per partition).

        Returns:
            List of index file reference names suitable for
            :meth:`load_index`.
        """
        index_refs: list[str] = []
        for shard in rdd.toLocalIterator():
            partition_index: faiss.Index = faiss.deserialize_index(shard)
            run_id: str = uuid.uuid1().hex[:8]
            index_file_name: str = f"__partition_index_{run_id}.index"
            faiss.write_index(
                partition_index,
                f"{self._local_tmp_dir}/{index_file_name}",
            )
            self.sp_s.sparkContext.addFile(
                f"{self._local_tmp_dir}/{index_file_name}"
            )
            index_refs.append(index_file_name)
            del partition_index
            gc.collect()
        return index_refs

    def load_index(self, link: str) -> faiss.Index:
        """Load a FAISS index distributed through SparkFiles.

        This method is invoked on executors during the predict phase.

        Args:
            link: File reference name returned by :meth:`collect_and_register`.

        Returns:
            The deserialized FAISS index.
        """
        return faiss.read_index(SparkFiles.get(link))

    @staticmethod
    def cleanup() -> None:
        """Delete all temporary local directories created by FaissIndexStorage.

        Iterates over :attr:`LOCAL_DIRS`, removes each directory if it
        exists, and resets the list. Exceptions during removal are
        silently suppressed.
        """
        for directory in FaissIndexStorage.LOCAL_DIRS:
            if os.path.exists(directory):
                try:
                    shutil.rmtree(directory)
                except Exception:
                    pass
        FaissIndexStorage.LOCAL_DIRS = []


class CachingIndex:
    """Thread-safe LRU cache for FAISS indexes on Spark executors.

    Prevents repeated deserialization of the same index file when
    processing multiple query batches within a single executor process.
    The cache is stored as a module-level singleton via
    :func:`hypex.extensions.faiss.get_executor_cache`.

    Args:
        max_index: Maximum number of indexes to hold in cache.
            When the limit is reached, the least-recently-used entry
            is evicted. If ``None``, the cache grows without bound.

    Example:
        >>> cache = CachingIndex(max_index=4)
        >>> index = cache.get("partition_index_0.index", storage, nprobe=8)
    """

    def __init__(self, max_index: int | None = None) -> None:
        self._max: int | None = max_index
        self._cache: OrderedDict[str, faiss.Index] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()

    def get(
        self,
        reference: str,
        storage: FaissIndexStorage,
        nprobe: int,
    ) -> faiss.Index:
        """Retrieve a FAISS index from cache or load it from storage.

        If the index identified by ``reference`` is already cached, it is
        moved to the most-recently-used position and returned immediately.
        Otherwise, the index is loaded via ``storage.load_index``, configured
        with the given ``nprobe``, and inserted into the cache. If the cache
        has reached its capacity limit, the least-recently-used entry is
        evicted first.

        Args:
            reference: File reference name for the index.
            storage: The :class:`FaissIndexStorage` instance used to load
                the index if it is not cached.
            nprobe: Number of IVF clusters to probe during search.
                Applied to the inner index if it supports the attribute.

        Returns:
            The loaded or cached FAISS index ready for search.
        """
        with self._lock:
            if reference in self._cache:
                self._cache.move_to_end(key=reference)
                return self._cache[reference]

            if self._max is not None and len(self._cache) >= self._max:
                _, evicted = self._cache.popitem(last=False)
                del evicted
                gc.collect()

            tmp_index: faiss.Index = storage.load_index(reference)
            inner: faiss.Index = faiss.downcast_index(tmp_index)
            if hasattr(inner, "nprobe"):
                inner.nprobe = nprobe

            self._cache[reference] = tmp_index
            return tmp_index

"""Tests for FaissIndexStorage and CachingIndex (no Spark job required)."""
from __future__ import annotations

import os
import pickle

import faiss
import numpy as np
import pytest

from hypex.utils.index_utils import CachingIndex, FaissIndexStorage


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """FaissIndexStorage creates directories relative to the CWD."""
    monkeypatch.chdir(tmp_path)
    previous = list(FaissIndexStorage.LOCAL_DIRS)
    yield tmp_path
    FaissIndexStorage.cleanup()
    FaissIndexStorage.LOCAL_DIRS = previous


class _FakeSession:
    """Minimal stand-in for SparkSession (only referenced, never used)."""

    sparkContext = None


def _flat_index(dim: int = 3, n: int = 5) -> faiss.Index:
    index = faiss.IndexFlatL2(dim)
    index.add(np.arange(n * dim, dtype="float32").reshape(n, dim))
    return index


def _ivf_index(dim: int = 4) -> faiss.Index:
    rng = np.random.RandomState(0)
    data = rng.rand(200, dim).astype("float32")
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 4)
    index.train(data)
    index.add(data)
    return index


# ---------------------------------------------------------------------------
# FaissIndexStorage
# ---------------------------------------------------------------------------
def test_storage_creates_local_directory_and_registers_it(isolated_cwd) -> None:
    storage = FaissIndexStorage(_FakeSession())
    assert os.path.isdir(storage._local_tmp_dir)
    assert storage._local_tmp_dir.startswith("__partition_indexes_")
    assert storage._local_tmp_dir in FaissIndexStorage.LOCAL_DIRS


def test_each_storage_gets_its_own_directory() -> None:
    a, b = FaissIndexStorage(_FakeSession()), FaissIndexStorage(_FakeSession())
    assert a._local_tmp_dir != b._local_tmp_dir


def test_backward_compatible_attributes() -> None:
    storage = FaissIndexStorage(_FakeSession(), base_dir="ignored")
    assert storage._distributed is False
    assert storage._distributed_dir is None
    assert FaissIndexStorage.DISTRIBUTED_DIRS == []


def test_pickling_drops_spark_session() -> None:
    storage = FaissIndexStorage(_FakeSession())
    state = storage.__getstate__()
    assert "sp_s" not in state
    assert state["_local_tmp_dir"] == storage._local_tmp_dir
    restored = pickle.loads(pickle.dumps(storage))
    assert not hasattr(restored, "sp_s")
    assert restored._local_tmp_dir == storage._local_tmp_dir


def test_save_index_roundtrips_through_faiss_serialization() -> None:
    storage = FaissIndexStorage(_FakeSession())
    original = _flat_index()
    restored = faiss.deserialize_index(storage.save_index(original))
    assert restored.ntotal == original.ntotal and restored.d == original.d
    query = np.ones((1, 3), dtype="float32")
    assert restored.search(query, 2)[1].tolist() == original.search(query, 2)[1].tolist()


def test_load_index_reads_file_resolved_through_sparkfiles(monkeypatch, isolated_cwd) -> None:
    path = isolated_cwd / "stored.index"
    faiss.write_index(_flat_index(), str(path))
    monkeypatch.setattr("hypex.utils.index_utils.SparkFiles.get", lambda name: str(path))
    loaded = FaissIndexStorage(_FakeSession()).load_index("stored.index")
    assert loaded.ntotal == 5


def test_collect_and_register_writes_files_and_registers_them(isolated_cwd) -> None:
    added = []

    class _Context:
        def addFile(self, path):
            added.append(path)

    class _Session:
        sparkContext = _Context()

    class _FakeRDD:
        def __init__(self, shards):
            self.shards = shards

        def toLocalIterator(self):
            return iter(self.shards)

    storage = FaissIndexStorage(_Session())
    shards = [faiss.serialize_index(_flat_index(n=3)), faiss.serialize_index(_flat_index(n=7))]
    refs = storage.collect_and_register(_FakeRDD(shards))

    assert len(refs) == 2 and len(set(refs)) == 2
    assert all(r.startswith("__partition_index_") and r.endswith(".index") for r in refs)
    assert added == [f"{storage._local_tmp_dir}/{r}" for r in refs]
    sizes = [faiss.read_index(p).ntotal for p in added]
    assert sizes == [3, 7]


def test_collect_and_register_empty_rdd() -> None:
    class _Empty:
        @staticmethod
        def toLocalIterator():
            return iter(())

    assert FaissIndexStorage(_FakeSession()).collect_and_register(_Empty()) == []


def test_cleanup_removes_directories_and_resets_list(isolated_cwd) -> None:
    storages = [FaissIndexStorage(_FakeSession()) for _ in range(2)]
    dirs = [s._local_tmp_dir for s in storages]
    FaissIndexStorage.cleanup()
    assert not any(os.path.exists(d) for d in dirs)
    assert FaissIndexStorage.LOCAL_DIRS == []


def test_cleanup_tolerates_missing_directories() -> None:
    storage = FaissIndexStorage(_FakeSession())
    os.rmdir(storage._local_tmp_dir)
    FaissIndexStorage.cleanup()
    assert FaissIndexStorage.LOCAL_DIRS == []


# ---------------------------------------------------------------------------
# CachingIndex
# ---------------------------------------------------------------------------
class _CountingStorage:
    def __init__(self, factory=_flat_index):
        self.factory = factory
        self.loads = []

    def load_index(self, reference):
        self.loads.append(reference)
        return self.factory()


def test_cache_loads_once_per_reference() -> None:
    cache, storage = CachingIndex(), _CountingStorage()
    first = cache.get("a", storage)
    second = cache.get("a", storage)
    assert first is second
    assert storage.loads == ["a"]


def test_cache_returns_distinct_indexes_for_distinct_references() -> None:
    cache, storage = CachingIndex(), _CountingStorage()
    assert cache.get("a", storage) is not cache.get("b", storage)
    assert storage.loads == ["a", "b"]


def test_cache_evicts_least_recently_used() -> None:
    cache, storage = CachingIndex(max_index=2), _CountingStorage()
    cache.get("a", storage)
    cache.get("b", storage)
    cache.get("a", storage)       # refresh "a" -> "b" becomes LRU
    cache.get("c", storage)       # evicts "b"
    assert list(cache._cache) == ["a", "c"]
    cache.get("b", storage)       # must be reloaded
    assert storage.loads == ["a", "b", "c", "b"]


def test_cache_unbounded_when_max_is_none() -> None:
    cache, storage = CachingIndex(max_index=None), _CountingStorage()
    for i in range(20):
        cache.get(f"i{i}", storage)
    assert len(cache._cache) == 20


def test_cache_limit_of_one_keeps_only_latest() -> None:
    cache, storage = CachingIndex(max_index=1), _CountingStorage()
    cache.get("a", storage)
    cache.get("b", storage)
    assert list(cache._cache) == ["b"]


def test_cache_leaves_ivf_nprobe_untouched() -> None:
    # nprobe is no longer configured by CachingIndex (it is set by the faiss
    # extension on the index itself); the cache must return the loaded index as is.
    cache, storage = CachingIndex(), _CountingStorage(_ivf_index)
    expected = faiss.downcast_index(_ivf_index()).nprobe
    index = cache.get("ivf", storage)
    assert faiss.downcast_index(index).nprobe == expected


def test_cache_returns_flat_index_unchanged() -> None:
    cache, storage = CachingIndex(), _CountingStorage(_flat_index)
    index = cache.get("flat", storage)
    assert not hasattr(faiss.downcast_index(index), "nprobe")


def test_cached_index_is_reused_on_repeated_get() -> None:
    cache, storage = CachingIndex(), _CountingStorage(_ivf_index)
    first = cache.get("ivf", storage)
    index = cache.get("ivf", storage)
    assert index is first
    assert storage.loads == ["ivf"]


def test_cache_is_thread_safe_for_concurrent_gets() -> None:
    from concurrent.futures import ThreadPoolExecutor

    cache, storage = CachingIndex(max_index=3), _CountingStorage()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: cache.get(f"i{i % 5}", storage), range(100)))
    assert all(r is not None for r in results)
    assert len(cache._cache) <= 3

"""The registry operates through Dataset without requiring a Spark backend."""

from copy import copy
from uuid import uuid4

import pandas as pd
import pytest

from hypex.dataset import Dataset, FeatureRole
from hypex.dataset.backends import PandasDataset


class CachedBackend(PandasDataset):
    """A tiny in-memory caching backend using the same interface as Spark."""

    def __init__(self, state):
        super().__init__(pd.DataFrame({"x": [1, 3]}))
        self.state = state
        self.key = uuid4().hex

    @property
    def persist_key(self):
        return self.key

    @property
    def is_persisted(self):
        return self.state["cached"]

    def persist(self, storage_level="MEMORY_AND_DISK"):
        self.state["cached"] = True
        return self

    def cache_snapshot(self):
        return copy(self)

    def same_cache(self, other):
        return isinstance(other, CachedBackend) and self.state is other.state

    def materialize(self, action):
        self.state["action"] = action
        if action == "count" and self.state.get("fail"):
            raise RuntimeError("materialization failed")

    def unpersist(self, blocking=False):
        self.state["cached"] = False
        return self


def make_dataset(state):
    return Dataset(data=CachedBackend(state), roles={"x": FeatureRole()})


def test_independent_owners_with_another_backend(fresh_manager):
    state = {"cached": False}
    first, second = make_dataset(state), make_dataset(state)
    first.persist()
    second.persist()
    assert fresh_manager.total_entries() == 2
    first.unpersist()
    assert second.is_persisted
    second.unpersist()
    assert not state["cached"]
    assert fresh_manager.total_entries() == 0


def test_failed_action_keeps_snapshot_reachable(fresh_manager):
    dataset = make_dataset({"cached": False, "fail": True})
    with pytest.raises(RuntimeError, match="materialization failed"):
        dataset.persist()
    assert fresh_manager.total_entries() == 1
    assert dataset.is_persisted
    dataset.unpersist()
    assert not dataset.is_persisted
    assert fresh_manager.total_entries() == 0


def test_materialization_uses_acquired_snapshot(fresh_manager, monkeypatch):
    acquired = {"cached": False}
    dataset = make_dataset(acquired)
    backend = dataset.backend_data
    snapshot = backend.cache_snapshot

    def snapshot_then_change_source():
        cached = snapshot()
        backend.state = {"cached": False}
        return cached

    monkeypatch.setattr(backend, "cache_snapshot", snapshot_then_change_source)
    dataset.persist()
    assert acquired["action"] == "count"
    assert "action" not in backend.state
    dataset.unpersist()
    assert not acquired["cached"]
    assert fresh_manager.total_entries() == 0

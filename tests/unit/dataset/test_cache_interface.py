"""The Dataset cache interface works with either backend."""

import pandas as pd
import pytest

from hypex.dataset import FeatureRole
from hypex.utils import BackendsEnum


@pytest.mark.parametrize("action", ["none", "count", "head"])
def test_cache_interface_preserves_data(make_dataset, backend, action):
    dataset = make_dataset(pd.DataFrame({"x": [1, 3]}), {"x": FeatureRole()})
    assert not dataset.is_persisted
    assert dataset.get_storage_level() is None

    assert dataset.persist(storage_level="DISK_ONLY", action=action) is dataset
    expected = backend == BackendsEnum.spark
    assert dataset.is_persisted == expected
    assert dataset.get_cache_info() == {
        "is_persisted": expected,
        "storage_level": "DISK_ONLY" if expected else None,
        "backend_type": backend,
        "in_memory": True,
    }
    assert dataset.unpersist(blocking=True) is dataset
    assert not dataset.is_persisted
    assert dataset.get_storage_level() is None
    raw = dataset.raw_data
    frame = raw.to_pandas() if hasattr(raw, "to_pandas") else raw
    assert frame.sort_index()["x"].tolist() == [1, 3]


def test_pandas_cache_methods_are_noops():
    from hypex.dataset.backends import PandasDataset

    backend = PandasDataset(pd.DataFrame({"x": [1, 3]}))
    raw = backend.data
    assert backend.persist() is backend
    assert backend.unpersist(blocking=True) is backend
    assert backend.checkpoint(eager=False) is backend
    assert backend.data is raw
    assert not backend.is_persisted
    assert backend.get_storage_level() is None

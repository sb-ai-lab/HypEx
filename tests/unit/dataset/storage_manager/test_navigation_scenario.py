"""Key inheritance, copying and wrappers using the production Dataset."""

from copy import deepcopy

import pytest

pytest.importorskip("pyspark")

from pyspark import StorageLevel

from hypex.dataset import ExperimentData, TempTargetRole

pytestmark = pytest.mark.spark


@pytest.mark.parametrize("operation", ["select", "filter", "groupby"])
def test_derived_dataset_inherits_key(spark_dataset, operation):
    dataset = spark_dataset()
    if operation == "select":
        child = dataset[["value"]]
    elif operation == "filter":
        child = dataset.filter(items=["value"])
    else:
        child = dataset.groupby("group").agg({"value": "sum"})
    assert child.backend_data._persist_key == dataset.backend_data._persist_key


def test_sibling_branches_share_key(spark_dataset):
    dataset = spark_dataset()
    first, second = dataset[["value"]], dataset[["group"]]
    assert first.backend_data._persist_key == second.backend_data._persist_key


def test_deepcopy_gets_independent_key(spark_dataset):
    dataset = spark_dataset()
    clone = deepcopy(dataset)
    assert clone.backend_data._persist_key != dataset.backend_data._persist_key
    assert clone.raw_data is dataset.raw_data


def test_unpersist_copy_preserves_original_cache(spark_dataset, fresh_manager):
    dataset = spark_dataset()
    dataset.persist()
    frame = fresh_manager.frames(dataset.backend_data._persist_key)[0]
    clone = deepcopy(dataset)

    assert not clone.is_persisted
    clone.unpersist(blocking=True)

    assert dataset.is_persisted
    assert frame.storageLevel != StorageLevel.NONE
    dataset.unpersist(blocking=True)
    assert frame.storageLevel == StorageLevel.NONE


def test_tmp_roles_preserve_registered_cache(spark_dataset, fresh_manager):
    experiment = ExperimentData(spark_dataset())
    experiment.ds.persist()
    key = experiment.ds.backend_data._persist_key
    frames = fresh_manager.frames(key)
    experiment.ds.tmp_roles = {"value": TempTargetRole()}

    assert experiment.ds.is_persisted
    assert experiment.ds.get_storage_level() == "MEMORY_AND_DISK"
    experiment.ds.unpersist(blocking=True)

    assert not experiment.ds.is_persisted
    assert fresh_manager.count(key) == 0
    assert all(frame.storageLevel == StorageLevel.NONE for frame in frames)

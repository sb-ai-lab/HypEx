"""Persistence lifecycle through the production Dataset and Spark backend."""

import pytest

pytest.importorskip("pyspark")

from pyspark import StorageLevel

pytestmark = pytest.mark.spark


def test_groupby_releases_both_cached_plans(spark_dataset, fresh_manager):
    dataset = spark_dataset()
    dataset.persist(action="none")
    key = dataset.backend_data._persist_key
    result = dataset.groupby("group").agg({"value": "sum"})
    result.persist(action="none")
    frames = fresh_manager.frames(key)
    assert len(frames) == 2
    for frame in frames:
        frame.count()
        assert frame.storageLevel != StorageLevel.NONE

    result.unpersist(blocking=True)

    assert all(frame.storageLevel == StorageLevel.NONE for frame in frames)
    assert fresh_manager.count(key) == 0
    assert not dataset.is_persisted
    assert not result.is_persisted


def test_materialized_persist_releases_previous_plan(spark_dataset, fresh_manager):
    dataset = spark_dataset()
    dataset.persist(action="none")
    key = dataset.backend_data._persist_key
    old_frame = fresh_manager.frames(key)[0]
    result = dataset.groupby("group").agg({"value": "sum"})
    result.persist(action="count")

    frames = fresh_manager.frames(key)
    assert len(frames) == 1
    assert frames[0] is not old_frame
    assert old_frame.storageLevel == StorageLevel.NONE
    assert frames[0].storageLevel != StorageLevel.NONE
    assert result.is_persisted

    result.unpersist(blocking=True)
    assert frames[0].storageLevel == StorageLevel.NONE


def test_independent_datasets_keep_their_caches(spark_dataset, fresh_manager):
    first, second = spark_dataset(12), spark_dataset(18)
    first.persist()
    second.persist()
    second_frame = fresh_manager.frames(second.backend_data._persist_key)[0]

    first.unpersist(blocking=True)

    assert not first.is_persisted
    assert second.is_persisted
    assert second_frame.storageLevel != StorageLevel.NONE
    second.unpersist(blocking=True)
    assert second_frame.storageLevel == StorageLevel.NONE

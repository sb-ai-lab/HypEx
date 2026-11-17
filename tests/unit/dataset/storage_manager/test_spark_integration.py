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
        frame.materialize("count")
        assert frame.backend_data._internal_frame_api.storageLevel != StorageLevel.NONE

    result.unpersist(blocking=True)

    assert all(
        frame.backend_data._internal_frame_api.storageLevel == StorageLevel.NONE
        for frame in frames
    )
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
    assert old_frame.backend_data._internal_frame_api.storageLevel == StorageLevel.NONE
    assert frames[0].backend_data._internal_frame_api.storageLevel != StorageLevel.NONE
    assert result.is_persisted

    result.unpersist(blocking=True)
    assert frames[0].backend_data._internal_frame_api.storageLevel == StorageLevel.NONE


def test_independent_datasets_keep_their_caches(spark_dataset, fresh_manager):
    first, second = spark_dataset(12), spark_dataset(18)
    first.persist()
    second.persist()
    second_frame = fresh_manager.frames(second.backend_data._persist_key)[0]

    first.unpersist(blocking=True)

    assert not first.is_persisted
    assert second.is_persisted
    assert (
        second_frame.backend_data._internal_frame_api.storageLevel != StorageLevel.NONE
    )
    second.unpersist(blocking=True)
    assert (
        second_frame.backend_data._internal_frame_api.storageLevel == StorageLevel.NONE
    )


@pytest.mark.parametrize(
    "operation", ["arithmetic", "split", "get", "loc", "iloc", "quantile"]
)
def test_transformation_releases_parent(
    spark_dataset, spark_session, fresh_manager, operation
):
    sc = spark_session.sparkContext
    before = {rdd.id() for rdd in sc._jsc.sc().getRDDStorageInfo()}
    dataset = spark_dataset()
    dataset.persist()
    key = dataset.backend_data._persist_key
    parent_frame = fresh_manager.frames(key)[0]
    if operation == "arithmetic":
        result = dataset + 1
    elif operation == "split":
        result = dataset.random_split_labels(
            edges=[5000000, 10000000], labels=["a", "b"]
        )
    elif operation == "get":
        result = dataset.get("value")
    elif operation == "loc":
        result = dataset.loc[[0, 1]]
    elif operation == "iloc":
        result = dataset.iloc[[0, 1]]
    else:
        result = dataset.quantile(q=[0.0, 1.0])
    assert result.backend_data._persist_key == key
    result.persist()
    frames = fresh_manager.frames(key)
    result.unpersist(blocking=True)
    assert fresh_manager.count(key) == 0
    assert (
        parent_frame.backend_data._internal_frame_api.storageLevel == StorageLevel.NONE
    )
    assert all(
        frame.backend_data._internal_frame_api.storageLevel == StorageLevel.NONE
        for frame in frames
    )
    assert {rdd.id() for rdd in sc._jsc.sc().getRDDStorageInfo()} <= before


def test_equivalent_plans_have_shared_ownership(spark_dataset, fresh_manager):
    first, second = spark_dataset(), spark_dataset()
    first.persist()
    second.persist()
    frame = fresh_manager.frames(second.backend_data._persist_key)[0]
    first.unpersist(blocking=True)
    assert second.is_persisted
    assert frame.backend_data._internal_frame_api.storageLevel != StorageLevel.NONE
    second.unpersist(blocking=True)
    assert frame.backend_data._internal_frame_api.storageLevel == StorageLevel.NONE


def test_parent_flag_after_aggregate(spark_dataset):
    parent = spark_dataset()
    parent.persist()
    child = parent.groupby("group").agg({"value": "sum"})
    child.persist()
    assert not parent.is_persisted
    assert parent.get_storage_level() is None
    assert child.is_persisted


def test_head_keeps_parent_cache(spark_dataset, fresh_manager):
    parent = spark_dataset()
    parent.persist()
    child = parent.groupby("group").agg({"value": "sum"})
    child.persist(action="head")
    assert parent.is_persisted
    assert fresh_manager.count(parent.backend_data._persist_key) == 2
    child.unpersist(blocking=True)


def test_shell_cleans_failed_materialization(
    spark_dataset, spark_session, fresh_manager, monkeypatch
):
    import pyspark.pandas as ps

    from hypex import ABTest

    dataset = spark_dataset()
    cache = spark_session._jsparkSession.sharedState().cacheManager()
    before = cache.cachedData().size()

    def fail(self):
        raise RuntimeError("materialization failed")

    monkeypatch.setattr(ps.DataFrame, "count", fail)
    with pytest.raises(RuntimeError, match="materialization failed"):
        ABTest().execute(dataset)
    assert fresh_manager.total_entries() == 0
    assert not dataset.is_persisted
    assert cache.cachedData().size() == before


@pytest.mark.parametrize("failure_stage", ["execute", "extract"])
@pytest.mark.parametrize("cached", [False, True])
def test_shell_preserves_input_on_failure(
    spark_dataset, spark_session, fresh_manager, failure_stage, cached
):
    from hypex import ABTest

    dataset = spark_dataset()
    if cached:
        dataset.persist()
    key = dataset.backend_data._persist_key
    cache = spark_session._jsparkSession.sharedState().cacheManager()
    before = cache.cachedData().size()
    shell = ABTest()

    def fail(*args):
        raise RuntimeError("pipeline failed")

    if failure_stage == "execute":
        shell._experiment.execute = fail
    else:
        shell._experiment.execute = lambda data: data
        shell._out.extract = fail
    with pytest.raises(RuntimeError, match="pipeline failed"):
        shell.execute(dataset)
    assert fresh_manager.keys() == ([key] if cached else [])
    assert dataset.is_persisted == cached
    assert cache.cachedData().size() == before


def test_stratified_splitter_cleans_failed_materialization(
    spark_dataset, spark_session, fresh_manager, monkeypatch
):
    import pyspark.pandas as ps

    from hypex.dataset import ExperimentData
    from hypex.splitters.aa import AASplitterWithStratification

    dataset = spark_dataset()
    cache = spark_session._jsparkSession.sharedState().cacheManager()
    before = cache.cachedData().size()

    def fail(self):
        raise RuntimeError("split materialization failed")

    monkeypatch.setattr(ps.DataFrame, "count", fail)
    with pytest.raises(RuntimeError, match="split materialization failed"):
        AASplitterWithStratification().execute(ExperimentData(dataset))
    assert fresh_manager.total_entries() == 0
    assert not dataset.is_persisted
    assert cache.cachedData().size() == before


@pytest.mark.parametrize("cached", [False, True])
def test_stratified_splitter_preserves_input_cache(
    spark_dataset, fresh_manager, cached
):
    from hypex.dataset import ExperimentData, StratificationRole
    from hypex.splitters.aa import AASplitterWithStratification

    dataset = spark_dataset(60)
    dataset.roles["group"] = StratificationRole()
    if cached:
        dataset.persist()
        frame = fresh_manager.frames(dataset.backend_data._persist_key)[0]
    result = AASplitterWithStratification(random_state=42).execute(
        ExperimentData(dataset)
    )
    assert len(result.ds) == 60
    if cached:
        assert frame.backend_data._internal_frame_api.storageLevel != StorageLevel.NONE
        assert dataset.is_persisted


def test_computed_column_is_cached(spark_session, fresh_manager):
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    from hypex.dataset import Dataset, FeatureRole

    calls = spark_session.sparkContext.accumulator(0)

    @F.udf(T.LongType())
    def compute(value):
        calls.add(1)
        return value + 1

    raw = spark_session.range(10).pandas_api(index_col="id").assign(x=1)
    source = Dataset(data=raw, roles={"x": FeatureRole()})
    source.persist()
    projected = raw.assign(x=raw.x.spark.transform(compute))
    dataset = Dataset(data=projected, roles={"x": FeatureRole()})
    assert not dataset.is_persisted
    dataset.persist()
    warmed = calls.value
    assert warmed == 10
    assert dataset.is_persisted
    for _ in range(2):
        assert dataset.raw_data.spark.frame().agg(F.sum("x")).first()[0] == 20
        assert calls.value == warmed


@pytest.mark.parametrize("cached", [False, True])
def test_persist_copy_preserves_source_wrapper(spark_dataset, fresh_manager, cached):
    from copy import deepcopy

    source = spark_dataset()
    if cached:
        source.persist()
    wrapper = source.raw_data
    internal = wrapper._internal
    clone = deepcopy(source)
    clone.persist(action="none")
    assert source.raw_data is wrapper
    assert source.raw_data._internal is internal
    assert clone.raw_data is not wrapper
    assert clone.is_persisted
    clone.unpersist(blocking=True)
    assert source.is_persisted == cached
    assert fresh_manager.total_entries() == int(cached)


@pytest.mark.parametrize("scenario", ["aa", "ab", "matching"])
@pytest.mark.parametrize("cached", [False, True])
def test_pipeline_cache_lifecycle(spark_session, fresh_manager, scenario, cached):
    import numpy as np
    import pandas as pd

    from hypex import AATest, ABTest, Matching
    from hypex.dataset import Dataset, FeatureRole, TargetRole, TreatmentRole
    from hypex.utils import BackendsEnum

    rng = np.random.default_rng(42)
    dataset = Dataset(
        data=pd.DataFrame(
            {"y": rng.normal(size=60), "g": np.arange(60) % 2, "x": rng.normal(size=60)}
        ),
        roles={"y": TargetRole(), "g": TreatmentRole(), "x": FeatureRole()},
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    if cached:
        dataset.persist()
    internal = dataset.raw_data._internal
    original = dataset.raw_data.to_pandas().sort_index()
    cache = spark_session._jsparkSession.sharedState().cacheManager()
    before = cache.cachedData().size()
    shell = {
        "aa": lambda: AATest(n_iterations=2, random_states=[0, 1]),
        "ab": ABTest,
        "matching": lambda: Matching(
            distance="l2", faiss_mode="base", compute_indexes=True
        ),
    }[scenario]()
    output = shell.execute(dataset)
    assert not output.resume.is_empty()
    assert cache.cachedData().size() == before
    assert fresh_manager.total_entries() == int(cached)
    assert dataset.is_persisted == cached
    assert dataset.raw_data._internal is internal
    pd.testing.assert_frame_equal(dataset.raw_data.to_pandas().sort_index(), original)
    # Outputs backed by Spark must remain readable after the shell's cleanup.
    if scenario == "aa":
        assert len(output.best_split.raw_data.to_pandas()) > 0
    elif scenario == "matching":
        assert len(output.indexes.raw_data.to_pandas()) == len(original)


@pytest.mark.parametrize("index_kind", ["unnamed", "named", "multi"])
def test_unpersist_preserves_index_and_wrapper(
    spark_session, fresh_manager, index_kind
):
    import pandas as pd

    from hypex.dataset import Dataset, FeatureRole
    from hypex.utils import BackendsEnum

    index = pd.Index([5, 7], name="row_id" if index_kind == "named" else None)
    if index_kind == "multi":
        index = pd.MultiIndex.from_tuples(
            [("a", 5), ("b", 7)], names=["group_id", "row_id"]
        )
    dataset = Dataset(
        data=pd.DataFrame({"x": [1, 3]}, index=index),
        roles={"x": FeatureRole()},
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    dataset.persist()
    wrapper = dataset.raw_data
    before = wrapper.to_pandas().sort_index()
    dataset.unpersist(blocking=True)
    assert dataset.raw_data is wrapper
    pd.testing.assert_frame_equal(dataset.raw_data.to_pandas().sort_index(), before)
    assert not dataset.is_persisted

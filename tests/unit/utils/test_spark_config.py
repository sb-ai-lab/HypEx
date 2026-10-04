"""Tests for SparkSessionCalculator and its settings/recommendation dataclasses."""

from __future__ import annotations

import pytest

from hypex.config import MatchingConfig
from hypex.utils.spark_config import (
    SparkRecommendation,
    SparkSessionCalculator,
    SparkSettings,
)

GB = 1024**3
MB = 1024**2


def _calc(**kwargs) -> SparkSessionCalculator:
    return SparkSessionCalculator(**kwargs)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
def test_spark_settings_defaults() -> None:
    settings = SparkSettings()
    assert settings.executor_instances == 10
    assert settings.executor_memory == "4g"
    assert settings.serializer.endswith("KryoSerializer")
    assert settings.extra_configs == {}


def test_spark_settings_extra_configs_not_shared_between_instances() -> None:
    a, b = SparkSettings(), SparkSettings()
    a.extra_configs["k"] = 1
    assert b.extra_configs == {}


def test_recommendation_default_priority() -> None:
    rec = SparkRecommendation("p", 1, 2, "because")
    assert rec.priority == "medium"


# ---------------------------------------------------------------------------
# Construction / size estimation
# ---------------------------------------------------------------------------
def test_numeric_columns_default_to_total_minus_categorical() -> None:
    assert _calc(num_columns=10, num_categorical_columns=3).num_numeric_columns == 7


def test_explicit_numeric_columns_win() -> None:
    assert (
        _calc(
            num_columns=10, num_categorical_columns=3, num_numeric_columns=5
        ).num_numeric_columns
        == 5
    )


def test_estimate_prefers_explicit_size() -> None:
    assert _calc(data_size_bytes=123, num_rows=10**9)._estimate_data_size() == 123


def test_estimate_from_rows() -> None:
    calc = _calc(num_rows=1000, num_columns=8, num_categorical_columns=2)
    assert calc._estimate_data_size() == 6 * 1000 * 8 + 2 * 1000 * 4


def test_estimate_default_is_one_gigabyte() -> None:
    assert _calc()._estimate_data_size() == 1_000_000_000


# ---------------------------------------------------------------------------
# Partitions / executors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "size,expected",
    [
        (1, 10),  # clamped to the 10-partition floor
        (150 * MB * 10, 10),
        (150 * MB * 100, 100),
        (150 * MB * 100 + 1, 100),  # floor division
        (150 * MB * 10_000, 500),  # clamped to the 500 ceiling
    ],
)
def test_optimal_partitions(size, expected) -> None:
    assert _calc()._calculate_optimal_partitions(size) == expected


@pytest.mark.parametrize(
    "partitions,cores,expected",
    [(10, 4, 4), (100, 4, 50), (40, 4, 20), (1000, 4, 50), (16, 2, 16)],
)
def test_optimal_executors(partitions, cores, expected) -> None:
    calc = _calc(target_executor_cores=cores)
    # base = partitions // cores, doubled, clamped to [4, 50]
    assert calc._calculate_optimal_executors(10**9, partitions) == expected


def test_executor_memory_has_floor() -> None:
    assert _calc(num_columns=2)._calculate_executor_memory(10_000, 10) == "4g"


def test_executor_memory_formula_for_large_data() -> None:
    calc = _calc(num_columns=8, target_executor_cores=4)
    size, partitions = 100 * GB, 100
    partition_size = size / partitions
    total = (
        partition_size * calc.FAISS_INDEX_OVERHEAD * partitions
        + partition_size * 4
        + MatchingConfig.FAISS_CHUNK_SIZE * 8 * 4 * 4
        + GB
    ) / GB
    assert calc._calculate_executor_memory(size, partitions) == f"{int(total) + 2}g"


def test_executor_memory_grows_with_data() -> None:
    calc = _calc()
    small = int(calc._calculate_executor_memory(10 * GB, 100).rstrip("g"))
    large = int(calc._calculate_executor_memory(100 * GB, 100).rstrip("g"))
    assert large > small


@pytest.mark.parametrize(
    "memory_gb,expected", [(4, "2048m"), (6, "2048m"), (10, "3072m"), (100, "30720m")]
)
def test_memory_overhead(memory_gb, expected) -> None:
    assert _calc()._calculate_memory_overhead(memory_gb) == expected


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def test_driver_memory_has_floor() -> None:
    assert _calc(num_rows=10)._calculate_driver_memory(1000) == "8g"


def test_driver_memory_formula() -> None:
    calc = _calc(num_columns=8, num_rows=10**9)
    sample_gb = MatchingConfig.FAISS_SAMPLE_TARGET * 8 * 4 / GB
    index_gb = 10**9 * 8 / GB
    expected = max(8, int(4 + sample_gb + index_gb) + 2)
    assert calc._calculate_driver_memory(10**10) == f"{expected}g"


def test_driver_memory_without_row_count_uses_size_per_column() -> None:
    calc = _calc(num_columns=8)
    size = 80 * GB
    expected = max(
        8,
        int(4 + MatchingConfig.FAISS_SAMPLE_TARGET * 8 * 4 / GB + (size // 8) / GB) + 2,
    )
    assert calc._calculate_driver_memory(size) == f"{expected}g"


def test_driver_max_result_size() -> None:
    assert _calc(num_rows=10)._calculate_driver_max_result_size() == "8g"
    assert (
        _calc(num_rows=4 * 10**9)._calculate_driver_max_result_size()
        == f"{int(4e9 * 8 / GB * 2) + 2}g"
    )
    assert _calc(data_size_bytes=20 * GB)._calculate_driver_max_result_size() == "42g"


# ---------------------------------------------------------------------------
# Whole settings
# ---------------------------------------------------------------------------
def test_calculate_optimal_settings_is_consistent() -> None:
    calc = _calc(data_size_bytes=50 * GB, num_columns=10, num_categorical_columns=2)
    settings = calc.calculate_optimal_settings()
    partitions = calc._calculate_optimal_partitions(50 * GB)
    assert settings.shuffle_partitions == settings.default_parallelism == partitions
    assert settings.executor_instances == calc._calculate_optimal_executors(
        50 * GB, partitions
    )
    assert settings.executor_cores == 4
    assert settings.driver_cores == 4
    assert settings.executor_memory == calc._calculate_executor_memory(
        50 * GB, partitions
    )
    assert (
        settings.driver_max_result_size
        == settings.extra_configs["spark.driver.maxResultSize"]
    )
    assert settings.extra_configs["spark.sql.adaptive.enabled"] == "true"


def test_driver_cores_follow_executor_cores_above_four() -> None:
    assert _calc(target_executor_cores=8).calculate_optimal_settings().driver_cores == 8


def test_overhead_matches_chosen_executor_memory() -> None:
    settings = _calc(data_size_bytes=500 * GB).calculate_optimal_settings()
    memory_gb = int(settings.executor_memory.rstrip("g"))
    assert (
        settings.executor_memory_overhead
        == f"{max(2048, int(memory_gb * 1024 * 0.3))}m"
    )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def _optimal() -> SparkSettings:
    return SparkSettings(
        executor_instances=20,
        executor_cores=4,
        executor_memory="16g",
        driver_memory="12g",
        driver_max_result_size="16g",
        shuffle_partitions=200,
    )


def _matching_current() -> dict:
    return {
        "executor_instances": 20,
        "executor_cores": 4,
        "executor_memory": "16g",
        "driver_memory": "12g",
        "driver_max_result_size": "16g",
        "shuffle_partitions": 200,
        "serializer": SparkSettings().serializer,
    }


def test_no_recommendations_when_settings_match() -> None:
    calc = _calc()
    assert calc.generate_recommendations(_matching_current(), _optimal()) == []


def test_each_deviation_yields_one_prioritised_recommendation() -> None:
    calc = _calc()
    current = {
        "executor_instances": 2,
        "executor_cores": 8,
        "executor_memory": "2g",
        "driver_memory": "1g",
        "driver_max_result_size": "1g",
        "shuffle_partitions": 10,
        "serializer": "java",
    }
    recs = {r.parameter: r for r in calc.generate_recommendations(current, _optimal())}
    assert set(recs) == {
        "spark.executor.instances",
        "spark.executor.cores",
        "spark.executor.memory",
        "spark.driver.memory",
        "spark.driver.maxResultSize",
        "spark.sql.shuffle.partitions",
        "spark.serializer",
    }
    assert recs["spark.executor.cores"].priority == "critical"
    assert recs["spark.driver.memory"].priority == "critical"
    assert recs["spark.driver.maxResultSize"].priority == "critical"
    assert recs["spark.executor.instances"].priority == "high"
    assert recs["spark.serializer"].priority == "medium"
    assert recs["spark.executor.instances"].recommended_value == 20
    assert recs["spark.executor.instances"].current_value == 2


def test_more_executors_than_needed_is_not_flagged() -> None:
    current = dict(_matching_current(), executor_instances=99)
    assert not [
        r
        for r in _calc().generate_recommendations(current, _optimal())
        if r.parameter == "spark.executor.instances"
    ]


def test_fewer_cores_than_target_is_not_flagged() -> None:
    current = dict(_matching_current(), executor_cores=2)
    assert not [
        r
        for r in _calc().generate_recommendations(current, _optimal())
        if r.parameter == "spark.executor.cores"
    ]


def test_recommendations_are_stored_and_printed(capsys) -> None:
    calc = _calc()
    calc.print_recommendations()
    assert "All settings are optimal" in capsys.readouterr().out
    calc.generate_recommendations(
        dict(_matching_current(), driver_memory="1g"), _optimal()
    )
    calc.print_recommendations()
    out = capsys.readouterr().out
    assert "[CRITICAL] spark.driver.memory" in out
    assert "Recommended value: 12g" in out


def test_check_current_settings_without_session() -> None:
    assert _calc().check_current_settings(None) == {}


@pytest.mark.spark
def test_check_current_settings_reads_session_conf(spark_session) -> None:
    calc = _calc()
    settings = calc.check_current_settings(spark_session)
    assert settings["shuffle_partitions"] == 2
    assert {
        "executor_cores",
        "driver_memory",
        "serializer",
        "driver_max_result_size",
    } <= set(settings)
    assert calc._current_settings == settings


# ---------------------------------------------------------------------------
# Config mapping / filesystem helpers
# ---------------------------------------------------------------------------
def test_every_settings_field_has_a_spark_conf_mapping() -> None:
    fields = set(SparkSettings.__dataclass_fields__) - {"extra_configs"}
    assert fields == set(SparkSessionCalculator.SETTINGS_TO_SPARK_CONF)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: broadcast_block_size ('4m') is mapped to spark.sql.broadcast.timeout "
    "(a duration in seconds), not a block-size setting",
)
def test_broadcast_block_size_maps_to_a_size_setting() -> None:
    mapped = SparkSessionCalculator.SETTINGS_TO_SPARK_CONF["broadcast_block_size"]
    assert "timeout" not in mapped


def test_define_fs_defaults_to_local_file_system() -> None:
    class _Broken:
        @property
        def sparkContext(self):
            raise RuntimeError

        @property
        def conf(self):
            raise RuntimeError

    assert SparkSessionCalculator._define_fs(_Broken()) == "file:///"


def test_define_fs_rewrites_viewfs_to_hdfs() -> None:
    class _Hadoop:
        @staticmethod
        def get(key, default):
            return "viewfs://cluster/data"

    class _Ctx:
        class _jsc:
            @staticmethod
            def hadoopConfiguration():
                return _Hadoop()

    class _Session:
        sparkContext = _Ctx()

    assert SparkSessionCalculator._define_fs(_Session()) == "hdfs://cluster/data"


@pytest.mark.spark
def test_define_fs_on_real_local_session(spark_session) -> None:
    assert SparkSessionCalculator._define_fs(spark_session) == "file:///"


@pytest.mark.spark
@pytest.mark.xfail(
    strict=True,
    reason="Issue: apply_settings sets values on a copy returned by SparkContext.getConf(), "
    "so the running session never sees the new runtime settings",
)
def test_apply_settings_changes_runtime_conf(spark_session, capsys) -> None:
    before = spark_session.conf.get("spark.sql.files.maxPartitionBytes")
    _calc().apply_settings(spark_session, SparkSettings(max_partition_bytes="64m"))
    assert spark_session.conf.get("spark.sql.files.maxPartitionBytes") == "64m"
    spark_session.conf.set("spark.sql.files.maxPartitionBytes", before)


# ---------------------------------------------------------------------------
# optimize_config
# ---------------------------------------------------------------------------
def test_optimize_config_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="config must be"):
        _calc().optimize_config("not a config")  # type: ignore[arg-type]


def test_optimize_config_from_none_sets_optimal_values(capsys) -> None:
    calc = _calc(data_size_bytes=50 * GB)
    conf = calc.optimize_config(None)
    settings = calc.calculate_optimal_settings()
    assert conf.get("spark.executor.memory") == settings.executor_memory
    assert conf.get("spark.driver.memory") == settings.driver_memory
    assert conf.get("spark.sql.shuffle.partitions") == str(settings.shuffle_partitions)


def test_optimize_config_keeps_unrelated_keys(capsys) -> None:
    conf = _calc().optimize_config({"spark.app.name": "MyApp", "custom.key": "v"})
    assert conf.get("spark.app.name") == "MyApp"
    assert conf.get("custom.key") == "v"


def test_optimize_config_accepts_spark_conf_without_mutating_it(capsys) -> None:
    from pyspark import SparkConf

    original = SparkConf().setAppName("keep").set("spark.executor.memory", "1g")
    optimized = _calc(data_size_bytes=50 * GB).optimize_config(original)
    assert original.get("spark.executor.memory") == "1g"
    assert optimized.get("spark.app.name") == "keep"
    assert optimized.get("spark.executor.memory") != "1g"

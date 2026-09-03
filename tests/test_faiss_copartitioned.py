"""Equivalence and safety checks for the co-partitioned Spark FAISS search.

The co-partitioned strategy groups control rows by their nearest IVF centroid
and routes queries to the same groups, so a task only ever holds one cluster.
These tests pin down the properties the strategy is supposed to guarantee:

* at equal probe counts it returns exactly the same neighbours as the legacy
  per-partition strategy;
* raising the probe count only improves recall, up to the exact 1-NN answer;
* splitting oversized clusters into salted sub-groups does not change results;
* every query gets a neighbour (a null would raise ``PairsNotFoundError``
  downstream in ``FaissNearestNeighbors``).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

faiss = pytest.importorskip("faiss")
pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402

from hypex.config import MatchingConfig  # noqa: E402
from hypex.dataset import Dataset, FeatureRole  # noqa: E402
from hypex.extensions.faiss import SparkFaissExtension  # noqa: E402
from hypex.utils.enums import BackendsEnum  # noqa: E402

DIM = 6
N_CONTROL = 6_000
N_QUERY = 1_500


@pytest.fixture(scope="module")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")
    session = (
        SparkSession.builder.master("local[2]")
        .appName("faiss_copartitioned_tests")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def data(spark):
    rng = np.random.default_rng(11)
    control = rng.standard_normal((N_CONTROL, DIM))
    control[: N_CONTROL // 4] *= 0.2  # a few dense clusters, as in real data
    query = rng.standard_normal((N_QUERY, DIM))

    columns = [f"f{i}" for i in range(DIM)]

    def to_dataset(matrix, start):
        frame = pd.DataFrame(matrix, columns=columns)
        frame.insert(0, "idx", np.arange(start, start + len(frame)))
        return Dataset(
            roles={c: FeatureRole() for c in columns},
            data=spark.createDataFrame(frame),
            backend=BackendsEnum.spark,
        ).set_index("idx")

    return {
        "control": control,
        "query": query,
        "control_ds": to_dataset(control, 0),
        "query_ds": to_dataset(query, N_CONTROL),
    }


def _match(data, mode, probes, n_neighbors=1, max_group=250_000):
    original = (
        MatchingConfig.FAISS_SEARCH_MODE,
        MatchingConfig.FAISS_N_PROBES,
        MatchingConfig.FAISS_MAX_GROUP_ROWS,
    )
    MatchingConfig.FAISS_SEARCH_MODE = mode
    MatchingConfig.FAISS_N_PROBES = probes
    MatchingConfig.FAISS_MAX_GROUP_ROWS = max_group
    extension = SparkFaissExtension(n_neighbors=n_neighbors)
    try:
        matched = extension.calc(
            data=data["control_ds"],
            test_data=data["query_ds"],
            mode="auto",
            fit_mode="sample",
        )
        frame = matched._backend_data.data.to_pandas().sort_index()
    finally:
        extension.unpersist()
        (
            MatchingConfig.FAISS_SEARCH_MODE,
            MatchingConfig.FAISS_N_PROBES,
            MatchingConfig.FAISS_MAX_GROUP_ROWS,
        ) = original
    return frame


def _recall_at_1(data, frame):
    """Share of queries matched to their true nearest neighbour."""
    control, query = data["control"], data["query"]
    exact = faiss.IndexFlatL2(DIM)
    exact.add(np.ascontiguousarray(control, dtype="float32"))
    _, truth = exact.search(np.ascontiguousarray(query, dtype="float32"), 1)

    positions = frame.index.to_numpy("int64") - N_CONTROL
    matched = frame["1"].to_numpy("int64")
    got = ((query[positions] - control[matched]) ** 2).sum(axis=1)
    best = ((query[positions] - control[truth[positions, 0]]) ** 2).sum(axis=1)
    return float(np.isclose(got, best, rtol=1e-6, atol=1e-9).mean())


def test_every_query_is_matched(data):
    frame = _match(data, "copartitioned", probes=8)
    assert len(frame) == N_QUERY
    assert not frame.isna().to_numpy().any()


def test_matches_legacy_strategy_at_equal_probes(data):
    copartitioned = _match(data, "copartitioned", probes=2)
    legacy = _match(data, "legacy", probes=2)
    pd.testing.assert_series_equal(copartitioned["1"], legacy["1"])


def test_more_probes_only_improve_recall(data):
    recalls = [
        _recall_at_1(data, _match(data, "copartitioned", probes=probes))
        for probes in (2, 8, 32)
    ]
    assert recalls == sorted(recalls)
    assert recalls[-1] > 0.99


def test_salting_oversized_clusters_does_not_change_results(data, monkeypatch):
    """Splitting clusters must not move a single match.

    The layout is captured so the test fails loudly if the fixture ever stops
    producing clusters large enough to be split — otherwise this would silently
    degrade into a comparison of two unsalted runs.
    """
    seen = {}
    original = SparkFaissExtension._layout_frame

    def capture(self, session, n_buckets):
        seen["salts"] = max(self._cluster_salts.values())
        seen["buckets"] = max(n_buckets.values()) if n_buckets else 1
        return original(self, session, n_buckets)

    unsalted = _match(data, "copartitioned", probes=8)

    monkeypatch.setattr(SparkFaissExtension, "_layout_frame", capture)
    salted = _match(data, "copartitioned", probes=8, max_group=200)

    assert seen["salts"] > 1, "control side was never split — the test proves nothing"
    assert seen["buckets"] > 1, "query side was never split — the test proves nothing"
    pd.testing.assert_frame_equal(unsalted, salted)


def test_multiple_neighbours_are_distinct_and_complete(data):
    frame = _match(data, "copartitioned", probes=32, n_neighbors=3)
    assert list(frame.columns) == ["1", "2", "3"]
    assert not frame.isna().to_numpy().any()
    neighbours = frame.to_numpy("int64")
    assert all(len(set(row)) == 3 for row in neighbours)

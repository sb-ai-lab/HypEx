"""Additional tests for hypex.extensions.faiss (Pandas IVF path, Spark fit modes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from hypex.config import MatchingConfig  # noqa: E402
from hypex.dataset import Dataset, FeatureRole  # noqa: E402
from hypex.extensions import PandasFaissExtension, SparkFaissExtension  # noqa: E402
from hypex.extensions.faiss import FaissExtension, get_executor_cache  # noqa: E402
from hypex.utils import BackendsEnum  # noqa: E402
from hypex.utils.registry import backend_factory  # noqa: E402


def _points(n, seed, offset=0.0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.normal(offset, 1, (n, 2)), columns=["f1", "f2"])


def _ds(df, backend=BackendsEnum.pandas, session=None) -> Dataset:
    return Dataset(
        roles={c: FeatureRole() for c in df.columns},
        data=df.copy(),
        backend=backend,
        session=session if backend == BackendsEnum.spark else None,
    )


def _pdf(ds) -> pd.DataFrame:
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


def _brute(control: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    c, t = control.to_numpy(), test.to_numpy()
    dist = ((t[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    return control.index.to_numpy()[np.argmin(dist, axis=1)]


# ---------------------------------------------------------------------------
# Pandas
# ---------------------------------------------------------------------------
def test_backend_factory_resolves_pandas_faiss() -> None:
    ds = _ds(_points(5, 0))
    assert backend_factory.resolve_backend(FaissExtension, ds) is PandasFaissExtension


def test_mahalanobis_transform_selects_matching_columns_only() -> None:
    df = _points(6, 0)
    df["extra"] = 1.0
    matrix = Dataset(
        roles={"a": FeatureRole(), "b": FeatureRole()},
        data=pd.DataFrame([[2.0, 0.0], [0.0, 3.0]], index=["f1", "f2"], columns=["a", "b"]),
        backend=BackendsEnum.pandas,
    )
    out = FaissExtension._mahalanobis_transform(_ds(df), matrix)
    assert out.shape == (6, 2)
    np.testing.assert_allclose(
        _pdf(out).to_numpy(dtype=float),
        df[["f1", "f2"]].to_numpy() * np.array([2.0, 3.0]),
        atol=1e-9,
    )


def test_mahalanobis_scaling_changes_nearest_neighbour() -> None:
    control = pd.DataFrame({"f1": [0.0, 1.0], "f2": [1.0, 0.0]})
    test = pd.DataFrame({"f1": [0.0], "f2": [0.0]})
    # equal distances in raw space are resolved in favour of f2-heavy scaling:
    matrix = Dataset(
        roles={"a": FeatureRole(), "b": FeatureRole()},
        data=pd.DataFrame([[10.0, 0.0], [0.0, 1.0]], index=["f1", "f2"], columns=["a", "b"]),
        backend=BackendsEnum.pandas,
    )
    ext = PandasFaissExtension(mahalanobis=matrix)
    # f1 differences are inflated x10, so control row 0 (f1=0) is the nearest
    assert _pdf(ext.calc(_ds(control), _ds(test))).iloc[0, 0] == 0


def test_fast_mode_small_data_still_uses_flat_index() -> None:
    import faiss

    control, test = _points(30, 0), _points(10, 1)
    ext = PandasFaissExtension(faiss_mode="fast")
    ext.calc(_ds(control), _ds(test), mode="fit")
    # IVF is only used above 1000 rows on both sides
    assert isinstance(faiss.downcast_index(ext.index.index), faiss.IndexFlatL2)


def test_fast_mode_large_data_builds_ivf_index_and_finds_matches() -> None:
    import faiss

    control, test = _points(1_200, 0), _points(1_100, 1)
    ext = PandasFaissExtension(faiss_mode="fast")
    result = ext.calc(_ds(control), _ds(test))
    assert isinstance(faiss.downcast_index(ext.index.index), faiss.IndexIVFFlat)
    assert ext.index.ntotal == len(control)
    found = _pdf(result).iloc[:, 0].to_numpy()
    # IVF is approximate: most queries must still hit their exact nearest neighbour
    assert (found == _brute(control, test)).mean() > 0.8


def test_n_neighbors_two_returns_two_columns() -> None:
    control, test = _points(30, 0), _points(10, 1)
    found = _pdf(PandasFaissExtension(n_neighbors=2).calc(_ds(control), _ds(test)))
    assert found.shape == (10, 2)
    assert (found.iloc[:, 0] != found.iloc[:, 1]).all()


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="FaissExtension.fit forwards target_data= to MLExtension.calc, which "
    "has no such parameter (and would recurse into self.fit)",
)
def test_public_fit_builds_index() -> None:
    ext = PandasFaissExtension()
    ext.fit(_ds(_points(5, 0)))
    assert ext.index.ntotal == 5


# ---------------------------------------------------------------------------
# Executor cache
# ---------------------------------------------------------------------------
def test_executor_cache_is_singleton() -> None:
    assert get_executor_cache() is get_executor_cache()


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------
@pytest.fixture
def spark_pair(spark_session):
    control, test = _points(60, 0), _points(25, 1, 0.3)
    return (
        control,
        test,
        _ds(control, BackendsEnum.spark, spark_session),
        _ds(test, BackendsEnum.spark, spark_session),
    )


@pytest.mark.spark
@pytest.mark.parametrize("fit_mode", ["sample", "cluster", "full", "shuffle"])
def test_spark_fit_modes_return_valid_neighbours(
    monkeypatch, spark_pair, fit_mode
) -> None:
    control, test, ctrl_ds, test_ds = spark_pair
    monkeypatch.setattr(MatchingConfig, "FAISS_FIT_MODE", fit_mode)
    monkeypatch.setattr(MatchingConfig, "BUCKET_SIZE", 20)
    ext = SparkFaissExtension(n_neighbors=1)
    found = _pdf(ext.calc(ctrl_ds, test_ds)).sort_index()
    ext.unpersist()
    assert len(found) == len(test)
    values = found.iloc[:, 0].to_numpy()
    assert set(values) <= set(control.index)
    expected = _brute(control, test)
    if fit_mode == "full":
        np.testing.assert_array_equal(values, expected)
    else:
        # approximate indexes: most queries find the exact neighbour
        assert (values == expected).mean() > 0.6


@pytest.mark.spark
def test_spark_n_neighbors_three_full_mode(monkeypatch, spark_pair) -> None:
    control, test, ctrl_ds, test_ds = spark_pair
    monkeypatch.setattr(MatchingConfig, "FAISS_FIT_MODE", "full")
    ext = SparkFaissExtension(n_neighbors=3)
    found = _pdf(ext.calc(ctrl_ds, test_ds)).sort_index()
    ext.unpersist()
    c, t = control.to_numpy(), test.to_numpy()
    order = np.argsort(((t[:, None, :] - c[None]) ** 2).sum(-1), axis=1)[:, :3]
    for row, exp in zip(found.to_numpy(), order):
        assert set(row) == set(exp)


@pytest.mark.spark
def test_spark_invalid_fit_mode_raises(monkeypatch, spark_pair) -> None:
    _, _, ctrl_ds, test_ds = spark_pair
    monkeypatch.setattr(MatchingConfig, "FAISS_FIT_MODE", "bogus")
    with pytest.raises(ValueError, match="Incorrect faiss fit mode"):
        SparkFaissExtension().calc(ctrl_ds, test_ds)


@pytest.mark.spark
def test_spark_predict_before_fit_raises(spark_pair) -> None:
    _, _, ctrl_ds, test_ds = spark_pair
    with pytest.raises(ValueError, match="not created yet"):
        SparkFaissExtension().calc(ctrl_ds, test_ds, mode="predict")


@pytest.mark.spark
def test_spark_predict_requires_test_data(spark_pair) -> None:
    _, _, ctrl_ds, _ = spark_pair
    with pytest.raises(ValueError, match="test_data is needed"):
        SparkFaissExtension().calc(ctrl_ds, None, mode="auto")


@pytest.mark.spark
def test_spark_fit_mode_returns_self_and_marks_fitted(monkeypatch, spark_pair) -> None:
    _, _, ctrl_ds, test_ds = spark_pair
    monkeypatch.setattr(MatchingConfig, "FAISS_FIT_MODE", "full")
    ext = SparkFaissExtension()
    assert ext.calc(ctrl_ds, test_ds, mode="fit") is ext
    assert ext._fitted is True
    ext.unpersist()
    assert ext._sharded_rdd is None


@pytest.mark.spark
def test_spark_string_feature_is_rejected(spark_session) -> None:
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": ["a", "b"]})
    ds = Dataset(
        roles={"f1": FeatureRole(), "f2": FeatureRole()},
        data=df,
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    with pytest.raises(TypeError, match="categorical"):
        SparkFaissExtension().calc(ds, ds)


@pytest.mark.spark
@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="SparkFaissExtension defines __enter__ but not __exit__",
)
def test_spark_context_manager_returns_self() -> None:
    ext = SparkFaissExtension()
    with ext as entered:
        assert entered is ext


@pytest.mark.spark
def test_spark_compute_cluster_params() -> None:
    ext = SparkFaissExtension(n_neighbors=5)
    ext._data_size = 10_000
    ext._compute_cluster_params()
    assert ext.k == 50  # sqrt(10000 / 4)
    assert ext._nprobe == 10  # max(k // 10, 10) clipped to [.., 50], >= n_neighbors
    ext._data_size = 100
    ext._compute_cluster_params()
    assert ext.k == 5
    assert ext._nprobe == 10

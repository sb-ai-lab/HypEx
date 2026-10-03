"""Tests for FaissNearestNeighbors and the Pandas/Spark FAISS extensions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    TreatmentRole,
)
from hypex.extensions import FaissExtension, PandasFaissExtension, SparkFaissExtension
from hypex.ml import FaissNearestNeighbors
from hypex.utils import BackendsEnum
from hypex.utils.errors import AbstractMethodError, PairsNotFoundError


def _points(n: int = 30, seed: int = 0, offset: float = 0.0, start: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        rng.normal(offset, 1, (n, 2)),
        columns=["f1", "f2"],
        index=range(start, start + n),
    )


def _ds(df: pd.DataFrame, backend=BackendsEnum.pandas, session=None) -> Dataset:
    return Dataset(
        roles={"f1": FeatureRole(), "f2": FeatureRole()},
        data=df.copy(),
        backend=backend,
        session=session if backend == BackendsEnum.spark else None,
    )


def _pdf(ds) -> pd.DataFrame:
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


def _brute_force(control: pd.DataFrame, test: pd.DataFrame, k: int) -> np.ndarray:
    """Row i -> index labels of the k closest control rows (L2)."""
    c, t = control.to_numpy(), test.to_numpy()
    dist = ((t[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    order = np.argsort(dist, axis=1)[:, :k]
    return control.index.to_numpy()[order]


@pytest.fixture
def control() -> pd.DataFrame:
    return _points(40, seed=0)


@pytest.fixture
def test_df() -> pd.DataFrame:
    return _points(25, seed=1, offset=0.3, start=100)


# ---------------------------------------------------------------------------
# PandasFaissExtension
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3])
def test_pandas_extension_matches_brute_force(control, test_df, k) -> None:
    result = PandasFaissExtension(n_neighbors=k).calc(_ds(control), _ds(test_df))
    found = _pdf(result)
    assert list(found.index) == list(test_df.index)
    expected = _brute_force(control, test_df, k)
    if k == 1:
        assert found.iloc[:, 0].tolist() == expected[:, 0].tolist()
    else:
        for row_found, row_expected in zip(found.to_numpy(), expected):
            assert set(row_found) == set(row_expected)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: matches whose index label exceeds len(data)+len(test_data) are replaced "
    "by -1, so non-default (e.g. shuffled or offset) index labels are lost",
)
def test_pandas_extension_returns_global_control_index_labels() -> None:
    control = _points(10, seed=0, start=500)
    test = _points(5, seed=1, start=0)
    found = _pdf(PandasFaissExtension().calc(_ds(control), _ds(test)))
    assert set(found.iloc[:, 0]) <= set(control.index)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: matches whose index label exceeds len(data)+len(test_data) are replaced "
    "by -1, so non-default (e.g. shuffled or offset) index labels are lost",
)
def test_pandas_extension_exact_duplicate_is_nearest() -> None:
    control = pd.DataFrame({"f1": [0.0, 10.0, 20.0], "f2": [0.0, 10.0, 20.0]}, index=[7, 8, 9])
    test = pd.DataFrame({"f1": [10.0], "f2": [10.0]}, index=[0])
    found = _pdf(PandasFaissExtension().calc(_ds(control), _ds(test)))
    assert found.iloc[0, 0] == 8


@pytest.mark.parametrize("mode", ["base", "auto", "fast"])
def test_faiss_modes_give_same_result_on_small_data(control, test_df, mode) -> None:
    base = _pdf(PandasFaissExtension(faiss_mode="base").calc(_ds(control), _ds(test_df)))
    other = _pdf(PandasFaissExtension(faiss_mode=mode).calc(_ds(control), _ds(test_df)))
    assert base.equals(other)


def test_prepare_indexes_selects_k_unique_distance_levels() -> None:
    index = np.array([[5, 6, 7, 8]])
    dist = np.array([[0.0, 0.0, 1.0, 2.0]])
    result = PandasFaissExtension._prepare_indexes(index, dist, 1)
    assert result.tolist() == [[5, 6]]  # both ties at the best distance are kept


def test_mahalanobis_transform_noop_without_matrix(control) -> None:
    ds = _ds(control)
    assert FaissExtension._mahalanobis_transform(ds, None) is ds


def test_mahalanobis_transform_applies_matrix(control) -> None:
    matrix = Dataset(
        roles={"a": FeatureRole(), "b": FeatureRole()},
        data=pd.DataFrame([[2.0, 0.0], [0.0, 3.0]], index=["f1", "f2"], columns=["a", "b"]),
        backend=BackendsEnum.pandas,
    )
    transformed = FaissExtension._mahalanobis_transform(_ds(control), matrix)
    np.testing.assert_allclose(
        _pdf(transformed).to_numpy(dtype=float),
        control.to_numpy() @ np.array([[2.0, 0.0], [0.0, 3.0]]),
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def test_predict_before_fit_raises(control, test_df) -> None:
    with pytest.raises(ValueError, match="index is not created yet"):
        PandasFaissExtension().calc(_ds(control), _ds(test_df), mode="predict")


def test_predict_requires_test_data(control) -> None:
    with pytest.raises(ValueError, match="test_data is needed"):
        PandasFaissExtension().calc(_ds(control), None, mode="auto")


def test_fit_mode_returns_self_with_populated_index(control, test_df) -> None:
    extension = PandasFaissExtension()
    assert extension.index is None
    returned = extension.calc(_ds(control), _ds(test_df), mode="fit")
    assert returned is extension
    assert extension.index.ntotal == len(control)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="Issue: mode='predict' queries the index with `data` (the indexed set) instead of "
    "`test_data`, so the result has the wrong number of rows",
)
def test_fit_then_predict_equals_auto(control, test_df) -> None:
    auto = _pdf(PandasFaissExtension().calc(_ds(control), _ds(test_df)))
    extension = PandasFaissExtension()
    extension.calc(_ds(control), _ds(test_df), mode="fit")
    split = _pdf(extension.calc(_ds(control), _ds(test_df), mode="predict"))
    assert split.shape == auto.shape


def test_abstract_calc_raises() -> None:
    with pytest.raises(AbstractMethodError):
        FaissExtension.calc(PandasFaissExtension(), None)


# ---------------------------------------------------------------------------
# SparkFaissExtension
# ---------------------------------------------------------------------------
@pytest.mark.spark
def test_spark_extension_matches_brute_force(control, test_df, spark_session) -> None:
    ctrl = _ds(control.reset_index(drop=True), BackendsEnum.spark, spark_session)
    test = _ds(test_df.reset_index(drop=True), BackendsEnum.spark, spark_session)
    result = _pdf(SparkFaissExtension(n_neighbors=1).calc(ctrl, test))
    expected = _brute_force(control.reset_index(drop=True), test_df.reset_index(drop=True), 1)[:, 0]
    # Spark returns partitions in arbitrary row order; the index carries the query row id.
    assert result.sort_index().iloc[:, 0].tolist() == expected.tolist()


@pytest.mark.spark
@pytest.mark.xfail(
    strict=True,
    reason="Issue: Spark FAISS result drops the query-set index (0..n-1) while the Pandas "
    "result keeps the test index",
)
def test_spark_result_keeps_test_index(control, test_df, spark_session) -> None:
    ctrl = _ds(control, BackendsEnum.spark, spark_session)
    test = _ds(test_df, BackendsEnum.spark, spark_session)
    result = _pdf(SparkFaissExtension(n_neighbors=1).calc(ctrl, test))
    assert sorted(result.index) == sorted(test_df.index)


# ---------------------------------------------------------------------------
# FaissNearestNeighbors executor
# ---------------------------------------------------------------------------
def _experiment(n_each: int = 20, seed: int = 0) -> tuple[pd.DataFrame, ExperimentData]:
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (2 * n_each, 2))
    df = pd.DataFrame(
        {"f1": X[:, 0], "f2": X[:, 1], "t": np.r_[np.zeros(n_each, int), np.ones(n_each, int)]}
    )
    roles = {"f1": FeatureRole(), "f2": FeatureRole(), "t": TreatmentRole()}
    return df, ExperimentData(Dataset(roles=roles, data=df.copy(), backend=BackendsEnum.pandas))


def test_executor_defaults() -> None:
    executor = FaissNearestNeighbors()
    assert (executor.n_neighbors, executor.two_sides, executor.test_pairs) == (1, False, False)
    assert executor.faiss_mode == "auto"


def test_inner_function_matches_brute_force(control, test_df) -> None:
    result = FaissNearestNeighbors._inner_function(_ds(control), _ds(test_df), n_neighbors=1)
    assert _pdf(result).iloc[:, 0].tolist() == _brute_force(control, test_df, 1)[:, 0].tolist()


def test_execute_inner_function_returns_test_only_by_default(control, test_df) -> None:
    grouping = [("0", _ds(control)), ("1", _ds(test_df))]
    result = FaissNearestNeighbors._execute_inner_function(
        grouping, tmp_roles={}, n_neighbors=1, two_sides=False, test_pairs=False
    )
    assert set(result) == {"test"}


def test_execute_inner_function_two_sides(control, test_df) -> None:
    grouping = [("0", _ds(control)), ("1", _ds(test_df))]
    result = FaissNearestNeighbors._execute_inner_function(
        grouping, tmp_roles={}, n_neighbors=1, two_sides=True, test_pairs=False
    )
    assert set(result) == {"test", "control"}
    assert len(_pdf(result["control"])) == len(control)
    assert len(_pdf(result["test"])) == len(test_df)


def test_execute_inner_function_test_pairs_returns_control(control, test_df) -> None:
    grouping = [("0", _ds(control)), ("1", _ds(test_df))]
    result = FaissNearestNeighbors._execute_inner_function(
        grouping, tmp_roles={}, n_neighbors=1, two_sides=False, test_pairs=True
    )
    assert set(result) == {"control"}


@pytest.mark.xfail(
    strict=True,
    raises=(FutureWarning, AttributeError),
    reason="Issue: PandasDataset.count_groups does int(Series) (FutureWarning) and, past "
    "that, execute() calls Dataset.reindex (defined only on SmallDataset) when the "
    "one-sided result is shorter than the dataset, so default matching cannot run",
)
def test_execute_default_one_sided_matching() -> None:
    df, data = _experiment()
    out = FaissNearestNeighbors(grouping_role=TreatmentRole()).execute(data)
    matched = [
        c for c in out.ds.columns if isinstance(out.ds.roles[c], AdditionalMatchingRole)
    ]
    assert matched


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, emitting a FutureWarning (TypeError for several group cols)",
)
def test_execute_two_sides_stores_matched_indexes() -> None:
    df, data = _experiment()
    executor = FaissNearestNeighbors(two_sides=True, grouping_role=TreatmentRole())
    out = executor.execute(data)
    matched = [c for c in out.ds.columns if isinstance(out.ds.roles[c], AdditionalMatchingRole)]
    assert len(matched) == 1
    found = out.ds.backend_data.data[matched[0]]
    ctrl = df[df.t == 0][["f1", "f2"]]
    test = df[df.t == 1][["f1", "f2"]]
    np.testing.assert_array_equal(
        found.loc[test.index].to_numpy(), _brute_force(ctrl, test, 1)[:, 0]
    )
    np.testing.assert_array_equal(
        found.loc[ctrl.index].to_numpy(), _brute_force(test, ctrl, 1)[:, 0]
    )


@pytest.mark.xfail(
    strict=True,
    raises=FutureWarning,
    reason="Issue: PandasDataset.count_groups does int(df[cols].nunique()) on a Series, emitting a FutureWarning (TypeError for several group cols)",
)
def test_execute_two_sides_neighbours_come_from_opposite_group() -> None:
    df, data = _experiment()
    out = FaissNearestNeighbors(two_sides=True, grouping_role=TreatmentRole()).execute(data)
    col = [c for c in out.ds.columns if isinstance(out.ds.roles[c], AdditionalMatchingRole)][0]
    found = out.ds.backend_data.data[col]
    for idx, match in found.items():
        assert df.t.loc[match] != df.t.loc[idx]


def test_pairs_not_found_error_is_exception() -> None:
    assert issubclass(PairsNotFoundError, Exception)


# ---------------------------------------------------------------------------
# NaN in the faiss result must still be reported as PairsNotFoundError
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::FutureWarning", "ignore::UserWarning")
def test_execute_nan_matches_raise_pairs_not_found(monkeypatch) -> None:
    """The warning loop only counts NaNs; the later per-group check must still raise."""
    df, data = _experiment()
    executor = FaissNearestNeighbors(grouping_role=TreatmentRole())
    n_test = int((df.t == 1).sum())
    nan_matches = Dataset(
        roles={"indexes": AdditionalMatchingRole()},
        data=pd.DataFrame(
            {"indexes": [np.nan, *range(n_test - 1)]},
            index=df[df.t == 1].index,
        ),
        backend=BackendsEnum.pandas,
    )
    monkeypatch.setattr(executor, "calc", lambda **kwargs: {"test": nan_matches})
    with pytest.warns(UserWarning, match="nans"), pytest.raises(PairsNotFoundError):
        executor.execute(data)

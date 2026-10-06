"""Tests for BiasExtension pandas / Spark implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import (
    AdditionalMatchingRole,
    Dataset,
    FeatureRole,
    TargetRole,
    TreatmentRole,
)
from hypex.extensions import BiasExtension
from hypex.extensions.bias import PandasBisaExtesion
from hypex.utils import BackendsEnum
from hypex.utils.registry import backend_factory

ROLES = {
    "treat": TreatmentRole(),
    "x": FeatureRole(),
    "y": TargetRole(),
    "nn": AdditionalMatchingRole(),
}


def _frame(nn=None) -> pd.DataFrame:
    rng = np.random.RandomState(1)
    n = 10
    x = rng.uniform(0, 10, 2 * n)
    treat = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    y = 2.0 * x + 3.0 * treat
    if nn is None:
        nn = np.r_[np.arange(n, 2 * n), np.arange(0, n)]
    return pd.DataFrame({"treat": treat, "x": x, "y": y, "nn": nn})


def _ext() -> BiasExtension:
    return PandasBisaExtesion(TreatmentRole(), [TargetRole()])


def _pd_ds(df: pd.DataFrame) -> Dataset:
    return Dataset(roles=dict(ROLES), data=df, backend=BackendsEnum.pandas)


def test_backend_factory_resolves_pandas() -> None:
    ds = _pd_ds(_frame())
    assert backend_factory.resolve_backend(BiasExtension, ds) is PandasBisaExtesion


def test_extract_info_without_neighbors_raises() -> None:
    ds = Dataset(
        roles={"treat": TreatmentRole(), "x": FeatureRole(), "y": TargetRole()},
        data=_frame().drop(columns="nn"),
        backend=BackendsEnum.pandas,
    )
    with pytest.raises(ValueError, match="No indexes"):
        BiasExtension._extract_info(ds)


def test_calc_exact_linear_relation_gives_known_bias() -> None:
    df = _frame()
    out = _ext().calc(_pd_ds(df))
    res = out.raw_data
    assert list(res.columns) == ["bias", "matched_target"]
    # y = 2x + 3*treat: within a group the matched y is exactly linear in matched x
    # with slope 2, so bias = (x_matched - x) * 2
    x, nn = df.x.to_numpy(), df.nn.to_numpy()
    np.testing.assert_allclose(res["bias"].to_numpy(), (x[nn] - x) * 2.0, atol=1e-8)
    np.testing.assert_allclose(res["matched_target"].to_numpy(), df.y.to_numpy()[nn])


def test_prepare_data_drops_dummy_minus_one_matches() -> None:
    df = _frame()
    df.loc[0, "nn"] = -1
    ds = _pd_ds(df)
    _, matched = PandasBisaExtesion._prepare_data(ds, ["nn"], ["x", "y"])
    assert 0 not in matched.index
    assert list(matched.columns) == ["x_matched", "y_matched"]
    assert len(matched) == len(df) - 1


def test_prepare_data_all_invalid_returns_empty_matched() -> None:
    ds = _pd_ds(_frame(nn=-1))
    _, matched = PandasBisaExtesion._prepare_data(ds, "nn", ["x", "y"])
    assert matched.empty
    assert list(matched.columns) == ["x_matched", "y_matched"]


def test_calc_no_valid_matches_warns_and_returns_nan() -> None:
    ds = _pd_ds(_frame(nn=-1))
    with pytest.warns(UserWarning, match="no valid matches"):
        out = _ext().calc(ds)
    res = out.raw_data
    assert len(res) == 20
    assert res["bias"].isna().all()
    assert res["matched_target"].isna().all()


def test_calc_partial_matches_warns_and_keeps_nan_target() -> None:
    df = _frame()
    df.loc[[0, 15], "nn"] = -1
    with pytest.warns(UserWarning, match="2 of 20 observations"):
        out = _ext().calc(_pd_ds(df))
    res = out.raw_data
    assert res.loc[[0, 15], "matched_target"].isna().all()
    assert (res.loc[[0, 15], "bias"] == 0.0).all()
    assert res["matched_target"].drop([0, 15]).notna().all()


def test_calc_coefs_too_few_rows_returns_zeros() -> None:
    ext = _ext()
    ext.group_field, ext.target_field, ext.features = "treat", "y", ["x"]
    data = pd.DataFrame(
        {
            "treat": [0, 1, 1, 1],
            "x_matched": [1.0, 1.0, 2.0, 3.0],
            "y_matched": [1.0, 2.0, 4.0, 6.0],
        }
    )
    coefs = ext._calc_coefs(data)
    assert coefs.shape == (2, 1)
    assert coefs[0, 0] == 0.0  # one control row: below n_features + 1
    assert coefs[1, 0] == pytest.approx(2.0)


def test_calc_coefs_ignores_nan_and_inf_rows() -> None:
    ext = _ext()
    ext.group_field, ext.target_field, ext.features = "treat", "y", ["x"]
    data = pd.DataFrame(
        {
            "treat": [0, 0, 0, 0, 1, 1, 1],
            "x_matched": [1.0, 2.0, 3.0, np.nan, 1.0, 2.0, np.inf],
            "y_matched": [3.0, 5.0, 7.0, 9.0, 1.0, 1.0, 5.0],
        }
    )
    coefs = ext._calc_coefs(data)
    assert coefs[0, 0] == pytest.approx(2.0)
    # treated: inf row is removed by the finite mask leaving 2 rows (>= 2 needed)
    assert coefs[1, 0] == pytest.approx(0.0, abs=1e-12)


def test_calc_coefs_all_infinite_returns_zeros() -> None:
    ext = _ext()
    ext.group_field, ext.target_field, ext.features = "treat", "y", ["x"]
    data = pd.DataFrame(
        {
            "treat": [0, 0, 1, 1],
            "x_matched": [np.inf, np.inf, 1.0, 2.0],
            "y_matched": [1.0, 2.0, 1.0, 3.0],
        }
    )
    coefs = ext._calc_coefs(data)
    assert coefs[0, 0] == 0.0
    assert coefs[1, 0] == pytest.approx(2.0)


def test_calc_coefs_linalg_error_returns_zeros(monkeypatch) -> None:
    def boom(*a, **k):
        raise np.linalg.LinAlgError("no convergence")

    monkeypatch.setattr(np.linalg, "lstsq", boom)
    ext = _ext()
    ext.group_field, ext.target_field, ext.features = "treat", "y", ["x"]
    data = pd.DataFrame(
        {
            "treat": [0, 0, 1, 1],
            "x_matched": [1.0, 2.0, 1.0, 2.0],
            "y_matched": [1.0, 2.0, 1.0, 3.0],
        }
    )
    assert (ext._calc_coefs(data) == 0.0).all()


@pytest.mark.spark
def test_spark_bias_matches_pandas(spark_session) -> None:
    df = _frame()
    ds = Dataset(
        roles=dict(ROLES), data=df, backend=BackendsEnum.spark, session=spark_session
    )
    cls = backend_factory.resolve_backend(BiasExtension, ds)
    assert cls.__name__ == "SparkBisaExtesion"
    ext = cls(TreatmentRole(), [TargetRole()])
    out = ext.calc(ds)
    res = out.raw_data
    res = res.to_pandas() if hasattr(res, "to_pandas") else res
    res = res.sort_index()
    x, nn = df.x.to_numpy(), df.nn.to_numpy()
    # regParam=0.01 shrinks the Spark coefficients slightly: allow loose tolerance
    np.testing.assert_allclose(res["bias"].to_numpy(), (x[nn] - x) * 2.0, atol=0.2)
    np.testing.assert_allclose(res["matched_target"].to_numpy(), df.y.to_numpy()[nn])


@pytest.mark.spark
@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="SparkBisaExtesion.prepare_data calls result_to_dataset without the "
    "required `roles` argument",
)
def test_spark_prepare_data_returns_indexes_and_matched(spark_session) -> None:
    df = _frame()
    ds = Dataset(
        roles=dict(ROLES), data=df, backend=BackendsEnum.spark, session=spark_session
    )
    cls = backend_factory.resolve_backend(BiasExtension, ds)
    indexes, matched = cls.prepare_data(ds)
    assert list(indexes.columns) == ["nn"]
    mdf = matched.raw_data
    mdf = mdf.to_pandas() if hasattr(mdf, "to_pandas") else mdf
    mdf = mdf.sort_index()
    np.testing.assert_allclose(
        mdf["x_matched"].to_numpy(), df.x.to_numpy()[df.nn.to_numpy()]
    )

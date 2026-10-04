"""Tests for CUPACExecutor and CupacExtension."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from hypex.dataset import (
    Dataset,
    ExperimentData,
    FeatureRole,
    PreTargetRole,
    TargetRole,
    TreatmentRole,
)
from hypex.extensions.cupac import CupacExtension
from hypex.ml import CUPACExecutor
from hypex.utils import ID_SPLIT_SYMBOL, BackendsEnum
from hypex.utils.cuped_theta import cuped_theta
from hypex.utils.models import CUPAC_MODELS

TOL = 1e-6


def _frames(n: int = 400, noise: float = 0.5, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"0": rng.normal(0, 1, n), "1": rng.normal(0, 1, n)})
    y = pd.DataFrame({"t": 3.0 * X["0"] - 2.0 * X["1"] + rng.normal(0, noise, n)})
    return X, y


def _ds(df: pd.DataFrame, role) -> Dataset:
    return Dataset(
        roles={c: role() for c in df.columns},
        data=df.copy(),
        backend=BackendsEnum.pandas,
    )


# ---------------------------------------------------------------------------
# cuped_theta helper
# ---------------------------------------------------------------------------
def test_cuped_theta_is_cov_over_var() -> None:
    rng = np.random.RandomState(0)
    x = rng.normal(5, 2, 500)
    y = 1.5 * x + rng.normal(0, 1, 500)
    assert cuped_theta(y, x) == pytest.approx(
        np.cov(y, x, ddof=0)[0, 1] / x.var(), abs=TOL
    )


def test_cuped_theta_zero_variance_covariate() -> None:
    assert cuped_theta([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]) == 0.0


def test_cuped_theta_is_stable_for_large_mean() -> None:
    rng = np.random.RandomState(1)
    x = 1e9 + rng.normal(0, 1, 1000)
    y = 2.0 * (x - 1e9) + rng.normal(0, 0.1, 1000)
    assert cuped_theta(y, x) == pytest.approx(2.0, abs=0.05)


# ---------------------------------------------------------------------------
# Variance reduction helper
# ---------------------------------------------------------------------------
def test_variance_reduction_percentage() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert CupacExtension._calculate_variance_reduction(y, y * 0.5) == pytest.approx(
        75.0
    )


def test_variance_reduction_never_negative() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert CupacExtension._calculate_variance_reduction(y, y * 2) == 0.0


def test_variance_reduction_constant_target_is_zero() -> None:
    y = np.ones(10)
    assert CupacExtension._calculate_variance_reduction(y, y) == 0.0


# ---------------------------------------------------------------------------
# Fold importances
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["linear", "ridge", "lasso"])
def test_linear_models_report_coefficients(name) -> None:
    X, y = _frames()
    model = CUPAC_MODELS[name]["pandasdataset"]
    from sklearn.base import clone

    fitted = clone(model).fit(X, y["t"])
    imp = CupacExtension._extract_fold_importances(fitted, name, ["0", "1"])
    assert imp == {
        "0": pytest.approx(float(fitted.coef_[0])),
        "1": pytest.approx(float(fitted.coef_[1])),
    }


def test_unknown_model_gives_no_importances() -> None:
    assert CupacExtension._extract_fold_importances(object(), "mystery", ["a"]) == {}


# ---------------------------------------------------------------------------
# kfold_fit / fit / predict
# ---------------------------------------------------------------------------
def test_kfold_fit_matches_manual_oof_computation() -> None:
    X, y = _frames()
    extension = CupacExtension(n_folds=5, random_state=0)
    var_red, importances = extension._calc_pandas(
        data=_ds(X, FeatureRole), mode="kfold_fit", model="linear", Y=_ds(y, TargetRole)
    )

    oof = np.full(len(y), np.nan)
    coefs = []
    for train, val in KFold(5, shuffle=True, random_state=0).split(X):
        m = LinearRegression().fit(X.iloc[train], y["t"].iloc[train])
        oof[val] = m.predict(X.iloc[val])
        coefs.append(m.coef_)
    yv = y["t"].to_numpy()
    theta = cuped_theta(yv, oof)
    adjusted = yv - theta * (oof - oof.mean())
    expected = (1 - adjusted.var() / yv.var()) * 100
    assert var_red == pytest.approx(expected, abs=TOL)
    assert importances["0"] == pytest.approx(np.mean([c[0] for c in coefs]), abs=TOL)
    assert importances["1"] == pytest.approx(np.mean([c[1] for c in coefs]), abs=TOL)


def test_kfold_fit_strong_signal_has_high_variance_reduction() -> None:
    X, y = _frames(noise=0.1)
    var_red, _ = CupacExtension(random_state=0)._calc_pandas(
        data=_ds(X, FeatureRole), mode="kfold_fit", model="linear", Y=_ds(y, TargetRole)
    )
    assert var_red > 95.0


def test_kfold_fit_uninformative_features_have_near_zero_reduction() -> None:
    rng = np.random.RandomState(3)
    X = pd.DataFrame({"0": rng.normal(0, 1, 400), "1": rng.normal(0, 1, 400)})
    y = pd.DataFrame({"t": rng.normal(0, 1, 400)})
    var_red, _ = CupacExtension(random_state=0)._calc_pandas(
        data=_ds(X, FeatureRole), mode="kfold_fit", model="linear", Y=_ds(y, TargetRole)
    )
    assert 0.0 <= var_red < 5.0


def test_kfold_fit_is_reproducible_with_seed() -> None:
    X, y = _frames()
    results = [
        CupacExtension(random_state=7)._calc_pandas(
            data=_ds(X, FeatureRole),
            mode="kfold_fit",
            model="ridge",
            Y=_ds(y, TargetRole),
        )
        for _ in range(2)
    ]
    assert results[0][0] == results[1][0]
    assert results[0][1] == results[1][1]


def test_fit_then_predict_recovers_linear_signal() -> None:
    X, y = _frames(noise=0.01)
    extension = CupacExtension(random_state=0)
    model = extension._calc_pandas(
        data=_ds(X, FeatureRole), mode="fit", model="linear", Y=_ds(y, TargetRole)
    )
    pred = extension._calc_pandas(data=_ds(X, FeatureRole), mode="predict", model=model)
    assert list(pred.columns) == ["predict"]
    np.testing.assert_allclose(
        pred.backend_data.data["predict"].to_numpy(), y["t"].to_numpy(), atol=0.1
    )


def test_unknown_model_name_raises_key_error() -> None:
    X, y = _frames()
    with pytest.raises(KeyError):
        CupacExtension()._calc_pandas(
            data=_ds(X, FeatureRole),
            mode="kfold_fit",
            model="nope",
            Y=_ds(y, TargetRole),
        )


@pytest.mark.catboost
def test_catboost_kfold_importances() -> None:
    pytest.importorskip("catboost")
    X, y = _frames(n=200, noise=0.1)
    var_red, importances = CupacExtension(n_folds=3, random_state=0)._calc_pandas(
        data=_ds(X, FeatureRole),
        mode="kfold_fit",
        model="catboost",
        Y=_ds(y, TargetRole),
    )
    assert var_red > 50.0
    assert sum(importances.values()) == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# CUPACExecutor
# ---------------------------------------------------------------------------
def test_default_models_are_all_available() -> None:
    executor = CUPACExecutor()
    executor._validate_models()
    assert executor.cupac_models == list(CUPAC_MODELS)


def test_single_model_string_becomes_list() -> None:
    executor = CUPACExecutor(cupac_models="linear")
    executor._validate_models()
    assert executor.cupac_models == ["linear"]


def test_unknown_model_is_rejected() -> None:
    executor = CUPACExecutor(cupac_models=["linear", "bogus"])
    with pytest.raises(ValueError, match="bogus"):
        executor._validate_models()


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="Issue: _validate_models lower-cases the name for the membership check but then "
    "indexes CUPAC_MODELS with the original case",
)
def test_model_names_are_case_insensitive() -> None:
    CUPACExecutor(cupac_models="Linear")._validate_models()


def _temporal_experiment() -> ExperimentData:
    rng = np.random.RandomState(0)
    n = 300
    f = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "y": f + rng.normal(0, 0.3, n),
            "y_l1": f + rng.normal(0, 0.3, n),
            "y_l2": f + rng.normal(0, 0.3, n),
            "f_l1": f,
            "f_l2": f + 0.1,
            "g": rng.randint(0, 2, n),
        }
    )
    roles = {
        "y": TargetRole(cofounders=["f"]),
        "y_l1": PreTargetRole(parent="y", lag=1),
        "y_l2": PreTargetRole(parent="y", lag=2),
        "f_l1": FeatureRole(parent="f", lag=1),
        "f_l2": FeatureRole(parent="f", lag=2),
        "g": TreatmentRole(),
    }
    return ExperimentData(Dataset(roles=roles, data=df, backend=BackendsEnum.pandas))


def test_prepare_data_builds_train_and_predict_structures() -> None:
    prepared = CUPACExecutor._prepare_data(_temporal_experiment())
    assert prepared == {
        "y": {
            "X_train": [["f_l2"], ["y_l2"]],
            "Y_train": ["y_l1"],
            "X_predict": [["f_l1"], ["y_l1"]],
        }
    }


def test_prepare_data_requires_lagged_history() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    ed = ExperimentData(
        Dataset(
            roles={"y": TargetRole(cofounders=[])}, data=df, backend=BackendsEnum.pandas
        )
    )
    with pytest.raises(ValueError, match="no lag periods"):
        CUPACExecutor._prepare_data(ed)


def test_agg_data_standardises_column_names() -> None:
    ed = _temporal_experiment()
    X = CUPACExecutor._agg_data_from_cupac_data(ed, [["f_l2"]])
    assert list(X.columns) == ["0"]
    assert len(X) == 300


def test_execute_adds_cupac_report_and_adjusted_target() -> None:
    ed = _temporal_experiment()
    executor = CUPACExecutor(cupac_models=["linear"], random_state=0)
    out = executor.execute(ed)
    report = out.analysis_tables[f"{executor.id}{ID_SPLIT_SYMBOL}y"]
    row = report.backend_data.data.iloc[0]
    assert row["cupac_best_model"] == "linear"
    assert row["cupac_variance_reduction_cv"] > 50


def test_public_calc_dispatches_to_pandas_implementation() -> None:
    X, y = _frames()
    CupacExtension(random_state=0).calc(
        data=_ds(X, FeatureRole), mode="kfold_fit", model="linear", Y=_ds(y, TargetRole)
    )

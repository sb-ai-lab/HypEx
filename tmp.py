from __future__ import annotations

import os
from pathlib import Path

from hypex import ABTest
from hypex.dataset import (
    Dataset,
    FeatureRole,
    InfoRole,
    PreTargetRole,
    TargetRole,
    TreatmentRole,
)
from hypex.utils.tutorial_data_creation import DataGenerator


def build_real_target_data(seed: int = 42) -> Dataset:
    gen = DataGenerator(
        n_samples=1_000,
        distributions={
            "X1": {"type": "normal", "mean": 1, "std": 1},
            "X2": {"type": "bernoulli", "p": 0.5},
            "y0": {"type": "normal", "mean": 1, "std": 5},
        },
        time_correlations={"X1": 0.2, "X2": 0.1, "y0": 0.8},
        effect_size=0.1,
        seed=seed,
    )

    df = gen.generate()
    df = df.drop(columns=["y0", "z", "U", "D", "y1"])
    df = df.rename(columns={"y0_lag_1": "y_lag1", "y0_lag_2": "y_lag2"})

    return Dataset(
        roles={
            "d": TreatmentRole(),
            "y": TargetRole(cofounders=["X1", "X2"]),
            "y_lag1": PreTargetRole(parent="y", lag=1),
            "X1_lag1": FeatureRole(parent="X1", lag=1),
            "X2_lag1": FeatureRole(parent="X2", lag=1),
            "y_lag2": PreTargetRole(parent="y", lag=2),
            "X1_lag2": FeatureRole(parent="X1", lag=2),
            "X2_lag2": FeatureRole(parent="X2", lag=2),
        },
        data=df,
        default_role=InfoRole(),
    )


def build_virtual_target_data(seed: int = 42) -> Dataset:
    gen = DataGenerator(
        n_samples=1_000,
        distributions={
            "X1": {"type": "normal", "mean": 1, "std": 1},
            "X2": {"type": "bernoulli", "p": 0.5},
            "y0": {"type": "normal", "mean": 1, "std": 5},
        },
        time_correlations={"X1": 0.2, "X2": 0.1, "y0": 0.8},
        effect_size=0.1,
        seed=seed,
    )

    df = gen.generate()
    df = df.drop(columns=["y0", "z", "U", "D", "y1", "y"])
    df = df.rename(columns={"y0_lag_1": "y_lag1", "y0_lag_2": "y_lag2"})

    return Dataset(
        roles={
            "d": TreatmentRole(),
            "y_lag1": PreTargetRole(parent="y", cofounders=["X1", "X2"], lag=1),
            "X1_lag1": FeatureRole(parent="X1", lag=1),
            "X2_lag1": FeatureRole(parent="X2", lag=1),
            "y_lag2": PreTargetRole(parent="y", lag=2),
            "X1_lag2": FeatureRole(parent="X1", lag=2),
            "X2_lag2": FeatureRole(parent="X2", lag=2),
        },
        data=df,
        default_role=InfoRole(),
    )


def latest_experiment_id() -> str:
    root = Path(".hypex_experiments")
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise RuntimeError("No experiments found in .hypex_experiments")
    return max(dirs, key=lambda p: p.stat().st_mtime).name


def smoke_baseline_ab() -> None:
    data = build_real_target_data(seed=1)
    result = ABTest().execute(data)
    assert result.resume is not None
    print("[OK] baseline ABTest")


def smoke_cupac_fit_predict() -> None:
    data = build_real_target_data(seed=2)
    result = ABTest(cupac_mode="fit_predict").execute(data)
    assert result.resume is not None
    assert result.cupac.variance_reductions is not None
    print("[OK] CUPAC fit_predict")


def smoke_cupac_fit_then_predict() -> None:
    fit_data = build_virtual_target_data(seed=3)
    ABTest(cupac_mode="fit").execute(fit_data)

    exp_id = latest_experiment_id()

    pred_data = build_real_target_data(seed=4)
    result = ABTest(
        cupac_mode="predict",
        experiment_id=exp_id,
    ).execute(pred_data)

    assert result.resume is not None
    assert result.cupac.variance_reductions is not None
    print(f"[OK] CUPAC fit->predict (exp_id={exp_id})")


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    smoke_baseline_ab()
    smoke_cupac_fit_predict()
    smoke_cupac_fit_then_predict()
    print("All smoke checks passed")


if __name__ == "__main__":
    main()

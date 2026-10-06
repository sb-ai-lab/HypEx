"""Shared helpers for scenario tests (synthetic data with a known truth)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hypex.dataset import (
    Dataset,
    FeatureRole,
    InfoRole,
    TargetRole,
    TreatmentRole,
)
from hypex.utils import BackendsEnum

AB_ROLES = {
    "id": InfoRole(),
    "treat": TreatmentRole(),
    "x": FeatureRole(),
    "y": TargetRole(),
}


def ab_frame(
    n: int = 600,
    effect: float = 0.0,
    groups: int = 2,
    seed: int = 0,
    noise: float = 1.0,
    confounded: bool = False,
) -> pd.DataFrame:
    """Frame with ``treat`` (0..groups-1), covariate ``x`` and outcome ``y``.

    Group ``g >= 1`` gets ``effect * g`` added to the outcome. When
    ``confounded`` is True the assignment probability depends on ``x`` (and so
    does ``y``), so the naive difference is biased.
    """
    rng = np.random.RandomState(seed)
    x = rng.normal(0, 1, n)
    if confounded:
        p = 1 / (1 + np.exp(-2 * x))
        treat = (rng.uniform(size=n) < p).astype(int)
    else:
        treat = rng.randint(0, groups, n)
    y = 2.0 * x + effect * treat + rng.normal(0, noise, n)
    return pd.DataFrame({"id": np.arange(n), "treat": treat, "x": x, "y": y})


def make_dataset(
    df: pd.DataFrame,
    roles: dict | None = None,
    backend: BackendsEnum = BackendsEnum.pandas,
    session=None,
) -> Dataset:
    roles = {c: r for c, r in (roles or AB_ROLES).items() if c in df.columns}
    return Dataset(
        roles=roles,
        data=df.copy(),
        backend=backend,
        session=session if backend == BackendsEnum.spark else None,
    )


def to_pandas(ds) -> pd.DataFrame:
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


def resume_row(output, feature: str = "y", group=None) -> pd.Series:
    """Row of ``output.resume`` for ``feature`` (and ``group`` when given)."""
    frame = to_pandas(output.resume)
    mask = frame["feature"] == feature
    if group is not None:
        mask &= frame["group"].astype(str) == str(group)
    return frame[mask].iloc[0]

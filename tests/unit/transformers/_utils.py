"""Helpers for transformer tests."""
from __future__ import annotations

import pandas as pd

from hypex.dataset import Dataset, ExperimentData
from hypex.utils import BackendsEnum


def make_ds(df: pd.DataFrame, roles: dict, backend=BackendsEnum.pandas, session=None) -> Dataset:
    return Dataset(
        roles=roles,
        data=df,
        backend=backend,
        session=session if backend == BackendsEnum.spark else None,
    )


def make_ed(df: pd.DataFrame, roles: dict, backend=BackendsEnum.pandas, session=None) -> ExperimentData:
    return ExperimentData(make_ds(df, roles, backend, session))


def to_pandas(ds) -> pd.DataFrame:
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data

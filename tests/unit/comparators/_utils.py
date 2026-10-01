"""Shared helpers for comparator tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hypex.dataset import (
    Dataset,
    ExperimentData,
    TargetRole,
    TreatmentRole,
)
from hypex.utils import BackendsEnum


def build_dataset(df: pd.DataFrame, roles: dict, backend=BackendsEnum.pandas, session=None) -> Dataset:
    """Create a Dataset on the requested backend."""
    return Dataset(
        roles=roles,
        data=df,
        backend=backend,
        session=session if backend == BackendsEnum.spark else None,
    )


def to_pandas(ds) -> pd.DataFrame:
    """Return the backend data of a Dataset/SmallDataset as pandas."""
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


def result_frame(experiment_data: ExperimentData, executor) -> pd.DataFrame:
    """Pandas frame of ``analysis_tables[executor.id]`` (rows = ``group┆column``)."""
    return to_pandas(experiment_data.analysis_tables[executor.id])


def three_groups_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Three treatment groups (a, b, c) with a continuous target ``y``."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "g": ["a"] * n + ["b"] * n + ["c"] * n,
            "y": np.r_[
                rng.normal(0.0, 1.0, n),
                rng.normal(0.5, 1.5, n),
                rng.normal(1.0, 1.0, n),
            ],
        }
    )


TREAT_TARGET_ROLES = {"g": TreatmentRole(), "y": TargetRole()}

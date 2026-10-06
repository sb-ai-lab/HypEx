"""Synthetic dataset generators for HypEx unit tests.

Provides factory functions that create minimal, deterministic
``pd.DataFrame`` instances for testing edge cases: empty datasets,
single-row datasets, all-NaN columns, constant targets, etc.

All functions return plain ``pd.DataFrame`` objects. Use the
``make_dataset`` fixture from ``conftest.py`` to wrap them into
``Dataset`` instances on the desired backend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_empty_dataset(
    columns: list[str] | None = None,
    n_columns: int = 3,
) -> pd.DataFrame:
    """Create an empty DataFrame with the specified columns.

    Args:
        columns: Explicit column names. If None, generates
            ``n_columns`` columns named ``col_0``, ``col_1``, etc.
        n_columns: Number of columns when ``columns`` is None.
            Ignored if ``columns`` is provided.

    Returns:
        pd.DataFrame: Empty DataFrame (0 rows) with the specified
        column structure. All columns have ``float64`` dtype.

    Example:
        .. code-block:: python

            >>> df = make_empty_dataset(columns=["x", "y"])
            >>> len(df)
            0
            >>> list(df.columns)
            ['x', 'y']
    """
    if columns is None:
        columns = [f"col_{i}" for i in range(n_columns)]
    return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})


def make_single_row_dataset(
    n_features: int = 3,
    treatment_value: int = 1,
    target_value: float = 42.0,
) -> pd.DataFrame:
    """Create a single-row DataFrame with treatment, features, and target.

    Useful for testing edge cases where aggregations like ``std``
    or ``var`` return NaN (ddof=1 with n=1).

    Args:
        n_features: Number of numeric feature columns.
        treatment_value: Value for the treatment column.
        target_value: Value for the target column.

    Returns:
        pd.DataFrame: DataFrame with 1 row and columns:
        ``treatment``, ``target``, ``feature_0``, ..., ``feature_{n-1}``.

    Example:
        .. code-block:: python

            >>> df = make_single_row_dataset(n_features=2)
            >>> df.shape
            (1, 4)
            >>> df["treatment"].iloc[0]
            1
    """
    data: dict[str, list] = {
        "treatment": [treatment_value],
        "target": [target_value],
    }
    for i in range(n_features):
        data[f"feature_{i}"] = [float(i + 1)]
    return pd.DataFrame(data)


def make_all_nan_dataset(
    n_rows: int = 5,
    n_columns: int = 3,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create a DataFrame filled entirely with NaN values.

    Tests how comparators and aggregators handle degenerate input
    where every value is missing.

    Args:
        n_rows: Number of rows.
        n_columns: Number of columns when ``columns`` is None.
        columns: Explicit column names. If None, generates
            ``n_columns`` columns named ``col_0``, ``col_1``, etc.

    Returns:
        pd.DataFrame: DataFrame of shape ``(n_rows, n_columns)``
        filled with ``np.nan``. All columns have ``float64`` dtype.

    Example:
        .. code-block:: python

            >>> df = make_all_nan_dataset(n_rows=3, n_columns=2)
            >>> df.isna().all().all()
            True
    """
    if columns is None:
        columns = [f"col_{i}" for i in range(n_columns)]
    data = {col: [np.nan] * n_rows for col in columns}
    return pd.DataFrame(data)


def make_constant_target_dataset(
    n_rows: int = 10,
    n_groups: int = 2,
    target_value: float = 5.0,
) -> pd.DataFrame:
    """Create a DataFrame with a constant target across all groups.

    The target column has zero variance, which is a degenerate case
    for statistical tests (t-test denominator is zero, CUPED theta
    is zero, etc.).

    Args:
        n_rows: Total number of rows.
        n_groups: Number of distinct treatment groups. Rows are
            distributed as evenly as possible.
        target_value: The constant value assigned to every row
            in the target column.

    Returns:
        pd.DataFrame: DataFrame with columns ``treatment``,
        ``target``, and ``feature_0``. The ``treatment`` column
        cycles through ``0, 1, ..., n_groups-1``. The ``target``
        column is constant. ``feature_0`` contains sequential
        floats for non-degenerate grouping.

    Example:
        .. code-block:: python

            >>> df = make_constant_target_dataset(n_rows=6, n_groups=2)
            >>> df["target"].var()
            0.0
            >>> sorted(df["treatment"].unique())
            [0, 1]
    """
    treatment = [i % n_groups for i in range(n_rows)]
    return pd.DataFrame(
        {
            "treatment": treatment,
            "target": [target_value] * n_rows,
            "feature_0": [float(i) for i in range(n_rows)],
        }
    )


def make_two_group_dataset(
    n_per_group: int = 5,
    control_mean: float = 10.0,
    treatment_mean: float = 12.0,
    std: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Create a two-group dataset with known means and standard deviation.

    Generates normally distributed data for control and treatment
    groups with exactly specified population parameters (approximate
    for small samples due to sampling variability; use large
    ``n_per_group`` for precise means).

    Args:
        n_per_group: Number of rows per group.
        control_mean: Mean of the target in the control group.
        treatment_mean: Mean of the target in the treatment group.
        std: Standard deviation for both groups.
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with columns ``treatment`` (0 or 1),
        ``target``, and ``feature_0``. Total rows = ``2 * n_per_group``.

    Example:
        .. code-block:: python

            >>> df = make_two_group_dataset(n_per_group=100, random_state=42)
            >>> abs(df[df["treatment"]==0]["target"].mean() - 10.0) < 0.5
            True
    """
    rng = np.random.default_rng(random_state)
    n_total = 2 * n_per_group

    treatment = [0] * n_per_group + [1] * n_per_group
    target = np.concatenate(
        [
            rng.normal(control_mean, std, n_per_group),
            rng.normal(treatment_mean, std, n_per_group),
        ]
    )
    feature = rng.normal(0, 1, n_total)

    return pd.DataFrame(
        {
            "treatment": treatment,
            "target": target,
            "feature_0": feature,
        }
    )


def make_categorical_dataset(
    n_rows: int = 20,
    n_categories: int = 3,
) -> pd.DataFrame:
    """Create a dataset with a categorical column for chi-squared tests.

    Args:
        n_rows: Total number of rows.
        n_categories: Number of distinct categories in the
            categorical column.

    Returns:
        pd.DataFrame: DataFrame with columns ``treatment`` (binary),
        ``category`` (string labels ``cat_0``, ``cat_1``, ...),
        and ``feature_0`` (numeric).

    Example:
        .. code-block:: python

            >>> df = make_categorical_dataset(n_rows=30, n_categories=3)
            >>> sorted(df["category"].unique())
            ['cat_0', 'cat_1', 'cat_2']
    """
    rng = np.random.default_rng(42)
    treatment = rng.binomial(1, 0.5, n_rows)
    categories = [f"cat_{rng.integers(0, n_categories)}" for _ in range(n_rows)]
    feature = rng.normal(0, 1, n_rows)

    return pd.DataFrame(
        {
            "treatment": treatment,
            "category": categories,
            "feature_0": feature,
        }
    )

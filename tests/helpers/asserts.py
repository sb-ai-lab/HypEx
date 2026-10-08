"""Assertion helpers for HypEx Dataset comparisons.

Provides functions that compare Dataset instances structurally
and semantically, abstracting away backend differences (pandas
vs Spark) and role metadata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hypex.dataset import ABCRole, Dataset


def assert_datasets_equal(
    actual: Dataset,
    expected: Dataset,
    *,
    check_roles: bool = True,
    check_index: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert that two Dataset instances are structurally and numerically equal.

    Converts both datasets to pandas for comparison, which handles
    Spark-to-pandas conversion transparently. Float comparisons use
    ``np.allclose`` with configurable tolerances.

    Args:
        actual: The dataset produced by the code under test.
        expected: The expected reference dataset.
        check_roles: If True, also compare the roles mapping.
            Roles are compared by class type and data_type attribute.
        check_index: If True, also compare the index values.
            When False (default), only column values are compared.
        rtol: Relative tolerance for float comparisons.
        atol: Absolute tolerance for float comparisons.

    Raises:
        AssertionError: If the datasets differ in shape, column names,
            values, or (optionally) roles or index.

    Example:
        .. code-block:: python

            >>> ds1 = make_dataset(pd.DataFrame({"x": [1.0, 2.0]}), {"x": FeatureRole()})
            >>> ds2 = make_dataset(pd.DataFrame({"x": [1.0, 2.0]}), {"x": FeatureRole()})
            >>> assert_datasets_equal(ds1, ds2)
    """
    # Check shape
    assert actual.shape == expected.shape, (
        f"Shape mismatch: actual={actual.shape}, expected={expected.shape}"
    )

    # Check column names
    actual_cols = sorted(actual.columns)
    expected_cols = sorted(expected.columns)
    assert actual_cols == expected_cols, (
        f"Column mismatch:\n"
        f"  actual:   {actual.columns}\n"
        f"  expected: {expected.columns}"
    )

    # Convert to pandas for value comparison
    actual_pdf = _to_pandas(actual)
    expected_pdf = _to_pandas(expected)

    # Compare values column by column
    for col in expected.columns:
        actual_series = actual_pdf[col]
        expected_series = expected_pdf[col]

        if actual_series.dtype.kind == "f" or expected_series.dtype.kind == "f":
            # Float comparison with tolerance
            actual_arr = actual_series.to_numpy(dtype=float, na_value=np.nan)
            expected_arr = expected_series.to_numpy(dtype=float, na_value=np.nan)

            # Handle NaN positions: both must be NaN or both must be close
            actual_nan = np.isnan(actual_arr)
            expected_nan = np.isnan(expected_arr)

            assert np.array_equal(actual_nan, expected_nan), (
                f"NaN pattern mismatch in column '{col}':\n"
                f"  actual NaN count:   {actual_nan.sum()}\n"
                f"  expected NaN count: {expected_nan.sum()}"
            )

            # Compare non-NaN values
            valid_mask = ~actual_nan
            if valid_mask.any():
                assert np.allclose(
                    actual_arr[valid_mask],
                    expected_arr[valid_mask],
                    rtol=rtol,
                    atol=atol,
                ), (
                    f"Value mismatch in column '{col}':\n"
                    f"  actual:   {actual_arr[valid_mask][:5]}...\n"
                    f"  expected: {expected_arr[valid_mask][:5]}..."
                )
        else:
            # Non-float: exact comparison (with NaN-aware equality)
            actual_list = actual_series.tolist()
            expected_list = expected_series.tolist()
            assert actual_list == expected_list, (
                f"Value mismatch in column '{col}':\n"
                f"  actual:   {actual_list[:5]}...\n"
                f"  expected: {expected_list[:5]}..."
            )

    # Optionally check index
    if check_index:
        actual_idx = list(actual_pdf.index)
        expected_idx = list(expected_pdf.index)
        assert actual_idx == expected_idx, (
            f"Index mismatch:\n"
            f"  actual:   {actual_idx[:5]}...\n"
            f"  expected: {expected_idx[:5]}..."
        )

    # Optionally check roles
    if check_roles:
        assert_roles_match(actual, expected)


def assert_roles_match(
    actual: Dataset,
    expected: Dataset,
    *,
    check_data_type: bool = True,
) -> None:
    """Assert that two Dataset instances have identical role assignments.

    Compares roles column-by-column. Two roles match if they are
    instances of the same class and (optionally) have the same
    ``data_type`` attribute.

    Args:
        actual: The dataset produced by the code under test.
        expected: The expected reference dataset.
        check_data_type: If True, also compare the ``data_type``
            attribute of each role. When False, only the role class
            is compared.

    Raises:
        AssertionError: If any column has a different role class
            or (optionally) a different data_type.

    Example:
        .. code-block:: python

            >>> from hypex.dataset import FeatureRole, TargetRole
            >>> ds1 = make_dataset(pd.DataFrame({"x": [1]}), {"x": FeatureRole()})
            >>> ds2 = make_dataset(pd.DataFrame({"x": [1]}), {"x": TargetRole()})
            >>> assert_roles_match(ds1, ds2)  # raises AssertionError
    """
    actual_roles = actual.roles
    expected_roles = expected.roles

    # Check that the same columns exist in both role mappings
    actual_cols = set(actual_roles.keys())
    expected_cols = set(expected_roles.keys())
    assert actual_cols == expected_cols, (
        f"Role column mismatch:\n"
        f"  actual roles for:   {sorted(actual_cols)}\n"
        f"  expected roles for: {sorted(expected_cols)}"
    )

    for col in expected_cols:
        actual_role = actual_roles[col]
        expected_role = expected_roles[col]

        # Check role class
        assert type(actual_role) is type(expected_role), (
            f"Role class mismatch for column '{col}':\n"
            f"  actual:   {type(actual_role).__name__}\n"
            f"  expected: {type(expected_role).__name__}"
        )

        # Check data_type attribute
        if check_data_type:
            actual_dt = getattr(actual_role, "data_type", None)
            expected_dt = getattr(expected_role, "data_type", None)
            assert actual_dt == expected_dt, (
                f"Role data_type mismatch for column '{col}':\n"
                f"  actual:   {actual_dt}\n"
                f"  expected: {expected_dt}"
            )


def assert_dataset_has_roles(
    dataset: Dataset,
    expected_roles: dict[str, type[ABCRole]],
) -> None:
    """Assert that a Dataset has specific roles for specific columns.

    A convenience wrapper that checks a subset of role assignments
    without requiring the full expected Dataset.

    Args:
        dataset: The dataset to inspect.
        expected_roles: Mapping of ``{column_name: role_class}``
            to verify. Only the specified columns are checked;
            other columns in the dataset are ignored.

    Raises:
        AssertionError: If any specified column is missing or has
            a role of the wrong class.

    Example:
        .. code-block:: python

            >>> from hypex.dataset import FeatureRole, TargetRole
            >>> ds = make_dataset(
            ...     pd.DataFrame({"x": [1], "y": [2]}),
            ...     {"x": FeatureRole(), "y": TargetRole()}
            ... )
            >>> assert_dataset_has_roles(ds, {"x": FeatureRole, "y": TargetRole})
    """
    for col, expected_role_cls in expected_roles.items():
        assert col in dataset.roles, (
            f"Column '{col}' not found in dataset roles. "
            f"Available: {list(dataset.roles.keys())}"
        )
        actual_role = dataset.roles[col]
        assert isinstance(actual_role, expected_role_cls), (
            f"Column '{col}' has role {type(actual_role).__name__}, "
            f"expected {expected_role_cls.__name__}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_pandas(dataset: Dataset) -> pd.DataFrame:
    """Convert a Dataset's underlying data to a pandas DataFrame.

    Handles both PandasDataset and SparkDataset backends transparently.

    Args:
        dataset: The dataset to convert.

    Returns:
        pd.DataFrame: The data as a pandas DataFrame.
    """
    backend_data = dataset.backend_data
    if hasattr(backend_data, "to_pandas"):
        # SparkDataset -> ps.DataFrame -> pd.DataFrame
        return backend_data.data.to_pandas()
    if hasattr(backend_data, "data"):
        # PandasDataset wraps a pd.DataFrame
        return backend_data.data
    # Fallback: try direct conversion
    return pd.DataFrame(dataset.data)

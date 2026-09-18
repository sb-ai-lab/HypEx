"""Transformer that downcasts float64 columns to float32 for memory efficiency."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..dataset.dataset import Dataset
from ..dataset.experiment_data import ExperimentData
from ..dataset.roles import ABCRole, FeatureRole, TargetRole
from ..utils import BackendsEnum
from ..utils.adapter import Adapter
from .abstract import Transformer


class Float32Caster(Transformer):
    """Downcasts numeric columns from float64 to float32.

    Reduces memory footprint (~2×) with negligible precision loss for
    most A/B-test metrics.  Operates on columns identified by semantic
    roles, not by name.

    For the **Spark** backend the cast is performed via
    ``pyspark.pandas.DataFrame.astype("float32")`` which maps to
    Spark ``FloatType``.  For **Pandas** the standard
    ``DataFrame.astype(np.float32)`` path is used.

    Args:
        target_roles: Role(s) identifying columns to downcast.
            Defaults to ``[FeatureRole(), TargetRole()]``.
        columns: Explicit column names to cast.  When provided,
            *target_roles* is ignored.  Defaults to ``None``
            (role-based discovery).
        key: Optional executor identifier.

    Example:
        .. code-block:: python

            from hypex.transformers import Float32Caster
            from hypex.dataset import FeatureRole, TargetRole

            caster = Float32Caster(
                target_roles=[FeatureRole(), TargetRole()],
            )
            result = caster.execute(experiment_data)
    """

    def __init__(
        self,
        target_roles: ABCRole | Sequence[ABCRole] | None = None,
        columns: str | Sequence[str] | None = None,
        key: Any = "",
    ) -> None:
        super().__init__(key=key)
        self.target_roles = (
            Adapter.to_list(target_roles)
            if target_roles is not None
            else [FeatureRole(), TargetRole()]
        )
        self.columns = Adapter.to_list(columns) if columns is not None else None

    @property
    def search_types(self) -> list[type] | None:
        """Only float64 columns are eligible for downcasting."""
        return [float]

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: list[str],
    ) -> Dataset:
        """Cast *target_cols* to float32.

        Args:
            data: Input dataset.
            target_cols: Column names to downcast.

        Returns:
            Dataset with float32 columns and updated role types.
        """
        if not target_cols:
            return data

        # Build per-column cast mapping.
        # np.float32 works for pandas; "float32" string is safer for
        # pyspark.pandas.
        if data.backend_type == BackendsEnum.spark:
            dtype_map: dict[str, Any] = {col: "float32" for col in target_cols}
        else:
            dtype_map = {col: np.float32 for col in target_cols}

        return data.astype(dtype=dtype_map)

    def execute(self, data: ExperimentData) -> ExperimentData:
        """Run float32 downcasting on the experiment dataset.

        Args:
            data: The experiment data container.

        Returns:
            Updated ``ExperimentData`` with downcasted columns.
        """
        if self.columns is not None:
            target_cols = [
                c for c in self.columns if c in data.ds.columns
            ]
        else:
            target_cols = data.ds.search_columns(
                roles=self.target_roles,
                search_types=self.search_types,
            )

        if not target_cols:
            return data

        result = data.copy(
            data=self.calc(data=data.ds, target_cols=target_cols),
        )
        return result

    @classmethod
    def calc(cls, data: Dataset, target_cols: list[str] | None = None, **kwargs) -> Dataset:
        """Stateless entry point for use outside the pipeline.

        Args:
            data: Input dataset.
            target_cols: Columns to downcast.  When ``None``, all
                float64 columns are selected.
            **kwargs: Forwarded to ``_inner_function``.

        Returns:
            Dataset with float32 columns.
        """
        if target_cols is None:
            target_cols = data.search_columns_by_type(float)
        return cls._inner_function(data, target_cols, **kwargs)
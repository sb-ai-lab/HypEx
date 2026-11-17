from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from ..dataset.dataset import Dataset
from ..dataset.experiment_data import ExperimentData
from ..dataset.roles import ABCRole, FeatureRole
from .abstract import Transformer


class TypeCaster(Transformer):
    """Cast column data types to specified types.

    This transformer converts column data types according to the provided
    mapping. It supports casting by column name or by data type, with
    optional automatic downcasting of float/decimal columns to float32.

    Args:
        dtype: Mapping of column names or data types to target types.
        roles: Role(s) identifying columns to cast. Defaults to FeatureRole().
        downcasting: Whether to automatically downcast float/decimal to float32. Defaults to True.
        key: Optional executor identifier.

    Example:
        >>> from hypex.transformers import TypeCaster
        >>> from hypex.dataset import FeatureRole
        >>> caster = TypeCaster(dtype={"col1": str, "col2": int})
        >>> result = caster.execute(experiment_data)
    """

    def __init__(
        self,
        dtype: dict[str, type] | dict[type, type],
        roles: ABCRole | Sequence[ABCRole] | None = None,
        downcasting: bool = True,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.dtype = dtype
        self.roles = roles or FeatureRole()
        self.downcasting = downcasting

    @staticmethod
    def _inner_function(
        data: Dataset,
        dtype: dict[str, type],
    ) -> Dataset:
        return data.astype(dtype=dtype)

    @staticmethod
    def _downcast(
        data: Dataset,
    ) -> Dataset:
        """Downcast float/decimal columns to float32.

        Args:
            data: Input dataset.

        Returns:
            Dataset with downcasted columns.
        """
        double_cols = [
            col
            for col, c_type in data.roles.items()
            if c_type.data_type is Decimal or c_type.data_type is float
        ]
        return data.astype({col: float for col in double_cols})

    @classmethod
    def calc(
        cls,
        data: Dataset,
        dtype: dict[str, type] | dict[type, type],
        roles: ABCRole | Sequence[ABCRole] | None = None,
        **kwargs,
    ):
        """Stateless entry point for type casting.

        Args:
            data: Input dataset.
            dtype: Mapping of column names or data types to target types.
            roles: Role(s) identifying columns to cast.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset with casted columns.
        """
        downcasting = kwargs.pop("downcasting", True)
        if downcasting:
            data = TypeCaster._downcast(data=data)

        cast_mapping = {}
        for k, v in dtype.items():
            if isinstance(k, str):
                cast_mapping[k] = v
            elif isinstance(k, type):
                cast_mapping.update({c: v for c in data.search_columns_by_type(k)})
        if roles:
            target_cols = data.search_columns(roles=roles)
            cast_mapping = {c: v for c, v in cast_mapping.items() if c in target_cols}

        return cls._inner_function(data, cast_mapping, **kwargs)

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(
            data=self.calc(
                data=data.ds,
                dtype=self.dtype,
                roles=self.roles,
                downcasting=self.downcasting,
            )
        )
        return result

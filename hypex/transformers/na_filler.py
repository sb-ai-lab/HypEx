from __future__ import annotations

from typing import Any, Literal, Sequence

from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import FeatureRole
from ..utils import ScalarType
from ..utils.adapter import Adapter
from .abstract import Transformer


class NaFiller(Transformer):
    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        values: ScalarType | dict[str, ScalarType] | None = None,
        method: Literal["bfill", "ffill"] | None = None,
        key: Any = "",
    ):
        """
        Initializes a NaFiller object.

        Args:
            target_roles (Optional[Union[str, Sequence[str]]], optional): The roles of the target columns. Defaults to None.
            key (Any, optional): The key for the NaFiller object. Defaults to "".
            values (Union[ScalarType, Dict[str, ScalarType]], optional): The values to fill missing values with. Defaults to None.
            method (Literal["bfill", "ffill"], optional): The method to fill missing values. Defaults to None.

        Returns:
            None
        """

        super().__init__(key=key)
        self.target_roles = target_roles or FeatureRole()
        self.values = values
        self.method = method

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: str | None = None,
        values: ScalarType | dict[str, ScalarType] | None = None,
        method: Literal["bfill", "ffill"] | None = None,
    ) -> Dataset:
        target_cols = Adapter.to_list(target_cols)
        for column in target_cols:
            value = values[column] if isinstance(values, dict) else values
            data[column] = data[column].fillna(values=value, method=method)
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(roles=self.target_roles)
        result = data.copy(
            data=self.calc(
                data=data.ds,
                target_cols=target_cols,
                values=self.values,
                method=self.method,
            )
        )
        return result

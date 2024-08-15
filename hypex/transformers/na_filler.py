from typing import Any, Optional, Union, Sequence, Literal, Dict

from hypex.dataset.dataset import Dataset
from hypex.dataset.dataset import ExperimentData
from hypex.dataset.roles import (
    FeatureRole,
)
from hypex.transformers.abstract import Transformer
from hypex.utils.adapter import Adapter
from hypex.utils import ScalarType


class NaFiller(Transformer):
    def __init__(
        self,
        target_roles: Optional[Union[str, Sequence[str]]] = None,
        values: Union[ScalarType, Dict[str, ScalarType], None] = None,
        method: Optional[Literal["bfill", "ffill"]] = None,
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
        target_cols: Optional[str] = None,
        values: Union[ScalarType, Dict[str, ScalarType], None] = None,
        method: Optional[Literal["bfill", "ffill"]] = None,
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

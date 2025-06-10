from __future__ import annotations

from typing import Any, Sequence

from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import FeatureRole
from ..utils import CategoricalTypes
from ..utils.adapter import Adapter
from .abstract import Transformer


class CategoryAggregator(Transformer):
    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        threshold: int | None = 15,
        new_group_name: str | None = None,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.target_roles = target_roles or FeatureRole()
        self.threshold = threshold
        self.new_group_name = new_group_name

    @property
    def search_types(self):
        return [CategoricalTypes]

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: str | None = None,
        threshold: int | None = 15,
        new_group_name: str | None = None,
    ) -> Dataset:
        target_cols = Adapter.to_list(target_cols)
        for column in target_cols:
            categories_counts = data[column].value_counts()
            values_to_replace = categories_counts[
                categories_counts["count"] < threshold
            ][column].get_values(column=column)
            data[column] = data[column].replace(
                to_replace=values_to_replace, value=new_group_name
            )

        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(
            roles=self.target_roles, search_types=self.search_types
        )
        result = data.copy(
            data=self.calc(
                data=data.ds,
                target_cols=target_cols,
                threshold=self.threshold,
                new_group_name=self.new_group_name,
            )
        )
        return result

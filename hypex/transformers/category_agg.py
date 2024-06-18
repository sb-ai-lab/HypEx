from typing import Any, Optional

from hypex.dataset.dataset import Dataset
from hypex.dataset.dataset import ExperimentData
from hypex.dataset.roles import (
    FeatureRole,
)
from hypex.transformers.abstract import Transformer
from hypex.utils import FieldKeyTypes


class CategoryAggregator(Transformer):
    def __init__(
        self,
        threshold: Optional[int] = 15,
        new_group_name: Optional[str] = None,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.threshold = threshold
        self.new_group_name = new_group_name

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: Optional[FieldKeyTypes] = None,
        threshold: Optional[int] = 15,
        new_group_name: Optional[str] = None,
    ) -> Dataset:
        target_cols = super(CategoryAggregator, CategoryAggregator)._list_unification(
            target_cols
        )
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
        target_cols = data.ds.search_columns(roles=FeatureRole())
        result = data.copy(
            data=self.calc(
                data=data.ds,
                target_cols=target_cols,
                threshold=self.threshold,
                new_group_name=self.new_group_name,
            )
        )
        return result

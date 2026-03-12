from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from ..dataset import Dataset
from ..dataset.roles import FeatureRole
from ..utils import CategoricalTypes
from ..utils.adapter import Adapter
from .ml_transformer import MLTransformer


class MLOneHotEncoder(MLTransformer):
    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        drop_first: bool = True,
        drop_original: bool = True,
        handle_unknown: str = "ignore",
        key: Any = "",
    ):
        super().__init__(
            key=key,
            drop_first=drop_first,
            drop_original=drop_original,
            handle_unknown=handle_unknown,
        )
        self.target_roles = target_roles or FeatureRole()

    @property
    def search_types(self):
        return [CategoricalTypes]

    @staticmethod
    def _fit(
        data: Dataset,
        target_cols: str | Sequence[str] | None = None,
        drop_first: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        target_cols = Adapter.to_list(target_cols)
        categories: dict[str, dict[str, list[Any]]] = {}

        for column in target_cols:
            if column not in data.columns:
                continue

            all_categories = list(data.data[column].dropna().unique())
            encoded_categories = all_categories[1:] if drop_first else all_categories

            categories[column] = {
                "all": all_categories,
                "encoded": encoded_categories,
            }

        return {
            "target_cols": [column for column in target_cols if column in categories],
            "categories": categories,
            "drop_first": drop_first,
        }

    @staticmethod
    def _transform(
        data: Dataset,
        fitted_params: dict[str, Any],
        drop_original: bool = True,
        handle_unknown: str = "ignore",
        **kwargs,
    ) -> Dataset:
        target_cols = fitted_params.get("target_cols", [])
        categories = fitted_params.get("categories", {})

        transformed_df = data.data.copy(deep=True)
        transformed_roles = deepcopy(data.roles)

        for column in target_cols:
            if column not in transformed_df.columns or column not in categories:
                continue

            column_info = categories[column]
            all_categories = set(column_info.get("all", []))
            encoded_categories = column_info.get("encoded", [])

            if handle_unknown == "error":
                unseen = set(transformed_df[column].dropna().unique()) - all_categories
                if unseen:
                    raise ValueError(
                        f"Unknown categories in column '{column}': {sorted(unseen)}"
                    )

            source_role = transformed_roles[column]
            for category in encoded_categories:
                new_column = f"{column}_{category}"
                transformed_df[new_column] = (transformed_df[column] == category).astype(int)
                transformed_roles[new_column] = source_role.asadditional(int)

            if drop_original:
                transformed_df = transformed_df.drop(columns=[column])
                transformed_roles.pop(column, None)

        return Dataset(data=transformed_df, roles=transformed_roles)

    def execute_fit(self, data):
        target_cols = data.ds.search_columns(
            roles=self.target_roles,
            search_types=self.search_types,
        )
        self.calc_kwargs["target_cols"] = target_cols
        return super().execute_fit(data)

    def execute_fit_predict(self, data):
        target_cols = data.ds.search_columns(
            roles=self.target_roles,
            search_types=self.search_types,
        )
        self.calc_kwargs["target_cols"] = target_cols
        return super().execute_fit_predict(data)

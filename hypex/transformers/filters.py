from __future__ import annotations

from typing import Any, Sequence

from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import FeatureRole, InfoRole, PreTargetRole, TargetRole
from ..utils.adapter import Adapter
from .abstract import Transformer


class CVFilter(Transformer):
    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        key: Any = "",
    ):
        """Initialize coefficient of variation filter of the columns in which it does not fit into the defined borders.

        Args:
            lower_bound:
                The minimum acceptable coefficient of variation below which we consider the column to be constant
            upper_bound:
                The maximum acceptable coefficient of variation above which we consider the to be incorrect
        """
        super().__init__(key=key)
        self.target_roles = target_roles or FeatureRole()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.type_filter: bool = True

    @property
    def search_types(self):
        return [float, int, bool]

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: str | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> Dataset:
        target_cols = Adapter.to_list(target_cols)
        for column in target_cols:
            cv = data[column].coefficient_of_variation()
            drop = False
            if (upper_bound and cv > upper_bound) or (lower_bound and cv < lower_bound):
                drop = True
            if drop:
                data.roles[column] = InfoRole()
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        if self.type_filter:
            target_cols = data.ds.search_columns(
                roles=self.target_roles, search_types=self.search_types
            )
        else:
            target_cols = data.ds.search_columns(roles=FeatureRole())
        result = data.copy(
            data=self.calc(
                data=data.ds,
                target_cols=target_cols,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
            )
        )
        return result


class ConstFilter(Transformer):
    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        threshold: float = 0.95,
        key: Any = "",
    ):
        """Initialize constants filter of the values which occur more often than defined by threshold.

        Args:
            target:
                The column or columns to be filtered
            threshold:
                The maximum acceptable frequency above which we consider the column to be constant
        """
        super().__init__(key=key)
        self.target_roles = target_roles or FeatureRole()
        self.threshold = threshold

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: str | None = None,
        threshold: float = 0.95,
    ) -> Dataset:
        target_cols = Adapter.to_list(target_cols)
        for column in target_cols:
            value_counts = data[column].value_counts(normalize=True, sort=True)
            if value_counts.get_values(0, "proportion") > threshold:
                data.roles[column] = InfoRole()
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(roles=self.target_roles)
        result = data.copy(
            data=self.calc(
                data=data.ds, target_cols=target_cols, threshold=self.threshold
            )
        )
        return result


class NanFilter(Transformer):
    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        threshold: float = 0.8,
        key: Any = "",
    ):
        """Initialize filter of the columns in which NaN occurs more often than defined by threshold.

        Args:
            target:
                The column or columns to be filtered
            threshold:
                The maximum acceptable frequency of NaN values in a column
        """
        super().__init__(key=key)
        self.target_roles = target_roles or FeatureRole()
        self.threshold = threshold

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: str | None = None,
        threshold: float = 0.8,
    ) -> Dataset:
        target_cols = Adapter.to_list(target_cols)
        for column in target_cols:
            nan_share = data[column].isna().sum() / len(data)
            if nan_share > threshold:
                data.roles[column] = InfoRole()
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(roles=self.target_roles)
        result = data.copy(
            data=self.calc(
                data=data.ds, target_cols=target_cols, threshold=self.threshold
            )
        )
        return result


class CorrFilter(Transformer):
    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        corr_space_roles: str | Sequence[str] | None = None,
        threshold: float = 0.8,
        method: str = "pearson",
        numeric_only: bool = True,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.target_roles = target_roles or FeatureRole()
        self.corr_space_roles = corr_space_roles or [FeatureRole(), TargetRole()]
        self.threshold = threshold
        self.method = method
        self.numeric_only = numeric_only

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: str | None = None,
        corr_space_cols: str | None = None,
        threshold: float = 0.8,
        method: str = "pearson",
        numeric_only: bool = True,
        drop_policy: str = "cv",
    ) -> Dataset:
        target_cols = Adapter.to_list(target_cols)
        corr_space_cols = Adapter.to_list(corr_space_cols)
        corr_matrix = data[corr_space_cols].corr(
            method=method, numeric_only=numeric_only
        )
        pre_target_column = None
        if drop_policy == "corr":
            pre_target_columns = data.search_columns([PreTargetRole()])
            if (pre_target_columns[0] not in corr_space_cols) | len(
                pre_target_columns
            ) != 1:
                raise ValueError(
                    "Correlation-based filtering cannot be applied if there are more than one PreTarget columns"
                )
            else:
                pre_target_column = pre_target_columns[0]
        corr_target_cols = [
            column for column in target_cols if column in corr_matrix.columns
        ]
        for target in corr_target_cols:
            for column in corr_matrix.columns:
                if (target != column) and (
                    abs(corr_matrix.get_values(row=target, column=column)) > threshold
                ):
                    drop = target
                    if data.roles[column] in corr_target_cols:
                        if drop_policy == "corr":
                            if abs(
                                corr_matrix.get_values(target, pre_target_column)
                            ) > abs(corr_matrix.get_values(column, pre_target_column)):
                                drop = target
                            else:
                                drop = column
                        elif drop_policy == "cv":
                            drop = (
                                target
                                if data[target].coefficient_of_variation()
                                < data[column].coefficient_of_variation()
                                else column
                            )
                    data.roles[drop] = InfoRole()
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(roles=self.target_roles)
        corr_space_cols = data.ds.search_columns(roles=self.corr_space_roles)
        result = data.copy(
            data=self.calc(
                data=data.ds,
                target_cols=target_cols,
                corr_space_cols=corr_space_cols,
                threshold=self.threshold,
                method=self.method,
                numeric_only=self.numeric_only,
            )
        )
        return result


class OutliersFilter(Transformer):
    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        lower_percentile: float = 0,
        upper_percentile: float = 1,
        key: Any = "",
    ):
        """Initialize outliers filter of the values laying beyond the given percentile and NaNs.

        Args:
            target:
                The name of target column to be filtered from outlier values
            percentile:
                The value of the percentile to filter outliers
        """
        super().__init__(key=key)
        self.target_roles = target_roles or FeatureRole()
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    @property
    def search_types(self):
        return [float, int, bool]

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: str | None = None,
        lower_percentile: float = 0,
        upper_percentile: float = 1,
    ) -> Dataset:
        mask = data[target_cols].apply(
            func=lambda x: (x < x.quantile(lower_percentile))
            | (x > x.quantile(upper_percentile)),
            role={column: InfoRole() for column in target_cols},
            axis=0,
        )
        mask = mask.apply(func=lambda x: x.any(), role={"filter": InfoRole()}, axis=1)
        drop_indexes = mask[mask["filter"]].dropna().index
        data = data.drop(drop_indexes, axis=0)
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(
            roles=self.target_roles,
            search_types=self.search_types,
        )
        t_ds = self.calc(
            data=data.ds,
            target_cols=target_cols,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
        )
        result = data.copy(data=t_ds)
        result.additional_fields = result.additional_fields.filter(t_ds.index, axis=0)
        return result

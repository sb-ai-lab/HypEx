from typing import Any, Optional, Union, Iterable

from hypex.dataset.dataset import Dataset
from hypex.dataset.dataset import ExperimentData
from hypex.dataset.roles import (
    ABCRole,
    InfoRole,
    FeatureRole,
    PreTargetRole,
    TargetRole,
)
from hypex.transformers.abstract import Transformer


class CVFilter(Transformer):
    def __init__(
        self,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
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
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_roles: Union[ABCRole, Iterable[ABCRole]] = FeatureRole(),
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        type_filter: bool = True,
    ) -> Dataset:
        target_roles = super()._list_unification(target_roles)
        if type_filter:
            addressable_columns = data.search_columns(
                roles=target_roles, search_types=[float, int, bool]
            )
        else:
            addressable_columns = data.search_columns(target_roles)
        for column in addressable_columns:
            cv = data[column].coefficient_of_variation()
            drop = False
            if upper_bound and cv > upper_bound:
                drop = True
            if lower_bound and cv < lower_bound:
                drop = True
            if drop:
                data.roles[column] = InfoRole()
        return data

    @classmethod
    def calc(
        cls,
        data: Dataset,
        target_roles: Union[ABCRole, Iterable[ABCRole]] = FeatureRole(),
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        type_filter: bool = True,
    ) -> Dataset:
        return cls._inner_function(
            data=data,
            target_roles=target_roles,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            type_filter=type_filter,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(
            data=self.calc(
                data=data.ds, lower_bound=self.lower_bound, upper_bound=self.upper_bound
            )
        )
        return result


class ConstFilter(Transformer):
    def __init__(
        self,
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
        self.threshold = threshold

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_roles: Union[ABCRole, Iterable[ABCRole]] = FeatureRole(),
        threshold: float = 0.95,
    ) -> Dataset:
        target_roles = super()._list_unification(target_roles)
        for column in data.search_columns(target_roles):
            value_counts = data[column].value_counts(normalize=True, sort=True)
            if value_counts.get_values(0, "proportion") > threshold:
                data.roles[column] = InfoRole()
        return data

    @classmethod
    def calc(
        cls,
        data: Dataset,
        target_roles: Union[ABCRole, Iterable[ABCRole]] = FeatureRole(),
        threshold: float = 0.95,
    ) -> Dataset:
        return cls._inner_function(
            data=data,
            target_roles=target_roles,
            threshold=threshold,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(data=self.calc(data=data.ds, threshold=self.threshold))
        return result


class NanFilter(Transformer):
    def __init__(
        self,
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
        self.threshold = threshold

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_roles: Union[ABCRole, Iterable[ABCRole]] = FeatureRole(),
        threshold: float = 0.8,
    ) -> Dataset:
        target_roles = super()._list_unification(target_roles)
        for column in data.search_columns(target_roles):
            nan_share = data[column].isna().sum() / len(data)
            if nan_share > threshold:
                data.roles[column] = InfoRole()
        return data

    @classmethod
    def calc(
        cls,
        data: Dataset,
        target_roles: Union[ABCRole, Iterable[ABCRole]] = FeatureRole(),
        threshold: float = 0.8,
    ) -> Dataset:
        return cls._inner_function(
            data=data, target_roles=target_roles, threshold=threshold
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(data=self.calc(data=data.ds, threshold=self.threshold))
        return result


class CorrFilter(Transformer):
    def __init__(
        self,
        threshold: float = 0.8,
        method: str = "pearson",
        numeric_only: bool = True,
        key: Any = "",
    ):
        super().__init__(key=key)
        self.threshold = threshold
        self.method = method
        self.numeric_only = numeric_only

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_roles: Union[ABCRole, Iterable[ABCRole]] = FeatureRole(),
        threshold: float = 0.8,
        method: str = "pearson",
        numeric_only: bool = True,
        corr_space_roles=None,
        drop_policy: str = "cv",
    ) -> Dataset:
        if corr_space_roles is None:
            corr_space_roles = [
                FeatureRole(),
                TargetRole(),
            ]
        target_roles = super()._list_unification(target_roles)
        corr_space_roles = super()._list_unification(corr_space_roles)
        corr_matrix = data[data.search_columns(corr_space_roles)].corr(
            method=method, numeric_only=numeric_only
        )
        pre_target_column = None
        if drop_policy == "corr":
            pre_target_columns = data.search_columns([PreTargetRole()])
            if (PreTargetRole() not in corr_space_roles) | len(pre_target_columns) != 1:
                raise ValueError(
                    "Correlation-based filtering cannot be applied if there are more than one PreTarget columns"
                )
            else:
                pre_target_column = pre_target_columns[0]
        target_roles_cols = corr_matrix.search_columns(target_roles)
        for target in target_roles_cols:
            for column in corr_matrix.columns:
                if (target != column) and (
                    abs(corr_matrix.get_values(target, column)) > threshold
                ):
                    drop = target
                    if data.roles[column] in target_roles_cols:
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

    @classmethod
    def calc(
        cls,
        data: Dataset,
        target_roles: Union[ABCRole, Iterable[ABCRole]] = FeatureRole(),
        threshold: float = 0.8,
        method: str = "pearson",
        numeric_only: bool = True,
        corr_space_roles=None,
        drop_policy: str = "cv",
    ) -> Dataset:
        return cls._inner_function(
            data=data,
            target_roles=target_roles,
            threshold=threshold,
            method=method,
            numeric_only=numeric_only,
            corr_space_roles=corr_space_roles,
            drop_policy=drop_policy,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(
            data=self.calc(
                data=data.ds,
                threshold=self.threshold,
                method=self.method,
                numeric_only=self.numeric_only,
            )
        )
        return result


class OutliersFilter(Transformer):
    def __init__(
        self,
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
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_roles: Union[ABCRole, Iterable[ABCRole]] = FeatureRole(),
        lower_percentile: float = 0,
        upper_percentile: float = 1,
    ) -> Dataset:
        addressable_roles = data.search_columns(
            roles=super()._list_unification(target_roles),
            search_types=[float, int, bool],
        )
        mask = data[addressable_roles].apply(
            func=lambda x: (x < x.quantile(lower_percentile))
            | (x > x.quantile(upper_percentile)),
            role={column: InfoRole() for column in addressable_roles},
            axis=0,
        )
        mask = mask.apply(func=lambda x: x.any(), role={"filter": InfoRole()}, axis=1)
        drop_indexes = mask[~mask].index.get_values(columns="filter")
        data = data.drop(drop_indexes, axis=0)
        return data

    @classmethod
    def calc(
        cls,
        data: Dataset,
        target_roles: Union[ABCRole, Iterable[ABCRole]] = FeatureRole(),
        lower_percentile: float = 0,
        upper_percentile: float = 1,
    ) -> Dataset:
        return cls._inner_function(
            data=data,
            target_roles=target_roles,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        t_ds = self.calc(
            data=data.ds,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
        )
        result = data.copy(data=t_ds)
        result.additional_fields = result.additional_fields.filter(t_ds.index, axis=0)
        return result

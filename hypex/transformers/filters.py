from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..dataset.dataset import Dataset
from ..dataset.experiment_data import ExperimentData
from ..dataset.roles import FeatureRole, InfoRole, PreTargetRole, TargetRole
from ..utils.adapter import Adapter
from .abstract import Transformer


class CVFilter(Transformer):
    """Filter columns based on coefficient of variation.

    This transformer identifies columns with unusual variability by computing
    the coefficient of variation (CV = std / mean) and filters out columns
    that fall outside specified bounds.

    Args:
        target_roles: Role(s) identifying columns to filter. Defaults to FeatureRole().
        lower_bound: Minimum acceptable CV. Columns below this are considered constant. Defaults to None.
        upper_bound: Maximum acceptable CV. Columns above this are considered unstable. Defaults to None.
        key: Optional executor identifier.

    Example:
        >>> from hypex.transformers import CVFilter
        >>> from hypex.dataset import FeatureRole
        >>> filter = CVFilter(lower_bound=0.1, upper_bound=5.0)
        >>> result = filter.execute(experiment_data)
    """

    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        key: Any = "",
    ):
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
    """Filter constant or near-constant columns.

    This transformer identifies columns where a single value occurs
    more frequently than a specified threshold and filters them out.

    Args:
        target_roles: Role(s) identifying columns to filter. Defaults to FeatureRole().
        threshold: Maximum acceptable frequency for a constant value. Defaults to 0.95.
        key: Optional executor identifier.

    Example:
        >>> from hypex.transformers import ConstFilter
        >>> from hypex.dataset import FeatureRole
        >>> filter = ConstFilter(threshold=0.9)
        >>> result = filter.execute(experiment_data)
    """

    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        threshold: float = 0.95,
        key: Any = "",
    ):
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
            value_counts = data[column].value_counts(normalize=True)
            if value_counts[column] > threshold:
                data.roles[column] = InfoRole()
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
            )
        )
        return result


class NanFilter(Transformer):
    """Filter columns with excessive NaN values.

    This transformer identifies columns where NaN values occur more
    frequently than a specified threshold and filters them out.

    Args:
        target_roles: Role(s) identifying columns to filter. Defaults to FeatureRole().
        threshold: Maximum acceptable frequency of NaN values in a column. Defaults to 0.8.
        key: Optional executor identifier.

    Example:
        >>> from hypex.transformers import NanFilter
        >>> from hypex.dataset import FeatureRole
        >>> filter = NanFilter(threshold=0.7)
        >>> result = filter.execute(experiment_data)
    """

    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        threshold: float = 0.8,
        key: Any = "",
    ):
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
            nan_ratio = data[column].isna().mean()
            if nan_ratio > threshold:
                data.roles[column] = InfoRole()
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
            )
        )
        return result


class CorrFilter(Transformer):
    """Filter columns based on correlation with target columns.

    This transformer identifies and filters columns that are highly
    correlated with target columns. It uses the correlation matrix
    to determine which columns to keep based on the specified method
    (pearson, kendall, or spearman).

    Args:
        target_roles: Role(s) identifying target columns. Defaults to TargetRole().
        corr_space_roles: Role(s) identifying columns to consider for correlation. Defaults to [FeatureRole(), TargetRole()].
        threshold: Correlation threshold for filtering. Defaults to 0.8.
        method: Correlation method - 'pearson', 'kendall', or 'spearman'. Defaults to 'pearson'.
        numeric_only: Whether to compute correlations only for numeric columns. Defaults to True.
        key: Optional executor identifier.

    Example:
        >>> from hypex.transformers import CorrFilter
        >>> from hypex.dataset import TargetRole, FeatureRole
        >>> filter = CorrFilter(threshold=0.7, method='pearson')
        >>> result = filter.execute(experiment_data)
    """

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
        self.target_roles = target_roles or [FeatureRole(), TargetRole()]
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
    ) -> Dataset:
        corr_matrix = data.ds.corr(method=method, numeric_only=numeric_only)
        corr_target_cols = Adapter.to_list(target_cols)
        corr_space_cols = Adapter.to_list(corr_space_cols)
        for target in corr_target_cols:
            for pre_target_column in corr_space_cols:
                if target not in corr_target_cols or pre_target_column not in corr_space_cols:
                    continue
                if target == pre_target_column:
                    continue
                drop_policy = "corr" if method == "pearson" else "cv"
                if data.roles[target] in corr_target_cols:
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
    """Filter rows with outlier values based on percentiles.

    This transformer identifies and removes rows where column values
    fall outside specified percentile ranges, effectively filtering
    out extreme outlier values.

    Args:
        target_roles: Role(s) identifying columns to filter. Defaults to FeatureRole().
        lower_percentile: Lower percentile threshold. Defaults to 0.
        upper_percentile: Upper percentile threshold. Defaults to 1.
        key: Optional executor identifier.

    Example:
        >>> from hypex.transformers import OutliersFilter
        >>> from hypex.dataset import FeatureRole
        >>> filter = OutliersFilter(lower_percentile=0.01, upper_percentile=0.99)
        >>> result = filter.execute(experiment_data)
    """

    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        lower_percentile: float = 0,
        upper_percentile: float = 1,
        key: Any = "",
    ):
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
            func=lambda x: (
                (x < x.quantile(lower_percentile)) | (x > x.quantile(upper_percentile))
            ),
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

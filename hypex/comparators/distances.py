from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from ..dataset import (
    ABCRole,
    AdditionalFeatureRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    GroupedDataset,
    GroupingRole,
    TargetRole,
)
from ..executor import Calculator
from ..extensions.scipy_linalg import (
    CholeskyExtension,
    InverseExtension,
    UniteCovExtension,
)
from ..utils import ExperimentDataEnum, NotSuitableFieldError
from ..utils.adapter import Adapter


class MahalanobisDistance(Calculator):
    """
    Calculator for computing the Mahalanobis distance between groups.

    This class is typically used in matching algorithms to find similar observations 
    across treatment and control groups based on multiple features. It accounts for 
    feature correlations by applying a linear transformation (via Cholesky decomposition 
    of the pooled covariance matrix) and optionally applies user-defined feature weights.

    Inherits from:
        Calculator: The base class for stateless calculation executors.
    """
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        key: Any = "",
        weights: dict[str, float] | None = None,
    ):
        """
        Initialize the Mahalanobis distance calculator.

        Args:
            grouping_role (ABCRole | None, optional): The role defining the grouping 
                column (e.g., treatment assignment). Defaults to `GroupingRole()`.
            key (Any, optional): Optional identifier for the calculator instance. 
                Defaults to "".
            weights (dict[str, float] | None, optional): Optional dictionary mapping 
                feature names to their relative importance weights. If None, all 
                features are weighted equally. Defaults to None.
        """
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()
        self.weights = weights

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: list[str] | None = None,
        **kwargs,
    ) -> Dataset:
        """
        Execute the inner distance calculation logic on grouped data.

        Iterates through the grouped data, treating the first group as the baseline 
        (control) and subsequent group as compared (test) group, calculating the 
        Mahalanobis transformation matrix to both.

        Args:
            grouping_data: Grouped dataset containing baseline and compared data slices.
            target_fields (list[str] | None, optional): Optional list of target field 
                names to compute the distance on. Defaults to None.
            **kwargs: Additional keyword arguments passed to `_inner_function`.

        Returns:
            Dataset:  The corr matrix Dataset to transform into Mahalanobis metric.
        """
        if len(grouping_data) > 1:
            return cls._inner_function(
                data=(
                    grouping_data[0][1][target_fields]
                    if target_fields
                    else grouping_data[0][1]
                ),
                test_data=(
                    grouping_data[1][1][target_fields]
                    if target_fields
                    else grouping_data[1][1]
                ),
                **kwargs,
            )
        else:
            return cls._inner_function(
                data=(
                    grouping_data[0][1][target_fields]
                    if target_fields
                    else grouping_data[0][1]
                ),
                test_data=None,
                **kwargs,
            )

    def _set_value(
            self, data: ExperimentData, value: Dataset | None = None, key: Any = None
    ) -> ExperimentData:
        """
        Store the calculated distance results into the ExperimentData object.

        Args:
            data (ExperimentData): The experiment data object to update.
            value (Dataset | None, optional): The corr matrix Dataset to transform
                into Mahalanobis metric. Defaults to None.
            key (Any, optional): Optional key for the stored value. Defaults to None.

        Returns:
        """
        data.set_value(
            ExperimentDataEnum.variables,
            self.id,
            value,
            self.key
        )
        return data

    def _get_fields(self, data: ExperimentData):
        """
        Retrieve the grouping and target fields from the experiment data.

        Args:
            data (ExperimentData): The input experiment data.

        Returns:
            tuple: A tuple containing `(group_field, target_fields)`.
        """
        group_field = data.field_search(self.grouping_role)
        target_fields = data.field_search(
            [FeatureRole(), AdditionalFeatureRole()], search_types=self.search_types
        )
        return group_field, target_fields

    @property
    def search_types(self) -> list[type] | None:
        """
        Return the allowed data types for this statistical operation.

        Returns:
            list[type]: List of numeric types supported (`int`, `float`).
        """
        return [int, float]

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        weights: dict[str, float] | None = None,
        **kwargs,
    ) -> Dataset:
        """
        Compute the Mahalanobis transformation for the given datasets.

        Calculates the pooled covariance matrix, performs Cholesky decomposition, 
        and applies the inverse transformation to project the data into a space 
        where the covariance is the identity matrix. Optional feature weights are 
        applied via a diagonal matrix before the final projection.

        Args:
            data (Dataset): The baseline (control) dataset.
            test_data (Dataset | None, optional): The compared (test) dataset. 
                Defaults to None.
            weights (dict[str, float] | None, optional): Feature weights dictionary. 
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the transformed "control" dataset, and 
            optionally the "test" dataset if `test_data` was provided.
        """
        test_data = cls._check_test_data(test_data)
        cov = UniteCovExtension().calc(data, test_data)

        cholesky = CholeskyExtension().calc(cov)
        mahalanobis_transform = InverseExtension().calc(cholesky)
        # mah_cols = mahalanobis_transform.columns
        if weights is not None:
            features = data.columns
            w_list = np.array(
                [weights[col] if col in weights.keys() else 1 for col in features]
            )
            w_matrix = np.sqrt(np.diag(w_list / w_list.sum()))
            mahalanobis_transform = mahalanobis_transform.dot(w_matrix)

        mahalanobis_transform: Dataset = mahalanobis_transform.transpose()
        # mahalanobis_transform = mahalanobis_transform.rename({col: new_col for col, new_col in zip(mahalanobis_transform.columns, mah_cols)})
        return mahalanobis_transform

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Sequence[str] | str | None = None,
        grouping_data: GroupedDataset | None = None,
        target_fields: str | list[str] | None = None,
        weights: dict[str, float] | None = None,
        **kwargs,
    ) -> Dataset:
        """
        Stateless entry point to calculate Mahalanobis distance.

        Allows the comparator to be run outside the experiment pipeline. Pass either 
        pre-grouped `grouping_data` or the raw `data` and `group_field` to have the 
        data grouped here.

        Args:
            data (Dataset): The input dataset.
            group_field (Sequence[str] | str | None, optional): Column name(s) to 
                group by. Defaults to None.
            grouping_data (GroupedDataset | None, optional): Pre-grouped data. 
                Defaults to None.
            target_fields (str | list[str] | None, optional): Target column(s) to 
                compute distance on. Defaults to None.
            weights (dict[str, float] | None, optional): Feature weights. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary of transformed datasets grouped by the specified field.

        Raises:
            NotSuitableFieldError: If the grouping field is not suitable (e.g., only one group).
        """
        group_field = Adapter.to_list(group_field)

        if grouping_data is None:
            grouping_data = list(data.groupby(group_field))
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise NotSuitableFieldError(group_field, "Grouping")

        return cls._execute_inner_function(
            grouping_data,
            # tmp_roles=tmp_roles,
            target_fields=target_fields,
            old_data=data,
            weights=weights,
            **kwargs,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute the Mahalanobis distance calculation on the given experiment data.

        Retrieves the appropriate fields, handles temporary roles if necessary, and 
        delegates the calculation to `calc`. The results are then stored back into 
        the `ExperimentData` object.

        Args:
            data (ExperimentData): The ExperimentData to execute the calculator on.

        Returns:
            ExperimentData: The ExperimentData with the distance calculation results 
            stored in the `groups` space.
        """
        group_field, target_fields = self._get_fields(data=data)
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.initial_ds.tmp_roles
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
            return data
        if group_field[0] in data.groups:  # TODO: to recheck if this is a correct check
            grouping_data = list(data.groups[group_field[0]].items())
        else:
            grouping_data = None
        t_data = deepcopy(data.initial_ds)
        for field in target_fields:
            if field not in t_data.columns:
                t_data = t_data.add_column(
                    data.additional_fields[field],
                    role={field: TargetRole()},
                )
        mahalanobis_transform = self.calc(
            data=t_data,
            group_field=group_field,
            target_fields=target_fields,
            grouping_data=grouping_data,
            weights=self.weights or None,
        )
        return self._set_value(data, mahalanobis_transform)

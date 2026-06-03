from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np

from ..dataset import (
    ABCRole,
    AdditionalMatchingRole,
    AdditionalTargetRole,
    Dataset,
    SmallDataset,
    ExperimentData,
    FeatureRole,
    InfoRole,
    TargetRole,
)
from ..extensions.scipy_stats import NormCDF
from ..extensions.scipy_linalg import LstsqExtension
from ..utils.enums import ExperimentDataEnum
from ..utils.errors import NoneArgumentError
from ..utils.registry import backend_factory
from .abstract import GroupOperator


class SMD(GroupOperator):
    def execute(self, data: ExperimentData) -> ExperimentData:
        pass

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Any:
        test_data = cls._check_test_data(test_data=test_data)
        return (data.mean() + test_data.mean()) / data.std()


class MatchingMetrics(GroupOperator):
    """
    Calculator for estimating treatment effects (ATT, ATC, ATE) and their 
    statistical significance after matching.

    This class computes the Individual Treatment Effect on the Treated (ITT) 
    and Control (ITC), applies optional bias correction, and calculates the 
    final Average Treatment Effects along with their standard errors, p-values, 
    and confidence intervals. It also handles the calculation of scaled counts 
    (weights) to account for multiple matches or specific matching strategies.

    Inherits from:
        GroupOperator: The base class for group-based operators in the HypEx library.
    """
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
        metric: Literal["auto", "atc", "att", "ate"] | None = None,
        n_neighbors: int = 1,
        key: Any = "",
    ):
        """
        Initialize the matching metrics calculator.

        Args:
            grouping_role (ABCRole | None, optional): The role defining the grouping 
                column. Defaults to None.
            target_roles (ABCRole | list[ABCRole] | None, optional): The role(s) 
                defining the target column(s). Defaults to None.
            metric (Literal["auto", "atc", "att", "ate"] | None, optional): The type 
                of treatment effect to estimate. "atc" = average treatment effect on 
                controls, "att" = average treatment effect on treated, "ate" = average 
                treatment effect, "auto" = calculates all. Defaults to "auto".
            n_neighbors (int, optional): The number of neighbors used in the matching 
                process, used for scaling counts. Defaults to 1.
            key (Any, optional): Optional identifier for the operator instance. 
                Defaults to "".
        """
        self.metric = metric or "auto"
        self.n_neighbors = n_neighbors
        self.__scaled_counts = {}
        target_roles = target_roles or TargetRole()
        super().__init__(
            grouping_role=grouping_role,
            target_roles=(
                target_roles if isinstance(target_roles, list) else [target_roles]
            ),
            key=key,
        )

    def _calc_scaled_counts(self, matches: Dataset, indexes: Dataset, group: str):
        """
        Calculate the scaled counts (weights) for matched observations.

        This method computes the frequency of each index in the matched dataset 
        and scales it by the number of neighbors. This is crucial for correctly 
        weighting observations when a single control unit is matched to multiple 
        treatment units (or vice versa).

        Args:
            matches (Dataset): The dataset containing the matched observations.
            indexes (Dataset): The dataset containing the original indexes of the matches.
            group (str): The name of the group ("control" or "test") for which 
                the scaled counts are being calculated.
        """
        matches_indexes = indexes.reset_index().select('index')
        matches_counts = (
            matches
            .apply(list, role={'indexes': InfoRole()}, axis=1)
            .explode('indexes')
            .value_counts()
        )

        matches_counts = (
            matches_indexes
            .merge(left_on='index', right_on='indexes', right=matches_counts, how='left')
            .drop(columns=['indexes'])
            .fillna(0)
            .set_index('index')
        )
        matches_counts.index.name = None
        self.__scaled_counts[group] = matches_counts["count"] / self.n_neighbors

    @staticmethod
    def _calc_vars(value: Dataset) -> float:
        """
        Calculate the variance of a dataset, handling potential NaN values.

        Args:
            value (Dataset): The dataset for which to calculate the variance.

        Returns:
            float: The variance of the dataset. Returns 0.0 if the dataset 
            contains any NaN values.
        """
        var = 0 if int(value[value.columns[0]].na_counts()) > 0 else float(value.var())
        return var

    @staticmethod
    def _calc_se(
        n_c: int, n_t: int, var_c: float, var_t: float, scaled_counts: dict[str, Dataset], group=None
    ):
        """
        Calculate the standard error for the treatment effect estimates.

        Args:
            n_c (int): The number of observations in the control group.
            n_t (int): The number of observations in the treatment group.
            var_c (float): The variance of the control group.
            var_t (float): The variance of the treatment group.
            scaled_counts (dict[str, Dataset]): A dictionary containing the scaled 
                counts (weights) for the groups.
            group (str | None, optional): The specific group to calculate the SE for. 
                If None, calculates the pooled SE for ATE. Defaults to None.

        Returns:
            float: The calculated standard error.
        """
        if group is not None:
            groups = list(scaled_counts.keys())
            groups.remove(group)
            group_other = groups[0]
            weights_c = scaled_counts[group_other] * 0 + 1
            weights_t = scaled_counts[group] * n_t / n_c
        else:
            n = n_c + n_t
            weights_c = (n_c / n) * (scaled_counts["test"] + 1)
            weights_t = (n_t / n) * (scaled_counts["control"] + 1)

        return np.sqrt(
            (weights_t**2 * var_t).sum() / n_t**2
            + (weights_c**2 * var_c).sum() / n_c**2
        )

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        target_fields: list[str] | None = None,
        **kwargs,
    ) -> Any:
        """
        Core calculation logic for treatment effects.

        Computes the Individual Treatment Effect on the Treated (ITT) and Control (ITC), 
        applies optional bias correction, and calculates the final metrics (ATT, ATC, ATE) 
        along with their standard errors, p-values, and 95% confidence intervals.

        Args:
            data (Dataset): The baseline (control) dataset.
            test_data (Dataset | None, optional): The compared (treatment) dataset. 
                Defaults to None.
            target_fields (list[str] | None, optional): The names of the target fields 
                to calculate effects on. Defaults to None.
            **kwargs: Additional keyword arguments, including:
                - `metric` (str): The type of effect to estimate ("att", "atc", "ate").
                - `scaled_counts` (dict): Pre-calculated scaled counts for weighting.
                - `bias` (dict | None): Optional bias correction values.

        Returns:
            Any: A dictionary containing the calculated metrics. Keys depend on the 
            `metric` argument and include effect size, standard error, p-value, and 
            lower/upper confidence interval bounds.
        """
        if target_fields is None or test_data is None:
            raise NoneArgumentError(
                ["target_fields", "test_data"], "att, atc, ate estimation"
            )
        metric = kwargs.get("metric", "ate")
        scaled_counts = kwargs.get("scaled_counts")
        itt = test_data[target_fields[0]] - test_data[target_fields[1]]
        itc = data[target_fields[1]] - data[target_fields[0]]

        bias = kwargs.get("bias", {})
        if bias and len(bias) > 0:
            if metric in ["atc", "ate"]:
                control_bias: Dataset = bias["control"]
                control_bias = control_bias.add_column(itc.index, {'index': InfoRole()}).set_index('index')
                control_bias.index.name = None
                # itc -= Dataset.from_dict(
                #     {"test": bias["control"]}, roles={}, index=itc.index
                # )
                itc -= control_bias
            if metric in ["att", "ate"]:
                test_bias: Dataset = bias["test"]
                test_bias = test_bias.add_column(itt.index, {'index': InfoRole()}).set_index('index')
                test_bias.index.name = None
                # itt += Dataset.from_dict(
                #     {"control": bias["test"]}, roles={}, index=itt.index
                # )
                itt += test_bias

        itc_len = len(itc)
        itt_len = len(itt)
        var_t = cls._calc_vars(itc)
        var_c = cls._calc_vars(itt)
        itt_se = cls._calc_se(itc_len, itt_len, var_c, var_t, scaled_counts, "control")
        itc_se = cls._calc_se(itt_len, itc_len, var_t, var_c, scaled_counts, "test")
        itt = itt.mean()
        itc = itc.mean()
        p_val_itt = (
            NormCDF()
            .calc(
                SmallDataset.from_dict(
                    {"value": [itt / itt_se]}, roles={"value": InfoRole()}
                )
            )
            .get_values()[0][0]
        )
        p_val_itc = (
            NormCDF()
            .calc(
                SmallDataset.from_dict(
                    {"value": [itc / itc_se]}, roles={"value": InfoRole()}
                )
            )
            .get_values()[0][0]
        )
        if metric == "atc":
            return {
                "ATC": [
                    itc,
                    itc_se,
                    p_val_itc,
                    itc - 1.96 * itc_se,
                    itc + 1.96 * itc_se,
                ]
            }
        if metric == "att":
            return {
                "ATT": [
                    itt,
                    itt_se,
                    p_val_itt,
                    itt - 1.96 * itt_se,
                    itt + 1.96 * itt_se,
                ]
            }
        len_control, len_test = len(data), len(test_data)
        ate = (itt * len_test + itc * len_control) / (len_test + len_control)
        ate_se = cls._calc_se(itc_len, itt_len, var_c, var_t, scaled_counts)
        p_val_ate = (
            NormCDF()
            .calc(
                SmallDataset.from_dict(
                    {"value": [ate / ate_se]}, roles={"value": InfoRole()}
                )
            )
            .get_values()[0][0]
        )
        return {
            "ATT": [itt, itt_se, p_val_itt, itt - 1.96 * itt_se, itt + 1.96 * itt_se],
            "ATC": [itc, itc_se, p_val_itc, itc - 1.96 * itc_se, itc + 1.96 * itc_se],
            "ATE": [ate, ate_se, p_val_ate, ate - 1.96 * ate_se, ate + 1.96 * ate_se],
        }

    @classmethod
    def _execute_inner_function(
        cls, grouping_data, target_fields: list[str] | None = None, **kwargs
    ) -> dict:
        """
        Wrapper to execute the inner function over the grouped data.

        Args:
            grouping_data: Grouped dataset containing the control and treatment slices.
            target_fields (list[str] | None, optional): The names of the target fields. 
                Defaults to None.
            **kwargs: Additional keyword arguments passed to `_inner_function`.

        Returns:
            dict: The result of the `_inner_function` calculation.
        """
        metric = kwargs.get("metric", "ate")
        if target_fields is None or len(target_fields) != 2:
            raise ValueError(
                f"This operator works with 2 targets, but got {len(target_fields) if target_fields else None}"
            )
        return cls._inner_function(
            data=grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            metric=metric,
            bias=kwargs.get("bias_estimation", None),
            scaled_counts=kwargs.get("scaled_counts"),
        )

    def _prepare_new_target(
        self,
        data: ExperimentData,
        t_data: Dataset,
        group_field: str,
    ) -> Dataset:
        """
        Prepare a new target variable by merging matched data and calculating scaled counts.

        This method is used when a secondary target field needs to be constructed from 
        the matched pairs. It aligns the matched data, calculates the scaled counts for 
        both groups, and returns the aggregated matched dataset.

        Args:
            data (ExperimentData): The original experiment data.
            t_data (Dataset): The dataset to be updated with the new target.
            group_field (str): The name of the grouping field.

        Returns:
            Dataset: The prepared matched dataset with aggregated values.
        """
        new_target = data.ds.search_columns(TargetRole())[0]
        indexes, matched_data = Bias.prepare_data(data, t_data)
        matched_data = matched_data[new_target + "_matched"]
        grouped_column = data.ds[group_field]
        (_, control_indexes), (_, test_indexes), *_ = (
            grouped_column
            .merge(right=indexes, right_index=True, left_index=True)
            .groupby(group_field)
        )

        control_indexes, test_indexes = control_indexes[indexes.columns], test_indexes[indexes.columns]
        self._calc_scaled_counts(control_indexes, test_indexes, "test")
        self._calc_scaled_counts(test_indexes, control_indexes, "control")

        return matched_data

    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Main execution method for calculating matching metrics.

        Orchestrates the calculation process: retrieves fields, prepares targets 
        if necessary (e.g., when a second target is missing), calculates the metrics 
        using the `calc` method, and stores the results in the `ExperimentData` object.

        Args:
            data (ExperimentData): The experiment data containing the matched dataset.

        Returns:
            ExperimentData: The updated ExperimentData object with the calculated 
            matching metrics stored in the `variables` space.
        """
        group_field, target_fields = self._get_fields(data=data)
        bias = (
            data.variables[data.get_one_id(Bias, ExperimentDataEnum.variables)]
            if len(
                data.get_ids(Bias, ExperimentDataEnum.variables)["Bias"]["variables"]
            )
            > 0
            else None
        )
        t_data = deepcopy(data.ds)
        if len(target_fields) != 2:
            matched_data = self._prepare_new_target(data, t_data, group_field)
            target_fields += [matched_data.search_columns(TargetRole())[0]]
            data.set_value(
                ExperimentDataEnum.additional_fields,
                self.id,
                matched_data,
                role=AdditionalTargetRole(),
            )
            t_data = t_data.add_column(
                # matched_data.reindex(t_data.index), 
                matched_data,
                role={target_fields[1]: TargetRole()},
            )
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
            return data

        compare_result = self.calc(
            data=t_data,
            group_field=group_field,
            target_fields=target_fields,
            metric=self.metric,
            bias_estimation=bias,
            scaled_counts=self.__scaled_counts,
        )
        return self._set_value(data, compare_result)


class Bias(GroupOperator):
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: list[ABCRole] | None = None,
        key: Any = "",
    ):
        super().__init__(
            grouping_role=grouping_role, target_roles=target_roles, key=key
        )

    @staticmethod
    def calc_bias(
        X: Dataset, X_matched: Dataset, coefficients: np.ndarray[float]
    ) -> list[float]:
        return (X - X_matched).dot(coefficients)

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        target_fields: list[str] | None = None,
        features_fields: list[str] | None = None,
        **kwargs,
    ) -> dict:
        if target_fields is None or features_fields is None or test_data is None:
            raise NoneArgumentError(
                ["target_fields", "features_fields", "test_data"], "bias_estimation"
            )
        if data[target_fields[1]].na_counts() > 0:
            coef_cls = backend_factory.resolve_backend(LstsqExtension, test_data)
            coefficients = coef_cls().calc(test_data[[target_fields[1]] + features_fields[len(features_fields) // 2 :]])
            return {
                "test": cls.calc_bias(
                    test_data[features_fields[: len(features_fields) // 2]],
                    test_data[features_fields[len(features_fields) // 2 :]],
                    coefficients,
                )
            }
        
        if test_data[target_fields[1]].na_counts() > 0:
            coef_cls = backend_factory.resolve_backend(LstsqExtension, data)
            coefficients = coef_cls().calc(data[[target_fields[1]] + features_fields[len(features_fields) // 2 :]])
            return {
                "control": cls.calc_bias(
                    data[features_fields[: len(features_fields) // 2]],
                    data[features_fields[len(features_fields) // 2 :]],
                    coefficients,
                )
            }
        coef_cls = backend_factory.resolve_backend(LstsqExtension, test_data)
        test_coefficients = coef_cls().calc(test_data[[target_fields[1]] + features_fields[len(features_fields) // 2 :]])

        coef_cls = backend_factory.resolve_backend(LstsqExtension, data)
        control_coefficients = coef_cls().calc(data[[target_fields[1]] + features_fields[len(features_fields) // 2 :]])
        return {
            "test": cls.calc_bias(
                test_data[features_fields[: len(features_fields) // 2]],
                test_data[features_fields[len(features_fields) // 2 :]],
                test_coefficients,
            ),
            "control": cls.calc_bias(
                data[features_fields[: len(features_fields) // 2]],
                data[features_fields[len(features_fields) // 2 :]],
                control_coefficients,
            ),
        }

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: list[str] | None = None,
        features_fields: list[str] | None = None,
        **kwargs,
    ) -> dict:
        return cls._inner_function(
            grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            features_fields=features_fields,
            **kwargs,
        )

    @staticmethod
    def prepare_data(data: ExperimentData, t_data: Dataset) -> Dataset:
        indexes = data.field_search(AdditionalMatchingRole())
        if len(indexes) == 0:
            raise ValueError("No indexes were found")
        indexes = data.additional_fields[indexes]
        # additional fields are already allignet according to index
         
        # indexes.index = t_data.index
        # indexes = indexes.add_column(t_data.index, {'index': InfoRole()}).set_index('index')
        # indexes.index.name = None
        # filtered_field = indexes
        # filtered_field = indexes.reset_index(drop=True)

        numeric_cols = t_data.search_columns(
            [FeatureRole(), TargetRole()], search_types=[int, float]
        )
        
        matched_data = (
            indexes
            # filtered_field
            .apply(list, axis=1, role={'_index' : InfoRole()})
            .explode('_index')
            .merge(t_data.select(numeric_cols), 
                   left_on='_index' ,right_index=True)
            .drop(columns=['_index'])
            .reset_index()
        )
        matched_data = (
            matched_data
            .add_column(matched_data['index'], {'_index': InfoRole()})
            .groupby(by='index')
            .agg('mean')
            .rename({col: col + "_matched" for col in numeric_cols})
            .set_index('_index') #grouping field will not disappear after `groupby`
        )
        # print(matched_data)
        matched_data.index.name = None

        return indexes, matched_data

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data)
        t_data = deepcopy(data.ds)
        if len(target_fields) < 2:
            _, matched_data = self.prepare_data(data, t_data)
            target_fields += [matched_data.search_columns(TargetRole())[0]]
            t_data = t_data.add_column(matched_data)
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
            return data  

        compare_result = self.calc(
            data=t_data,
            group_field=group_field,
            target_fields=target_fields,
            features_fields=t_data.search_columns(
                FeatureRole(), search_types=[int, float]
            ),
        )
        return self._set_value(data, compare_result)

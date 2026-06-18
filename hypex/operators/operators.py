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
    AdditionalStatisticRole
)
from ..extensions.scipy_stats import NormCDF
from ..extensions.scipy_linalg import LstsqExtension
from ..utils import ID_SPLIT_SYMBOL
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
                control_bias = bias["control"]
                itc -= control_bias
            if metric in ["att", "ate"]:
                test_bias = bias["test"]
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

    # TODO: fix bias, as now it is in `additional_fields` not in `variables`
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
        bias = data.field_search(AdditionalStatisticRole())
        if len(bias) > 0:
            bias_groups = list(
                data.ds[group_field]
                .merge(right=data.additional_fields[bias], left_index=True, right_index=True)
                .groupby(group_field)
            )
            bias = {
                "control": bias_groups[0][1][bias],
                "test": bias_groups[1][1][bias]
            }

        else:
            bias = None
            
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
    """
    Calculator for estimating selection bias after matching.

    This operator quantifies the residual bias between treatment and control
    groups that remains after the matching procedure. It uses a linear
    regression model (via ``LstsqExtension``) trained on the matched sample
    to predict the counterfactual outcome, then computes the difference
    between the observed and predicted values.

    The bias is defined as:
        - For treatment group:  bias_t = E[Y(0) | T=1] - E[Y(0) | T=0]
        - For control group:    bias_c = E[Y(1) | T=1] - E[Y(1) | T=0]

    where Y(0) and Y(1) are potential outcomes under control and treatment,
    and T is the treatment assignment indicator.

    The computation proceeds as follows:
        1. For each group, fit a linear model: target ~ features
           using the matched observations as the training signal.
        2. Compute the matched (counterfactual) features by averaging
           over the matched pairs for each observation.
        3. Estimate bias as: (X - X_matched) · coefficients,
           where X are the original features and X_matched are the
           pair-averaged features.

    The operator automatically handles cases where one group lacks
    matched observations (e.g., one-sided matching) by computing bias
    only for the group with available matched data.

    Inherits from:
        GroupOperator: The base class for group-based operators in the HypEx library.

    Examples:
        ```python
            # Typically used internally by the Matching pipeline
            from hypex.operators import Bias
            from hypex.dataset import TreatmentRole, TargetRole

            bias_calc = Bias(
                grouping_role=TreatmentRole(),
                target_roles=[TargetRole()],
            )
            result = bias_calc.execute(experiment_data)
        ```
    Args:
        grouping_role (ABCRole | None, optional): The role defining the
            treatment assignment column (e.g., ``TreatmentRole()``).
            Defaults to None, which falls back to ``GroupingRole()``.
        target_roles (list[ABCRole] | None, optional): The role(s) defining
            the target outcome column(s) for which bias should be estimated.
            Defaults to None, which falls back to ``TargetRole()``.
        key (Any, optional): Optional identifier for the operator instance.
            Defaults to "".

    Attributes:
        calc_bias (staticmethod): Core computation that applies the dot
            product between feature differences and regression coefficients.
        prepare_data (staticmethod): Helper that constructs the matched
            dataset by exploding match indices and aggregating features.

    See Also:
        MatchingMetrics: The operator that uses the bias estimates to
            compute bias-corrected treatment effects (ATT, ATC, ATE).
        LstsqExtension: The backend-agnostic least-squares solver used
            to fit the regression coefficients.
    """
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: list[ABCRole] | None = None,
        key: Any = "",
    ):
        """
        Initialize the Bias calculator.

        Args:
            grouping_role (ABCRole | None, optional): The role defining the
                treatment assignment column. Defaults to None.
            target_roles (list[ABCRole] | None, optional): The role(s) defining
                the target outcome column(s). Defaults to None.
            key (Any, optional): Optional identifier for the operator instance.
                Defaults to "".
        """
        super().__init__(
            grouping_role=grouping_role, target_roles=target_roles, key=key
        )
    
    def _set_value(
            self, data: ExperimentData, value: Dataset, key=None
    ) -> ExperimentData:
        """
        Store the calculated bias values into the ExperimentData object.

        This method saves the bias estimates as additional fields in the
        ExperimentData, making them available for downstream operators
        (e.g., MatchingMetrics) to apply bias correction.

        Args:
            data (ExperimentData): The experiment data object to update.
            value (Dataset): The dataset containing bias estimates for
                treatment and/or control groups.
            key (Any, optional): Optional key for the stored value.
                Defaults to None.

        Returns:
            ExperimentData: The updated experiment data with bias values
                stored in the ``additional_fields`` space.
        """
        return data.set_value(
            space=ExperimentDataEnum.additional_fields,
            executor_id=self.id,
            value=value,
            role=value.roles,
        )
    
    @staticmethod
    def calc_bias(
        X: Dataset, X_matched: Dataset, coefficients: np.ndarray[float]
    ) -> list[float]:
        """
        Calculate bias as the dot product of feature differences and coefficients.

        Computes the residual bias by projecting the difference between
        original features (X) and matched (counterfactual) features
        (X_matched) onto the regression coefficients.

        The formula is: bias = (X - X_matched) · coefficients

        Args:
            X (Dataset): The original feature values for the group.
            X_matched (Dataset): The matched (counterfactual) feature values,
                typically computed by averaging over matched pairs.
            coefficients (np.ndarray[float]): The regression coefficients
                from the least-squares fit of target ~ features.

        Returns:
            list[float]: A list (or Dataset) containing the bias estimate
                for each observation in the group.

        Examples:
            ```python
                bias = Bias.calc_bias(
                    X=treatment_features,
                    X_matched=matched_features,
                    coefficients=regression_coefs
                )
            ```
        """
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
        """
        Core calculation logic for bias estimation.

        Fits a linear regression model (target ~ features) for each group
        using the matched observations, then computes the bias as the
        difference between original and matched features projected onto
        the regression coefficients.

        The method handles three scenarios:
            1. Only control group has matched data → compute bias for control
            2. Only treatment group has matched data → compute bias for treatment
            3. Both groups have matched data → compute bias for both

        Args:
            data (Dataset): The baseline (control) dataset.
            test_data (Dataset | None, optional): The compared (treatment) dataset.
                Defaults to None.
            target_fields (list[str] | None, optional): Names of the target fields.
                The first element is the original target, the second is the
                matched target. Defaults to None.
            features_fields (list[str] | None, optional): Names of the feature fields.
                The first half are original features, the second half are
                matched features. Defaults to None.
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            dict: A dictionary with keys "test" and/or "control" mapping to
                the bias estimates (Dataset) for each group.

        Raises:
            NoneArgumentError: If any of the required arguments (target_fields,
                features_fields, test_data) are None.
        """
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
        """
        Execute the inner bias calculation on grouped data.

        This classmethod serves as a wrapper that unpacks the grouped data
        and delegates to ``_inner_function``.

        Args:
            grouping_data: Grouped dataset containing the control and treatment slices.
                Expected format: [(control_name, control_data), (test_name, test_data)]
            target_fields (list[str] | None, optional): Names of the target fields.
                Defaults to None.
            features_fields (list[str] | None, optional): Names of the feature fields.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to ``_inner_function``.

        Returns:
            dict: The result of the ``_inner_function`` calculation, containing
                bias estimates for treatment and/or control groups.
        """
        return cls._inner_function(
            grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            features_fields=features_fields,
            **kwargs,
        )

    @staticmethod
    def prepare_data(data: ExperimentData, t_data: Dataset) -> Dataset:
        """
        Prepare matched data by aggregating features over matched pairs.

        This method constructs the counterfactual (matched) features by:
            1. Retrieving match indices from additional fields
            2. Exploding the indices to create one row per match
            3. Merging with the original data to get matched feature values
            4. Grouping by original index and computing the mean of matched features

        The result is a dataset where each observation has its matched
        (counterfactual) feature values, computed as the average over all
        its matches.

        Args:
            data (ExperimentData): The experiment data containing match indices
                in the ``additional_fields`` space.
            t_data (Dataset): The original dataset with features and targets.

        Returns:
            tuple[Dataset, Dataset]: A tuple containing:
                - indexes: The original match indices dataset
                - matched_data: The dataset with matched features, where each
                  column is suffixed with "_matched"

        Raises:
            ValueError: If no match indices are found in the additional fields.

        Examples:
            ```python
                indexes, matched_data = Bias.prepare_data(experiment_data, original_data)
                # matched_data contains columns like "feat1_matched", "feat2_matched"
            ```
        """
        indexes = data.field_search(AdditionalMatchingRole())
        if len(indexes) == 0:
            raise ValueError("No indexes were found")
        indexes = data.additional_fields[indexes]
        # additional fields are already allignet according to index

        numeric_cols = t_data.search_columns(
            [FeatureRole(), TargetRole()], search_types=[int, float]
        )
        
        matched_data = (
            indexes
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
            .set_index('_index')
        )
        matched_data.index.name = None

        return indexes, matched_data

    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute the bias estimation on the given experiment data.

        This method orchestrates the entire bias calculation process:
            1. Retrieves grouping and target fields
            2. Prepares matched data if necessary (when second target is missing)
            3. Computes bias estimates for treatment and/or control groups
            4. Stores the results in the ExperimentData object

        The bias estimates are stored as additional fields with the role
        ``AdditionalStatisticRole``, making them available for downstream
        operators (e.g., ``MatchingMetrics``) to apply bias correction.

        Args:
            data (ExperimentData): The experiment data containing the matched
                dataset and match indices.

        Returns:
            ExperimentData: The updated ExperimentData object with bias estimates
                stored in the ``additional_fields`` space.

        Examples:
            ```python
                bias_calc = Bias(
                    grouping_role=TreatmentRole(),
                    target_roles=[TargetRole()]
                )
                result_data = bias_calc.execute(experiment_data)
                # Bias estimates are now in result_data.additional_fields
            ```
        """
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
        bais_ds = Dataset.create_empty(
            backend=data.ds.backend_type,
            session=data.ds.session
        )
        for bais_res in compare_result.values():
            bais_ds = bais_ds.append(bais_res)
        
        bais_ds = bais_ds.rename({f"{col}": col for col in bais_ds.columns})
        bais_ds.roles = {col: AdditionalStatisticRole() for col in bais_res.columns}
        return self._set_value(data, bais_ds)

from __future__ import annotations

from typing import Any, Literal

from ..dataset import (
    ABCRole,
    AdditionalStatisticRole,
    AdditionalTargetRole,
    Dataset,
    ExperimentData,
    TargetRole,
)
from ..extensions import BiasExtension, MatchingMetricsExtension
from ..utils.enums import ExperimentDataEnum
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
        target_roles = target_roles or TargetRole()
        super().__init__(
            grouping_role=grouping_role,
            target_roles=(
                target_roles if isinstance(target_roles, list) else [target_roles]
            ),
            key=key,
        )

    def _write_log(file: str, result: str, time: str, mode: str = "a"):
        with open(file, mode) as f:
            f.write(result + ": " + time + "\n")

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        target_fields: list[str] | None = None,
        **kwargs,
    ) -> Any:
        pass

    @classmethod
    def _execute_inner_function(
        cls, grouping_data, target_fields: list[str] | None = None, **kwargs
    ) -> dict:
        pass

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
        _, target_fields = self._get_fields(data=data)
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.initial_ds.tmp_roles
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
            return data

        cls = backend_factory.resolve_backend(MatchingMetricsExtension, data.ds)
        compare_result = cls(self.grouping_role, self.target_roles, self.metric, self.n_neighbors).calc(data.ds)

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

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        target_fields: list[str] | None = None,
        features_fields: list[str] | None = None,
        **kwargs,
    ) -> dict:
        pass

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: list[str] | None = None,
        features_fields: list[str] | None = None,
        **kwargs,
    ) -> dict:
        pass

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
        _, target_fields = self._get_fields(data)

        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.initial_ds.tmp_roles
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
            return data
        cls = backend_factory.resolve_backend(BiasExtension, data.ds)
        compare_result: Dataset = cls(self.grouping_role, self.target_roles).calc(data.ds)
        compare_result.roles["bias"] = AdditionalStatisticRole()
        compare_result.roles["matched_target"] = AdditionalTargetRole()

        output = self._set_value(data, compare_result)
        if compare_result.is_persisted:
            compare_result.unpersist()
        return output

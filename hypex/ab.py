from __future__ import annotations

from typing import Literal

from .analyzers.ab import ABAnalyzer
from .comparators import Chi2Test, GroupDifference, GroupSizes, TTest, UTest
from .dataset import TargetRole, TreatmentRole
from .experiments.base import Experiment, OnRoleExperiment
from .ui.ab import ABOutput
from .ui.base import ExperimentShell
from .utils import ABNTestMethodsEnum


class ABTest(ExperimentShell):
    """A class for conducting A/B tests with configurable statistical tests and multiple testing correction.

    This class provides functionality to run A/B tests with options for different statistical tests
    (t-test, u-test, chi-square test) and multiple testing correction methods.

    Args:
        additional_tests (Union[str, List[str], None], optional): Statistical test(s) to run in addition to
            the default group difference calculation. Valid options are "t-test", "u-test", and "chi2-test".
            Can be a single test name or list of test names. Defaults to ["t-test"].
        multitest_method (str, optional): Method to use for multiple testing correction. Valid options are:
            "bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel", "fdr_bh", "fdr_by",
            "fdr_tsbh", "fdr_tsbhy", "quantile". Defaults to "holm".

            For more information refer to the statsmodels documentation:
            https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html

    Examples:
        Basic A/B test with default t-test::

            >>> ab_test = ABTest()
            >>> results = ab_test.execute(data)

        A/B test with multiple statistical tests::

            >>> ab_test = ABTest(
            ...     additional_tests=["t-test", "chi2-test"],
            ...     multitest_method="bonferroni"
            ... )
            >>> results = ab_test.execute(data)
    """

    @staticmethod
    def _make_experiment(additional_tests, multitest_method):
        """Creates an experiment configuration with specified statistical tests.

        Args:
            Args:
        additional_tests (Union[str, List[str], None], optional): Statistical test(s) to run in addition to
            the default group difference calculation. Valid options are "t-test", "u-test", and "chi2-test".
            Can be a single test name or list of test names. Defaults to ["t-test"].
        multitest_method (str, optional): Method to use for multiple testing correction. Valid options are:
            "bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel", "fdr_bh", "fdr_by",
            "fdr_tsbh", "fdr_tsbhy", "quantile". Defaults to "holm".
         For more information refer to the statsmodels documentation:
         <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html>

        Returns:
            Experiment: Configured experiment object with specified tests and correction method.
        """
        test_mapping = {
            "t-test": TTest(compare_by="groups", grouping_role=TreatmentRole()),
            "u-test": UTest(compare_by="groups", grouping_role=TreatmentRole()),
            "chi2-test": Chi2Test(compare_by="groups", grouping_role=TreatmentRole()),
        }
        on_role_executors = [GroupDifference(grouping_role=TreatmentRole())]
        additional_tests = ["t-test"] if additional_tests is None else additional_tests
        additional_tests = (
            additional_tests
            if isinstance(additional_tests, list)
            else [additional_tests]
        )
        for i in additional_tests:
            on_role_executors += [test_mapping[i]]
        return Experiment(
            executors=[
                GroupSizes(grouping_role=TreatmentRole()),
                OnRoleExperiment(
                    executors=on_role_executors,
                    role=TargetRole(),
                ),
                ABAnalyzer(
                    multitest_method=(
                        ABNTestMethodsEnum(multitest_method)
                        if multitest_method
                        else None
                    )
                ),
            ]
        )

    def __init__(
        self,
        additional_tests: (
            Literal["t-test", "u-test", "chi2-test"]
            | list[Literal["t-test", "u-test", "chi2-test"]]
            | None
        ) = None,
        multitest_method: (
            Literal[
                "bonferroni",
                "sidak",
                "holm-sidak",
                "holm",
                "simes-hochberg",
                "hommel",
                "fdr_bh",
                "fdr_by",
                "fdr_tsbh",
                "fdr_tsbhy",
                "quantile",
            ]
            | None
        ) = "holm",
        t_test_equal_var: bool | None = None,
    ):
        super().__init__(
            experiment=self._make_experiment(additional_tests, multitest_method),
            output=ABOutput(),
        )
        if t_test_equal_var is not None:
            self.experiment.set_params({TTest: {"calc_kwargs": {"equal_var": t_test_equal_var}}})

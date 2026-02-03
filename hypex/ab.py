from __future__ import annotations

from typing import Literal

from .analyzers.ab import ABAnalyzer
from .comparators import Chi2Test, GroupDifference, GroupSizes, KSTest, TTest, UTest
from .dataset import AdditionalTargetRole, TargetRole, TreatmentRole
from .executor.executor import Executor
from .experiments.base import Experiment, OnRoleExperiment
from .transformers import CUPEDTransformer
from .ui.ab import ABOutput
from .ui.base import ExperimentShell
from .utils import ABNTestMethodsEnum, ABTestTypesEnum


class ABTest(ExperimentShell):
    """A class for conducting A/B tests with configurable statistical tests and multiple testing correction.

    This class provides functionality to run A/B tests with options for different statistical tests
    (t-test, u-test, chi-square test) and multiple testing correction methods.

    Args:
        additional_tests (Union[str, ABTestTypesEnum, List[Union[str, ABTestTypesEnum]], None], optional): Statistical test(s) to run in addition to
            the default group difference calculation. Valid options are 't-test', 'u-test', 'chi2-test' or ABTestTypesEnum.t_test, ABTestTypesEnum.u_test, and ABTestTypesEnum.chi2_test.
            Can be a single test name/enum or list of test names/enums. Defaults to [ABTestTypesEnum.t_test].
        multitest_method (ABNTestMethodsEnum, optional): Method to use for multiple testing correction. Valid options are:
            ABNTestMethodsEnum.bonferroni, ABNTestMethodsEnum.sidak, etc. Defaults to ABNTestMethodsEnum.holm.

            For more information refer to the statsmodels documentation:
            https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html

    Examples
    --------
    .. code-block:: python

        # Basic A/B test with default t-test
        ab_test = ABTest()
        results = ab_test.execute(data)

        # A/B test with multiple statistical tests
        ab_test = ABTest(
            additional_tests=[ABTestTypesEnum.t_test, ABTestTypesEnum.chi2_test],
            multitest_method=ABNTestMethodsEnum.bonferroni,
            cuped_features={"target_feature": "pre_target_feature"},
            enable_cupac=True,
            cupac_models=['linear', 'ridge']
        )
        results = ab_test.execute(data)
    """

    @staticmethod
    def _make_experiment(
        additional_tests: str | ABTestTypesEnum | list[str | ABTestTypesEnum] | None,
        multitest_method: ABNTestMethodsEnum | str | None,
        cuped_features: dict[str, str] | None,
        cupac_models: str | list[str] | None,
        enable_cupac: bool,
    ) -> Experiment:
        test_mapping: dict[str, Executor] = {
            "t-test": TTest(compare_by="groups", grouping_role=TreatmentRole()),
            "ks-test": KSTest(compare_by="groups", grouping_role=TreatmentRole()),
            "u-test": UTest(compare_by="groups", grouping_role=TreatmentRole()),
            "chi2-test": Chi2Test(compare_by="groups", grouping_role=TreatmentRole()),
        }
        on_role_executors: list[Executor] = [
            GroupDifference(grouping_role=TreatmentRole())
        ]
        additional_tests = (
            [ABTestTypesEnum.t_test] if additional_tests is None else additional_tests
        )
        multitest_method = (
            ABNTestMethodsEnum(multitest_method)
            if (
                multitest_method is not None
                and multitest_method in ABNTestMethodsEnum.__members__.values()
            )
            else ABNTestMethodsEnum.holm
        )
        if additional_tests:
            if isinstance(additional_tests, list):
                additional_tests = [
                    ABTestTypesEnum(test) if isinstance(test, str) else test
                    for test in additional_tests
                ]
            else:
                additional_tests = (
                    ABTestTypesEnum(additional_tests)
                    if isinstance(additional_tests, str)
                    else additional_tests
                )
        additional_tests = (
            additional_tests
            if isinstance(additional_tests, list)
            else [additional_tests]
        )
        for test_name in additional_tests:
            on_role_executors.append(test_mapping[test_name.value])

        # Build base executors list
        executors: list[Executor] = [
            GroupSizes(grouping_role=TreatmentRole()),
            OnRoleExperiment(
                executors=on_role_executors,
                role=(
                    [TargetRole(), AdditionalTargetRole()]
                    if enable_cupac
                    else TargetRole()
                ),
            ),
            ABAnalyzer(
                multitest_method=(
                    ABNTestMethodsEnum(multitest_method) if multitest_method else None
                )
            ),
        ]
        if cuped_features:
            executors.insert(0, CUPEDTransformer(cuped_features=cuped_features))

        if enable_cupac:
            from .ml import CUPACExecutor

            executors.insert(0, CUPACExecutor(cupac_models=cupac_models))

        return Experiment(executors=executors)

    def __init__(
        self,
        additional_tests: (
            str | ABTestTypesEnum | list[str | ABTestTypesEnum] | None
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
        cuped_features: dict[str, str] | None = None,
        cupac_models: str | list[str] | None = None,
        enable_cupac: bool = False,
    ):
        """
        Args:
            additional_tests: Statistical test(s) to run in addition to the default group difference calculation. Valid options are 't-test', 'u-test', 'chi2-test' or ABTestTypesEnum.t_test, ABTestTypesEnum.u_test, and ABTestTypesEnum.chi2_test. Can be a single test name/enum or list of test names/enums. Defaults to [ABTestTypesEnum.t_test].
            multitest_method: Method to use for multiple testing correction. Valid options are ABNTestMethodsEnum.bonferroni, ABNTestMethodsEnum.sidak, etc. Defaults to ABNTestMethodsEnum.holm.
            t_test_equal_var: Whether to use equal variance in t-test (optional).
            cuped_features: dict[str, str] — Dictionary {target_feature: pre_target_feature} for CUPED. Only dict is allowed.
            cupac_models: str | list[str] — model name (e.g. 'linear', 'ridge', 'lasso', 'catboost') or list of model names to try. If None, all available models will be tried and the best will be selected by variance reduction.
            enable_cupac: bool — Enable CUPAC variance reduction. CUPAC configuration is extracted from dataset.features_mapping.
        """
        super().__init__(
            experiment=self._make_experiment(
                additional_tests,
                multitest_method,
                cuped_features,
                cupac_models,
                enable_cupac,
            ),
            output=ABOutput(),
        )
        if t_test_equal_var is not None:
            self.experiment.set_params(
                {TTest: {"calc_kwargs": {"equal_var": t_test_equal_var}}}
            )

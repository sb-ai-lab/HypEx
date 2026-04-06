from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from .analyzers.ab import ABAnalyzer
from .comparators import Chi2Test, GroupDifference, GroupSizes, KSTest, TTest, UTest
from .dataset import AdditionalTargetRole, TargetRole, TreatmentRole
from .executor.executor import Executor
from .experiments.base import Experiment, OnRoleExperiment
from .transformers import CUPEDTransformer
from .ui.ab import ABOutput
from .ui.base import ExperimentShell
from .utils import ABNTestMethodsEnum, ABTestTypesEnum
from .utils.enums import MLModeEnum


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

        # A/B test with CUPAC - train and apply
        ab_test = ABTest(
            cupac_mode='fit_predict',
            cupac_models=['linear', 'ridge']
        )
        results = ab_test.execute(data)
        
        # CUPAC with fit/predict pattern
        # Step 1: Fit on historical data
        ab_test_fit = ABTest(
            cupac_mode='fit',
            cupac_models=['linear', 'ridge']
        )
        results = ab_test_fit.execute(historical_data)
        
        # Step 2: Predict on current data
        ab_test_predict = ABTest(
            cupac_mode='predict',
            experiment_id='experiment_id_from_first_run'
        )
        results = ab_test_predict.execute(current_data)
    """

    @staticmethod
    def _make_experiment(
        additional_tests: Optional[
            Union[str, ABTestTypesEnum, List[Union[str, ABTestTypesEnum]]]
        ],
        multitest_method: Optional[Union[ABNTestMethodsEnum, str]],
        cuped_features: Optional[Dict[str, str]],
        cupac_models: Optional[Union[str, List[str]]],
        cupac_mode: Optional[Union[str, MLModeEnum]],
        experiment_id: Optional[str],
    ) -> Experiment:
        test_mapping: Dict[str, Executor] = {
            "t-test": TTest(compare_by="groups", grouping_role=TreatmentRole()),
            "ks-test": KSTest(compare_by="groups", grouping_role=TreatmentRole()),
            "u-test": UTest(compare_by="groups", grouping_role=TreatmentRole()),
            "chi2-test": Chi2Test(compare_by="groups", grouping_role=TreatmentRole()),
        }
        on_role_executors: List[Executor] = [
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

        use_cupac = cupac_mode is not None

        # Build base executors list
        executors: List[Executor] = [
            GroupSizes(grouping_role=TreatmentRole()),
            OnRoleExperiment(
                executors=on_role_executors,
                role=(
                    [TargetRole(), AdditionalTargetRole()]
                    if use_cupac
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

        if use_cupac:
            from .experiments.cupac import CupacExperiment

            ml_experiment = CupacExperiment(
                cupac_models=cupac_models,
                mode=cupac_mode,
                experiment_id=experiment_id,
            )
            executors.insert(0, ml_experiment)

        return Experiment(executors=executors)

    def __init__(
        self,
        additional_tests: Optional[
            Union[str, ABTestTypesEnum, List[Union[str, ABTestTypesEnum]]]
        ] = None,
        multitest_method: Optional[
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
        ] = "holm",
        t_test_equal_var: Optional[bool] = None,
        cuped_features: Optional[Dict[str, str]] = None,
        cupac_models: Optional[Union[str, List[str]]] = None,
        cupac_mode: Optional[Literal["fit", "predict", "fit_predict"]] = None,
        experiment_id: Optional[str] = None,
    ):
        """
        Args:
            additional_tests: Statistical test(s) to run in addition to the default group difference calculation. 
                Valid options are 't-test', 'u-test', 'chi2-test' or ABTestTypesEnum.t_test, ABTestTypesEnum.u_test, 
                and ABTestTypesEnum.chi2_test. Can be a single test name/enum or list of test names/enums. 
                Defaults to [ABTestTypesEnum.t_test].
            multitest_method: Method to use for multiple testing correction. Valid options are 
                ABNTestMethodsEnum.bonferroni, ABNTestMethodsEnum.sidak, etc. Defaults to ABNTestMethodsEnum.holm.
            t_test_equal_var: Whether to use equal variance in t-test (optional).
            cuped_features: dict[str, str] — Dictionary {target_feature: pre_target_feature} for CUPED. Only dict is allowed.
            cupac_models: str | list[str] — model name (e.g. 'linear', 'ridge', 'lasso', 'catboost') or list of model names to try. 
                If None, all available models will be tried and the best will be selected by variance reduction.
            cupac_mode: str | None — Execution mode for CUPAC. If None, CUPAC is disabled:
                - 'fit_predict': Train models and apply adjustment (default)
                - 'fit': Only train models, save for later use
                - 'predict': Only apply adjustment using pre-trained models
            experiment_id: str | None — When cupac_mode='predict', specifies the experiment ID to load models from.
                This is the folder name created during a previous run with mode='fit' or 'fit_predict'.
        
        Examples:
            # Standard CUPAC - train and apply in one go
            ab_test = ABTest(
                cupac_mode='fit_predict',
                cupac_models=['linear', 'ridge']
            )
            
            # Fit on virtual target with lagged features
            ab_test_fit = ABTest(
                cupac_mode='fit',
                cupac_models=['linear']
            )
            result = ab_test_fit.execute(virtual_data)
            
            # Apply to real target (no lagged features needed)
            import os
            exp_id = os.listdir('.hypex_experiments')[0]
            ab_test_predict = ABTest(
                cupac_mode='predict',
                experiment_id=exp_id
            )
            result = ab_test_predict.execute(real_data)
        """
        super().__init__(
            experiment=self._make_experiment(
                additional_tests,
                multitest_method,
                cuped_features,
                cupac_models,
                cupac_mode,
                experiment_id,
            ),
            output=ABOutput(),
        )
        if t_test_equal_var is not None:
            self.experiment.set_params(
                {TTest: {"calc_kwargs": {"equal_var": t_test_equal_var}}}
            )

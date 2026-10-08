from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .analyzers.aa import AADryTestAnalyzer, AAScoreAnalyzer, OneAAStatAnalyzer
from .comparators import Chi2Test, GroupDifference, GroupSizes, KSTest, TTest
from .comparators.abstract import Comparator
from .dataset import AdditionalTreatmentRole, FeatureRole, TargetRole
from .executor import Executor
from .experiments.base import Experiment, OnRoleExperiment
from .experiments.base_complex import IfParamsExperiment, ParamsExperiment
from .forks.aa import IfAAExecutor
from .reporters import AATestReporter, DatasetReporter, DictReporter
from .splitters import AASplitter, AASplitterWithStratification
from .transformers.float32_caster import Float32Caster
from .transformers.na_dropper import NaDropper
from .ui.aa import AAOutput
from .ui.base import ExperimentShell
from .utils import SpaceEnum


class AATest(ExperimentShell):
    """A class for conducting A/A tests with configurable parameters.

    Args:
        precision_mode (bool, optional): If True, runs more iterations (2000) in order to tackle type 1 error.
            If False, runs fewer iterations (10) for quicker results. Defaults to False.
        control_size (float, optional): The proportion of data to allocate to control group.
            Must be between 0 and 1. Defaults to 0.5.
        stratification (bool, optional): Whether to use stratified sampling when splitting data.
            Defaults to False.
        n_iterations (int, optional): Number of test iterations to run. If None, determined by
            precision_mode. Defaults to None.
        sample_size (float, optional): Fraction of data to sample for each test.
            Must be between 0 and 1. If None, uses full dataset. Defaults to None.
        additional_params (Dict[str, Any], optional): Additional parameters to pass to the
            experiment pipeline. Defaults to None.
        random_states (Iterable[int], optional): Random seeds to use for each iteration.
            If None, uses range(n_iterations). Defaults to None.
        equal_variance (bool | None, optional): Assume equal variance in t-test.
            If True, use Student's t-test. If False, use Welch's t-test.
            If None (default), Welch's t-test is used.
        groups_sizes (list[float] | None, optional): Custom group size proportions. Defaults to None.
        float32 (bool, optional): Cast float columns to float32 for memory savings. Defaults to False.
        early_stopping (bool, optional): Stop when all features pass (no differences
            detected) on any iteration. Defaults to False.
        t_test_equal_var (bool | None, optional): Deprecated alias for ``equal_variance``.
    ...
    """

    @staticmethod
    def _make_experiment(
        stratification: bool,
        n_iterations: int,
        control_size: float,
        sample_size: float | None,
        additional_params: dict[str, Any] | None,
        random_states: Iterable[int] | None,
        groups_sizes: list[float] | None,
        dry_test: bool,
        float32: bool = False,
        early_stopping: bool = False,
    ) -> Experiment:
        """Builds the experiment pipeline for A/A testing."""
        aa_metrics = Experiment(
            executors=[
                GroupSizes(grouping_role=AdditionalTreatmentRole()),
                OnRoleExperiment(
                    executors=[
                        GroupDifference(
                            compare_by="groups", grouping_role=AdditionalTreatmentRole()
                        ),
                        TTest(
                            grouping_role=AdditionalTreatmentRole(),
                            target_roles=TargetRole(),
                            reliability=0.05,
                        ),
                        KSTest(
                            compare_by="groups", grouping_role=AdditionalTreatmentRole()
                        ),
                        Chi2Test(
                            compare_by="groups", grouping_role=AdditionalTreatmentRole()
                        ),
                    ],
                    role=[TargetRole(), FeatureRole()],
                ),
                OneAAStatAnalyzer(),
            ]
        )

        pre_executors: list[Executor] = [NaDropper()]

        one_aa_base = Experiment(executors=[*pre_executors, AASplitter(), aa_metrics])
        one_aa_strat = Experiment(
            executors=[*pre_executors, AASplitterWithStratification(), aa_metrics]
        )
        base_experiment = one_aa_strat if stratification else one_aa_base

        # Float32Caster is applied ONCE before the iterative ParamsExperiment,
        # not inside each iteration.
        outer_executors: list[Executor] = []
        if float32:
            outer_executors.append(Float32Caster())

        params = AATest._prepare_params(
            n_iterations,
            control_size,
            random_states,
            sample_size,
            additional_params,
            groups_sizes,
        )

        experiment_params = [
            ParamsExperiment(
                executors=[base_experiment],
                params=params,
                reporter=DatasetReporter(
                    AATestReporter(
                        dict_reporter=DictReporter(front=False),
                        output_format="dict",
                    ),
                    single_row=True,
                ),
                stopping_criterion=(
                    IfAAExecutor(all_features_passed=True) if early_stopping else None
                ),
            )
        ]

        if sample_size:
            params_no_sample = AATest._prepare_params(
                n_iterations,
                control_size,
                random_states,
                sample_size=None,
                additional_params=additional_params,
                groups_sizes=groups_sizes,
            )
            experiment_params.append(
                IfParamsExperiment(
                    executors=[base_experiment],
                    params=params_no_sample,
                    reporter=DatasetReporter(
                        AATestReporter(
                            dict_reporter=DictReporter(front=False),
                            output_format="dict",
                        ),
                        single_row=True,
                    ),
                    stopping_criterion=IfAAExecutor(sample_size=sample_size),
                )
            )

        if dry_test:
            experiment_params.append(AADryTestAnalyzer())
        experiment_params.append(AAScoreAnalyzer())

        return Experiment([*outer_executors, *experiment_params], key="AATest")

    @staticmethod
    def _prepare_params(
        n_iterations: int,
        control_size: float,
        random_states: Iterable[int] | None = None,
        sample_size: float | None = None,
        additional_params: dict[str, Any] | None = None,
        groups_sizes: list[float] | None = None,
    ) -> dict[type, dict[str, Any]]:
        """Prepares parameters for the A/A test experiment.

        Args:
            n_iterations (int): Number of test iterations to run.
            control_size (float): The proportion of data to allocate to control group.
            random_states (Iterable[int] | None): Random seeds to use for each iteration.
            sample_size (float | None): Fraction of data to sample for each test.
            additional_params (dict[str, Any] | None): Additional parameters to pass.
            groups_sizes (list[float] | None): Custom group size proportions.

        Returns:
            Dict[type, Dict[str, Any]]: Dictionary mapping executor classes to their
                parameter configurations.
        """
        random_states = random_states or range(n_iterations)
        additional_params = additional_params or {}
        params = {
            AASplitter: {
                "random_state": random_states,
                "control_size": [control_size],
                "sample_size": [sample_size],
                "groups_sizes": [groups_sizes],
            },
            Comparator: {
                "grouping_role": [AdditionalTreatmentRole()],
                "space": [SpaceEnum.additional],
            },
        }

        params.update(additional_params)
        return params

    def __init__(
        self,
        precision_mode: bool = False,
        control_size: float = 0.5,
        stratification: bool = False,
        n_iterations: int | None = None,
        sample_size: float | None = None,
        additional_params: dict[str, Any] | None = None,
        random_states: Iterable[int] | None = None,
        equal_variance: bool | None = None,
        groups_sizes: list[float] | None = None,
        float32: bool = False,
        early_stopping: bool = False,
        t_test_equal_var: bool | None = None,
        dry_test: bool = False,
    ):
        import warnings

        if t_test_equal_var is not None:
            warnings.warn(
                "t_test_equal_var is deprecated and will be removed in a "
                "future version. Use equal_variance instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if equal_variance is None:
                equal_variance = t_test_equal_var

        if n_iterations is None:
            n_iterations = 2000 if precision_mode else 10
        if early_stopping and precision_mode:
            import warnings

            warnings.warn(
                "early_stopping=True combined with precision_mode=True may "
                "stop after very few iterations, making AA-score and FPR "
                "estimates unreliable. Consider disabling one of them.",
                UserWarning,
                stacklevel=2,
            )

        if early_stopping and dry_test:
            import warnings

            warnings.warn(
                "early_stopping=True combined with dry_test=True may produce "
                "a p-value distribution from too few iterations for "
                "meaningful uniformity diagnostics.",
                UserWarning,
                stacklevel=2,
            )

        super().__init__(
            experiment=self._make_experiment(
                stratification=stratification,
                n_iterations=n_iterations,
                control_size=control_size,
                sample_size=sample_size,
                additional_params=additional_params,
                random_states=random_states,
                groups_sizes=groups_sizes,
                float32=float32,
                early_stopping=early_stopping,
                dry_test=dry_test,
            ),
            output=AAOutput(),
        )

        if equal_variance is not None:
            self.experiment.set_params(
                {TTest: {"calc_kwargs": {"equal_variance": equal_variance}}}
            )

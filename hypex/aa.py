from __future__ import annotations

from typing import Any, Iterable

from .analyzers.aa import AAScoreAnalyzer, OneAAStatAnalyzer
from .comparators import GroupDifference, GroupSizes
from .comparators.abstract import Comparator
from .comparators.hypothesis_testing import Chi2Test, KSTest, TTest
from .dataset import AdditionalTreatmentRole, TargetRole
from .experiments.base import Experiment, OnRoleExperiment
from .experiments.base_complex import IfParamsExperiment, ParamsExperiment
from .forks.aa import IfAAExecutor
from .reporters import DatasetReporter
from .reporters.aa import OneAADictReporter
from .splitters import AASplitter, AASplitterWithStratification
from .ui.aa import AAOutput
from .ui.base import ExperimentShell
from .utils import SpaceEnum

AA_METRICS = Experiment(
    executors=[
        GroupSizes(grouping_role=AdditionalTreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(
                    compare_by="groups", grouping_role=AdditionalTreatmentRole()
                ),
                TTest(compare_by="groups", grouping_role=AdditionalTreatmentRole()),
                KSTest(compare_by="groups", grouping_role=AdditionalTreatmentRole()),
                Chi2Test(compare_by="groups", grouping_role=AdditionalTreatmentRole()),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)

ONE_AA_TEST = Experiment(executors=[AASplitter(), AA_METRICS])
ONE_AA_TEST_WITH_STRATIFICATION = Experiment(
    executors=[AASplitterWithStratification(), AA_METRICS]
)

AA_TEST = Experiment(
    [
        ParamsExperiment(
            executors=([ONE_AA_TEST]),
            params={
                AASplitter: {"random_state": range(2000), "control_size": [0.5]},
                Comparator: {
                    "grouping_role": [AdditionalTreatmentRole()],
                    "space": [SpaceEnum.additional],
                },
            },
            reporter=DatasetReporter(OneAADictReporter(front=False)),
        ),
        AAScoreAnalyzer(),
    ],
    key="AATest",
)
AA_TEST_WITH_STRATIFICATION = Experiment(
    [
        ParamsExperiment(
            executors=([ONE_AA_TEST_WITH_STRATIFICATION]),
            params={
                AASplitter: {"random_state": range(2000), "control_size": [0.5]},
                Comparator: {
                    "grouping_role": [AdditionalTreatmentRole()],
                    "space": [SpaceEnum.additional],
                },
            },
            reporter=DatasetReporter(OneAADictReporter(front=False)),
        ),
        AAScoreAnalyzer(),
    ],
    key="AATest",
)


class AATest(ExperimentShell):
    """A class for conducting A/A tests with configurable parameters.

    This class provides functionality to run A/A tests with options for stratification,
    precision control (fast or with type 1 error control), and sample size specification.
    It sets up the experiment pipeline with appropriate parameters, performs homogeneity
    tests for each split in order to evaluate their quality and to identify the best one.

    Args:
        precision_mode (bool, optional): If True, runs more iterations (2000) in order to tuckle type 1 error.
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
        t_test_equal_var (bool, optional): If True (default), perform a standard independent 2 sample
            test that assumes equal population variances. If False, perform Welch's t-test,
            which does not assume equal population variance.


    Examples
    --------
    .. code-block:: python

        # Basic A/A test with default parameters
        aa_test = AATest()
        results = aa_test.execute(data)

        # High precision A/A test with stratification
        aa_test = AATest(
            precision_mode=True,
            stratification=True,
            control_size=0.5
        )
        results = aa_test.execute(data)

        # A/A test with custom sample size and iterations
        aa_test = AATest(
            sample_size=0.8,
            n_iterations=100
        )
        results = aa_test.execute(data)
    """

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

        Returns:
            Dict[type, Dict[str, Any]]: Dictionary mapping executor classes to their
                parameter configurations.

        Examples
        --------
        .. code-block:: python

            params = AATest._prepare_params(
                n_iterations=10,
                control_size=0.5,
                sample_size=0.8
            )
            print(params[AASplitter]["control_size"])
            # [0.5]
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
        t_test_equal_var: bool | None = None,
        groups_sizes: list[float] | None = None,
    ):
        if n_iterations is None:
            if precision_mode:
                n_iterations = 2000
            else:
                n_iterations = 10
        experiment_params = [
            ParamsExperiment(
                executors=(
                    [ONE_AA_TEST_WITH_STRATIFICATION if stratification else ONE_AA_TEST]
                ),
                params=self._prepare_params(
                    n_iterations,
                    control_size,
                    random_states,
                    sample_size,
                    additional_params,
                    groups_sizes,
                ),
                reporter=DatasetReporter(OneAADictReporter(front=False)),
            )
        ]
        if sample_size:
            experiment_params.append(
                IfParamsExperiment(
                    executors=(
                        [
                            (
                                ONE_AA_TEST_WITH_STRATIFICATION
                                if stratification
                                else ONE_AA_TEST
                            )
                        ]
                    ),
                    params=self._prepare_params(
                        n_iterations,
                        control_size,
                        random_states,
                        additional_params,
                        groups_sizes,
                    ),
                    reporter=DatasetReporter(OneAADictReporter(front=False)),
                    stopping_criterion=IfAAExecutor(sample_size=sample_size),
                )
            )
        experiment_params.append(AAScoreAnalyzer())
        super().__init__(
            experiment=Experiment(
                experiment_params,
                key="AATest",
            ),
            output=AAOutput(),
        )
        if t_test_equal_var is not None:
            self.experiment.set_params(
                {TTest: {"calc_kwargs": {"equal_var": t_test_equal_var}}}
            )

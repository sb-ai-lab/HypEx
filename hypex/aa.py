from typing import Optional, Dict, Any, Iterable

from .forks.aa import IfAAExecutor
from .analyzers.aa import OneAAStatAnalyzer, AAScoreAnalyzer
from .comparators import GroupDifference, GroupSizes
from .comparators.abstract import Comparator
from .comparators.hypothesis_testing import TTest, KSTest, Chi2Test
from .dataset import AdditionalTreatmentRole
from .dataset import TargetRole
from .experiments.base import Experiment, OnRoleExperiment
from .experiments.base_complex import ParamsExperiment, IfParamsExperiment
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
    @staticmethod
    def _prepare_params(
        n_iterations: int,
        control_size: float,
        random_states: Optional[Iterable[int]] = None,
        sample_size: Optional[float] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[type, Dict[str, Any]]:
        random_states = random_states or range(n_iterations)
        additional_params = additional_params or {}
        params = {
            AASplitter: {
                "random_state": random_states,
                "control_size": [control_size],
                "sample_size": [sample_size],
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
        t1_error_handling: bool = False,
        control_size: float = 0.5,
        stratification: bool = False,
        n_iterations: Optional[int] = None,
        sample_size: Optional[float] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        random_states: Optional[Iterable[int]] = None,
    ):
        if n_iterations is None:
            if t1_error_handling:
                n_iterations = 2000
            else:
                n_iterations = 10
        super().__init__(
            experiment=Experiment(
                [
                    ParamsExperiment(
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
                            sample_size,
                            additional_params,
                        ),
                        reporter=DatasetReporter(OneAADictReporter(front=False)),
                    ),
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
                        ),
                        reporter=DatasetReporter(OneAADictReporter(front=False)),
                        stopping_criterion=IfAAExecutor(sample_size=sample_size),
                    ),
                    AAScoreAnalyzer(),
                ],
                key="AATest",
            ),
            output=AAOutput(),
        )

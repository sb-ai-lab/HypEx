from typing import Optional, Dict, Any, Iterable

from hypex.analyzers.aa import OneAAStatAnalyzer, AAScoreAnalyzer
from hypex.comparators import (
    Comparator,
    GroupDifference,
    GroupSizes,
    TTest,
    KSTest,
    Chi2Test,
)
from hypex.dataset import AdditionalTreatmentRole, TargetRole, TreatmentRole
from hypex.experiments.base import Experiment, OnRoleExperiment
from hypex.experiments.base_complex import ParamsExperiment, IfParamsExperiment
from hypex.forks.aa import IfAAExecutor
from hypex.reporters import DatasetReporter
from hypex.reporters.aa import OneAADictReporter
from hypex.splitters import AASplitter, AASplitterWithStratification
from hypex.ui.aa import AAOutput
from hypex.ui.base import ExperimentShell
from hypex.utils import SpaceEnum

ONE_AA_TEST = Experiment(
    executors=[
        AASplitter(),
        GroupSizes(grouping_role=TreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(compare_by="groups", grouping_role=TreatmentRole()),
                TTest(compare_by="groups", grouping_role=TreatmentRole()),
                KSTest(compare_by="groups", grouping_role=TreatmentRole()),
                Chi2Test(compare_by="groups", grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)

ONE_AA_TEST_WITH_STRATIFICATION = Experiment(
    executors=[
        AASplitterWithStratification(),
        GroupSizes(grouping_role=TreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(compare_by="groups", grouping_role=TreatmentRole()),
                TTest(compare_by="groups", grouping_role=TreatmentRole()),
                KSTest(compare_by="groups", grouping_role=TreatmentRole()),
                Chi2Test(compare_by="groups", grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
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
        n_iterations: int = 2000,
        control_size: float = 0.5,
        stratification: bool = False,
        sample_size: Optional[float] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        random_states: Optional[Iterable[int]] = None,
    ):
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

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

    def create_experiment(self, **kwargs) -> Experiment:
        return Experiment(
            [
                ParamsExperiment(
                    executors=(
                        [
                            (
                                ONE_AA_TEST_WITH_STRATIFICATION
                                if kwargs.get("stratification")
                                else ONE_AA_TEST
                            )
                        ]
                    ),
                    params=self._prepare_params(
                        kwargs.get("n_iterations", 2000),
                        kwargs.get("control_size", 0.5),
                        kwargs.get("random_states", None),
                        kwargs.get("sample_size", None),
                        kwargs.get("additional_params", None),
                    ),
                    reporter=DatasetReporter(OneAADictReporter(front=False)),
                ),
                IfParamsExperiment(
                    executors=(
                        [
                            (
                                ONE_AA_TEST_WITH_STRATIFICATION
                                if kwargs.get("stratification")
                                else ONE_AA_TEST
                            )
                        ]
                    ),
                    params=self._prepare_params(
                        kwargs.get("n_iterations", 2000),
                        kwargs.get("control_size", 0.5),
                        kwargs.get("random_states", None),
                        kwargs.get("additional_params", None),
                    ),
                    reporter=DatasetReporter(OneAADictReporter(front=False)),
                    stopping_criterion=IfAAExecutor(
                        sample_size=kwargs.get("sample_size", None)
                    ),
                ),
                AAScoreAnalyzer(),
            ],
            key="AATest",
        )

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
            output=AAOutput(),
            create_experiment_kwargs={
                "n_iterations": n_iterations,
                "control_size": control_size,
                "stratification": stratification,
                "sample_size": sample_size,
                "additional_params": additional_params,
                "random_states": random_states,
            },
        )

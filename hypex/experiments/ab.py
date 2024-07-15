from typing import Optional

from hypex.analyzers.ab import ABAnalyzer
from hypex.comparators import GroupDifference, GroupSizes, TTest, UTest, Chi2Test
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments import Experiment, OnRoleExperiment
from hypex.ui.ab import ABOutput
from hypex.ui.base import ExperimentShell
from hypex.utils import ABNTestMethodsEnum

AB_TEST = Experiment(
    executors=[
        GroupSizes(grouping_role=TreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(grouping_role=TreatmentRole()),
                TTest(grouping_role=TreatmentRole()),
                UTest(grouping_role=TreatmentRole()),
                Chi2Test(grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
        ABAnalyzer(multitest_method=ABNTestMethodsEnum.quantile),
    ]
)


class ABTest(ExperimentShell):

    def __init__(
        self,
        multitest_method: Optional[str] = None,
    ):
        experiment = AB_TEST
        experiment.executors[2].multitest_method = ABNTestMethodsEnum(multitest_method)
        super().__init__(
            experiment=experiment,
            output=ABOutput(),
        )

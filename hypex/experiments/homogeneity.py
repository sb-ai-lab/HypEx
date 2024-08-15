from hypex.analyzers.aa import OneAAStatAnalyzer
from hypex.comparators import GroupDifference, GroupSizes
from hypex.comparators import TTest, KSTest, Chi2Test
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments.base import Experiment, OnRoleExperiment
from hypex.ui.base import ExperimentShell
from hypex.ui.homo import HomoOutput
from hypex.utils import SpaceEnum

HOMOGENEITY_TEST = Experiment(
    executors=[
        OnRoleExperiment(
            executors=[
                GroupSizes(grouping_role=TreatmentRole(), compare_by="groups"),
                GroupDifference(grouping_role=TreatmentRole(), compare_by="groups"),
                TTest(grouping_role=TreatmentRole(), compare_by="groups"),
                KSTest(grouping_role=TreatmentRole(), compare_by="groups"),
                Chi2Test(grouping_role=TreatmentRole(), compare_by="groups"),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)


class HomogeneityTest(ExperimentShell):
    def __init__(self):
        super().__init__(
            experiment=HOMOGENEITY_TEST,
            output=HomoOutput(),
        )

from hypex.analyzers.aa import OneAAStatAnalyzer
from hypex.comparators import GroupDifference, GroupSizes
from hypex.comparators import TTest, KSTest, Chi2Test
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments import Experiment, OnRoleExperiment
from hypex.utils import SpaceEnum

HOMOGENEITY_TEST = Experiment(
    executors=[
        OnRoleExperiment(
            executors=[
                GroupSizes(grouping_role=TreatmentRole(), space=SpaceEnum.data),
                GroupDifference(grouping_role=TreatmentRole(), space=SpaceEnum.data),
                TTest(grouping_role=TreatmentRole(), space=SpaceEnum.data),
                KSTest(grouping_role=TreatmentRole(), space=SpaceEnum.data),
                Chi2Test(grouping_role=TreatmentRole(), space=SpaceEnum.data),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)

from hypex.analyzers import OneAAStatAnalyzer
from hypex.comparators import GroupDifference, GroupSizes
from hypex.comparators.hypothesis_testing import TTest, KSTest, Chi2Test
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments import Experiment, OnRoleExperiment
from hypex.splitters import AASplitter
from hypex.utils import SpaceEnum

ONE_AA_TEST = Experiment(
    executors=[
        AASplitter(),
        GroupSizes(grouping_role=TreatmentRole(), space=SpaceEnum.additional),
        OnRoleExperiment(
            executors=[
                GroupDifference(
                    grouping_role=TreatmentRole(), space=SpaceEnum.additional
                ),
                TTest(grouping_role=TreatmentRole(), space=SpaceEnum.additional),
                KSTest(grouping_role=TreatmentRole(), space=SpaceEnum.additional),
                Chi2Test(grouping_role=TreatmentRole(), space=SpaceEnum.additional),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)

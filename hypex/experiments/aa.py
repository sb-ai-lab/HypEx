from hypex.analyzers.aa import OneAAStatAnalyzer
from hypex.comparators.comparators import GroupDifference, GroupSizes
from hypex.comparators.hypothesis_testing import TTest, KSTest
from hypex.dataset.roles import TargetRole, TreatmentRole
from hypex.experiments.base import Experiment, OnRoleExperiment
from hypex.splitters.aa import AASplitter
from hypex.utils.enums import SpaceEnum

AA_TEST = Experiment(
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
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)

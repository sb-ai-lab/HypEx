from hypex.analyzer.aa import OneAASplitAnalyzer
from hypex.comparators.comparators import GroupDifference, GroupSizes
from hypex.comparators.hypothesis_testing import TTest, KSTest
from hypex.dataset.roles import TargetRole, TreatmentRole
from hypex.experiment.experiment import Experiment, OnRoleExperiment
from hypex.utils.enums import SpaceEnum

HOMOGENEITY_TEST = Experiment(
    executors=[
        OnRoleExperiment(
            executors=[
                GroupSizes(grouping_role=TreatmentRole(), space=SpaceEnum.additional),
                GroupDifference(
                    grouping_role=TreatmentRole(), space=SpaceEnum.additional
                ),
                TTest(grouping_role=TreatmentRole(), space=SpaceEnum.additional),
                KSTest(grouping_role=TreatmentRole(), space=SpaceEnum.additional),
            ],
            role=TargetRole(),
        ),
        OneAASplitAnalyzer(),
    ]
)

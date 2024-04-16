from hypex.analyzers.ab import ABAnalyzer
from hypex.comparators.comparators import GroupDifference, GroupSizes, ATE
from hypex.comparators.hypothesis_testing import TTest, UTest
from hypex.dataset.roles import TargetRole, TreatmentRole
from hypex.experiments.base import Experiment, OnRoleExperiment

AB_TEST = Experiment(
    executors=[
        GroupSizes(grouping_role=TreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(grouping_role=TreatmentRole()),
                ATE(
                    grouping_role=TreatmentRole(),
                ),
                TTest(grouping_role=TreatmentRole()),
                UTest(grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
        ABAnalyzer(),
    ]
)

from hypex.comparators.comparators import GroupDifference, GroupSizes
from hypex.comparators.hypothesis_testing import TTest, MannWhitney
from hypex.dataset.roles import TargetRole, TreatmentRole
from hypex.experiments.base import Experiment, OnRoleExperiment

AB_TEST = Experiment(
    executors=[
        GroupSizes(grouping_role=TreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(grouping_role=TreatmentRole()),
                TTest(grouping_role=TreatmentRole()),
                MannWhitney(grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
    ]
)

from hypex.comparators.comparators import GroupDifference, GroupSizes
from hypex.comparators.hypothesis_testing import TTest, MannWhitney
from hypex.dataset.roles import TargetRole, TreatmentRole
from hypex.experiment.experiment import Experiment, OnRoleExperiment

AB_TEST = Experiment(
    executors=[
        OnRoleExperiment(
            executors=[
                GroupSizes(grouping_role=TreatmentRole()),
                GroupDifference(grouping_role=TreatmentRole()),
                TTest(grouping_role=TreatmentRole()),
                MannWhitney(grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
    ]
)

from hypex.analyzers.aa import OneAAStatAnalyzer
from hypex.comparators.comparators import GroupDifference
from hypex.comparators.hypothesis_testing import TTest, KSTest
from hypex.dataset.roles import TargetRole, TreatmentRole
from hypex.experiments.base import Experiment, OnRoleExperiment

HOMOGENEITY_TEST = Experiment(
    executors=[
        OnRoleExperiment(
            executors=[
                GroupDifference(grouping_role=TreatmentRole()),
                TTest(grouping_role=TreatmentRole()),
                KSTest(grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)

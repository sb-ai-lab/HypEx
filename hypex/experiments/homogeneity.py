from hypex.analyzers import OneAASplitAnalyzer
from hypex.comparators import GroupDifference, GroupSizes
from hypex.analyzers import OneAAStatAnalyzer
from hypex.comparators import GroupDifference
from hypex.comparators import TTest, KSTest
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments import Experiment, OnRoleExperiment

HOMOGENEITY_TEST = Experiment(
    executors=[
        OnRoleExperiment(
            executors=[
                GroupSizes(grouping_role=TreatmentRole()),
                GroupDifference(grouping_role=TreatmentRole()),
                TTest(grouping_role=TreatmentRole()),
                KSTest(grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)

from hypex.experiment.experiment import Experiment, OnRoleExperiment
from hypex.splitters.aa_splitter import AASplitter
from hypex.comparators.hypothesis_testing import TTest, KSTest
from hypex.comparators.comparators import GroupDifference, GroupSizes
from hypex.dataset.roles import TargetRole, TreatmentRole
from hypex.analyzer.aa import OneAASplitAnalyzer

AA_TEST = Experiment(
    executors=[
        AASplitter(),
        OnRoleExperiment(
            executors=[
                GroupDifference(grouping_role=TreatmentRole(), space="additional"),
                TTest(grouping_role=TreatmentRole(), space="additional"),
                KSTest(grouping_role=TreatmentRole(), space="additional"),
            ],
            role=TargetRole(),
        ),
        OneAASplitAnalyzer()
    ]
)

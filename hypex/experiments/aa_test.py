from hypex.experiment.experiment import Experiment, OnRoleExperiment
from hypex.splitters.aa_splitter import AASplitter
from hypex.comparators.hypothesis_testing import TTest, KSTest
from hypex.comparators.comparators import GroupDifference, GroupSizes
from hypex.dataset.roles import TargetRole
from hypex.analyzers.aa import OneAASplitAnalyzer

AA_TEST = Experiment(
    executors=[
        AASplitter(),
        OnRoleExperiment(
            executors=[
                GroupDifference(),
                TTest(),
                KSTest(),
            ],
            role=TargetRole(),
        ),
        OneAASplitAnalyzer()
    ]
)
from hypex.analyzers import OneAAStatAnalyzer
from hypex.comparators import GroupDifference, GroupSizes
from hypex.comparators.abstract import GroupComparator
from hypex.comparators.hypothesis_testing import TTest, KSTest
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments import Experiment, OnRoleExperiment
from hypex.experiments.base_complex import ParamsExperiment
from hypex.reporters import DatasetReporter, AADictReporter
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
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)

AA_TEST = Experiment(
    executors=[
        ParamsExperiment(
            executors=[
                AASplitter(),
                OnRoleExperiment(
                    executors=[
                        GroupSizes(),
                        GroupDifference(),
                        TTest(),
                        KSTest(),
                    ],
                    role=TargetRole(),
                ),
                OneAAStatAnalyzer(),
            ],
            reporter=DatasetReporter(AADictReporter(front=False)),
            params={
                AASplitter: {"random_state": range(2000)},
                GroupComparator: {
                    "grouping_role": [TreatmentRole()],
                    "space": [SpaceEnum.additional],
                },
            },
        )
    ]
)

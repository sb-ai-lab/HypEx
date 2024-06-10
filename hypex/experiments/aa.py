from hypex.analyzers import OneAAStatAnalyzer
from hypex.comparators import GroupDifference, GroupSizes
from hypex.comparators.hypothesis_testing import TTest, KSTest
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments import Experiment, OnRoleExperiment
from hypex.splitters import AASplitter
from hypex.utils import SpaceEnum
from hypex.transformers.filters import CVFilter, ConstFilter, NanFilter, CorrFilter, OutliersFilter

AA_TEST = Experiment(
    executors=[
        CVFilter(lower_bound=0.5),
        # ConstFilter(),
        # NanFilter(threshold=0.04),
        # CorrFilter(),
        # OutliersFilter(lower_percentile=0.05, upper_percentile=0.95),
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

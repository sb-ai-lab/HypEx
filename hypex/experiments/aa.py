from hypex.analyzers import OneAAStatAnalyzer
from hypex.comparators import GroupDifference, GroupSizes
from hypex.comparators.hypothesis_testing import TTest, KSTest
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments import Experiment, OnRoleExperiment
from hypex.splitters import AASplitter
from hypex.utils import SpaceEnum
from hypex.transformers.filters import CVFilter, ConstFilter, NanFilter, CorrFilter, OutliersFilter
from hypex.transformers.category_agg import CategoryAggregator

AA_TEST = Experiment(
    executors=[
        # CVFilter(),
        # ConstFilter(threshold=0.4),
        # NanFilter(threshold=0.4),
        # CorrFilter(),
        # OutliersFilter(lower_percentile=0.05, upper_percentile=0.95),
        CategoryAggregator(new_group_name="n/a"),
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

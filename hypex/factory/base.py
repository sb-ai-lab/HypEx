import sys

from hypex.analyzers import ABAnalyzer, OneAASplitAnalyzer
from hypex.comparators import GroupDifference, GroupSizes, ATE, TTest, KSTest, UTest
from hypex.dataset import (
    ExperimentData,
    Arg1Role,
    Arg2Role,
    InfoRole,
    TargetRole,
    FeatureRole,
    GroupingRole,
    PreTargetRole,
    StatisticRole,
    StratificationRole,
    TreatmentRole,
    TmpTreatmentRole,
    TempGroupingRole,
    TempTargetRole,
)
from hypex.describers import Unique
from hypex.experiments import (
    Experiment,
    OnRoleExperiment,
    GroupExperiment,
    CycledExperiment,
)
from hypex.operators import (
    MetricRatio,
    MetricLogRatio,
    MetricPercentageRatio,
    MetricDelta,
    MetricRelativeDelta,
    MetricAbsoluteDelta,
    MetricPercentageDelta,
)
from hypex.reporters import AADictReporter
from hypex.splitters import (
    AASplitter,
    AASplitterWithGrouping,
    AASplitterWithStratification,
)
from hypex.stats import Min, Max, Mode, Mean, Median, Size, Std, Variance
from hypex.transformers import Shuffle
from hypex.utils import ExperimentDataEnum, SpaceEnum

all_classes = [
    ABAnalyzer,
    OneAASplitAnalyzer,
    GroupDifference,
    GroupSizes,
    ATE,
    TTest,
    KSTest,
    UTest,
    Arg1Role,
    Arg2Role,
    InfoRole,
    TargetRole,
    FeatureRole,
    GroupingRole,
    PreTargetRole,
    StatisticRole,
    StratificationRole,
    TreatmentRole,
    TmpTreatmentRole,
    TempGroupingRole,
    TempTargetRole,
    Unique,
    OnRoleExperiment,
    GroupExperiment,
    CycledExperiment,
    MetricRatio,
    MetricLogRatio,
    MetricPercentageRatio,
    MetricDelta,
    MetricRelativeDelta,
    MetricAbsoluteDelta,
    MetricPercentageDelta,
    AADictReporter,
    AASplitter,
    AASplitterWithGrouping,
    AASplitterWithStratification,
    Min,
    Max,
    Mode,
    Mean,
    Median,
    Size,
    Std,
    Variance,
    Shuffle,
    ExperimentDataEnum,
    SpaceEnum,
]


class Factory:
    def __init__(self, hypothesis):
        self.hypothesis = hypothesis

    def make_experiment(self, experiment):
        executors = []
        for key, items in experiment.items():
            class_ = getattr(sys.modules[__name__], key)
            if "executors" in items or "inner_executors" in items:
                item = "executors" if "executors" in items else "inner_executors"
                items[f"{item}"] = self.make_experiment(experiment[key][f"{item}"])
            if "role" in items or "grouping_role" in items:
                item = "role" if "role" in items else "grouping_role"
                items[f"{item}"] = getattr(
                    sys.modules[__name__], items[item] + "Role"
                )()
            items = {i: None if j == "None" else j for i, j in items.items()}
            executors.append(class_(**items))
        return executors

    def execute(self):
        experiment_data = ExperimentData(self.hypothesis.dataset)
        experiment = Experiment(
            executors=self.make_experiment(self.hypothesis.experiment)
        )
        return experiment_data, experiment

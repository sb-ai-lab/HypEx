# import sys
#
# from ..analyzers import ABAnalyzer, OneAAStatAnalyzer
# from ..comparators import GroupDifference, GroupSizes, ATE, TTest, KSTest, UTest
# from ..dataset import (
#     ExperimentData,
#     Arg1Role,
#     Arg2Role,
#     InfoRole,
#     TargetRole,
#     FeatureRole,
#     GroupingRole,
#     PreTargetRole,
#     StatisticRole,
#     StratificationRole,
#     TreatmentRole,
#     TempTreatmentRole,
#     TempGroupingRole,
#     TempTargetRole,
# )
# from ..experiments import (
#     Experiment,
#     OnRoleExperiment,
#     GroupExperiment,
#     CycledExperiment,
# )
# from ..reporters import OneAADictReporter
# from ..transformers import Shuffle
# from ..utils import ExperimentDataEnum, SpaceEnum
#
# all_classes = [
#     ABAnalyzer,
#     OneAAStatAnalyzer,
#     GroupDifference,
#     GroupSizes,
#     ATE,
#     TTest,
#     KSTest,
#     UTest,
#     Arg1Role,
#     Arg2Role,
#     InfoRole,
#     TargetRole,
#     FeatureRole,
#     GroupingRole,
#     PreTargetRole,
#     StatisticRole,
#     StratificationRole,
#     TreatmentRole,
#     TempTreatmentRole,
#     TempGroupingRole,
#     TempTargetRole,
#     OnRoleExperiment,
#     GroupExperiment,
#     CycledExperiment,
#     OneAADictReporter,
#     Shuffle,
#     ExperimentDataEnum,
#     SpaceEnum,
# ]
#
# spaces = {
#     "additional": SpaceEnum.additional,
#     "auto": SpaceEnum.auto,
#     "data": SpaceEnum.data,
# }
#
#
# class Factory:
#     def __init__(self, hypothesis):
#         self.hypothesis = hypothesis
#
#     def make_experiment(self, experiment):
#         executors = []
#         for key, items in experiment.items():
#             class_ = getattr(sys.modules[__name__], key)
#             if "executors" in items or "inner_executors" in items:
#                 item = "executors" if "executors" in items else "inner_executors"
#                 items[f"{item}"] = self.make_experiment(experiment[key][f"{item}"][0])
#             if "role" in items or "grouping_role" in items:
#                 item = "role" if "role" in items else "grouping_role"
#                 items[f"{item}"] = getattr(
#                     sys.modules[__name__], items[item] + "Role"
#                 )()
#             if "space" in items:
#                 items["space"] = spaces.get(items["space"])
#             items = {i: None if j == "None" else j for i, j in items.items()}
#             executors.append(class_(**items))
#         return executors
#
#     def execute(self):
#         experiment_data = ExperimentData(self.hypothesis.dataset)
#         experiment = Experiment(
#             executors=self.make_experiment(self.hypothesis.experiment)
#         )
#         return experiment_data, experiment

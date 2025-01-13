from typing import Optional, Literal, Union, List

from .analyzers.ab import ABAnalyzer
from .comparators import GroupDifference, GroupSizes, TTest, UTest, Chi2Test
from .dataset import TargetRole, TreatmentRole
from .experiments.base import Experiment, OnRoleExperiment
from .ui.ab import ABOutput
from .ui.base import ExperimentShell
from .utils import ABNTestMethodsEnum


class ABTest(ExperimentShell):
    def create_experiment(self, **kwargs) -> Experiment:
        test_mapping = {
            "t-test": TTest(compare_by="groups", grouping_role=TreatmentRole()),
            "u-test": UTest(compare_by="groups", grouping_role=TreatmentRole()),
            "chi2-test": Chi2Test(compare_by="groups", grouping_role=TreatmentRole()),
        }
        on_role_executors = [GroupDifference(grouping_role=TreatmentRole())]
        additional_tests = kwargs.get("additional_tests", ["t-test"])
        additional_tests = (
            additional_tests
            if isinstance(additional_tests, List)
            else [additional_tests]
        )
        for i in additional_tests:
            on_role_executors += [test_mapping[i]]
        return Experiment(
            executors=[
                GroupSizes(grouping_role=TreatmentRole()),
                OnRoleExperiment(
                    executors=on_role_executors,
                    role=TargetRole(),
                ),
                ABAnalyzer(
                    multitest_method=(
                        ABNTestMethodsEnum(kwargs["multitest_method"])
                        if kwargs.get("multitest_method")
                        else None
                    )
                ),
            ]
        )


    def __init__(
        self,
        additional_tests: Union[
            Literal["t-test", "u-test", "chi2-test"],
            List[Literal["t-test", "u-test", "chi2-test"]],
            None,
        ] = None,
        multitest_method: Optional[
            Literal[
                "bonferroni",
                "sidak",
                "holm-sidak",
                "holm",
                "simes-hochberg",
                "hommel",
                "fdr_bh",
                "fdr_by",
                "fdr_tsbh",
                "fdr_tsbhy",
                "quantile",
            ]
        ] = "holm",
    ):

        super().__init__(
            output=ABOutput(),
            create_experiment_kwargs={
                "additional_tests": additional_tests,
                "multitest_method": multitest_method
            }
        )

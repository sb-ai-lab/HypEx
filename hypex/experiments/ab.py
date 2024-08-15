from typing import Optional, Literal, Union, List

from hypex.analyzers.ab import ABAnalyzer
from hypex.comparators import GroupDifference, GroupSizes, TTest, UTest, Chi2Test
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments.base import Experiment, OnRoleExperiment
from hypex.ui.ab import ABOutput
from hypex.ui.base import ExperimentShell
from hypex.utils import ABNTestMethodsEnum


class ABTest(ExperimentShell):

    @staticmethod
    def _make_experiment(additional_tests, multitest_method):
        test_mapping = {
            "t-test": TTest(compare_by="groups", grouping_role=TreatmentRole()),
            "u-test": UTest(compare_by="groups", grouping_role=TreatmentRole()),
            "chi2-test": Chi2Test(compare_by="groups", grouping_role=TreatmentRole()),
        }
        on_role_executors = [GroupDifference(grouping_role=TreatmentRole())]
        additional_tests = ["t-test"] if additional_tests is None else additional_tests
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
                        ABNTestMethodsEnum(multitest_method)
                        if multitest_method
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
        ] = None,
    ):
        super().__init__(
            experiment=self._make_experiment(additional_tests, multitest_method),
            output=ABOutput(),
        )

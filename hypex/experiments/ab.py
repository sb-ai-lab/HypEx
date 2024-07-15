from typing import Optional, Literal, Union, List

from hypex.analyzers.ab import ABAnalyzer
from hypex.comparators import GroupDifference, GroupSizes, TTest, UTest, Chi2Test
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments import Experiment, OnRoleExperiment
from hypex.ui.ab import ABOutput
from hypex.ui.base import ExperimentShell
from hypex.utils import ABNTestMethodsEnum

AB_TEST_T = Experiment(
    executors=[
        GroupSizes(grouping_role=TreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(grouping_role=TreatmentRole()),
                TTest(grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
        ABAnalyzer(multitest_method=ABNTestMethodsEnum.quantile),
    ]
)

AB_TEST_TU = Experiment(
    executors=[
        GroupSizes(grouping_role=TreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(grouping_role=TreatmentRole()),
                TTest(grouping_role=TreatmentRole()),
                UTest(grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
        ABAnalyzer(multitest_method=ABNTestMethodsEnum.quantile),
    ]
)

AB_TEST_TC = Experiment(
    executors=[
        GroupSizes(grouping_role=TreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(grouping_role=TreatmentRole()),
                TTest(grouping_role=TreatmentRole()),
                Chi2Test(grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
        ABAnalyzer(multitest_method=ABNTestMethodsEnum.quantile),
    ]
)

AB_TEST_TUC = Experiment(
    executors=[
        GroupSizes(grouping_role=TreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(grouping_role=TreatmentRole()),
                TTest(grouping_role=TreatmentRole()),
                UTest(grouping_role=TreatmentRole()),
                Chi2Test(grouping_role=TreatmentRole()),
            ],
            role=TargetRole(),
        ),
        ABAnalyzer(multitest_method=ABNTestMethodsEnum.quantile),
    ]
)


class ABTest(ExperimentShell):

    @staticmethod
    def _set_experiment(additional_tests):
        if additional_tests:
            if len(additional_tests) == 2:
                return AB_TEST_TUC
            elif "u-test" in additional_tests:
                return AB_TEST_TU
            else:
                return AB_TEST_TC
        return AB_TEST_T

    def __init__(
        self,
        additional_tests: Union[
            Literal["u-test", "chi2-test"],
            List[Literal["u-test", "chi2-test"]],
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
        additional_tests = (
            additional_tests
            if isinstance(additional_tests, List)
            else [additional_tests]
        )
        experiment = self._set_experiment(additional_tests)
        experiment.executors[2].multitest_method = (
            ABNTestMethodsEnum(multitest_method) if multitest_method else None
        )
        super().__init__(
            experiment=experiment,
            output=ABOutput(),
        )

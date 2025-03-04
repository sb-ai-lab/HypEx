import warnings
from typing import List, Literal, Union

from .experiments import GroupExperiment
from .reporters.matching import MatchingDatasetReporter
from .analyzers.matching import MatchingAnalyzer
from .comparators import TTest, PSI, KSTest
from .comparators.distances import MahalanobisDistance
from .dataset import TreatmentRole, TargetRole, AdditionalTargetRole
from .executor import Executor
from .experiments.base import Experiment, OnRoleExperiment
from .ml.faiss import FaissNearestNeighbors
from .operators.operators import MatchingMetrics, Bias
from .ui.base import ExperimentShell
from .ui.matching import MatchingOutput


class Matching(ExperimentShell):
    """A class for performing matching analysis with configurable distance metrics and quality tests.

    This class provides functionality to identify groups that are similar to each other
    using various distance metrics and quality assessment methods.

    Args:
        group_match (bool, optional): Whether to perform group matching. Defaults to False.
        distance (Literal["mahalanobis", "l2"], optional): Distance metric to use for matching.
            Options are "mahalanobis" or "l2". Defaults to "mahalanobis".
        metric (Literal["atc", "att", "ate"], optional): Type of treatment effect to estimate.
            "atc" = average treatment effect on controls
            "att" = average treatment effect on treated
            "ate" = average treatment effect
            Defaults to "ate".
        bias_estimation (bool, optional): Whether to estimate bias. Defaults to True.
        quality_tests (Union[str, List[str]], optional): Quality tests to perform.
            Options are "smd", "psi", "ks-test", "repeats", "t-test", or "auto".
            Can be a single test or list of tests. Defaults to "auto".

    Examples:
        Basic matching with default settings:
        >>> matching = Matching()
        >>> results = matching.execute(data)

        Matching with L2 distance and specific quality tests:
        >>> matching = Matching(
        ...     distance="l2",
        ...     quality_tests=["t-test", "ks-test"]
        ... )
        >>> results = matching.execute(data)

        Group matching with ATT estimation:
        >>> matching = Matching(
        ...     group_match=True,
        ...     metric="att",
        ...     bias_estimation=True
        ... )
        >>> results = matching.execute(data)
    """

    @staticmethod
    def _make_experiment(
        group_match: bool = False,
        distance: Literal["mahalanobis", "l2"] = "mahalanobis",
        metric: Literal["atc", "att", "ate"] = "ate",
        bias_estimation: bool = True,
        quality_tests: Union[
            Literal["smd", "psi", "ks-test", "repeats", "t-test", "auto"],
            List[Literal["smd", "psi", "ks-test", "repeats", "t-test", "auto"]],
        ] = "auto",
    ) -> Experiment:
        """Creates an experiment configuration with specified matching parameters.

        Args:
            group_match (bool, optional): Whether to perform group matching. Defaults to False.
        distance (Literal["mahalanobis", "l2"], optional): Distance metric to use for matching.
            Options are "mahalanobis" or "l2". Defaults to "mahalanobis".
        metric (Literal["atc", "att", "ate"], optional): Type of treatment effect to estimate.
            "atc" = average treatment effect on controls
            "att" = average treatment effect on treated
            "ate" = average treatment effect
            Defaults to "ate".
        bias_estimation (bool, optional): Whether to estimate bias. Defaults to True.
        quality_tests (Union[str, List[str]], optional): Quality tests to perform.
            Options are "ks-test", "t-test", or "auto".
            Can be a single test or list of tests. Defaults to "auto", which performs all tests.

        Returns:
            Experiment: Configured experiment object with specified matching parameters.

        Examples:
            >>> exp = Matching._make_experiment(
            ...     distance="l2",
            ...     metric="att",
            ...     quality_tests=["t-test"]
            ... )
        """
        distance_mapping = {
            "mahalanobis": MahalanobisDistance(grouping_role=TreatmentRole())
        }
        test_mapping = {
            "t-test": TTest(compare_by="groups", grouping_role=TreatmentRole()),
            # "psi": PSI(grouping_role=TreatmentRole(), compare_by="groups"),
            "ks-test": KSTest(grouping_role=TreatmentRole(), compare_by="groups"),
        }
        two_sides = metric == "ate"
        test_pairs = metric == "atc"
        executors: List[Executor] = [
            FaissNearestNeighbors(
                grouping_role=TreatmentRole(),
                two_sides=two_sides,
                test_pairs=test_pairs,
            )
        ]
        if bias_estimation:
            executors += [
                Bias(grouping_role=TreatmentRole(), target_roles=[TargetRole()]),
            ]
        executors += [
            MatchingMetrics(
                grouping_role=TreatmentRole(),
                target_roles=[TargetRole()],
                metric=metric,
            ),
            MatchingAnalyzer(),
        ]
        if quality_tests != "auto":
            # warnings.warn("Now quality tests aren't supported yet")
            executors += [
                OnRoleExperiment(
                    executors=[test_mapping[test] for test in quality_tests],
                    role=TargetRole(),
                )
            ]
        return (
            Experiment(
                executors=(
                    executors
                    if distance == "l2"
                    else [distance_mapping[distance]] + executors
                )
            )
            if not group_match
            else GroupExperiment(
                executors=(
                    executors
                    if distance == "l2"
                    else [distance_mapping[distance]] + executors
                ),
                reporter=MatchingDatasetReporter(),
            )
        )

    def __init__(
        self,
        group_match: bool = False,
        distance: Literal["mahalanobis", "l2"] = "mahalanobis",
        metric: Literal["atc", "att", "ate"] = "ate",
        bias_estimation: bool = True,
        quality_tests: Union[
            Literal["smd", "psi", "ks-test", "repeats", "t-test", "auto"],
            List[Literal["smd", "psi", "ks-test", "repeats", "t-test", "auto"]],
        ] = "auto",
    ):
        super().__init__(
            experiment=self._make_experiment(
                group_match, distance, metric, bias_estimation, quality_tests
            ),
            output=MatchingOutput(GroupExperiment if group_match else MatchingAnalyzer),
        )

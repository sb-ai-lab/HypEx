from __future__ import annotations

from typing import Literal

from .analyzers.matching import MatchingAnalyzer
from .comparators import Chi2Test, KSTest, TTest
from .comparators.distances import MahalanobisDistance
from .dataset import AdditionalMatchingRole, FeatureRole, TargetRole, TreatmentRole
from .encoders.encoders import DummyEncoder
from .executor import Executor
from .experiments import GroupExperiment
from .experiments.base import Experiment, OnRoleExperiment
from .ml.faiss import FaissNearestNeighbors
from .operators.operators import Bias, MatchingMetrics
from .reporters.matching import MatchingDatasetReporter
from .transformers import TypeCaster
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
        faiss_mode (Literal["base", "fast", "auto"], optional): Faiss mode to use for matching.
            Options are "base", "fast", or "auto". Defaults to "auto".
        n_neighbors (int, optional): Number of neighbors to use for matching. Defaults to 1.

    Examples
    --------
    .. code-block:: python

        # Basic matching with default settings
        matching = Matching()
        results = matching.execute(data)

        # Matching with L2 distance and specific quality tests
        matching = Matching(
            distance="l2",
            quality_tests=["t-test", "ks-test"]
        )
        results = matching.execute(data)

        # Group matching with ATT estimation
        matching = Matching(
            group_match=True,
            metric="att",
            bias_estimation=True
        )
        results = matching.execute(data)
    """

    @staticmethod
    def _make_experiment(
        group_match: bool = False,
        distance: Literal["mahalanobis", "l2"] = "mahalanobis",
        metric: Literal["atc", "att", "ate"] = "ate",
        bias_estimation: bool = True,
        quality_tests: (
            Literal["smd", "psi", "ks-test", "repeats", "t-test", "chi2-test", "auto"]
            | list[
                Literal[
                    "smd", "psi", "ks-test", "repeats", "t-test", "chi2-test", "auto"
                ]
            ]
        ) = "auto",
        faiss_mode: Literal["base", "fast", "auto"] = "auto",
        n_neighbors: int = 1,
        weights: dict[str, float] | None = None,
        encode_categories: bool = True,
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
        faiss_mode (Literal["base", "fast", "auto"], optional): Faiss mode to use for matching.
            Options are "base", "fast", or "auto". Defaults to "auto".
        n_neighbors (int, optional): Number of neighbors to use for matching. Defaults to 1.

        Returns:
            Experiment: Configured experiment object with specified matching parameters.

        Examples
        --------
        .. code-block:: python

            exp = Matching._make_experiment(
                distance="l2",
                metric="att",
                quality_tests=["t-test"]
            )
        """
        distance_mapping = {
            "mahalanobis": MahalanobisDistance(
                grouping_role=TreatmentRole(), weights=weights
            ),
            # "l2": L2Distance(grouping_role=TreatmentRole(), weights=weights),
        }
        test_mapping = {
            "t-test": TTest(
                compare_by="matched_pairs",
                grouping_role=TreatmentRole(),
                baseline_role=AdditionalMatchingRole(),
            ),
            # "psi": PSI(grouping_role=TreatmentRole(), compare_by="groups"),
            "ks-test": KSTest(
                grouping_role=TreatmentRole(),
                compare_by="matched_pairs",
                baseline_role=AdditionalMatchingRole(),
            ),
            "chi2-test": Chi2Test(
                grouping_role=TreatmentRole(),
                compare_by="matched_pairs",
                baseline_role=AdditionalMatchingRole(),
            ),
        }
        two_sides = metric == "ate"
        test_pairs = metric == "atc"
        executors: list[Executor] = [
            TypeCaster(
                dtype={int: float},
                roles=[FeatureRole(), TargetRole()],
            ),
            FaissNearestNeighbors(
                grouping_role=TreatmentRole(),
                two_sides=two_sides,
                test_pairs=test_pairs,
                faiss_mode=faiss_mode,
                n_neighbors=n_neighbors,
            ),
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
                n_neighbors=n_neighbors,
            ),
            MatchingAnalyzer(),
        ]
        if quality_tests == "auto":
            executors += [
                OnRoleExperiment(
                    executors=list(test_mapping.values()),
                    role=FeatureRole(),
                )
            ]
        else:
            # warnings.warn("Now quality tests aren't supported yet")
            executors += [
                OnRoleExperiment(
                    executors=[test_mapping[test] for test in quality_tests],
                    role=FeatureRole(),
                )
            ]
        executors = (
            executors if distance == "l2" else [distance_mapping[distance], *executors]
        )
        executors = executors if not encode_categories else [DummyEncoder(), *executors]
        return (
            Experiment(executors=executors)
            if not group_match
            else GroupExperiment(
                executors=executors,
                reporter=MatchingDatasetReporter(),
            )
        )

    def __init__(
        self,
        group_match: bool = False,
        distance: Literal["mahalanobis", "l2"] = "mahalanobis",
        # metric: Literal["atc", "att", "ate"] = "ate",
        bias_estimation: bool = True,
        quality_tests: (
            Literal["smd", "psi", "ks-test", "repeats", "t-test", "chi2-test", "auto"]
            | list[
                Literal[
                    "smd", "psi", "ks-test", "repeats", "t-test", "chi2-test", "auto"
                ]
            ]
        ) = "auto",
        faiss_mode: Literal["base", "fast", "auto"] = "auto",
        n_neighbors: int = 1,
        weights: dict[str, float] | None = None,
        encode_categories: bool = True,
    ):
        metric = "ate"
        super().__init__(
            experiment=self._make_experiment(
                group_match,
                distance,
                metric,
                bias_estimation,
                quality_tests,
                faiss_mode,
                n_neighbors,
                weights,
                encode_categories,
            ),
            output=MatchingOutput(GroupExperiment if group_match else MatchingAnalyzer),
        )

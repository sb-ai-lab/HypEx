from __future__ import annotations

from ..analyzers.ab import ABAnalyzer
from ..comparators import GroupDifference, GroupSizes
from ..dataset import Dataset, ExperimentData, InfoRole, StatisticRole, TreatmentRole
from ..reporters.ab import ABDatasetReporter
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum
from .base import Output


class CupacOutput:
    """Container for CUPAC-specific outputs.

    Attributes:
        variance_reductions (Dataset | None): Variance reduction metrics from CUPAC models.
        feature_importances (Dataset | None): Feature importance scores from CUPAC models.
    """

    def __init__(self):
        self.variance_reductions: Dataset | None = None
        self.feature_importances: Dataset | None = None

    def __repr__(self) -> str:
        has_vr = self.variance_reductions is not None
        has_fi = self.feature_importances is not None

        if not has_vr and not has_fi:
            return "CupacOutput(no CUPAC data available)"

        parts = []
        if has_vr:
            n_targets = len(self.variance_reductions.data)
            parts.append(f"variance_reductions: {n_targets} target(s)")
        if has_fi:
            n_features = len(self.feature_importances.data)
            parts.append(f"feature_importances: {n_features} feature(s)")

        return f"CupacOutput({', '.join(parts)})"


class ABOutput(Output):
    multitest: Dataset | str
    sizes: Dataset
    cupac: CupacOutput

    def __init__(self):
        self._groups = []
        self.cupac = CupacOutput()
        super().__init__(resume_reporter=ABDatasetReporter())

    def _extract_multitest_result(self, experiment_data: ExperimentData):
        multitest_id = experiment_data.get_one_id(
            ABAnalyzer, ExperimentDataEnum.analysis_tables
        )
        if multitest_id and "MultiTest" in multitest_id:
            self.multitest = experiment_data.analysis_tables[multitest_id]
        else:
            self.multitest = (
                "There was less than three groups or multitest method wasn't provided"
            )

    def _extract_differences(self, experiment_data: ExperimentData):
        targets = []
        groups = []
        ids = experiment_data.get_ids(
            GroupDifference,
            searched_space=ExperimentDataEnum.analysis_tables,
        )["GroupDifference"]["analysis_tables"]
        self._groups = list(
            experiment_data.groups[
                experiment_data.ds.search_columns(TreatmentRole())[0]
            ].keys()
        )[1:]
        for i in self._groups:
            groups += [i] * len(ids)
        diff = Dataset.create_empty()
        for i in range(len(ids)):
            diff = diff.append(experiment_data.analysis_tables[ids[i]])
            targets += [ids[i].split(ID_SPLIT_SYMBOL)[-1]]
        return diff.add_column(groups, role={"group": StatisticRole()}).add_column(
            targets * len(self._groups), role={"feature": StatisticRole()}
        )

    def _extract_sizes(self, experiment_data: ExperimentData):
        ids = experiment_data.get_ids(
            GroupSizes,
            searched_space=ExperimentDataEnum.analysis_tables,
        )["GroupSizes"]["analysis_tables"]
        self.sizes = experiment_data.analysis_tables[ids[0]].add_column(
            self._groups, role={"group": StatisticRole()}
        )

    def _extract_variance_reductions(self, experiment_data: ExperimentData):
        """Extract variance reduction data from analysis_tables."""
        # Find all CUPAC report keys in analysis_tables
        cupac_report_keys = [
            key
            for key in experiment_data.analysis_tables.keys()
            if key.endswith("_cupac_report")
        ]

        if not cupac_report_keys:
            self.cupac.variance_reductions = None
            return

        # Aggregate all CUPAC reports into a single dataset
        variance_data = []
        for key in cupac_report_keys:
            report = experiment_data.analysis_tables[key]
            target_name = key.replace("_cupac_report", "")

            control_mean_bias = None
            test_mean_bias = None

            resume_data = self.resume.data
            if (
                "feature" in resume_data.columns
                and target_name in resume_data["feature"].values
            ):
                original_row = resume_data[resume_data["feature"] == target_name]
                cupac_row = resume_data[
                    resume_data["feature"] == f"{target_name}_cupac"
                ]

                control_mean_bias = (
                    original_row["control mean"].iloc[0]
                    - cupac_row["control mean"].iloc[0]
                )
                test_mean_bias = (
                    original_row["test mean"].iloc[0] - cupac_row["test mean"].iloc[0]
                )

            variance_data.append(
                {
                    "target": target_name,
                    "best_model": report.get("cupac_best_model"),
                    "variance_reduction_cv": report.get("cupac_variance_reduction_cv"),
                    "variance_reduction_real": report.get(
                        "cupac_variance_reduction_real"
                    ),
                    "control_mean_bias": control_mean_bias,
                    "test_mean_bias": test_mean_bias,
                }
            )

        self.cupac.variance_reductions = Dataset.from_dict(
            data=variance_data,
            roles={
                "target": InfoRole(str),
                "best_model": InfoRole(str),
                "variance_reduction_cv": StatisticRole(),
                "variance_reduction_real": StatisticRole(),
                "control_mean_bias": StatisticRole(),
                "test_mean_bias": StatisticRole(),
            },
        )

    def _extract_feature_importances(self, experiment_data: ExperimentData):
        """Extract feature importances from CUPAC models."""
        # Find all CUPAC report keys in analysis_tables
        cupac_report_keys = [
            key
            for key in experiment_data.analysis_tables.keys()
            if key.endswith("_cupac_report")
        ]

        if not cupac_report_keys:
            self.cupac.feature_importances = None
            return

        # Aggregate all feature importances into a single dataset
        importance_data = []
        for key in cupac_report_keys:
            report = experiment_data.analysis_tables[key]
            target_name = key.replace("_cupac_report", "")
            model_name = report.get("cupac_best_model")
            importances = report.get("cupac_feature_importances", {})

            if not importances:
                continue

            # Convert feature importances to rows
            for feature_idx, importance_value in importances.items():
                importance_data.append(
                    {
                        "target": target_name,
                        "feature": feature_idx,
                        "importance": importance_value,
                        "model": model_name,
                    }
                )

        if not importance_data:
            self.cupac.feature_importances = None
            return

        self.cupac.feature_importances = Dataset.from_dict(
            data=importance_data,
            roles={
                "target": InfoRole(str),
                "feature": InfoRole(str),
                "importance": StatisticRole(),
                "model": InfoRole(str),
            },
        )

    @property
    def variance_reduction_report(self) -> Dataset | str:
        """Get variance reduction report for CUPED/CUPAC transformations."""
        if hasattr(self, "_experiment_data"):
            return self.resume_reporter.report_variance_reductions(
                self._experiment_data
            )
        return "No experiment data available."

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_differences(experiment_data)
        self._extract_multitest_result(experiment_data)
        self._extract_sizes(experiment_data)
        self._extract_variance_reductions(experiment_data)
        self._extract_feature_importances(experiment_data)

"""CUPAC-specific reporters for extracting variance reductions and feature importances."""

from ..comparators import GroupDifference
from ..dataset import Dataset, ExperimentData, InfoRole, StatisticRole
from ..utils import ExperimentDataEnum, ID_SPLIT_SYMBOL


class CupacReporter:
    """Reporter for extracting CUPAC analysis results from experiment data."""

    @staticmethod
    def extract_resume(experiment_data: ExperimentData) -> Dataset | None:
        """Generate summary resume for CUPAC results.
        
        Args:
            experiment_data: Experiment data containing CUPAC reports
            
        Returns:
            Dataset with target, best_model, variance_reduction_cv, variance_reduction_real columns
        """
        cupac_report_keys = [
            key
            for key in experiment_data.analysis_tables.keys()
            if key.endswith("_cupac_report")
        ]

        if not cupac_report_keys:
            return None

        resume_data = []
        for key in cupac_report_keys:
            report = experiment_data.analysis_tables[key]
            target_name = key.replace("_cupac_report", "")
            
            vr_cv = report.get("cupac_variance_reduction_cv")
            vr_real = report.get("cupac_variance_reduction_real")
            
            resume_data.append({
                "target": target_name,
                "best_model": report.get("cupac_best_model"),
                "variance_reduction_cv": f"{vr_cv:.1f}%" if vr_cv is not None else "-",
                "variance_reduction_real": f"{vr_real:.1f}%" if vr_real is not None else "-",
            })

        return Dataset.from_dict(
            data=resume_data,
            roles={
                "target": InfoRole(str),
                "best_model": InfoRole(str),
                "variance_reduction_cv": InfoRole(str),
                "variance_reduction_real": InfoRole(str),
            },
        )

    @staticmethod
    def _get_group_means(experiment_data: ExperimentData, target_name: str) -> dict[str, float | None]:
        """Extract control and test means for a target from GroupDifference comparators.
        
        Args:
            experiment_data: Experiment data containing GroupDifference results
            target_name: Name of the target variable (e.g., 'y', 'y_cupac')
            
        Returns:
            Dict with 'control_mean' and 'test_mean' keys, or None values if not found
        """
        group_diff_ids = experiment_data.get_ids(
            GroupDifference, 
            searched_space=ExperimentDataEnum.analysis_tables
        )["GroupDifference"]["analysis_tables"]
        
        for gd_id in group_diff_ids:
            # GroupDifference IDs format: "GroupDifference┴params_hash┴target_name"
            gd_target = gd_id.split(ID_SPLIT_SYMBOL)[-1]
            if gd_target == target_name:
                gd_data = experiment_data.analysis_tables[gd_id]
                # Extract scalar values from Dataset
                control_mean = gd_data.data["control mean"].iloc[0] if "control mean" in gd_data.columns else None
                test_mean = gd_data.data["test mean"].iloc[0] if "test mean" in gd_data.columns else None
                return {
                    "control_mean": control_mean,
                    "test_mean": test_mean
                }
        
        return {"control_mean": None, "test_mean": None}

    @staticmethod
    def extract_variance_reductions(
        experiment_data: ExperimentData
    ) -> Dataset | None:
        """Extract variance reduction data from CUPAC reports.
        
        Args:
            experiment_data: Experiment data containing CUPAC reports in analysis_tables
            
        Returns:
            Dataset with variance reduction metrics or None if no CUPAC data found
        """
        cupac_report_keys = [
            key
            for key in experiment_data.analysis_tables.keys()
            if key.endswith("_cupac_report")
        ]

        if not cupac_report_keys:
            return None

        variance_data = []
        for key in cupac_report_keys:
            report = experiment_data.analysis_tables[key]
            target_name = key.replace("_cupac_report", "")

            control_mean_bias = None
            test_mean_bias = None

            # Extract means for original and CUPAC-transformed targets
            original_means = CupacReporter._get_group_means(experiment_data, target_name)
            cupac_means = CupacReporter._get_group_means(experiment_data, f"{target_name}_cupac")
            
            if original_means["control_mean"] is not None and cupac_means["control_mean"] is not None:
                control_mean_bias = original_means["control_mean"] - cupac_means["control_mean"]
            
            if original_means["test_mean"] is not None and cupac_means["test_mean"] is not None:
                test_mean_bias = original_means["test_mean"] - cupac_means["test_mean"]

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

        return Dataset.from_dict(
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

    @staticmethod
    def extract_feature_importances(
        experiment_data: ExperimentData,
    ) -> Dataset | None:
        """Extract feature importances from CUPAC models.
        
        Args:
            experiment_data: Experiment data containing CUPAC reports in analysis_tables
            
        Returns:
            Dataset with feature importances or None if no data found
        """
        cupac_report_keys = [
            key
            for key in experiment_data.analysis_tables.keys()
            if key.endswith("_cupac_report")
        ]

        if not cupac_report_keys:
            return None

        importance_data = []
        for key in cupac_report_keys:
            report = experiment_data.analysis_tables[key]
            target_name = key.replace("_cupac_report", "")
            model_name = report.get("cupac_best_model")
            importances = report.get("cupac_feature_importances", {})

            if not importances:
                continue

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
            return None

        return Dataset.from_dict(
            data=importance_data,
            roles={
                "target": InfoRole(str),
                "feature": InfoRole(str),
                "importance": StatisticRole(),
                "model": InfoRole(str),
            },
        )

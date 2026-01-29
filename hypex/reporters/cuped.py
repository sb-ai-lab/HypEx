"""CUPED-specific reporter for extracting variance reduction metrics."""

from ..dataset import Dataset, ExperimentData, InfoRole, StatisticRole


class CupedReporter:
    """Reporter for extracting CUPED variance reduction results from experiment data."""

    @staticmethod
    def extract_resume(data: ExperimentData) -> Dataset | str:
        """Generate summary resume for CUPED results.
        
        Args:
            data: Experiment data containing variance reduction metrics
            
        Returns:
            Dataset with target, covariate, and variance_reduction columns
        """
        variance_cols = [
            col
            for col in data.additional_fields.columns
            if col.endswith("_variance_reduction")
        ]
        
        if not variance_cols:
            return "No CUPED data available"

        # Extract CUPED features mapping from additional_fields column names
        resume_data = []
        for col in variance_cols:
            metric_name = col.replace("_variance_reduction", "")
            reduction_value = data.additional_fields.data[col].iloc[0]
            
            # Try to find covariate name from dataset roles or use simple format
            covariate = f"{metric_name.replace('_cuped', '')}_lag"
            
            resume_data.append({
                "target": metric_name.replace("_cuped", ""),
                "covariate": covariate,
                "variance_reduction": f"{reduction_value:.1f}%"
            })

        return Dataset.from_dict(
            data=resume_data,
            roles={
                "target": InfoRole(str),
                "covariate": InfoRole(str),
                "variance_reduction": InfoRole(str),
            },
        )

    @staticmethod
    def extract_variance_reductions(data: ExperimentData) -> Dataset | str:
        """Generate variance reduction report for CUPED transformations.
        
        Args:
            data: Experiment data containing variance reduction metrics in additional_fields
            
        Returns:
            Dataset with variance reduction metrics or error message string
        """
        variance_cols = [
            col
            for col in data.additional_fields.columns
            if col.endswith("_variance_reduction")
        ]
        
        if not variance_cols:
            return "No variance reduction data available. Ensure CUPED was applied."

        # Create report data
        report_data = []
        for col in variance_cols:
            metric_name = col.replace("_variance_reduction", "")
            # Get the scalar value from the additional_fields
            reduction_value = data.additional_fields.data[col].iloc[0]
            report_data.append(
                {
                    "Transformed Metric Name": metric_name,
                    "Variance Reduction (%)": reduction_value,
                }
            )

        return Dataset.from_dict(
            data=report_data,
            roles={
                "Transformed Metric Name": InfoRole(str),
                "Variance Reduction (%)": StatisticRole(),
            },
        )

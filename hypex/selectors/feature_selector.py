"""Feature selection class."""

import logging
import warnings

import pandas as pd

logger = logging.getLogger("feature_selector")
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format="[%(asctime)s | %(name)s | %(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    level=logging.INFO,
)


class FeatureSelector:
    """Class of Feature selector. Select top features. By default, use LGM."""

    def __init__(
        self,
        outcome: str,
        treatment: str,
        feature_selection_method,
    ):
        """Initialize the FeatureSelector.

        Args:
            outcome:
                The target column
            treatment:
                The column that determines control and test groups
            feature_selection_method:
                List of names of algorithms for feature selection

        ..warnings::
            FeatureSelector does not rule out the possibility of overlooked features,
            the complex impact of features on target description, or
            the significance of features from a business logic perspective.
        """
        warnings.warn(
            "FeatureSelector does not rule out the possibility of overlooked features, "
            "the complex impact of features on target description, or "
            "the significance of features from a business logic perspective."
        )
        self.outcome = outcome
        self.treatment = treatment
        self.feature_selection_method = feature_selection_method

    def perform_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trains a model and returns feature scores.

        This method defines metrics, applies the model, creates a report, and returns feature scores

        Args:
            df:
                Input data

        Returns:
            A DataFrame containing the feature scores from the model

        """
        report_df = self.feature_selection_method(
            df=df,
            info_col_list=None,
            target=self.outcome,
            treatment_col=self.treatment,
            weights_col_list=None,
            category_col_list=None,
        )
        return report_df

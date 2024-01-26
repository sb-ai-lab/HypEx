"""Feature selection class."""
import logging
from typing import List

import pandas as pd
from hypex.selectors import selector_primal_methods

logger = logging.getLogger("feature_selector")
console_out = logging.StreamHandler()
logging.basicConfig(
    handlers=(console_out,),
    format="[%(asctime)s | %(name)s | %(levelname)s]: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    level=logging.INFO,
)


class FeatureSelector:
    """Class of Feature selector. Select top features. By default, use LGM.
    # TODO: write some feature selector"""

    def __init__(
            self,
            outcome: str,
            treatment: str,
            use_algos: str,
    ):
        """Initialize the FeatureSelector.

        Args:
            outcome:
                The target column
            treatment:
                The column that determines control and test groups
            use_algos:
                List of names of algorithms for feature selection
        """
        self.outcome = outcome
        self.treatment = treatment
        self.use_algos = use_algos

    def perform_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trains a model and returns feature scores.

        This method defines metrics, applies the model, creates a report, and returns feature scores

        Args:
            df:
                Input data

        Returns:
            A DataFrame containing the feature scores from the model

        """
        roles = {
            "target": self.outcome,
            "drop": [self.treatment],
        }

        if self.use_algos == 'lgb':
            selector = selector_primal_methods.pd_lgbm_feature_selector
        elif self.use_algos == 'catboost':
            selector = selector_primal_methods.pd_catboost_feature_selector
        elif self.use_algos == 'ridgecv':
            selector = selector_primal_methods.pd_ridgecv_feature_selector
        else:
            raise Exception(f"Unknown input algorithm used on feature_selector: {self.use_algos}")
        report_df = selector(
            df=df,
            info_col_list=None,
            target=self.outcome,
            treatment_col=self.treatment,
            weights_col_list=None,
            category_col_list=None,
            model=None
        )
        return report_df

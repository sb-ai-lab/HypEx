from typing import Dict, Any

from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu  # type: ignore

from hypex.comparators.base import StatHypothesisTestingWithScipy

import logging
from funcy import log_durations


import logging
import funcy

logger = logging.getLogger(__name__)
f_handler = logging.FileHandler(f"{__name__}.log")
f_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
logger.addHandler(f_handler)
logger.setLevel(logging.DEBUG)


class TTest(StatHypothesisTestingWithScipy):
    @funcy.log_durations(logger.debug)
    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        return ttest_ind(
            control_data.data.values.flatten(), test_data.data.values.flatten()
        )


class KSTest(StatHypothesisTestingWithScipy):
    @funcy.log_durations(logger.debug)
    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        return ks_2samp(
            control_data.data.values.flatten(), test_data.data.values.flatten()
        )


class UTest(StatHypothesisTestingWithScipy):
    def _comparison_function(self, control_data, test_data):
        return mannwhitneyu(
            control_data.data.values.flatten(), test_data.data.values.flatten()
        )

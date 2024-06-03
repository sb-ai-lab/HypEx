from typing import Dict, Any

from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu  # type: ignore

from .abstract import StatHypothesisTestingWithScipy


class TTest(StatHypothesisTestingWithScipy):
    def _inner_function(self, control_data, test_data) -> Dict[str, Any]:
        return ttest_ind(
            control_data.data.values.flatten(), test_data.data.values.flatten()
        )


class KSTest(StatHypothesisTestingWithScipy):
    def _inner_function(self, control_data, test_data) -> Dict[str, Any]:
        return ks_2samp(
            control_data.data.values.flatten(), test_data.data.values.flatten()
        )


class UTest(StatHypothesisTestingWithScipy):
    def _inner_function(self, control_data, test_data):
        return mannwhitneyu(
            control_data.data.values.flatten(), test_data.data.values.flatten()
        )

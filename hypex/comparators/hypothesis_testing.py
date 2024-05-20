from typing import Dict, Any

from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu  # type: ignore

from .abstract import StatHypothesisTesting
from ..dataset import Dataset
from ..extensions.hypothesis_testing import TTestExtension, KSTestExtension, UTestExtension, Chi2TestExtension


class TTest(StatHypothesisTesting):
    def _comparison_function(self, control_data: Dataset, test_data: Dataset) -> Dataset:
        return TTestExtension(self.reliability).calc(control_data, test_data)


class KSTest(StatHypothesisTesting):
    def _comparison_function(self, control_data, test_data) -> Dataset:
        return KSTestExtension(self.reliability).calc(control_data, test_data)


class UTest(StatHypothesisTesting):
    def _comparison_function(self, control_data, test_data) -> Dataset:
        return UTestExtension(self.reliability).calc(control_data, test_data)


class Chi2Test(StatHypothesisTesting):
    def _comparison_function(self, control_data, test_data) -> Dataset:
        return Chi2TestExtension(reliability=self.reliability).calc(control_data, test_data)

from typing import Dict, Any

from hypex.comparators.base import GroupComparator
from hypex.dataset.roles import TempTargetRole
from hypex.experiments.base import Executor
from hypex.operators.binary import MetricDelta
from hypex.stats.agg import Mean, Size

import logging
import funcy

logger = logging.getLogger(__name__)
f_handler = logging.FileHandler(f"{__name__}.log")
f_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
logger.addHandler(f_handler)
logger.setLevel(logging.DEBUG)


class GroupDifference(GroupComparator):
    default_inner_executors: Dict[str, Executor] = {
        "mean": Mean(),
    }

    @funcy.log_durations(logger.debug, label="GroupDifference")
    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        target_field = control_data.get_columns_by_roles(
            TempTargetRole(), tmp_role=True
        )[0]
        ed_control = self.inner_executors["mean"].calc(control_data)
        ed_test = self.inner_executors["mean"].calc(test_data)

        mean_a = ed_control.iloc[0]
        mean_b = ed_test.iloc[0]

        return {
            f"{target_field} control mean": mean_a,
            f"{target_field} test mean": mean_b,
            f"{target_field} difference": mean_b - mean_a,
            f"{target_field} difference %": (mean_b / mean_a - 1) * 100,
        }


class GroupSizes(GroupComparator):
    default_inner_executors: Dict[str, Executor] = {
        "size": Size(),
    }

    @funcy.log_durations(logger.debug, label="GroupComparator")
    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        size_a = self.inner_executors["size"].calc(control_data)
        size_b = self.inner_executors["size"].calc(test_data)

        return {
            "control size": size_a,
            "test size": size_b,
            "control size %": (size_a / (size_a + size_b)) * 100,
            "test size %": (size_b / (size_a + size_b)) * 100,
        }


class ATE(GroupComparator):
    default_inner_executors: Dict[str, Executor] = {
        "delta": MetricDelta(),
        "mean": Mean(),
        "size": Size(),
    }

    def _comparison_function(self, control_data, test_data) -> Dict[str, Any]:
        target_field = control_data.get_columns_by_roles(
            TempTargetRole(), tmp_role=True
        )[0]
        size_a = self.inner_executors["size"].calc(control_data)
        size_b = self.inner_executors["size"].calc(test_data)
        control_mean = self.inner_executors["mean"].calc(control_data)
        test_mean = self.inner_executors["mean"].calc(test_data)

        ate = (size_a / (size_a + size_b)) * control_mean + (
            size_b / (size_a + size_b)
        ) * test_mean

        return {f"{target_field} ATE": ate.iloc[0]}

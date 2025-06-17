from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from ..dataset import ABCRole, Dataset, ExperimentData
from ..utils import ExperimentDataEnum
from .abstract import Comparator


class MDEBySize(Comparator):

    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        significance: float = 0.05,
        power: float = 0.8,
        key: Any = "",
    ):
        super().__init__(
            compare_by="groups",
            grouping_role=grouping_role,
            key=key,
        )
        self.significance = significance
        self.power = power

    def _set_value(
        self, data: ExperimentData, value: Dataset | None = None, key: Any = None
    ) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.variables,
            self.id,
            value,
        )
        return data

    @classmethod
    def calc(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        significance: float = 0.05,
        power: float = 0.8,
        **kwargs,
    ) -> float:
        m = norm.ppf((1 + significance) / 2) + norm.ppf(power)
        if not test_data:
            raise ValueError("test_data is required")

        n_test, n_control = len(test_data), len(data)

        var_test, var_control = test_data.var(ddof=1), data.var(ddof=1)
        s = np.sqrt(var_test / n_test + var_control / n_control)

        return m * s

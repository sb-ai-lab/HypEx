"""CUPED (Controlled-experiment Using Pre-Experiment Data) transformer."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from ..dataset.dataset import Dataset, SmallDataset
from ..dataset.experiment_data import ExperimentData
from ..dataset.roles import StatisticRole, TargetRole
from ..utils.enums import ExperimentDataEnum
from .abstract import Transformer


class CUPEDTransformer(Transformer):
    """Apply CUPED variance reduction to target features.

    For each ``(target, pre_target)`` pair computes
    ``theta = Cov(Y, X) / Var(X)`` and produces an adjusted column
    ``{target}_cuped = Y - theta * (X - mean(X))``.

    Adjusted columns are added with ``TargetRole`` so downstream
    comparators pick them up automatically.  Variance-reduction
    percentages are stored in ``analysis_tables`` under the
    transformer's executor ID as a ``SmallDataset``.

    Args:
        cuped_features: Mapping ``{target_feature: pre_target_feature}``.
        key: Optional executor identifier.
    """

    def __init__(
        self,
        cuped_features: dict[str, str],
        key: Any = "",
    ) -> None:
        super().__init__(key=key)
        self.cuped_features = cuped_features

    @staticmethod
    def _inner_function(
        data: Dataset,
        cuped_features: dict[str, str],
    ) -> Dataset:
        """Compute CUPED-adjusted columns.

        Args:
            data: Input dataset containing target and pre-target columns.
            cuped_features: Mapping ``{target: pre_target}``.

        Returns:
            A copy of *data* with ``{target}_cuped`` columns appended.
        """
        result = deepcopy(data)
        for target_feature, pre_target_feature in cuped_features.items():
            mean_xy = (result[target_feature] * result[pre_target_feature]).mean()
            mean_x = result[pre_target_feature].mean()
            mean_y = result[target_feature].mean()
            cov_xy = mean_xy - mean_x * mean_y
            mean_xx = (result[pre_target_feature] * result[pre_target_feature]).mean()
            var_x = mean_xx - mean_x * mean_x

            theta = 0.0 if (var_x == 0 or np.isnan(var_x)) else cov_xy / var_x

            pre_target_mean = result[pre_target_feature].mean()
            new_values_ds = (
                result[target_feature]
                - (result[pre_target_feature] - pre_target_mean) * theta
            )
            result = result.add_column(
                data=new_values_ds,
                role={f"{target_feature}_cuped": TargetRole()},
            )
        return result

    @classmethod
    def calc(
        cls,
        data: Dataset,
        cuped_features: dict[str, str],
        **kwargs: Any,
    ) -> Dataset:
        """Stateless entry point for CUPED transformation.

        Args:
            data: Input dataset.
            cuped_features: Mapping ``{target: pre_target}``.
            **kwargs: Ignored.

        Returns:
            Dataset with CUPED-adjusted columns.
        """
        return cls._inner_function(data, cuped_features)

    def execute(self, data: ExperimentData) -> ExperimentData:
        """Run CUPED and store variance reductions in analysis_tables.

        Args:
            data: The experiment data container.

        Returns:
            Updated ``ExperimentData`` with adjusted targets in ``ds``
            and variance-reduction report in ``analysis_tables``.
        """
        new_ds = self.calc(data=data.ds, cuped_features=self.cuped_features)

        # ── Compute variance reductions ──────────────────────────────
        report_rows: list[dict[str, Any]] = []
        for target_feature in self.cuped_features:
            original_var = data.ds[target_feature].var()
            adjusted_var = new_ds[f"{target_feature}_cuped"].var()
            variance_reduction = (
                (1 - adjusted_var / original_var) * 100
                if original_var > 0
                else 0.0
            )
            report_rows.append({
                "feature": f"{target_feature}_cuped",
                "variance_reduction_pct": variance_reduction,
            })

        # ── Store in analysis_tables (principle #9) ─────────────────
        report_ds = SmallDataset.from_dict(
            report_rows,
            roles={
                "feature": StatisticRole(str),
                "variance_reduction_pct": StatisticRole(float),
            },
        )
        data = data.set_value(
            ExperimentDataEnum.analysis_tables,
            self.id,
            report_ds,
        )

        return data.copy(data=new_ds)

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import StatisticRole, TargetRole
from .abstract import Transformer


class CUPEDTransformer(Transformer):
    def __init__(
        self,
        cuped_features: dict[str, str],
        key: Any = "",
    ):
        """
        Transformer that applies the CUPED adjustment to target features.

        Args:
            cuped_features (dict[str, str]): A mapping {target_feature: pre_target_feature}.
        """
        super().__init__(key=key)
        self.cuped_features = cuped_features

    @staticmethod
    def _inner_function(
        data: Dataset,
        cuped_features: dict[str, str],
    ) -> Dataset:
        result = deepcopy(data)
        for target_feature, pre_target_feature in cuped_features.items():
            x = result[pre_target_feature]
            y = result[target_feature]

            mean_x = x.mean()
            mean_y = y.mean()

            cov_xy = ((x - mean_x) * (y - mean_y)).mean()
            var_x = ((x - mean_x) ** 2).mean()

            if var_x == 0 or var_x != var_x:
                theta = 0
            else:
                theta = cov_xy / var_x

            new_values_ds = y - (x - mean_x) * theta

            result = result.add_column(
                data=new_values_ds,
                role={f"{target_feature}_cuped": TargetRole()},
            )
        return result

    @classmethod
    def calc(cls, data: Dataset, cuped_features: dict[str, str], **kwargs) -> Dataset:
        return cls._inner_function(data, cuped_features)

    def execute(self, data: ExperimentData) -> ExperimentData:
        new_ds = self.calc(data=data.ds, cuped_features=self.cuped_features)
        # Calculate variance reductions
        variance_reductions = {}
        for target_feature, pre_target_feature in self.cuped_features.items():
            original_var = data.ds[target_feature].var()
            adjusted_var = new_ds[f"{target_feature}_cuped"].var()
            variance_reduction = (
                (1 - adjusted_var / original_var) * 100 if original_var > 0 else 0.0
            )
            variance_reductions[f"{target_feature}_cuped"] = variance_reduction
        # Save variance reductions to additional_fields
        for metric, reduction in variance_reductions.items():
            data.additional_fields = data.additional_fields.add_column(
                data=[reduction], role={f"{metric}_variance_reduction": StatisticRole()}
            )
        return data.copy(data=new_ds)

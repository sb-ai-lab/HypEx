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
            mean_xy = (result[target_feature] * result[pre_target_feature]).mean()
            mean_x = result[pre_target_feature].mean()
            mean_y = result[target_feature].mean()
            cov_xy = mean_xy - mean_x * mean_y

            std_y = result[target_feature].std()
            std_x = result[pre_target_feature].std()

            # Handle zero variance or NaN case (single observation)
            if std_y == 0 or std_x == 0 or std_y != std_y or std_x != std_x:
                theta = 0
            else:
                theta = cov_xy / (std_y * std_x)
            pre_target_mean = result[pre_target_feature].mean()
            new_values_ds = (
                result[target_feature]
                - (result[pre_target_feature] - pre_target_mean) * theta
            )
            result = result.add_column(
                data=new_values_ds, role={f"{target_feature}_cuped": TargetRole()}
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

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
from ..dataset.dataset import Dataset
from ..dataset.experiment_data import ExperimentData
from ..dataset.roles import TargetRole, ABCRole
from .abstract import Transformer

class NaDropper(Transformer):
    """Transformer that drops rows with NaN values in target columns.
    
    This is useful for preparing data for statistical tests that cannot handle 
    missing values (e.g., scipy.stats.ks_2samp).
    
    Args:
        target_roles: Roles to search for target columns. Defaults to TargetRole().
        how: Whether to drop rows if 'any' or 'all' targets are NaN. Defaults to 'any'.
    """
    def __init__(
        self,
        target_roles: ABCRole | Sequence[ABCRole] | None = None,
        how: str = "any",
        key: Any = "",
    ):
        super().__init__(key=key)
        self.target_roles = target_roles or TargetRole()
        self.how = how

    @staticmethod
    def _inner_function(
        data: Dataset,
        target_cols: list[str],
        how: str = "any",
    ) -> Dataset:
        if not target_cols:
            return data
        # Drop rows where any of the target columns have NaN
        return data.dropna(subset=target_cols, how=how)

    def execute(self, data: ExperimentData) -> ExperimentData:
        # Find all columns matching the target roles
        target_cols = data.ds.search_columns(roles=self.target_roles)
        
        if not target_cols:
            return data
            
        result = data.copy(
            data=self.calc(
                data=data.ds,
                target_cols=target_cols,
                how=self.how,
            )
        )
        return result
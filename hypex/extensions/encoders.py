from __future__ import annotations

import copy

import pandas as pd  # type: ignore
import pyspark.pandas as ps

from ..dataset import Dataset, DatasetAdapter, ABCRole
from ..dataset.backends import PandasDataset, SparkDataset
from .abstract import Extension

from ..utils.registry import backend_factory
from ..utils import Adapter

# TODO: needs to be removed due to migration to ml module.
class DummyEncoderExtension(Extension):
    """
    Master-backend class for DummyEncoder. 
    """
    @staticmethod
    def _resolve_source_column(
        dummy_col: str,
        source_cols: list[str],
    ) -> str:
        best = None
        for src in source_cols:
            if dummy_col.startswith(f"{src}_"):
                if best is None or len(src) > len(best):
                    best = src
        if best is None:
            raise KeyError(
                f"Cannot map dummy column '{dummy_col}' to any "
                f"source column in {source_cols}"
            )
        return best

    @classmethod
    def _build_roles(
        cls,
        dummies_columns: list[str],
        source_cols: list[str],
        source_roles: dict[str, ABCRole],
    ) -> dict[str, ABCRole]:
        roles = {}
        for col in dummies_columns:
            src = cls._resolve_source_column(col, source_cols)
            roles[col] = source_roles[src].asadditional(int)
            roles[col].data_type = bool
        return roles

@backend_factory.register(DummyEncoderExtension, PandasDataset)
class PandasDummyEncoderExtension(DummyEncoderExtension):
    """
    Slave-backend class on pandas for DummyEncoder. 
    """
    def calc(
        data: Dataset, target_cols: str | list[str] | None = None, **kwargs
    ):
        target_cols = Adapter.to_list(target_cols)
        dummies_df = pd.get_dummies(
            data=data[target_cols].raw_data, drop_first=True, dtype=int
        )
        # Setting roles to the dummies in additional fields based on the original
        # roles by searching based on the part of the dummy column name
        roles = DummyEncoderExtension._build_roles(
            dummies_columns=list(dummies_df.columns),
            source_cols=target_cols,
            source_roles=data.roles,
        )
        new_roles = copy.deepcopy(roles)
        for role in roles.values():
            role.data_type = bool
        return DatasetAdapter.to_dataset(dummies_df, roles=new_roles, small=False)

@backend_factory.register(DummyEncoderExtension, SparkDataset)
class SparkDummyEncoderExtension(DummyEncoderExtension):
    """
    Slave-backend class on pyspark for DummyEncoder. 
    """

    def calc(
        data: Dataset, target_cols: str | list[str] | None = None, **kwargs
    ):
        target_cols = Adapter.to_list(target_cols)
        dummies_df = ps.get_dummies(
            data=data[target_cols].raw_data, drop_first=True, dtype=int
        )

        roles = DummyEncoderExtension._build_roles(
            dummies_columns=list(dummies_df.columns),
            source_cols=target_cols,
            source_roles=data.roles,
        )
        new_roles = copy.deepcopy(roles)
        for role in roles.values():
            role.data_type = bool
        return DatasetAdapter.to_dataset(dummies_df, roles=new_roles, small=False)
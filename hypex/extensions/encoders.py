from __future__ import annotations

import copy

import pandas as pd  # type: ignore
import pyspark.pandas as ps

from ..dataset import Dataset, DatasetAdapter
from ..dataset.backends import PandasDataset, SparkDataset
from .abstract import Extension

from ..utils.registry import backend_factory

# TODO: needs to be removed due to migration to ml module.
class DummyEncoderExtension(Extension):
    """
    Master-backend class for DummyEncoder. 
    """

@backend_factory.register(DummyEncoderExtension, PandasDataset)
class PandasDummyEncoderExtension(DummyEncoderExtension):
    """
    Slave-backend class on pandas for DummyEncoder. 
    """
    def calc(
        data: Dataset, target_cols: str | None = None, **kwargs
    ):
        dummies_df = pd.get_dummies(
            data=data[target_cols].data, drop_first=True, dtype=int
        )
        # Setting roles to the dummies in additional fields based on the original
        # roles by searching based on the part of the dummy column name
        roles = {
            col: data.roles[col[: col.rfind("_")]].asadditional(int)
            for col in dummies_df.columns
        }
        new_roles = copy.deepcopy(roles)
        for role in roles.values():
            role.data_type = bool
        return DatasetAdapter.to_dataset(dummies_df, roles=new_roles)

@backend_factory.register(DummyEncoderExtension, SparkDataset)
class SparkDummyEncoderExtension(DummyEncoderExtension):
    """
    Slave-backend class on pyspark for DummyEncoder. 
    """

    def calc(
        data: Dataset, target_cols: str | None = None, **kwargs
    ):
        dummies_df = ps.get_dummies(
            data=data[target_cols].data, drop_first=True, dtype=int
        )

        roles = {
            col: data.roles[col[: col.rfind("_")]].asadditional(int)
            for col in dummies_df.columns
        }
        new_roles = copy.deepcopy(roles)
        for role in roles.values():
            role.data_type = bool
        return DatasetAdapter.to_dataset(dummies_df, roles=new_roles)
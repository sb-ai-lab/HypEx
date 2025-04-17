from __future__ import annotations

import copy

import pandas as pd  # type: ignore

from ..dataset import Dataset, DatasetAdapter
from .abstract import Extension


class DummyEncoderExtension(
    Extension
):  # TODO: role types are being rewritten, needs to be fixed
    @staticmethod
    def _calc_pandas(data: Dataset, target_cols: str | None = None, **kwargs):
        dummies_df = pd.get_dummies(data=data[target_cols].data, drop_first=True)
        # Setting roles to the dummies in additional fields based on the original
        # roles by searching based on the part of the dummy column name
        roles = {col: data.roles[col[: col.rfind("_")]] for col in dummies_df.columns}
        new_roles = copy.deepcopy(roles)
        for role in roles.values():
            role.data_type = bool
        return DatasetAdapter.to_dataset(dummies_df, roles=new_roles)

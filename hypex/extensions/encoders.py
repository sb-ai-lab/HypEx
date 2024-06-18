from typing import Optional
import pandas as pd

from hypex.dataset import Dataset
from hypex.extensions.abstract import Extension
from hypex.utils import (
    FieldKeyTypes,
)
from hypex.utils.adapter import Adapter


class DummyEncoderExtension(Extension):

    @staticmethod
    def _calc_pandas(
        data: Dataset, target_cols: Optional[FieldKeyTypes] = None, **kwargs
    ):
        dummies_df = pd.get_dummies(data=data[target_cols].data, drop_first=True)
        # Setting roles to the dummies in additional fields based on the original
        # roles by searching based on the part of the dummy column name
        roles = {col: data.roles[col[: col.rfind("_")]] for col in dummies_df.columns}
        for role in roles.values():
            role.data_type = bool
        return Adapter.to_dataset(dummies_df, roles=roles)

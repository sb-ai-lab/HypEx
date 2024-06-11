from typing import Union, Dict, Any, List

import pandas as pd

from hypex.dataset import Dataset, InfoRole
from hypex.utils import ScalarType


class Adapter:

    @staticmethod
    def to_dataset(data: Any, col_name: Union[str, List]) -> Dataset:
        """
        Convert a number to a Dataset
        """
        if isinstance(data, dict):
            return Adapter.dict_to_dataset(data)
        elif isinstance(data, list):
            if not isinstance(col_name, list):
                col_name = [col_name]
            return Adapter.list_to_dataset(data, col_name)
        elif isinstance(data, ScalarType):
            return Adapter.value_to_dataset(data, col_name)
        else:
            raise ValueError(f"Unsupported data type {type(data)}")

    @staticmethod
    def value_to_dataset(data: ScalarType, col_name: str) -> Dataset:
        """
        Convert a float to a Dataset
        """
        return Dataset(
            roles={col_name: InfoRole()}, data=pd.DataFrame(data=[data], columns=[col_name])
        )

    @staticmethod
    def dict_to_dataset(data: Dict) -> Dataset:
        """
        Convert a dict to a Dataset
        """
        return Dataset.from_dict(
            data=[data], roles={name: InfoRole() for name in data.keys()}
        )

    @staticmethod
    def list_to_dataset(data: List, col_name: List) -> Dataset:
        """
        Convert a list to a Dataset
        """
        return Dataset(
            roles={name: InfoRole() for name in col_name}, data=pd.DataFrame(data=data, columns=[col_name])
        )

from typing import Union, Dict, Any, List

import pandas as pd  # type: ignore

from hypex.dataset import Dataset, ABCRole
from hypex.utils import ScalarType, FieldKeyTypes


class Adapter:

    @staticmethod
    def to_dataset(
        data: Any, roles: Union[ABCRole, Dict[FieldKeyTypes, ABCRole]]
    ) -> Dataset:
        """
        Convert various data types to a Dataset object.
        Args:
        data (Any): The input data to convert.
        col_name (Union[str, List]): The column name or list of column names.
        Returns:
        Dataset: A Dataset object generated from the input data.
        Raises:
        ValueError: If the data type is not supported.
        """
        # Convert data based on its type
        if isinstance(data, dict):
            return Adapter.dict_to_dataset(data, roles)
        elif isinstance(data, pd.DataFrame):
            return Adapter.frame_to_dataset(data, roles)
        elif isinstance(data, list):
            return Adapter.list_to_dataset(data, roles)
        elif any(isinstance(data, t) for t in [str, int, float, bool]):
            return Adapter.value_to_dataset(data, roles)
        elif isinstance(data, Dataset):
            return data
        else:
            raise ValueError(f"Unsupported data type {type(data)}")

    @staticmethod
    def value_to_dataset(
        data: ScalarType, roles: Dict[FieldKeyTypes, ABCRole]
    ) -> Dataset:
        """
        Convert a float to a Dataset
        """
        return Dataset(
            roles=roles,
            data=pd.DataFrame(data=data, columns=[list(roles.keys())[0]]),
        )

    @staticmethod
    def dict_to_dataset(
        data: Dict, roles: Union[ABCRole, Dict[FieldKeyTypes, ABCRole]]
    ) -> Dataset:
        """
        Convert a dict to a Dataset
        """
        roles_names = list(data.keys())
        if isinstance(roles, Dict):
            return Dataset.from_dict(data=[data], roles=roles)
        elif isinstance(roles, ABCRole):
            return Dataset.from_dict(
                data=[data], roles={name: roles for name in roles_names}
            )

    @staticmethod
    def list_to_dataset(data: List, roles: Dict[FieldKeyTypes, ABCRole]) -> Dataset:
        """
        Convert a list to a Dataset
        """
        return Dataset(
            roles=roles,
            data=pd.DataFrame(data=data, columns=[list(roles.keys())[0]]),
        )

    @staticmethod
    def frame_to_dataset(
        data: pd.DataFrame, roles: Dict[FieldKeyTypes, ABCRole]
    ) -> Dataset:
        """
        Convert a list to a Dataset
        """
        return Dataset(
            roles=roles,
            data=data,
        )

    @staticmethod
    def to_list(data: Any) -> List:
        if not isinstance(data, list):
            return [data]
        else:
            return data

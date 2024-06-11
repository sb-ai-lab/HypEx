from typing import Union, Dict, Any, List

import pandas as pd

from hypex.dataset import Dataset, InfoRole


class Adapter:

    @staticmethod
    def to_list(data: Any) -> List:
        if not isinstance(data, list):
            return [data]
        else:
            return data

    # def to_dataset(name: str, data: Union[float, int]) -> Dataset:

    @staticmethod
    def float_to_dataset(name: str, data: Union[float, int]) -> Dataset:
        """
        Convert a float to a Dataset
        """
        return Dataset(
            roles={name: InfoRole()}, data=pd.DataFrame(data=[data], columns=[name])
        )

    @staticmethod
    def dict_to_dataset(data: Dict) -> Dataset:
        """
        Convert a dict to a Dataset
        """
        return Dataset.from_dict(
            [data], roles={name: InfoRole() for name in data.keys()}
        )

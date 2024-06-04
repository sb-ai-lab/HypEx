from typing import Union, Dict

import pandas as pd

from hypex.dataset import Dataset, InfoRole


class Adapter:

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

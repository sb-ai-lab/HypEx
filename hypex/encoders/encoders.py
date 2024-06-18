from typing import Optional

from hypex.dataset import Dataset
from hypex.encoders.abstract import Encoder
from hypex.extensions.encoders import DummyEncoderExtension
from hypex.utils import FieldKeyTypes


class DummyEncoder(Encoder):
    @staticmethod
    def _inner_function(
        data: Dataset, target_cols: Optional[FieldKeyTypes] = None, **kwargs
    ) -> Dataset:
        return DummyEncoderExtension().calc(
            data=data, target_cols=target_cols, **kwargs
        )

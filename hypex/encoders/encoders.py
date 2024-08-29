from typing import Optional

from ..dataset import Dataset
from .abstract import Encoder
from ..extensions.encoders import DummyEncoderExtension


class DummyEncoder(Encoder):
    @staticmethod
    def _inner_function(
        data: Dataset, target_cols: Optional[str] = None, **kwargs
    ) -> Dataset:
        if not target_cols:
            return data
        return DummyEncoderExtension().calc(
            data=data, target_cols=target_cols, **kwargs
        )

from __future__ import annotations

from ..dataset import Dataset
from ..extensions.encoders import DummyEncoderExtension
from .abstract import Encoder
from ..utils.registry import backend_factory

class DummyEncoder(Encoder):
    @staticmethod
    def _inner_function(
        data: Dataset, target_cols: str | None = None, **kwargs
    ) -> Dataset:
        if not target_cols:
            return Dataset.create_empty()
        
        encoder_cls = backend_factory(DummyEncoderExtension, data)
        return encoder_cls().calc(
            data=data, target_cols=target_cols, **kwargs
        )

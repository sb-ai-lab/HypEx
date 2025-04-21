from __future__ import annotations

from ..dataset import Dataset
from ..extensions.encoders import DummyEncoderExtension
from .abstract import Encoder


class DummyEncoder(Encoder):
    @classmethod
    def calc(cls, data: Dataset, **kwargs: str | None) -> Dataset:
        target_cols = kwargs.get("target_cols")
        if not target_cols:
            return data
        return DummyEncoderExtension().calc(
            data=data, target_cols=target_cols, **kwargs
        )

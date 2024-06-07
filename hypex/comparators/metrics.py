from typing import Optional

from .abstract import GroupComparator
from ..dataset import Dataset


class ATC(GroupComparator):

    # TODO переопределить функцию подготовки данных
    @classmethod
    def _inner_function(cls, data: Dataset, other: Optional[Dataset] = None, **kwargs):
        return {"ATC": (data - other).mean()}


class ATT(GroupComparator):
    @classmethod
    def _inner_function(cls, data: Dataset, other: Optional[Dataset] = None, **kwargs):
        return {"ATT": (other - data).mean()}


class ATE(GroupComparator):
    @classmethod
    def _inner_function(
        cls, data: Dataset, other: Optional[Dataset] = None, **kwargs
    ) -> float:
        atc = ATC.calc(data, other)
        att = ATT.calc(data, other)
        return (att * len(data) + atc * len(other)) / (len(data) + len(other))

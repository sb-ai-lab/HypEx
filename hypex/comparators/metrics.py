from typing import Optional

from .abstract import MatchingComparator
from ..dataset import Dataset


class ATC(MatchingComparator):

    # TODO переопределить функцию подготовки данных
    @classmethod
    def _inner_function(cls, data: Dataset, other: Optional[Dataset] = None, **kwargs):
        return {"ATC": (data - other).mean()}


class ATT(MatchingComparator):
    @classmethod
    def _inner_function(cls, data: Dataset, other: Optional[Dataset] = None, **kwargs):
        return {"ATT": (other - data).mean()}


class ATE(MatchingComparator):
    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        other: Optional[Dataset] = None,
        att: Optional[float] = None,
        atc: Optional[float] = None,
        **kwargs
    ) -> float:
        att = att if att is not None else ATT().calc(data, other).get("ATT")
        atc = atc if atc is not None else ATC().calc(data, other).get("ATC")
        return (att * len(data) + atc * len(other)) / (len(data) + len(other))

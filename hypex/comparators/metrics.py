from typing import Optional

from .abstract import MatchingComparator
from ..dataset import Dataset


class ATC(MatchingComparator):
    @classmethod
    def _inner_function(cls, data: Dataset, other: Optional[Dataset] = None, **kwargs):
        other = cls._check_test_data(test_data=other)
        return {"ATC": (data - other).mean()}


class ATT(MatchingComparator):

    @classmethod
    def _inner_function(cls, data: Dataset, other: Optional[Dataset] = None, **kwargs):
        other = cls._check_test_data(test_data=other)
        return {"ATT": (other - data).mean()}


class ATE(MatchingComparator):
    @staticmethod
    def _check_test_data(
        test_data: Optional[Dataset] = None,
        att: Optional[float] = None,
        atc: Optional[float] = None,
    ) -> Dataset:
        if test_data is None:
            raise ValueError("test_data is needed for evaluation")
        if att is None or atc is None:
            raise ValueError("ATC and ATT must be provided.")
        return test_data

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        other: Optional[Dataset] = None,
        att: Optional[float] = None,
        atc: Optional[float] = None,
        **kwargs
    ) -> float:
        other = cls._check_test_data(test_data=other, att=att, atc=atc)
        return (att * len(data) + atc * len(other)) / (len(data) + len(other))

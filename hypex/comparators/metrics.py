from typing import Optional

from .abstract import MatchingComparator
from ..dataset import Dataset


class ATC(MatchingComparator):
    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ):
        test_data = cls._check_test_data(test_data=test_data)
        return {"ATC": (data - test_data).mean()}


class ATT(MatchingComparator):

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ):
        test_data = cls._check_test_data(test_data=test_data)
        return {"ATT": (test_data - data).mean()}


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
        test_data: Optional[Dataset] = None,
        att: Optional[float] = None,
        atc: Optional[float] = None,
        **kwargs
    ) -> float:
        test_data = cls._check_test_data(test_data=test_data, att=att, atc=atc)
        return (att * len(data) + atc * len(test_data)) / (len(data) + len(test_data))

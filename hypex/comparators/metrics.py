from typing import Optional

import numpy as np
from mpmath import ln
from numpy import arange

from .abstract import MatchingComparator
from ..dataset import Dataset

NUM_OF_BUCKETS: int = 20


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


class PSI(MatchingComparator):

    @classmethod
    def _inner_function(cls, data: Dataset, other: Optional[Dataset] = None, **kwargs):
        other = cls._check_test_data(test_data=other)
        data.sort(ascending=False)
        other.sort(ascending=False)
        data_column = data.iloc[:, 0]
        other_column = other.iloc[:, 0]
        data_bins = arange(
            data_column.min(),
            data_column.max(),
            (data_column.max() - data_column.min()) / NUM_OF_BUCKETS,
        )
        other_bins = arange(
            other_column.min(),
            other_column.max(),
            (other_column.max() - other_column.min()) / NUM_OF_BUCKETS,
        )
        data_groups = data_column.groupby(
            data_column.cut(data_bins).get_values(column=data.columns[0])
        )
        other_groups = other_column.groupby(
            other_column.cut(other_bins).get_values(column=other.columns[0])
        )
        data_psi = np.array(x[1].count() for x in data_groups)
        other_psi = np.array(x[1].count() for x in other_groups)
        psi = (data_psi - other_psi) * ln(data_psi / other_psi)
        return {"PSI": psi.sum()}


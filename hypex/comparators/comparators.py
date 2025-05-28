from __future__ import annotations

from typing import Literal

import numpy as np

from ..dataset import ABCRole, Dataset
from ..utils.constants import NUMBER_TYPES_LIST
from .abstract import Comparator

NUM_OF_BUCKETS = 10


class GroupDifference(Comparator):
    def __init__(
        self,
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ] = "groups",
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
    ):
        super().__init__(
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=target_roles,
        )

    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        **kwargs,
    ) -> dict:
        test_data = cls._check_test_data(test_data)
        control_mean = data.mean()
        test_mean = test_data.mean()

        return {
            "control mean": control_mean,
            "test mean": test_mean,
            "difference": test_mean - control_mean,
            "difference %": (
                (test_mean / control_mean - 1) * 100 if control_mean != 0 else None
            ),
        }


class GroupSizes(Comparator):
    def __init__(
        self,
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross", "matched_pairs"
        ] = "groups",
        grouping_role: ABCRole | None = None,
    ):
        super().__init__(
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=grouping_role,
        )

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> dict:
        size_a = len(data)
        size_b = len(test_data) if isinstance(test_data, Dataset) else 0

        return {
            "control size": size_a,
            "test size": size_b,
            "control size %": (size_a / (size_a + size_b)) * 100,
            "test size %": (size_b / (size_a + size_b)) * 100,
        }


class PSI(Comparator):
    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> dict[str, float]:
        test_data = cls._check_test_data(test_data=test_data)
        data.sort(ascending=False)
        test_data.sort(ascending=False)
        data_column = data.iloc[:, 0]
        test_data_column = test_data.iloc[:, 0]
        data_bins = np.arange(
            data_column.min(),
            data_column.max(),
            (data_column.max() - data_column.min()) / NUM_OF_BUCKETS,
        )
        test_data_bins = np.arange(
            test_data_column.min(),
            test_data_column.max(),
            (test_data_column.max() - test_data_column.min()) / NUM_OF_BUCKETS,
        )
        data_groups = data_column.groupby(
            data_column.cut(data_bins).get_values(column=data.columns[0])
        )
        test_data_groups = test_data_column.groupby(
            test_data_column.cut(test_data_bins).get_values(column=test_data.columns[0])
        )

        data_psi = [x[1].count() / len(data) for x in data_groups]
        test_data_psi = [x[1].count() / len(test_data) for x in test_data_groups]
        psi = [(y - x) * np.log(y / x) for x, y in zip(data_psi, test_data_psi)]
        return {"PSI": sum(psi)}

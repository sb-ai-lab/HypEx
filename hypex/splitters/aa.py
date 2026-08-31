from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..dataset import (
    AdditionalTreatmentRole,
    Dataset,
    ExperimentData,
    StratificationRole,
)
from ..dataset.roles import ConstGroupRole
from ..executor import Calculator
from ..utils import ExperimentDataEnum


class AASplitter(Calculator):
    def __init__(
        self,
        control_size: float = 0.5,
        random_state: int | None = None,
        sample_size: float | None = None,
        constant_key: bool = True,
        save_groups: bool = True,
        groups_sizes: list[float] | None = None,
        key: Any = "",
    ):
        self.control_size = control_size
        self.random_state = random_state
        self._key = key
        self.constant_key = constant_key
        self.save_groups = save_groups
        self.sample_size = sample_size
        self.groups_sizes = groups_sizes
        super().__init__(key)

    def _generate_params_hash(self):
        hash_parts: list[str] = []
        if self.control_size != 0.5:
            hash_parts.append(f"cs {self.control_size}")
        if self.random_state is not None:
            hash_parts.append(f"rs {self.random_state}")
        if self.groups_sizes is not None:
            hash_parts.append(f"gs {self.groups_sizes}")
        self._params_hash = "|".join(hash_parts)

    def init_from_hash(self, params_hash: str):
        hash_parts: list[str] = params_hash.split("|")
        for hash_part in hash_parts:
            if hash_part.startswith("cs"):
                self.control_size = float(hash_part[hash_part.rfind(" ") + 1 :])
            elif hash_part.startswith("rs"):
                self.random_state = int(hash_part[hash_part.rfind(" ") + 1 :])
            elif hash_part.startswith("gs"):
                self.groups_sizes = []
                groups_sizes = (
                    hash_part[hash_part.find(" ") + 1 :].strip("[]").split(",")
                )
                self.groups_sizes = [float(gs) for gs in groups_sizes]
        self._generate_id()

    @property
    def key(self) -> Any:
        return self._key

    @key.setter
    def key(self, value: Any):
        if not self.constant_key:
            self._key = value
            self._generate_id()

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        data = data.set_value(
            ExperimentDataEnum.additional_fields,
            self._id,
            value,
            role=AdditionalTreatmentRole(),
        )

        if self.save_groups:
            data.groups[self.id] = {
                group: data.ds.loc[group_data.index]
                for group, group_data in data.additional_fields.groupby(self.id)
            }
        return data

    @staticmethod
    def _apply_const_groups(
        split_series: pd.Series,
        const_data: dict[Any, Dataset],
        label_map: dict[int, str],
        const_group_field: str | None,
    ) -> None:
        """Write the groups the user pinned over the split of the free rows.

        ``control`` and ``test_N`` are the labels of the split itself, ``test`` is
        the documented alias for ``test_1`` of a two-group split. Anything else is
        a typo or a group that was not asked for, and both used to end up silently
        in ``test_1``.
        """
        codes = {label: code for code, label in label_map.items()}
        codes.setdefault("test", 1)
        for group, group_data in const_data.items():
            code = codes.get(str(group))
            if code is None:
                raise ValueError(
                    f"Unknown constant group {str(group)!r} in column "
                    f"'{const_group_field}'. Expected one of {sorted(codes)}, or a "
                    f"missing value (None / np.nan) for a row that takes part in "
                    f"the split - note that np.where(mask, 'test', np.nan) writes "
                    f"the string 'nan', which is not a missing value."
                )
            split_series[group_data.index] = code

    @staticmethod
    def _inner_function(
        data: Dataset,
        random_state: int | None = None,
        control_size: float = 0.5,
        groups_sizes: list[float] | None = None,
        sample_size: float | None = None,
        const_group_field: str | None = None,
        **kwargs,
    ) -> list[str]:
        sample_size = 1.0 if sample_size is None else sample_size
        control_indexes = []
        const_data: dict[Any, Dataset] = {}
        free_size = len(data)
        if const_group_field:
            const_data = dict(data.groupby(const_group_field))
            control_data = const_data.get("control")
            if control_data is not None:
                control_indexes = list(control_data.index)
            # rows with a missing value in the const column are not grouped,
            # so they are the ones left to be split
            free_size = len(data) - sum(len(cd) for cd in const_data.values())
            control_size = (
                0.0
                if free_size == 0
                else max(
                    0.0,
                    (len(data) * control_size - len(control_indexes)) / free_size,
                )
            )
        # every row can already be pinned to a constant group: then there is
        # nothing to split and the constant assignment is the split itself
        addition_indexes: list = []
        if free_size:
            experiment_data = (
                data[data[const_group_field].isna()] if const_group_field else data
            )
            addition_indexes = list(
                experiment_data.sample(
                    frac=sample_size, random_state=random_state
                ).index
            )
        edges = []
        if groups_sizes:
            if sum(groups_sizes) != 1:
                raise ValueError("Groups sizes must sum to 1")
            for group_size in groups_sizes:
                size = int(len(addition_indexes) * group_size) + (
                    0 if not edges else edges[-1]
                )
                size = min(size, len(addition_indexes))
                if size not in edges:
                    edges += [size]
        else:
            edges = [int(len(addition_indexes) * control_size), len(addition_indexes)]
        control_indexes += addition_indexes[: edges[0]]
        test_indexes = [
            addition_indexes[edges[i - 1] : edges[i]] for i in range(1, len(edges))
        ]

        split_series = pd.Series(
            np.ones(data.data.shape[0], dtype="int"), index=data.data.index
        )
        split_series[control_indexes] -= 1
        for i, test_index in enumerate(test_indexes):
            split_series[test_index] += i

        # groups can be requested but stay empty - with no free rows to split, or
        # with sizes too small to take a row - and they still need their label
        requested_groups = len(groups_sizes) if groups_sizes else 2
        label_map = {0: "control"}
        label_map.update(
            {i: f"test_{i}" for i in range(1, max(len(edges), requested_groups))}
        )

        # pinned rows are written last: their group is fixed and must not be
        # shifted by the random split of the free rows
        AASplitter._apply_const_groups(
            split_series, const_data, label_map, const_group_field
        )

        split_series = split_series.map(label_map)

        return split_series.to_list()

    def execute(self, data: ExperimentData) -> ExperimentData:
        const_group_fields = data.ds.search_columns(ConstGroupRole())
        const_group_fields = (
            const_group_fields[0] if len(const_group_fields) > 0 else None
        )
        result = self.calc(
            data.ds,
            random_state=self.random_state,
            control_size=self.control_size,
            sample_size=self.sample_size,
            const_group_field=const_group_fields,
            groups_sizes=self.groups_sizes,
        )
        return self._set_value(
            data,
            result,
        )


class AASplitterWithStratification(AASplitter):
    @staticmethod
    def _inner_function(
        data: Dataset,
        random_state: int | None = None,
        control_size: float = 0.5,
        grouping_fields=None,
        **kwargs,
    ) -> list[str]:
        if not grouping_fields:
            return AASplitter._inner_function(
                data, random_state, control_size, **kwargs
            )

        splits = []
        index = []
        for _, group_data in data.groupby(grouping_fields):
            splits.extend(
                AASplitter._inner_function(
                    group_data, random_state, control_size, **kwargs
                )
            )
            index.extend(list(group_data.index))
        # groupby breaks the original row order, so restore it before returning
        # a flat list: the value is written as a column of the source dataset.
        return pd.Series(splits, index=index).reindex(data.index).to_list()

    def execute(self, data: ExperimentData) -> ExperimentData:
        grouping_fields = data.ds.search_columns(StratificationRole())
        result = self.calc(
            data.ds,
            random_state=self.random_state,
            control_size=self.control_size,
            grouping_fields=grouping_fields,
            groups_sizes=self.groups_sizes,
        )
        return self._set_value(data, result)


#
# class AASplitterWithStratification(AASplitter):
#     def __init__(
#         self,
#         control_size: float = 0.5,
#         random_state: Optional[int] = None,
# #         key: Any = "",
#     ):
#         super().__init__(control_size, random_state,  key)
#
#     def calc(self, data: Dataset):
#         stratification_columns = data.get_columns_by_roles(StratificationRole())
#
#         groups = data.groupby(stratification_columns)
#         result = Dataset._create_empty()
#         for _, gd in groups:
#             ged = ExperimentData(gd)
#             ged = super().execute(ged)
#
#             result = ged if result is None else result.append(ged)
#         return result["group"]


# As idea
# class SplitterAAMulti(ExperimentMulti):
#     def execute(self, data):
#         raise NotImplementedError

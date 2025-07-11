from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..dataset import (
    AdditionalTreatmentRole,
    Dataset,
    ExperimentData,
    StratificationRole,
    TreatmentRole,
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
                groups_sizes = hash_part[hash_part.find(" ") + 1 :].strip("[]").split(",")
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
        if const_group_field:
            const_data = dict(data.groupby(const_group_field))
            control_data = const_data.get("control")
            if control_data is not None:
                control_indexes = list(control_data.index)
            const_size = sum(len(cd) for cd in const_data.values())
            control_size = (len(data) * control_size - const_size) / (
                    len(data) - const_size
            )
        experiment_data = (
            data[data[const_group_field].isna()] if const_group_field else data
        )
        experiment_data_index = experiment_data.sample(
            frac=sample_size, random_state=random_state
        ).index
        addition_indexes = list(experiment_data_index)
        edges = []
        if groups_sizes:
            if sum(groups_sizes) != 1:
                raise ValueError("Groups sizes must sum to 1")
            for group_size in groups_sizes:
                size = int(len(addition_indexes) * group_size) + (0 if not edges else edges[-1])
                size = min(size, len(addition_indexes))
                if not size in edges:
                    edges += [size]
        else:
            edges = [int(len(addition_indexes) * control_size), len(addition_indexes)]
        control_indexes += addition_indexes[:edges[0]]
        test_indexes = [addition_indexes[edges[i - 1]:edges[i]] for i in range(1, len(edges))]

        split_series = pd.Series(np.ones(data.data.shape[0], dtype="int"), index=data.data.index)
        split_series[control_indexes] -= 1
        for i, test_index in enumerate(test_indexes):
            split_series[test_index] += i

        label_map = {0: "control"}
        label_map.update({i: f"test_{i}" for i in range(1, len(edges))})
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
    ) -> list[str] | Dataset:
        if not grouping_fields:
            return AASplitter._inner_function(
                data, random_state, control_size, **kwargs
            )

        result = {"split": []}
        index = []
        for group, group_data in data.groupby(grouping_fields):
            result["split"].extend(
                AASplitter._inner_function(group_data, random_state, control_size)
            )
            index.extend(list(group_data.index))
        return Dataset.from_dict(result, index=index, roles={"split": TreatmentRole()})

    def execute(self, data: ExperimentData) -> ExperimentData:
        grouping_fields = data.ds.search_columns(StratificationRole())
        result = self.calc(
            data.ds,
            random_state=self.random_state,
            control_size=self.control_size,
            grouping_fields=grouping_fields,
            groups_sizes=self.groups_sizes
        )
        if isinstance(result, Dataset):
            result = result.replace_roles({"split": AdditionalTreatmentRole()})
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

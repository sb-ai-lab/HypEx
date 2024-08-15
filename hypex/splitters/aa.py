from collections import Counter
from typing import Any, List, Optional, Dict

from hypex.dataset import (
    Dataset,
    ExperimentData,
    TreatmentRole,
    StratificationRole,
    AdditionalTreatmentRole,
)
from hypex.executor import Calculator
from hypex.utils import ExperimentDataEnum


class AASplitter(Calculator):
    def __init__(
        self,
        control_size: float = 0.5,
        random_state: Optional[int] = None,
        constant_key: bool = True,
        save_groups: bool = True,
        key: Any = "",
    ):
        self.control_size = control_size
        self.random_state = random_state
        self._key = key
        self.constant_key = constant_key
        self.save_groups = save_groups
        super().__init__(key)

    def _generate_params_hash(self):
        hash_parts: List[str] = []
        if self.control_size != 0.5:
            hash_parts.append(f"cs {self.control_size}")
        if self.random_state is not None:
            hash_parts.append(f"rs {self.random_state}")
        self._params_hash = "|".join(hash_parts)

    def init_from_hash(self, params_hash: str):
        hash_parts: List[str] = params_hash.split("|")
        for hash_part in hash_parts:
            if hash_part.startswith("cs"):
                self.control_size = float(hash_part[hash_part.rfind(" ") + 1 :])
            elif hash_part.startswith("rs"):
                self.random_state = int(hash_part[hash_part.rfind(" ") + 1 :])
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
        random_state: Optional[int] = None,
        control_size: float = 0.5,
        **kwargs,
    ) -> List[str]:
        experiment_data = data.shuffle(random_state)
        addition_indexes = list(experiment_data.index)
        edge = int(len(addition_indexes) * control_size)
        control_indexes = addition_indexes[:edge]

        return ["control" if i in control_indexes else "test" for i in data.index]

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = self.calc(
            data.ds, random_state=self.random_state, control_size=self.control_size
        )
        return self._set_value(
            data,
            result,
        )


class AASplitterWithStratification(AASplitter):
    @staticmethod
    def _inner_function(
        data: Dataset,
        random_state: Optional[int] = None,
        control_size: float = 0.5,
        grouping_fields=None,
        **kwargs,
    ) -> Dataset:
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

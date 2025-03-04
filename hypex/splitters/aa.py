from typing import Any, List, Optional, Union

from ..dataset import (
    Dataset,
    ExperimentData,
    TreatmentRole,
    StratificationRole,
    AdditionalTreatmentRole,
)
from ..dataset.roles import ConstGroupRole
from ..executor import Calculator
from ..utils import ExperimentDataEnum


class AASplitter(Calculator):
    def __init__(
        self,
        control_size: float = 0.5,
        random_state: Optional[int] = None,
        sample_size: Optional[float] = None,
        constant_key: bool = True,
        save_groups: bool = True,
        key: Any = "",
    ):
        self.control_size = control_size
        self.random_state = random_state
        self._key = key
        self.constant_key = constant_key
        self.save_groups = save_groups
        self.sample_size = sample_size
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
        sample_size: Optional[float] = None,
        const_group_field: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
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
        experiment_data = experiment_data.sample(
            frac=sample_size, random_state=random_state
        )
        addition_indexes = list(experiment_data.index)
        edge = int(len(addition_indexes) * control_size)
        control_indexes += addition_indexes[:edge]

        return ["control" if i in control_indexes else "test" for i in data.index]

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
    ) -> Union[List[str], Dataset]:
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

from typing import Dict, Any, Union

from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.dataset.roles import GroupingRole, StratificationRole, TreatmentRole
from hypex.experiment.experiment import Executor, ComplexExecutor
from hypex.transformers.transformers import Shuffle
from hypex.utils.enums import ExperimentDataEnum


class AASplitter(ComplexExecutor):
    def __init__(
        self,
        control_size: float = 0.5,
        random_state: Union[int, None] = None,
        inner_executors: Union[Dict[str, Executor], None] = None,
        full_name: Union[str, None] = None,
        key: Any = 0,
    ):
        self.control_size = control_size
        self.random_state = random_state
        self.default_inner_executors = {"shuffle": Shuffle(self.random_state)}

        super().__init__(inner_executors, full_name, key)

    def generate_params_hash(self) -> str:
        return f"{self.random_state}"

    def _set_value(self, data: ExperimentData, value=None, key=None) -> ExperimentData:
        data = data.set_value(
            ExperimentDataEnum.additional_fields,
            self._id,
            str(self.full_name),
            value,
            role=TreatmentRole(),
        )
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data: ExperimentData = self.inner_executors["shuffle"].execute(data)

        addition_indexes = list(experiment_data.index)
        edge = int(len(addition_indexes) * self.control_size)

        result_group = ["A" if i < edge else "B" for i in addition_indexes]
        data = self._set_value(data, result_group)

        return data


class AASplitterWithGrouping(AASplitter):
    def execute(self, data: ExperimentData):
        group_field = data.get_columns_by_roles(GroupingRole())
        groups = list(data.groupby(group_field))
        edge = len(groups) // 2
        result = []
        for i, group in enumerate(groups):
            group_ds = Dataset.from_dict(
                [{"group_for_split": group[0], "group": "A" if i < edge else "B"}]
                * len(group[1]),
                roles={"group_for_split": GroupingRole(), "group": TreatmentRole()},
                index=group[1].index,
            )
            result = group_ds if result is None else result.append(group_ds)

        self._set_value(data, result["group"])
        return data


class AASplitterWithStratification(AASplitter):
    def __init__(
        self,
        control_size: float = 0.5,
        random_state: int = None,
        full_name: str = None,
        key: Any = 0,
    ):
        super().__init__(control_size, random_state, full_name, key)

    def execute(self, data):
        stratification_columns = data.get_columns_by_roles(StratificationRole())

        groups = data.groupby(stratification_columns)
        result = None
        for _, gd in groups:
            ged = ExperimentData(gd)
            ged = self.super().execute(ged)

            result = ged if result is None else result.append(ged)

        self._set_value(data, result["group"])
        return data


# As idea
# class SplitterAAMulti(ExperimentMulti):
#     def execute(self, data):
#         raise NotImplementedError

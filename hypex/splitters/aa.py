from typing import Dict, Any, List, Optional

from hypex.dataset import (
    ExperimentData,
    Dataset,
    GroupingRole,
    StratificationRole,
    TreatmentRole,
)
from hypex.executor import ComplexExecutor, Executor
from hypex.transformers import Shuffle
from hypex.utils import ExperimentDataEnum


class AASplitter(ComplexExecutor):
    def __init__(
        self,
        control_size: float = 0.5,
        random_state: Optional[int] = None,
        full_name: Optional[str] = None,
        key: Any = "",
        inner_executors: Optional[Dict[str, Executor]] = None,
    ):
        self.control_size = control_size
        self.random_state = random_state
        self.default_inner_executors = {"shuffle": Shuffle(self.random_state)}

        super().__init__(inner_executors, full_name, key)

    def generate_params_hash(self) -> str:
        return f"{self.random_state}"

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        data = data.set_value(
            ExperimentDataEnum.additional_fields,
            self._id,
            str(self.full_name),
            value,
            role=TreatmentRole(),
        )
        return data

    def calc(self, data: ExperimentData) -> List[str]:
        experiment_data: ExperimentData = self.inner_executors["shuffle"].execute(data)

        addition_indexes = list(experiment_data.index)
        edge = int(len(addition_indexes) * self.control_size)

        return ["A" if i < edge else "B" for i in addition_indexes]


class AASplitterWithGrouping(AASplitter):
    def calc(self, data: Dataset):
        group_field = data.get_columns_by_roles(GroupingRole())
        groups = list(data.groupby(group_field))
        edge = len(groups) // 2
        result: Dataset = Dataset._create_empty({})
        for i, group in enumerate(groups):
            group_ds = Dataset.from_dict(
                [{"group_for_split": group[0], "group": "A" if i < edge else "B"}]
                * len(group[1]),
                roles={"group_for_split": GroupingRole(), "group": TreatmentRole()},
                index=group[1].index,
            )
            result = group_ds if result is None else result.append(group_ds)
        return result["group"]


class AASplitterWithStratification(AASplitter):
    def __init__(
        self,
        control_size: float = 0.5,
        random_state: Optional[int] = None,
        full_name: Optional[str] = None,
        key: Any = "",
    ):
        super().__init__(control_size, random_state, full_name, key)

    def calc(self, data: Dataset):
        stratification_columns = data.get_columns_by_roles(StratificationRole())

        groups = data.groupby(stratification_columns)
        result = Dataset._create_empty({})
        for _, gd in groups:
            ged = ExperimentData(gd)
            ged = super().execute(ged)

            result = ged if result is None else result.append(ged)
        return result["group"]


# As idea
# class SplitterAAMulti(ExperimentMulti):
#     def execute(self, data):
#         raise NotImplementedError

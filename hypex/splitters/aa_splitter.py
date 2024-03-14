from typing import List, Dict
import warnings

from hypex.experiment.experiment import Executor, Experiment
from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.dataset.roles import GroupingRole, StratificationRole, TreatmentRole
from hypex.transformers.transformers import Shuffle
from hypex.describers.describers import Unique

# TODO: To Experiment
# TODO: Set group


class AASplitter(ComplexExecutor):
    def get_inner_executors(
        self, inner_executors: Dict[str, Executor] = None
    ) -> Dict[str, Executor]:
        if inner_executors and "shuffle" not in inner_executors:
            warnings.warn(
                "Shuffle executor not found in inner_executors. Will shuffle will be used by default",
                category=Warning,
            )
            inner_executors = {}
        if not inner_executors:
            inner_executors = {"shuffle": Shuffle(self.random_state)}
        return inner_executors

    def __init__(
        self,
        control_size: float = 0.5,
        random_state: int = None,
        inner_executors: Dict[str, Executor] = None,
        full_name: str = None,
        index: int = 0,
    ):
        self.control_size = control_size
        self.random_state = random_state

        super().__init__(self.get_inner_executors(inner_executors), full_name, index)

    def generate_params_hash(self) -> str:
        return f"{self.random_state}"

    def _set_value(self, data: ExperimentData, value) -> ExperimentData:
        return data.set_value(
            "additional_fields",
            self._id,
            self.full_name,
            value,
            role=TreatmentRole,
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data: ExperimentData = self.inner_executors["shuffle"].execute(data)

        addition_indexes = list(experiment_data.index)
        edge = int(len(addition_indexes) * self.control_size)

        result_group = ["A" if i < edge else "B" for i in addition_indexes]
        data = self._set_value(data, result_group)

        return data


class AASplitterWithGrouping(AASplitter):
    def execute(self, data: ExperimentData):
        group_field = data.get_columns_by_roles(GroupingRole)
        groups = list(data.groupby(group_field))
        edge = len(groups) // 2
        result = None
        for i, group in enumerate(groups):
            group_ds = Dataset(roles=[GroupingRole, TreatmentRole]).from_dict(
                [{"group_for_split": group[0], "group": "A" if i < edge else "B"}]
                * len(group[1]),
                index=group[1].index,
            )
            result = group_ds if result is None else result.append(group_ds)
        
        # TODO: check index
        self._set_value(data, result["group"])
        return data


class AASplitterWithStratification(SplitterAA):
    def __init__(
        self,
        control_size: float = 0.5,
        random_state: int = None,
        full_name: str = None,
        index: int = 0,
    ):
        super().__init__(control_size, random_state, full_name, index)

    def execute(self, data):
        control_indexes = []
        test_indexes = []
        stratification_columns = data.get_columns_by_roles(StratificationRole)

        groups = data.groupby(stratification_columns)
        result = None
        for _, gd in groups:
            ged = ExperimentData(gd)
            ged = self.super().execute(ged)
            
            result = ged if result is None else result.append(ged)
        # TODO: check index
        self._set_value(data, result["group"])
        return data

# As idea
# class SplitterAAMulti(ExperimentMulti):
#     def execute(self, data):
#         raise NotImplementedError

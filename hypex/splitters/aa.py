from typing import Dict, Any, List, Optional

from hypex.dataset.dataset import ExperimentData, Dataset
from hypex.dataset.roles import GroupingRole, StratificationRole, TreatmentRole
from hypex.experiments.base import Executor, ComplexExecutor
from hypex.transformers.shuffle import Shuffle
from hypex.utils.enums import ExperimentDataEnum


import logging
import funcy

logger = logging.getLogger(__name__)
f_handler = logging.FileHandler(f"{__name__}.log")
f_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
logger.addHandler(f_handler)
logger.setLevel(logging.DEBUG)


class AASplitter(ComplexExecutor):
    def __init__(
        self,
        control_size: float = 0.5,
        random_state: Optional[int] = None,
        full_name: Optional[str] = None,
        constant_key: bool = False,
        key: Any = "",
        inner_executors: Optional[Dict[str, Executor]] = None,
    ):
        self.control_size = control_size
        self.random_state = random_state
        self.default_inner_executors = {"shuffle": Shuffle(self.random_state)}
        self._key = key
        self.constant_key = constant_key
        super().__init__(inner_executors, full_name, key)
    
    @property
    def key(self) -> Any:
        return self._key
    
    @key.setter
    def key(self, value: Any):
        if not self.constant_key:
            self._key = value
            self._generate_id()

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

    @funcy.log_durations(logger.debug)
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

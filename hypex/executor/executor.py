from abc import abstractmethod, ABC
from copy import copy
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple, Iterable

from .executor_state import ExecutorState, DatasetSpace
from ..dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    GroupingRole,
    FeatureRole,
    AdditionalMatchingRole,
    TargetRole,
    DatasetAdapter,
)
from ..dataset.roles import AdditionalRole
from ..utils import (
    AbstractMethodError,
    ID_SPLIT_SYMBOL,
    SetParamsDictTypes,
    ExperimentDataEnum,
    NotSuitableFieldError,
)
from ..utils.adapter import Adapter


class Executor(ABC):
    def _get_params_dict(self):
        return {k: str(v) for k, v in copy(self.__dict__).items()}

    def __init__(self, key: str = "", save_space: Optional[DatasetSpace] = None):
        self._state = ExecutorState(
            self.__class__.__name__, self._get_params_dict(), key, save_space
        )

    @property
    def state(self):
        return self._state

    def refresh_state(
        self, key: Optional[str] = None, save_space: Optional[DatasetSpace] = None
    ):
        if key is not None:
            self._state.key = key
        if save_space is not None:
            self._state.save_space = save_space
        self._state.set_params(self._get_params_dict())

    def check_and_setattr(self, params: Dict[str, Any]):
        for key, value in params.items():
            if key in self.__dir__():
                setattr(self, key, value)

    def set_params(self, params: SetParamsDictTypes) -> None:
        if isinstance(list(params)[0], str):
            self.check_and_setattr(params)
        elif isinstance(list(params)[0], type):
            for executor_class, class_params in params.items():
                if isinstance(self, executor_class):
                    self.check_and_setattr(class_params)
        else:
            raise ValueError(
                "params must be a dict of str to dict or a dict of class to dict"
            )
        self.refresh_state()

    def refresh_from_state(self, state: [ExecutorState, str]) -> None:
        self._state = ExecutorState.create_from_str(str(state))
        self.check_and_setattr(self._state.get_params_dict())

    @property
    def _is_transformer(self) -> bool:
        return False

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        # defined in order to avoid  unnecessary redefinition in classes like transformer
        return data

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        raise AbstractMethodError


class Calculator(Executor, ABC):
    @classmethod
    def calc(cls, data: Dataset, **kwargs):
        return cls._inner_function(data, **kwargs)

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Any:
        raise AbstractMethodError

    @staticmethod
    def _check_test_data(
        test_data: Optional[Dataset] = None,
    ) -> Dataset:  # TODO to move away from Calculator. Where to?
        if test_data is None:
            raise ValueError("test_data is needed for comparison")
        return test_data


class MLExecutor(Calculator, ABC):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_role: Optional[ABCRole] = None,
        key: Any = "",
    ):
        self.target_role = target_role or TargetRole()
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()

    def _get_fields(self, data: ExperimentData):
        group_field = data.field_search(self.grouping_role)
        target_field = data.field_search(
            self.target_role, search_types=self.search_types
        )
        return group_field, target_field

    @abstractmethod
    def fit(self, X: Dataset, Y: Optional[Dataset] = None) -> "MLExecutor":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Dataset) -> Dataset:
        raise NotImplementedError

    def score(self, X: Dataset, Y: Dataset) -> float:
        raise NotImplementedError

    @property
    def search_types(self):
        return [int, float]

    @classmethod
    @abstractmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        target_data: Optional[Dataset] = None,
        **kwargs,
    ) -> Any:
        raise AbstractMethodError

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_field: Optional[str] = None,
        **kwargs,
    ) -> Any:
        if target_field:
            return cls._inner_function(
                data=grouping_data[0][1].drop(target_field),
                target_data=grouping_data[0][1][target_field],
                test_data=grouping_data[1][1].drop(target_field),
                **kwargs,
            )
        return cls._inner_function(
            data=grouping_data[0][1],
            test_data=grouping_data[1][1],
            **kwargs,
        )

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.additional_fields,
            self.state,
            value=value,
            key=key,
            role=AdditionalMatchingRole(),
        )

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Union[Sequence[str], str, None] = None,
        grouping_data: Optional[List[Tuple[str, Dataset]]] = None,
        target_field: Union[str, List[str], None] = None,
        features_fields: Union[str, List[str], None] = None,
        **kwargs,
    ) -> Dataset:
        group_field = Adapter.to_list(group_field)
        features_fields = Adapter.to_list(features_fields)
        if grouping_data is None:
            grouping_data = data.groupby(group_field, fields_list=features_fields)
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise NotSuitableFieldError(group_field, "Grouping")
        result = cls._execute_inner_function(
            grouping_data, target_field=target_field, **kwargs
        )
        return DatasetAdapter.to_dataset(
            result,
            {i: AdditionalMatchingRole() for i in list(result.keys())},
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data=data)
        features_fields = data.ds.search_columns(
            FeatureRole(), search_types=self.search_types
        )
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data
        if group_field[0] in data.groups:  # TODO: to recheck if this is a correct check
            grouping_data = list(data.groups[group_field[0]].items())
        else:
            grouping_data = None
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            grouping_data=grouping_data,
            target_fields=target_fields,
            features_fields=features_fields,
        )
        return self._set_value(data, compare_result)


class IfExecutor(Executor, ABC):
    def __init__(
        self,
        if_executor: Optional[Executor] = None,
        else_executor: Optional[Executor] = None,
        key: Any = "",
    ):
        self.if_executor = if_executor
        self.else_executor = else_executor
        super().__init__(key)

    @abstractmethod
    def check_rule(self, data, **kwargs) -> bool:
        raise AbstractMethodError

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.variables, self.id, value, key="response"
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        if self.check_rule(data):
            return (
                self.if_executor.execute(data)
                if self.if_executor is not None
                else self._set_value(data, True)
            )
        return (
            self.else_executor.execute(data)
            if self.else_executor is not None
            else self._set_value(data, False)
        )

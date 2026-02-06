from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..dataset import (
    ABCRole,
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    GroupingRole,
    TargetRole,
)
from ..utils import (
    ID_SPLIT_SYMBOL,
    AbstractMethodError,
    ExperimentDataEnum,
    NotSuitableFieldError,
    SetParamsDictTypes,
)
from ..utils.adapter import Adapter


class Executor(ABC):
    def __init__(
        self,
        key: Any = "",
        **calc_kwargs,
    ):
        self._id: str = ""
        self._params_hash = ""

        self.key: Any = key
        self._generate_id()
        self.calc_kwargs = calc_kwargs

    def check_and_setattr(self, params: dict[str, Any]):
        for key, value in params.items():
            if key in self.__dir__():
                setattr(self, key, value)

    def _generate_params_hash(self):
        self._params_hash = ""

    def _generate_id(self):
        self._generate_params_hash()
        self._id = ID_SPLIT_SYMBOL.join(
            [
                self.__class__.__name__,
                self._params_hash.replace(ID_SPLIT_SYMBOL, "|"),
                str(self._key).replace(ID_SPLIT_SYMBOL, "|"),
            ]
        )

    def set_params(self, params: SetParamsDictTypes) -> None:
        if isinstance(next(iter(params)), str):
            self.check_and_setattr(params)
        elif isinstance(next(iter(params)), type):
            for executor_class, class_params in params.items():
                if isinstance(self, executor_class):
                    self.check_and_setattr(class_params)
        else:
            raise ValueError(
                "params must be a dict of str to dict or a dict of class to dict"
            )
        self._generate_id()

    def init_from_hash(self, hash: str) -> None:
        self._params_hash = hash
        self._generate_id()

    @classmethod
    def build_from_id(cls, executor_id: str):
        splitted_id = executor_id.split(ID_SPLIT_SYMBOL)
        if splitted_id[0] != cls.__name__:
            raise ValueError(f"{executor_id} is not a valid {cls.__name__} id")
        result = cls()
        result.init_from_hash(splitted_id[1])
        return result

    @property
    def id(self) -> str:
        return self._id

    @property
    def key(self) -> Any:
        return self._key

    @key.setter
    def key(self, value: Any):
        self._key = value
        self._generate_id()

    @property
    def params_hash(self) -> str:
        return self._params_hash

    @property
    def id_for_name(self) -> str:
        return self.id.replace(ID_SPLIT_SYMBOL, "_")

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

    @property
    def search_types(self):
        raise AbstractMethodError

    @staticmethod
    def _check_test_data(
        test_data: Dataset | None = None,
    ) -> Dataset:  # TODO to move away from Calculator. Where to?
        if test_data is None:
            raise ValueError("test_data is needed for comparison")
        return test_data


class MLExecutor(Calculator, ABC):
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_role: ABCRole | None = None,
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
    def fit(self, X: Dataset, Y: Dataset | None = None) -> MLExecutor:
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
        test_data: Dataset | None = None,
        target_data: Dataset | None = None,
        **kwargs,
    ) -> Any:
        raise AbstractMethodError

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_field: str | None = None,
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
        for i in range(value.shape[1]):
            data.set_value(
                ExperimentDataEnum.additional_fields,
                f"{self.id}{ID_SPLIT_SYMBOL}{i}",
                value=value.iloc[:, i],
                key=key,
                role=AdditionalMatchingRole(),
            )
        return data

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Sequence[str] | str | None = None,
        grouping_data: list[tuple[str, Dataset]] | None = None,
        target_field: str | list[str] | None = None,
        features_fields: str | list[str] | None = None,
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
        return result

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
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
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
        # TODO: add roles to compare_result
        return self._set_value(data, compare_result)


class IfExecutor(Executor, ABC):
    def __init__(
        self,
        if_executor: Executor | None = None,
        else_executor: Executor | None = None,
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

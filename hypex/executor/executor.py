from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..dataset import (
    Dataset,
    ExperimentData,
)
from ..utils import (
    AbstractMethodError,
    ExperimentDataEnum,
    SetParamsDictTypes,
)
from ..utils import ID_SPLIT_SYMBOL


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

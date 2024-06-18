from abc import ABC, abstractmethod
from typing import Any, Optional

from hypex.dataset import Dataset, ExperimentData, ABCRole
from hypex.utils import ID_SPLIT_SYMBOL, AbstractMethodError


class Executor(ABC):
    def __init__(
        self,
        key: Any = "",
        random_state: Optional[int] = None,
    ):
        self._id: str = ""
        self._params_hash = ""
        self.random_state = random_state

        self.key: Any = key
        self.refresh_params_hash()

    def _generate_params_hash(self):
        self._params_hash = ""

    def _generate_id(self):
        self._id = ID_SPLIT_SYMBOL.join(
            [
                self.__class__.__name__,
                self.params_hash.replace(ID_SPLIT_SYMBOL, "|"),
                str(self._key).replace(ID_SPLIT_SYMBOL, "|"),
            ]
        )

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

    def refresh_params_hash(self):
        self._generate_params_hash()
        self._generate_id()

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

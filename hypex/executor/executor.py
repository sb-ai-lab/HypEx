from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from hypex.dataset import Dataset, ExperimentData
from hypex.utils import ID_SPLIT_SYMBOL, AbstractMethodError


class Executor(ABC):
    def __init__(
        self,
        full_name: Optional[str] = None,
        key: Any = "",
        random_state: Optional[int] = None,
    ):
        self._id: str = ""
        self._params_hash = ""
        self.full_name = full_name
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
    def full_name(self) -> Union[str, None]:
        return self._full_name

    @full_name.setter
    def full_name(self, value: Optional[str]):
        self._full_name: str = str(value or self.__class__.__name__)

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
        return data

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        raise AbstractMethodError


class Calculator(Executor, ABC):
    @staticmethod
    @abstractmethod
    def calc(data: Dataset, **kwargs):
        raise AbstractMethodError

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Any, Union, Dict

from hypex.dataset import ExperimentData, Dataset
from hypex.utils import ID_SPLIT_SYMBOL


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
    def calc(self, data: Dataset):
        raise NotImplementedError

    def execute(self, data: ExperimentData) -> ExperimentData:
        value = self.calc(data)
        return self._set_value(data, value)


class ComplexExecutor(Executor, ABC):
    default_inner_executors: Dict[str, Executor] = {}

    def _get_inner_executors(
        self, inner_executors: Optional[Dict[str, Executor]] = None
    ) -> Dict[str, Executor]:
        result = {}
        inner_executors = inner_executors or {}
        for key, executor in self.default_inner_executors.items():
            if key not in inner_executors:
                if len(inner_executors):
                    warnings.warn(
                        f"{key} executor not found in inner_executors. Will {key} will be used by default."
                    )
                result[key] = executor
            else:
                result[key] = inner_executors[key]
        return result

    def __init__(
        self,
        inner_executors: Optional[Dict[str, Executor]] = None,
        full_name: Optional[str] = None,
        key: Any = "",
    ):
        super().__init__(full_name=full_name, key=key)
        self.inner_executors = self._get_inner_executors(inner_executors)

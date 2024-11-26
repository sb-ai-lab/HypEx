from enum import Enum
from typing import Union, Optional, Dict, Any


class DatasetSpace(Enum):
    data = "data"
    additional_fields = "additional_fields"
    variables = "variables"
    groups = "groups"
    analysis_tables = "analysis_tables"


class ExecutorState:
    SPLIT_SYMBOL = "\u256B"
    NAME_BORDER_SYMBOL = "\u2506"
    PARAMETER_SPLIT_SYMBOL = "\u2534"

    @staticmethod
    def dict_to_str(new_parameters):
        return "".join(f"{key}:{value}" for key, value in new_parameters.items())

    def set_params(self, new_parameters: Union[str, dict, None]):
        if new_parameters is None:
            self.parameters = ""
        if isinstance(new_parameters, str):
            self.parameters = new_parameters
        elif isinstance(new_parameters, dict):
            self.parameters = self.dict_to_str(new_parameters)

    def set_executor(self, executor: Union[str, type]):
        self.executor = (
            executor.__class__.__name__ if isinstance(executor, type) else executor
        )

    def __init__(
        self,
        executor: Union[str, type],
        parameters: Union[str, dict, None] = None,
        key: Optional[str] = None,
        *,
        save_space: Optional[DatasetSpace] = None,
    ):
        self.executor = (
            executor.__class__.__name__ if isinstance(executor, type) else executor
        )
        self.executor = None
        self.parameters = None

        self.set_executor(executor)
        self.key = key
        self.save_space = save_space
        self.set_params(parameters)

    def __str__(self):
        return f"{self.executor}{self.SPLIT_SYMBOL}{self.parameters}{self.SPLIT_SYMBOL}{self.key}"

    @staticmethod
    def create_from_str(string: str) -> "ExecutorState":
        return ExecutorState(*string.split(ExecutorState.SPLIT_SYMBOL))

    def get_params_dict(self) -> Dict[str, Any]:
        return {
            param[: param.find(":")]: param[param.find(":") + 1 :]
            for param in self.parameters.split(self.PARAMETER_SPLIT_SYMBOL)
        }

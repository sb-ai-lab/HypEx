from enum import Enum
from typing import Union, Optional


class DatasetSpace(Enum):
    data = "data"
    additional_fields = "additional_fields"
    variables = "variables"
    groups = "groups"
    analysis_tables = "analysis_tables"


class ExecutorState:
    def __init__(
        self,
        executor: Union[str, type],
        parameters: Union[str, dict, None] = None,
        key: Optional[str] = None,
        save_space: Optional[DatasetSpace] = None,
    ):
        self.executor = (
            executor.__class__.__name__ if isinstance(executor, type) else executor
        )
        self.parameters = parameters
        self.key = key
        self.save_space = save_space

from copy import deepcopy
from typing import Any, Union, Iterable, Dict, List, Optional

from ..dataset import Dataset
from ..utils import BackendsEnum
from ..executor.executor_state import DatasetSpace, ExecutorState


class ExperimentData:
    def __init__(self, data: Dataset):
        self._data = {
            "data": data,
        }
        self._index = {
            "additional_fields": {},
            "variables": {},
            "groups": {},
            "analysis_tables": {},
        }

    @property
    def ds(self):
        """
        Returns the dataset property.

        Returns:
            The dataset property.
        """
        return self._data["data"]

    @staticmethod
    def create_empty(
        roles=None, backend=BackendsEnum.pandas, index=None
    ) -> "ExperimentData":
        """Creates an empty ExperimentData object"""
        ds = Dataset.create_empty(backend, roles, index)
        return ExperimentData(ds)

    def set_value(
        self,
        executor_state: ExecutorState,
        value: Any,
    ) -> "ExperimentData":
        def refresh_index_structure():
            if executor_state.save_space is None:
                raise ValueError(
                    "The ExecutorState must contain an indication of the save_space"
                )
            if executor_state.executor not in self._index:  # add executor in index
                self._index[executor_state.save_space][
                    executor_state.executor
                ] = executor_state.get_dict_for_index()
            elif (
                executor_state.parameters
                not in self._index[executor_state.save_space][executor_state.executor]
            ):  # add executor parameters in index
                self._index[executor_state.save_space][executor_state.executor][
                    executor_state.parameters
                ].update(executor_state.get_dict_for_index())

        refresh_index_structure()
        self._data[str(executor_state)] = value
        self._index[executor_state.save_space][executor_state.executor][
            executor_state.parameters
        ][executor_state.key] = str(executor_state)
        return self

    def search_state(
        self,
        space: Union[Iterable[DatasetSpace], DatasetSpace, None] = None,
        executor: Union[type, str, None] = None,
        parameters: Union[Dict[str, Any], str, None] = None,
        key: Union[str, None] = None,
    ) -> List[ExecutorState]:
        """
        After each step of the search takes place at a lower level.
        """
        def space_step():
            if space is None:
                return self._index.keys()
            return [space] if isinstance(space, DatasetSpace) else space

        def executor_step(spaces):
            executors = {s: list(self._index[s].keys()) for s in spaces}
            if executor is None:
                return executors
            str_executor = executor if isinstance(executor, str) else executor.__name__
            executors = {
                s: [e for e in executors[s] if e == str_executor] for s in executors
            }
            return {s: e for s, e in executors.items() if len(e) > 0}

        def parameters_step(executors):
            if not executors:
                return None

            params = {
                s: {e: list(self._index[s][e].keys()) for e in executors[s]}
                for s in executors
            }
            if parameters is None:
                return params
            param_str = (
                parameters
                if isinstance(parameters, str)
                else ExecutorState.dict_to_str(parameters)
            )

            return {
                s: {e: [p for p in params[s][e] if p == param_str] for e in params[s]}
                for s in params
            }

        def key_step(params):
            if not params:
                return None

            executors = []
            for s in params:
                for e in params[s]:
                    for p in params[s][e]:
                        if key is None:
                            for k in self._index[s][e][p]:
                                executors.append(self._index[s][e][p][k])
                        elif key in self._index[s][e][p]:
                            executors.append(self._index[s][e][p][key])
            return executors

        #-----------------------------------------------
        result = key_step(parameters_step(executor_step(space_step())))
        return result or []

    def copy(self, data: Optional[Dataset] = None) -> "ExperimentData":
        result = deepcopy(self)
        if data is not None:
            result._data = data
        return result


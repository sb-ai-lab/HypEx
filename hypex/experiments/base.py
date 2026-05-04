from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Any

from ..dataset import ABCRole, AdditionalTargetRole, ExperimentData, TempTargetRole, Dataset
from ..executor import Executor
from ..utils import ExperimentDataEnum
from ..utils.registry import backend_factory

import time
import inspect



class Experiment(Executor):
    def _detect_transformer(self) -> bool:
        return any(executor._is_transformer for executor in self.executors)

    def get_executor_ids(
        self, searched_classes: type | Iterable[type] | None = None
    ) -> dict[type, list[str]]:
        if not searched_classes:
            return {}

        searched_classes = (
            searched_classes
            if isinstance(searched_classes, Iterable)
            else [searched_classes]
        )
        return {
            searched_class: [
                executor.id
                for executor in self.executors
                if isinstance(executor, searched_class)
            ]
            for searched_class in searched_classes
        }

    def __init__(
        self,
        executors: Sequence[Executor],
        transformer: bool | None = None,
        key: Any = "",
    ):
        self.executors: Sequence[Executor] = executors
        self.transformer: bool = (
            transformer if transformer is not None else self._detect_transformer()
        )
        super().__init__(key)

    def set_params(self, params: dict[str, Any] | dict[type, dict[str, Any]]) -> None:
        if isinstance(next(iter(params)), str):
            super().set_params(params)
        elif isinstance(next(iter(params)), type):
            for executor in self.executors:
                executor.set_params(params)
        else:
            raise ValueError(
                "params must be a dict of str to dict or a dict of class to dict"
            )

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(ExperimentDataEnum.analysis_tables, self.id, value)
    
    @staticmethod
    def _get_executor_backend(executor: Executor, ds: Dataset):
        """
        Class for selecting backend-dependent realization for direct executor
        """
        executor_cls = type(executor)
        # print(executor_cls.__name__, type(ds.backend_data))
        backend_cls = backend_factory.resolve_backend(executor_cls, ds)
        if backend_cls is None:
             return executor
        # else:
        #     executor_params = getattr(executor, 'experiment_kwargs', {})
        #     cur_executer = executor_dep_on_backend(**executor_params)
        
        # return cur_executer
        sig = inspect.signature(backend_cls.__init__)
        expected_params = {p.name for p in sig.parameters.values() if p.name != 'self'}

        init_kwargs = {k: getattr(executor, k) for k in expected_params if hasattr(executor, k)}

        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_keyword and hasattr(executor, 'calc_kwargs'):
            init_kwargs['calc_kwargs'] = executor.calc_kwargs

        new_executor = backend_cls(**init_kwargs)

        if hasattr(executor, 'key'):
            new_executor.key = executor.key

        return new_executor
            

    def execute(self, data: ExperimentData) -> ExperimentData:
        experiment_data = deepcopy(data) if self.transformer else data # TODO: fix `deepcopy` issue 
        # experiment_data = data
        for executor in self.executors:
            start = time.perf_counter()
            cur_executor = self._get_executor_backend(executor, experiment_data.ds)
            cur_executor.key = self.key #TODO: do we need to send here slave-backend class key?
            print(cur_executor.__class__.__name__)
            experiment_data = cur_executor.execute(experiment_data)
            end = time.perf_counter()

            print(f"executor.key = {executor.id}; dt = {end - start:.4f}c")
            
        return experiment_data


class OnRoleExperiment(Experiment):
    def __init__(
        self,
        executors: list[Executor],
        role: ABCRole | Sequence[ABCRole],
        transformer: bool | None = None,
        key: Any = "",
    ):
        self.role: list[ABCRole] = [role] if isinstance(role, ABCRole) else list(role)
        super().__init__(executors, transformer, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        for field in data.field_search(self.role):
            if field in data.ds.columns:
                data.ds.tmp_roles = {field: TempTargetRole()}
            elif field in data.additional_fields.columns:
                data.additional_fields.tmp_roles = {field: AdditionalTargetRole()}
            data = super().execute(data)
            data.ds.tmp_roles = {}
            data.additional_fields.tmp_roles = {}
        return data

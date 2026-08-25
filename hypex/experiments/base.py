from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Any

from ..dataset import ABCRole, AdditionalTargetRole, ExperimentData, TempTargetRole, Dataset
from ..executor import Executor
from ..utils import ExperimentDataEnum, HypExLogger
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

        # Создаем логгер для эксперимента
        self.logger = HypExLogger(
            name="hypex.experiment",
            level="INFO",
            log_file="experiment.log",
            console_out=False
        )

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
        backend_cls = backend_factory.resolve_backend(executor_cls, ds)
        if backend_cls is None:
             return executor

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
        # Логируем информацию о Spark-сессии
        self.logger.log_spark_info()

        experiment_data = deepcopy(data) if self.transformer else data
        for executor in self.executors:
            # start = time.perf_counter()
            # cur_executor = self._get_executor_backend(executor, experiment_data.ds)
            # cur_executor.key = self.key 
            # experiment_data = cur_executor.execute(experiment_data)
            # # end = time.perf_counter()
            # # if logg_file is not None:
            # #     with open(logg_file, "a") as f:
            # #         f.write(f"{type(cur_executor).__name__}: {end - start} sec\n")
            
            with self.logger.process(
                name=executor.__class__.__name__,
                backend=experiment_data.ds.backend_type.value,
                log_spark=False  # можно включить для детального логирования Spark-процессов
            ):
                cur_executor = self._get_executor_backend(executor, experiment_data.ds)
                cur_executor.key = self.key
                experiment_data = cur_executor.execute(experiment_data)
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

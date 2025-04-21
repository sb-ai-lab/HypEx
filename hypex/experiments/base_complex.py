from __future__ import annotations

from itertools import product
from typing import Any, Sequence

from tqdm import tqdm

from ..dataset import ABCRole, Dataset, ExperimentData, GroupingRole
from ..executor import Executor, IfExecutor
from ..reporters import DatasetReporter, Reporter
from ..utils.enums import ExperimentDataEnum
from .base import Experiment


class ExperimentWithReporter(Experiment):
    def __init__(
        self,
        executors: Sequence[Executor],
        reporter: Reporter,
        transformer: bool | None = None,
        key: str = "",
    ):
        super().__init__(executors, transformer, key)
        self.reporter = reporter

    def one_iteration(
        self, data: ExperimentData, key: str = "", set_key_as_index: bool = False
    ):
        t_data = ExperimentData(data.ds)
        self.key = key
        t_data = super().execute(t_data)
        result = self.reporter.report(t_data)
        if set_key_as_index:
            result.index = [key]
        return result

    def _set_result(
        self, data: ExperimentData, result: list[Dataset], reset_index: bool = True
    ):
        result = (
            result[0].append(result[1:], reset_index=reset_index)
            if len(result) > 1
            else result[0]
        )
        return self._set_value(data, result)


class CycledExperiment(ExperimentWithReporter):
    def __init__(
        self,
        executors: list[Executor],
        reporter: DatasetReporter,
        n_iterations: int,
        transformer: bool | None = None,
        key: str = "",
    ):
        super().__init__(executors, reporter, transformer, key)
        self.n_iterations: int = n_iterations

    def generate_params_hash(self) -> str:
        return f"{self.reporter.__class__.__name__} x {self.n_iterations}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        result: list[Dataset] = [
            self.one_iteration(data, str(i)) for i in tqdm(range(self.n_iterations))
        ]
        return self._set_result(data, result)


class GroupExperiment(ExperimentWithReporter):
    def __init__(
        self,
        executors: Sequence[Executor],
        reporter: Reporter,
        searching_role: ABCRole = GroupingRole(),
        transformer: bool | None = None,
        key: str = "",
    ):
        self.searching_role = searching_role
        super().__init__(executors, reporter, transformer, key)

    def generate_params_hash(self) -> str:
        return f"GroupExperiment: {self.reporter.__class__.__name__}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field = data.ds.search_columns(self.searching_role)
        result: list[Dataset] = [
            self.one_iteration(
                ExperimentData(group_data), str(group[0]), set_key_as_index=True
            )
            for group, group_data in tqdm(data.ds.groupby(group_field))
        ]
        return self._set_result(data, result, reset_index=False)


class ParamsExperiment(ExperimentWithReporter):
    def __init__(
        self,
        executors: Sequence[Executor],
        reporter: DatasetReporter,
        params: dict[type, dict[str, Sequence[Any]]],
        transformer: bool | None = None,
        key: str = "",
    ):
        super().__init__(executors, reporter, transformer, key)
        self._params = params
        self._flat_params: list[dict[type, dict[str, Any]]] = []

    def generate_params_hash(self) -> str:
        return f"ParamsExperiment: {self.reporter.__class__.__name__}"

    def _update_flat_params(self):
        classes = list(self._params)
        param_combinations = [
            list(
                product(
                    *[
                        product([parameter], values)
                        for parameter, values in class_params.items()
                    ]
                )
            )
            for class_params in self._params.values()
        ]
        new_flat_params = [
            {
                classes[i]: dict(param_combination[i])
                for i in range(len(param_combination))
            }
            for param_combination in product(*param_combinations)
        ]
        self._flat_params = new_flat_params

    @property
    def flat_params(self) -> list[dict[type, dict[str, Any]]]:
        return self._flat_params

    @property
    def params(self) -> dict[type, dict[str, Sequence[Any]]]:
        return self._params

    @params.setter
    def params(self, params: dict[type, dict[str, Sequence[Any]]]):
        self._params = params
        self._update_flat_params()

    def execute(self, data: ExperimentData) -> ExperimentData:
        results = []
        self._update_flat_params()
        for flat_param in tqdm(self._flat_params):
            t_data = ExperimentData(data.ds)
            for executor in self.executors:
                executor.set_params(flat_param)
                t_data = executor.execute(t_data)
            report = self.reporter.report(t_data)
            results.append(report)
        return self._set_result(data, results)


class IfParamsExperiment(ParamsExperiment):
    def __init__(
        self,
        executors: Sequence[Executor],
        reporter: DatasetReporter,
        params: dict[type, dict[str, Sequence[Any]]],
        stopping_criterion: IfExecutor,
        transformer: bool | None = None,
        key: str = "",
    ):
        self.stopping_criterion = stopping_criterion
        super().__init__(executors, reporter, params, transformer, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        self._update_flat_params()
        for flat_param in tqdm(self._flat_params):
            t_data = ExperimentData(data.ds)
            for executor in self.executors:
                executor.set_params(flat_param)
                t_data = executor.execute(t_data)
            if_result = self.stopping_criterion.execute(t_data)
            if_executor_id = if_result.get_one_id(
                self.stopping_criterion.__class__, ExperimentDataEnum.variables
            )
            if if_result.variables[if_executor_id]["response"]:
                return self._set_result(data, [self.reporter.report(t_data)])
        return data

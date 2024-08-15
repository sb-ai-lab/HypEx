from itertools import product
from typing import Optional, List, Dict, Sequence, Any

from tqdm.auto import tqdm

from hypex.dataset import ExperimentData, Dataset
from hypex.dataset import TempGroupingRole
from hypex.executor import Executor
from hypex.experiments.base import Experiment
from hypex.reporters import Reporter, DatasetReporter


class ExperimentWithReporter(Experiment):
    def __init__(
        self,
        executors: Sequence[Executor],
        reporter: Reporter,
        transformer: Optional[bool] = None,
        key: str = "",
    ):
        super().__init__(executors, transformer, key)
        self.reporter = reporter

    def one_iteration(self, data: ExperimentData, key: str = ""):
        t_data = ExperimentData(data.ds)
        self.key = key
        t_data = super().execute(t_data)
        return self.reporter.report(t_data)

    def _set_result(self, data: ExperimentData, result: List[Dataset]):
        result = result[0].append(result[1:], True)
        return self._set_value(data, result)


class CycledExperiment(ExperimentWithReporter):
    def __init__(
        self,
        executors: List[Executor],
        reporter: DatasetReporter,
        n_iterations: int,
        transformer: Optional[bool] = None,
        key: str = "",
    ):
        super().__init__(executors, reporter, transformer, key)
        self.n_iterations: int = n_iterations

    def generate_params_hash(self) -> str:
        return f"{self.reporter.__class__.__name__} x {self.n_iterations}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        result: List[Dataset] = [
            self.one_iteration(data, str(i)) for i in tqdm(range(self.n_iterations))
        ]
        return self._set_result(data, result)


class GroupExperiment(ExperimentWithReporter):
    def generate_params_hash(self) -> str:
        return f"GroupExperiment: {self.reporter.__class__.__name__}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field = data.ds.search_columns(TempGroupingRole(), tmp_role=True)
        result: List[Dataset] = [
            self.one_iteration(group_data, str(group))
            for group, group_data in tqdm(data.ds.groupby(group_field))
        ]
        return self._set_result(data, result)


class ParamsExperiment(ExperimentWithReporter):
    def __init__(
        self,
        executors: Sequence[Executor],
        reporter: DatasetReporter,
        params: Dict[type, Dict[str, Sequence[Any]]],
        transformer: Optional[bool] = None,
        key: str = "",
    ):
        super().__init__(executors, reporter, transformer, key)
        self._params = params
        self._flat_params: List[Dict[type, Dict[str, Any]]] = []

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
    def flat_params(self) -> List[Dict[type, Dict[str, Any]]]:
        return self._flat_params

    @property
    def params(self) -> Dict[type, Dict[str, Sequence[Any]]]:
        return self._params

    @params.setter
    def params(self, params: Dict[type, Dict[str, Sequence[Any]]]):
        self._params = params
        self._update_flat_params()

    def execute(self, data: ExperimentData) -> ExperimentData:
        results = []
        self._update_flat_params()
        for flat_param in self._flat_params:
            t_data = ExperimentData(data.ds)
            for executor in self.executors:
                executor.set_params(flat_param)
                t_data = executor.execute(t_data)
            report = self.reporter.report(t_data)
            results.append(report)
        return self._set_result(data, results)

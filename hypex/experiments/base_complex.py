from itertools import product
from typing import Optional, List, Dict, Sequence, Any

from tqdm.auto import tqdm

from hypex.dataset import ExperimentData, Dataset, ABCRole
from hypex.executor import Executor
from hypex.experiments import Experiment
from hypex.reporters import Reporter, DatasetReporter


class ExperimentWithReporter(Experiment):
    def __init__(
        self,
        executors: list[Executor],
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
        result = result[0].append(result[1:], range(self.n_iterations))
        return self._set_value(data, result)


class CycledExperiment(ExperimentWithReporter):
    def __init__(
        self,
        executors: list[Executor],
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
    def __init__(
        self,
        executors: list[Executor],
        reporter: DatasetReporter,
        grouping_role: Optional[ABCRole] = None,
        transformer: Optional[bool] = None,
        key: str = "",
    ):
        super().__init__(executors, reporter, transformer, key)
        self.grouping_role = grouping_role

    def generate_params_hash(self) -> str:
        return f"GroupExperiment: {self.reporter.__class__.__name__}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field = data.ds.search_columns(self.grouping_role)
        result = [
            self.one_iteration(ExperimentData(group_data), str(group))
            for group, group_data in tqdm(data.ds.groupby(group_field))
        ]
        return self._set_result(data, result)


class ParamsExperiment(ExperimentWithReporter):
    def __init__(
        self,
        executors: list[Executor],
        reporter: DatasetReporter,
        parameters: Dict[type, Dict[str, Sequence[Any]]],
        transformer: Optional[bool] = None,
        key: str = "",
    ):
        super().__init__(executors, reporter, transformer, key)
        self._params = parameters
        self._flat_params: List[Dict[type, Dict[str, Any]]] = []

    def generate_params_hash(self) -> str:
        return f"ParamsExperiment: {self.reporter.__class__.__name__}"

    @property
    def parameters(self) -> Dict[type, Dict[str, Sequence[Any]]]:
        return self._params

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

    @parameters.setter
    def parameters(self, params: Dict[type, Dict[str, Sequence[Any]]]):
        self._params = params
        self._update_flat_params()

    def execute(self, data: ExperimentData) -> ExperimentData:
        results = []
        self._update_flat_params()
        for flat_param in self._flat_params:
            t_data = ExperimentData(data.ds)
            for executor in self.executors:
                for class_, params in flat_param.items():
                    if isinstance(executor, class_):
                        executor.set_params(params)
                t_data = executor.execute(t_data)
            report = self.reporter.report(t_data)
            results.append(report)
        return self._set_result(data, results)

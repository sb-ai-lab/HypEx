from itertools import product
from typing import Optional, List, Dict, Sequence, Any

from tqdm.auto import tqdm

from hypex.dataset import ExperimentData, Dataset, TempGroupingRole
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
    def generate_params_hash(self) -> str:
        return f"GroupExperiment: {self.reporter.__class__.__name__}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field = data.ds.search_columns(TempGroupingRole(), tmp_role=True)
        result: List[Dataset] = [
            self.one_iteration(group_data, str(group))
            for group, group_data in tqdm(data.ds.groupby(group_field))
        ]
        return self._set_result(data, result)


# params = {
#     AASplitter: {"random_state": range(5)},
#     GroupComparator: {
#         "grouping_role": [TreatmentRole(), TargetRole()],
#         "space": [SpaceEnum.additional],
#     },
# }
#
# flat_params = []
# param_combinations = []
# classes = list(params)
# for c, ps in params.items():
#     param_combinations.append(
#         list(product(*[product([p], vs) for p, vs in ps.items()]))
#     )
# for pc in product(*param_combinations):
#     flat_params.append({classes[i]: {p: v for p, v in pc[i]} for i in range(len(pc))})
# flat_params


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
        self._flat_params = []

    def generate_params_hash(self) -> str:
        return f"ParamsExperiment: {self.reporter.__class__.__name__}"

    @property
    def parameters(self) -> Dict[type, Dict[str, Sequence[Any]]]:
        return self._params

    def _update_flat_params(self):
        new_flat_params = []
        param_combinations = []
        classes = list(self._params)
        for class_params in self._params.values():
            param_combinations.append(
                list(
                    product(
                        *[
                            product([parameter], values)
                            for parameter, values in class_params.items()
                        ]
                    )
                )
            )
        for param_combination in product(*param_combinations):
            new_flat_params.append(
                {
                    classes[i]: {
                        parameter: value for parameter, value in param_combination[i]
                    }
                    for i in range(len(param_combination))
                }
            )
        self._flat_params = new_flat_params

    @property
    def flat_params(self) -> List[Dict[type, Dict[str, Any]]]:
        return self._flat_params

    @parameters.setter
    def parameters(self, params: Dict[type, Dict[str, Sequence[Any]]]):
        self._params = params
        self._update_flat_params()

    def execute(self, data: ExperimentData) -> ExperimentData:
        pass

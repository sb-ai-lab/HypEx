from copy import deepcopy
from itertools import product
from typing import Optional, List, Dict, Sequence, Any, Iterable, Literal

from tqdm.auto import tqdm

from ..comparators import Comparator
from ..dataset import (
    ExperimentData,
    Dataset,
    GroupingRole,
    ABCRole,
    FeatureRole,
    TargetRole,
)
from ..executor import Executor, IfExecutor
from .base import Experiment
from ..extensions.abstract import MLExtension
from ..reporters import Reporter, DatasetReporter
from ..utils.enums import ExperimentDataEnum


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
        self, data: ExperimentData, result: List[Dataset], reset_index: bool = True
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
    def __init__(
        self,
        executors: Sequence[Executor],
        reporter: Reporter,
        searching_role: ABCRole = GroupingRole(),
        transformer: Optional[bool] = None,
        key: str = "",
    ):
        self.searching_role = searching_role
        super().__init__(executors, reporter, transformer, key)

    def generate_params_hash(self) -> str:
        return f"GroupExperiment: {self.reporter.__class__.__name__}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field = data.ds.search_columns(self.searching_role)
        result: List[Dataset] = [
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


class IfParamsExperiment(ParamsExperiment):
    def __init__(
        self,
        executors: Sequence[Executor],
        reporter: DatasetReporter,
        params: Dict[type, Dict[str, Sequence[Any]]],
        stopping_criterion: IfExecutor,
        transformer: Optional[bool] = None,
        key: str = "",
    ):
        self.stopping_criterion = stopping_criterion
        super().__init__(executors, reporter, params, transformer, key)

    def execute(self, data: ExperimentData) -> ExperimentData:
        self._update_flat_params()
        for flat_param in self._flat_params:
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


class MLExperiment(Executor):
    def __init__(
        self,
        ml_pipeline: Sequence[MLExtension],
        scores: Optional[Iterable[Comparator]] = None,
        feature_role: Optional[ABCRole] = None,
        target_role: Optional[ABCRole] = None,
        grouping_role: Optional[ABCRole] = None,
        key: Any = "",
    ):
        super().__init__(key)
        self.feature_role = feature_role or FeatureRole()
        self.target_role = target_role or TargetRole()
        self.grouping_role = grouping_role or GroupingRole()
        self.ml_pipeline = ml_pipeline
        self.scores = scores

    def _get_group_data(
        self, data: ExperimentData, group_label: Literal["train", "test", "validation"]
    ):
        group = group_label if group_label in data.groups[self.id] else ""
        if group not in data.groups[self.id]:
            raise ValueError(
                "Error in GroupingRole: ML group is one field contains values of 'train', 'test' or 'validation'"
            )
        return data.groups[self.id][group]

    def fit(self, data: ExperimentData, features, target):
        fit_data = self._get_group_data(data, "train")
        for ml_extension in self.ml_pipeline:
            ml_extension.fit(fit_data, features, target)
            data.set_value(
                ExperimentDataEnum.executors_states,
                self.id,
                ml_extension,
                ml_extension.__class__.__name__,
            )

    def transform(self, data):
        data.trans_ds = deepcopy(data.ds) if data.trans_ds is None else data.trans_ds
        for ml_extension in self.ml_pipeline:
            if isinstance(ml_extension, MLTransformer):
                data.trans_ds = ml_extension.transform(data.trans_ds)

    def predict(self, data):
        t_data = data.ds if data.trans_ds is None else data.trans_ds
        for ml_extension in self.ml_pipeline:
            if isinstance(ml_extension, MLPredictor):
                data.trans_ds = ml_extension.predict(data.trans_ds)

    def score(self, data):
        pass

    def execute(self, data: ExperimentData) -> ExperimentData:
        features = data.ds.search_columns(self.feature_role)
        target = data.ds.search_columns(self.target_role)
        group = data.ds.search_columns(self.grouping_role)
        if group > 0:
            for key, group in data.ds.groupby(group):
                data.set_value(ExperimentDataEnum.groups, self.id, group, key)
        else:
            data.set_value(ExperimentDataEnum.groups, self.id, deepcopy(data.ds), "")
        self.fit(data, features, target)
        self.transform(data)
        self.predict(data)
        self.score(data)
        return data

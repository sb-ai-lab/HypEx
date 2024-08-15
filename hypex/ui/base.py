from typing import Dict, Union, Any, Optional

from hypex.dataset import ExperimentData, Dataset
from hypex.experiments.base import Experiment
from hypex.reporters import Reporter
from hypex.utils import ID_SPLIT_SYMBOL
from hypex.utils.enums import RenameEnum


class Output:
    resume: Dataset

    def __init__(
        self,
        resume_reporter: Reporter,
        additional_reporters: Optional[Dict[str, Reporter]] = None,
    ):
        self.resume_reporter = resume_reporter
        self.additional_reporters = additional_reporters or {}

    def _extract_by_reporters(self, experiment_data: ExperimentData):
        self.resume = self.resume_reporter.report(experiment_data)
        for attribute, reporter in self.additional_reporters.items():
            setattr(self, attribute, reporter.report(experiment_data))

    @staticmethod
    def replace_splitters(
        data: Dataset, mode: RenameEnum = RenameEnum.columns
    ) -> Dataset:
        result = data
        if mode in (RenameEnum.all, RenameEnum.columns):
            result = result.rename(
                {c: c.replace(ID_SPLIT_SYMBOL, " ") for c in result.columns}
            )
        if mode in (RenameEnum.all, RenameEnum.index):
            result.index = [i.replace(ID_SPLIT_SYMBOL, " ") for i in result.index]
        return result

    def extract(self, experiment_data: ExperimentData):
        self._extract_by_reporters(experiment_data)


class ExperimentShell:
    def __init__(
        self,
        experiment: Experiment,
        output: Output,
        experiment_params: Optional[Dict[str, Any]] = None,
    ):
        if experiment_params:
            experiment.set_params(experiment_params)
        self._out = output
        self._experiment = experiment

    @property
    def experiment(self):
        return self._experiment

    def execute(self, data: Union[Dataset, ExperimentData]):
        if isinstance(data, Dataset):
            data = ExperimentData(data)
        result_experiment_data = self._experiment.execute(data)
        self._out.extract(result_experiment_data)
        return self._out

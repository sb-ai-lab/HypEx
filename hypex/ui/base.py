from __future__ import annotations

from typing import Any

from ..dataset import Dataset, ExperimentData
from ..experiments.base import Experiment
from ..reporters import Reporter
from ..utils import ID_SPLIT_SYMBOL
from ..utils.enums import RenameEnum


class Output:
    """A class for handling experiment output reporting and formatting.

    This class manages the reporting and formatting of experiment results, allowing for both
    a primary resume report and additional custom reports.

    Attributes:
        resume (Dataset): The main summary report of the experiment results.
        _experiment_data (ExperimentData): Internal storage of the experiment data.

    Args:
        resume_reporter (Reporter): The main reporter that generates the resume output.
        additional_reporters (Optional[Dict[str, Reporter]]): Dictionary mapping attribute
            names to additional reporters for custom reporting. Defaults to None.

    Examples
    --------
    .. code-block:: python

        # Basic usage with just a resume reporter
        from my_reporters import MyResumeReporter
        output = Output(resume_reporter=MyResumeReporter())
        output.extract(experiment_data)
        print(output.resume)

        # Using additional custom reporters
        from my_reporters import StatsReporter, PlotReporter
        additional = {
            'statistics': StatsReporter(),
            'plots': PlotReporter()
        }
        output = Output(
            resume_reporter=MyResumeReporter(),
            additional_reporters=additional
        )
        output.extract(experiment_data)
        print(output.statistics)  # Access additional report
        print(output.plots)  # Access additional report
    """

    resume: Dataset
    _experiment_data: ExperimentData

    def __init__(
        self,
        resume_reporter: Reporter,
        additional_reporters: dict[str, Reporter] | None = None,
    ):
        self.resume_reporter = resume_reporter
        self.additional_reporters = additional_reporters or {}

    def _extract_by_reporters(self, experiment_data: ExperimentData):
        """Extracts reports from all configured reporters.

        Args:
            experiment_data (ExperimentData): The experiment data to generate reports from.
        """
        self.resume = self.resume_reporter.report(experiment_data)
        for attribute, reporter in self.additional_reporters.items():
            setattr(self, attribute, reporter.report(experiment_data))
        self._experiment_data = experiment_data

    @staticmethod
    def _replace_splitters(
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
        """Extracts and processes all reports from the experiment data.

        Args:
            experiment_data (ExperimentData): The experiment data to generate reports from.

        Examples
        --------
        .. code-block:: python

            output = Output(resume_reporter=MyReporter())
            output.extract(experiment_data)
            print(output.resume)  # Access the main report
        """
        self._extract_by_reporters(experiment_data)


class ExperimentShell:
    """Base class for experiment execution with configurable output handling.

    This class provides a shell for executing experiments with customizable parameters
    and output formatting. It serves as a base class for specific experiment types
    like A/B tests and A/A tests.

    Args:
        experiment (Experiment): The experiment configuration to execute.
        output (Output): Output handler that defines how results are formatted.
        experiment_params (Optional[Dict[str, Any]], optional): Additional parameters
            to configure the experiment. Defaults to None.

    Examples
    --------
    .. code-block:: python

        # Basic usage with default parameters
        experiment = Experiment([...])  # Configure experiment
        output = Output(resume_reporter=MyReporter())
        shell = ExperimentShell(experiment, output)
        results = shell.execute(data)

        # With custom experiment parameters
        params = {
            "random_state": 42,
            "test_size": 0.3
        }
        shell = ExperimentShell(
            experiment=experiment,
            output=output,
            experiment_params=params
        )
        results = shell.execute(data)
    """

    def __init__(
        self,
        experiment: Experiment,
        output: Output,
        experiment_params: dict[str, Any] | None = None,
    ):
        if experiment_params:
            experiment.set_params(experiment_params)
        self._out = output
        self._experiment = experiment

    @property
    def experiment(self):
        """Gets the configured experiment instance.

        Returns:
            Experiment: The experiment configuration object.
        """
        return self._experiment

    def execute(self, data: Dataset | ExperimentData) -> Output:
        """Executes the experiment on the provided data.

        Runs the configured experiment on the input data and formats the results
        using the configured output handler.

        Args:
            data (Union[Dataset, ExperimentData]): Input data for the experiment.
                Can be either a Dataset or ExperimentData instance.

        Returns:
            Output: Formatted experiment results through the configured output handler.

        Examples
        --------
        .. code-block:: python

            shell = ExperimentShell(experiment, output)
            dataset = Dataset(...)  # Your input data
            results = shell.execute(dataset)
            print(results.resume)  # Access formatted results
        """
        if isinstance(data, Dataset):
            data = ExperimentData(data)
        result_experiment_data = self._experiment.execute(data)
        self._out.extract(result_experiment_data)
        return self._out

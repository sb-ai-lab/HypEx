from __future__ import annotations

from typing import Any

from ..dataset import Dataset, ExperimentData
from ..experiments.base import Experiment
from ..reporters import Reporter
from ..utils import ID_SPLIT_SYMBOL
from ..utils.enums import RenameEnum


def _html_section(title: str, level: int = 2) -> str:
    """Generate HTML section header."""
    border = '3px' if level == 2 else '2px'
    return (f'<div style="margin: {"25" if level == 2 else "20"}px 0 {"12" if level == 2 else "8"}px 0; '
            f'padding: {"8" if level == 2 else "4"}px 0; border-bottom: {border} solid {"#333" if level == 2 else "#ddd"};">'
            f'<strong style="font-size: {1.2 if level == 2 else 1.1}em;">{title}</strong></div>')


def _html_content(value: Any) -> str:
    """Generate HTML content for a value."""
    if value is None:
        return '<div style="color: #888;">None</div>'
    return value._repr_html_() if hasattr(value, '_repr_html_') else f'<pre>{str(value)}</pre>'


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

    def _get_output_fields(self) -> list[str]:
        """Get list of output fields to display in __repr__.
        
        This method can be overridden in subclasses for custom field ordering
        or filtering. By default, it returns all annotated attributes from the
        class hierarchy.
        
        Returns:
            list[str]: List of field names to display, with 'resume' always first.
        
        Examples
        --------
        .. code-block:: python
        
            # Default behavior - automatic from annotations
            class MyOutput(Output):
                resume: Dataset
                custom_field: Dataset
            
            # Custom override
            class MyOutput(Output):
                def _get_output_fields(self) -> list[str]:
                    return ['resume', 'custom_field', 'special_metric']
        """
        all_annotations = {}
        for cls in reversed(self.__class__.__mro__):
            if cls is object:
                continue
            if hasattr(cls, '__annotations__'):
                all_annotations.update(cls.__annotations__)
        
        fields = [
            name for name in all_annotations.keys()
            if not name.startswith('_') and hasattr(self, name)
        ]
        
        # Ensure 'resume' is always first if it exists
        if 'resume' in fields:
            fields.remove('resume')
            fields.insert(0, 'resume')
        
        return fields
    
    def __repr__(self) -> str:
        """Return string representation showing all output fields with their data.
        
        Displays all experiment output fields with their actual data tables.
        
        Returns:
            str: Formatted string showing all output fields with data.
        
        Examples
        --------
        .. code-block:: python
        
            output = ABOutput()
            output.extract(experiment_data)
            print(output)  # Shows all tables: resume, multitest, sizes, cupac
        """
        class_name = self.__class__.__name__
        fields = self._get_output_fields()
        
        if not fields:
            return f"{class_name}(no fields available)"
        
        output_parts = [f"{class_name}:"]
        
        for field_name in fields:
            if not hasattr(self, field_name):
                continue
            
            field_value = getattr(self, field_name)
            
            output_parts.append(f"\n{'=' * 60}")
            output_parts.append(f"{field_name}:")
            output_parts.append('=' * 60)
            
            if field_value is None:
                output_parts.append("None")
            else:
                output_parts.append(str(field_value))
        
        return "\n".join(output_parts)
    
    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        fields = self._get_output_fields()
        if not fields:
            return f"<div><b>{self.__class__.__name__}:</b> no fields available</div>"
        
        return '\n'.join(
            _html_section(f, 3) + _html_content(getattr(self, f))
            for f in fields if hasattr(self, f)
        )

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


class ExperimentOutput:
    """Container for experiment outputs with automatic delegation to main output.
    
    This class acts as a facade over main_output and additional_outputs, providing
    seamless access to all experiment results while maintaining backward compatibility.
    
    Attributes:
        main_output (Output): Primary output (e.g., ABOutput, AAOutput).
        additional_outputs (dict[str, Output]): Dict of supplementary outputs (e.g., {'cupac': CupacOutput}).
    
    Examples
    --------
    .. code-block:: python
    
        # Access main output fields directly
        result = ExperimentOutput(main_output=ABOutput(...))
        print(result.resume)  # Delegates to main_output.resume
        print(result.multitest)  # Delegates to main_output.multitest
        
        # Access additional outputs by name
        result = ExperimentOutput(
            main_output=ABOutput(...),
            additional_outputs={'cupac': CupacOutput(...)}
        )
        print(result.cupac.variance_reductions)
        
        # List all available outputs
        print(result.outputs)  # ['main', 'cupac']
    """
    
    def __init__(
        self, 
        main_output: Output, 
        additional_outputs: dict[str, Output] | None = None
    ):
        """Initialize ExperimentOutput with main and additional outputs.
        
        Args:
            main_output: Primary output object containing main experiment results.
            additional_outputs: Optional dict of named additional outputs.
        """
        self.main_output = main_output
        self.additional_outputs = additional_outputs or {}
    
    def extract(self, experiment_data: ExperimentData) -> None:
        """Extract data from experiment_data for all outputs.
        
        Calls extract() on main_output and all additional_outputs that have this method.
        For outputs that need additional context (like CupacOutput needing resume_data),
        passes it as a parameter.
        
        Args:
            experiment_data: Experiment data to extract from.
        """
        # Extract main output
        self.main_output.extract(experiment_data)
        
        # Extract additional outputs
        for name, output in self.additional_outputs.items():
            output.extract(experiment_data)
    
    @property
    def outputs(self) -> list[str]:
        """Get list of all available output names.
        
        Returns:
            List of output names including 'main' and any additional outputs.
        
        Examples
        --------
        .. code-block:: python
        
            result = ExperimentOutput(
                main_output=ABOutput(...),
                additional_outputs={'cupac': CupacOutput(...)}
            )
            print(result.outputs)  # ['main', 'cupac']
        """
        return ['main_output'] + list(self.additional_outputs.keys())
    
    def __getattr__(self, name: str):
        """Delegate attribute access to main_output first, then additional_outputs.
        
        Args:
            name: Attribute name to access.
            
        Returns:
            Attribute value from main_output or additional_outputs.
            
        Raises:
            AttributeError: If attribute not found in any output.
        """
        # Avoid recursion for private/special attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Try main_output first
        if hasattr(self.main_output, name):
            return getattr(self.main_output, name)
        
        # Then check additional_outputs
        if name in self.additional_outputs:
            return self.additional_outputs[name]
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __dir__(self):
        """Support for dir() and IDE autocompletion.
        
        Returns:
            Sorted list of all available attributes.
        """
        main_attrs = dir(self.main_output)
        additional_attrs = list(self.additional_outputs.keys())
        own_attrs = ['main_output', 'additional_outputs', 'outputs']
        return sorted(set(main_attrs + additional_attrs + own_attrs))
    
    def __repr__(self) -> str:
        """Print resume from all outputs.
        
        Returns:
            Formatted string with resumes from main and additional outputs.
        """
        parts = []
        
        if hasattr(self.main_output, 'resume'):
            if len(self.additional_outputs) > 0:
                parts.append("MAIN RESULTS")
            parts.append(str(self.main_output.resume))
        
        for name, output in self.additional_outputs.items():
            if hasattr(output, 'resume'):
                parts.append(f"\n{name.upper()} RESULTS")
                parts.append(str(output.resume))
        
        return "\n".join(parts)
    
    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        parts = []
        
        if hasattr(self.main_output, 'resume'):
            if self.additional_outputs:
                parts.append(_html_section('MAIN RESULTS'))
            parts.append(_html_content(self.main_output.resume))
        
        for name, output in self.additional_outputs.items():
            if hasattr(output, 'resume'):
                parts.extend([_html_section(f'{name.upper()} RESULTS'), _html_content(output.resume)])
        
        return '\n'.join(parts)


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

    def execute(self, data: Dataset | ExperimentData):
        """Executes the experiment on the provided data.

        Runs the configured experiment on the input data and formats the results
        using the configured output handler.

        Args:
            data (Union[Dataset, ExperimentData]): Input data for the experiment.
                Can be either a Dataset or ExperimentData instance.

        Returns:
            Output | ExperimentOutput: Formatted experiment results.

        Examples
        --------
        .. code-block:: python

            shell = ExperimentShell(experiment, output)
            dataset = Dataset(...)  # Your input data
            results = shell.execute(dataset)
            print(results.resume)  # Access main output
            print(results.cupac.variance_reductions)  # Access additional output
        """
        if isinstance(data, Dataset):
            data = ExperimentData(data)
        result_experiment_data = self._experiment.execute(data)
        
        # Extract data - works for both Output and ExperimentOutput
        self._out.extract(result_experiment_data)
        
        return self._out


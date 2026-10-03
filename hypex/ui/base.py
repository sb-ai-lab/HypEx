from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

from ..dataset import Dataset, ExperimentData
from ..experiments.base import Experiment
from ..reporters import Reporter
from ..utils import ID_SPLIT_SYMBOL, BackendsEnum
from ..utils.enums import RenameEnum


def _html_section(title: str) -> str:
    """Generate an HTML section header."""
    return (
        '<div style="margin: 20px 0 8px 0; padding: 4px 0; '
        'border-bottom: 2px solid #ddd;">'
        f'<strong style="font-size: 1.1em;">{title}</strong></div>'
    )


def _html_content(value: Any) -> str:
    """Generate HTML content for a value."""
    if value is None:
        return '<div style="color: #888;">None</div>'
    if hasattr(value, "_repr_html_"):
        return value._repr_html_()
    return f"<pre>{value}</pre>"


class _Report:
    """Full report of an output: every table under its own titled section.

    Renders as plain text in a console and as HTML tables in Jupyter.
    """

    def __init__(self, title: str, tables: dict[str, Any]):
        self.title = title
        self.tables = dict(tables)

    def __repr__(self) -> str:
        if not self.tables:
            return f"{self.title}(no tables available)"
        parts = [f"{self.title}:"]
        for name, table in self.tables.items():
            parts.append(f"\n{'=' * 60}")
            parts.append(f"{name}:")
            parts.append("=" * 60)
            parts.append("None" if table is None else str(table))
        return "\n".join(parts)

    def _repr_html_(self) -> str:
        if not self.tables:
            return f"<div><b>{self.title}:</b> no tables available</div>"
        return "\n".join(
            _html_section(name) + _html_content(table)
            for name, table in self.tables.items()
        )


def _warn_resume_deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"`{old}` is deprecated and will be removed in a future release, "
        f"use `{new}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class Output:
    """A class for handling experiment output reporting and formatting.

    This class manages the reporting and formatting of experiment results, allowing for both
    a primary summary report and additional custom reports. ``repr(output)``
    shows the full report with every table.

    Attributes:
        summary (Dataset): The main summary table of the experiment results.
            ``resume`` is a deprecated alias.
        _experiment_data (ExperimentData): Internal storage of the experiment data.

    Args:
        summary_reporter (Reporter): The main reporter that generates the summary output.
        resume_reporter (Reporter): Deprecated alias of ``summary_reporter``.
        additional_reporters (Optional[Dict[str, Reporter]]): Dictionary mapping attribute
            names to additional reporters for custom reporting. Defaults to None.

    Examples
    --------
    .. code-block:: python

        # Basic usage with just a summary reporter
        from my_reporters import MySummaryReporter
        output = Output(summary_reporter=MySummaryReporter())
        output.extract(experiment_data)
        print(output.summary)

        # Using additional custom reporters
        from my_reporters import StatsReporter, PlotReporter
        additional = {
            'statistics': StatsReporter(),
            'plots': PlotReporter()
        }
        output = Output(
            summary_reporter=MySummaryReporter(),
            additional_reporters=additional
        )
        output.extract(experiment_data)
        print(output.statistics)  # Access additional report
        print(output.plots)  # Access additional report
    """

    summary: Dataset
    _experiment_data: ExperimentData

    def __init__(
        self,
        summary_reporter: Reporter | None = None,
        additional_reporters: dict[str, Reporter] | None = None,
        *,
        resume_reporter: Reporter | None = None,
    ):
        if resume_reporter is not None:
            _warn_resume_deprecated("resume_reporter", "summary_reporter")
            if summary_reporter is None:
                summary_reporter = resume_reporter
        if summary_reporter is None:
            raise TypeError("Output requires `summary_reporter`")
        self.summary_reporter = summary_reporter
        self.additional_reporters = additional_reporters or {}

    @property
    def resume_reporter(self) -> Reporter:
        """Deprecated alias of ``summary_reporter``."""
        _warn_resume_deprecated("resume_reporter", "summary_reporter")
        return self.summary_reporter

    @property
    def resume(self) -> Dataset:
        """Deprecated alias of ``summary``."""
        _warn_resume_deprecated("resume", "summary")
        return self.summary

    @resume.setter
    def resume(self, value: Dataset) -> None:
        _warn_resume_deprecated("resume", "summary")
        self.summary = value

    def _get_output_fields(self) -> list[str]:
        """Names of the fields shown in ``repr``, ``summary`` first.

        All annotated public attributes of the class hierarchy that are set on
        the instance. Subclasses may override it for custom ordering.
        """
        annotations: dict[str, Any] = {}
        for cls in reversed(type(self).__mro__):
            annotations.update(vars(cls).get("__annotations__", {}))
        fields = [
            name
            for name in annotations
            if not name.startswith("_") and name in vars(self)
        ]
        if "summary" in fields:
            fields.remove("summary")
            fields.insert(0, "summary")
        return fields

    def _report(self) -> _Report:
        return _Report(
            type(self).__name__,
            {name: getattr(self, name) for name in self._get_output_fields()},
        )

    def __repr__(self) -> str:
        return repr(self._report())

    def _repr_html_(self) -> str:
        return self._report()._repr_html_()

    def _extract_by_reporters(self, experiment_data: ExperimentData):
        """Extracts reports from all configured reporters.

        Args:
            experiment_data (ExperimentData): The experiment data to generate reports from.
        """
        self.summary = self.summary_reporter.report(experiment_data)
        for attribute, reporter in self.additional_reporters.items():
            setattr(self, attribute, reporter.report(experiment_data))
        self._experiment_data = experiment_data

    @staticmethod
    def _replace_splitters(
        data: Dataset, mode: RenameEnum = RenameEnum.columns
    ) -> Dataset:
        if data.is_empty() or len(data.columns) == 0:
            return data

        result = data
        if mode in (RenameEnum.all, RenameEnum.columns):
            rename_map = {c: c.replace(ID_SPLIT_SYMBOL, " ") for c in result.columns}
            try:
                result.raw_data = result.raw_data.rename(columns=rename_map)
            except Exception:
                if hasattr(result._backend_data, "data"):
                    result._backend_data.data = result._backend_data.data.rename(
                        columns=rename_map
                    )
            result._roles = {
                rename_map.get(c, c): role for c, role in result._roles.items()
            }

        if mode in (RenameEnum.all, RenameEnum.index):
            result.index = [str(i).replace(ID_SPLIT_SYMBOL, " ") for i in result.index]
        return result

    def extract(self, experiment_data: ExperimentData):
        """Extracts and processes all reports from the experiment data.

        Args:
            experiment_data (ExperimentData): The experiment data to generate reports from.

        Examples
        --------
        .. code-block:: python

            output = Output(summary_reporter=MyReporter())
            output.extract(experiment_data)
            print(output.summary)  # Access the main report
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
        output = Output(summary_reporter=MyReporter())
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
        auto_persist: bool = True,
    ):
        if experiment_params:
            experiment.set_params(experiment_params)
        self._out = output
        self._experiment = experiment
        self.auto_persist = auto_persist

    def execute(self, data: Dataset | ExperimentData) -> Output:
        """Execute the experiment pipeline on the provided data.

        Orchestrates the full experiment lifecycle: data preparation, optional
        caching for Spark backends, pipeline execution, and result extraction.

        **Auto-persist behaviour (Spark only):**
        When ``auto_persist`` is enabled (default) and the input dataset uses
        the Spark backend, the method automatically persists the dataset with
        ``MEMORY_AND_DISK`` storage level before the pipeline starts. This
        avoids costly recomputation of the source DataFrame across multiple
        stages (splitters, comparators, analyzers). After the pipeline
        completes, the dataset is unpersisted **only** if it was persisted by
        this method — datasets that the user cached manually are left untouched.

        For the Pandas backend, ``persist`` / ``unpersist`` are no-ops, so
        the method behaves identically regardless of backend.

        Args:
            data: Input data for the experiment. Accepts either a raw
                :class:`~hypex.dataset.Dataset` (which will be wrapped in an
                :class:`~hypex.dataset.ExperimentData` container) or an
                already-prepared :class:`~hypex.dataset.ExperimentData` instance.

        Returns:
            Output: The experiment output object containing the formatted
            results (summary, multitest table, quality reports, etc.),
            populated by the configured :class:`Output` handler.

        Example:
            .. code-block:: python

                ab_test = ABTest(multitest_method="bonferroni")
                result = ab_test.execute(spark_dataset)
                print(result.summary)
                print(result.multitest)

        See Also:
            :meth:`Dataset.persist`: Manual caching control.
            :class:`ExperimentShell`: Constructor accepting ``auto_persist`` flag.
        """
        if isinstance(data, Dataset):
            data = ExperimentData(deepcopy(data))
        elif isinstance(data, ExperimentData):
            # Copy the container so that set_value() calls do not leak
            # back into the caller's ExperimentData.
            data = data.copy()

        # ── Auto-persist for Spark backend ──────────────────────────
        original_ds = data.ds
        persisted_by_us = False
        if (
            self.auto_persist
            and original_ds.backend_type == BackendsEnum.spark
            and not original_ds.is_persisted
        ):
            original_ds.persist(storage_level="MEMORY_AND_DISK", action="count")
            persisted_by_us = True

        try:
            result_experiment_data = self._experiment.execute(data)
            self._out.extract(result_experiment_data)
            return self._out
        finally:
            if persisted_by_us and original_ds.is_persisted:
                original_ds.unpersist()

    @property
    def experiment(self):
        """Gets the configured experiment instance.

        Returns:
            Experiment: The experiment configuration object.
        """
        return self._experiment

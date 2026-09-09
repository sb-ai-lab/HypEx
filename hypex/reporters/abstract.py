from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..comparators import GroupDifference, GroupSizes
from ..dataset import Dataset, ExperimentData, SmallDataset
from ..dataset.roles import InfoRole, StatisticRole, TreatmentRole
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum
from ..utils.constants import TEST_NAME_NORMALIZATION, NAME_BORDER_SYMBOL
from ..utils.errors import AbstractMethodError

REPORTABLE_METRICS = frozenset({
    "pass", "p-value", "difference", "difference %", "control mean", "test mean"
})

@dataclass(frozen=True)
class ResultKey:
    """Dataclass representing a parsed composite identifier.

    Provides a structured and type-safe alternative to manually parsing
    IDs separated by ``ID_SPLIT_SYMBOL``.
    """
    executor: str
    params_hash: str
    field: str

    @classmethod
    def from_id(cls, id_str: str) -> "ResultKey":
        """Parse a composite ID string into a ``ResultKey`` instance.

        Args:
            id_str: The raw ID string, typically containing executor name,
                parameter hash, and field name separated by ``ID_SPLIT_SYMBOL``.

        Returns:
            A ``ResultKey`` instance. If the string cannot be split into
            three parts, the entire string is assigned to both ``executor``
            and ``field``.
        """
        parts = id_str.split(ID_SPLIT_SYMBOL)
        if len(parts) == 3:
            return cls(executor=parts[0], params_hash=parts[1], field=parts[2])
        return cls(executor=id_str, params_hash="", field=id_str)

def _normalize_value(val: Any) -> Any:
    """Normalize cell values to basic Python types.

    Converts numpy/pandas scalars, lists, or arrays to native Python
    types. Replaces ``NaN`` floats with ``None`` and unwraps single-element
    collections.

    Args:
        val: The raw value extracted from a dataset.

    Returns:
        The normalized value, or ``None`` if empty/NaN.
    """
    if val is None:
        return None
        
    if isinstance(val, (int, float, str, bool)):
        if isinstance(val, float) and np.isnan(val):
            return None
        return val
        
    if hasattr(val, 'item'):
        try:
            item_val = val.item()
            if isinstance(item_val, float) and np.isnan(item_val):
                return None
            return item_val
        except (ValueError, TypeError, AttributeError):
            pass
            
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        return _normalize_value(val[0])
        
    return val

def _get_index_values(table: Dataset | SmallDataset) -> list[Any]:
    """Extract index values from a dataset in a backend-agnostic way.

    Handles differences between pandas (``tolist``) and pyspark.pandas
    (``to_list``) APIs.

    Args:
        table: The dataset instance.

    Returns:
        A list of index values.
    """
    index_obj = table.index
    if hasattr(index_obj, "to_list"):
        return index_obj.to_list()
    if hasattr(index_obj, "tolist"):
        return index_obj.tolist()
    return list(index_obj)

def _normalize_group_name(group: str) -> str:
    """Normalize group names by stripping tuple notation.

    StatsComparator may produce tuple keys like '(1,)' while
    GroupsComparator produces plain strings like '1'.
    This function unifies them so the reporter merges rows correctly.

    Examples:
        '(1,)'   -> '1'
        '(2,)'   -> '2'
        "('a',)" -> "'a'"  (multi-value tuples are left as-is)
        '1'      -> '1'   (already normal)
    """
    stripped = group.strip()
    if stripped.startswith('(') and stripped.endswith(')'):
        inner = stripped[1:-1]
        # Remove trailing comma for single-element tuples: "1," -> "1"
        if inner.endswith(','):
            inner = inner[:-1]
        # Only simplify if there's no remaining comma (single-element tuple)
        if ',' not in inner:
            return inner.strip()
    return group

def _extract_from_comparator(data: ExperimentData, comparator_id: str, front: bool) -> dict[str, Any]:
    """Extract and flatten metrics from a comparator's analysis table.

    Args:
        data: The experiment data container.
        comparator_id: The unique ID of the comparator in ``analysis_tables``.
        front: If ``True``, uses spaces as separators in output keys;
            otherwise uses ``ID_SPLIT_SYMBOL``.

    Returns:
        A flat dictionary mapping composite keys to normalized metric values.
    """
    table = data.analysis_tables.get(comparator_id)
    if table is None or table.is_empty():
        return {}
    key = ResultKey.from_id(comparator_id)
    sep = " " if front else ID_SPLIT_SYMBOL
    result = {}
    records = table.to_records()
    index_values = _get_index_values(table)

    for idx_val, row_dict in zip(index_values, records):
        idx_str = str(idx_val)
        if NAME_BORDER_SYMBOL in idx_str:
            group, feature = idx_str.split(NAME_BORDER_SYMBOL, 1)
        else:
            group = idx_str
            feature = key.field
        group = _normalize_group_name(group)

        for col, val in row_dict.items():
            full_key = f"{feature}{sep}{key.executor}{sep}{col}{sep}{group}"
            result[full_key] = _normalize_value(val)
    return result

def extract_tests(data: ExperimentData, test_classes: list[type], front: bool) -> dict[str, Any]:
    """Extract test outcomes (p-values and pass flags) for specified test classes.

    Args:
        data: The experiment data container.
        test_classes: List of comparator classes to search for.
        front: Formatting flag for the output dictionary keys.

    Returns:
        A dictionary containing only ``'pass'`` and ``'p-value'`` entries
        for the requested tests.
    """
    result = {}
    for cls_ in test_classes:
        ids = data.get_ids(cls_, searched_space=ExperimentDataEnum.analysis_tables)
        for id_list in ids.get(cls_.__name__, {}).values():
            for cid in id_list:
                for k, v in _extract_from_comparator(data, cid, front).items():
                    if "pass" in k or "p-value" in k:
                        result[k] = v
    return result

def extract_group_difference(data: ExperimentData, front: bool) -> dict[str, Any]:
    """Extract group difference metrics (e.g., means, differences).

    Args:
        data: The experiment data container.
        front: Formatting flag for the output dictionary keys.

    Returns:
        A dictionary of group difference results.
    """
    ids = data.get_ids(GroupDifference)[GroupDifference.__name__][ExperimentDataEnum.analysis_tables.value]
    out = {}
    for cid in ids:
        out.update(_extract_from_comparator(data, cid, front))
    return out

def extract_group_sizes(data: ExperimentData, front: bool) -> dict[str, Any]:
    """Extract group size information.

    Args:
        data: The experiment data container.
        front: Formatting flag for the output dictionary keys.

    Returns:
        A dictionary containing group sizes and their percentages.
    """
    cid = data.get_one_id(GroupSizes, ExperimentDataEnum.analysis_tables)
    return _extract_from_comparator(data, cid, front)

def extract_analyzer_data(data: ExperimentData, analyzer_class: type | str) -> dict[str, Any]:
    """Extract aggregated metrics from a specific analyzer.

    Args:
        data: The experiment data container.
        analyzer_class: The class or name of the analyzer to retrieve data from.

    Returns:
        A flat dictionary of the first row of the analyzer's results.
    """
    cid = data.get_one_id(analyzer_class, ExperimentDataEnum.analysis_tables)
    table = data.analysis_tables[cid]
    if table.is_empty():
        return {}
        
    records = table.to_records()
    if not records:
        return {}
        
    row_dict = records[0]
    return {col: _normalize_value(val) for col, val in row_dict.items()}

class Reporter(ABC):
    """Abstract base class for all experiment reporters."""
    @abstractmethod
    def report(self, data: ExperimentData) -> Any:
        """Generate a report from the experiment data.

        Args:
            data: The experiment data container.

        Returns:
            The formatted report content.

        Raises:
            AbstractMethodError: If not implemented by subclass.
        """
        raise AbstractMethodError

class DictReporter(Reporter, ABC):
    """Base reporter that outputs results as a flat dictionary."""
    def __init__(self, front: bool = True):
        """Initialize the dictionary reporter.

        Args:
            front: If ``True``, formats keys for front-end display (e.g.,
                using spaces instead of symbols). Defaults to ``True``.
        """
        self.front = front

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        """Internal method to build the dictionary report.

        Subclasses should override this to implement specific extraction logic.

        Args:
            data: The experiment data container.

        Returns:
            An empty dictionary by default.
        """
        return {}

    def report(self, data: ExperimentData) -> dict[str, Any]:
        """Generate the final dictionary report.

        Args:
            data: The experiment data container.

        Returns:
            The formatted dictionary.
        """
        return self._report(data)

class TestDictReporter(DictReporter, ABC):
    """Reporter specialized for statistical tests with dict-to-dataset conversion."""
    tests: list[type] = []

    @staticmethod
    def _get_struct_dict(data: dict) -> dict:
        """Convert a flat dictionary into a nested hierarchical structure.

        Parses keys containing ``ID_SPLIT_SYMBOL`` to group metrics by
        field, index, executor, and metric type.

        Args:
            data: The flat dictionary with composite keys.

        Returns:
            A nested dictionary structured as 
            ``{field: {index: {executor: {metric: value}}}}``.
        """
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for key, value in data.items():
            if ID_SPLIT_SYMBOL not in key:
                continue
            parts = key.split(ID_SPLIT_SYMBOL)
            if len(parts) >= 4 and parts[2] in REPORTABLE_METRICS:
                tree[parts[0]][parts[3]][parts[1]][parts[2]] = value
        return dict(tree)

    @staticmethod
    def _convert_struct_dict_to_dataset(data: dict) -> SmallDataset:
        """Transform a nested dictionary into a ``SmallDataset``.

        Flattens the hierarchical structure into rows, mapping metrics to
        appropriate columns and converting boolean pass flags to strings
        (``'OK'`` / ``'NOT OK'``).

        Args:
            data: The nested dictionary from ``_get_struct_dict``.

        Returns:
            A ``SmallDataset`` containing the structured test results.
        """
        result = []
        for feature, groups in data.items():
            for group, tests in groups.items():
                row = {"feature": feature, "group": group}
                
                if "GroupDifference" in tests:
                    metrics = tests["GroupDifference"]
                    for k in ("control mean", "test mean", "difference", "difference %"):
                        if k in metrics:
                            row[k] = metrics.get(k)
                            
                for test_name, metrics in tests.items():
                    if test_name == "GroupDifference":
                        continue
                    norm_name = TEST_NAME_NORMALIZATION.get(test_name, test_name)
                    row[f"{norm_name} pass"] = metrics.get("pass")
                    row[f"{norm_name} p-value"] = metrics.get("p-value")
                    
                result.append(row)
                
        for row in result:
            for k, v in list(row.items()):
                if "pass" in k:
                    row[k] = "OK" if v is True or str(v).lower() in ("true", "1") else "NOT OK"
                    
        if not result:
            return SmallDataset.from_dict(
                {"feature": [], "group": []},
                roles={"feature": InfoRole(), "group": TreatmentRole()}
            )
        return SmallDataset.from_dict(
            result,
            roles={"feature": InfoRole(), "group": TreatmentRole()},
        )

    def extract_tests(self, data: ExperimentData) -> dict[str, Any]:
        """Extract test results using the reporter's configured test classes and format.

        Args:
            data: The experiment data container.

        Returns:
            A dictionary of test results formatted according to the ``front`` flag.
        """
        return extract_tests(data, self.tests, self.front)

class DatasetReporter(Reporter):
    """Reporter that outputs results as a structured ``Dataset`` or dictionary."""

    def __init__(
        self,
        dict_reporter: DictReporter | None = None,
        output_format: Literal["dict", "dataset"] = "dataset",
        single_row: bool = False,
    ):
        self.dict_reporter = dict_reporter or DictReporter()
        self.output_format = output_format
        self.single_row = single_row

    @property
    def front(self) -> bool:
        return self.dict_reporter.front

    @front.setter
    def front(self, value: bool) -> None:
        self.dict_reporter.front = value

    def _with_front(self, data, front_flag, func):
        old_front = self.dict_reporter.front
        self.dict_reporter.front = front_flag
        try:
            return func(data)
        finally:
            self.dict_reporter.front = old_front

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        return self.dict_reporter.report(data)

    def report(self, data: ExperimentData) -> dict | Dataset:
        old_front = self.dict_reporter.front
        self.dict_reporter.front = False
        try:
            dict_result = self._report(data)
        finally:
            self.dict_reporter.front = old_front

        if self.output_format == "dict":
            return dict_result
        if self.single_row:
            return self._to_single_row_dataset(dict_result)
        struct_dict = TestDictReporter._get_struct_dict(dict_result)
        return TestDictReporter._convert_struct_dict_to_dataset(struct_dict)

    @staticmethod
    def _to_single_row_dataset(data: dict) -> SmallDataset:
        """Convert flat dict with composite keys to a single-row dataset.

        Each key like ``feature┆Executor┆metric┆group`` becomes a column
        ``feature Executor metric group`` (with normalised test names).
        """
        row: dict[str, Any] = {}
        for key, value in data.items():
            col = str(key)
            if ID_SPLIT_SYMBOL in col:
                col = col.replace(ID_SPLIT_SYMBOL, " ")
            # Normalize executor names: StatsTTest → TTest, etc.
            for raw_name, norm_name in TEST_NAME_NORMALIZATION.items():
                if raw_name != norm_name:
                    col = col.replace(raw_name, norm_name)
            row[col] = _normalize_value(value)
        if not row:
            return SmallDataset.create_empty()
        return SmallDataset.from_dict(
            [row], roles={c: StatisticRole() for c in row}
        )

    @staticmethod
    def convert_to_dataset(data: dict) -> Dataset | SmallDataset:
        struct_dict = TestDictReporter._get_struct_dict(data)
        return TestDictReporter._convert_struct_dict_to_dataset(struct_dict)
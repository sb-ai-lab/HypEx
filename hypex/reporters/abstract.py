from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from ..dataset import Dataset, SmallDataset, ExperimentData
from ..dataset.roles import InfoRole, TreatmentRole
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum
from ..utils.errors import AbstractMethodError

REPORTABLE_METRICS = frozenset({
    "pass", "p-value", "difference", "difference %", "control mean", "test mean"
})

@dataclass(frozen=True)
class ResultKey:
    """Safe, typed replacement for manual ID_SPLIT_SYMBOL parsing."""
    executor: str
    params_hash: str
    field: str

    @classmethod
    def from_id(cls, id_str: str) -> "ResultKey":
        parts = id_str.split(ID_SPLIT_SYMBOL)
        if len(parts) == 3:
            return cls(executor=parts[0], params_hash=parts[1], field=parts[2])
        return cls(executor=id_str, params_hash="", field=id_str)

def _normalize_value(val: Any) -> Any:
    """
    Приводит значения из ячеек Dataset к базовым Python-типам.
    Работает только со скалярами и стандартными коллекциями, 
    которые возвращает метод to_records().
    """
    if val is None:
        return None
        
    # Базовые типы Python
    if isinstance(val, (int, float, str, bool)):
        if isinstance(val, float) and np.isnan(val):
            return None
        return val
        
    # Numpy / Pandas скаляры (np.int64, np.float64 и т.д.)
    if hasattr(val, 'item'):
        try:
            item_val = val.item()
            if isinstance(item_val, float) and np.isnan(item_val):
                return None
            return item_val
        except (ValueError, TypeError, AttributeError):
            pass
            
    # Если в ячейке лежит список или массив (например, после специфичных агрегаций)
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        return _normalize_value(val[0])
        
    return val

def _get_index_values(table: Dataset | SmallDataset) -> list[Any]:
    """
    Безопасное извлечение списка индексов без нарушения абстракции бэкенда.
    Учитывает различия в API pandas (tolist) и pyspark.pandas (to_list).
    """
    index_obj = table.index
    if hasattr(index_obj, "to_list"):
        return index_obj.to_list()
    if hasattr(index_obj, "tolist"):
        return index_obj.tolist()
    return list(index_obj)

def _extract_from_comparator(data: ExperimentData, comparator_id: str, front: bool) -> dict[str, Any]:
    table = data.analysis_tables.get(comparator_id)
    if table is None or table.is_empty():
        return {}
        
    key = ResultKey.from_id(comparator_id)
    sep = " " if front else ID_SPLIT_SYMBOL
    result = {}
    
    # Используем API Dataset: to_records() сам заботится о конвертации из Spark/Pandas
    records = table.to_records()
    index_values = _get_index_values(table)
    
    for idx_val, row_dict in zip(index_values, records):
        for col, val in row_dict.items():
            result[f"{key.field}{sep}{key.executor}{sep}{col}{sep}{idx_val}"] = _normalize_value(val)
            
    return result

def extract_tests(data: ExperimentData, test_classes: list[type], front: bool) -> dict[str, Any]:
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
    from ..comparators import GroupDifference
    ids = data.get_ids(GroupDifference)[GroupDifference.__name__][ExperimentDataEnum.analysis_tables.value]
    out = {}
    for cid in ids:
        out.update(_extract_from_comparator(data, cid, front))
    return out

def extract_group_sizes(data: ExperimentData, front: bool) -> dict[str, Any]:
    from ..comparators import GroupSizes
    cid = data.get_one_id(GroupSizes, ExperimentDataEnum.analysis_tables)
    return _extract_from_comparator(data, cid, front)

def extract_analyzer_data(data: ExperimentData, analyzer_class: type | str) -> dict[str, Any]:
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
    @abstractmethod
    def report(self, data: ExperimentData) -> Any:
        raise AbstractMethodError

class DictReporter(Reporter, ABC):
    """Base dict-producing reporter with safe front-toggle."""
    def __init__(self, front: bool = True):
        self.front = front

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        return {}

    def report(self, data: ExperimentData) -> dict[str, Any]:
        return self._report(data)

class TestDictReporter(DictReporter, ABC):
    """Specialized for statistical tests with flattened dict->dataset conversion."""
    tests: list[type] = []

    @staticmethod
    def _get_struct_dict(data: dict) -> dict:
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
        result = []
        for feature, groups in data.items():
            for group, tests in groups.items():
                row = {"feature": feature, "group": group}
                for test_name, metrics in tests.items():
                    if test_name == "GroupDifference":
                        row.update({k: metrics.get(k) for k in REPORTABLE_METRICS if k not in ("pass", "p-value")})
                    else:
                        row[f"{test_name} pass"] = metrics.get("pass")
                        row[f"{test_name} p-value"] = metrics.get("p-value")
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
        return extract_tests(data, self.tests, self.front)

class DatasetReporter(Reporter):
    def __init__(
        self,
        dict_reporter: DictReporter | None = None,
        output_format: Literal["dict", "dataset"] = "dataset"
    ):
        self.dict_reporter = dict_reporter or DictReporter()
        self.output_format = output_format
        
    @property
    def front(self) -> bool:
        return self.dict_reporter.front

    @front.setter
    def front(self, value: bool) -> None:
        self.dict_reporter.front = value

    def _with_front(
        self,
        data: ExperimentData,
        front_flag: bool,
        func: Callable[[ExperimentData], dict | Dataset],
    ) -> dict | Dataset:
        old_front = self.dict_reporter.front
        self.dict_reporter.front = front_flag
        try:
            return func(data)
        finally:
            self.dict_reporter.front = old_front

    def report(self, data: ExperimentData) -> dict | Dataset:
        dict_result = self.dict_reporter.report(data)
        
        if self.output_format == "dict":
            return dict_result
        struct_dict = TestDictReporter._get_struct_dict(dict_result)
        return TestDictReporter._convert_struct_dict_to_dataset(struct_dict)

    @staticmethod
    def convert_to_dataset(data: dict) -> Dataset | SmallDataset:
        struct_dict = TestDictReporter._get_struct_dict(data)
        return TestDictReporter._convert_struct_dict_to_dataset(struct_dict)
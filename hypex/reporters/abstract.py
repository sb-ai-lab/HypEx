from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Literal

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
    if val is None:
        return None
    if isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return _normalize_value(val[0])
    # Dataset / SmallDataset
    if hasattr(val, 'iget_values'):
        try:
            return _normalize_value(val.iget_values(0, 0))
        except Exception:
            pass
    # pandas / pyspark Series
    if hasattr(val, 'iloc') and hasattr(val, '__len__'):
        try:
            return _normalize_value(val.iloc[0] if len(val) > 0 else val)
        except Exception:
            pass
    # numpy scalar
    if hasattr(val, 'item'):
        try:
            return val.item()
        except Exception:
            pass
    return val

def _extract_from_comparator(data: ExperimentData, comparator_id: str, front: bool) -> dict[str, Any]:
    table = data.analysis_tables.get(comparator_id)
    if table is None:
        return {}
    key = ResultKey.from_id(comparator_id)
    sep = " " if front else ID_SPLIT_SYMBOL
    result = {}
    
    df = table.data
    if hasattr(df, 'to_pandas'):
        df = df.to_pandas()
    elif hasattr(df, 'toPandas'):
        df = df.toPandas()
        
    for idx_val, row in df.iterrows():
        for col, val in row.items():
            print(f"[DEBUG] _extract_from_comparator | idx={idx_val}, col={col}, val.type={type(val).__name__}, val={val}")
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
        
    row = table.data
    if hasattr(row, 'to_pandas'):
        row = row.to_pandas()
    elif hasattr(row, 'toPandas'):
        row = row.toPandas()
        
    if hasattr(row, 'iloc'):
        row = row.iloc[0]
        
    items = row.to_dict() if hasattr(row, 'to_dict') else (row if isinstance(row, dict) else {})
    return {col: _normalize_value(val) for col, val in items.items()}

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
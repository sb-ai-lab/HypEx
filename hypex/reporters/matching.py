from __future__ import annotations
from typing import Any, ClassVar, Literal
import warnings

from ..analyzers.matching import MatchingAnalyzer
from ..comparators import GroupChi2Test, GroupKSTest, GroupTTest
from ..dataset import Dataset, ExperimentData, StatisticRole
from ..ml import FaissNearestNeighbors
from ..utils import ID_SPLIT_SYMBOL, MATCHING_INDEXES_SPLITTER_SYMBOL, ExperimentDataEnum
from .abstract import DatasetReporter, DictReporter, Reporter, TestDictReporter

class MatchingReporter(DatasetReporter):
    def __init__(self, searching_class: type = MatchingAnalyzer, output_format: Literal["dict", "dataset"] = "dataset"):
        dict_rep = DictReporter()
        super().__init__(dict_rep, output_format)
        self.searching_class = searching_class

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        result = self._extract_from_analyser(data)
        if self.searching_class == MatchingAnalyzer:
            result.update(self._extract_indexes(data))
        return result

    def _extract_from_analyser(self, data: ExperimentData) -> dict[str, Any]:
        analyzer_id = data.get_one_id(self.searching_class, ExperimentDataEnum.analysis_tables)
        table = data.analysis_tables[analyzer_id].data
        return {f"{col}{ID_SPLIT_SYMBOL}{idx}": val for col in table.columns for idx, row in table.iterrows() for val in [row[col]]}

    def _extract_indexes(self, data: ExperimentData) -> dict[str, str]:
        ids = data.get_ids(FaissNearestNeighbors, ExperimentDataEnum.additional_fields)[FaissNearestNeighbors.__name__][ExperimentDataEnum.additional_fields.value]
        return {f"indexes{ID_SPLIT_SYMBOL}{col.split(ID_SPLIT_SYMBOL)[3]}": MATCHING_INDEXES_SPLITTER_SYMBOL.join(str(i) for i in data.additional_fields[col].data.tolist()) for col in ids}

class MatchingQualityReporter(DatasetReporter):
    tests: ClassVar[list] = [GroupTTest, GroupKSTest, GroupChi2Test]
    def _report(self, data: ExperimentData) -> dict: return extract_tests(data, self.tests, self.front)

# Aliases
class MatchingDictReporter(MatchingReporter):
    def __init__(self, searching_class=MatchingAnalyzer):
        super().__init__(searching_class, output_format="dict")
        warnings.warn("MatchingDictReporter is deprecated.", DeprecationWarning, stacklevel=2)

class MatchingQualityDictReporter(MatchingQualityReporter):
    def __init__(self, front=True):
        super().__init__(output_format="dict")
        warnings.warn("MatchingQualityDictReporter is deprecated.", DeprecationWarning, stacklevel=2)

class MatchingDatasetReporter(MatchingReporter):
    def __init__(self, searching_class=MatchingAnalyzer):
        super().__init__(searching_class, output_format="dataset")
        warnings.warn("MatchingDatasetReporter is deprecated.", DeprecationWarning, stacklevel=2)

class MatchingQualityDatasetReporter(MatchingQualityReporter):
    def __init__(self):
        super().__init__(output_format="dataset")
        warnings.warn("MatchingQualityDatasetReporter is deprecated.", DeprecationWarning, stacklevel=2)
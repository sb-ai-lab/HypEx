# hypex/reporters/matching.py
from __future__ import annotations

from typing import Any, ClassVar, Literal
import warnings

from ..analyzers.matching import MatchingAnalyzer
from ..comparators import GroupChi2Test, GroupKSTest, GroupTTest
from ..dataset import Dataset, ExperimentData, StatisticRole
from ..ml import FaissNearestNeighbors
from ..utils import ID_SPLIT_SYMBOL, MATCHING_INDEXES_SPLITTER_SYMBOL, ExperimentDataEnum
from .abstract import (
    DatasetReporter,
    DictReporter,
    Reporter,
    TestDictReporter,
    extract_tests,
)


class MatchingReporter(DatasetReporter):
    """Reporter for matching experiment results.
    
    Extracts treatment effect metrics and matched neighbor indices,
    formatting them into a structured dataset or dictionary.
    """

    def __init__(
        self, 
        searching_class: type = MatchingAnalyzer, 
        output_format: Literal["dict", "dataset"] = "dataset"
    ):
        """Initialize the matching reporter.
        
        Args:
            searching_class: The analyzer class used to compute matching metrics.
            output_format: The desired output format ('dict' or 'dataset').
        """
        dict_rep = DictReporter()
        super().__init__(dict_rep, output_format)
        self.searching_class = searching_class

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        """Construct the internal dictionary report for matching.
        
        Args:
            data: The experiment data container.
            
        Returns:
            A dictionary containing analyzer metrics and matched indices.
        """
        result = self._extract_from_analyser(data)
        if self.searching_class == MatchingAnalyzer:
            result.update(self._extract_indexes(data))
        return result

    def _extract_from_analyser(self, data: ExperimentData) -> dict[str, Any]:
        """Extract flattened metrics from the matching analyzer table.
        
        Args:
            data: The experiment data container.
            
        Returns:
            A flat dictionary mapping composite keys to metric values.
        """
        analyzer_id = data.get_one_id(self.searching_class, ExperimentDataEnum.analysis_tables)
        table = data.analysis_tables[analyzer_id].data
        return {
            f"{col}{ID_SPLIT_SYMBOL}{idx}": val 
            for col in table.columns 
            for idx, row in table.iterrows() 
            for val in [row[col]]
        }

    def _extract_indexes(self, data: ExperimentData) -> dict[str, str]:
        """Extract matched neighbor indices from additional fields.
        
        Uses the backend-agnostic ``_to_numpy()`` method to safely collect 
        values from single-column datasets. In the Spark backend, column 
        selection returns a ``DataFrame`` (not a ``Series``), which lacks 
        the ``.tolist()`` method and causes an ``AttributeError`` if 
        accessed directly via ``.data.tolist()``.
        
        Args:
            data: The experiment data container.
            
        Returns:
            A dictionary mapping composite index keys to a string of 
            neighbor indices joined by the matching splitter symbol.
        """
        ids = data.get_ids(
            FaissNearestNeighbors, ExperimentDataEnum.additional_fields
        )[FaissNearestNeighbors.__name__][ExperimentDataEnum.additional_fields.value]
        
        return {
            f"indexes{ID_SPLIT_SYMBOL}{col.split(ID_SPLIT_SYMBOL)[3]}": 
            MATCHING_INDEXES_SPLITTER_SYMBOL.join(
                str(int(i)) if isinstance(i, float) and i.is_integer() else str(i) 
                for i in data.additional_fields[col]._to_numpy()
            ) 
            for col in ids
        }


class MatchingQualityReporter(DatasetReporter):
    """Reporter for matching quality tests (T-Test, KS-Test, Chi2-Test)."""
    
    tests: ClassVar[list] = [GroupTTest, GroupKSTest, GroupChi2Test]
    
    def _report(self, data: ExperimentData) -> dict: 
        """Extract quality test outcomes.
        
        Args:
            data: The experiment data container.
            
        Returns:
            A dictionary of test pass flags and p-values.
        """
        return extract_tests(data, self.tests, self.front)


# ── Backwards-compatible aliases ─────────────────────────────────────────────

class MatchingDictReporter(MatchingReporter):
    """Legacy reporter wrapper for dictionary output.
    
    Deprecated: Use ``MatchingReporter(output_format='dict')`` instead.
    """
    def __init__(self, searching_class=MatchingAnalyzer):
        super().__init__(searching_class, output_format="dict")
        warnings.warn(
            "MatchingDictReporter is deprecated. Use MatchingReporter(output_format='dict')", 
            DeprecationWarning, 
            stacklevel=2
        )


class MatchingQualityDictReporter(MatchingQualityReporter):
    """Legacy reporter wrapper for dictionary output.
    
    Deprecated: Use ``MatchingQualityReporter(output_format='dict')`` instead.
    """
    def __init__(self, front=True):
        super().__init__(output_format="dict")
        self.front = front
        warnings.warn(
            "MatchingQualityDictReporter is deprecated.", 
            DeprecationWarning, 
            stacklevel=2
        )


class MatchingDatasetReporter(MatchingReporter):
    """Legacy reporter wrapper for dataset output.
    
    Deprecated: Use ``MatchingReporter()`` instead.
    """
    def __init__(self, searching_class=MatchingAnalyzer):
        super().__init__(searching_class, output_format="dataset")
        warnings.warn(
            "MatchingDatasetReporter is deprecated.", 
            DeprecationWarning, 
            stacklevel=2
        )
        
    def report(self, data: ExperimentData) -> Dataset:
        """Directly return the valid dataset from MatchingAnalyzer.
        
        Bypasses the broken dict-to-dataset conversion via TestDictReporter
        that previously returned an empty dataset due to key parsing mismatches.
        
        Args:
            data: The experiment data container.
            
        Returns:
            The pre-computed matching metrics dataset.
        """
        analyzer_id = data.get_one_id(self.searching_class, ExperimentDataEnum.analysis_tables)
        return data.analysis_tables[analyzer_id]


class MatchingQualityDatasetReporter(MatchingQualityReporter):
    """Legacy reporter wrapper for dataset output.
    
    Deprecated: Use ``MatchingQualityReporter()`` instead.
    """
    def __init__(self):
        super().__init__(output_format="dataset")
        warnings.warn(
            "MatchingQualityDatasetReporter is deprecated.", 
            DeprecationWarning, 
            stacklevel=2
        )
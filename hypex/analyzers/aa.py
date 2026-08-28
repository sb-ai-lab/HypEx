from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd  # type: ignore

from ..comparators import (
    GroupChi2Test,
    GroupKSTest,
    GroupTTest,
    StatsChi2Test,
    StatsKSTest,
    StatsTTest,
    StatsZTest,
)
from ..dataset import (
    Dataset,
    ExperimentData,
    InfoRole,
    SmallDataset,
    StatisticRole,
    StratificationRole,
)
from ..executor import Executor
from ..experiments import IfParamsExperiment, ParamsExperiment
from ..splitters import AASplitter, AASplitterWithStratification
from ..utils import ID_SPLIT_SYMBOL, ExperimentDataEnum, timeit
from ..utils.constants import NAME_BORDER_SYMBOL
from ..utils.naming import _parse_metric_col, normalize_test_name

# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean_key(class_name: str, field: str) -> str:
    """Build the composite key used in analysis_data."""
    return f"mean{ID_SPLIT_SYMBOL}{class_name}{ID_SPLIT_SYMBOL}{field}{ID_SPLIT_SYMBOL}all"


def _is_passed(value: Any) -> bool:
    """Check whether a cell value represents a passed test."""
    #: Values treated as "passed" when parsing pass-columns.
    _PASS_TRUTHY = frozenset({"OK", "TRUE", "1"})
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) >= 1.0
    # ──────────────────────────────────────────────────────
    return str(value).strip().upper() in _PASS_TRUTHY


def _collect_tables(
    data: ExperimentData,
    analysis_ids: list[str],
) -> Dataset:
    """Merge multiple analysis tables into one."""
    tables = [data.analysis_tables[aid] for aid in analysis_ids]
    if len(tables) == 1:
        return tables[0]
    return tables[0].append(tables[1:])


def _resolve_column_parts(col: str) -> tuple[str, str, str] | None:
    feature, raw_test, _, group = _parse_metric_col(col)
    if raw_test:
        if NAME_BORDER_SYMBOL in group:
            group = group.split(NAME_BORDER_SYMBOL)[0]
        return feature, raw_test, group

    if ID_SPLIT_SYMBOL in col:
        parts = col.split(ID_SPLIT_SYMBOL)
        if len(parts) >= 4:
            grp = parts[3]
            if NAME_BORDER_SYMBOL in grp:
                grp = grp.split(NAME_BORDER_SYMBOL)[0]
            return parts[0], parts[1], grp
    return None


# ── OneAAStatAnalyzer ─────────────────────────────────────────────────────────

class OneAAStatAnalyzer(Executor):
    """Aggregates statistical test results across multiple A/A test splits.

    Computes average p-values, pass rates, and a weighted composite quality
    score to evaluate the overall consistency of data splitting configurations.
    """
    #: Registered test classes whose results are aggregated by OneAAStatAnalyzer.
    ANALYSIS_TEST_CLASSES: ClassVar[tuple[type]] = tuple([
        GroupTTest,
        GroupKSTest,
        GroupChi2Test,
        StatsTTest,
        StatsChi2Test,
        StatsZTest,
        StatsKSTest]
    )

    #: (preferred_class, fallback_class, weight) for composite score computation.
    #: Preferred = Spark-backed (Stats*), fallback = Pandas-backed (Group*).
    _SCORE_RULES: ClassVar[tuple[tuple[str, str, int]]] = tuple([
            ("StatsTTest", "GroupTTest", 1),
            ("StatsKSTest", "GroupKSTest", 2),
            ("StatsChi2Test", "GroupChi2Test", 2),
        ]
    )

    #: Weights for the final best-split scoring formula.
    PVALUE_WEIGHT: float = 2 / 3
    TEST_SCORE_WEIGHT: float = 1 / 3

    def _set_value(self, data: ExperimentData, value: Any) -> ExperimentData:
        """Stores the aggregated metrics in the experiment data container."""
        return data.set_value(ExperimentDataEnum.analysis_tables, self.id, value)

    @timeit(level="ANALYZER", prefix="AA_STAT")
    def execute(self, data: ExperimentData) -> ExperimentData:
        executor_ids = data.get_ids(
            self.ANALYSIS_TEST_CLASSES,
            searched_space=ExperimentDataEnum.analysis_tables,
        )

        analysis_data = self._collect_means(data, executor_ids)
        analysis_data = self._sanitize_nan(analysis_data)
        analysis_data["mean test score"] = self._compute_composite_score(analysis_data)

        return self._set_value(
            data,
            SmallDataset.from_dict(
                analysis_data,
                {field: StatisticRole(float) for field in analysis_data},
            ),
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _collect_means(
        self,
        data: ExperimentData,
        executor_ids: dict[str, dict[str, list[str]]],
    ) -> dict[str, float]:
        """Compute mean p-value and pass rate per test class."""
        result: dict[str, float] = {}

        for class_name, spaces in executor_ids.items():
            ids = spaces.get("analysis_tables", [])
            if not ids:
                continue

            table = _collect_tables(data, ids)
            for field in ("p-value", "pass"):
                key = _mean_key(class_name, field)
                if field in table.columns:
                    result[key] = table[field].mean()
                else:
                    result[key] = 0.0

        return result

    @staticmethod
    def _sanitize_nan(data: dict[str, float]) -> dict[str, float]:
        """Replace NaN values with 0."""
        return {k: (0.0 if np.isnan(v) else v) for k, v in data.items()}

    @classmethod
    def _compute_composite_score(cls, analysis_data: dict[str, float]) -> float:
        """Compute the weighted composite score from individual test p-values.

        Rules:
        - T-test:  weight 1 (checks means only).
        - KS-test: weight 2 (checks full distribution).
        - Chi2:    weight 2 (categorical targets).

        Within each family, Stats* (Spark) takes priority over Group* (Pandas).
        """
        score = 0.0
        total_weight = 0

        for preferred, fallback, weight in cls._SCORE_RULES:
            pref_key = _mean_key(preferred, "p-value")
            fall_key = _mean_key(fallback, "p-value")

            if pref_key in analysis_data:
                score += weight * analysis_data[pref_key]
                total_weight += weight
            elif fall_key in analysis_data:
                score += weight * analysis_data[fall_key]
                total_weight += weight

        return score / total_weight if total_weight else 0.0


# ── AAScoreAnalyzer ───────────────────────────────────────────────────────────

class AAScoreAnalyzer(Executor):
    """Evaluates A/A test split quality and identifies the optimal splitting configuration.

    Analyzes statistical test pass rates against a target significance level
    (alpha), computes reliability weights, and selects the split with the best
    composite score.
    """

    #: Pass-rate threshold multiplier relative to alpha.
    THRESHOLD_FACTOR: float = 1.2

    PVALUE_WEIGHT: float = 2 / 3
    TEST_SCORE_WEIGHT: float = 1 / 3

    SPLITTER_CLASS_MAPPING: ClassVar[dict[str, type]] = {
        cls.__name__: cls for cls in (AASplitter, AASplitterWithStratification)
    }

    def __init__(self, alpha: float = 0.05, key: str = ""):
        """Initializes the A/A score analyzer.

        Args:
            alpha: Target significance level for hypothesis testing.
            key: Optional identifier key for storing results.
        """
        super().__init__(key=key)
        self.alpha = alpha
        self.threshold = 1 - (self.alpha * self.THRESHOLD_FACTOR)
        self._feature_weights: dict[str, float] = {}

    # ── Storage ───────────────────────────────────────────────────────────

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        """Stores analysis results in the experiment data."""
        return data.set_value(
            ExperimentDataEnum.analysis_tables,
            executor_id=self.id,
            key=self.key,
            value=value,
        )

    # ── Public API ────────────────────────────────────────────────────────

    @timeit(level="ANALYZER", prefix="AA_SCORE")
    def execute(self, data: ExperimentData) -> ExperimentData:
        """Executes the full A/A scoring and split selection pipeline.

        Args:
            data: Experiment data containing parameter experiment results.

        Returns:
            Updated ExperimentData with AA scores and the best split applied.
        """
        param_id = data.get_one_id(ParamsExperiment, ExperimentDataEnum.analysis_tables)
        score_table = data.analysis_tables[param_id]
        if_param_scores = self._get_if_param_scores(data)

        data = self._analyze_aa_score(data, score_table)
        return self._analyze_best_split(data, score_table, if_param_scores)

    def build_splitter_from_id(self, splitter_id: str) -> AASplitter:
        """Reconstructs a splitter instance from its serialized identifier.

        Args:
            splitter_id: String identifier with class name and parameters.

        Returns:
            An instantiated splitter object ready for execution.

        Raises:
            ValueError: If the identifier does not match any registered class.
        """
        class_name = splitter_id[: splitter_id.find(ID_SPLIT_SYMBOL)]
        splitter_class = self.SPLITTER_CLASS_MAPPING.get(class_name)
        if splitter_class is None:
            raise ValueError(f"{splitter_id} is not a valid splitter id")
        return splitter_class.build_from_id(splitter_id)

    # ── AA score computation ──────────────────────────────────────────────
    def _analyze_aa_score(self, data, score_table):
        """Compute per-feature pass-rate weights and store as 'aa score'."""

        self._feature_weights = {}
        aa_rows = []
        pass_cols = [c for c in score_table.columns if "pass" in c.lower()]
        for col in pass_cols:
            parts = _resolve_column_parts(col)
            if parts is None:
                continue
            feature, raw_test, group = parts
            if feature == "mean":
                continue
            test_name = normalize_test_name(raw_test)
            col_data = score_table[col]
            pass_rate = (
                sum(1 for v in col_data if _is_passed(v)) / len(col_data)
                if len(col_data) > 0
                else 0.0
            )
            weight = 1 - abs(self.alpha - pass_rate)
            index_label = f"{feature} {test_name} {group}".strip()
            self._feature_weights[index_label] = weight
            aa_rows.append({
                "_idx": index_label,
                "score": weight,
                "pass": weight >= self.threshold,
            })
        result_ds = self._build_aa_score_dataset(aa_rows)
        self.key = "aa score"
        return self._set_value(data, result_ds)

    @staticmethod
    def _build_aa_score_dataset(rows: list[dict[str, Any]]) -> SmallDataset:
        if not rows:
            return SmallDataset.from_dict([{}], roles={})
        df = pd.DataFrame(rows).set_index("_idx")
        return SmallDataset(
            roles={"score": StatisticRole(), "pass": StatisticRole()},
            data=df,
        )

    # ── Best split selection ──────────────────────────────────────────────

    def _analyze_best_split(
        self,
        data: ExperimentData,
        score_table: Dataset,
        if_param_scores: Dataset | None = None,
    ) -> ExperimentData:
        """Orchestrates identification and application of the best split."""
        best_split_id, best_data = self._get_best_split(data, score_table, if_param_scores)
        return self._set_best_split(best_data, best_split_id)

    def _get_best_split(
        self,
        data: ExperimentData,
        score_table: Dataset,
        if_param_scores: Dataset | None = None,
    ) -> tuple[str, ExperimentData]:
        """Determine the best split index and store its statistics.

        Returns:
            Tuple of (best_split_id, updated ExperimentData).
        """
        best_index = self._find_best_index(score_table, if_param_scores)
        best_split_id = self._extract_splitter_id(score_table, best_index)

        row_df = score_table.data.iloc[[best_index]]
        best_score_stat = SmallDataset(
            roles={col: score_table.roles.get(col, InfoRole()) for col in row_df.columns},
            data=row_df,
        )
        self.key = "best split statistics"
        result_data = self._set_value(data, best_score_stat)
        return best_split_id, result_data

    def _find_best_index(
        self,
        score_table: Dataset,
        if_param_scores: Dataset | None,
    ) -> int:
        """Select the row index with the highest composite score."""
        if if_param_scores is not None or not self._feature_weights:
            return 0

        weighted_pvalues = self._compute_weighted_pvalues(score_table)
        mean_test_score = self._get_mean_test_score_column(score_table)

        score_col = (
            weighted_pvalues * AAScoreAnalyzer.PVALUE_WEIGHT
            + mean_test_score * AAScoreAnalyzer.TEST_SCORE_WEIGHT
        )
        return score_col.idxmax()

    def _compute_weighted_pvalues(self, score_table):
        weighted = None
        pval_cols = [c for c in score_table.columns if "p-value" in c.lower()]
        for col in pval_cols:
            parts = _resolve_column_parts(col)
            if parts is None:
                continue
            feature, raw_test, group = parts
            test_name = normalize_test_name(raw_test)
            lookup_key = f"{feature} {test_name} {group}".strip()
            weight = self._feature_weights.get(lookup_key, 0)
            if weight == 0:
                weight = self._feature_weights.get(test_name, 0)
            if weight <= 0:
                continue
            col_data = score_table.data[col].astype(float)
            contribution = col_data * weight
            weighted = contribution if weighted is None else weighted + contribution
        if weighted is None:
            return pd.Series(0.0, index=range(len(score_table)))
        return weighted / len(self._feature_weights)

    @staticmethod
    def _get_mean_test_score_column(score_table: Dataset) -> pd.Series | float:
        """Extract the 'mean test score' column or return 0."""
        if "mean test score" in score_table.columns:
            return score_table.data["mean test score"].astype(float)
        return 0.0

    @staticmethod
    def _extract_splitter_id(score_table: Dataset, best_index: int) -> str:
        """Get the splitter ID for the best row."""
        if "splitter_id" in score_table.columns:
            return score_table.data.loc[best_index, "splitter_id"]
        return f"AASplitter{ID_SPLIT_SYMBOL}rs {int(best_index)}{ID_SPLIT_SYMBOL}"

    def _set_best_split(
        self, data: ExperimentData, best_splitter_id: str
    ) -> ExperimentData:
        """Saves the optimal splitter configuration and executes it.

        Args:
            data: The experiment data container.
            best_splitter_id: Identifier of the best-performing split.

        Returns:
            Updated ExperimentData with the best splitter applied.
        """
        self.key = "best splitter"

        strat_cols = data.ds.search_columns(StratificationRole())
        if strat_cols:
            cleaned_ds = data.ds.dropna(subset=strat_cols)
            if len(cleaned_ds) < len(data.ds):
                data = data.copy(data=cleaned_ds)

        result = data.set_value(
            ExperimentDataEnum.variables, self.id, best_splitter_id, self.key
        )
        best_splitter = self.build_splitter_from_id(best_splitter_id)
        best_splitter.save_groups = False
        best_splitter.constant_key = False
        best_splitter.key = "best"
        return best_splitter.execute(result)

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _get_if_param_scores(data: ExperimentData) -> Dataset | None:
        """Retrieve IfParamsExperiment scores if they exist."""
        ids = data.get_ids(IfParamsExperiment, ExperimentDataEnum.analysis_tables)
        table_ids = ids.get("IfParamsExperiment", {}).get("analysis_tables", [])
        if not table_ids:
            return None
        return data.analysis_tables[table_ids[0]]
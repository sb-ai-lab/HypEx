from ..analyzers.aa import AAScoreAnalyzer
from ..dataset import Dataset, ExperimentData, SmallDataset
from ..reporters.aa import AABestSplitReporter, AAPassedReporter
from ..reporters.abstract import TestDictReporter
from ..utils import ExperimentDataEnum
from ..utils.constants import (
    ID_SPLIT_SYMBOL,
    TEST_NAME_NORMALIZATION,
    normalize_test_name,
)
from ..utils.enums import RenameEnum
from .base import Output


class AAOutput(Output):
    best_split: Dataset
    experiments: Dataset
    aa_score: Dataset
    best_split_statistic: Dataset

    def __init__(self):
        super().__init__(
            resume_reporter=AAPassedReporter(),
            additional_reporters={"best_split": AABestSplitReporter()}
        ) 
    def _extract_experiments(self, experiment_data: ExperimentData):
        id_ = experiment_data.get_one_id(
            "ParamsExperiment", ExperimentDataEnum.analysis_tables
        )
        raw_table = experiment_data.analysis_tables[id_]

        # raw_table is a SmallDataset where each row = one iteration,
        # columns are like "pre_spends┆StatsTTest┆pass┆test_1", "splitter_id", etc.
        pdf = raw_table.data  # pd.DataFrame

        # Build the legacy-format experiments table directly
        result_rows = []
        for row_idx in range(len(pdf)):
            row: dict = {}
            for col in pdf.columns:
                val = pdf.iloc[row_idx].get(col)
                if ID_SPLIT_SYMBOL in str(col):
                    # "feature┆Executor┆metric┆group" → "feature Executor metric group"
                    parts = str(col).split(ID_SPLIT_SYMBOL)
                    if len(parts) >= 4:
                        feature, executor, metric, group = parts[0], parts[1], parts[2], parts[3]
                        norm_executor = normalize_test_name(executor)
                        new_col = f"{feature} {norm_executor} {metric} {group}"
                    elif len(parts) == 3:
                        feature, executor, metric = parts
                        norm_executor = normalize_test_name(executor)
                        new_col = f"{feature} {norm_executor} {metric}"
                    else:
                        new_col = str(col).replace(ID_SPLIT_SYMBOL, " ")
                else:
                    new_col = str(col)
                row[new_col] = val
            result_rows.append(row)

        if result_rows:
            self.experiments = SmallDataset.from_dict(result_rows, roles={})
        else:
            self.experiments = SmallDataset.from_dict([{"feature": [], "group": []}], roles={})

        self.experiments = self._replace_splitters(self.experiments, RenameEnum.columns)

    def _extract_aa_score(self, experiment_data: ExperimentData):
        def get_analyzer_id(key: str):
            target_id = [i for i in aa_score_analyser_ids if i.endswith(key)]
            if len(target_id):
                return target_id[0]
            raise ValueError("Result of AAScoreAnalyzer does not found.")

        aa_score_analyser_ids = experiment_data.get_ids(
            AAScoreAnalyzer, ExperimentDataEnum.analysis_tables
        )[AAScoreAnalyzer.__name__][ExperimentDataEnum.analysis_tables.value]

        self.aa_score = experiment_data.analysis_tables[get_analyzer_id("aa score")]
        self.aa_score = self._replace_splitters(self.aa_score, RenameEnum.index)

        self.best_split_statistic = experiment_data.analysis_tables[
            get_analyzer_id("best split statistics")
        ]

        # --- normalize column names in best_split_statistic ---
        # Rename "StatsTTest pass" → "TTest pass" etc.
        rename_map = {}
        for col in self.best_split_statistic.columns:
            for raw, norm in TEST_NAME_NORMALIZATION.items():
                if col.startswith(raw + " ") and raw != norm:
                    rename_map[col] = col.replace(raw, norm, 1)
                    break
        if rename_map:
            try:
                self.best_split_statistic.data = (
                    self.best_split_statistic.data.rename(columns=rename_map)
                )
                self.best_split_statistic._roles = {
                    rename_map.get(c, c): r
                    for c, r in self.best_split_statistic._roles.items()
                }
            except Exception:
                pass

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_experiments(experiment_data)
        self._extract_aa_score(experiment_data)

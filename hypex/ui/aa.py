from ..analyzers.aa import AAScoreAnalyzer
from ..dataset import Dataset, ExperimentData, InfoRole, SmallDataset, StatisticRole
from ..reporters.aa import AABestSplitReporter, AAPassedReporter
from ..utils import ExperimentDataEnum, _parse_metric_col
from ..utils.constants import (
    ID_SPLIT_SYMBOL,
    NAME_BORDER_SYMBOL,
    TEST_NAME_NORMALIZATION,
)
from ..utils.enums import RenameEnum
from ..utils.naming import normalize_test_name
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

    def _extract_best_split_statistic(self, experiment_data: ExperimentData):
        aa_score_analyser_ids = experiment_data.get_ids(
            AAScoreAnalyzer, ExperimentDataEnum.analysis_tables
        )[AAScoreAnalyzer.__name__][ExperimentDataEnum.analysis_tables.value]

        best_split_id = None
        for id_ in aa_score_analyser_ids:
            if id_.endswith("best split statistics"):
                best_split_id = id_
                break

        if best_split_id is None:
            self.best_split_statistic = SmallDataset.create_empty()
            return

        raw_table = experiment_data.analysis_tables[best_split_id]
        if raw_table.is_empty():
            self.best_split_statistic = SmallDataset.create_empty()
            return

        records = raw_table.to_records()
        if not records:
            self.best_split_statistic = SmallDataset.create_empty()
            return

        row = records[0]

        feature_groups: set[tuple[str, str]] = set()
        test_names: set[str] = set()

        for k in row.keys():
            if NAME_BORDER_SYMBOL in k:
                continue
            feature, test, metric, group = _parse_metric_col(k)
            if feature and feature != "mean":
                feature_groups.add((feature, group))
                if test and test != "GroupDifference":
                    test_names.add(normalize_test_name(test))

        order_map = {"TTest": 0, "KSTest": 1, "Chi2Test": 2, "ZTest": 3}
        ordered_tests = sorted(test_names, key=lambda x: order_map.get(x, 99))

        result_rows = []
        for feature, group in sorted(feature_groups):
            rec = {
                "feature": feature,
                "group": group,
                "control mean": None,
                "test mean": None,
                "difference": None,
                "difference %": None,
            }

            for k, v in row.items():
                f, t, m, g = _parse_metric_col(k)
                if f == feature and g == group and t == "GroupDifference":
                    if m == "control mean":
                        rec["control mean"] = v
                    elif m == "test mean":
                        rec["test mean"] = v
                    elif m == "difference":
                        rec["difference"] = v
                    elif m == "difference %":
                        rec["difference %"] = v

            for tn in ordered_tests:
                rec[f"{tn} pass"] = None
                rec[f"{tn} p-value"] = None
                for k, v in row.items():
                    f, t, m, g = _parse_metric_col(k)
                    if f == feature and g == group and normalize_test_name(t) == tn:
                        if m == "pass":
                            is_significant = str(v).strip().upper() in ("OK", "TRUE", "1")
                            rec[f"{tn} pass"] = "NOT OK" if is_significant else "OK"
                        elif m == "p-value":
                            rec[f"{tn} p-value"] = v

            result_rows.append(rec)

        roles = {}
        if result_rows:
            for c in result_rows[0]:
                if c in ("feature", "group"):
                    roles[c] = InfoRole()
                else:
                    roles[c] = StatisticRole()

        self.best_split_statistic = SmallDataset.from_dict(result_rows, roles=roles)

    def _extract_experiments(self, experiment_data: ExperimentData):
        id_ = experiment_data.get_one_id(
            "ParamsExperiment", ExperimentDataEnum.analysis_tables
        )
        raw_table = experiment_data.analysis_tables[id_]
        pdf = raw_table.data
        result_rows = []
        for row_idx in range(len(pdf)):
            row: dict = {}
            for col in pdf.columns:
                val = pdf.iloc[row_idx].get(col)
                col_str = str(col)
                if NAME_BORDER_SYMBOL in col_str:
                    continue
                if ID_SPLIT_SYMBOL in col_str:
                    parts = col_str.split(ID_SPLIT_SYMBOL)
                    norm_parts = [normalize_test_name(p) for p in parts]
                    new_col = " ".join(norm_parts)
                else:
                    new_col = col_str

                for raw_name, norm_name in TEST_NAME_NORMALIZATION.items():
                    if raw_name != norm_name:
                        new_col = new_col.replace(raw_name, norm_name)

                new_col = new_col.replace(" all", "")

                if "pass" in new_col and not new_col.startswith("mean "):
                    if val is not None and not (isinstance(val, float) and val != val):
                        if isinstance(val, str):
                            val = val.strip().lower() in ("true", "1", "ok")
                        else:
                            val = bool(val)

                row[new_col] = val
            result_rows.append(row)

        if result_rows:
            self.experiments = SmallDataset.from_dict(result_rows, roles={})
        else:
            self.experiments = SmallDataset.from_dict(
                [{"feature": [], "group": []}], roles={}
            )
        self.experiments = self._replace_splitters(
            self.experiments, RenameEnum.columns
        )

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
        self._extract_best_split_statistic(experiment_data)

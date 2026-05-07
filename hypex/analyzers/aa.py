from __future__ import annotations
from typing import Any, ClassVar
import numpy as np
import pandas as pd

from ..comparators import GroupChi2Test, GroupKSTest, GroupTTest
from ..comparators import StatsTTest, StatsChi2Test, StatsZTest
from ..dataset import Dataset, ExperimentData, StatisticRole, InfoRole
from ..dataset.dataset import SmallDataset
from ..executor import Executor
from ..experiments.base_complex import IfParamsExperiment, ParamsExperiment
from ..reporters.aa import OneAADictReporter
from ..splitters import AASplitter, AASplitterWithStratification
from ..utils import ID_SPLIT_SYMBOL, BackendsEnum, ExperimentDataEnum, NotFoundInExperimentDataError

class OneAAStatAnalyzer(Executor):
    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(ExperimentDataEnum.analysis_tables, self.id, value)

    def execute(self, data: ExperimentData) -> ExperimentData:
        analysis_tests: list[type] = [GroupTTest, GroupKSTest, GroupChi2Test,
                                      StatsTTest, StatsChi2Test, StatsZTest]
        executor_ids = data.get_ids(analysis_tests, searched_space=ExperimentDataEnum.analysis_tables)
        analysis_data: dict[str, float] = {}

        for class_, spaces in executor_ids.items():
            analysis_ids = spaces.get("analysis_tables", [])
            if len(analysis_ids) > 0:
                t_data = (
                    data.analysis_tables[analysis_ids[0]].append([data.analysis_tables[k] for k in analysis_ids[1:]])
                    if len(analysis_ids) > 1 else data.analysis_tables[analysis_ids[0]]
                )
                for field in ["p-value", "pass"]:
                    analysis_data[f"mean{ID_SPLIT_SYMBOL}{class_}{ID_SPLIT_SYMBOL}{field}{ID_SPLIT_SYMBOL}all"] = t_data[field].mean()

        analysis_data["mean test score"] = 0
        sum_weight = 0
        analysis_data = {key: (0 if np.isnan(value) else value) for key, value in analysis_data.items()}

        gtt_p = f"mean{ID_SPLIT_SYMBOL}GroupTTest{ID_SPLIT_SYMBOL}p-value{ID_SPLIT_SYMBOL}all"
        gks_p = f"mean{ID_SPLIT_SYMBOL}GroupKSTest{ID_SPLIT_SYMBOL}p-value{ID_SPLIT_SYMBOL}all"
        stt_p = f"mean{ID_SPLIT_SYMBOL}StatsTTest{ID_SPLIT_SYMBOL}p-value{ID_SPLIT_SYMBOL}all"
        chi2_p = f"mean{ID_SPLIT_SYMBOL}GroupChi2Test{ID_SPLIT_SYMBOL}p-value{ID_SPLIT_SYMBOL}all"

        if gtt_p in analysis_data and gks_p in analysis_data:
            analysis_data["mean test score"] = analysis_data[gtt_p] + 2 * analysis_data[gks_p]
            sum_weight += 3
        if stt_p in analysis_data:
            analysis_data["mean test score"] = analysis_data[stt_p]
            sum_weight += 3
        if chi2_p in analysis_data:
            analysis_data["mean test score"] += 2 * analysis_data[chi2_p]
            sum_weight += 2
        if sum_weight:
            analysis_data["mean test score"] /= sum_weight

        return self._set_value(data, SmallDataset.from_dict(
            analysis_data, {field: StatisticRole(float) for field in analysis_data}
        ))


class AAScoreAnalyzer(Executor):
    AA_SPLITER_CLASS_MAPPING: ClassVar[dict] = {
        class_.__name__: class_ for class_ in [AASplitter, AASplitterWithStratification]
    }

    def __init__(self, alpha: float = 0.05, key: str = ""):
        super().__init__(key=key)
        self.alpha = alpha
        self.__feature_weights = {}
        self.threshold = 1 - (self.alpha * 1.2)

    def _set_value(self, data: ExperimentData, value: Any, key: Any = None) -> ExperimentData:
        return data.set_value(ExperimentDataEnum.analysis_tables, executor_id=self.id, key=self.key, value=value)

    def _analyze_aa_score(self, data: ExperimentData, score_table: Dataset) -> ExperimentData:
        self.__feature_weights = {}
        pass_cols = [c for c in score_table.columns if "pass" in c.lower()]
        
        for col in pass_cols:
            test_name = col.split(" pass")[0].strip() if ID_SPLIT_SYMBOL not in col else col.split(ID_SPLIT_SYMBOL)[1]
            
            col_data = score_table[col]
            passed = sum(1 for v in col_data if str(v).strip().upper() in ["OK", "TRUE", "1"])
            pass_rate = passed / len(col_data) if len(col_data) > 0 else 0
            self.__feature_weights[test_name] = 1 - abs(self.alpha - pass_rate)

        flat_row = {}
        for test_name, weight in self.__feature_weights.items():
            flat_row[f"{test_name} score"] = weight
            flat_row[f"{test_name} pass"] = weight >= self.threshold
            
        self.key = "aa score"
        return self._set_value(data, SmallDataset.from_dict([flat_row], roles={}))

    def build_splitter_from_id(self, splitter_id: str):
        splitter_class = self.AA_SPLITER_CLASS_MAPPING.get(splitter_id[: splitter_id.find(ID_SPLIT_SYMBOL)])
        if splitter_class is None:
            raise ValueError(f"{splitter_id} is not a valid splitter id")
        return splitter_class.build_from_id(splitter_id)

    def _get_best_split(self, data, score_table, if_param_scores=None):
        if if_param_scores is not None:
            best_index = 0
        elif not self.__feature_weights:
            best_index = 0
        else:
            pvalue_weight = 2 / 3
            test_score_weight = 1 / 3
            weighted_pvalues = None

            pval_cols = [c for c in score_table.columns if "p-value" in c.lower()]
            for col in pval_cols:
                test_name = col.split(" p-value")[0].strip() if " " in col else col.split(ID_SPLIT_SYMBOL)[1]
                weight = self.__feature_weights.get(test_name, 0)
                if weight > 0:
                    col_data = score_table.data[col].astype(float)
                    weighted_pvalues = (col_data * weight) if weighted_pvalues is None else weighted_pvalues + (col_data * weight)

            if weighted_pvalues is None:
                weighted_pvalues = pd.Series([0.0] * len(score_table))
            else:
                weighted_pvalues = weighted_pvalues / len(self.__feature_weights)

            mean_test_score = score_table["mean test score"] if "mean test score" in score_table.columns else 0.0
            score_col = (weighted_pvalues * pvalue_weight + mean_test_score * test_score_weight)
            best_index = score_col.idxmax()

        print(f"best_index = {best_index}")

        if "splitter_id" in score_table.columns:
            best_split_id = score_table.data.loc[best_index, "splitter_id"]
        else:
            best_split_id = f"AASplitter{ID_SPLIT_SYMBOL}rs {int(best_index)}{ID_SPLIT_SYMBOL}"


        row_df = score_table.data.iloc[[best_index]]
        best_score_stat = SmallDataset(
            roles={col: score_table.roles.get(col, InfoRole()) for col in row_df.columns},
            data=row_df
        )

        self.key = "best split statistics"
        result = self._set_value(data, best_score_stat)
        return {"best_split_id": best_split_id, "data": result}

    def _set_best_split(self, data: ExperimentData, best_splitter_id: str) -> ExperimentData:
        self.key = "best splitter"
        result = data.set_value(ExperimentDataEnum.variables, self.id, best_splitter_id, self.key)
        best_splitter = self.build_splitter_from_id(best_splitter_id)
        best_splitter.save_groups = False
        best_splitter.constant_key = False
        best_splitter.key = "best"
        return best_splitter.execute(result)

    def _analyze_best_split(self, 
                            data: ExperimentData, 
                            score_table: Dataset, 
                            if_param_scores: Dataset | None = None) -> ExperimentData:
        best_split = self._get_best_split(data, score_table, if_param_scores)
        return self._set_best_split(best_split["data"], best_split["best_split_id"])

    def execute(self, data: ExperimentData) -> ExperimentData:
        print(f"[DEBUG] AAScoreAnalyzer | id={id(data)} keys={list(data.analysis_tables.keys())}")
        print(f"[DEBUG] AAScoreAnalyzer | keys в analysis_tables: {list(data.analysis_tables.keys())}")

        param_experiment_id = data.get_one_id(ParamsExperiment, ExperimentDataEnum.analysis_tables)
        ifparam_experiment_id = data.get_ids(IfParamsExperiment, ExperimentDataEnum.analysis_tables)
        score_table = data.analysis_tables[param_experiment_id]
        if_param_scores = (
            None
            if len(ifparam_experiment_id["IfParamsExperiment"]["analysis_tables"]) == 0
            else data.analysis_tables[ifparam_experiment_id["IfParamsExperiment"]["analysis_tables"][0]]
        )
        data = self._analyze_aa_score(data, score_table)
        return self._analyze_best_split(data, score_table, if_param_scores)
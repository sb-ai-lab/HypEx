from __future__ import annotations

from typing import Any

from ..analyzers.matching import MatchingAnalyzer
from ..dataset import Dataset, ExperimentData
from ..reporters.abstract import Reporter
from ..utils import ExperimentDataEnum, ID_SPLIT_SYMBOL
from ..utils.errors import NotFoundInExperimentDataError


class MatchingReporter(Reporter):
    """
    Репортер для основных метрик матчинга (ATT, ATC, ATE).
    Забирает готовый Dataset из MatchingAnalyzer или GroupExperiment.
    """

    def report(self, data: ExperimentData) -> Dataset:
        # 1. Пытаемся забрать результат обычного Matching (MatchingAnalyzer)
        try:
            analyzer_id = data.get_one_id(MatchingAnalyzer, ExperimentDataEnum.analysis_tables)
            result_ds = data.analysis_tables[analyzer_id]
            print(f"[DEBUG MatchingReporter] analyzer_id: {analyzer_id}")
            print(f"[DEBUG MatchingReporter] result_ds type: {type(result_ds)}")
            print(f"[DEBUG MatchingReporter] result_ds roles: {list(result_ds.roles.keys())}")
            print(f"[DEBUG MatchingReporter] result_ds columns: {result_ds.columns}")
            print(f"[DEBUG MatchingReporter] result_ds data empty? {result_ds.data.empty if hasattr(result_ds.data, 'empty') else 'N/A'}")
            return result_ds
        except NotFoundInExperimentDataError:
            pass

        # 2. Если был group_match=True, результат лежит в GroupExperiment.
        # ВАЖНО: Передаем имя класса строкой ("GroupExperiment"), чтобы избежать 
        # циклического импорта между reporters и experiments.base_complex!
        try:
            group_exp_id = data.get_one_id("GroupExperiment", ExperimentDataEnum.analysis_tables)
            return data.analysis_tables[group_exp_id]
        except NotFoundInExperimentDataError:
            pass

        return Dataset.create_empty()


class MatchingQualityReporter(Reporter):
    """
    Репортер для тестов качества матчинга (T-Test, Chi2, KS).
    Собирает результаты OnRoleExperiment из analysis_tables.
    """

    def report(self, data: ExperimentData) -> dict[str, Dataset]:
        quality_results = {}

        for exec_id, table in data.analysis_tables.items():
            # Фильтруем только итоговые таблицы тестов (исключаем MatchingAnalyzer, Bias и сырые stats)
            is_test = any(test in exec_id for test in ["TTest", "Chi2Test", "KSTest", "UTest"])
            is_raw_stats = exec_id.endswith("stats") or exec_id.endswith("┆stats")

            if is_test and not is_raw_stats:
                # exec_id обычно имеет вид "StatsTTest┴┴feat_num_1" или "GroupTTest┴┴feat_cat"
                parts = exec_id.split(ID_SPLIT_SYMBOL)
                feature_name = parts[-1] if len(parts) > 1 else exec_id

                # Группируем тесты по имени фичи
                if feature_name not in quality_results:
                    quality_results[feature_name] = table
                else:
                    # Если на одну фичу несколько тестов, аппендим их
                    quality_results[feature_name] = quality_results[feature_name].append(table)

        return quality_results
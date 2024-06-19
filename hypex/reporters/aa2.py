from typing import Dict, Any
from itertools import product

from hypex.analyzers.aa2 import AAScoreAnalyzer
from hypex.dataset import ExperimentData, Dataset, InfoRole, StatisticRole
from hypex.reporters import Reporter
from hypex.utils import AAReporterModsEnum, ID_SPLIT_SYMBOL, ExperimentDataEnum


class AAReporter(Reporter):
    def __init__(self, mode: AAReporterModsEnum = AAReporterModsEnum.passed):
        self.mode = mode

    def _reformat_aa_score_table(self, table: Dataset) -> Dataset:
        result = {}
        for ind in table.index:
            splitted_index = ind.split(ID_SPLIT_SYMBOL)
            row_index = f"{splitted_index[0]}{ID_SPLIT_SYMBOL}{splitted_index[-1]}"
            value = table.get_values(ind, "pass")
            if row_index not in result:
                result[row_index] = {splitted_index[1]: value}
            else:
                result[row_index][splitted_index[1]] = value
        return Dataset.from_dict(result, roles={}).transpose() * 1

    def _reformat_best_split_table(self, table: Dataset) -> Dataset:
        passed = table.loc[:, [c for c in table.columns if c.endswith("pass")]]
        new_index = table.apply(
            lambda x: f"{x['feature']}{ID_SPLIT_SYMBOL}{x['group']}",
            {"index": InfoRole()},
            axis=1,
        )
        passed.index = new_index.get_values(column="index")
        passed = passed.rename(
            names={c: c[: c.rfind("pass") - 1] for c in passed.columns}
        )
        passed = passed.replace("OK", 1).replace("NOT OK", 0)
        return passed

    def _pass_mode(self, analyzer_tables: Dict[str, Dataset]):
        score_table = self._reformat_aa_score_table(analyzer_tables["aa score"])
        best_split_table = self._reformat_best_split_table(
            analyzer_tables["best split statistics"]
        )
        resume_table = score_table * best_split_table
        resume_table = resume_table.apply(
            lambda x: "OK" if x.sum() > 0 else "NOT OK",
            axis=1,
            role={"result": StatisticRole()},
        )
        result = score_table.merge(
            best_split_table,
            suffixes=(" aa test", " best split"),
            left_index=True,
            right_index=True,
        )
        result = result.merge(resume_table, left_index=True, right_index=True)

        result = result.replace(0, "NOT OK").replace(1, "OK")
        splitted_index = [i.split(ID_SPLIT_SYMBOL) for i in result.index]
        result.add_column([i[0] for i in splitted_index], role={"feature": InfoRole()})
        result.add_column([i[1] for i in splitted_index], role={"group": InfoRole()})
        result.index = range(len(splitted_index))
        return result

    def report(self, data: ExperimentData) -> Dataset:
        if self.mode == AAReporterModsEnum.passed:
            analyser_ids = data.get_ids(
                AAScoreAnalyzer, ExperimentDataEnum.analysis_tables
            )
            analyser_tables = {
                id_[id_.rfind(ID_SPLIT_SYMBOL) + 1 :]: data.analysis_tables[id_]
                for id_ in analyser_ids[AAScoreAnalyzer][
                    ExperimentDataEnum.analysis_tables.value
                ]
            }
            return self._pass_mode(analyser_tables)
        elif self.mode == AAReporterModsEnum.best_split:
            best_split_id = []
            for c in data.additional_fields.columns:
                if c.endswith("best"):
                    best_split_id = c
                    break

            markers = data.additional_fields.loc[:, best_split_id]
            markers = markers.rename({markers.columns[0]: "split"})
            return data.ds.merge(markers, left_index=True, right_index=True)

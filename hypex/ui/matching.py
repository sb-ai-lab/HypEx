from typing import Dict, Any

from .base import Output
from ..analyzers.matching import MatchingAnalyzer
from ..dataset import (
    Dataset,
    ExperimentData,
    AdditionalMatchingRole,
    StatisticRole,
    GroupingRole,
)
from ..reporters.matching import MatchingDictReporter
from ..utils import ID_SPLIT_SYMBOL


class MatchingOutput(Output):
    resume: Dataset
    full_data: Dataset

    def __init__(self, searching_class: type = MatchingAnalyzer):
        super().__init__(resume_reporter=MatchingDictReporter(searching_class))

    def _extract_full_data(self, experiment_data: ExperimentData, indexes):
        indexes.index = experiment_data.ds.index
        filtered_field = indexes.drop(
            indexes[indexes[indexes.columns[0]] == -1], axis=0
        )
        matched_data = experiment_data.ds.loc[
            list(map(lambda x: x[0], filtered_field.get_values()))
        ].rename({i: i + "_matched" for i in experiment_data.ds.columns})
        matched_data.index = filtered_field.index
        self.indexes = indexes
        self.full_data = experiment_data.ds.append(
            matched_data.reindex(experiment_data.ds.index), axis=1
        )

    def extract(self, experiment_data: ExperimentData):
        resume = self.resume_reporter.report(experiment_data)
        reformatted_resume: Dict[str, Any] = {}
        for key, value in resume.items():
            if ID_SPLIT_SYMBOL in key:
                keys = key.split(ID_SPLIT_SYMBOL)
                temp_key = keys[0] if len(keys) < 3 else f"{keys[2]} {keys[0]}"
                if temp_key not in reformatted_resume:
                    reformatted_resume[temp_key] = {}
                reformatted_resume[temp_key].update({keys[1]: value})
        if "indexes" in reformatted_resume.keys():
            group_indexes_id = experiment_data.ds.search_columns(GroupingRole())
            indexes = [
                Dataset.from_dict(
                    {"indexes": list(map(int, values.split("||")))},
                    index=experiment_data.ds[
                        experiment_data.ds[group_indexes_id] == group
                    ].index,
                    roles={"indexes": StatisticRole()},
                )
                for group, values in reformatted_resume.pop("indexes").items()
            ]
            indexes = indexes[0].append(indexes[1:]).sort()
        else:
            indexes = Dataset.from_dict(
                {"indexes": list(map(int, resume["indexes"].split("||")))},
                roles={"indexes": AdditionalMatchingRole()},
            )
        self.resume = Dataset.from_dict(
            reformatted_resume,
            roles={
                column: StatisticRole() for column in list(reformatted_resume.keys())
            },
        )
        self._extract_full_data(
            experiment_data,
            indexes,
        )

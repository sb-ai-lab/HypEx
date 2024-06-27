from .base import Output
from ..analyzers.aa2 import AAScoreAnalyzer
from ..dataset import Dataset, ExperimentData
from ..reporters.aa2 import AAPassedReporter, AABestSplitReporter
from ..utils import ExperimentDataEnum


class AAOutput(Output):
    best_split: Dataset
    experiments: Dataset
    aa_score: Dataset
    best_split_statistic: Dataset

    def __init__(self):
        super().__init__(
            resume_reporter=AAPassedReporter(),
            additional_reporters={"best_split": AABestSplitReporter()},
        )

    def _extract_experiments(self, experiment_data: ExperimentData):
        pass

    def _extract_aa_score(self, experiment_data: ExperimentData):
        def get_analyzer_id(key: str):
            target_id = [i for i in aa_score_analyser_ids if i.endswith(key)]
            if len(target_id):
                return target_id[0]
            else:
                raise ValueError("Result of AAScoreAnalyzer does not found.")

        aa_score_analyser_ids = experiment_data.get_ids(
            AAScoreAnalyzer, ExperimentDataEnum.analysis_tables
        )[AAScoreAnalyzer][ExperimentDataEnum.analysis_tables.value]

        self.aa_score = experiment_data.analysis_tables[get_analyzer_id("aa_score")]
        self.best_split_statistic = experiment_data.analysis_tables[
            get_analyzer_id("best split statistics")
        ]

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        self._extract_experiments(experiment_data)
        self._extract_aa_score(experiment_data)

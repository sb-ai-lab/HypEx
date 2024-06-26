from .base import Output
from ..dataset import Dataset
from ..reporters.aa2 import AAPassedReporter, AABestSplitReporter


class AAOutput(Output):
    best_split: Dataset

    def __init__(self):
        super().__init__(
            resume_reporter=AAPassedReporter(),
            additional_reporters={"best_split": AABestSplitReporter()},
        )

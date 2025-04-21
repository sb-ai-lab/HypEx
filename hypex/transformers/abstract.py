
from ..dataset import Dataset, ExperimentData
from ..executor import Calculator


class Transformer(Calculator):
    @property
    def _is_transformer(self):
        return True

    @classmethod
    def calc(cls, data: Dataset, **kwargs):
        return cls.calc(data, **kwargs)

    def execute(self, data: ExperimentData) -> ExperimentData:
        data = data.copy(data=self.calc(data=data.ds))
        return data

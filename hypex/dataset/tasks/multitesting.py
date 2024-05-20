from statsmodels.stats.multitest import multipletests  # type: ignore

from .abstract import Task
from .. import Dataset, StatisticRole


class ABMultiTest(Task):
    def __init__(self, method, alpha: float = 0.05):
        self.method = method
        self.alpha = alpha
        super().__init__()

    @staticmethod
    def multitest_result_to_dataset(result):
        result = {"rejected": result[0], "new p-values": result[1]}
        return Dataset.from_dict(
            result, roles={"rejected": StatisticRole(), "new p-values": StatisticRole()}
        )

    def _calc_pandas(self, data: Dataset, **kwargs):
        p_values = data.data.values.flatten()
        return self.multitest_result_to_dataset(
            multipletests(p_values, method=self.method, alpha=self.alpha, **kwargs)
        )

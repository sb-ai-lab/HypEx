from typing import Callable

from scipy.stats import ttest_ind, ks_2samp

from typing import Union, List, Callable

import pandas as pd

from hypex.dataset import Dataset, StatisticRole
from hypex.dataset.tasks.abstract import CompareTask, Task
from hypex.utils import BackendsEnum


class StatTest(CompareTask):
    def __init__(self, test_function: Callable, alpha: float = 0.05):
        super().__init__()
        self.test_function = test_function
        self.alpha = alpha

    @staticmethod
    def check_other(other: Union[Dataset, List[Dataset]]):
        if len(other) == 0:
            raise ValueError("No other dataset provided")

    @staticmethod
    def check_dataset(data: Dataset):
        if len(data.columns) != 1:
            raise ValueError("Data must be one-dimensional")

    def _calc_pandas(
        self, data: Dataset, other: Union[Dataset, List[Dataset], None] = None, **kwargs
    ) -> Union[float, Dataset]:
        other = other or []
        self.check_other(other)

        if isinstance(other, Dataset):
            other = [other]

        self.check_dataset(data)

        result = []
        for o in other:
            self.check_dataset(o)
            one_result = self.test_function(
                data.data.values.flatten(), o.data.values.flatten()
            )
            result.append(
                {
                    "p-value": one_result.pvalue,
                    "statistic": one_result.statistic,
                    "pass": one_result.pvalue < self.alpha,
                }
            )

        df_result = pd.DataFrame(result)
        return Dataset(
            roles={str(f): StatisticRole() for f in df_result.columns},
            data=df_result,
            backend=BackendsEnum.pandas,
        )


class TTest(StatTest):
    def __init__(self, alpha: float = 0.05):
        super().__init__(ttest_ind, alpha=alpha)


class KSTest(StatTest):
    def __init__(self, alpha: float = 0.05):
        super().__init__(ks_2samp, alpha=alpha)


class Chi2Test(Task):
    pass
    # def _calc_pandas(
    #         self, data: Dataset, other: Union[Dataset, List[Dataset], None] = None, **kwargs
    # ) -> Union[float, Dataset]:
    #     df = df.sort_values(by=treated_column)
    #     groups = df[treated_column].unique().tolist()
    #     all_pvalues = {}
    #     for target_field in self.target_fields:
    #         group_a, group_b = (
    #             df[df[treated_column] == groups[0]][target_field],
    #             df[df[treated_column] == groups[1]][target_field],
    #         )
    #         proportions = group_a.shape[0] / (group_a.shape[0] + group_b.shape[0])
    #         group_a = group_a.value_counts().rename(target_field).sort_index()
    #         group_b = group_b.value_counts().rename(target_field).sort_index()
    #         index = (
    #             pd.Series(group_a.index.tolist() + group_b.index.tolist())
    #             .unique()
    #             .tolist()
    #         )
    #         group_a, group_b = [
    #             group.reindex(index, fill_value=0) for group in [group_a, group_b]
    #         ]
    #         merged_data = pd.DataFrame(
    #             {
    #                 "index_x": group_a * (1 - proportions),
    #                 "index_y": group_b * proportions,
    #             }
    #         ).fillna(0)
    #         sub_group = merged_data.sum(axis=1).sort_values()
    #         _filter = sub_group <= 15
    #         if _filter.sum():
    #             other = {
    #                 "index_x": merged_data["index_x"][_filter].sum(),
    #                 "index_y": merged_data["index_y"][_filter].sum(),
    #             }
    #             merged_data.drop(sub_group[_filter].index, inplace=True)
    #             merged_data = pd.concat(
    #                 [merged_data, pd.DataFrame([other], index=["Other"])]
    #             )
    #         all_pvalues[target_field] = chi2_contingency(
    #             merged_data[["index_x", "index_y"]]
    #         ).pvalue
    #     return all_pvalues

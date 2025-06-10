from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..comparators import TTest, UTest
from ..dataset import Dataset, ExperimentData, StatisticRole, TargetRole, TreatmentRole
from ..experiments.base import Executor
from ..extensions.statsmodels import MultiTest, MultitestQuantile
from ..utils import (
    ID_SPLIT_SYMBOL,
    NAME_BORDER_SYMBOL,
    ABNTestMethodsEnum,
    BackendsEnum,
    ExperimentDataEnum,
)


class ABAnalyzer(Executor):
    def __init__(
        self,
        multitest_method: ABNTestMethodsEnum | None = None,
        alpha: float = 0.05,
        equal_variance: bool = True,
        quantiles: float | list[float] | None = None,
        iteration_size: int = 20000,
        random_state: int | None = None,
        key: Any = "",
    ):
        self.multitest_method = multitest_method
        self.alpha = alpha
        self.equal_variance = equal_variance
        self.quantiles = quantiles
        self.iteration_size = iteration_size
        self.random_state = random_state
        super().__init__(key)

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables,
            self.id + key if key else self.id,
            value,
        )

    def execute_multitest(self, data: ExperimentData, p_values: Dataset, **kwargs):
        group_field = data.ds.search_columns(TreatmentRole())[0]
        target_fields = data.ds.search_columns(TargetRole(), search_types=[int, float])
        if self.multitest_method and len(data.groups[group_field]) > 2:
            if self.multitest_method != ABNTestMethodsEnum.quantile:
                multitest_result = MultiTest(self.multitest_method).calc(
                    p_values, **kwargs
                )
                groups = []
                for i in list(data.groups[group_field].keys())[1:]:
                    groups += [i] * len(target_fields)
                multitest_result = multitest_result.add_column(
                    groups
                    * (
                        len(multitest_result)
                        // len(target_fields)
                        // (len(data.groups[group_field]) - 1)
                    ),
                    role={"group": StatisticRole()},
                )

            else:
                multitest_result = Dataset.create_empty()
                for target_field in target_fields:
                    multitest_result = multitest_result.append(
                        MultitestQuantile(
                            self.alpha,
                            self.iteration_size,
                            self.equal_variance,
                            self.random_state,
                        ).calc(
                            p_values,
                            group_field=group_field,
                            target_field=target_field,
                            quantiles=self.quantiles,
                        )
                    )
            return self._set_value(data, multitest_result, key="MultiTest")
        return data

    def _add_pvalues(self, multitest_pvalues, value, field):
        if (
            self.multitest_method
            and field == "p-value"
            and self.multitest_method != "quantile"
        ):
            multitest_pvalues = multitest_pvalues.append(value)
        return multitest_pvalues

    def execute(self, data: ExperimentData) -> ExperimentData:
        executor_ids = data.get_ids([TTest, UTest])
        num_groups = len(data.groups[data.ds.search_columns(TreatmentRole())[0]]) - 1
        groups = list(data.groups[data.ds.search_columns(TreatmentRole())[0]].items())
        multitest_pvalues = Dataset.create_empty()
        analysis_data = {}
        for c, spaces in executor_ids.items():
            analysis_ids = spaces.get("analysis_tables", [])
            if len(analysis_ids) == 0:
                continue
            t_data = deepcopy(data.analysis_tables[analysis_ids[0]])
            for aid in analysis_ids[1:]:
                t_data = t_data.append(data.analysis_tables[aid])
            if len(analysis_ids) < len(t_data):
                analysis_ids *= num_groups
            t_data.data.index = analysis_ids
            for f in ["p-value", "pass"]:
                for i in range(0, len(analysis_ids), len(analysis_ids) // num_groups):
                    value = t_data.iloc[i : i + len(analysis_ids) // num_groups][f]
                    multitest_pvalues = self._add_pvalues(multitest_pvalues, value, f)
                    analysis_data[f"{c} {f} {groups[i // num_groups + 1][0]}"] = (
                        value.mean()
                    )
            if c not in ["UTest", "TTest"]:
                indexes = t_data.index
                values = t_data.data.values.tolist()
                for idx, value in zip(indexes, values):
                    name = idx.split(ID_SPLIT_SYMBOL)[-1]
                    analysis_data[
                        f"{c} {name[name.find(NAME_BORDER_SYMBOL) + 1 : name.rfind(NAME_BORDER_SYMBOL)]}"
                    ] = value[0]

        analysis_dataset = Dataset.from_dict(
            [analysis_data],
            {f: StatisticRole() for f in analysis_data},
            BackendsEnum.pandas,
        )
        data = self.execute_multitest(
            data,
            (
                multitest_pvalues
                if not multitest_pvalues.is_empty()
                and self.multitest_method != ABNTestMethodsEnum.quantile
                else data.ds
            ),
        )

        return self._set_value(data, analysis_dataset)

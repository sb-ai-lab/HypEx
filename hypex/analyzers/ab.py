from typing import Optional, Any, Union, List

from hypex.comparators import TTest, UTest
from hypex.dataset import (
    Dataset,
    ExperimentData,
    StatisticRole,
    TreatmentRole,
    TargetRole,
)
from hypex.extensions.statsmodels import ABMultiTest, ABMultitestQuantile
from hypex.experiments.base import Executor
from hypex.utils import (
    ID_SPLIT_SYMBOL,
    NAME_BORDER_SYMBOL,
    BackendsEnum,
    ExperimentDataEnum,
    ABNTestMethodsEnum,
)


class ABAnalyzer(Executor):

    def __init__(
        self,
        multitest_method: Optional[ABNTestMethodsEnum] = None,
        alpha: float = 0.05,
        equal_variance: bool = True,
        quantiles: Optional[Union[float, List[float]]] = None,
        iteration_size: int = 20000,
        random_state: Optional[int] = None,
        full_name: Optional[str] = None,
        key: Any = "",
    ):
        self.multitest_method = multitest_method
        self.alpha = alpha
        self.equal_variance = equal_variance
        self.quantiles = quantiles
        self.iteration_size = iteration_size
        self.random_state = random_state
        super().__init__(full_name, key)

    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables,
            self.id + key if key else self.id,
            str(self.full_name),
            value,
        )

    def execute_multitest(self, data: ExperimentData, p_values: Dataset, **kwargs):
        group_field = data.ds.get_columns_by_roles(TreatmentRole())[0]
        target_field = data.ds.get_columns_by_roles(TargetRole())[0]
        if self.multitest_method:
            multitest_result = (
                ABMultiTest(self.multitest_method).calc(p_values, **kwargs)
                if self.multitest_method != ABNTestMethodsEnum.quantile
                else ABMultitestQuantile(
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

    def _add_pvalues(self, multitest_pvalues, value, field, class_name):
        if (
            class_name == "TTest"
            and self.multitest_method
            and field == "p-value"
            and self.multitest_method != "quantile"
        ):
            multitest_pvalues = multitest_pvalues.append(value)
        return multitest_pvalues

    def execute(self, data: ExperimentData) -> ExperimentData:
        executor_ids = data.get_ids([TTest, UTest])
        multitest_pvalues = Dataset.create_empty()
        analysis_data = {}
        for c, spaces in executor_ids.items():
            analysis_ids = spaces.get("analysis_tables", [])
            if len(analysis_ids) == 0:
                continue
            t_data = data.analysis_tables[analysis_ids[0]]
            for aid in analysis_ids[1:]:
                t_data = t_data.append(data.analysis_tables[aid])
            t_data.data.index = analysis_ids * len(t_data)
            for f in ["p-value", "pass"]:
                value = t_data[f]
                multitest_pvalues = self._add_pvalues(
                    multitest_pvalues, value, f, c.__name__
                )
                analysis_data[f"{c.__name__} {f}"] = value.mean()
            if c.__name__ not in ["UTest", "TTest"]:
                indexes = t_data.index
                values = t_data.data.values.tolist()
                for idx, value in zip(indexes, values):
                    name = idx.split(ID_SPLIT_SYMBOL)[-1]
                    analysis_data[
                        f"{c.__name__} {name[name.find(NAME_BORDER_SYMBOL) + 1: name.rfind(NAME_BORDER_SYMBOL)]}"
                    ] = value[0]

        analysis_dataset = Dataset.from_dict(
            [analysis_data],
            {f: StatisticRole() for f in analysis_data},
            BackendsEnum.pandas,
        )
        data = self.execute_multitest(
            data, multitest_pvalues if multitest_pvalues.is_empty() else data.ds
        )

        return self._set_value(data, analysis_dataset)

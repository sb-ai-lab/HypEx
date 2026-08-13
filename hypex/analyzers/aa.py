from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np

from ..comparators import Chi2Test, GroupDifference, KSTest, TTest
from ..dataset import Dataset, ExperimentData, MaximizationRole, StatisticRole
from ..executor import Executor
from ..experiments.base_complex import IfParamsExperiment, ParamsExperiment
from ..reporters.aa import OneAADictReporter
from ..splitters import AASplitter, AASplitterWithStratification
from ..utils import ID_SPLIT_SYMBOL, BackendsEnum, ExperimentDataEnum


class OneAAStatAnalyzer(Executor):
    def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
        return data.set_value(ExperimentDataEnum.analysis_tables, self.id, value)

    def execute(self, data: ExperimentData) -> ExperimentData:
        analysis_tests: list[type] = [TTest, KSTest, Chi2Test]
        executor_ids = data.get_ids(
            analysis_tests, searched_space=ExperimentDataEnum.analysis_tables
        )
        # num_groups = len(data.groups[data.ds.search_columns(TreatmentRole())[0]]) - 1
        # groups = list(data.groups[data.ds.search_columns(TreatmentRole())[0]].items())
        # multitest_pvalues = Dataset.create_empty()
        # analysis_data = {}

        analysis_data: dict[str, float] = {}
        for class_, spaces in executor_ids.items():
            analysis_ids = spaces.get("analysis_tables", [])
            if len(analysis_ids) > 0:
                if len(analysis_ids) > 1:
                    t_data = data.analysis_tables[analysis_ids[0]].append(
                        [data.analysis_tables[k] for k in analysis_ids[1:]]
                    )
                else:
                    t_data = data.analysis_tables[analysis_ids[0]]
                # t_data.data.index = analysis_ids
                for field in ["p-value", "pass"]:
                    analysis_data[f"mean {class_} {field}"] = t_data[field].mean()
        analysis_data["mean test score"] = 0
        sum_weight = 0
        analysis_data = {
            key: (0 if np.isnan(value) else value)
            for key, value in analysis_data.items()
        }
        if (
            "mean TTest p-value" in analysis_data
            and "mean KSTest p-value" in analysis_data
        ):
            analysis_data["mean test score"] = (
                analysis_data["mean TTest p-value"]
                + 2 * analysis_data["mean KSTest p-value"]
            )
            sum_weight += 3
        if "mean Chi2Test p-value" in analysis_data:
            analysis_data["mean test score"] += (
                2 * analysis_data["mean Chi2Test p-value"]
            )
            sum_weight += 2
        if sum_weight:
            analysis_data["mean test score"] /= sum_weight

        analysis_dataset = Dataset.from_dict(
            [analysis_data],
            {field: StatisticRole() for field in analysis_data},
            BackendsEnum.pandas,
        )

        return self._set_value(data, analysis_dataset)


class AAScoreAnalyzer(Executor):
    """Scores the A/A iterations and picks the best split among them.

    By default the best split is the one with the highest homogeneity score. If a
    column carries ``MaximizationRole`` and ``maximize_group`` is set, the split is
    instead chosen among the iterations where no test flagged a difference on any
    feature: of those the one with the highest mean of the tagged metric in
    ``maximize_group`` wins. The metric itself is never tested for homogeneity.

    Args:
        alpha (float, optional): Significance level the per-feature A/A scores are
            compared against. Defaults to 0.05.
        key (str, optional): Executor key. Defaults to "".
        maximize_group (str, optional): Group the metric is maximized in, as labeled
            by the splitter ("control", "test_1", ...). Required whenever a column
            carries MaximizationRole, and meaningless without one. Defaults to None.
    """

    AA_SPLITER_CLASS_MAPPING: ClassVar[dict] = {
        class_.__name__: class_ for class_ in [AASplitter, AASplitterWithStratification]
    }
    PASS_COLUMN_MARKER: ClassVar[str] = f"{ID_SPLIT_SYMBOL}pass{ID_SPLIT_SYMBOL}"
    P_VALUE_COLUMN_MARKER: ClassVar[str] = f"{ID_SPLIT_SYMBOL}p-value{ID_SPLIT_SYMBOL}"
    GROUP_DIFFERENCE_MARKER: ClassVar[str] = (
        f"{ID_SPLIT_SYMBOL}{GroupDifference.__name__}{ID_SPLIT_SYMBOL}"
    )

    # TODO: rename alpha
    def __init__(
        self,
        alpha: float = 0.05,
        key: str = "",
        maximize_group: str | None = None,
    ):
        super().__init__(key=key)
        self.alpha = alpha
        self.maximize_group = maximize_group
        self.__feature_weights = {}
        self.threshold = 1 - (self.alpha * 1.2)

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.analysis_tables,
            executor_id=self.id,
            key=self.key,
            value=value,
        )

    def _analyze_aa_score(
        self, data: ExperimentData, score_table: Dataset
    ) -> ExperimentData:
        self.__feature_weights = {
            column: 1 - abs(self.alpha - score_table.loc[:, column].mean())
            for column in score_table.columns
            if self.PASS_COLUMN_MARKER in column
        }
        aa_scores = {
            class_.replace(f"{ID_SPLIT_SYMBOL}pass", ""): value
            for class_, value in self.__feature_weights.items()
        }
        aa_passed = {
            class_: value >= self.threshold for class_, value in aa_scores.items()
        }
        result = Dataset.from_dict({"score": aa_scores, "pass": aa_passed}, roles={})
        self.key = "aa score"
        return self._set_value(data, result)

    @staticmethod
    def _is_flagged(value: Any) -> bool:
        """Whether a test flagged a difference, i.e. its "pass" value is truthy.

        A test that could not be computed (a missing value) is not a flag. Values
        may arrive as strings, so bare bool() is not enough: it would read both
        "False" and nan as a flag.
        """
        if value is None:
            return False
        if isinstance(value, str):
            return value.lower() == "true"
        if isinstance(value, float) and np.isnan(value):
            return False
        return bool(value)

    def _find_maximized_metric_column(self, score_table: Dataset, metric: str) -> str:
        """Finds the column to maximize: mean of the metric in maximize_group."""
        if self.maximize_group == "control":
            # control mean is repeated for every test group, so any of them fits
            prefix = (
                f"{metric}{self.GROUP_DIFFERENCE_MARKER}control mean{ID_SPLIT_SYMBOL}"
            )
            found = [c for c in score_table.columns if c.startswith(prefix)]
        else:
            column = (
                f"{metric}{self.GROUP_DIFFERENCE_MARKER}"
                f"test mean{ID_SPLIT_SYMBOL}{self.maximize_group}"
            )
            found = [column] if column in score_table.columns else []
        if len(found) == 0:
            groups = {"control"}
            for column in score_table.columns:
                if self.GROUP_DIFFERENCE_MARKER in column:
                    groups.add(column[column.rfind(ID_SPLIT_SYMBOL) + 1 :])
            raise ValueError(
                f"Mean of {metric} in group {self.maximize_group} is not found in the "
                f"A/A score table. Groups available: {sorted(groups)}. Note that only "
                f"a numeric metric can be maximized."
            )
        return found[0]

    def _resolve_metric_to_maximize(self, data: ExperimentData) -> str | None:
        """The column tagged for maximization, if maximization is configured at all.

        The metric (a column with MaximizationRole) and the group to maximize it in
        are two halves of one setting: either both are given, or neither is.
        """
        metric_fields = data.ds.search_columns(MaximizationRole())
        role_name = MaximizationRole().role_name
        if len(metric_fields) == 0:
            if self.maximize_group is not None:
                raise ValueError(
                    f"Group {self.maximize_group} is set to be maximized, but no "
                    f"column carries the {role_name} role, so there is no metric to "
                    f"maximize in it."
                )
            return None
        if self.maximize_group is None:
            raise ValueError(
                f"Column {metric_fields[0]} carries the {role_name} role, but the "
                f"group to maximize it in is not set. Pass maximize_group "
                f"('control', 'test_1', ...)."
            )
        if len(metric_fields) > 1:
            warnings.warn(
                f"Only one metric can be maximized, but {len(metric_fields)} columns "
                f"carry the {role_name} role: {metric_fields}. {metric_fields[0]} "
                f"will be used.",
                stacklevel=2,
            )
        return metric_fields[0]

    def _get_best_index_by_metric(
        self, data: ExperimentData, score_table: Dataset
    ) -> float | None:
        """Index of the homogeneous split with the highest metric in the given group.

        Only the iterations where no test flagged a difference on any feature take
        part in the comparison. Returns None if there is nothing to maximize or if
        no iteration is homogeneous.
        """
        metric = self._resolve_metric_to_maximize(data)
        if metric is None:
            return None
        maximized_column = self._find_maximized_metric_column(score_table, metric)
        pass_columns = [c for c in score_table.columns if self.PASS_COLUMN_MARKER in c]

        def is_homogeneous(row) -> bool:
            return not any(self._is_flagged(row[column]) for column in pass_columns)

        metric_scores = score_table.apply(
            lambda x: x[maximized_column] if is_homogeneous(x) else -np.inf,
            axis=1,
            role={"metric score": StatisticRole()},
        )
        if metric_scores.max() == -np.inf:
            warnings.warn(
                f"No split is homogeneous on all features, so the mean of {metric} "
                f"in group {self.maximize_group} cannot be maximized. Falling back "
                f"to the best split by homogeneity.",
                stacklevel=2,
            )
            return None
        return metric_scores.idxmax()

    def build_splitter_from_id(self, splitter_id: str):
        splitter_class = self.AA_SPLITER_CLASS_MAPPING.get(
            splitter_id[: splitter_id.find(ID_SPLIT_SYMBOL)]
        )
        if splitter_class is None:
            raise ValueError(f"{splitter_id} is not a valid splitter id")
        return splitter_class.build_from_id(splitter_id)

    def _get_best_index_by_homogeneity(self, score_table: Dataset) -> float | int:
        """Index of the split with the highest homogeneity score."""
        if len(self.__feature_weights) < 1:
            return 0
        aa_split_scores = score_table.apply(
            lambda x: (
                (
                    (
                        (
                            sum(
                                x[
                                    key.replace(
                                        self.PASS_COLUMN_MARKER,
                                        self.P_VALUE_COLUMN_MARKER,
                                    )
                                ]
                                * value
                                for key, value in self.__feature_weights.items()
                                if (isinstance(value, float) and value > 0)
                                and (
                                    key.replace(
                                        self.PASS_COLUMN_MARKER,
                                        self.P_VALUE_COLUMN_MARKER,
                                    )
                                    in x["splitter_id"]
                                )
                            )
                            / len(self.__feature_weights)
                        )
                        * 2
                    )
                    / 3
                )
                + x["mean test score"] / 3
            ),
            axis=1,
            role={"aa split score": StatisticRole()},
        )
        return aa_split_scores.idxmax()

    def _get_best_split(
        self,
        data: ExperimentData,
        score_table: Dataset,
        if_param_scores: Dataset | None = None,
    ) -> dict[str, Any]:
        # TODO: add split_scores in ExperimentData
        if if_param_scores is None:
            best_index = self._get_best_index_by_metric(data, score_table)
            if best_index is None:
                best_index = self._get_best_index_by_homogeneity(score_table)
            best_split_id = score_table.loc[best_index, "splitter_id"].get_values(0, 0)
            score_dict = score_table.loc[best_index, :].transpose().to_records()[0]
        else:
            best_index = 0
            best_split_id = score_table.loc[best_index, "splitter_id"].get_values(0, 0)
            score_dict = if_param_scores.loc[best_index, :].transpose().to_records()[0]
        best_score_stat = OneAADictReporter.convert_flat_dataset(score_dict)
        self.key = "best split statistics"
        result = self._set_value(data, best_score_stat)
        return {"best_split_id": best_split_id, "data": result}

    def _set_best_split(
        self,
        data: ExperimentData,
        best_splitter_id: str,
    ) -> ExperimentData:
        self.key = "best splitter"
        result = data.set_value(
            ExperimentDataEnum.variables, self.id, best_splitter_id, self.key
        )
        best_splitter = self.build_splitter_from_id(best_splitter_id)
        best_splitter.save_groups = False
        best_splitter.constant_key = False
        best_splitter.key = "best"
        result = best_splitter.execute(result)
        return result

    def _analyze_best_split(
        self,
        data: ExperimentData,
        score_table: Dataset,
        if_param_scores: Dataset | None = None,
    ) -> ExperimentData:
        best_split = self._get_best_split(data, score_table, if_param_scores)
        return self._set_best_split(best_split["data"], best_split["best_split_id"])

    def execute(self, data: ExperimentData) -> ExperimentData:
        param_experiment_id = data.get_one_id(
            ParamsExperiment, ExperimentDataEnum.analysis_tables, "AATest"
        )
        ifparam_experiment_id = data.get_ids(
            IfParamsExperiment,
            ExperimentDataEnum.analysis_tables,
        )
        score_table = data.analysis_tables[param_experiment_id]
        score_table = score_table.dropna(axis=1, how="all")
        if_param_scores = (
            None
            if len(ifparam_experiment_id["IfParamsExperiment"]["analysis_tables"]) == 0
            else data.analysis_tables[
                ifparam_experiment_id["IfParamsExperiment"]["analysis_tables"][0]
            ]
        )
        data = self._analyze_aa_score(data, score_table)
        return self._analyze_best_split(data, score_table, if_param_scores)

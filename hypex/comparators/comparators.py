from typing import Dict, Optional, List, Union

from hypex.comparators.abstract import Comparator
from hypex.dataset import Dataset, ABCRole, FeatureRole
from hypex.utils.constants import NUMBER_TYPES_LIST
from hypex.utils import SpaceEnum


class GroupDifference(Comparator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        space: SpaceEnum = SpaceEnum.auto,
    ):
        super().__init__(compare_by="groups", grouping_role=grouping_role, space=space)

    @property
    def search_types(self) -> Optional[List[type]]:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        **kwargs,
    ) -> Dict:
        test_data = cls._check_test_data(test_data)
        control_mean = data.mean()
        test_mean = test_data.mean()

        return {
            "control mean": control_mean,
            "test mean": test_mean,
            "difference": test_mean - control_mean,
            "difference %": (test_mean / control_mean - 1) * 100,
        }


class GroupSizes(Comparator):
    def __init__(
        self, grouping_role: Optional[ABCRole] = None, space: SpaceEnum = SpaceEnum.auto
    ):
        super().__init__(compare_by="groups", grouping_role=grouping_role, space=space)

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Dict:
        size_a = len(data)
        size_b = len(test_data) if isinstance(test_data, Dataset) else 0

        return {
            "control size": size_a,
            "test size": size_b,
            "control size %": (size_a / (size_a + size_b)) * 100,
            "test size %": (size_b / (size_a + size_b)) * 100,
        }

class SMD(Comparator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_roles: Union[ABCRole, List[ABCRole], None] = None,
        space: SpaceEnum = SpaceEnum.auto
    ):
        super().__init__(
        compare_by="columns",
        grouping_role=grouping_role,
        target_roles=FeatureRole(),
        space=space
        )

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Union[float, Dataset]:
        test_data = cls._check_test_data(test_data=test_data)
        return (data.mean() + test_data.mean()) / data.std()


# class MatchingMetrics(Calculator):
#     def __init__(
#         self,
#         target_roles: Union[ABCRole, List[ABCRole], None] = None,
#         baseline_role: Optional[ABCRole] = None,
#         metric: Optional[Literal["auto", "atc", "att", "ate"]] = None,
#         key: Any = "",
#     ):
#         self.metric = metric or "auto"
#         super().__init__(compare_by="columns", target_roles=target_roles, baseline_role=baseline_role, key=key)
#
#     def execute(self, data: ExperimentData) -> ExperimentData:
#         group_field, target_fields, baseline_field = self._get_fields(data=data)
#         t_data = deepcopy(data.ds)
#         if (len(target_fields) + len(baseline_field)) != 2:
#             target_fields += self._field_searching(
#                 data=data,
#                 field=self.target_roles,
#                 search_types=self.search_types,
#                 space=SpaceEnum.additional,
#             )
#         if (len(target_fields) + len(baseline_field)) != 2:
#             distances_keys = data.get_ids(FaissNearestNeighbors, ExperimentDataEnum.groups)
#             if len(distances_keys["FaissNearestNeighbors"]["groups"]) > 0:
#                 target_fields += data.groups[distances_keys["FaissNearestNeighbors"]["groups"][0]]["matched_df"].search_columns(
#                     self.target_roles)
#             else:
#                 raise ValueError
#         if target_fields[0] not in t_data.columns:
#             t_data = t_data.add_column(
#                 data.groups[distances_keys["FaissNearestNeighbors"]["groups"][0]]["matched_df"][target_fields[0]],
#                 role={target_fields[0]: TargetRole()},
#             )
#         self.key = str(
#             baseline_field if len(target_fields) == 0 else (target_fields or "")
#         )
#         if (
#             not target_fields and data.ds.tmp_roles
#         ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
#             return data
#
#         compare_result = self.calc(
#             data=t_data,
#             compare_by="columns",
#             target_fields=target_fields,
#             baseline_field=baseline_field,
#             metric=self.metric,
#         )
#         result_dataset = self._local_extract_dataset(
#             compare_result, {key: StatisticRole() for key in compare_result}
#         )
#         return self._set_value(data=data, space=ExperimentDataEnum.analysis_tables, value=result_dataset)
#
#     @classmethod
#     def _execute_inner_function(
#         cls, grouping_data, target_fields: Optional[List[str]] = None, **kwargs
#     ) -> Dict:
#         metric = kwargs.get("metric", "auto")
#         if target_fields is None or len(target_fields) != 1:
#             raise ValueError(
#                 "This operator works with 2 targets, but got {}".format(
#                     len(target_fields) if target_fields else None
#                 )
#             )
#         if metric == "auto":
#             if len(
#                 grouping_data[0][1][grouping_data[0][1][target_fields[1]] == 0]
#             ) == len(grouping_data[0][1]):
#                 metric = "atc"
#             else:
#                 metric = (
#                     "att"
#                     if len(
#                         grouping_data[1][1][grouping_data[1][1][target_fields[1]] == 0]
#                     )
#                     == len(grouping_data[1][1])
#                     else "ate"
#                 )
#         att = (
#             grouping_data[0][1][target_fields[0]]
#             - grouping_data[0][1][target_fields[1]]
#         ).mean()
#         if metric == "att":
#             return {"ATT": att}
#         atc = (
#             grouping_data[1][1][target_fields[0]]
#             - grouping_data[1][1][target_fields[1]]
#         ).mean()
#         if metric == "atc":
#             return {"ATC": atc}
#         len_test, len_control = len(grouping_data[0][1]), len(grouping_data[1][1])
#         return {
#             "ATT": att,
#             "ATC": atc,
#             "ATE": (att * len_test + atc * len_control) / (len_test + len_control),
#         }
#
#     @classmethod
#     def _inner_function(
#         cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
#     ) -> Any:
#         raise NotImplementedError
from typing import Optional, Any, List, Literal, Union, Dict

from hypex.dataset import Dataset, ABCRole, ExperimentData, MatchingRole
from hypex.operators.abstract import GroupOperator
from hypex.utils import SpaceEnum, FieldKeyTypes


class SMD(GroupOperator):
    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        test_data = cls._check_test_data(test_data=test_data)
        return (data.mean() + test_data.mean()) / data.std()


class MatchingMetrics(GroupOperator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_roles: Union[ABCRole, List[ABCRole], None] = None,
        space: SpaceEnum = SpaceEnum.auto,
        metric: Optional[Literal["auto", "atc", "att", "ate"]] = None,
        key: Any = "",
    ):
        self.metric = metric or "auto"
        super().__init__(
            grouping_role=grouping_role, target_roles=target_roles, space=space, key=key
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data=data)
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data
        if group_field[0] in data.groups:  # TODO: to recheck if this is a correct check
            grouping_data = list(data.groups[group_field[0]].items())
        else:
            grouping_data = data.ds.groupby(group_field)
        if len(target_fields) < 2:
            t_grouping_data = []
            target_fields += [target_fields[0] + "_matched"]
            index_fields_names = data.additional_fields.search_columns(MatchingRole())
            if not index_fields_names:
                raise Exception("Сюда тоже нужна ошибка")
            for i in range(len(index_fields_names)):
                index_field = data.additional_fields[index_fields_names[i]].fillna(-1)
                filtered_field = index_field.drop(
                    index_field[index_field[index_fields_names[i]] == -1], axis=0
                )
                new_target = data.ds.iloc[
                    list(map(lambda x: x[0], filtered_field.get_values()))
                ][target_fields[0]]
                new_target.index = filtered_field.index
                new_target = new_target.reindex(
                    grouping_data[i][1].index,
                    fill_value=0,
                ).rename({target_fields[0]: target_fields[0] + "_matched"})
                t_grouping_data += [
                    (grouping_data[i][0], grouping_data[i][1].add_column(new_target))
                ]
            if len(t_grouping_data) < 2:
                t_grouping_data.append(grouping_data[1])
            grouping_data = t_grouping_data
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            grouping_data=grouping_data,
            target_fields=target_fields,
            metric=self.metric,
        )
        return self._set_value(data, compare_result)

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: Optional[List[FieldKeyTypes]] = None,
        **kwargs
    ) -> Dict:
        metric = kwargs.get("metric", "auto")
        if target_fields is None or len(target_fields) != 2:
            raise ValueError("Нужно дописать ошибку")
        if metric == "auto":
            if (
                target_fields[1] in grouping_data[0][1].columns
                and target_fields[1] in grouping_data[1][1].columns
            ):
                metric = "ate"
            else:
                metric = (
                    "att" if target_fields[1] in grouping_data[0][1].columns else "atc"
                )
        if metric in ["atc", "att"]:
            return {
                metric.upper(): (
                    grouping_data[0][1][target_fields[0]]
                    - grouping_data[0][1][target_fields[1]]
                ).mean()
            }
        else:
            att = (
                grouping_data[0][1][target_fields[0]]
                - grouping_data[0][1][target_fields[1]]
            ).mean()
            atc = (
                grouping_data[1][1][target_fields[0]]
                - grouping_data[1][1][target_fields[1]]
            ).mean()
            len_test, len_control = len(grouping_data[0][1]), len(grouping_data[1][1])
            return {
                "ATT": att,
                "ATC": atc,
                "ATE": (att * len_test + atc * len_control) / (len_test + len_control),
            }

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        raise NotImplementedError

from copy import deepcopy
from typing import Optional, Any, List, Literal, Union, Dict

from hypex.dataset import Dataset, ABCRole, ExperimentData, TargetRole
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.abstract import GroupOperator
from hypex.utils import SpaceEnum
from hypex.utils.enums import ExperimentDataEnum


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
        bias_estimation: bool = True,
        key: Any = "",
    ):
        self.metric = metric or "auto"
        self.bias_estimation = bias_estimation
        super().__init__(
            grouping_role=grouping_role, target_roles=target_roles, space=space, key=key
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data=data)
        t_data = deepcopy(data.ds)
        if len(target_fields) != 2:
            distances_keys = data.get_ids(
                FaissNearestNeighbors, ExperimentDataEnum.groups
            )
            if len(distances_keys["FaissNearestNeighbors"]["groups"]) > 0:
                target_fields += data.groups[
                    distances_keys["FaissNearestNeighbors"]["groups"][0]
                ]["matched_df"].search_columns(self.target_roles)
            else:
                raise ValueError
        if target_fields[1] not in t_data.columns:
            t_data = t_data.add_column(
                data.groups[distances_keys["FaissNearestNeighbors"]["groups"][0]][
                    "matched_df"
                ][target_fields[1]],
                role={target_fields[1]: TargetRole()},
            )
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data

        compare_result = self.calc(
            data=t_data,
            group_field=group_field,
            target_fields=target_fields,
            metric=self.metric,
            bias_estimation=self.bias_estimation,
        )
        return self._set_value(data, compare_result)

    @staticmethod
    def bias_coefs(matches, Y_m, X_m):

        # Computes OLS coefficient in bias correction regression. Constructs
        # data for regression by including (possibly multiple times) every
        # observation that has appeared in the matched sample.

        flat_idx = reduce(lambda x, y: np.concatenate((x, y)), matches)
        N, K = len(flat_idx), X_m.shape[1]

        Y = Y_m[flat_idx]
        X = np.empty((N, K + 1))
        X[:, 0] = 1  # intercept term
        X[:, 1:] = X_m[flat_idx]

        return np.linalg.lstsq(X, Y)[0][1:]  # don't need intercept coef

    @staticmethod
    def bias(X, X_m, matches, coefs):
        X_m_mean = [X_m[idx].mean(0) for idx in matches]
        bias_list = [(X_j - X_i).dot(coefs) for X_i, X_j in zip(X, X_m_mean)]

        return np.array(bias_list)

    @classmethod
    def _execute_inner_function(
        cls, grouping_data, target_fields: Optional[List[str]] = None, **kwargs
    ) -> Dict:
        metric = kwargs.get("metric", "auto")
        if target_fields is None or len(target_fields) != 2:
            raise ValueError(
                "This operator works with 2 targets, but got {}".format(
                    len(target_fields) if target_fields else None
                )
            )
        if metric == "auto":
            if len(
                grouping_data[0][1][grouping_data[0][1][target_fields[1]] == 0]
            ) == len(grouping_data[0][1]):
                metric = "atc"
            else:
                metric = (
                    "att"
                    if len(
                        grouping_data[1][1][grouping_data[1][1][target_fields[1]] == 0]
                    )
                    == len(grouping_data[1][1])
                    else "ate"
                )
        itt = (
            grouping_data[0][1][target_fields[0]]
            - grouping_data[0][1][target_fields[1]]
        ).mean()
        itc = (
            grouping_data[1][1][target_fields[0]]
            - grouping_data[1][1][target_fields[1]]
        ).mean()
        # if kwargs.get("bias_estimation", False):
        #     bias_c = cls.bias(
        #         X_c,
        #         X_t,
        #         matches_c,
        #         cls.bias_coefs(grouping_data[0][1].drop(target_fields), Y_t, X_t),
        #     )
        #     bias_t = cls.bias(
        #         X_t,
        #         X_c,
        #         matches_t,
        #         cls.bias_coefs(grouping_data[1][1].drop(target_fields), Y_c, X_c),
        #     )
        #     itt = itt - bias_c
        #     itc = itc + bias_t

        if metric == "atc":
            return {"ATC": itc}
        if metric == "att":
            return {"ATC": itt}
        len_test, len_control = len(grouping_data[0][1]), len(grouping_data[1][1])
        return {
            "ATT": itt,
            "ATC": itc,
            "ATE": (itt * len_test + itc * len_control) / (len_test + len_control),
        }

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        raise NotImplementedError

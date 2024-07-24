from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple, Literal

from hypex.dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    GroupingRole,
    TempTargetRole,
    FeatureRole,
    MatchingRole,
    TargetRole,
    DatasetAdapter,
)
from hypex.utils import (
    ComparisonNotSuitableFieldError,
    NoColumnsError,
    SpaceEnum,
    AbstractMethodError,
    ID_SPLIT_SYMBOL,
    SetParamsDictTypes,
    ExperimentDataEnum,
    NoRequiredArgumentError,
)
from hypex.utils.adapter import Adapter


class Executor(ABC):
    def __init__(
        self,
        key: Any = "",
    ):
        self._id: str = ""
        self._params_hash = ""

        self.key: Any = key
        self._generate_id()

    def check_and_setattr(self, params: Dict[str, Any]):
        for key, value in params.items():
            if key in self.__dir__():
                setattr(self, key, value)

    def _generate_params_hash(self):
        self._params_hash = ""

    def _generate_id(self):
        self._generate_params_hash()
        self._id = ID_SPLIT_SYMBOL.join(
            [
                self.__class__.__name__,
                self._params_hash.replace(ID_SPLIT_SYMBOL, "|"),
                str(self._key).replace(ID_SPLIT_SYMBOL, "|"),
            ]
        )

    def set_params(self, params: SetParamsDictTypes) -> None:
        if isinstance(list(params)[0], str):
            self.check_and_setattr(params)
        elif isinstance(list(params)[0], type):
            for executor_class, class_params in params.items():
                if isinstance(self, executor_class):
                    self.check_and_setattr(class_params)
        else:
            raise ValueError(
                "params must be a dict of str to dict or a dict of class to dict"
            )
        self._generate_id()

    def init_from_hash(self, hash: str) -> None:
        self._params_hash = hash
        self._generate_id()

    @classmethod
    def build_from_id(cls, executor_id: str):
        splitted_id = executor_id.split(ID_SPLIT_SYMBOL)
        if splitted_id[0] != cls.__name__:
            raise ValueError(f"{executor_id} is not a valid {cls.__name__} id")
        result = cls()
        result.init_from_hash(splitted_id[1])
        return result

    @property
    def id(self) -> str:
        return self._id

    @property
    def key(self) -> Any:
        return self._key

    @key.setter
    def key(self, value: Any):
        self._key = value
        self._generate_id()

    @property
    def params_hash(self) -> str:
        return self._params_hash

    @property
    def id_for_name(self) -> str:
        return self.id.replace(ID_SPLIT_SYMBOL, "_")

    @property
    def _is_transformer(self) -> bool:
        return False

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        # defined in order to avoid  unnecessary redefinition in classes like transformer
        return data

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        raise AbstractMethodError


class Calculator(Executor, ABC):
    @classmethod
    def calc(cls, data: Dataset, **kwargs):
        return cls._inner_function(data, **kwargs)

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Any:
        raise AbstractMethodError


class GroupCalculator(Calculator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_roles: Optional[List[ABCRole]] = None,
        space: SpaceEnum = SpaceEnum.auto,
        key: Any = "",
    ):
        self.grouping_role = grouping_role or GroupingRole()
        self.space = space
        self.__additional_mode = space == SpaceEnum.additional
        self.target_roles = target_roles or []
        super().__init__(key=key)

    @property
    def search_types(self):
        raise AbstractMethodError

    def _field_searching(
        self,
        data: ExperimentData,
        field,
        tmp_role: bool = False,
        search_types=None,
        space: Optional[SpaceEnum] = None,
    ):
        space = space or self.space
        searched_field = []
        if space in [SpaceEnum.auto, SpaceEnum.data]:
            searched_field = data.ds.search_columns(
                field, tmp_role=tmp_role, search_types=search_types
            )
        if (
            space in [SpaceEnum.auto, SpaceEnum.additional]
            and searched_field == []
            and isinstance(data, ExperimentData)
        ):
            searched_field = data.additional_fields.search_columns(
                field, tmp_role=tmp_role, search_types=search_types
            )
            self.__additional_mode = True
        return searched_field

    @staticmethod
    def _check_test_data(test_data: Optional[Dataset] = None) -> Dataset:
        if test_data is None:
            raise ValueError("test_data is needed for comparison")
        return test_data


    @staticmethod
    def _field_arg_universalization(
        field: Union[Sequence[str], str, None]
    ) -> List[str]:
        if not field:
            raise NoColumnsError(field)
        elif isinstance(field, str):
            return [field]
        return list(field)

    @classmethod
    @abstractmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        raise AbstractMethodError

    @classmethod
    @abstractmethod
    def _execute_inner_function(
        cls,
        baseline_data: Union[Tuple[str, Dataset], List[Tuple[str, Dataset]]],
        compared_data: Union[Tuple[str, Dataset], List[Tuple[str, Dataset]]],
        **kwargs,
    ) -> Any:
        raise AbstractMethodError

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Union[Sequence[str], str, None] = None,
        grouping_data: Optional[List[Tuple[str, Dataset]]] = None,
        target_fields: Union[
            str, List[str], None
        ] = None,  # why is it possible to leave it None?
        baseline_column: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        group_field = Adapter.to_list(group_field)

        if grouping_data is None:
            grouping_data = cls._split_data_to_buckets(
                data=data,
                group_field=group_field,
                target_fields=target_fields,
                baseline_column=baseline_column,
            )
        if len(grouping_data) < 2:
            raise ComparisonNotSuitableFieldError(group_field)
        baseline_data, compared_data = grouping_data
        return cls._execute_inner_function(
            baseline_data=baseline_data, compared_data=compared_data, **kwargs
        )

    @staticmethod
    def _split_data_to_buckets(
        data: Dataset,
        group_field: Union[str, List[str]],
        target_fields: Union[str, List[str]],
        compare_by: Literal[
            "groups", "columns", "columns_in_groups", "cross"
        ] = "groups",
        baseline_column: Optional[str] = None,
    ) -> Tuple:
        """
        Splits the given dataset into buckets into baseline and compared data, based on the specified comparison mode.

        Args:
            data (Dataset): The dataset to be split.
            group_field (Union[Sequence[str], str]): The field(s) to group the data by.
            target_fields (Union[str, List[str]]): The field(s) to target for comparison.
            compare_by (Literal['groups', 'columns', 'columns_in_groups', 'cross'], optional): The method to compare the data. Defaults to 'groups'.
            baseline_column (Optional[str], optional): The column to use as the baseline for comparison. Required if `compare_by` is 'columns' or 'columns_in_groups'. Defaults to None.

        Returns:
            Tuple: A tuple containing the baseline data and the compared data.

        Raises:
            NoRequiredArgumentError: If `baseline_column` is None and `compare_by` is 'columns' or 'columns_in_groups' or 'cross'.
            ValueError: If `compare_by` is not one of the allowed values.
        """
        target_fields = Adapter.to_list(target_fields)
        if compare_by == "groups":
            data_buckets = data.groupby(by=group_field, fields_list=target_fields)
            baseline_data = data_buckets.pop(0)
            compared_data = data_buckets
        else:
            if baseline_column is None:
                raise NoRequiredArgumentError("baseline_column")
            elif compare_by == "columns":
                baseline_data = (baseline_column, data[baseline_column])
                compared_data = [
                    (column, data[column])
                    for column in target_fields
                    if column != baseline_column
                ]
            elif compare_by == "columns_in_groups":
                baseline_data = data.groupby(
                    by=group_field, fields_list=baseline_column
                )
                compared_data = data.groupby(by=group_field, fields_list=target_fields)
            elif compare_by == "cross":
                data_buckets = data.groupby(by=group_field, fields_list=baseline_column)
                baseline_data = data_buckets.pop(0)
                compared_data = data.groupby(by=group_field, fields_list=target_fields)
                compared_data.pop(0)
            else:
                raise ValueError("compare_by")
        return baseline_data, compared_data




class MLExecutor(GroupCalculator, ABC):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_role: Optional[ABCRole] = None,
        space: SpaceEnum = SpaceEnum.auto,
        key: Any = "",
    ):
        self.target_role = target_role or TargetRole()
        super().__init__(
            grouping_role=grouping_role or GroupingRole(), space=space, key=key
        )

    def _get_fields(self, data: ExperimentData):
        group_field = self._field_searching(data, self.grouping_role)
        target_field = self._field_searching(
            data, self.target_role, search_types=self.search_types
        )
        return group_field, target_field

    @abstractmethod
    def fit(self, X: Dataset, Y: Optional[Dataset] = None) -> "MLExecutor":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Dataset) -> Dataset:
        raise NotImplementedError

    def score(self, X: Dataset, Y: Dataset) -> float:
        raise NotImplementedError

    @property
    def search_types(self):
        return [int, float]

    @classmethod
    @abstractmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        target_data: Optional[Dataset] = None,
        **kwargs,
    ) -> Any:
        raise AbstractMethodError

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_field: Optional[str] = None,
        **kwargs,
    ) -> Any:
        if target_field:
            return cls._inner_function(
                data=grouping_data[0][1].drop(target_field),
                target_data=grouping_data[0][1][target_field],
                test_data=grouping_data[1][1].drop(target_field),
                **kwargs,
            )
        return cls._inner_function(
            data=grouping_data[0][1],
            test_data=grouping_data[1][1],
            **kwargs,
        )

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.groups,
            self.id,
            str(self.__class__.__name__),
            value=value,
            key=key,
        )

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Union[Sequence[str], str, None] = None,
        grouping_data: Optional[List[Tuple[str, Dataset]]] = None,
        target_field: Union[str, List[str], None] = None,
        features_fields: Union[str, List[str], None] = None,
        **kwargs,
    ) -> Dataset:
        group_field = Adapter.to_list(group_field)
        features_fields = Adapter.to_list(features_fields)
        if grouping_data is None:
            grouping_data = data.groupby(group_field, fields_list=features_fields)
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise ComparisonNotSuitableFieldError(group_field)
        result = cls._execute_inner_function(
            grouping_data, target_field=target_field, **kwargs
        )
        return DatasetAdapter.to_dataset(
            result,
            {i: MatchingRole() for i in list(result.keys())},
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data=data)
        features_fields = data.ds.search_columns(
            FeatureRole(), search_types=self.search_types
        )
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
            grouping_data = None
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            grouping_data=grouping_data,
            target_fields=target_fields,
            features_fields=features_fields,
        )
        return self._set_value(data, compare_result)

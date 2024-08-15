from abc import abstractmethod, ABC
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple, Iterable

from hypex.dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    GroupingRole,
    FeatureRole,
    AdditionalMatchingRole,
    TargetRole,
    DatasetAdapter,
)
from hypex.dataset.roles import AdditionalRole
from hypex.utils import (
    AbstractMethodError,
    ID_SPLIT_SYMBOL,
    SetParamsDictTypes,
    ExperimentDataEnum,
    FieldNotSuitableFieldError,
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

    @property
    def search_types(self):
        raise AbstractMethodError

    def _field_searching(
        self,
        data: ExperimentData,
        roles: Union[ABCRole, Iterable[ABCRole]],
        tmp_role: bool = False,
        search_types=None,
    ):
        searched_field = []
        roles = Adapter.to_list(roles)
        field_in_additional = [
            role for role in roles if isinstance(role, AdditionalRole)
        ]
        field_in_data = [role for role in roles if role not in field_in_additional]
        if field_in_data:
            searched_field += data.ds.search_columns(
                field_in_data, tmp_role=tmp_role, search_types=search_types
            )
        if field_in_additional and isinstance(data, ExperimentData):
            searched_field += data.additional_fields.search_columns(
                field_in_additional, tmp_role=tmp_role, search_types=search_types
            )
        return searched_field

    @staticmethod
    def _check_test_data(
        test_data: Optional[Dataset] = None,
    ) -> Dataset:  # TODO to move away from Calculator. Where to?
        if test_data is None:
            raise ValueError("test_data is needed for comparison")
        return test_data


class MLExecutor(Calculator, ABC):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_role: Optional[ABCRole] = None,
        key: Any = "",
    ):
        self.target_role = target_role or TargetRole()
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()

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
            ExperimentDataEnum.additional_fields,
            self.id,
            value=value,
            key=key,
            role=AdditionalMatchingRole(),
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
            raise FieldNotSuitableFieldError(group_field, "Grouping")
        result = cls._execute_inner_function(
            grouping_data, target_field=target_field, **kwargs
        )
        return DatasetAdapter.to_dataset(
            result,
            {i: AdditionalMatchingRole() for i in list(result.keys())},
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

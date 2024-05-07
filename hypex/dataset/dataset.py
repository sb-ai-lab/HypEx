import warnings
from copy import copy
from typing import Union, List, Iterable, Any, Dict, Callable, Hashable, Optional

import pandas as pd  # type: ignore

from hypex.dataset.abstract import DatasetBase
from hypex.dataset.roles import (
    StatisticRole,
    InfoRole,
    ABCRole,
    FilterRole,
)
from hypex.utils import (
    ID_SPLIT_SYMBOL,
    ExperimentDataEnum,
    BackendsEnum,
    ConcatDataError,
    ConcatBackendError,
    NotFoundInExperimentDataError,
    FromDictType,
    MergeDataError,
    MergeBackendError,
    MergeOnError, FieldKeyTypes,
)


class Dataset(DatasetBase):
    class Locker:
        def __init__(self, backend, roles):
            self.backend = backend
            self.roles = roles

        def __getitem__(self, item):
            t_data = self.backend.loc(item)
            return Dataset(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

    class ILocker:
        def __init__(self, backend, roles):
            self.backend = backend
            self.roles = roles

        def __getitem__(self, item):
            t_data = self.backend.iloc(item)
            return Dataset(
                data=t_data,
                roles={k: v for k, v in self.roles.items() if k in t_data.columns},
            )

    def __init__(
        self,
        roles: Union[
            Dict[ABCRole, Union[List[Union[str, int]], str, int]],
            Dict[Union[str, int], ABCRole],
        ],
        data: Optional[Union[pd.DataFrame, str]] = None,
        backend: Optional[BackendsEnum] = None,
    ):
        super().__init__(roles, data, backend)
        self.loc = self.Locker(self._backend, self.roles)
        self.iloc = self.ILocker(self._backend, self.roles)

    def __getitem__(self, item: Union[Iterable, str, int]):
        items = (
            [item] if isinstance(item, str) or not isinstance(item, Iterable) else item
        )
        roles: Dict = {
            column: (
                self.roles[column]
                if column in self.columns and self.roles.get(column, False)
                else InfoRole()
            )
            for column in items
        }
        result = Dataset(data=self._backend.__getitem__(item), roles=roles)
        result.tmp_roles = self.tmp_roles
        return result

    def __setitem__(self, key: str, value: Any):
        if key not in self.columns and isinstance(key, str):
            self.add_column(value, {key: InfoRole()})
            warnings.warn("Column must be added by add_column", category=SyntaxWarning)
        self.data[key] = value

    @staticmethod
    def _create_empty(backend=BackendsEnum.pandas, roles=None, index=None):
        if roles is None:
            roles = {}
        index = [] if index is None else index
        columns = list(roles.keys())
        ds = Dataset(roles=roles, backend=backend)
        ds._backend = ds._backend._create_empty(index, columns)
        ds.data = ds._backend.data
        return ds

    def _convert_data_after_agg(self, result):
        if isinstance(result, float):
            return result
        return Dataset(
            data=result,
            roles={column: StatisticRole() for column in self.roles},
        )

    def add_column(
        self,
        data,
        role: Optional[Dict[str, ABCRole]] = None,
        index: Optional[Iterable[Hashable]] = None,
    ):
        if role is None:
            if not isinstance(data, Dataset):
                raise ValueError("Козьёль")
            self.roles.update(data.roles)
            self._backend.add_column(
                data._backend.data[list(data._backend.data.columns)[0]],
                list(data.roles.keys())[0],
                index,
            )
        else:
            self.roles.update(role)
            self._backend.add_column(data, list(role.keys())[0], index)

    def append(self, other, index=None):
        if not isinstance(other, Dataset):
            raise ConcatDataError(type(other))
        if type(other._backend) is not type(self._backend):
            raise ConcatBackendError(type(other._backend), type(self._backend))
        self.roles.update(other.roles)
        return Dataset(
            roles=self.roles, data=self._backend.append(other._backend, index)
        )

    @staticmethod
    def from_dict(
        data: FromDictType,
        roles: Union[
            Dict[ABCRole, Union[List[Union[str, int]], str, int]],
            Dict[Union[str, int], ABCRole],
        ],
        backend: BackendsEnum = BackendsEnum.pandas,
        index=None,
    ):
        ds = Dataset(roles=roles, backend=backend)
        ds._backend = ds._backend.from_dict(data, index)
        ds.data = ds._backend.data
        return ds

    # What is going to happen when a matrix is returned?
    def apply(
        self,
        func: Callable,
        role: Dict[FieldKeyTypes, ABCRole],
        axis: int=0,
        **kwargs,
    ):
        return Dataset(
            data=self._backend.apply(func=func, axis=axis, **kwargs).rename(
                list(role.keys())[0]
            ),
            roles=role,
        )

    def map(self, func, na_action=None, **kwargs):
        return Dataset(
            roles=self.roles,
            data=self._backend.map(func=func, na_action=na_action, **kwargs),
        )

    def unique(self):
        return self._backend.unique()

    def nunique(self, dropna: bool = False):
        return self._backend.nunique(dropna)

    def isin(self, values: Iterable):
        return Dataset(
            roles={column: FilterRole() for column in self.roles.keys()},
            data=self._backend.isin(values),
        )

    def groupby(
        self,
        by: Any,
        func: Optional[Union[str, List]] = None,
        fields_list: Optional[Union[str, List]] = None,
        **kwargs,
    ):

        datasets = [
            (i, Dataset(roles=self.roles, data=data))
            for i, data in self._backend.groupby(by=by, **kwargs)
        ]
        if fields_list:
            fields_list = (
                fields_list if isinstance(fields_list, Iterable) else [fields_list]
            )
            datasets = [(i, data[fields_list]) for i, data in datasets]
        if func:
            datasets = [(i, data.agg(func)) for i, data in datasets]
        for dataset in datasets:
            dataset[1].tmp_roles = self.tmp_roles
        return datasets

    def mean(self):
        return self._convert_data_after_agg(self._backend.mean())

    def max(self):
        return self._convert_data_after_agg(self._backend.max())

    def min(self):
        return self._convert_data_after_agg(self._backend.min())

    def count(self):
        return self._convert_data_after_agg(self._backend.count())

    def sum(self):
        return self._convert_data_after_agg(self._backend.sum())

    def agg(self, func: Union[str, List]):
        return self._convert_data_after_agg(self._backend.agg(func))

    def std(self):
        return self._convert_data_after_agg(self._backend.std())

    def coefficient_of_variation(self):
        return self._convert_data_after_agg(self._backend.coefficient_of_variation())

    def value_counts(self, dropna: bool = False):
        return self._convert_data_after_agg(self._backend.value_counts(dropna=dropna))

    def dropna(self, subset: Union[str, Iterable[str]] = None):
        return self._backend.dropna(subset=subset)

    def isna(self):
        return self._backend.isna()

    def na_counts(self):
        return self._convert_data_after_agg(self._backend.na_counts())

    def quantile(self, q: float = 0.5):
        return self._convert_data_after_agg(self._backend.quantile(q=q))

    def select_dtypes(self, include: Any = None, exclude: Any = None):
        t_data = self._backend.select_dtypes(include=include, exclude=exclude)
        roles = {k: v for k, v in self.roles.items() if k in t_data.columns}
        return Dataset(roles=roles, data=t_data)

    def merge(
        self,
        right,
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        suffixes=("_x", "_y"),
    ):
        # use backend check
        for on_ in [on, left_on, right_on]:
            if on_ and (on_ not in [*self.columns, *right.columns]):
                raise MergeOnError(on_)
        if not isinstance(right, Dataset):
            raise MergeDataError(type(right))
        #TODO type is type
        if not isinstance(right._backend, type(self._backend)):
            raise MergeBackendError(type(right._backend), type(self._backend))
        t_data = self._backend.merge(
            right=right._backend,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=suffixes,
        )
        t_roles = copy(self.roles)
        t_roles.update(right.roles)

        for c in t_data.columns:
            if f"{c}".endwith(suffixes[0]) and c[:-len(suffixes[0])] in self.columns:
                t_roles[c] = self.roles[c[:-len(suffixes[0])]]
            if f"{c}".endwith(suffixes[1]) and c[:-len(suffixes[1])] in right.columns:
                t_roles[c] = right.roles[c[:-len(suffixes[1])]]

        new_roles = {c: t_roles[c] for c in t_data.columns}
        return Dataset(roles=new_roles, data=t_data)

    def drop(self, labels: Any = None, axis: int = 1):
        t_data = self._backend.drop(labels=labels, axis=axis)
        if axis == 1:
            self.roles = {c: self.roles[c] for c in t_data.columns}
        return Dataset(roles=self.roles, data=t_data)


class ExperimentData(Dataset):
    def __init__(self, data: Dataset):
        self.additional_fields = Dataset._create_empty(index=data.index)
        self.stats = Dataset._create_empty(index=data.columns)
        self.additional_fields = Dataset._create_empty(index=data.index)
        self.analysis_tables: Dict[str, Dataset] = {}
        self.id_name_mapping: Dict[str, str] = {}

        super().__init__(data=data.data, roles=data.roles)

    @staticmethod
    def _create_empty(roles=None, backend=BackendsEnum.pandas, index=None):
        ds = Dataset._create_empty(backend, roles, index)
        return ExperimentData(ds)

    def check_hash(self, executor_id: int, space: ExperimentDataEnum) -> bool:
        if space == ExperimentDataEnum.additional_fields:
            return executor_id in self.additional_fields.columns
        elif space == ExperimentDataEnum.stats:
            return executor_id in self.stats.columns
        elif space == ExperimentDataEnum.analysis_tables:
            return executor_id in self.analysis_tables
        else:
            return any(self.check_hash(executor_id, s) for s in ExperimentDataEnum)

    def set_value(
        self,
        space: ExperimentDataEnum,
        executor_id: str,
        name: str,
        value: Any,
        key: Optional[str] = None,
        role=None,
    ):
        if space == ExperimentDataEnum.additional_fields:
            self.additional_fields.add_column(data=value, role={executor_id: role})
        elif space == ExperimentDataEnum.analysis_tables:
            self.analysis_tables[executor_id] = value
        elif space == ExperimentDataEnum.stats:
            if executor_id not in self.stats.columns:
                self.stats.add_column(
                    data=[None] * len(self.stats),
                    role={executor_id: StatisticRole()},
                )
            self.stats[executor_id][key] = value
        self.id_name_mapping[executor_id] = name
        return self

    def get_ids(
        self, classes: Union[type, List[type]]
    ) -> Dict[type, Dict[str, List[str]]]:
        classes = classes if isinstance(classes, Iterable) else [classes]
        return {
            class_: {
                ExperimentDataEnum.stats.value: [
                    str(_id)
                    for _id in self.stats.columns
                    if _id.split(ID_SPLIT_SYMBOL)[0] == class_.__name__
                ],
                ExperimentDataEnum.additional_fields.value: [
                    str(_id)
                    for _id in self.additional_fields.columns
                    if _id.split(ID_SPLIT_SYMBOL)[0] == class_.__name__
                ],
                ExperimentDataEnum.analysis_tables.value: [
                    str(_id)
                    for _id in self.analysis_tables
                    if _id.split(ID_SPLIT_SYMBOL)[0] == class_.__name__
                ],
            }
            for class_ in classes
        }

    def _get_one_id(self, class_: type, space: ExperimentDataEnum) -> str:
        result = self.get_ids(class_)
        if not len(result):
            raise NotFoundInExperimentDataError(class_)
        return result[class_][space.value][0]

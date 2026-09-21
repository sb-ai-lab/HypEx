from __future__ import annotations

import inspect
import html

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from ..dataset import (
    ABCRole,
    AdditionalMatchingRole,
    Dataset,
    GroupedDataset,
    ExperimentData,
    FeatureRole,
    GroupingRole,
    TargetRole,
)
from ..utils import (
    ID_SPLIT_SYMBOL,
    AbstractMethodError,
    ExperimentDataEnum,
    NotSuitableFieldError,
    SetParamsDictTypes,
)
from ..utils.adapter import Adapter
from ..utils.constants import NAME_BORDER_SYMBOL

class Executor(ABC):
    # Maximum number of list / dict elements expanded into separate
    # sub-cells before the rendered output is truncated.
    _HTML_MAX_ITEMS: int = 10
    # Inline CSS applied to every nested HTML table.
    _HTML_TABLE: str = (
        "border-collapse:collapse; margin:2px 0 2px 10px; "
        "font-family:monospace; font-size:12px;"
    )
     # Inline CSS applied to every HTML table cell.
    _HTML_CELL: str = (
        "border:1px solid #d0d0d0; padding:2px 8px; "
        "text-align:left; vertical-align:top;"
    )

    def __init__(
        self,
        key: Any = "",
        **calc_kwargs,
    ):
        self._id: str = ""
        self._params_hash = ""

        self.key: Any = key
        self._generate_id()
        self.calc_kwargs = calc_kwargs

    def check_and_setattr(self, params: dict[str, Any]):
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
        if isinstance(next(iter(params)), str):
            self.check_and_setattr(params)
        elif isinstance(next(iter(params)), type):
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

    def get_params(self, deep: bool=False):
        """Return the initialization parameters of this executor.

        Mirrors the ``scikit-learn`` ``get_params`` API: the signature of
        ``__init__`` of the concrete class is inspected and the current
        value of every named argument is read from the instance attribute
        with the same name (falling back to the declared default).

        Args:
            deep: If ``True``, parameters of nested objects that implement
                ``get_params`` are collected recursively and stored under
                ``<name>__<nested_name>`` keys. Defaults to ``False``.

        Returns:
            A mapping ``{parameter_name: current_value}`` for every named
            ``__init__`` argument. ``*args`` / ``**kwargs`` are skipped.
        """
        init_params = inspect.signature(self.__init__).parameters
        out: dict[str, Any] = {}

        for name, param in init_params.items():
            if name == "self":
                continue

            #  **kwargs
            if param.kind == param.VAR_POSITIONAL:
                stored = getattr(self, name, None)
                if isinstance(stored, dict):
                    out.update(stored)
                continue

            # *args
            if param.kind == param.VAR_KEYWORD:
                continue

            value = getattr(self, name, param.default)

            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((name + NAME_BORDER_SYMBOL + k, val) for k, val in deep_items)

            out[name] = value

        return out

    def _repr_params(self) -> dict[str, Any]:
        """Collect the parameters shown by ``__repr__`` and ``_repr_html_``.

        Equivalent to ``get_params(deep=False)`` with one addition: when
        ``calc_kwargs`` was passed through ``**kwargs`` (and therefore is
        absent from the ``__init__`` signature), it is appended to the
        result so that implicitly passed calculation options stay visible.

        Returns:
            A mapping ``{parameter_name: value}`` ready to be rendered.
        """
        params = self.get_params(deep=False)
        if (
            hasattr(self, "calc_kwargs")
            and self.calc_kwargs
            and "calc_kwargs" not in params
        ):
            params = {**params, "calc_kwargs": self.calc_kwargs}
        return params

    @classmethod
    def _html_value(cls, value: Any) -> str:
        """Render a single parameter value as an HTML snippet.

        Rendering rules (applied recursively):

        * :class:`Executor` -- a collapsible ``<details>`` block whose
          content is the executor's own parameter table;
        * ``range`` -- a compact ``repr`` (ranges such as
          ``random_states`` may contain thousands of elements);
        * ``list`` / ``tuple`` -- a sub-table with one row per element,
          truncated after ``_HTML_MAX_ITEMS`` rows;
        * ``Mapping`` -- a sub-table with one row per key; class objects
          used as keys are displayed via their ``__name__``;
        * any other value -- an escaped ``repr`` inside a ``<pre>`` block.

        Args:
            value: The parameter value to render.

        Returns:
            An HTML string safe for embedding into a table cell.
        """
        if isinstance(value, Executor):
            return (
                f"<details><summary style='cursor:pointer; font-family:monospace;'>"
                f"<b>{type(value).__name__}</b></summary>"
                f"{value._repr_html_()}</details>"
            )
        if isinstance(value, range):
            return f"<pre style='margin:0;'>{html.escape(repr(value))}</pre>"
        if isinstance(value, (list, tuple)):
            items = list(value)
            truncated = len(items) > cls._HTML_MAX_ITEMS
            shown = items[: cls._HTML_MAX_ITEMS] if truncated else items
            body = "".join(
                f"<tr>"
                f"<td style='{cls._HTML_CELL}; color:#999;'>[{i}]</td>"
                f"<td style='{cls._HTML_CELL}'>{cls._html_value(item)}</td>"
                f"</tr>"
                for i, item in enumerate(shown)
            )
            if truncated:
                body += (
                    f"<tr><td colspan='2' style='{cls._HTML_CELL}; color:#999;'>"
                    f"… и ещё {len(items) - cls._HTML_MAX_ITEMS}</td></tr>"
                )
            return f"<table style='{cls._HTML_TABLE}'>{body}</table>"
        if isinstance(value, Mapping):
            body = "".join(
                f"<tr>"
                f"<td style='{cls._HTML_CELL}'><code>"
                f"{html.escape(getattr(k, '__name__', None) or repr(k))}</code></td>"
                f"<td style='{cls._HTML_CELL}'>{cls._html_value(v)}</td>"
                f"</tr>"
                for k, v in value.items()
            )
            return f"<table style='{cls._HTML_TABLE}'>{body}</table>"
        return f"<pre style='margin:0;'>{html.escape(repr(value))}</pre>"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        params = self._repr_params()
        if not params:
            return f"{class_name}()"
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{class_name}({params_str})"

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr>"
            f"<td style='{self._HTML_CELL}'><code>{html.escape(str(name))}</code></td>"
            f"<td style='{self._HTML_CELL}'>{self._html_value(value)}</td>"
            f"</tr>"
            for name, value in self._repr_params().items()
        )
        header = (
            f"<tr><th style='{self._HTML_CELL}'>Parameter</th>"
            f"<th style='{self._HTML_CELL}'>Value</th></tr>"
        )
        return (
            f"<div style='font-family:monospace; display:inline-block;'>"
            f"<b>{html.escape(type(self).__name__)}</b>"
            f"<table style='{self._HTML_TABLE}'>{header}{rows}</table>"
            f"</div>"
        )


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

    @staticmethod
    def _check_test_data(
        test_data: Dataset | None = None,
    ) -> Dataset:  # TODO to move away from Calculator. Where to?
        if test_data is None:
            raise ValueError("test_data is needed for comparison")
        return test_data


class MLExecutor(Calculator, ABC):
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_role: ABCRole | None = None,
        key: Any = "",
    ):
        self.target_role = target_role or TargetRole()
        super().__init__(key=key)
        self.grouping_role = grouping_role or GroupingRole()

    def _get_fields(self, data: ExperimentData):
        group_field = data.field_search(self.grouping_role)
        target_field = data.field_search(
            self.target_role, search_types=self.search_types
        )
        return group_field, target_field

    @abstractmethod
    def fit(self, X: Dataset, Y: Dataset | None = None) -> MLExecutor:
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
        test_data: Dataset | None = None,
        target_data: Dataset | None = None,
        **kwargs,
    ) -> Any:
        raise AbstractMethodError

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        tmp_roles,
        target_field: str | None = None,
        **kwargs,
    ) -> Any:
        (_, _data), (_, _test_data), *_ = grouping_data
        _data.tmp_roles = tmp_roles

        if target_field:
            return cls._inner_function(
                data=_data.drop(target_field),
                target_data=_data[target_field],
                test_data=_test_data.drop(target_field),
                **kwargs,
            )
        return cls._inner_function(
            data=_data,
            test_data=_test_data,
            **kwargs,
        )

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        for i in range(value.shape[1]):
            data.set_value(
                ExperimentDataEnum.additional_fields,
                f"{self.id}{ID_SPLIT_SYMBOL}{i}",
                value=value.iloc[:, i],
                key=key,
                role=AdditionalMatchingRole(),
            )
        return data

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Sequence[str] | str | None = None,
        # grouping_data: list[tuple[str, Dataset]] | None = None,
        grouping_data: GroupedDataset | None = None, 
        target_field: str | list[str] | None = None,
        features_fields: str | list[str] | None = None,
        **kwargs,
    ) -> Dataset:
        group_field = Adapter.to_list(group_field)
        features_fields = Adapter.to_list(features_fields)
        if grouping_data is None:
            grouping_data = data[group_field + features_fields].groupby(group_field)
        if len(grouping_data) > 1:
            # grouping_data[0][1].tmp_roles = data.tmp_roles
            tmp_roles = data.tmp_roles
        else:
            raise NotSuitableFieldError(group_field, "Grouping")
        result = cls._execute_inner_function(
            grouping_data, tmp_roles, target_field=target_field, **kwargs
        )
        return result

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
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
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
        # TODO: add roles to compare_result
        return self._set_value(data, compare_result)


class IfExecutor(Executor, ABC):
    def __init__(
        self,
        if_executor: Executor | None = None,
        else_executor: Executor | None = None,
        key: Any = "",
    ):
        self.if_executor = if_executor
        self.else_executor = else_executor
        super().__init__(key)

    @abstractmethod
    def check_rule(self, data, **kwargs) -> bool:
        raise AbstractMethodError

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        return data.set_value(
            ExperimentDataEnum.variables, self.id, value, key="response"
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        if self.check_rule(data):
            return (
                self.if_executor.execute(data)
                if self.if_executor is not None
                else self._set_value(data, True)
            )
        return (
            self.else_executor.execute(data)
            if self.else_executor is not None
            else self._set_value(data, False)
        )

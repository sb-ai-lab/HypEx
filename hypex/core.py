import importlib
import pkgutil
import re
from typing import Type, Optional, Any, Dict, Union, get_type_hints, TypeVar, Callable
from types import MethodType
import pandas as pd
from functools import wraps
from .context import Context
from .pipeline import Pipeline


T = TypeVar('T', bound='Hypex')




class Hypex:
    """Главный класс модуля для работы с пайплайнами"""


    def __init__(self, df: pd.DataFrame):
        self.context = Context(df)
        self._discover_pipelines()

    @staticmethod
    def _chainable_method(func: Callable[..., T]) -> Callable[..., T]:
        """Декоратор для методов, возвращающих Hypex"""

        @wraps(func)
        def wrapper(self, **kwargs) -> T:
            return func(self, **kwargs)

        # Явно указываем тип возвращаемого значения
        wrapper.__annotations__ = {'return': 'Hypex'}
        return wrapper

    def _update_type_hints(self):
        """Обновляем подсказки типов для динамических методов"""
        type_hints = get_type_hints(self.__class__)
        for name, method in self.__class__.__dict__.items():
            if isinstance(method, MethodType) and not name.startswith('_'):
                type_hints[name] = MethodType[..., 'Hypex']

        if hasattr(self.__class__, '__annotations__'):
            self.__class__.__annotations__.update(type_hints)

    def __dir__(self):
        """Возвращает список атрибутов для автодополнения"""
        base_attrs = set(super().__dir__())
        pipeline_methods = {
            name for name in self.__class__.__dict__.keys()
            if not name.startswith('_') and callable(getattr(self.__class__, name))
        }
        return sorted(base_attrs | pipeline_methods)

    def _discover_pipelines(self) -> None:
        """Обнаружение и регистрация всех доступных пайплайнов"""
        self._register_builtin_pipelines()
        self._discover_plugin_pipelines()

    def _register_builtin_pipelines(self) -> None:
        """Регистрация пайплайнов из hypex.pipelines"""
        try:
            pipelines_module = importlib.import_module("hypex.pipelines")
            for module_name in getattr(pipelines_module, "__all__", []):
                full_path = f"hypex.pipelines.{module_name}"
                self._register_pipeline_module(full_path, module_name)
        except Exception as e:
            print(f"Error loading builtin pipelines: {str(e)}")

    def _discover_plugin_pipelines(self) -> None:
        """Обнаружение плагинов пайплайнов"""
        for _, name, _ in pkgutil.iter_modules():
            if name.startswith("hypex_pipeline_"):
                try:
                    module = importlib.import_module(name)
                    pipeline_name = name.replace("hypex_pipeline_", "")
                    self._register_pipeline_from_module(module, pipeline_name)
                except Exception as e:
                    print(f"Error loading plugin {name}: {str(e)}")

    def _register_pipeline_module(self, module_path: str, pipeline_name: str) -> None:
        """Регистрация пайплайна из модуля"""
        try:
            module = importlib.import_module(module_path)
            self._register_pipeline_from_module(module, pipeline_name)
        except Exception as e:
            print(f"Error registering pipeline {pipeline_name}: {str(e)}")

    def _register_pipeline_from_module(self, module: Any, pipeline_name: str) -> None:
        """Поиск и регистрация классов пайплайнов в модуле"""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and issubclass(attr, Pipeline) and attr != Pipeline):
                self._add_pipeline_method(attr, pipeline_name)

    def _add_pipeline_method(self, pipeline_class: Type[Pipeline], method_name: str) -> None:
        """Добавление метода с поддержкой автодополнения"""

        @self._chainable_method
        def pipeline_method(self: T, **kwargs) -> T:
            pipeline = pipeline_class(self.context, **kwargs)
            pipeline.run()
            return self

        pipeline_method.__doc__ = pipeline_class.__doc__
        method_name = self._to_snake_case(method_name)

        # Добавляем метод в класс
        setattr(self.__class__, method_name, pipeline_method)

        # Обновляем аннотации типов
        if not hasattr(self.__class__, '__annotations__'):
            self.__class__.__annotations__ = {}
        self.__class__.__annotations__[method_name] = Callable[..., 'Hypex']

    @classmethod
    def register_pipeline(cls, pipeline_class: Type[Pipeline]) -> None:
        """Явная регистрация пайплайна"""
        method_name = cls._to_snake_case(pipeline_class.__name__)

        @cls._chainable_method
        def pipeline_method(self: T, **kwargs) -> T:
            pipeline = pipeline_class(self.context, **kwargs)
            pipeline.run()
            return self

        pipeline_method.__doc__ = pipeline_class.__doc__
        setattr(cls, method_name, pipeline_method)

        if not hasattr(cls, '__annotations__'):
            cls.__annotations__ = {}
        cls.__annotations__[method_name] = Callable[..., 'Hypex']

    @classmethod
    def _register_explicit_pipeline(cls, pipeline_class: Type[Pipeline], method_name: str) -> None:
        """Внутренний метод регистрации"""

        def pipeline_method(self, **kwargs) -> "Hypex":
            pipeline = pipeline_class(self.context, **kwargs)
            pipeline.run()
            return self

        pipeline_method.__doc__ = pipeline_class.__doc__
        setattr(cls, method_name, pipeline_method)

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Конвертация CamelCase в snake_case"""
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
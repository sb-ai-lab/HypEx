from typing import List, Callable,Any, Generic, TypeVar

from .context import Context

T = TypeVar('T', bound='Pipeline')

class Pipeline:
    """Базовый класс для всех pipeline-обработчиков"""

    def __init__(self, context: Context, **kwargs):
        self.context = context
        context.push_pipeline_context()
        self.context.pipeline_context['kwargs'] = kwargs
        self.funcs: List[Callable] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Автоматически регистрируем подклассы при импорте
        if cls.__module__.startswith('hypex.pipelines.'):
            from hypex import Hypex
            Hypex.register_pipeline(cls)

    def custom_run(self) -> 'Pipeline':
        """Основной метод для переопределения в дочерних классах"""
        pass

    def run(self) -> 'Pipeline':
        """Основной метод выполнения pipeline"""
        if hasattr(self, 'custom_run') and callable(self.custom_run):
            self.custom_run()
        else:
            for func in self.funcs:
                func(self.context)

        self.context.pop_pipeline_context()
        return self
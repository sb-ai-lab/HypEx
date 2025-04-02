from typing import Dict

import pandas as pd


class Context:
    """Класс для хранения контекста выполнения"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.pipeline_context = {}
        self.pipeline_context_stack = []

    def push_pipeline_context(self) -> None:
        """Сохраняет текущий контекст в стек и создает новый пустой"""
        self.pipeline_context_stack.append(self.pipeline_context)
        self.pipeline_context = {}

    def pop_pipeline_context(self) -> Dict:
        """Восстанавливает предыдущий контекст из стека"""
        if self.pipeline_context_stack:
            self.pipeline_context = self.pipeline_context_stack.pop()
        return self.pipeline_context
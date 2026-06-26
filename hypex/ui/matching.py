from __future__ import annotations

from typing import Any

from ..dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    TargetRole,
)
from ..reporters.matching import MatchingReporter, MatchingQualityReporter
from .base import Output


class MatchingOutput(Output):
    """
    Output handler for Matching experiments.
    
    Automatically extracts:
    - resume: Dataset with ATT, ATC, ATE metrics.
    - quality_results: Dict of Datasets with quality tests (T-Test, Chi2, KS) per feature.
    - full_data: Original dataset enriched with matched features (_matched_0, _matched_1, etc.).
    """
    resume: Dataset
    full_data: Dataset
    quality_results: dict[str, Dataset]

    def __init__(self, *args, **kwargs):
        # ВАЖНО: additional_reporters должен быть словарем, где ключ - это имя атрибута,
        # в который базовый класс Output автоматически сохранит результат репортера.
        super().__init__(
            resume_reporter=MatchingReporter(),
            additional_reporters={
                "quality_results": MatchingQualityReporter()
            },
        )

    def extract(self, experiment_data: ExperimentData):
        # Базовый extract сам вызовет .report() у resume_reporter и additional_reporters
        # и сохранит результаты в self.resume и self.quality_results
        super().extract(experiment_data)
        
        # Округляем resume для красивого вывода в Jupyter
        if self.resume is not None and not self.resume.is_empty():
            print(f"[DEBUG MatchingOutput] self.resume.is_empty(): {self.resume.is_empty()}")
            print(f"[DEBUG MatchingOutput] self.resume.columns: {self.resume.columns}")
            print(f"[DEBUG MatchingOutput] self.resume.roles: {list(self.resume.roles.keys())}")
            print(f"[DEBUG MatchingOutput] self.resume.data:\n{self.resume.data}")
            # Округляем resume для красивого вывода (теперь отработает корректно через Dataset.__round__)
            if self.resume is not None and not self.resume.is_empty():
                self.resume = round(self.resume, 4)
            
        # Собираем full_data (исходный датасет + сматченные признаки)
        self._extract_full_data(experiment_data)

    def _extract_full_data(self, experiment_data: ExperimentData):
        """
        Собирает full_data, опираясь на additional_fields (где лежат индексы пар).
        FaissNearestNeighbors сохраняет колонки с индексами сматченных пар 
        с ролью AdditionalMatchingRole().
        """
        self.full_data = experiment_data.ds
        
        # Находим колонки с индексами сматченных пар
        index_cols = experiment_data.ds.search_columns(AdditionalMatchingRole())
        if not index_cols:
            return

        # Сортируем колонки, чтобы порядок соседей был детерминированным
        for i, col in enumerate(sorted(index_cols)):
            t_indexes = experiment_data.ds[[col]]
            
            # Получаем значения индексов и оригинальные индексы строк
            idx_values = t_indexes[col].get_values()
            orig_indices = list(t_indexes.index)
            
            valid_orig_indices = []
            valid_matched_indices = []
            
            # Фильтруем валидные пары (исключаем -1, который означает отсутствие пары, и NaN)
            for orig_idx, matched_idx in zip(orig_indices, idx_values):
                if matched_idx != -1 and matched_idx is not None and str(matched_idx) != 'nan':
                    valid_orig_indices.append(orig_idx)
                    valid_matched_indices.append(matched_idx)
            
            if not valid_matched_indices:
                continue
                
            try:
                # Берем строки из оригинального датасета по matched индексам
                matched_data = experiment_data.ds.loc[valid_matched_indices]
                
                # Переименовываем колонки, добавляя суффикс _matched_{i}
                rename_dict = {c: f"{c}_matched_{i}" for c in experiment_data.ds.columns}
                matched_data = matched_data.rename(rename_dict)
                
                # Возвращаем оригинальный индекс для выравнивания
                matched_data.index = valid_orig_indices
                
                # Reindex до полного размера, чтобы выровнять с full_data (заполнит NaN там, где нет пар)
                matched_data = matched_data.reindex(experiment_data.ds.index)
                
                # Джойним с основным датасетом по колонкам (axis=1)
                self.full_data = self.full_data.append(matched_data, axis=1)
            except Exception:
                # В случае проблем с бэкендом (например, специфичный loc в Spark) просто пропускаем
                # эту колонку, чтобы не ломать весь пайплайн
                continue
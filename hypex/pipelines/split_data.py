import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from scipy import stats
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

from hypex import Pipeline
from hypex.context import Context


class SplitData(Pipeline):
    """Пайплайн для оптимального разделения данных на группы"""
    def _validate_params(self):
        """Проверка корректности параметров"""
        params = self.context.pipeline_context['kwargs']

        if 'group_column' not in params:
            raise ValueError("Не указано имя колонки для групп (group_column)")

        if 'group_sizes' not in params or not isinstance(params['group_sizes'], dict):
            raise ValueError("Не указаны размеры групп (group_sizes должен быть dict)")

        if not np.isclose(sum(params['group_sizes'].values()), 1.0, atol=0.01):
            raise ValueError("Сумма долей групп должна быть равна 1")

        if 'n_iter' not in params:
            self.context.pipeline_context['kwargs']['n_iter'] = 100

        if 'random_state' not in params:
            self.context.pipeline_context['kwargs']['random_state'] = 42

    def _calculate_p_value(self, group1: pd.Series, group2: pd.Series, test: str) -> float:
        """Вычисляет p-value для двух групп с указанным тестом"""
        if test == 'ks':
            _, p_value = stats.ks_2samp(group1, group2)
        elif test == 'anderson':
            result = stats.anderson_ksamp([group1, group2])
            p_value = result.significance_level
        else:
            raise ValueError(f"Неизвестный тест: {test}")

        return p_value

    def _evaluate_split(self, df: pd.DataFrame, split_indices: Dict[str, np.ndarray]) -> float:
        """Оценивает качество разделения по всем фичам и тестам"""
        total_p_value = 0
        n_tests = 0

        # Получаем список всех групп
        groups = list(split_indices.keys())

        # Сравниваем каждую пару групп
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1 = df.iloc[split_indices[groups[i]]]
                group2 = df.iloc[split_indices[groups[j]]]

                # Для каждой фичи и каждого теста
                for col in df.select_dtypes(include=[np.number]).columns:
                    for test in self.context.pipeline_context['stats_tests']:
                        try:
                            p_value = self._calculate_p_value(group1[col], group2[col], test)
                            total_p_value += p_value
                            n_tests += 1
                        except:
                            continue

        return total_p_value / n_tests if n_tests > 0 else 0

    def _generate_split(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Генерирует случайное разделение данных"""
        params = self.context.pipeline_context['kwargs']
        group_sizes = params['group_sizes']
        indices = np.arange(len(df))

        # Сначала делим на первую группу и остаток
        group1, remaining = train_test_split(
            indices,
            test_size=1 - group_sizes[list(group_sizes.keys())[0]],
            random_state=params['random_state']
        )

        split_indices = {list(group_sizes.keys())[0]: group1}

        # Затем последовательно делим остаток
        remaining_groups = list(group_sizes.keys())[1:]
        for i, group_name in enumerate(remaining_groups):
            if i == len(remaining_groups) - 1:
                split_indices[group_name] = remaining
            else:
                next_group_size = group_sizes[group_name] / (1 - sum(list(group_sizes.values())[:i + 1]))
                group, remaining = train_test_split(
                    remaining,
                    test_size=1 - next_group_size,
                    random_state=params['random_state']
                )
                split_indices[group_name] = group

        return split_indices

    def _find_best_split(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Находит оптимальное разделение данных"""
        params = self.context.pipeline_context['kwargs']
        best_score = -1
        best_split = None

        for _ in range(params['n_iter']):
            # Генерируем случайное разделение
            split_indices = self._generate_split(df)

            # Оцениваем качество разделения
            score = self._evaluate_split(df, split_indices)

            # Обновляем лучшее разделение
            if score > best_score:
                best_score = score
                best_split = split_indices

            # Обновляем random_state для следующей итерации
            params['random_state'] += 1

        self.context.pipeline_context['best_score'] = best_score
        return best_split

    def custom_run(self):
        """Основной метод выполнения пайплайна"""

        self.context.pipeline_context['stats_tests']= ['ks', 'anderson']  # Тесты по умолчанию
        self._validate_params()
        params = self.context.pipeline_context['kwargs']
        df = self.context.df.copy()

        # Находим оптимальное разделение
        best_split = self._find_best_split(df)

        # Создаем колонку с группами
        group_column = params['group_column']
        df[group_column] = ""

        for group_name, indices in best_split.items():
            df.loc[df.index[indices], group_column] = group_name

        # Сохраняем результат в контексте
        self.context.df= df

        return  self
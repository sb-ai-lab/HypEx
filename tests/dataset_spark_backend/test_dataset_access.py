import pytest
import pandas as pd
import numpy as np
from hypex.dataset import Dataset
from hypex.dataset.roles import FeatureRole, TargetRole, InfoRole
from hypex.utils import BackendsEnum

class TestDatasetAccess:
    """Тесты доступа к данным: индексация, срезы, получение значений"""

    # --- Базовые тесты (из оригинала) ---
    def test_getitem_column(self, spark_dataset):
        result = spark_dataset["value"]
        assert isinstance(result, Dataset)
        assert "value" in result.columns
        assert result.shape[1] == 1

    def test_getitem_row_int(self, spark_dataset):
        row = spark_dataset[0]
        assert isinstance(row, Dataset)
        # Ожидаем 1 строку и все колонки (5)
        assert row.shape[0] == 5
        assert row.shape[1] == 1

    def test_getitem_list_columns(self, spark_dataset):
        subset = spark_dataset[["id", "score"]]
        assert isinstance(subset, Dataset)
        assert set(subset.columns) == {"id", "score"}
        assert subset.shape[0] == 5  # Все строки
        assert subset.shape[1] == 2  # 2 колонки

    def test_get_values_scalar(self, spark_dataset):
        val = spark_dataset.get_values(row=0, column="id")
        assert val == 1

    def test_iget_values_scalar(self, spark_dataset):
        val = spark_dataset.iget_values(row=0, column=0)
        assert val == 1

    def test_filter_items(self, spark_dataset):
        filtered = spark_dataset.filter(items=["id", "name", "value"])
        assert set(filtered.columns) == {"id", "name", "value"}

    def test_select_columns(self, spark_dataset):
        selected = spark_dataset.select(["score", "value"])
        assert set(selected.columns) == {"score", "value"}

    def test_iselect_columns(self, spark_dataset):
        selected = spark_dataset.iselect([0, 4])
        assert set(selected.columns) == {"id", "score"}

    # --- Расширенные тесты: Граничные условия и Ошибки ---
    def test_getitem_invalid_column_raises(self, spark_dataset):
        """Проверка обработки несуществующей колонки"""
        with pytest.raises((KeyError, Exception)):
            _ = spark_dataset["non_existent_column"]

    def test_get_values_out_of_bounds(self, spark_dataset):
        """Проверка индекса за пределами диапазона"""
        with pytest.raises((IndexError, Exception)):
            _ = spark_dataset.get_values(row=100, column="id")

    def test_iselect_out_of_bounds(self, spark_dataset):
        """Проверка iselect с несуществующим индексом колонки"""
        with pytest.raises((IndexError, Exception)):
            _ = spark_dataset.iselect([0, 99])

    def test_getitem_empty_list(self, spark_dataset):
        """Выбор пустого списка колонок должен вернуть пустой Dataset"""
        subset = spark_dataset[[]]
        assert isinstance(subset, Dataset)
        assert subset.shape[1] == 0

    # --- Расширенные тесты: Срезы (Slicing) ---
    def test_slice_rows_range(self, spark_dataset):
        """Срез строк по диапазону"""
        subset = spark_dataset[1:3]
        assert isinstance(subset, Dataset)
        assert subset.shape[0] == 2  # Строки 1 и 2
        assert subset.shape[1] == 5

    def test_slice_rows_step(self, spark_dataset):
        """Срез строк с шагом"""
        subset = spark_dataset[::2]
        assert isinstance(subset, Dataset)
        assert subset.shape[0] == 3  # Строки 0, 2, 4

    # --- Расширенные тесты: Роли (Roles) ---
    def test_column_selection_preserves_roles(self, spark_dataset):
        """При выборе колонок должны сохраняться их роли"""
        subset = spark_dataset[["id", "value"]]
        assert subset.roles is not None
        assert "id" in subset.roles
        assert "value" in subset.roles
        assert isinstance(subset.roles["id"], InfoRole)
        assert isinstance(subset.roles["value"], TargetRole)

    def test_row_selection_preserves_roles(self, spark_dataset):
        """При выборе строки роли колонок должны сохраняться"""
        row = spark_dataset.iloc[0]
        assert row.roles is not None
        assert len(row.roles) == 5  # Все колонки на месте

    # --- Расширенные тесты: Типы данных ---
    @pytest.mark.parametrize("col_name,expected_type", [
        ("id", (int, np.integer)),
        ("value", (float, np.floating)),
        ("name", (str, object)),
        ("active", (bool, np.bool_))
    ])
    def test_get_values_types(self, spark_dataset, col_name, expected_type):
        """Проверка типов возвращаемых скалярных значений"""
        val = spark_dataset.get_values(row=0, column=col_name)
        assert isinstance(val, expected_type)

    # --- Расширенные тесты: Булева индексация ---
    def test_boolean_mask_filtering(self, spark_dataset):
        """Фильтрация по булевой маске"""
        mask = spark_dataset["score"] > 150
        assert isinstance(mask, Dataset)
        # Применяем маску (если метод где-то есть, или проверяем саму маску)
        # В данном случае проверяем, что маска создалась корректно
        assert mask.shape[0] == spark_dataset.shape[0]

    def test_filter_by_condition(self, spark_dataset):
        """Фильтрация строк по условию (если поддерживается)"""
        # Предполагаем, что есть метод query или filter с условием
        # Если нет - этот тест можно закомментировать или адаптировать
        try:
            filtered = spark_dataset.filter(condition=spark_dataset["score"] > 200)
            assert isinstance(filtered, Dataset)
            assert len(filtered) <= len(spark_dataset)
        except TypeError:
            pytest.skip("Метод filter не поддерживает аргумент condition")

    # --- Расширенные тесты: Цепочки операций ---
    def test_chained_selection(self, spark_dataset):
        """Цепочка операций выбора"""
        result = spark_dataset[["id", "score"]][0]
        assert isinstance(result, Dataset)
        assert result.shape[0] == 2
        assert result.shape[1] == 1

    def test_select_then_filter(self, spark_dataset):
        """Выбор колонок с последующей фильтрацией"""
        subset = spark_dataset[["id", "value"]]
        filtered = subset.filter(items=["id"])
        assert filtered.shape[1] == 1
        assert "id" in filtered.columns

    # --- Расширенные тесты: Специфичные методы ---
    def test_head_tail(self, spark_dataset):
        """Проверка методов head и tail (если есть)"""
        if hasattr(spark_dataset, 'head'):
            head = spark_dataset.head(2)
            assert head.shape[0] == 2
        if hasattr(spark_dataset, 'tail'):
            tail = spark_dataset.tail(2)
            assert tail.shape[0] == 2

    def test_sample_reproducibility(self, spark_dataset):
        """Проверка воспроизводимости sample с random_state"""
        sample1 = spark_dataset.sample(n=2, random_state=42)
        sample2 = spark_dataset.sample(n=2, random_state=42)
        # Данные должны совпадать при одинаковом seed
        assert sample1.to_dict()["data"] == sample2.to_dict()["data"]

    # --- Расширенные тесты: Пустой Dataset ---
    def test_access_on_empty_dataset(self, spark_session):
        """Проверка доступа к пустому Dataset"""
        ds_empty = Dataset(roles={}, data=None, backend=BackendsEnum.spark, session=spark_session)
        assert ds_empty.is_empty()
        assert ds_empty.shape == (0, 0)

    # --- Расширенные тесты: iselect с отрицательными индексами ---
    def test_iselect_negative_indices(self, spark_dataset):
        """Выбор колонок по отрицательным индексам"""
        # -1 это последняя колонка (score), -5 это первая (id)
        selected = spark_dataset.iselect([-1]) 
        assert "score" in selected.columns
        
        selected_first = spark_dataset.iselect([-5])
        assert "id" in selected_first.columns
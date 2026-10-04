# Известные ошибки библиотеки (найдены при покрытии тестами)

Источник: ветка `dev/test-coverage-80`, замер 2026-10-04. Для каждой ошибки в `tests/` стоит `xfail(strict=True)` с `reason=`: когда ошибку исправят, тест выдаст XPASS и упадёт — тогда снять маркер. Поиск: `grep -rn "xfail" tests`.

## 1. Сквозные причины (ломают много тестов)

| Ошибка | Где | Эффект |
|---|---|---|
| ИСПРАВЛЕНО: `PandasDataset.count_groups` делает `int(df[cols].nunique())` | `hypex/dataset/backends/pandas_backend.py` | FutureWarning (в pandas), `TypeError` при нескольких колонках групп и на Spark; ломает любой `list(groupby)`: `GroupedDataset`, `GroupOperator.calc`, `GroupExperiment`, `MahalanobisDistance`, split-modes, calculators, `MLExecutor`, `MinSampleSize` |
| `HomogeneityTest()` создаёт deprecated `HomoDatasetReporter` | `hypex/ui/homo.py` | DeprecationWarning → ошибка из-за `filterwarnings` в `pyproject.toml`; фасад нельзя вызвать в тестах |
| `Matching()` / `MatchingOutput()` создают deprecated `MatchingDictReporter` / `MatchingDatasetReporter` / `MatchingQualityDatasetReporter` | `hypex/matching.py`, `hypex/ui/matching.py` | то же; старые xfail-причины Matching (соседи из противоположной группы, ближайший по ковариате, `_match_pandas`, `group_match`) нельзя перепроверить, пока репортёры deprecated |

## 2. Dataset / adapter

- `DatasetAdapter.list_to_dataset` / `ndarray_to_dataset` с `small=False` вызывают несуществующий `Dataset.to_dataset()`.
- `Dataset.get(key)` пересоздаёт Dataset со всеми исходными ролями → `RoleColumnError`.
- `Dataset.dot` с 1-D вектором: pandas — `IndexError`, Spark — `.assign` на Series.
- `GroupedDataset` со списком групп использует `dataset_class._backend.concat`, у `Dataset` нет `_backend` → `AttributeError` в `agg` и `apply`.
- `GroupedDataset.apply` на pandas: DeprecationWarning о группирующих колонках.

## 3. Spark-бэкенд

- Результаты `agg` не адресуются как в pandas (`KeyError`); `quantile` возвращает транспонированный фрейм.
- `std(ddof=1)` на одной строке: `float(None)`.
- Сеттер индекса передаёт список в `set_index` как имена колонок (`KeyError`) — `reset_index(drop=True)`.
- `add_column` присваивает DataFrame колонке.
- `__and__`, `__or__`, `__pos__` работают на pyspark.pandas DataFrame и не работают.
- `SparkBisaExtesion.prepare_data` вызывает `result_to_dataset` без обязательного `roles` (`TypeError`).
- `SparkKSTestExtension` с `nan_policy="omit"` не отбрасывает NaN (результат 1.0 / 0.0, в pandas иначе).
- `SparkFaissExtension` определяет `__enter__`, но не `__exit__`.
- `MultiTest._calc_spark` теряет составной строковый индекс при `to_backend` — поправка ничего не делает.
- `StatsUTest._execute_spark` → `ImportError`: `StatsUTestExtension` не существует (строки ~922–962 не покрыты).

## 4. Статистика / comparators / extensions

- `GroupChi2TestExtension.calc` передаёт `_form_results(p_value, statistic, ...)` в перепутанном порядке (pandas и Spark).
- `StatsKSTest` / `StatsUTest` на pandas: код ставит `delegate._id`, а результаты лежат под `GroupKSTest┴┴y` / `GroupUTest┴┴y` — поиск по id Stats-исполнителя не находит.
- `PandasLstsqExtension.calc` через `create_empty().fillna()` даёт pandas FutureWarning (3 теста lstsq).
- `MultitestQuantile._calc_pandas` не тестируется: вызов `groupby(fields_list=...)` падает дальше, т.к. в группах остаётся строковая колонка группы.
- `MahalanobisDistance` при одной группе всегда бросает `ValueError("test_data is needed ...")` — ветка `test_data=None` не может сработать.
- Сопоставление ролей игнорирует `data_type`: `Float32Caster(target_roles=InfoRole(int))` выбирает float-колонку Info.

## 5. ML / matching / faiss

- `FaissExtension.fit` (публичный) передаёт `target_data=` в `MLExtension.calc`, такого параметра нет (`TypeError`); то же на уровне `FaissNearestNeighbors.fit`.
- `FaissNearestNeighbors.execute` в режимах one-sided и `test_pairs` вызывает `Dataset.reindex` (есть только у `SmallDataset`) — `AttributeError`, `hypex/ml/faiss.py:~332`.
- `FaissNearestNeighbors._execute_inner_function` при `test_pairs=True, two_sides=True`: запись "test" повторяет запрос control вместо сопоставления test→control.
- Текст предупреждения "nans ... replaced with dummy matches" неверен: NaN по-прежнему вызывают `PairsNotFoundError`.
- `MLExecutor._execute_inner_function` с `target_field` вызывает `_data.drop(target_field)` без `axis=1` — pandas пытается удалить строку.
- `MLExecutor.execute` передаёт `target_fields=`, а `calc` принимает `target_field` — таргет не доходит до `_inner_function` (теста нет).

## 6. UI / reporters

- `ABOutput.variance_reduction_report` вызывает `ABTestReporter.report_variance_reductions`, которого больше нет.

## 7. Прочее наблюдаемое

- `strict_abc` вычисляет аннотации «как написано», поэтому в файлах с `from __future__ import annotations` проверки возвращаемых типов не работают (тесты `test_strict_abc.py` намеренно без этого импорта).
- `faiss` в Spark: партиционные функции исполняются на executor'ах, `coverage` их не видит (`extensions/faiss.py` ~72%).

## Тесты, где поведение изменилось и тест подогнан под новый контракт

`CachingIndex.get(reference, storage)` (nprobe вынесен из кэша); Welch — t-test по умолчанию; в AB «significant = OK»; AA-резюме — одна строка на признак; shell копирует `ExperimentData`. Подробности: `git log` ветки, коммиты шага «fix failing tests».

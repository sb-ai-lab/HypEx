# Известные ошибки библиотеки (найдены при покрытии тестами)

Источник: ветка `dev/test-coverage-80`, замер 2026-10-04. Для каждой ошибки в `tests/` стоит `xfail(strict=True)` с `reason=`: когда ошибку исправят, тест выдаст XPASS и упадёт — тогда снять маркер. Поиск: `grep -rn "xfail" tests`.

## 1. Сквозные причины (ломают много тестов)

| Ошибка | Где | Эффект |
|---|---|---|
| ИСПРАВЛЕНО: `PandasDataset.count_groups` делает `int(df[cols].nunique())` | `hypex/dataset/backends/pandas_backend.py` | FutureWarning (в pandas), `TypeError` при нескольких колонках групп и на Spark; ломает любой `list(groupby)`: `GroupedDataset`, `GroupOperator.calc`, `GroupExperiment`, `MahalanobisDistance`, split-modes, calculators, `MLExecutor`, `MinSampleSize` |
| ИСПРАВЛЕНО: `HomogeneityTest()` создаёт deprecated `HomoDatasetReporter` | `hypex/ui/homo.py` | DeprecationWarning → ошибка из-за `filterwarnings` в `pyproject.toml`; фасад нельзя вызвать в тестах |
| ИСПРАВЛЕНО (deprecated-репортёры удалены, фасады используют `MatchingReporter`/`MatchingQualityReporter`/`HomogeneityReporter`): `Matching()` / `MatchingOutput()` создают deprecated `MatchingDictReporter` / `MatchingDatasetReporter` / `MatchingQualityDatasetReporter` | `hypex/matching.py`, `hypex/ui/matching.py` | то же; старые xfail-причины Matching (соседи из противоположной группы, ближайший по ковариате, `_match_pandas`, `group_match`) нельзя перепроверить, пока репортёры deprecated |

## 2. Dataset / adapter

- ИСПРАВЛЕНО: `DatasetAdapter.list_to_dataset` / `ndarray_to_dataset` с `small=False` вызывают несуществующий `Dataset.to_dataset()`.
- ИСПРАВЛЕНО: `Dataset.get(key)` пересоздаёт Dataset со всеми исходными ролями → `RoleColumnError`.
- ИСПРАВЛЕНО: `Dataset.dot` с 1-D вектором: pandas — `IndexError`, Spark — `.assign` на Series.
- ИСПРАВЛЕНО: `GroupedDataset` со списком групп использует `dataset_class._backend.concat`, у `Dataset` нет `_backend` → `AttributeError` в `agg` и `apply`.
- `GroupedDataset.apply` на pandas: DeprecationWarning о группирующих колонках.

## 3. Spark-бэкенд

- Результаты `agg` не адресуются как в pandas (`KeyError`); `quantile` возвращает транспонированный фрейм.
- ИСПРАВЛЕНО: `std(ddof=1)` на одной строке: `float(None)`.
- ИСПРАВЛЕНО: Сеттер индекса передаёт список в `set_index` как имена колонок (`KeyError`) — `reset_index(drop=True)`.
- ИСПРАВЛЕНО: `add_column` присваивает DataFrame колонке.
- ИСПРАВЛЕНО: `__and__`, `__or__`, `__pos__` работают на pyspark.pandas DataFrame и не работают.
- ИСПРАВЛЕНО: `SparkBisaExtesion.prepare_data` вызывает `result_to_dataset` без обязательного `roles` (`TypeError`).
- `SparkKSTestExtension` с `nan_policy="omit"` не отбрасывает NaN (результат 1.0 / 0.0, в pandas иначе).
- ИСПРАВЛЕНО: `SparkFaissExtension` определяет `__enter__`, но не `__exit__`.
- `MultiTest._calc_spark` теряет составной строковый индекс при `to_backend` — поправка ничего не делает.
- ИСПРАВЛЕНО: `StatsUTest._execute_spark` → `ImportError`: `StatsUTestExtension` не существует (строки ~922–962 не покрыты).

## 4. Статистика / comparators / extensions

- ИСПРАВЛЕНО (числа изменились, см. «Изменения численных результатов»): `GroupChi2TestExtension.calc` передаёт `_form_results(p_value, statistic, ...)` в перепутанном порядке (pandas и Spark).
- `StatsKSTest` / `StatsUTest` на pandas: код ставит `delegate._id`, а результаты лежат под `GroupKSTest┴┴y` / `GroupUTest┴┴y` — поиск по id Stats-исполнителя не находит.
- ИСПРАВЛЕНО: `PandasLstsqExtension.calc` через `create_empty().fillna()` даёт pandas FutureWarning (3 теста lstsq).
- ИСПРАВЛЕНО (добавлен и `calc`-диспетчер): `MultitestQuantile._calc_pandas` не тестируется: вызов `groupby(fields_list=...)` падает дальше, т.к. в группах остаётся строковая колонка группы.
- ИСПРАВЛЕНО: `MahalanobisDistance` — недостижимая ветка `test_data=None` удалена; при одной группе `_execute_inner_function` явно бросает `ValueError("test_data is needed ...")`, а `calc` — `NotSuitableFieldError`.
- ИСПРАВЛЕНО локально в `Float32Caster` (общий `search_columns` не менялся — у него 46 вызывающих): `Float32Caster(target_roles=InfoRole(int))` больше не выбирает float-колонку Info. `Dataset.search_columns` по-прежнему сопоставляет роли только по классу.

## 5. ML / matching / faiss

- ИСПРАВЛЕНО: `FaissExtension.fit` (публичный) передаёт `target_data=` в `MLExtension.calc`, такого параметра нет (`TypeError`); то же на уровне `FaissNearestNeighbors.fit`.
- ИСПРАВЛЕНО (`Dataset.reindex` добавлен): `FaissNearestNeighbors.execute` в режимах one-sided и `test_pairs` вызывает `Dataset.reindex` (есть только у `SmallDataset`) — `AttributeError`, `hypex/ml/faiss.py:~332`.
- ИСПРАВЛЕНО: `FaissNearestNeighbors._execute_inner_function` при `test_pairs=True, two_sides=True`: запись "test" повторяет запрос control вместо сопоставления test→control.
- ИСПРАВЛЕНО (текст предупреждения и docstring; поведение прежнее): NaN в результате faiss вызывают `PairsNotFoundError`, а не заменяются заглушками.
- ИСПРАВЛЕНО: `MLExecutor._execute_inner_function` с `target_field` вызывает `_data.drop(target_field)` без `axis=1` — pandas пытается удалить строку.
- `MLExecutor.execute` передаёт `target_fields=`, а `calc` принимает `target_field` — таргет не доходит до `_inner_function` (теста нет).

## 6. UI / reporters

- ИСПРАВЛЕНО: `ABOutput.variance_reduction_report` вызывает `ABTestReporter.report_variance_reductions`, которого больше нет.
- `Matching().execute(...)` на Spark и при `compute_indexes=False` по умолчанию возвращает пустой `indexes` (`len == 0`), тесты ожидают строки на каждую запись (`test_matching_output_structure_on_spark`, `test_indexes_cover_every_row`).
- `Matching(group_match=True)`: `ValueError: No group keys passed!` на обычных ролях.
- ИСПРАВЛЕНО попутно: `OnRoleExperiment.execute` не восстанавливал `self.executors` при исключении (общий `HOMOGENEITY_TEST` портился после неудачного запуска) — теперь `try/finally`.

## 7. Прочее наблюдаемое

- `strict_abc` вычисляет аннотации «как написано», поэтому в файлах с `from __future__ import annotations` проверки возвращаемых типов не работают (тесты `test_strict_abc.py` намеренно без этого импорта).
- `faiss` в Spark: партиционные функции исполняются на executor'ах, `coverage` их не видит (`extensions/faiss.py` ~72%).

## Тесты, где поведение изменилось и тест подогнан под новый контракт

`CachingIndex.get(reference, storage)` (nprobe вынесен из кэша); Welch — t-test по умолчанию; в AB «significant = OK»; AA-резюме — одна строка на признак; shell копирует `ExperimentData`. Подробности: `git log` ветки, коммиты шага «fix failing tests».

## Изменения численных результатов (записано при исправлении)

Эти исправления меняют выдаваемые числа; прежние результаты были неверными.

- **Chi2 (`GroupChi2TestExtension.calc`, pandas и Spark).** Раньше в `p-value` попадала статистика хи-квадрат, а в `statistic` — p-value (и флаг `pass` считался по статистике). Теперь значения совпадают с `scipy.stats.chi2_contingency`. Это затрагивает все места, где используется `GroupChi2Test` / `Chi2Test`: A/A-тест (`mean p-value`, `pass` и выбор лучшего разбиения через композитный балл с весом 2 для Chi2), гомогенность, A/B(n)-тест с `additional_tests=["chi2-test"]` и проверки качества Matching. Сохранённые ранее результаты с chi2 пересчитать.

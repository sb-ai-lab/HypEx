# Известные ошибки библиотеки (найдены при покрытии тестами)

Источник: ветка `dev/test-coverage-80`, замер 2026-10-04. Для каждой ошибки в `tests/` стоит `xfail(strict=True)` с `reason=`: когда ошибку исправят, тест выдаст XPASS и упадёт — тогда снять маркер. Поиск: `grep -rn "xfail" tests`.

## 1. Сквозные причины (ломают много тестов)

| Ошибка | Где | Эффект |
|---|---|---|
| ИСПРАВЛЕНО: `PandasDataset.count_groups` делает `int(df[cols].nunique())` | `hypex/dataset/backends/pandas_backend.py` | FutureWarning (в pandas), `TypeError` при нескольких колонках групп и на Spark; ломает любой `list(groupby)`: `GroupedDataset`, `GroupOperator.calc`, `GroupExperiment`, `MahalanobisDistance`, split-modes, calculators, `MLExecutor`, `MinSampleSize` |
| ИСПРАВЛЕНО: `HomogeneityTest()` создаёт deprecated `HomoDatasetReporter` | `hypex/ui/homo.py` | DeprecationWarning → ошибка из-за `filterwarnings` в `pyproject.toml`; фасад нельзя вызвать в тестах |
| ИСПРАВЛЕНО (deprecated-репортёры восстановлены как обратно-совместимые алиасы с `DeprecationWarning`, фасады используют `MatchingReporter`/`MatchingQualityReporter`/`HomogeneityReporter`): `Matching()` / `MatchingOutput()` создают deprecated `MatchingDictReporter` / `MatchingDatasetReporter` / `MatchingQualityDatasetReporter` | `hypex/matching.py`, `hypex/ui/matching.py` | то же; старые xfail-причины Matching (соседи из противоположной группы, ближайший по ковариате, `_match_pandas`, `group_match`) нельзя перепроверить, пока репортёры deprecated |

## 2. Dataset / adapter

- ИСПРАВЛЕНО: `DatasetAdapter.list_to_dataset` / `ndarray_to_dataset` с `small=False` вызывают несуществующий `Dataset.to_dataset()` (теперь `small` для list/ndarray/scalar задокументирован как игнорируемый: всегда `Dataset`).
- ИСПРАВЛЕНО (проверки типов фреймов убраны из `DatasetBase.get`: backend `get` теперь возвращает фрейм — одна колонка как однокалоночный фрейм — или сам `default`): `Dataset.get(key)` пересоздаёт Dataset со всеми исходными ролями → `RoleColumnError`.
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
- ИСПРАВЛЕНО (числа изменились при наличии NaN, см. «Изменения численных результатов»): `SparkKSTestExtension` (путь `GroupKSTest` на Spark) с `nan_policy="omit"` не отбрасывал NaN (результат 1.0 / 0.0, в pandas иначе). Пользовательский путь `StatsKSTest`/`StatsUTest` на Spark (гистограммы в `StatsKSTestExtension`) тоже игнорировал NaN неверно (NaN в max давал p=1.0) — исправлено отдельно: NaN трактуется как null в bounds и unpivot, без новых Spark-задач.
- ИСПРАВЛЕНО: `SparkFaissExtension` определяет `__enter__`, но не `__exit__`.
- ПРИЧИНА УТОЧНЕНА (`MultiTest._calc_spark` удалён: `MultiTest.calc` одним путём конвертирует данные в pandas через `to_backend`): тест падал не из-за `to_backend`, а потому что `Dataset(data=pd.DataFrame, backend=spark)` создаёт Spark-фрейм через `createDataFrame` и теряет pandas-индекс (составные id p-value пропадают ещё до `MultiTest`). Тест теперь строит Spark-датасет через `ps.from_pandas` (индекс сохраняется). В пайплайне `ABAnalyzer` p-value всегда собираются в `SmallDataset` (pandas), реальные A/B-результаты на Spark это не затрагивало.
- ОТКРЫТО (вне плана): `SparkNavigation.__init__` для `pd.DataFrame` использует `createDataFrame(data)` и теряет индекс pandas.
- ИСПРАВЛЕНО: `StatsUTest._execute_spark` → `ImportError`: `StatsUTestExtension` не существует (переиспользован `StatsKSTestExtension`). `statistic` — U1 базовой группы, как в scipy/pandas (раньше в логике было `min(U1, U2)`). ВАЖНО: KS/U на Spark — приближение по гистограмме (`n_bins` равных бинов на [min, max], нормальное приближение с поправкой на ties): на нормальных данных близко к scipy (~1%), но один экстремальный выброс растягивает бины и p-value может быть сильно неверным.

## 4. Статистика / comparators / extensions

- ИСПРАВЛЕНО (числа изменились, см. «Изменения численных результатов»): `GroupChi2TestExtension.calc` передаёт `_form_results(p_value, statistic, ...)` в перепутанном порядке (pandas и Spark).
- ИСПРАВЛЕНО (вариант a: таблица кладётся под id самого `Stats*`-исполнителя, как на Spark; нигде в фасадах не используется, имена в отчётах не меняются): `StatsKSTest` / `StatsUTest` на pandas (теперь Spark-only: `execute` на не-Spark данных бросает `TypeError`, на pandas `KSTest`/`UTest` резолвятся в `Group*`; pandas-fallback удалён): код ставил `delegate._id`, а результаты лежат под `GroupKSTest┴┴y` / `GroupUTest┴┴y` — поиск по id Stats-исполнителя не находит.
- ИСПРАВЛЕНО: `PandasLstsqExtension.calc` через `create_empty().fillna()` даёт pandas FutureWarning (3 теста lstsq).
- ИСПРАВЛЕНО на pandas (`PandasMultitestQuantile`, выбирается через `backend_factory` в `ABAnalyzer`; `SparkMultitestQuantile` явно бросает `NotImplementedError` — не поддержан; `calc` базового класса бросает `NotImplementedError` на любом бэкенде; `accepted hypothesis` теперь int): `MultitestQuantile._calc_pandas` не тестируется: вызов `groupby(fields_list=...)` падает дальше, т.к. в группах остаётся строковая колонка группы.
- ИСПРАВЛЕНО: `MahalanobisDistance` — недостижимая ветка `test_data=None` удалена; при одной группе `_execute_inner_function` передаёт `test_data=None`, и единственная проверка `_check_test_data` в `_inner_function` бросает `ValueError("test_data is needed ...")`, а `calc` — `NotSuitableFieldError`.
- ИСПРАВЛЕНО локально в `Float32Caster` (общий `search_columns` не менялся — у него 46 вызывающих): `Float32Caster(target_roles=InfoRole(int))` больше не выбирает float-колонку Info. `Dataset.search_columns` по-прежнему сопоставляет роли только по классу.

## 5. ML / matching / faiss

- ИСПРАВЛЕНО: `FaissExtension.fit` (публичный) передаёт `target_data=` в `MLExtension.calc`, такого параметра нет (`TypeError`); то же на уровне `FaissNearestNeighbors.fit`.
- ИСПРАВЛЕНО (`DatasetBase.reindex` добавлен без импорта `dataset` в `abstract` — возвращает класс `self`; `SmallDataset.reindex` возвращает `Dataset`): `FaissNearestNeighbors.execute` в режимах one-sided и `test_pairs` вызывает `Dataset.reindex` (есть только у `SmallDataset`) — `AttributeError`, `hypex/ml/faiss.py:~332`.
- ИСПРАВЛЕНО: `FaissNearestNeighbors._execute_inner_function` при `test_pairs=True, two_sides=True`: запись "test" повторяет запрос control вместо сопоставления test→control.
- ИСПРАВЛЕНО (текст предупреждения и docstring; поведение прежнее): NaN в результате faiss вызывают `PairsNotFoundError`, а не заменяются заглушками.
- ИСПРАВЛЕНО: `MLExecutor._execute_inner_function` с `target_field` вызывает `_data.drop(target_field)` без `axis=1` — pandas пытается удалить строку.
- ИСПРАВЛЕНО: публичный `FaissExtension.predict` уходил в `RecursionError` (`super().calc` -> `MLExtension.calc` -> `self.predict`); теперь `predict` = `self.calc(data=X, test_data=X, mode="predict")` с `AdditionalMatchingRole`. `PandasFaissExtension(faiss_mode="fast").fit(X)` при >1000 строк падал с `TypeError` (`len(None)`) — исправлено.
- ИСПРАВЛЕНО (числа изменились, см. «Изменения численных результатов»): pandas FAISS `_predict` заменял на -1 совпадения с метками контрольной группы > `len(control)+len(test)` (граница сравнивала метку строки с числом строк); граница удалена, как и на Spark.
- `FaissNearestNeighbors.predict` (executor) не хранит индекс (stateless) и теперь бросает `NotImplementedError` с указанием `fit(X).predict(Y)` (раньше `RecursionError`).
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

- **Chi2 (`GroupChi2TestExtension.calc`, pandas и Spark).** Раньше в `p-value` попадала статистика хи-квадрат, а в `statistic` — p-value (и флаг `pass` считался по статистике). Теперь значения совпадают с `scipy.stats.chi2_contingency`. На Spark фасад `StatsChi2Test` и раньше считал верно; изменились только `GroupChi2Test` / `SparkChi2TestExtension`. Это затрагивает все места, где используется `GroupChi2Test` / `Chi2Test`: A/A-тест (`mean p-value`, `pass` и выбор лучшего разбиения через композитный балл с весом 2 для Chi2), гомогенность, A/B(n)-тест с `additional_tests=["chi2-test"]` и проверки качества Matching. Сохранённые ранее результаты с chi2 пересчитать.
- **Spark KS (`SparkKSTestExtension.calc`, `nan_policy="omit"` по умолчанию).** Раньше NaN/null отравляли min/max и бакеты, и результат при наличии пропусков был `p-value=1.0, statistic=0.0`-подобным. Теперь строки с NaN/null отбрасываются в обеих выборках, как в pandas `ks_2samp(nan_policy="omit")`. Без пропусков числа не изменились.
- **Spark U (`StatsUTest`).** Поле `statistic` теперь U1 базовой группы (совпадает с scipy и pandas `GroupUTest`); раньше Spark выдавал `min(U1, U2)`. p-value не менялось.
- **Spark `StatsKSTest`/`StatsUTest` с NaN.** Раньше один NaN в колонке давал `p-value=1.0, statistic=0.0`; теперь NaN игнорируются.
- **Pandas FAISS `_predict` (non-default индексы).** Раньше совпадение, чья метка контрольной строки превышала `len(control)+len(test)`, превращалось в `-1` («не найдено»); теперь возвращается настоящая метка ближайшего соседа. Числа меняются только для датасетов, у которых индекс не `0..N-1`.

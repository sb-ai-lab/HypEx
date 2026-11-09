# Техническое описание веток `fix/issue-*`

Документ описывает 26 веток, вошедших в итоговую интеграцию: исходные 22, issue 13 и добавленные после интеграционной проверки 27–29. Issue 12 и 19 уже были решены ранее, а issue 23 решена лишь частично, поэтому они в интеграцию не входили.

Для каждой ветки указаны затронутые модули, исходная ошибка реализации, конкретное изменение и его последствия.

## `fix/issue-01-groupscomparator-natural-order`

- **Модули:** `hypex/comparators/abstract.py`; регрессионные тесты в `tests/unit/comparators/test_hypothesis_testing.py`.
- **Что ломалось:** в `GroupsComparator._grouping_data_split` группы сортировались через `key=str`. Поэтому порядок `1, 10, 2` считался нормальным, baseline выбирался неправильно, а знак разности и t-статистики мог инвертироваться.
- **Что изменено:** выбор baseline и последовательность сравнений переведены на естественную сортировку значений групп; NaN обрабатывается отдельно и остаётся в конце.
- **Пояснение:** порядок `1, 2, 10` больше не зависит от строкового представления. Контроль и сравниваемые группы определяются детерминированно, а результаты сохраняют правильный знак.

## `fix/issue-02-statscomparator-result-order`

- **Модули:** `hypex/comparators/abstract.py`; тесты в `tests/unit/comparators/test_stats_tests_execute.py`.
- **Что ломалось:** `StatsComparator` получал словарь агрегированных статистик в одном порядке, а метки итоговых строк формировал через отдельную строковую сортировку. Значение сравнения одной группы могло быть записано под индексом другой.
- **Что изменено:** порядок обхода `group_col_stats` и порядок индексов результата унифицирован; больше нет независимого переупорядочивания меток после вычислений.
- **Пояснение:** p-value, статистика и effect size остаются связанными с исходным compared-group. Это было особенно заметно на группах `2` и `10`.

## `fix/issue-03-spark-chi2-column-order`

- **Модули:** `hypex/extensions/stats_hypothesis_testing.py`, класс `StatsChi2TestExtension`; тесты в `tests/dataset_spark_backend/test_group_order.py`.
- **Что ломалось:** Spark-агрегация возвращала строки в произвольном порядке, а словарь результатов затем сопоставлялся с `target_cols` позиционно. P-value одной целевой колонки мог оказаться под именем другой.
- **Что изменено:** итоговый словарь формируется по явному порядку `target_cols`, а не по порядку строк, полученных после `collect()`.
- **Пояснение:** результат chi-square адресуется по имени целевой переменной и больше не зависит от распределённого плана выполнения Spark.

## `fix/issue-04-abanalyzer-group-labels`

- **Модули:** `hypex/analyzers/ab.py`, `hypex/comparators/abstract.py`; тесты в `tests/scenario/test_ab_scenario.py` и `tests/unit/test_analyzers.py`.
- **Что ломалось:** `ABAnalyzer._build_row_index` использовал `zip` между group labels в естественном порядке и строками, построенными в строковом порядке. На группах `1, 2, 10` средние и p-value присваивались неверным строкам.
- **Что изменено:** построение индекса и извлечение результатов используют один canonical group order. Изменения в comparator и анализаторе согласованы, чтобы не исправить порядок только на одном уровне.
- **Пояснение:** значения группы `10` больше не попадают в строку группы `2`. Дополнительно проверены многозначные метки и соответствие средних исходным данным.

## `fix/issue-05-multitest-quantile-sample-size`

- **Модули:** `hypex/extensions/statsmodels.py`, класс `MultitestQuantile`; тесты в `tests/unit/test_extensions_math.py`.
- **Что ломалось:** в t-подобной статистике использовался `len(data)` — общее количество наблюдений во всех группах. Модельные квантили при этом соответствовали стандартизации по размерам отдельных групп, поэтому статистика завышалась примерно в `sqrt(k)` раз.
- **Что изменено:** размеры групп вычисляются отдельно как `len(sample[1])`; знаменатель формируется как `sqrt(var_j / n_j + var_i / n_i)`.
- **Пояснение:** теперь наблюдаемая статистика и Monte-Carlo нулевое распределение находятся в одной шкале. Это снижает ложные «победы» и корректно учитывает неравные размеры групп.

## `fix/issue-06-cupac-prediction-index`

- **Модули:** `hypex/extensions/cupac.py`, pandas-реализация `_predict_pandas`; тесты в `tests/unit/test_ml_cupac.py`.
- **Что ломалось:** результат `model.predict` оборачивался в Dataset с новым `RangeIndex`. Последующая index-aligned арифметика с исходной целевой колонкой сопоставляла прогнозы не тем наблюдениям.
- **Что изменено:** при создании prediction frame используется индекс входной `X_df`, а не автоматическая последовательность `0..n-1`.
- **Пояснение:** корректировка CUPAC выполняется по тем же строкам, для которых получены прогнозы. Исправлен рост дисперсии на отфильтрованных, перемешанных и нестандартно проиндексированных данных.

## `fix/issue-07-cupac-mixed-lags`

- **Модули:** `hypex/ml/cupac.py`, логика сортировки `searched_lags`; тесты в `tests/unit/test_ml_cupac.py`.
- **Что ломалось:** `sorted(..., key=lambda x: x[1])` пытался сравнить `None` у обычного `FeatureRole` с целочисленным лагом у `PreTargetRole`, что вызывало `TypeError`.
- **Что изменено:** ключ сортировки стал None-aware: текущий признак без лага упорядочивается совместимо с лаговыми признаками, а последующая обработка `None/0` сохраняется.
- **Пояснение:** в одной конфигурации CUPAC можно использовать текущие и лаговые переменные без искусственного заполнения отсутствующих лагов.

## `fix/issue-08-pandas-chi2-category-union`

- **Модули:** `hypex/extensions/scipy_stats.py`, `PandasChi2TestExtension.matrix_preparation`; тесты в `tests/unit/comparators/test_hypothesis_testing.py`.
- **Что ломалось:** `merge(on=col_name)` использовал `how="inner"` по умолчанию. Категория, присутствующая только в control или только в test, исчезала из contingency table, а последующий `fillna(0)` уже не мог её восстановить.
- **Что изменено:** объединение выполняется с `how="outer"`, после чего отсутствующие частоты заполняются нулями.
- **Пояснение:** таблица содержит объединение категорий обеих выборок. Это меняет статистику и p-value в случаях с редкими или односторонними категориями и устраняет неверные выводы о независимости.

## `fix/issue-09-abanalyzer-group-chi2`

- **Модули:** `hypex/analyzers/ab.py`; сценарные тесты в `tests/scenario/test_ab_scenario.py`.
- **Что ломалось:** набор классов, из которых `ABAnalyzer` собирал статистики, содержал `StatsChi2Test` для Spark, но не `GroupChi2Test` для pandas. Таблица теста существовала в `analysis_tables`, однако анализатор её игнорировал.
- **Что изменено:** `GroupChi2Test` включён в сбор результатов AB-анализа наравне с другими поддерживаемыми тестами.
- **Пояснение:** pandas chi-square теперь попадает в итоговую таблицу `ABAnalyzer` и может участвовать в последующей multitest-коррекции.

## `fix/issue-10-ab-multitest-categorical`

- **Модули:** `hypex/analyzers/ab.py`, `hypex/extensions/statsmodels.py`; тесты в `tests/scenario/test_ab_scenario.py`.
- **Что ломалось:** решение о применении поправки зависело только от количества числовых TargetRole-колонок. Категориальные p-values не попадали в коррекцию, а объединённый массив смешанных типов мог привести к `UFuncTypeError` в statsmodels.
- **Что изменено:** для стандартных методов размер семьи определяется по фактически собранным p-values; результаты нормализуются по имени теста и корректируются внутри соответствующего семейства. Quantile-режим по-прежнему использует числовые цели, как и задумано.
- **Пояснение:** смешанные numeric/categorical наборы проходят без dtype-ошибок, а chi-square результаты получают Holm/Bonferroni/FDR-коррекцию отдельно от t-test.

## `fix/issue-11-ab-enum-multitest`

- **Модули:** `hypex/ab.py`, нормализация аргумента `multitest_method`; тесты в `tests/scenario/test_ab_scenario.py`.
- **Что ломалось:** проверка `multitest_method in ABNTestMethodsEnum._value2member_map_` работала только для строк. Enum-значение не находилось в map и попадало в ветку неявного fallback на `holm`.
- **Что изменено:** сначала проверяется `isinstance(multitest_method, ABNTestMethodsEnum)`; строковые значения преобразуются отдельно.
- **Пояснение:** `ABNTestMethodsEnum.bonferroni` теперь действительно означает Bonferroni, а не Holm. Поведение enum и строкового имени согласовано.

## `fix/issue-13-spark-faiss-prefit-batches`

- **Модули:** `hypex/extensions/faiss.py`, `SparkFaissExtension._prefit`; тесты в `tests/unit/test_ml_faiss.py`.
- **Что ломалось:** остатки мелких партиций накапливались через `np.append(cache, np_batch)`, который без `axis` выравнивал матрицу в 1D. `MiniBatchKMeans.partial_fit` затем получал одномерный массив вместо `(n_samples, n_features)`.
- **Что изменено:** `cache` стал списком массивов; батчи добавляются через `cache.append`, а перед обучением объединяются через `np.concatenate(cache, axis=0)`. Мелкие остатки накапливаются до минимально допустимого размера `k`.
- **Пояснение:** prefit устойчив к большому числу мелких Spark-партиций и не зависит от количества ядер. Добавлена проверка сохранения размерностей и centroid-формы.

## `fix/issue-14-spark-value-counts`

- **Модули:** `hypex/dataset/backends/spark_backend.py`, `SparkDataset.value_counts`; тесты в `tests/unit/dataset/test_dataset_agg.py` и `tests/unit/transformers/test_filters.py`.
- **Что ломалось:** реализация фактически выполняла подсчёт только по первой колонке. При `normalize=True` результат назывался `count`, тогда как абстрактный слой искал `proportion`, что приводило к ошибке ролей.
- **Что изменено:** Spark DataFrame группируется по всем исходным колонкам через `groupBy`, подсчёт идёт по полным комбинациям значений. Для нормализации формируется отдельная `proportion` на основе общего количества строк.
- **Пояснение:** многоколоночный результат соответствует pandas `DataFrame.value_counts`, а `normalize=True` корректно проходит обёртку Dataset и назначение ролей.

## `fix/issue-15-spark-scalar-loc`

- **Модули:** `hypex/dataset/backends/spark_backend.py`, `SparkNavigation.loc`; тесты в `tests/unit/dataset/test_dataset_indexing.py`.
- **Что ломалось:** `DatasetBase.Locker` ожидал DataFrame и читал `.columns`, но скалярный `.loc` возвращал Series. Первая версия исправления учитывала только `pyspark.pandas.Series`, хотя в проверенной версии PySpark возвращался локальный `pandas.Series`.
- **Что изменено:** результат преобразуется через `.to_frame()` для обоих типов — `pd.Series` и `ps.Series`.
- **Пояснение:** форма результата соответствует pandas-контракту `(n_columns, 1)`. Локальная pandas-строка не отправляется обратно в Spark через Arrow, где смешанные int/float/str вызывали бы отдельную ошибку.

## `fix/issue-16-checkhash-groups`

- **Модули:** `hypex/dataset/experiment_data.py`, `ExperimentData.check_hash`; тесты в `tests/unit/dataset/test_experiment_data.py`.
- **Что ломалось:** `ExperimentDataEnum.groups` входил в поддерживаемые пространства, но отдельной ветки обработки не имел. Универсальный fallback повторно вызывал `check_hash` с теми же аргументами до `RecursionError`.
- **Что изменено:** для `groups` добавлена прямая проверка присутствия `exec_id_str` в `self.groups`; это пространство исключено из рекурсивного обхода fallback.
- **Пояснение:** проверка групп выполняется за постоянное время и соответствует фактической структуре хранения `ExperimentData`.

## `fix/issue-17-experimentdata-copy`

- **Модули:** `hypex/dataset/experiment_data.py`, `ExperimentData.copy`; тесты в `tests/unit/dataset/test_experiment_data.py`.
- **Что ломалось:** при передаче заменяющего `data` новый объект получал `result._data = data`. Копия и вызывающая сторона использовали один Dataset, поэтому `add_column` через копию изменял исходный объект.
- **Что изменено:** заменяющий Dataset копируется перед сохранением в новый `ExperimentData`, вместо простого присваивания ссылки.
- **Пояснение:** восстановлена независимость копии и заявленная deep-copy семантика. Остальные внутренние пространства `ExperimentData` продолжают копироваться по тем же правилам.

## `fix/issue-18-role-copy-metadata`

- **Модули:** `hypex/dataset/roles.py`, `ABCRole.__copy__` и производные преобразования ролей; тесты в `tests/unit/dataset/test_roles.py`.
- **Что ломалось:** копия создавалась через базовый конструктор без `parent`, `lag` и `cofounders`. Метаданные исчезали не только при `copy.copy`, но и при `astype()`/`asadditional()`.
- **Что изменено:** при копировании переносятся параметры исходной роли и связанные с ней метаданные, а не только её тип.
- **Пояснение:** `PreTargetRole` и производные dummy-роли сохраняют lineage. Это необходимо для корректного отбора признаков и лагов в CUPAC после преобразований Dataset.

## `fix/issue-20-pandas-remove-index-column`

- **Модули:** `hypex/dataset/backends/pandas_backend.py`, `PandasDataset.remove_index_col`; тесты в `tests/unit/dataset/test_dataset_indexing.py`.
- **Что ломалось:** `self.data.drop(index_col_name)` без `columns` или `axis=1` интерпретировал имя служебной колонки как метку строки.
- **Что изменено:** вызов заменён на `drop(columns=[index_col_name])`.
- **Пояснение:** round-trip `add_index_col()` → `remove_index_col()` возвращает DataFrame с теми же строками и колонками, без `KeyError` и потери данных.

## `fix/issue-21-additional-fields-cache`

- **Модули:** `hypex/dataset/experiment_data.py`, `ExperimentData._set_additional_fields`; тесты в `tests/unit/dataset/test_experiment_data.py`.
- **Что ломалось:** in-place ветки `add_column` возвращали `new_data is self._data`. Код вызывал `persist()` для нового состояния, а затем `unpersist()` по той же ссылке, сразу удаляя созданный Spark-кэш.
- **Что изменено:** выгрузка старого backend выполняется только при реальной замене объекта данных; кэш текущего `new_data` не снимается.
- **Пояснение:** `is_persisted` больше не меняется неожиданно с `True` на `False`. Это исправление управления ресурсами и повторными вычислениями, а не формул статистических методов.

## `fix/issue-22-naming-difference-feature`

- **Модули:** `hypex/utils/naming.py`, `_parse_metric_col`; тесты в `tests/unit/utils/test_naming.py`.
- **Что ломалось:** имя метрики `difference` находилось в начале имени колонки `difference TTest pass test_1`. Парсер принимал пользовательский признак за служебную метрику и возвращал пустое feature name.
- **Что изменено:** служебные метрики ищутся по окончанию соответствующего сегмента имени, а не по совпадению в произвольной/начальной позиции.
- **Пояснение:** признак `difference` корректно попадает в `AAScoreAnalyzer`, расчёт лучшего split и отчёты. Разбор составных executor id и метрик сохранён.

## `fix/issue-24-keyless-index-merge`

- **Модули:** `hypex/dataset/abstract.py`, `hypex/dataset/backends/abstract.py`, `hypex/dataset/backends/pandas_backend.py`, `hypex/dataset/backends/spark_backend.py`; тесты в `tests/unit/dataset/test_dataset_indexing.py`.
- **Что ломалось:** без `on` и других ключей pandas не знал, по каким колонкам объединять данные. После первого исправления условие `not all([on, left_on, right_on])` дополнительно включало index-join даже при явном `on=`, вызывая конфликт аргументов и ошибки chi-square.
- **Что изменено:** API использует `None` как отсутствие явного выбора. Только когда все `on/left_on/right_on/left_index/right_index` равны `None`, выставляются оба index-флага. Во всех остальных случаях незаданные флаги приводятся к `False`.
- **Пояснение:** keyless merge работает по индексу, а keyed merge — по указанным колонкам. Поведение согласовано на pandas и Spark и покрыто двумя независимыми regression-тестами.

## `fix/issue-25-spark-agg-non-numeric`

- **Модули:** `hypex/dataset/backends/spark_backend.py`, путь агрегации и преобразования результатов; тесты в `tests/unit/dataset/test_dataset_agg.py`.
- **Что ломалось:** Spark-агрегация молча ограничивала вход числовыми колонками. Пользователь получал неполный результат без предупреждения о том, что часть запрошенных данных была проигнорирована.
- **Что изменено:** добавлена проверка типов входных колонок и явное отклонение неподдерживаемых нечисловых агрегаций.
- **Пояснение:** неподдерживаемая операция теперь завершается понятной ошибкой вместо молчаливого удаления колонок. Это делает ограничения Spark-агрегаций видимыми и предотвращает неверную интерпретацию результата.

## `fix/issue-26-faiss-predict-query`

- **Модули:** `hypex/extensions/faiss.py`, predict-режим FAISS; тесты в `tests/unit/test_ml_faiss.py`, отметка в `tests/KNOWN_LIBRARY_BUGS.md`.
- **Что ломалось:** в `mode="predict"` матрица запросов строилась из `data`, то есть из обучающего набора, а не из `test_data`. При равной длине это давало позиционные self-match, при разной — ошибки формы или неверные соответствия.
- **Что изменено:** для поиска используются векторы `test_data`; обучающая матрица остаётся только источником индекса/соседей.
- **Пояснение:** последовательность `fit(X).predict(Y)` теперь эквивалентна поиску X→Y по смыслу запроса и корректно работает при разных размерах X и Y. Это отдельный дефект от проблемы prefit-батчей в issue 13.

## `fix/issue-27-pandas-chi2-raw-counts`

- **Модули:** `hypex/extensions/scipy_stats.py`, `PandasChi2TestExtension.matrix_preparation`; тесты в `tests/unit/test_extensions_scipy_stats.py` и `tests/unit/comparators/test_hypothesis_testing.py`; `tests/KNOWN_LIBRARY_BUGS.md`.
- **Что ломалось:** частоты control умножались на `1 - proportion`, а test — на `proportion`. Для равных групп общий размер contingency table уменьшался примерно вдвое, статистика и p-value рассчитывались не по исходным наблюдениям.
- **Что изменено:** удалены `proportion` и оба множителя; `data_vc` и `other_vc` используют фактические `count`. Ожидания тестов изменены на raw counts, а strict-xfail SciPy-тест снят.
- **Пояснение:** pandas и Spark теперь используют одинаковую частотную модель. После исправления AA-категория дала 11/200 отклонений (5,5%), а не 0/200, а AB p-value совпал со SciPy.

## `fix/issue-28-spark-aa-sample-size-null-groups`

- **Модули:** `hypex/comparators/abstract.py` (`StatsComparator._compute_stats`), `hypex/comparators/stats_hypothesis_testing.py` (chi-square и KS/U input preparation); сценарный тест в `tests/scenario/test_aa_scenario.py`.
- **Что ломалось:** `AASplitter` оставлял невыбранные строки в общем Dataset с null additional-treatment label. Spark groupBy включал null как отдельную группу, после чего `sorted(group_col_stats)` пытался сравнить `None` со строками.
- **Что изменено:** перед comparator-агрегациями вызывается `dropna(subset=group_cols)` для обычных статистик и chi-square; KS/U дополнительно фильтруют `subset` по `group_col` перед гистограммами.
- **Пояснение:** невыбранные строки не участвуют в статистических сравнениях, как и в pandas. Низкоуровневые `Stats*Extension` не изменены, поэтому их явные NaN-ordering тесты продолжают проходить.

## `fix/issue-29-matching-scalar-quality-tests`

- **Модули:** `hypex/matching.py`, `Matching._make_experiment`; тесты в `tests/dataset_spark_backend/test_matching_spark.py`.
- **Что ломалось:** ветка, отличная от `quality_tests="auto"`, выполняла list comprehension непосредственно по `quality_tests`. Для строки это означало поиск `test_mapping["t"]` и немедленный `KeyError`.
- **Что изменено:** перед построением `OnRoleExperiment` добавлен `selected_tests = [quality_tests]`, если аргумент является строкой; списки передаются без изменений.
- **Пояснение:** `"t-test"` и `["t-test"]` строят одинаковый pipeline и дают одинаковый summary на pandas и Spark. Обработка `auto` и явных списков не затронута.

---

## Итоговая проверка

- Полный набор: **1767 passed, 18 skipped, 78 xfailed, 0 failed**.
- Синтетическая матрица AA/AB/Homogeneity/Matching: **28/28 сценариев passed** на pandas и Spark.
- Ruff: проверка и форматирование проходят без ошибок.

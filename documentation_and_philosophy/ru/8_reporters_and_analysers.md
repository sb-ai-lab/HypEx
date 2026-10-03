# Репортёры, аналитики и извлечение результатов

---

## 1. Архитектура извлечения результатов

Путь данных от сырых `analysis_tables` до финального `Output` объекта проходит через **три слоя**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ЭКСПЕРИМЕНТ                                          │
│  Executor'ы пишут результаты в ExperimentData.analysis_tables               │
│  Ключ: "ClassName┆params_hash┆key"                                         │
│  Значение: SmallDataset с колонками (p-value, statistic, pass, ...)         │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     АНАЛИТИКИ (Analyzers)                                   │
│  Агрегируют, трансформируют, выбирают лучшее                                │
│  OneAAStatAnalyzer → AAScoreAnalyzer → ABAnalyzer → MatchingAnalyzer        │
│  Результат: новые записи в analysis_tables / variables                      │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     РЕПОРТЁРЫ (Reporters)                                   │
│  Извлекают данные из analysis_tables, форматируют в таблицы                 │
│  extract_tests() → extract_group_difference() → extract_group_sizes()       │
│  Результат: dict[str, Any] или Dataset                                      │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     OUTPUT (ui/)                                            │
│  Собирает результаты репортёров в атрибуты объекта                          │
│  AAOutput / ABOutput / MatchingOutput / HomoOutput                          │
│  Результат: готовый объект с .summary, .multitest, .sizes, ...             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Иерархия репортёров

### 2.1. Базовые классы

```
Reporter (ABC)
├── DictReporter (ABC)
│   └── TestDictReporter (ABC)
│       ├── AATestReporter → DatasetReporter
│       ├── ABTestReporter → DatasetReporter
│       ├── HomogeneityReporter → DatasetReporter
│       ├── MatchingReporter → DatasetReporter
│       └── MatchingQualityReporter → DatasetReporter
└── DatasetReporter
    ├── AAPassedReporter
    ├── AABestSplitReporter
    ├── CupedReporter
    └── CupacReporter
```

### 2.2. `Reporter` — абстрактный контракт

```python
class Reporter(ABC):
    @abstractmethod
    def report(self, data: ExperimentData) -> Any:
        raise AbstractMethodError
```

Единственный метод — `report()`, принимающий `ExperimentData` и возвращающий результат в произвольном формате.

### 2.3. `DictReporter` — вывод в виде словаря

```python
class DictReporter(Reporter, ABC):
    def __init__(self, front: bool = True):
        self.front = front

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        return {}

    def report(self, data: ExperimentData) -> dict[str, Any]:
        return self._report(data)
```

Флаг `front` определяет формат ключей:
- `front=True` → ключи через пробел: `"revenue TTest p-value 1"`
- `front=False` → ключи через `ID_SPLIT_SYMBOL`: `"revenue┆TTest┆p-value┆1"`

### 2.4. `TestDictReporter` — специализация для статистических тестов

Добавляет два критических метода:

#### `_get_struct_dict()` — парсинг плоского словаря в иерархию

```python
@staticmethod
def _get_struct_dict(data: dict) -> dict:
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for key, value in data.items():
        if ID_SPLIT_SYMBOL not in key:
            continue
        parts = key.split(ID_SPLIT_SYMBOL)
        if len(parts) >= 4 and parts[2] in REPORTABLE_METRICS:
            tree[parts[0]][parts[3]][parts[1]][parts[2]] = value
    return dict(tree)
```

Вход: `{"revenue┆StatsTTest┆p-value┆1": 0.03, ...}`
Выход: `{"revenue": {"1": {"StatsTTest": {"p-value": 0.03}}}}`

#### `_convert_struct_dict_to_dataset()` — конвертация в таблицу

```python
@staticmethod
def _convert_struct_dict_to_dataset(data: dict, *, invert_pass: bool = False) -> SmallDataset:
    result = []
    for feature, groups in data.items():
        for group, tests in groups.items():
            row = {"feature": feature, "group": group}
            # GroupDifference metrics
            if "GroupDifference" in tests:
                metrics = tests["GroupDifference"]
                for k in ("control mean", "test mean", "difference",
                          "difference %", "ci lower", "ci upper"):
                    if k in metrics:
                        row[k] = metrics.get(k)
            # Test results
            for test_name, metrics in tests.items():
                if test_name == "GroupDifference":
                    continue
                norm_name = TEST_NAME_NORMALIZATION.get(test_name, test_name)
                row[f"{norm_name} pass"] = metrics.get("pass")
                row[f"{norm_name} p-value"] = metrics.get("p-value")
            result.append(row)
    # Convert pass booleans to "OK"/"NOT OK"
    for row in result:
        for k, v in list(row.items()):
            if "pass" in k:
                truthy = _is_truthy(v)
                if invert_pass:
                    row[k] = "OK" if truthy else "NOT OK"
                else:
                    row[k] = "NOT OK" if truthy else "OK"
    return SmallDataset.from_dict(result, roles={...})
```

**Ключевое различие `invert_pass`:**
- **A/A-тест** (`invert_pass=False`): обнаруженное различие = **плохо** → `"NOT OK"`
- **A/B-тест** (`invert_pass=True`): значимый эффект = **хорошо** → `"OK"`

### 2.5. `DatasetReporter` — универсальный обёрточный репортёр

```python
class DatasetReporter(Reporter):
    def __init__(
        self,
        dict_reporter: DictReporter | None = None,
        output_format: Literal["dict", "dataset"] = "dataset",
        single_row: bool = False,
        invert_pass: bool = False,
    ):
        self.dict_reporter = dict_reporter or DictReporter()
        self.output_format = output_format
        self.single_row = single_row
        self._invert_pass = invert_pass
```

Метод `report()`:
1. Вызывает `self._report(data)` → плоский словарь
2. Если `output_format == "dict"` → возвращает словарь
3. Если `single_row == True` → конвертирует в однострочный `SmallDataset`
4. Иначе → парсит через `_get_struct_dict` → `_convert_struct_dict_to_dataset`

---

## 3. Механизм извлечения данных

### 3.1. Формат составных ключей

Все результаты в `analysis_tables` хранятся под ключами формата:

```
ClassName┆params_hash┆key
```

Внутри таблиц строки индексируются как:

```
{group}┆{column}        ← для режима "groups"
{group}┆{column}        ← для режима "matched_pairs"
```

### 3.2. `REPORTABLE_METRICS` — извлекаемые метрики

```python
REPORTABLE_METRICS = frozenset({
    "pass",
    "p-value",
    "difference",
    "difference %",
    "control mean",
    "test mean",
    "ci lower",
    "ci upper",
})
```

Только эти метрики попадают в финальный отчёт. Промежуточные значения (`statistic`, `histogram`, `value_counts`) остаются в `analysis_tables` для отладки.

### 3.3. `TEST_NAME_NORMALIZATION` — нормализация имён тестов

```python
TEST_NAME_NORMALIZATION: dict[str, str] = {
    "StatsTTest": "TTest",
    "StatsKSTest": "KSTest",
    "StatsChi2Test": "Chi2Test",
    "StatsZTest": "ZTest",
    "StatsUTest": "UTest",
    "GroupTTest": "TTest",
    "GroupKSTest": "KSTest",
    "GroupChi2Test": "Chi2Test",
    "GroupUTest": "UTest",
    "TTest": "TTest",
    "KSTest": "KSTest",
    "Chi2Test": "Chi2Test",
    "UTest": "UTest",
}
```

Пользователь видит **единое имя** (`TTest`), независимо от того, выполнялся ли тест через `GroupTTest` (Pandas/scipy) или `StatsTTest` (Spark/агрегаты).

### 3.4. Функции извлечения

#### `extract_tests()`

```python
def extract_tests(data: ExperimentData, test_classes: list[type], front: bool) -> dict[str, Any]:
    result = {}
    for cls_ in test_classes:
        ids = data.get_ids(cls_, searched_space=ExperimentDataEnum.analysis_tables)
        for id_list in ids.get(cls_.__name__, {}).values():
            for cid in id_list:
                for k, v in _extract_from_comparator(data, cid, front).items():
                    if "pass" in k or "p-value" in k:
                        result[k] = v
    return result
```

#### `extract_group_difference()`

```python
def extract_group_difference(data: ExperimentData, front: bool) -> dict[str, Any]:
    ids = data.get_ids(GroupDifference)[GroupDifference.__name__][
        ExperimentDataEnum.analysis_tables.value
    ]
    out = {}
    for cid in ids:
        out.update(_extract_from_comparator(data, cid, front))
    return out
```

#### `extract_group_sizes()`

```python
def extract_group_sizes(data: ExperimentData, front: bool) -> dict[str, Any]:
    cid = data.get_one_id(GroupSizes, ExperimentDataEnum.analysis_tables)
    return _extract_from_comparator(data, cid, front)
```

#### `_extract_from_comparator()` — ядро извлечения

```python
def _extract_from_comparator(data, comparator_id, front) -> dict[str, Any]:
    table = data.analysis_tables.get(comparator_id)
    if table is None or table.is_empty():
        return {}
    key = ResultKey.from_id(comparator_id)
    sep = " " if front else ID_SPLIT_SYMBOL
    result = {}
    records = table.to_records()
    index_values = _get_index_values(table)
    for idx_val, row_dict in zip(index_values, records):
        idx_str = str(idx_val)
        if NAME_BORDER_SYMBOL in idx_str:
            group, feature = idx_str.split(NAME_BORDER_SYMBOL, 1)
        else:
            group = idx_str
            feature = key.field
        group = _normalize_group_name(group)
        for col, val in row_dict.items():
            full_key = f"{feature}{sep}{key.executor}{sep}{col}{sep}{group}"
            result[full_key] = _normalize_value(val)
    return result
```

### 3.5. `_normalize_group_name()` — унификация групп

Spark `StatsComparator` может порождать ключи вида `"(1,)"`, а Pandas `GroupsComparator` — `"1"`. Функция унифицирует:

```python
def _normalize_group_name(group: str) -> str:
    stripped = group.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        inner = stripped[1:-1]
        if inner.endswith(","):
            inner = inner[:-1]
        if "," not in inner:
            return inner.strip()
    return group
```

---

## 4. Конкретные репортёры

### 4.1. `AATestReporter`

Извлекает результаты одной итерации A/A-теста:

```python
class AATestReporter(DatasetReporter):
    tests: ClassVar[list[type[BaseComparator]]] = [
        GroupTTest, GroupKSTest, GroupChi2Test,
        StatsTTest, StatsKSTest, StatsChi2Test, StatsUTest,
    ]

    def _build_dict_report(self, data: ExperimentData) -> dict[str, Any]:
        result = {"splitter_id": self.get_splitter_id(data)}
        front_flag = self.dict_reporter.front
        result.update(extract_group_difference(data, front_flag))
        result.update(extract_tests(data, self.tests, front_flag))
        result.update(extract_analyzer_data(data, "OneAAStatAnalyzer"))
        return result
```

### 4.2. `AAPassedReporter`

Формирует итоговую таблицу `summary` для A/A-теста:

| Колонка | Описание |
|---------|----------|
| `feature` | Имя метрики |
| `group` | Группа сравнения |
| `result` | `"OK"` / `"NOT OK"` |
| `{Test} aa score` | Прошёл ли тест по AA-score |
| `{Test} best split` | Прошёл ли тест на лучшем сплите |
| `control mean` | Среднее в контроле |
| `test mean` | Среднее в тесте |
| `difference` | Абсолютная разница |
| `difference %` | Относительная разница |

### 4.3. `AABestSplitReporter`

Присоединяет колонку `split` лучшего сплиттера к исходному датасету:

```python
class AABestSplitReporter(Reporter):
    def report(self, data: ExperimentData) -> Dataset:
        best_split_id = next(
            (c for c in data.additional_fields.columns if c.endswith("best")),
            None,
        )
        if best_split_id is None:
            return data.ds
        markers = data.additional_fields.select([best_split_id])
        markers = markers.rename({best_split_id: "split"})
        result = data.ds.merge(markers, left_index=True, right_index=True)
        return result
```

### 4.4. `ABTestReporter`

```python
class ABTestReporter(DatasetReporter):
    tests: ClassVar[list] = [
        GroupTTest, GroupKSTest, GroupUTest, GroupChi2Test,
        StatsTTest, StatsKSTest, StatsChi2Test, StatsUTest,
    ]

    def _report(self, data: ExperimentData) -> dict[str, Any]:
        result = {}
        result.update(extract_group_sizes(data, self.front))
        result.update(extract_group_difference(data, self.front))
        result.update(extract_tests(data, self.tests, self.front))
        result.update(extract_analyzer_data(data, ABAnalyzer))
        return result
```

### 4.5. `MatchingReporter`

Извлекает метрики эффектов и индексы матчинга:

```python
class MatchingReporter(DatasetReporter):
    def _report(self, data: ExperimentData) -> dict[str, Any]:
        result = self._extract_from_analyser(data)
        if self.searching_class == MatchingAnalyzer:
            result.update(self._extract_indexes(data))
        return result
```

Для Spark-бэкенда `_extract_indexes()` возвращает **пустой словарь** — индексы уже существуют как ленивые колонки в `ds`.

### 4.6. `CupedReporter` / `CupacReporter`

Извлекают отчёты о variance reduction из `analysis_tables`:

```python
class CupedReporter(Reporter):
    def report(self, data: ExperimentData) -> SmallDataset:
        ids = data.get_ids(CUPEDTransformer, searched_space=ExperimentDataEnum.analysis_tables)
        table_ids = ids.get(CUPEDTransformer.__name__, {}).get(
            ExperimentDataEnum.analysis_tables.value, []
        )
        if not table_ids:
            return SmallDataset.create_empty()
        table = data.analysis_tables[table_ids[0]]
        records = table.to_records()
        rows = [{"feature": r.get("feature"),
                 "variance_reduction_pct": r.get("variance_reduction_pct")}
                for r in records]
        return SmallDataset.from_dict(rows, roles={...})
```

---

## 5. Аналитики

### 5.1. `OneAAStatAnalyzer` — агрегация одной итерации

**Задача:** собрать результаты всех тестов одной итерации и вычислить composite score.

```python
class OneAAStatAnalyzer(Executor):
    ANALYSIS_TEST_CLASSES: ClassVar[tuple[type, ...]] = (
        GroupTTest, GroupKSTest, GroupChi2Test, GroupUTest,
        StatsTTest, StatsChi2Test, StatsZTest, StatsKSTest, StatsUTest,
    )

    _SCORE_RULES: ClassVar[tuple[tuple[str, str, int], ...]] = (
        ("StatsTTest", "GroupTTest", 1),      # T-test: вес 1
        ("StatsKSTest", "GroupKSTest", 2),     # KS: вес 2
        ("StatsChi2Test", "GroupChi2Test", 2), # Chi2: вес 2
        ("StatsUTest", "GroupUTest", 1),       # U-test: вес 1
    )
```

#### Алгоритм `execute()`:

```
1. data.get_ids(ANALYSIS_TEST_CLASSES, analysis_tables)
   → найти все таблицы тестов

2. _collect_means(data, executor_ids)
   → для каждого класса: средний p-value и средний pass
   → ключ: "mean┆ClassName┆p-value┆all"

3. _sanitize_nan(analysis_data)
   → NaN → 0.0

4. _compute_composite_score(analysis_data)
   → взвешенный средний p-value
   → Stats* приоритетнее Group*

5. set_value(analysis_tables, self.id, SmallDataset.from_dict(analysis_data))
```

#### Composite score:

```python
@classmethod
def _compute_composite_score(cls, analysis_data: dict[str, float]) -> float:
    score = 0.0
    total_weight = 0
    for preferred, fallback, weight in cls._SCORE_RULES:
        pref_key = _mean_key(preferred, "p-value")
        fall_key = _mean_key(fallback, "p-value")
        if pref_key in analysis_data:
            score += weight * analysis_data[pref_key]
            total_weight += weight
        elif fall_key in analysis_data:
            score += weight * analysis_data[fall_key]
            total_weight += weight
    return score / total_weight if total_weight else 0.0
```

### 5.2. `AAScoreAnalyzer` — выбор лучшего сплита

**Задача:** оценить качество каждой итерации и выбрать оптимальный сплит.

#### Параметры:

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `alpha` | `0.05` | Целевой уровень значимости |
| `THRESHOLD_FACTOR` | `1.2` | Множитель порога |
| `PVALUE_WEIGHT` | `2/3` | Вес p-values в финальном скоре |
| `TEST_SCORE_WEIGHT` | `1/3` | Вес mean_test_score |

#### Алгоритм `execute()`:

```
1. Получить таблицу результатов ParamsExperiment
   → param_id = data.get_one_id(ParamsExperiment, analysis_tables)
   → score_table = data.analysis_tables[param_id]

2. _analyze_aa_score(data, score_table)
   → для каждой колонки "pass":
     * pass_rate = доля прошедших итераций
     * weight = 1 - |alpha - pass_rate|
     * pass = weight >= threshold
   → сохранить как "aa score"

3. _analyze_best_split(data, score_table)
   → _find_best_index():
     * weighted_pvalues × 2/3 + mean_test_score × 1/3
     * best_index = score_col.idxmax()
   → _extract_splitter_id():
     * извлечь "AASplitter┆rs 7┆" из строки
   → сохранить "best split statistics"

4. _set_best_split(data, best_splitter_id):
   → build_splitter_from_id(best_splitter_id)
   → best_splitter.key = "best"
   → best_splitter.execute(data)
   → добавляет колонку "AASplitter┆rs 7┆best"
```

#### Формула выбора лучшего сплита:

```
score = weighted_pvalues × (2/3) + mean_test_score × (1/3)

где:
  weighted_pvalues = Σ (p-value_col × weight_col) / len(feature_weights)
  mean_test_score  = колонка "mean test score" из OneAAStatAnalyzer
```

#### Реконструкция сплиттера из ID:

```python
def build_splitter_from_id(self, splitter_id: str) -> AASplitter:
    class_name = splitter_id[: splitter_id.find(ID_SPLIT_SYMBOL)]
    splitter_class = self.SPLITTER_CLASS_MAPPING.get(class_name)
    if splitter_class is None:
        raise ValueError(f"{splitter_id} is not a valid splitter id")
    return splitter_class.build_from_id(splitter_id)
```

### 5.3. `AADryTestAnalyzer` — диагностика равномерности

**Задача:** проверить, что p-values при отсутствии эффекта распределены равномерно (Uniform(0,1)).

```python
class AADryTestAnalyzer(Executor):
    def execute(self, data: ExperimentData) -> ExperimentData:
        param_id = data.get_one_id(ParamsExperiment, analysis_tables)
        score_table = data.analysis_tables[param_id]
        target_columns = data.field_search(TargetRole())
        p_val_dist = self._uniform_test(score_table, target_columns)
        output = self._set_value(data, p_val_dist, analysis_tables)
        return self._set_value(output, self._fig, variables)
```

#### `_uniform_test()`:

Для каждой колонки `"TTest p-value"`:
1. Извлекает массив p-values
2. Применяет KS-тест против Uniform(0,1) через `UniformCheck`
3. Вычисляет FPR = доля p-values < alpha
4. Строит гистограмму с ожидаемой линией Uniform

Результат сохраняется в `variables` как `matplotlib.Figure` для визуализации.

### 5.4. `ABAnalyzer` — агрегация A/B-теста

**Задача:** собрать результаты тестов, вычислить средние по группам, применить поправку на множественное тестирование.

#### Алгоритм `execute()`:

```
1. data.get_ids([GroupTTest, GroupUTest, GroupKSTest,
                 StatsTTest, StatsChi2Test, StatsKSTest, StatsUTest])

2. Определить group_field и заполнить data.groups если пусто
   (для Spark/StatsComparator пути)

3. Для каждого класса тестов:
   a. Объединить все таблицы через append()
   b. Построить составной индекс строк:
      "TestName┆hash┆target┆group"
   c. Для каждой группы: средний p-value и pass
   d. Собрать p-values для multitest

4. execute_multitest(data, multitest_pvalues)
   → если num_comparisons > 1 и метод задан

5. Сохранить analysis_dataset в analysis_tables
```

#### `execute_multitest()`:

```python
def execute_multitest(self, data, p_values, **kwargs):
    group_field = data.ds.search_columns(TreatmentRole())[0]
    target_fields = data.ds.search_columns(TargetRole(), search_types=[int, float])
    num_groups = len(data.groups[group_field])
    num_comparisons = (num_groups - 1) * len(target_fields)

    if self.multitest_method and num_comparisons > 1:
        if self.multitest_method != ABNTestMethodsEnum.quantile:
            multitest_result = MultiTest(self.multitest_method, self.alpha).calc(p_values)
        else:
            multitest_result = SmallDataset.create_empty()
            for target_field in target_fields:
                multitest_result = multitest_result.append(
                    MultitestQuantile(...).calc(p_values, ...)
                )
        return self._set_value(data, multitest_result, key="MultiTest")
    return data
```

### 5.5. `MatchingAnalyzer` — форматирование метрик матчинга

```python
class MatchingAnalyzer(Executor):
    def execute(self, data: ExperimentData):
        variables = data.variables[
            data.get_one_id(MatchingMetrics, space=ExperimentDataEnum.variables)
        ]
        columns = ["Effect Size", "Standard Error", "P-value", "CI Lower", "CI Upper"]
        return self._set_value(
            data,
            DatasetAdapter.to_dataset(
                variables,
                {field: StatisticRole(float) for field in variables},
            ).transpose(roles={column: StatisticRole(float) for column in columns}),
        )
```

Транспонирует словарь `{"ATT": [...], "ATC": [...], "ATE": [...]}` в таблицу, где строки = метрики, колонки = статистики.

---

## 6. Поправка на множественное тестирование

### 6.1. `MultiTest` — обёртка над `statsmodels`

```python
class MultiTest(Extension):
    def __init__(self, method: ABNTestMethodsEnum, alpha: float = 0.05):
        self.method = method
        self.alpha = alpha

    def _calc_pandas(self, data: Dataset, **kwargs):
        p_values = data.raw_data.values.flatten()
        tests_raw, fields, groups = self._index_parts(data.index)
        tests = [TEST_NAME_NORMALIZATION.get(t, t) for t in tests_raw]

        corrected = np.empty(len(p_values), dtype=float)
        rejected = np.empty(len(p_values), dtype=bool)

        # Correction per statistical test family
        for test in dict.fromkeys(tests):
            positions = [i for i, name in enumerate(tests) if name == test]
            test_rejected, test_corrected = multipletests(
                [p_values[i] for i in positions],
                method=self.method.value,
                alpha=self.alpha,
                **kwargs,
            )[:2]
            corrected[positions] = test_corrected
            rejected[positions] = test_rejected

        return DatasetAdapter.to_dataset({
            "field": fields,
            "test": tests,
            "old p-value": p_values,
            "new p-value": corrected,
            "correction": [old/new if old != 0 else 0.0 for new, old in zip(corrected, p_values)],
            "rejected": rejected,
            "group": groups,
        }, StatisticRole())
```

**Ключевой принцип:** поправка применяется **внутри каждого семейства тестов** отдельно. T-test p-values корректируются вместе, KS-test p-values — вместе, но не смешиваются.

### 6.2. `MultitestQuantile` — симуляционный подход

Для метода `"quantile"` используется Монте-Карло симуляция:

```
1. Вычислить квантиль маргинального распределения:
   quantile_of_marginal_distribution(num_samples, quantile_level, variances)

2. Для каждой группы:
   a. Сгенерировать iteration_size выборок из N(0,1)
   b. Для каждой выборки: вычислить min t-value по всем парам
   c. Сравнить с критическим квантилем

3. Если min_t_value > quantile → гипотеза принимается
```

### 6.3. Поддерживаемые методы

| Метод | Описание |
|-------|----------|
| `bonferroni` | Коррекция Бонферрони |
| `sidak` | Коррекция Шидака |
| `holm-sidak` | Холм-Шидак |
| `holm` | Холм (по умолчанию) |
| `simes-hochberg` | Симес-Хохберг |
| `hommel` | Хоммел |
| `fdr_bh` | FDR Бенджамини-Хохберг |
| `fdr_by` | FDR Бенджамини-Йекутиэли |
| `fdr_tsbh` | FDR двухэтапный |
| `fdr_tsbky` | FDR двухэтапный KY |
| `quantile` | Симуляционный (Монте-Карло) |

---

## 7. Output extraction pipeline

### 7.1. `Output.extract()` — базовый механизм

```python
class Output:
    def _extract_by_reporters(self, experiment_data: ExperimentData):
        self.summary = self.summary_reporter.report(experiment_data)
        for attribute, reporter in self.additional_reporters.items():
            setattr(self, attribute, reporter.report(experiment_data))
        self._experiment_data = experiment_data

    def extract(self, experiment_data: ExperimentData):
        self._extract_by_reporters(experiment_data)
```

### 7.2. `AAOutput.extract()`

```python
class AAOutput(Output):
    best_split: Dataset
    experiments: Dataset
    aa_score: Dataset
    best_split_statistics: Dataset

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)  # summary + best_split
        self._extract_experiments(experiment_data)
        self._extract_aa_score(experiment_data)
        self._extract_best_split_statistics(experiment_data)
```

| Атрибут | Источник | Содержимое |
|---------|----------|------------|
| `summary` | `AAPassedReporter` | OK/NOT OK по каждой фиче |
| `best_split` | `AABestSplitReporter` | Данные с колонкой split |
| `experiments` | `ParamsExperiment` таблица | Все итерации |
| `aa_score` | `AAScoreAnalyzer` | Score и pass |
| `best_split_statistics` | `AAScoreAnalyzer` | Детали лучшего сплита |

### 7.3. `ABOutput.extract()`

```python
class ABOutput(Output):
    multitest: Dataset | str
    sizes: Dataset
    cuped: CupedOutput | None
    cupac: CupacOutput

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)  # summary
        self._extract_differences(experiment_data)
        self._extract_multitest_result(experiment_data)
        self._extract_sizes(experiment_data)
        self._extract_cupac(experiment_data)
        if self.cuped is not None:
            self.cuped.extract(experiment_data)
```

| Атрибут | Источник | Содержимое |
|---------|----------|------------|
| `summary` | `ABTestReporter` | Разницы, p-values, pass |
| `multitest` | `ABAnalyzer` → `MultiTest` | Скорректированные p-values |
| `sizes` | `GroupSizes` | Размеры групп |
| `cuped` | `CupedReporter` | Variance reduction |
| `cupac` | `CupacReporter` | Variance reduction + importances |

### 7.4. `MatchingOutput.extract()`

```python
class MatchingOutput(Output):
    summary: Dataset
    full_data: Dataset
    quality_results: Dataset

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)  # summary + quality_results
        reformatted_summary = self._reformat_summary(self.summary)

        if self.compute_indexes:
            if experiment_data.ds.backend_type == BackendsEnum.spark:
                indexes = self._get_spark_indexes(experiment_data)
            else:
                indexes = self._extract_driver_indexes(experiment_data, reformatted_summary)
        else:
            indexes = SmallDataset.create_empty()

        # Build summary table
        if reformatted_summary:
            first_key = next(iter(reformatted_summary.keys()))
            group_keys = list(reformatted_summary[first_key].keys())
            transposed_summary = {
                metric: [values[group] for group in group_keys]
                for metric, values in reformatted_summary.items()
            }
            self.summary = SmallDataset.from_dict(
                {"data": transposed_summary, "index": group_keys},
                roles={column: StatisticRole() for column in reformatted_summary},
            )

        if self.extract_full_data:
            self._extract_full_data(experiment_data, indexes)

        self.summary.raw_data = self.summary.raw_data.round(2)
```

**Оптимизация для Spark:**
- `compute_indexes=False` → индексы не собираются на драйвер
- `extract_full_data=False` → не выполняется итеративный цикл merge + checkpoint

---

## 8. Пример: путь данных от `TTest` до `result.summary`

### Шаг 1: `StatsTTest.execute()` записывает результат

```python
# analysis_tables["StatsTTest┆┆revenue"] = SmallDataset:
#   index: ["1┆revenue"]
#   columns: ["p-value", "statistic", "pass"]
#   values:  [0.03, -2.14, True]
```

### Шаг 2: `ABAnalyzer.execute()` агрегирует

```python
# analysis_tables["ABAnalyzer┆┆"] = SmallDataset:
#   columns: ["StatsTTest p-value 1", "StatsTTest pass 1", ...]
#   values:  [0.03, 1.0, ...]
```

### Шаг 3: `ABTestReporter._report()` извлекает

```python
result = {
    "revenue StatsTTest p-value 1": 0.03,
    "revenue StatsTTest pass 1": True,
    "revenue GroupDifference difference 1": 2.8,
    "revenue GroupDifference control mean 1": 152.3,
    ...
}
```

### Шаг 4: `DatasetReporter.report()` конвертирует

```python
# _get_struct_dict():
{"revenue": {"1": {"StatsTTest": {"p-value": 0.03, "pass": True},
                    "GroupDifference": {"difference": 2.8, ...}}}}

# _convert_struct_dict_to_dataset():
# feature | group | control mean | test mean | difference | TTest pass | TTest p-value
# revenue | 1     | 152.3        | 155.1     | 2.8        | OK         | 0.03
```

### Шаг 5: `ABOutput.extract()` присваивает

```python
self.summary = result_dataset  # таблица выше
self.multitest = multitest_result  # скорректированные p-values
self.sizes = sizes_dataset  # размеры групп
```

### Шаг 6: Пользователь видит результат

```python
result = ab.execute(data)
print(result.summary)
# feature | group | control mean | test mean | difference | TTest pass | TTest p-value
# revenue | 1     | 152.3        | 155.1     | 2.8        | OK         | 0.03
```

---

## 9. Контракт для создания кастомного репортёра

```python
from hypex.reporters.abstract import Reporter, DatasetReporter, DictReporter
from hypex.dataset import ExperimentData, Dataset, SmallDataset

# Вариант 1: наследование от Reporter
class MyReporter(Reporter):
    def report(self, data: ExperimentData) -> Dataset:
        # Извлечение из analysis_tables
        ids = data.get_ids(MyExecutor, ExperimentDataEnum.analysis_tables)
        table = data.analysis_tables[ids[MyExecutor.__name__]["analysis_tables"][0]]
        # Форматирование
        return table

# Вариант 2: наследование от DatasetReporter
class MyDatasetReporter(DatasetReporter):
    def _report(self, data: ExperimentData) -> dict[str, Any]:
        result = {}
        result.update(extract_tests(data, [MyTest], self.front))
        result.update(extract_group_difference(data, self.front))
        return result

# Использование в Output
class MyOutput(Output):
    custom_table: Dataset

    def __init__(self):
        super().__init__(
            summary_reporter=MyDatasetReporter(
                dict_reporter=DictReporter(),
                output_format="dataset",
            ),
            additional_reporters={"custom_table": MyReporter()},
        )
```

---

*Документация актуальна для версии **2.0.0a1**. Следующий раздел — «Расширяемость: создание кастомных экспериментов и блоков (Уровни 5–6)».*
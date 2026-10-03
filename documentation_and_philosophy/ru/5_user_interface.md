# Пользовательский интерфейс: оболочки экспериментов (Уровень 4)

---

## 1. `ExperimentShell` — универсальная обёртка

Каждая из четырёх оболочек (`AATest`, `ABTest`, `Matching`, `HomogeneityTest`) является наследником `ExperimentShell`. Это класс, который связывает **пайплайн** (`Experiment`) и **извлечение результатов** (`Output`) в единый вызов `execute()`.

```python
class ExperimentShell:
    def __init__(
        self,
        experiment: Experiment,
        output: Output,
        experiment_params: dict[str, Any] | None = None,
        auto_persist: bool = True,
    ):
        if experiment_params:
            experiment.set_params(experiment_params)
        self._out = output
        self._experiment = experiment
        self.auto_persist = auto_persist
```

### 1.1. Жизненный цикл `execute()`

```
ExperimentShell.execute(data)
│
├── 1. Упаковка данных
│   ├── Dataset → ExperimentData(deepcopy(data))
│   └── ExperimentData → data.copy()
│
├── 2. Авто-кэширование (только Spark)
│   └── если auto_persist=True и бэкенд=Spark и не persisted:
│       original_ds.persist("MEMORY_AND_DISK", action="count")
│       persisted_by_us = True
│
├── 3. Выполнение пайплайна
│   └── result = self._experiment.execute(data)
│
├── 4. Извлечение результатов
│   └── self._out.extract(result)
│
└── 5. Очистка (в finally)
    └── если persisted_by_us → original_ds.unpersist()
```

Ключевое свойство: **исходные данные пользователя не мутируются**. Если `data` — это `Dataset`, он оборачивается через `deepcopy`. Если `ExperimentData` — через `data.copy()`.

### 1.2. Свойство `experiment`

```python
@property
def experiment(self) -> Experiment:
    return self._experiment
```

Позволяет интроспектировать пайплайн после создания оболочки:

```python
ab = ABTest()
print(ab.experiment.executors)  # список Executor'ов
```

---

## 2. Система `Output`

### 2.1. Базовый класс

```python
class Output:
    summary: Dataset
    _experiment_data: ExperimentData

    def __init__(
        self,
        summary_reporter: Reporter | None = None,
        additional_reporters: dict[str, Reporter] | None = None,
        *,
        resume_reporter: Reporter | None = None,  # deprecated
    ):
        self.summary_reporter = summary_reporter
        self.additional_reporters = additional_reporters or {}
```

**Механизм извлечения:**

```python
def _extract_by_reporters(self, experiment_data: ExperimentData):
    self.summary = self.summary_reporter.report(experiment_data)
    for attribute, reporter in self.additional_reporters.items():
        setattr(self, attribute, reporter.report(experiment_data))
    self._experiment_data = experiment_data

def extract(self, experiment_data: ExperimentData):
    self._extract_by_reporters(experiment_data)
```

Каждый `additional_reporters` становится **атрибутом** объекта `Output` с тем же именем, что и ключ словаря.

### 2.2. `_Report` — рендеринг

`Output.__repr__()` и `Output._repr_html_()` делегируют в `_Report`, который рендерит **все таблицы** под заголовками:

```python
class _Report:
    def __init__(self, title: str, tables: dict[str, Any]):
        self.title = title
        self.tables = dict(tables)

    def __repr__(self) -> str:
        parts = [f"{self.title}:"]
        for name, table in self.tables.items():
            parts.append(f"\n{'=' * 60}")
            parts.append(f"{name}:")
            parts.append("=" * 60)
            parts.append("None" if table is None else str(table))
        return "\n".join(parts)

    def _repr_html_(self) -> str:
        return "\n".join(
            _html_section(name) + _html_content(table)
            for name, table in self.tables.items()
        )
```

В Jupyter Notebook `print(result)` покажет текстовый вывод, а просто `result` в ячейке — HTML-таблицы.

### 2.3. Обнаружение полей для рендеринга

```python
def _get_output_fields(self) -> list[str]:
    annotations: dict[str, Any] = {}
    for cls in reversed(type(self).__mro__):
        annotations.update(vars(cls).get("__annotations__", {}))
    fields = [
        name
        for name in annotations
        if not name.startswith("_") and name in vars(self)
    ]
    if "summary" in fields:
        fields.remove("summary")
        fields.insert(0, "summary")
    return fields
```

Каждый подкласс `Output` объявляет свои поля через **аннотации типов**. Например, `AAOutput` объявляет `experiments`, `aa_score`, `best_split`, `best_split_statistics` — и все они автоматически попадают в `repr`.

---

## 3. `AATest` — проверка качества рандомизации

### 3.1. Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `precision_mode` | `bool` | `False` | `True` → 2000 итераций, `False` → 10 |
| `control_size` | `float` | `0.5` | Доля контрольной группы (0–1) |
| `stratification` | `bool` | `False` | Стратифицированный сплит |
| `n_iterations` | `int \| None` | `None` | Явное число итераций (переопределяет `precision_mode`) |
| `sample_size` | `float \| None` | `None` | Доля данных для сэмплирования |
| `additional_params` | `dict \| None` | `None` | Доп. параметры для `ParamsExperiment` |
| `random_states` | `Iterable[int] \| None` | `None` | Зёрна для итераций. По умолчанию `range(n_iterations)` |
| `equal_variance` | `bool \| None` | `None` | `True` → Student, `False` → Welch, `None` → Welch |
| `groups_sizes` | `list[float] \| None` | `None` | Кастомные пропорции групп |
| `float32` | `bool` | `False` | Приведение к `float32` для экономии памяти |
| `early_stopping` | `bool` | `False` | Остановка при прохождении всех тестов |
| `dry_test` | `bool` | `False` | Диагностика равномерности p-values |

### 3.2. Внутренний пайплайн

```
Experiment(key="AATest")
│
├── [Float32Caster?]                      ← если float32=True
│
├── ParamsExperiment                      ← основная итерация
│   ├── executors: [base_experiment]
│   ├── params: {AASplitter: {random_state, control_size, ...},
│   │            Comparator: {grouping_role, space}}
│   ├── reporter: DatasetReporter(AATestReporter, single_row=True)
│   └── stopping_criterion: [IfAAExecutor?]  ← если early_stopping
│
│   base_experiment = Experiment:
│   ├── NaDropper()                       ← удаление NaN из таргетов
│   ├── AASplitter() | AASplitterWithStratification()
│   └── aa_metrics = Experiment:
│       ├── GroupSizes(AdditionalTreatmentRole)
│       ├── OnRoleExperiment(role=[TargetRole, FeatureRole]):
│       │   ├── GroupDifference(AdditionalTreatmentRole)
│       │   ├── TTest(AdditionalTreatmentRole, reliability=0.05)
│       │   ├── KSTest(AdditionalTreatmentRole)
│       │   └── Chi2Test(AdditionalTreatmentRole)
│       └── OneAAStatAnalyzer()           ← агрегация
│
├── [IfParamsExperiment?]                 ← если sample_size задан
│   └── stopping_criterion: IfAAExecutor(sample_size=...)
│
├── [AADryTestAnalyzer?]                  ← если dry_test=True
│
└── AAScoreAnalyzer()                     ← выбор лучшего сплита
```

### 3.3. `AAOutput`

```python
class AAOutput(Output):
    best_split: Dataset
    experiments: Dataset
    aa_score: Dataset
    best_split_statistics: Dataset

    def __init__(self):
        super().__init__(
            summary_reporter=AAPassedReporter(),
            additional_reporters={"best_split": AABestSplitReporter()},
        )
```

| Атрибут | Тип | Содержимое |
|---------|-----|------------|
| `summary` | `Dataset` | Итоговая таблица: `feature`, `group`, `result` (OK/NOT OK), `TTest aa score`, `TTest best split`, `control mean`, `test mean`, `difference`, `difference %` |
| `experiments` | `Dataset` | Таблица итераций: каждая строка = одна итерация, колонки = тесты |
| `aa_score` | `Dataset` | AA-score и `pass` для каждого `(фича, тест)` |
| `best_split` | `Dataset` | Исходные данные с колонкой `split` лучшего сплита |
| `best_split_statistics` | `Dataset` | Детальная статистика лучшего сплита |

### 3.4. Пример использования

```python
from hypex import AATest
from hypex.dataset import Dataset, TargetRole, TreatmentRole, FeatureRole

ds = Dataset(
    roles={
        TargetRole(): ["revenue", "spends"],
        FeatureRole(): ["age", "gender"],
        TreatmentRole(): "treat",
    },
    data="experiment.csv",
)

# Быстрый режим: 10 итераций
aa = AATest()
result = aa.execute(ds)

print(result.summary)            # OK/NOT OK по каждой фиче
print(result.aa_score)           # pass-rate и weight
print(result.best_split)         # колонка "split" лучшего сплита
print(result.best_split_statistics)  # детальная статистика

# Точный режим: 2000 итераций, стратификация, ранняя остановка
aa_precise = AATest(
    precision_mode=True,
    stratification=True,
    early_stopping=True,
    equal_variance=False,
)
result = aa_precise.execute(ds)
```

---

## 4. `ABTest` — сравнение групп

### 4.1. Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `additional_tests` | `str \| list \| None` | `["t-test"]` | Доп. тесты: `"t-test"`, `"ks-test"`, `"u-test"`, `"chi2-test"` |
| `multitest_method` | `str \| None` | `"holm"` | Поправка на множественное тестирование |
| `equal_variance` | `bool \| None` | `None` | `True` → Student, `False` → Welch |
| `cuped_features` | `dict[str, str] \| None` | `None` | `{target: pre_target}` для CUPED |
| `cupac_models` | `str \| list[str] \| None` | `None` | Модели для CUPAC: `"linear"`, `"ridge"`, `"lasso"`, `"catboost"` |
| `enable_cupac` | `bool` | `False` | Включить CUPAC variance reduction |
| `float32` | `bool` | `False` | Приведение к `float32` |

### 4.2. Внутренний пайплайн

```
Experiment
│
├── [CUPEDTransformer?]                   ← если cuped_features задан
│   └── создаёт {target}_cuped колонки
│
├── [CUPACExecutor?]                      ← если enable_cupac=True
│   └── ML-предсказания для variance reduction
│
├── [Float32Caster?]                      ← если float32=True
│
├── GroupSizes(TreatmentRole)             ← размеры групп
│
├── OnRoleExperiment(role=TargetRole [+ AdditionalTargetRole если CUPAC]):
│   ├── GroupDifference(TreatmentRole)    ← разница средних + 95% CI
│   ├── TTest(TreatmentRole)              ← t-тест
│   ├── [KSTest?]                         ← если в additional_tests
│   ├── [UTest?]                          ← если в additional_tests
│   └── [Chi2Test?]                       ← если в additional_tests
│
└── ABAnalyzer(multitest_method)          ← агрегация + поправка
```

### 4.3. `ABOutput`

```python
class ABOutput(Output):
    multitest: Dataset | str
    sizes: Dataset
    cuped: CupedOutput | None
    cupac: CupacOutput
```

| Атрибут | Тип | Содержимое |
|---------|-----|------------|
| `summary` | `Dataset` | Основная таблица: `feature`, `group`, `control mean`, `test mean`, `difference`, `difference %`, `ci lower`, `ci upper`, `{Test} pass`, `{Test} p-value` |
| `multitest` | `Dataset \| str` | Поправка на множественное тестирование или сообщение |
| `sizes` | `Dataset` | Размеры групп: `control size`, `test size`, `control size %`, `test size %` |
| `cuped` | `CupedOutput \| None` | Variance reduction от CUPED (если включён) |
| `cupac` | `CupacOutput` | Variance reduction от CUPAC |

**`CupedOutput`:**

| Атрибут | Содержимое |
|---------|------------|
| `variance_reductions` | `feature`, `variance_reduction_pct` |

**`CupacOutput`:**

| Атрибут | Содержимое |
|---------|------------|
| `variance_reductions` | `target`, `best_model`, `variance_reduction_cv`, `variance_reduction_real` |
| `feature_importances` | `target`, `feature`, `importance`, `model` |

### 4.4. Пример использования

```python
from hypex import ABTest
from hypex.dataset import Dataset, TargetRole, TreatmentRole, FeatureRole

ds = Dataset(
    roles={
        TargetRole(): ["revenue", "spends", "conversion"],
        TreatmentRole(): "treat",
        FeatureRole(): ["age", "gender"],
    },
    data="ab_data.csv",
)

# Базовый A/B-тест с t-тестом и поправкой Холма
ab = ABTest()
result = ab.execute(ds)

print(result.summary)      # разница средних, p-value, pass
print(result.multitest)    # скорректированные p-values
print(result.sizes)        # размеры групп

# Расширенный: несколько тестов + CUPED
ab_advanced = ABTest(
    additional_tests=["t-test", "chi2-test", "u-test"],
    multitest_method="bonferroni",
    cuped_features={"revenue": "revenue_lag_1", "spends": "spends_lag_1"},
    equal_variance=False,
)
result = ab_advanced.execute(ds)

print(result.summary)
print(result.cuped.variance_reductions)  # % редукции дисперсии

# С CUPAC (ML-based variance reduction)
ab_cupac = ABTest(
    additional_tests=["t-test"],
    cupac_models=["linear", "ridge"],
    enable_cupac=True,
)
result = ab_cupac.execute(ds)
print(result.cupac.variance_reductions)
print(result.cupac.feature_importances)
```

### 4.5. Когда применяется поправка на множественное тестирование

Поправка применяется, если:
1. `multitest_method` задан (не `None`), **и**
2. `(num_groups - 1) × num_target_fields > 1`

Для стандартных методов (`holm`, `bonferroni`, `fdr_bh`, ...) используется `MultiTest` (обёртка над `statsmodels.multipletests`). Для `"quantile"` — `MultitestQuantile` (Монте-Карло симуляция).

---

## 5. `Matching` — оценка причинно-следственных связей

### 5.1. Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `group_match` | `bool` | `False` | Матчинг по группам (через `GroupExperiment`) |
| `distance` | `"mahalanobis" \| "l2"` | `"mahalanobis"` | Метрика расстояния |
| `bias_estimation` | `bool` | `True` | Коррекция остаточного смещения |
| `quality_tests` | `str \| list[str]` | `"auto"` | Тесты качества: `"t-test"`, `"ks-test"`, `"chi2-test"` или `"auto"` |
| `faiss_mode` | `"base" \| "fast" \| "auto"` | `"auto"` | Режим FAISS |
| `n_neighbors` | `int` | `1` | Число соседей |
| `weights` | `dict[str, float] \| None` | `None` | Веса признаков для Махаланобиса |
| `encode_categories` | `bool` | `True` | One-hot кодирование категориальных |
| `extract_full_data` | `bool` | `False` | Извлекать полный датасет с матчингом |
| `compute_indexes` | `bool` | `False` | Извлекать индексы матчинга |

### 5.2. Внутренний пайплайн

```
Experiment
│
├── [DummyEncoder?]                       ← если encode_categories=True
│   └── one-hot кодирование категориальных признаков
│
├── TypeCaster(dtype={int: float},       ← приведение типов
│              roles=[FeatureRole, TargetRole])
│
├── [MahalanobisDistance?]                ← если distance="mahalanobis"
│   └── ковариация → Холецкий → обратная матрица → веса
│
├── FaissNearestNeighbors                 ← поиск соседей
│   ├── grouping_role=TreatmentRole
│   ├── two_sides=(metric=="ate")
│   ├── test_pairs=(metric=="atc")
│   ├── faiss_mode, n_neighbors
│   └── результат: AdditionalMatchingRole колонки
│
├── [Bias?]                               ← если bias_estimation=True
│   ├── grouping_role=TreatmentRole
│   ├── target_roles=[TargetRole]
│   └── линейная регрессия на матченных данных
│
├── MatchingMetrics                       ← расчёт ATT/ATC/ATE
│   ├── grouping_role=TreatmentRole
│   ├── target_roles=[TargetRole]
│   ├── metric="ate"
│   └── n_neighbors
│
├── MatchingAnalyzer                      ← транспонирование результатов
│
└── OnRoleExperiment(role=FeatureRole):   ← quality tests
    ├── [TTest(matched_pairs)?]
    ├── [KSTest(matched_pairs)?]
    └── [Chi2Test(matched_pairs)?]
```

Если `group_match=True`, весь пайплайн оборачивается в `GroupExperiment`, который выполняет его **для каждой группы** отдельно.

### 5.3. `MatchingOutput`

```python
class MatchingOutput(Output):
    summary: Dataset
    full_data: Dataset
    quality_results: Dataset
```

| Атрибут | Тип | Содержимое |
|---------|-----|------------|
| `summary` | `Dataset` | Таблица эффектов: строки = `ATT`, `ATC`, `ATE`; колонки = `Effect Size`, `Standard Error`, `P-value`, `CI Lower`, `CI Upper` |
| `quality_results` | `Dataset` | Результаты quality tests: p-values матчинга по признакам |
| `full_data` | `Dataset` | Полный датасет с `*_matched_{i}` колонками (если `extract_full_data=True`) |

### 5.4. Пример использования

```python
from hypex import Matching
from hypex.dataset import Dataset, TargetRole, TreatmentRole, FeatureRole

ds = Dataset(
    roles={
        TargetRole(): ["revenue"],
        TreatmentRole(): "treat",
        FeatureRole(): ["age", "gender", "industry", "spends_lag"],
    },
    data="matching_data.csv",
)

# Базовый матчинг: Махаланобис + FAISS + bias correction + ATE
m = Matching()
result = m.execute(ds)

print(result.summary)
#        Effect Size  Standard Error  P-value  CI Lower  CI Upper
# ATT         12.34           2.11    0.001      8.20     16.48
# ATC         10.87           1.95    0.002      7.05     14.69
# ATE         11.61           1.43    0.000      8.81     14.41

print(result.quality_results)
# p-values для каждого признака после матчинга

# Расширенный: несколько соседей, L2, конкретные тесты
m_adv = Matching(
    distance="l2",
    n_neighbors=3,
    bias_estimation=True,
    quality_tests=["t-test", "ks-test"],
    faiss_mode="fast",
    extract_full_data=True,
    compute_indexes=True,
)
result = m_adv.execute(ds)

print(result.summary)
print(result.full_data)       # полный датасет с матчингом
```

### 5.5. Оптимизация извлечения для Spark

По умолчанию `extract_full_data=False` и `compute_indexes=False`. Это **критично** для Spark:

- `compute_indexes=False` → индексы матчинга остаются ленивыми колонками в `ds`, **не собираются на драйвер**.
- `extract_full_data=False` → не выполняется итеративный цикл `merge + checkpoint` для построения `full_data`.

Для Spark-бэкенда индексы извлекаются через `_get_spark_indexes()` — чистая проекция колонок без материализации:

```python
def _get_spark_indexes(self, experiment_data):
    ids = experiment_data.get_ids(
        FaissNearestNeighbors, ExperimentDataEnum.additional_fields
    )[FaissNearestNeighbors.__name__][ExperimentDataEnum.additional_fields.value]
    additional = experiment_data.additional_fields
    available = sorted(
        (col for col in ids if col in additional.columns),
        key=lambda c: int(str(c).split(ID_SPLIT_SYMBOL)[-1]),
    )
    indexes = additional[available]  # ленивая Spark-проекция
    return indexes.rename({...})
```

---

## 6. `HomogeneityTest` — проверка однородности

### 6.1. Пайплайн

Самая простая оболочка — один `OnRoleExperiment` с батареей тестов:

```python
HOMOGENEITY_TEST = Experiment(
    executors=[
        OnRoleExperiment(
            executors=[
                GroupSizes(grouping_role=TreatmentRole(), compare_by="groups"),
                GroupDifference(grouping_role=TreatmentRole(), compare_by="groups"),
                GroupTTest(grouping_role=TreatmentRole(), compare_by="groups"),
                GroupKSTest(grouping_role=TreatmentRole(), compare_by="groups"),
                GroupChi2Test(grouping_role=TreatmentRole(), compare_by="groups"),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)

class HomogeneityTest(ExperimentShell):
    def __init__(self):
        super().__init__(
            experiment=HOMOGENEITY_TEST,
            output=HomoOutput(),
        )
```

### 6.2. `HomoOutput`

```python
class HomoOutput(Output):
    summary: Dataset

    def __init__(self):
        super().__init__(summary_reporter=HomoDatasetReporter())
```

Единственный атрибут — `summary` с результатами всех тестов.

### 6.3. Пример использования

```python
from hypex import HomogeneityTest
from hypex.dataset import Dataset, TargetRole, TreatmentRole

ds = Dataset(
    roles={
        TargetRole(): ["revenue", "spends", "conversion"],
        TreatmentRole(): "treat",
    },
    data="homogeneity_data.csv",
)

homo = HomogeneityTest()
result = homo.execute(ds)

print(result.summary)
# feature | group | control mean | test mean | difference | TTest pass | TTest p-value | ...
```

---

## 7. Сводная таблица оболочек

| Оболочка | Пайплайн | `summary_reporter` | Доп. атрибуты `Output` |
|----------|----------|--------------------|-----------------------|
| `AATest` | `NaDropper → Splitter → ParamsExperiment → AAScoreAnalyzer` | `AAPassedReporter` | `experiments`, `aa_score`, `best_split`, `best_split_statistics` |
| `ABTest` | `[CUPED] → [CUPAC] → GroupSizes → OnRoleExperiment → ABAnalyzer` | `ABTestReporter` | `multitest`, `sizes`, `cuped`, `cupac` |
| `Matching` | `[DummyEncoder → TypeCaster] → [Mahalanobis] → Faiss → [Bias] → Metrics → Analyzer → QualityTests` | `MatchingDictReporter` | `quality_results`, `full_data` |
| `HomogeneityTest` | `OnRoleExperiment(GroupSizes, GroupDifference, GroupTTest, GroupKSTest, GroupChi2Test) → OneAAStatAnalyzer` | `HomoDatasetReporter` | — |

---

## 8. Паттерн расширения: создание своей оболочки

Для создания новой оболочки достаточно:

```python
from hypex.experiments.base import Experiment, OnRoleExperiment
from hypex.comparators import GroupDifference, TTest
from hypex.dataset import TargetRole, TreatmentRole
from hypex.reporters import DatasetReporter, DictReporter
from hypex.ui.base import ExperimentShell, Output
from hypex.reporters.abstract import Reporter
from hypex.dataset import ExperimentData, Dataset

# 1. Свой репортёр
class MyReporter(Reporter):
    def report(self, data: ExperimentData) -> Dataset:
        # ... извлечение данных ...
        return result_dataset

# 2. Свой Output
class MyOutput(Output):
    custom_table: Dataset  # аннотация → попадёт в repr

    def __init__(self):
        super().__init__(
            summary_reporter=MyReporter(),
        )

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        # ... дополнительная логика ...

# 3. Своя оболочка
class MyExperiment(ExperimentShell):
    def __init__(self, param1: int = 10):
        experiment = Experiment(
            executors=[
                GroupSizes(grouping_role=TreatmentRole()),
                OnRoleExperiment(
                    executors=[GroupDifference(grouping_role=TreatmentRole()), TTest(grouping_role=TreatmentRole())],
                    role=TargetRole(),
                ),
            ]
        )
        super().__init__(
            experiment=experiment,
            output=MyOutput(),
            experiment_params={"some_key": param1},
        )

# Использование
my_exp = MyExperiment(param1=20)
result = my_exp.execute(data)
print(result.summary)
print(result.custom_table)
```

---

*Документация актуальна для версии **2.0.0a1**. Следующий раздел — «Предобработка: трансформеры, сплиттеры, энкодеры».*
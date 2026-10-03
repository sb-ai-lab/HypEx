# Обзор архитектуры HypEx 2.0

---

## 1. Миссия и позиционирование

**HypEx** (Hypothesis Experiments) — библиотека для проведения и анализа статистических экспериментов на данных любого масштаба: от нескольких тысяч строк в `pandas.DataFrame` до сотен миллионов записей в распределённом кластере Apache Spark.

### Что решает

| Задача | Класс в HypEx | Что делает |
|--------|---------------|------------|
| Проверка качества рандомизации | `AATest` | Многократный A/A-сплит с оценкой FPR, выбором лучшего сплита, ранней остановкой |
| Сравнение групп по метрикам | `ABTest` | Разница средних, t-тест, KS, χ², U-тест, поправка на множественное тестирование |
| Оценка причинно-следственных связей без рандомизации | `Matching` | FAISS-матчинг, коррекция смещения, расчёт ATT/ATC/ATE |
| Проверка однородности групп | `HomogeneityTest` | Батарея тестов на идентичность распределений |

### Чем отличается от альтернатив

- **`scipy.stats`** — даёт только один тест на два массива. HypEx строит полный пайплайн: сплиттинг → тестирование → агрегация → поправка на множественное тестирование → отчёт.
- **`statsmodels`** — не поддерживает распределённые вычисления. В HypEx `StatsComparator` агрегирует все метрики одним `groupBy().agg()`-джобом в Spark, а не `N × M` отдельными задачами.
- **Самописные пайплайны** — не переиспользуемы, не тестируемы. В HypEx каждый блок — `Executor` с идентификатором, параметрами и воспроизводимым поведением.
- **Библиотеки матчинга (`causalml`, `dowhy`)** — не имеют распределённого FAISS. HypEx реализует четыре стратегии распределённого индекса (`sample`, `cluster`, `full`, `shuffle`) с LRU-кэшем на экзекьюторах.

### Ключевая идея

Один и тот же код пользователя работает на **любом бэкенде**:

```python
from hypex import ABTest

ab = ABTest(additional_tests=["t-test", "chi2-test"])
result = ab.execute(data)   # data: Dataset на любом бэкенде
print(result.summary)
```

Если `data` создан на `pandas.DataFrame` — тесты выполняются через `scipy`. Если на `spark.DataFrame` — через агрегированные статистики без сбора данных на драйвер. **Пользователь не меняет ни строки кода.**

---

## 2. Восемь уровней абстракции

Архитектура построена по принципу **слоёной луковки**: каждый уровень использует только уровень ниже и не знает об уровнях выше. Это позволяет бизнес-аналитику запускать эксперимент кнопкой, а архитектору ядра — менять фундаментальное поведение без поломки пользовательского API.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Уровень 1   Пользователь на платформе (без кода)                      │
│  Уровень 2   Конструктор сценариев (без кода, со знанием статистики)    │
│  Уровень 3   Шаблонный код с параметрами (Dev-пакет)                   │
│  Уровень 4   Оболочки: AATest / ABTest / Matching / HomogeneityTest    │
│  Уровень 5   Сборка пайплайна из Executor'ов                           │
│  Уровень 6   Создание кастомного Executor'а наследованием              │
│  Уровень 7   Модификация ядра: бэкенды, роли, ExperimentData          │
│  Уровень 8   Архитектурные решения нового поколения                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Уровень 1 — Пользователь на платформе

| Параметр | Значение |
|----------|----------|
| **Пользователь** | Бизнес-аналитик, продакт-менеджер |
| **Сегмент** | Платформа (внешний продукт) |
| **Идея** | Запуск готового сценария одной кнопкой без знания статистики |
| **Пример** | Выбрать сценарий «A/B-тест конверсии», загрузить CSV, нажать «Запустить» |

На этом уровне HypEx — «чёрный ящик». Платформа вызывает `ABTest().execute(data)` и рендерит `result.summary` как таблицу.

### Уровень 2 — Конструктор сценариев

| Параметр | Значение |
|----------|----------|
| **Пользователь** | Финансист / аналитик со знанием статистики |
| **Сегмент** | Платформа (визуальный конструктор) |
| **Идея** | Сборка пайплайна из блоков в GUI без написания кода |
| **Пример** | Перетащить блоки: `NaDropper → AASplitter → TTest → AAScoreAnalyzer` |

Конструктор генерирует тот же самый объект `Experiment(executors=[...])`, что программист собрал бы кодом (Уровень 5).

### Уровень 3 — Шаблонный код с параметрами (Dev-пакет)

| Параметр | Значение |
|----------|----------|
| **Пользователь** | Аналитик с доступом к лабораторной зоне |
| **Сегмент** | Dev-пакет (шаблонный скрипт) |
| **Идея** | Запуск заранее запрограммированного сценария с настройкой параметров |
| **Пример** | `AATest(precision_mode=True, stratification=True, n_iterations=100)` |

### Уровень 4 — Оболочки экспериментов

| Параметр | Значение |
|----------|----------|
| **Пользователь** | Исследователь данных, прочитавший туториал |
| **Сегмент** | Библиотека |
| **Идея** | Базовый эксперимент в пару строк кода |

Все четыре оболочки — наследники `ExperimentShell`:

```python
class ExperimentShell:
    def __init__(self, experiment: Experiment, output: Output, ...):
        ...
    def execute(self, data: Dataset | ExperimentData) -> Output:
        ...
```

`ExperimentShell.execute()` принимает данные, вызывает `Experiment.execute()`, передаёт результат в `Output.extract()` и возвращает готовый объект с таблицами.

**Оболочка → внутренний пайплайн:**

| Оболочка | Ключевые Executor'ы пайплайна |
|----------|-------------------------------|
| `AATest` | `NaDropper → AASplitter → GroupSizes → OnRoleExperiment(GroupDifference, TTest, KSTest, Chi2Test) → OneAAStatAnalyzer → ParamsExperiment → AAScoreAnalyzer` |
| `ABTest` | `[CUPEDTransformer] → [CUPACExecutor] → [Float32Caster] → GroupSizes → OnRoleExperiment(GroupDifference, TTest, …) → ABAnalyzer(multitest)` |
| `Matching` | `[DummyEncoder → TypeCaster] → [MahalanobisDistance] → FaissNearestNeighbors → [Bias] → MatchingMetrics → MatchingAnalyzer → OnRoleExperiment(quality tests)` |
| `HomogeneityTest` | `OnRoleExperiment(GroupSizes, GroupDifference, GroupTTest, GroupKSTest, GroupChi2Test) → OneAAStatAnalyzer` |

### Уровень 5 — Сборка пайплайна из Executor'ов

| Параметр | Значение |
|----------|----------|
| **Пользователь** | Разработчик, знакомый с `Experiment` и `Executor` |
| **Сегмент** | Библиотека |
| **Идея** | Кастомный эксперимент из готовых блоков |

```python
from hypex.experiments.base import Experiment, OnRoleExperiment
from hypex.comparators import GroupDifference, TTest
from hypex.dataset import TargetRole, TreatmentRole

my_experiment = Experiment(executors=[
    GroupSizes(grouping_role=TreatmentRole()),
    OnRoleExperiment(
        executors=[
            GroupDifference(grouping_role=TreatmentRole()),
            TTest(grouping_role=TreatmentRole()),
        ],
        role=TargetRole(),
    ),
])
```

Типы `Experiment` на этом уровне:

| Класс | Назначение |
|-------|------------|
| `Experiment` | Последовательный пайплайн |
| `OnRoleExperiment` | Итерация по колонкам заданной роли; разделение на векторные и итеративные исполнители |
| `ParamsExperiment` | Sweep по декартову произведению параметров (для A/A итераций) |
| `IfParamsExperiment` | Sweep с ранним выходом по `IfExecutor` |
| `CycledExperiment` | N итераций с репортёром |
| `GroupExperiment` | Исполнение по группам (для `Matching` с `group_match=True`) |

### Уровень 6 — Создание кастомного Executor'а

| Параметр | Значение |
|----------|----------|
| **Пользователь** | Разработчик, знакомый с типовыми блоками |
| **Сегмент** | Библиотека |
| **Идея** | Новый блок наследованием от `Calculator`, `Transformer`, `GroupsComparator`, `StatsComparator` или `MLExecutor` |

Минимальный контракт:

```python
class MyTest(StatsHypothesisTesting):
    REQUIRED_STATS = ["mean", "var", "count"]

    @classmethod
    def _inner_function(cls, baseline_stats, compared_stats, **kwargs):
        # ... формула теста ...
        return {"p-value": p, "statistic": t, "pass": p < reliability}
```

Регистрация в фабрике:

```python
from hypex.utils.registry import backend_factory

@backend_factory.register(MyMaster, SparkDataset)
class MySparkTest(MyTest): ...
```

### Уровень 7 — Модификация ядра

| Параметр | Значение |
|----------|----------|
| **Пользователь** | Хорошо знакомый с внутренним устройством разработчик |
| **Сегмент** | Библиотека |
| **Идея** | Глубокие доработки: новый бэкенд, новая роль, изменение `ExperimentData` |

Точки входа для модификации:

- **Добавить бэкенд**: реализовать `DatasetBackendCalc` (40+ абстрактных методов) → `backend_factory.register_explicit()`
- **Добавить роль**: наследоваться от `ABCRole` → зарегистрировать в `default_roles`
- **Изменить пространство данных**: модифицировать `ExperimentData.set_value()` / `field_search()`

### Уровень 8 — Ядро

| Параметр | Значение |
|----------|----------|
| **Пользователь** | Архитектор HypEx |
| **Сегмент** | Библиотека |
| **Идея** | Изменение фундаментального поведения — нового поколения архитектуры |

Примеры решений Уровня 8 в текущем коде:
- Переход от отдельного `additional_fields` к единому `ds` с ролевым разделением
- Разделение `GroupsComparator` (сырые данные) и `StatsComparator` (агрегаты)
- `tmp_roles` для итеративных пайплайнов без дублирования данных

---

## 3. Ключевые архитектурные решения

### 3.1. Backend-agnostic `Dataset` с фабрикой

```
DatasetBase  (abstract.py)
  ├── Dataset        (dataset.py)       — основной контейнер
  ├── SmallDataset   (dataset.py)       — компактный (результаты анализа)
  └── _backend_data: PandasDataset | SparkDataset
                        │
                        ├── PandasNavigation + PandasDataset   (pandas_backend.py)
                        └── SparkNavigation  + SparkDataset    (spark_backend.py)
```

`DatasetBase` делегирует все операции `_backend_data` — объекту `PandasDataset` или `SparkDataset`. Пользователь работает с `Dataset`, не зная о бэкенде. Конвертация:

```python
spark_ds = pandas_ds.to_backend(BackendsEnum.spark, session=spark_session)
pandas_ds = spark_ds.to_backend(BackendsEnum.pandas)  # с проверкой лимита строк
```

`DatasetConfig.SPARK_PANDAS_CONVERSION_LIMIT = 100_000` защищает от OOM на драйвере.

### 3.2. Ролевая система колонок

Каждая колонка в `Dataset` имеет семантическую роль — экземпляр `ABCRole`. Роли определяют, **как пайплайн использует колонку**, а не как она называется.

```
ABCRole
├── TargetRole          — целевая метрика
├── FeatureRole         — признак (наследует LagRole: parent, lag)
├── TreatmentRole       — столбец обработки / группы
├── GroupingRole        — группировка
├── PreTargetRole       — пре-таргет для CUPED
├── StratificationRole  — стратификация
├── ConstGroupRole      — закреплённая группа
├── IndexRole / InfoRole / StatisticRole / FilterRole
├── DisabledRole        — «мягкое» удаление (хранит initial_role)
├── TempRole
│   ├── TempTargetRole
│   ├── TempTreatmentRole
│   └── TempGroupingRole
├── AdditionalRole
│   ├── AdditionalTargetRole
│   ├── AdditionalFeatureRole
│   ├── AdditionalTreatmentRole
│   ├── AdditionalMatchingRole
│   ├── AdditionalStatisticRole
│   └── AdditionalVarianceReductionRole
└── ReportRole / DefaultRole / ResumeRole
```

Поиск колонок по роли:

```python
targets = data.ds.search_columns(TargetRole(), search_types=[int, float])
```

Роли заменяют хардкод имён колонок: пайплайн не знает, что метрика называется `revenue` — он знает, что это `TargetRole(float)`.

### 3.3. Pipeline из Executor'ов

Фундаментальный паттерн:

```
Executor.execute(data: ExperimentData) → ExperimentData
```

Каждый `Executor` получает контейнер, мутирует его (добавляет колонки, заполняет `analysis_tables`, создаёт `variables`) и возвращает. `Experiment` последовательно вызывает `execute()` каждого исполнителя.

Иерархия:

```
Executor (ABC)
├── Calculator
│   ├── GroupsComparator    — сравнение на сырых данных
│   │   └── GroupHypothesisTesting (reliability)
│   ├── StatsComparator     — двухфазное: агрегация → сравнение
│   │   └── StatsHypothesisTesting (reliability)
│   ├── MLExecutor          — fit/predict, группировка по TreatmentRole
│   ├── Transformer         — _is_transformer=True, копирует данные
│   ├── Encoder             — кодирование категориальных колонок
│   └── GroupOperator       — оператор на двух таргетах (Bias, MatchingMetrics)
├── IfExecutor              — условное исполнение (check_rule)
└── Experiment              — последовательный пайплайн из executors
```

Идентификация каждого `Executor`:

```
id = ClassName ┆ params_hash ┆ key
```

где `┆` — `ID_SPLIT_SYMBOL = "┆"`. Это позволяет однозначно находить результаты в `ExperimentData` через `get_ids()`.

### 3.4. `ExperimentData` — единый контейнер состояния

```python
class ExperimentData:
    _data: Dataset              # основной датасет (ds)
    variables: dict[str, dict]  # скалярные параметры эксперимента
    groups: dict[str, dict[str, Dataset]]  # разрезанные по группам поднаборы
    analysis_tables: dict[str, SmallDataset]  # результаты тестов и анализаторов
    _initial_cols: list[str]    # колонки до пайплайна (для изоляции итераций)
```

Четыре пространства хранения:

| Пространство | Что хранит | Пример |
|-------------|------------|--------|
| `ds` (+ `additional_fields` как view) | Основной датасет + синтетические колонки | Сплит-метка, матчинг-индексы |
| `variables` | Скаляры, конфиги, модели | `MahalanobisDistance` матрица |
| `groups` | `Dict[group_label, Dataset]` | `{"control": ds_ctrl, "test_1": ds_test}` |
| `analysis_tables` | `Dict[executor_id, SmallDataset]` | Результаты `TTest`, `AAScoreAnalyzer` |

Критический метод — `_clean_ds_for_iteration()`: удаляет все колонки с `AdditionalRole` и вызывает `checkpoint()` на Spark. Вызывается перед каждой итерацией `ParamsExperiment`, `CycledExperiment`, `GroupExperiment` для **изоляции итераций**.

### 3.5. Мастер-класс → бэкенд-реализация через реестр

Паттерн **фабрика + реестр**:

```python
# comparators/comparators.py — мастер-класс (пустой)
class TTest(StatTestMasterAbstract): pass

# comparators/hypothesis_testing.py — Pandas-реализация
@backend_factory.register(TTest, PandasDataset)
class GroupTTest(GroupHypothesisTesting): ...

# comparators/stats_hypothesis_testing.py — Spark-реализация
@backend_factory.register(TTest, SparkDataset)
class StatsTTest(StatsHypothesisTesting): ...
```

В момент исполнения `Experiment._get_executor_backend()` резолвит:

```python
@staticmethod
def _get_executor_backend(executor, ds):
    executor_cls = type(executor)                    # TTest
    backend_cls = backend_factory.resolve_backend(executor_cls, ds)  # GroupTTest или StatsTTest
    if backend_cls is None:
        return executor
    # Перенос параметров конструктора + разворачивание calc_kwargs
    return backend_cls(**init_kwargs)
```

**Результат**: пользователь пишет `TTest(grouping_role=TreatmentRole())` — а выполняется `StatsTTest` на Spark или `GroupTTest` на Pandas.

### 3.6. `GroupsComparator` vs `StatsComparator`

Это ключевое разделение для производительности на Spark:

| | `GroupsComparator` | `StatsComparator` |
|---|---|---|
| **Данные** | Сырые `Dataset`-срезы | Предварительно агрегированные статистики |
| **Фаза 1** | — | `_compute_stats()`: один `groupBy().agg()` на все колонки |
| **Фаза 2** | `_inner_function(baseline_ds, compared_ds)` | `_inner_function(baseline_stats, compared_stats)` |
| **Spark-джобы** | `N × M` (группы × колонки) | **1** агрегация + скалярные вычисления на драйвере |
| **Режимы** | `groups`, `columns`, `columns_in_groups`, `cross`, `matched_pairs` | `groups`, `matched_pairs` |
| **Примеры** | `GroupTTest`, `GroupKSTest`, `GroupChi2Test`, `GroupUTest` | `StatsTTest`, `StatsKSTest`, `StatsChi2Test`, `StatsUTest`, `StatsZTest`, `GroupDifference`, `GroupSizes` |

`StatsComparator` хранит два артефакта:
- `{self.id}┆stats` — таблица агрегатов (строки = группы, колонки = `{stat}┆{col}`)
- `{self.id}` — попарные результаты тестов

### 3.7. `OnRoleExperiment`: векторные и итеративные исполнители

`OnRoleExperiment` оптимизирует итерацию по таргет-колонкам:

```python
# experiments/base.py, OnRoleExperiment.execute()
for ex in self.executors:
    if isinstance(ex, (StatsComparator, StatsHypothesisTesting)):
        vector_executors.append(ex)     # все колонки за 1 проход
    elif isinstance(ex, StatTestMasterAbstract):
        if backend == BackendsEnum.spark:
            vector_executors.append(ex)  # мастер → Stats* на Spark
        else:
            iterative_executors.append(ex)  # мастер → Group* на Pandas
    else:
        iterative_executors.append(ex)
```

Векторные исполнители получают `tmp_roles` со **всеми** таргетами сразу и выполняются один раз. Итеративные — в цикле по одной колонке, каждый раз устанавливая `tmp_roles = {field: TempTargetRole()}`.

---

## 4. Схема взаимодействия модулей

```
                    ┌──────────────────────────────────┐
                    │       Пользовательский код        │
                    │  AATest / ABTest / Matching / ... │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │       ui/base.py                  │
                    │  ExperimentShell + Output         │
                    │  (execute → extract → repr)       │
                    └──────────────┬───────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
   ┌──────────▼──────┐  ┌─────────▼────────┐  ┌───────▼──────────┐
   │  aa.py / ab.py   │  │  matching.py     │  │ homogeneity.py   │
   │  AATest / ABTest │  │  Matching        │  │ HomogeneityTest  │
   └──────────┬──────┘  └─────────┬────────┘  └───────┬──────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │  experiments/base.py              │
                    │  Experiment / OnRoleExperiment    │
                    │  experiments/base_complex.py      │
                    │  ParamsExperiment / GroupExperiment│
                    └──────────────┬───────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
   ┌──────────▼──────┐  ┌─────────▼────────┐  ┌───────▼──────────┐
   │ executor/        │  │ transformers/    │  │ splitters/       │
   │ Executor         │  │ Transformer      │  │ AASplitter       │
   │ Calculator       │  │ NaDropper        │  │ AASplitterWith   │
   │ MLExecutor       │  │ CUPEDTransformer │  │ Stratification   │
   │ IfExecutor       │  │ TypeCaster       │  │                  │
   └──────────┬──────┘  │ Filters          │  └──────────────────┘
              │          └─────────────────┘
              │
   ┌──────────▼──────────────────────────────────────────┐
   │              comparators/                            │
   │  abstract.py:  BaseComparator → GroupsComparator    │
   │                               → StatsComparator     │
   │  comparators.py: TTest, KSTest, Chi2Test, UTest,   │
   │                  ZTest, GroupDifference, GroupSizes  │
   │  hypothesis_testing.py: GroupTTest, GroupKSTest...  │
   │  stats_hypothesis_testing.py: StatsTTest, StatsKSTest│
   │  distances.py: MahalanobisDistance                  │
   └──────────────────────┬──────────────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────────────┐
   │              extensions/                             │
   │  faiss.py: PandasFaissExtension / SparkFaissExtension│
   │  bias.py: PandasBisaExtesion / SparkBisaExtesion     │
   │  matching_metric.py: MatchingMetricsExtension        │
   │  scipy_stats.py: GroupTTestExtension, ...            │
   │  stats_hypothesis_testing.py: StatsAggregationExt   │
   │  statsmodels.py: MultiTest, MultitestQuantile        │
   │  scipy_linalg.py: CholeskyExtension, InverseExtension│
   │  cupac.py: CupacExtension                           │
   └──────────────────────┬──────────────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────────────┐
   │              dataset/                                │
   │  abstract.py: DatasetBase (backend-agnostic API)     │
   │  dataset.py: Dataset, SmallDataset, DatasetAdapter   │
   │  experiment_data.py: ExperimentData                  │
   │  roles.py: ABCRole + 30+ ролей                      │
   │  groupby_dataset.py: GroupedDataset                 │
   │  backends/:                                          │
   │    abstract.py: DatasetBackendNavigation / Calc      │
   │    pandas_backend.py: PandasDataset                  │
   │    spark_backend.py: SparkDataset                    │
   └──────────────────────┬──────────────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────────────┐
   │              utils/ + config/                        │
   │  registry.py: BackendFactory (фабрика бэкендов)      │
   │  config.py: DatasetConfig, MatchingConfig            │
   │  logger.py: HypExLogger                              │
   │  profiling.py: @timeit                               │
   │  constants.py: ID_SPLIT_SYMBOL, NAME_BORDER_SYMBOL   │
   │  enums.py: BackendsEnum, ExperimentDataEnum          │
   └─────────────────────────────────────────────────────┘
```

### Поток данных при вызове `ABTest().execute(data)`

```
1. ExperimentShell.execute(data)
   ├── ExperimentData(deepcopy(data))           # упаковка в контейнер
   ├── [авто] persist для Spark                  # если auto_persist=True
   │
   ├── Experiment.execute(experiment_data)
   │   ├── [CUPEDTransformer]                    # если задан
   │   │   └── data.copy(data=new_ds)            # Transformer → deepcopy
   │   │
   │   ├── [CUPACExecutor]                       # если enable_cupac
   │   │
   │   ├── [Float32Caster]                       # если float32=True
   │   │
   │   ├── GroupSizes.execute()                  # StatsComparator
   │   │   ├── Phase 1: _compute_stats()         # 1 Spark job: count
   │   │   └── Phase 2: _inner_function()        # скалярная арифметика
   │   │   └── data.set_value(analysis_tables, ...)
   │   │
   │   ├── OnRoleExperiment.execute()
   │   │   ├── vector: TTest → StatsTTest        # Spark: 1 agg job
   │   │   │   └── Phase 1: mean, std, count
   │   │   │   └── Phase 2: t-statistic, p-value
   │   │   └── iterative: GroupDifference        # если на Pandas
   │   │       └── для каждого таргета:
   │   │           tmp_roles = {col: TempTargetRole()}
   │   │           super().execute(data)
   │   │
   │   └── ABAnalyzer.execute()
   │       ├── сбор результатов из analysis_tables
   │       ├── execute_multitest()               # MultiTest / MultitestQuantile
   │       └── data.set_value(analysis_tables, ...)
   │
   ├── Output.extract(result_experiment_data)
   │   ├── summary_reporter.report()             # ABTestReporter
   │   ├── _extract_differences()
   │   ├── _extract_multitest_result()
   │   ├── _extract_sizes()
   │   ├── _extract_cupac()
   │   └── [cuped.extract()]
   │
   └── return ABOutput
       ├── .summary        — таблица: feature | group | difference | TTest pass | ...
       ├── .multitest      — поправка на множественное тестирование
       ├── .sizes          — размеры групп
       ├── .cupac          — variance reduction отчёты
       └── .cuped          — variance reduction отчёты
```

---

## 5. Что нового в 2.0

### 5.1. Единый `Dataset` вместо разрозненных pandas-обёрток

**Было (гипотетическая 1.0)**: каждый компонент работал с `pd.DataFrame` напрямую. Для Spark нужен был отдельный код.

**Стало**: `DatasetBase` с делегированием в `PandasDataset` / `SparkDataset`. Один интерфейс, два бэкенда. Конвертация через `to_backend()` с сохранением индекса и ролей.

### 5.2. Ролевая система вместо хардкода имён

**Было**: `group_field="treatment"`, `target_field="revenue"` — строки.

**Стало**: `grouping_role=TreatmentRole()`, `target_roles=TargetRole()`. Роли несут тип данных (`data_type`), временные метаданные (`parent`, `lag` в `LagRole`), информацию об исходной роли (`DisabledRole.initial_role`).

### 5.3. `StatsComparator` для минимизации Spark-джобов

**Было**: `GroupsComparator` итерировал по парам (группа × колонка) — `N × M` Spark-джобов.

**Стало**: `StatsComparator` выполняет **одну** агрегацию (`_compute_stats()`), собирает скалярные словари `{col: {stat: value}}` на драйвер и сравнивает аналитически. Для 10 метрик и 5 групп: было 50 джобов → стало 1.

### 5.4. `ExperimentData` вместо передачи DataFrame по цепочке

**Было**: каждый `Executor` принимал и возвращал `pd.DataFrame`, результаты хранились в отдельных переменных.

**Стало**: единый контейнер с четырьмя пространствами (`ds`, `variables`, `groups`, `analysis_tables`). Любой `Executor` может прочитать результат любого другого через `data.get_ids(SomeClass, space)`.

### 5.5. Фабрика бэкендов

**Было**: `if backend == "spark": ... else: ...` в каждом компоненте.

**Стало**: `@backend_factory.register(MasterClass, BackendType)` — декларативная регистрация. Резолвинг в рантайме через `backend_factory.resolve_backend(master_cls, data)`. Добавление нового бэкенда не требует изменения существующего кода.

### 5.6. Синтетические колонки в `ds` вместо отдельного `additional_fields`

**Было**: отдельный `additional_fields: Dataset` с ручным управлением индексов.

**Стало**: синтетические колонки (сплит-метки, матчинг-индексы, CUPED-таргеты) хранятся прямо в `ds` с ролями `Additional*Role`. Свойство `additional_fields` — read-only view для обратной совместимости. `_clean_ds_for_iteration()` удаляет их перед каждой итерацией.

### 5.7. Распределённый FAISS с четырьмя режимами

`SparkFaissExtension` реализует:
- `"sample"` — обучение IVF-квантизатора на случайной выборке
- `"cluster"` — обучение на всём датасете через итеративный `MiniBatchKMeans` / `BIRCH`
- `"full"` — точный поиск, свой `IndexFlatL2` на каждой партиции
- `"shuffle"` (по умолчанию) — кластеризация → ко-группировка по `(cluster, c_bucket, t_bucket)` → per-cluster поиск с ограничением `BUCKET_SIZE`

### 5.8. CUPED и CUPAC как модули дисперсионной редукции

`CUPEDTransformer` — линейная корректировка `Y' = Y - θ(X - E[X])`, где `θ = Cov(Y, X) / Var(X)`.
`CUPACExecutor` — ML-предсказания (linear, ridge, lasso, catboost) с K-Fold cross-fitting для оценки θ. Результаты хранятся в `analysis_tables` с отчётом по `variance_reduction_pct`.

---

## 6. Глоссарий ключевых терминов

| Термин | Определение |
|--------|-------------|
| **Роль** (`ABCRole`) | Семантический ярлык колонки, определяющий её участие в пайплайне |
| **Исполнитель** (`Executor`) | Атомарная единица работы: принимает `ExperimentData`, возвращает `ExperimentData` |
| **Эксперимент** (`Experiment`) | Упорядоченная последовательность исполнителей |
| **Пространство** (`ExperimentDataEnum`) | Именованное хранилище внутри `ExperimentData`: `variables`, `groups`, `analysis_tables`, `additional_fields` |
| **Мастер-класс** | Абстрактный или пустой класс-заглушка (`TTest`, `KSTest`), который резолвится в бэкенд-реализацию |
| **`tmp_roles`** | Временные роли, устанавливаемые на время одной итерации `OnRoleExperiment` |
| **`checkpoint()`** | Материализация данных в Spark для обрезки DAG-линейки |
| **`ID_SPLIT_SYMBOL`** (`┆`) | Разделитель в составных идентификаторах: `ClassName┆params_hash┆key` |
| **`NAME_BORDER_SYMBOL`** (`┆`) | Разделитель внутри имени колонки: `stat┆col` |

---

*Документация актуальна для версии **2.0.0a1**. Следующий раздел — «Слой данных: Dataset, Roles, Backends».*
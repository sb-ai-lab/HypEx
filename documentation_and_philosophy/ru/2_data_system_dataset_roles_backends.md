# Слой данных: Dataset, Roles, Backends

---

## 1. Иерархия классов данных

Слой данных построен по принципу **трёхуровневой матрёшки**: абстрактный интерфейс → конкретные контейнеры → адаптер конвертации. Каждый уровень решает свою задачу, не зная о внутреннем устройстве других.

```
DatasetBase  (dataset/abstract.py)
│
├── Dataset          (dataset/dataset.py)    — основной контейнер эксперимента
│   └── to_small_dataset() → SmallDataset
│
├── SmallDataset     (dataset/dataset.py)    — компактный контейнер результатов
│   └── to_dataset() → Dataset
│
└── DatasetAdapter   (dataset/dataset.py)    — статические фабрики конвертации
        ├── to_dataset()
        ├── dict_to_dataset()
        ├── frame_to_dataset()
        ├── list_to_dataset()
        ├── ndarray_to_dataset()
        └── value_to_dataset()
```

### 1.1. `DatasetBase` — абстрактный контракт

`DatasetBase` определяет **полный интерфейс**, с которым работают все `Executor`'ы, `Experiment`'ы и `Comparator`'ы. Он **не хранит данные напрямую** — делегирует их бэкенду через `_backend_data`.

Ключевые атрибуты:

| Атрибут | Тип | Назначение |
|---------|-----|------------|
| `_backend_data` | `PandasDataset \| SparkDataset` | Реальный носитель данных |
| `_roles` | `dict[str, ABCRole]` | Семантические роли колонок |
| `_tmp_roles` | `dict[str, ABCRole]` | Временные роли для итеративных пайплайнов |
| `default_role` | `ABCRole \| None` | Роль по умолчанию для колонок без явной роли |
| `loc` / `iloc` | `Locker` / `ILocker` | Label- и position-based доступ |

Конструктор автоматически:
1. Определяет бэкенд из типа входных данных (`pd.DataFrame` → `PandasDataset`, `spark.DataFrame` / `ps.DataFrame` → `SparkDataset`).
2. Нормализует роли через `_parse_roles()` — поддерживает оба формата: `{Role: [cols]}` и `{col: Role}`.
3. Назначает `DefaultRole()` всем колонкам без явной роли через `_set_all_roles()`.
4. Инферирует `data_type` роли из dtype колонки через `_set_empty_types()`.

### 1.2. `Dataset` — основной контейнер

`Dataset` — это «рабочая лошадка» пайплайна. От `DatasetBase` добавляет:

```python
class Dataset(DatasetBase):
    def to_small_dataset(self) -> SmallDataset: ...
    def sort(self, by=None, ascending=True, **kwargs): ...
```

Создание из разных источников:

```python
from hypex.dataset import Dataset, TargetRole, TreatmentRole, FeatureRole

# Из pandas DataFrame
ds = Dataset(
    roles={
        TargetRole(): ["revenue", "spends"],
        TreatmentRole(): "treat",
        FeatureRole(): ["age", "gender"],
    },
    data=pd.read_csv("data.csv"),
)

# Из Spark DataFrame
ds_spark = Dataset(
    roles={TargetRole(): "revenue", TreatmentRole(): "treat"},
    data=spark_df,
    session=spark_session,
)

# Из файла
ds = Dataset(
    roles={TargetRole(): "revenue"},
    data="data.csv",
)
```

### 1.3. `SmallDataset` — компактный контейнер результатов

`SmallDataset` хранит **результаты анализа** — таблицы тестов, агрегаты, отчёты. Всегда работает на **Pandas-бэкенде**, даже если исходные данные были в Spark. Это осознанное решение: результаты тестов — это десятки строк, и перенос их на драйвер безопасен.

Ключевые отличия от `Dataset`:

| Метод | `Dataset` | `SmallDataset` |
|-------|-----------|----------------|
| `from_dict()` | — | ✅ Статическая фабрика |
| `transpose()` | — | ✅ С ролями |
| `reindex()` | — | ✅ |
| `idxmax()` | — | ✅ |
| Бэкенд | любой | только `PandasDataset` |

```python
from hypex.dataset import SmallDataset, StatisticRole

result = SmallDataset.from_dict(
    {"p-value": [0.03], "statistic": [2.14], "pass": [True]},
    roles=StatisticRole(),
)
```

### 1.4. `DatasetAdapter` — фабрики конвертации

`DatasetAdapter` предоставляет **статические методы** для создания `Dataset` / `SmallDataset` из произвольных данных. Используется внутри `Comparator`'ов и `Extension`'ов:

```python
from hypex.dataset.dataset import DatasetAdapter

# Из скаляра
ds = DatasetAdapter.to_dataset(0.05, StatisticRole())

# Из dict
ds = DatasetAdapter.to_dataset(
    {"p-value": [0.03], "pass": [True]},
    StatisticRole(),
)

# Из DataFrame
ds = DatasetAdapter.to_dataset(pdf, roles={"col": InfoRole()})
```

---

## 2. Бэкенды

### 2.1. Архитектура бэкенд-абстракции

```
DatasetBackendNavigation  (abstract.py)
│   to_backend, index, columns, __getitem__, __len__,
│   loc, iloc, from_dict, to_dict, create_empty, ...
│
└── DatasetBackendCalc  (abstract.py)
    │   mean, var, std, sum, count, agg, groupby,
    │   merge, dropna, value_counts, corr, dot, ...
    │
    ├── PandasNavigation + PandasDataset  (pandas_backend.py)
    │   data: pd.DataFrame
    │
    └── SparkNavigation + SparkDataset    (spark_backend.py)
        data: pyspark.pandas.DataFrame
        session: SparkSession
```

`DatasetBackendNavigation` определяет **навигационный контракт** (индексация, срезы, типы, структура). `DatasetBackendCalc` расширяет его **вычислительным контрактом** (агрегации, группировки, джойны). Каждый бэкенд реализует оба.

### 2.2. `PandasDataset`

Обёртка над `pd.DataFrame`. Все операции выполняются **немедленно** (eager) в памяти драйвера.

Особенности:
- `random_split_labels()` использует `pandas.util.hash_array` (SipHash) для детерминированного хэширования индекса.
- `checkpoint()` — **no-op** (Pandas не страдает от роста DAG).
- `_data_compression()` поддерживает сжатие при конвертации из Spark: `downcasting` (float64→float32) и `encoding` (StringIndexer для категориальных).

### 2.3. `SparkDataset`

Обёртка над `pyspark.pandas.DataFrame` (Koalas API поверх `pyspark.sql.DataFrame`). Все операции **ленивые** — формируют DAG, исполняются только при вызове действия (`.count()`, `.collect()`, `.to_pandas()`).

Особенности:
- `_ops_on_diff_frames()` — контекстный менеджер, временно включающий `compute.ops_on_diff_frames` для операций между DataFrame из разных источников.
- `_check_pandas_conversion()` — **защита от OOM**: блокирует `to_pandas()` при `> DatasetConfig.SPARK_PANDAS_CONVERSION_LIMIT` строк.
- `persist()` / `unpersist()` — управление кэшированием в Spark.
- `checkpoint()` — материализация через `spark.checkpoint()` или `local_checkpoint()` с предупреждением.
- `random_split_labels()` использует `F.hash()` (Murmur3) — **другой хэш**, чем в Pandas, поэтому точное распределение строк по группам может отличаться между бэкендами для одного и того же `random_state`. Детерминизм гарантирован **внутри** каждого бэкенда.
- `iter_groups()` — итерация по группам с сортировкой по ключу для детерминированного baseline.

### 2.4. Конвертация между бэкендами

```python
# Pandas → Spark (индекс сохраняется через временную колонку)
spark_ds = pandas_ds.to_backend(BackendsEnum.spark, session=spark_session)

# Spark → Pandas (с проверкой лимита строк)
pandas_ds = spark_ds.to_backend(BackendsEnum.pandas)
```

Механизм сохранения индекса при `pandas → spark`:

```
pd.DataFrame
  → reset_index() с именем "__hypex_temp_index__"
  → session.createDataFrame(pdf)
  → ps.DataFrame(spark_df)
  → set_index("__hypex_temp_index__")
  → index.name = None
```

### 2.5. `persist` / `unpersist` / `checkpoint`

| Метод | Что делает | Когда использовать |
|-------|------------|-------------------|
| `persist(storage_level)` | Кэширует DataFrame в памяти/диске | Перед многократным чтением (итерации, ветвления) |
| `unpersist(blocking)` | Освобождает кэш | После завершения работы с данными |
| `checkpoint(eager)` | Материализует и обрезает DAG | В итеративных циклах для предотвращения экспоненциального роста плана |

`checkpoint()` критичен для Spark: без него каждая итерация `ParamsExperiment` добавляет новый уровень в DAG, и к 100-й итерации план становится неуправляемым. `checkpoint()` создаёт физическую точку восстановления и обрезает линейку.

```python
# Автоматический checkpoint в ExperimentData._clean_ds_for_iteration()
if cleaned.backend_type == BackendsEnum.spark:
    cleaned.checkpoint(eager=True)
```

---

## 3. Система ролей

### 3.1. Философия

Роли отвечают на вопрос **«что эта колонка значит для пайплайна?»** вместо **«как она называется?»**. Пайплайн не ищет колонку `"revenue"` — он ищет колонку с ролью `TargetRole(float)`. Это позволяет:

- Переименовывать колонки без изменения пайплайна.
- Один и тот же пайплайн запускать на разных датасетах.
- Динамически менять назначение колонки (например, `TempTargetRole` в итеративном цикле).

### 3.2. Базовый класс `ABCRole`

```python
class ABCRole(ABC):
    _role_name: RoleNameType = "Abstract"

    def __init__(self, data_type: DefaultRoleTypes | None = None):
        self.data_type = data_type

    def astype(self, data_type) -> ABCRole: ...       # копия с новым типом
    def asadditional(self, data_type) -> ABCRole: ...  # поиск Additional-варианта
```

`data_type` используется в `search_columns(search_types=[int, float])` для фильтрации колонок по типу. `asadditional()` ищет в `default_roles` класс, наследующий одновременно от текущей роли и от `AdditionalRole`.

### 3.3. Полная иерархия ролей

```
ABCRole
│
├── IndexRole              — идентификатор строки
├── InfoRole               — вспомогательная колонка (не участвует в анализе)
├── DisabledRole           — «мягко» удалённая колонка (хранит initial_role)
├── StratificationRole     — стратификация при сплиттинге
├── GroupingRole           — группировка / сегментация
├── TreatmentRole          — столбец обработки / группы
├── ConstGroupRole         — закреплённая (нерандомизированная) группа
├── StatisticRole          — вычисленная статистика
├── ResumeRole             — итоговый отчёт
├── FilterRole             — булева маска для фильтрации
├── DefaultRole            — fallback для колонок без роли
├── ReportRole             — колонка для отчётов
│
├── TargetRole             — целевая метрика (+ cofounders)
│
├── LagRole (ABCRole)      — база для временных зависимостей (parent, lag)
│   ├── FeatureRole        — признак / предиктор
│   └── PreTargetRole      — пре-таргет для CUPED (+ cofounders)
│
├── TempRole               — база для временных ролей
│   ├── TempTargetRole     — временный таргет (итерация OnRoleExperiment)
│   ├── TempTreatmentRole  — временный treatment
│   └── TempGroupingRole   — временная группировка
│
└── AdditionalRole         — база для синтетических / производных колонок
    ├── AdditionalTreatmentRole   — доп. столбец обработки (сплит-метка)
    ├── AdditionalGroupingRole    — доп. группировка
    ├── AdditionalTargetRole      — доп. таргет (CUPED/CUPAC)
    ├── AdditionalFeatureRole     — доп. признак
    ├── AdditionalPreTargetRole   — доп. пре-таргет
    ├── AdditionalMatchingRole    — индексы матчинга (FAISS)
    ├── AdditionalVarianceReductionRole — CUPED θ
    └── AdditionalStatisticRole   — доп. статистика (bias)
```

### 3.4. Назначение каждой роли

| Роль | Кто ставит | Кто читает | Пример |
|------|-----------|------------|--------|
| `TargetRole` | Пользователь | `OnRoleExperiment`, `StatsComparator` | `revenue`, `spends` |
| `FeatureRole` | Пользователь | `MahalanobisDistance`, `FaissNearestNeighbors`, `DummyEncoder` | `age`, `gender` |
| `TreatmentRole` | Пользователь | `GroupSizes`, `GroupDifference`, `TTest` | `treat` |
| `GroupingRole` | Пользователь / пайплайн | `GroupsComparator._get_fields_data()` | `segment` |
| `PreTargetRole` | Пользователь | `CUPEDTransformer` | `revenue_lag_1` |
| `StratificationRole` | Пользователь | `AASplitterWithStratification` | `industry` |
| `ConstGroupRole` | Пользователь | `AASplitter` (закреплённые группы) | `fixed_group` |
| `AdditionalTreatmentRole` | `AASplitter._set_value()` | `GroupSizes`, `TTest` в A/A | `AASplitter┆rs 0┆` |
| `AdditionalMatchingRole` | `FaissNearestNeighbors._set_value()` | `Bias`, `MatchingMetrics` | `FaissNearestNeighbors┆┆0` |
| `AdditionalTargetRole` | `CUPEDTransformer`, `CUPACExecutor` | `OnRoleExperiment` (векторная ветка) | `revenue_cuped` |
| `DisabledRole` | `DummyEncoder._disable_target_cols()` | `ExperimentData.cleanup_additional()` | `gender` (после one-hot) |
| `TempTargetRole` | `OnRoleExperiment.execute()` | `BaseComparator._get_fields_data()` | текущая колонка итерации |
| `InfoRole` | По умолчанию | — | `user_id` |

### 3.5. `tmp_roles` vs `roles`

Это ключевой механизм для **итеративных пайплайнов**:

```
┌─────────────────────────────────────────────────────────────────┐
│  roles (постоянные)                                             │
│  ┌───────────┬───────────┬───────────┬───────────┐              │
│  │ revenue   │ spends    │ treat     │ age       │              │
│  │ Target    │ Target    │ Treatment │ Feature   │              │
│  └───────────┴───────────┴───────────┴───────────┘              │
│                                                                 │
│  Итерация 1: OnRoleExperiment устанавливает                     │
│  tmp_roles = {"revenue": TempTargetRole()}                      │
│  → Comparator видит только "revenue" как таргет                 │
│                                                                 │
│  Итерация 2:                                                    │
│  tmp_roles = {"spends": TempTargetRole()}                       │
│  → Comparator видит только "spends" как таргет                  │
│                                                                 │
│  После всех итераций:                                           │
│  tmp_roles = {}  (сброс)                                        │
└─────────────────────────────────────────────────────────────────┘
```

`BaseComparator._get_fields_data()` проверяет наличие `tmp_roles`:

```python
def _get_fields_data(self, data: ExperimentData) -> dict[str, Dataset]:
    tmp_role = bool(data.ds.tmp_roles or data.additional_fields.tmp_roles)
    target_fields_data = data.field_data_search(
        roles=(
            (TempTargetRole() if data.ds.tmp_roles else AdditionalTargetRole())
            if tmp_role
            else self.target_roles
        ),
        tmp_role=tmp_role,
        ...
    )
```

### 3.6. Реестр `default_roles`

Глобальный словарь для строковой инициализации ролей:

```python
default_roles: dict[RoleNameType, ABCRole] = {
    "info": InfoRole(),
    "default": DefaultRole(),
    "feature": FeatureRole(),
    "treatment": TreatmentRole(),
    "grouping": GroupingRole(),
    "target": TargetRole(),
    "pretarget": PreTargetRole(),
    "stratification": StratificationRole(),
    "statistic": StatisticRole(),
    "index": IndexRole(),
    "filter": FilterRole(),
    "constgroup": ConstGroupRole(),
    "additionaltreatment": AdditionalTreatmentRole(),
    ...
}
```

Используется в `DatasetBase._parse_roles()` для нормализации строковых ключей.

### 3.7. Поиск колонок по роли

```python
# Все числовые таргеты
targets = ds.search_columns(TargetRole(), search_types=[int, float])
# → ["revenue", "spends"]

# Все категориальные признаки
cats = ds.search_columns(FeatureRole(), search_types=[str])
# → ["gender", "industry"]

# Все Additional-колонки
add_cols = ds.search_columns(AdditionalRole())

# С учётом временных ролей
tmp = ds.search_columns(TempTargetRole(), tmp_role=True)
```

---

## 4. `ExperimentData` — единый контейнер состояния

### 4.1. Четыре пространства хранения

```
ExperimentData
├── ds (property → _data)                    # Основной датасет
│   └── Все колонки, включая синтетические (Additional*Role)
│
├── variables: dict[str, dict]               # Скаляры, матрицы, модели
│   └── "MahalanobisDistance┆┆" → {"": Dataset(матрица)}
│
├── groups: dict[str, dict[str, Dataset]]    # Разрезы по группам
│   └── "treat" → {"0": ds_control, "1": ds_test}
│
└── analysis_tables: dict[str, SmallDataset] # Результаты тестов
    ├── "StatsTTest┆hash┆revenue" → SmallDataset(p-value, statistic, pass)
    ├── "GroupSizes┆hash┆treat"   → SmallDataset(control size, test size, ...)
    └── "OneAAStatAnalyzer┆┆"     → SmallDataset(mean p-value, mean test score)
```

### 4.2. `_initial_cols` и `additional_fields`

При создании `ExperimentData` фиксируются колонки исходного датасета:

```python
self._initial_cols = deepcopy(self._data.columns)
```

Свойство `additional_fields` — **read-only view** колонок, которых нет в `_initial_cols`:

```python
@property
def additional_fields(self) -> Dataset | SmallDataset:
    initial = set(self._initial_cols)
    additional_cols = [c for c in self._data.columns if c not in initial]
    if not additional_cols:
        return self._data.create_empty(...)
    view = self._data[additional_cols]
    view.roles = {c: self._data.roles[c] for c in additional_cols}
    return view
```

Это позволяет:
- `field_search(space="ds")` — искать только в исходных колонках.
- `field_search(space="additional_fields")` — искать только в синтетических.
- `field_search(space="all")` — искать во всех.

### 4.3. `set_value()` — маршрутизация по пространствам

```python
def set_value(self, space, executor_id, value, key=None, role=None) -> Self:
    if space == ExperimentDataEnum.additional_fields:
        return self._set_additional_fields(exec_id, value, role)
    if space == ExperimentDataEnum.analysis_tables:
        return self._set_analysis_tables(exec_id, value)
    if space == ExperimentDataEnum.variables:
        return self._set_variables(exec_id, value, key)
    if space == ExperimentDataEnum.groups:
        return self._set_groups(exec_id, value, key)
```

`_set_additional_fields()` — самый сложный: добавляет колонку прямо в `ds`, обрабатывая три случая:
1. **Скаляр / список** → `add_column()` с именем `exec_id`.
2. **Одноколоночный Dataset** → извлекает колонку, добавляет с именем `exec_id`.
3. **Многоколоночный Dataset** → переименовывает колонки в `{exec_id}_{col}`, делает `merge()`.

### 4.4. `get_ids()` — поиск результатов по классу

```python
ids = data.get_ids(StatsTTest, ExperimentDataEnum.analysis_tables)
# → {"StatsTTest": {"analysis_tables": ["StatsTTest┆abc┆revenue", ...]}}
```

Парсит составные ID по `ID_SPLIT_SYMBOL` (`┴`): первая часть — имя класса, последняя — ключ. Фильтрует по имени класса и опциональному суффиксу `key`.

### 4.5. `field_search()` и `field_data_search()`

`field_search()` возвращает **список имён колонок**:

```python
targets = data.field_search(TargetRole(), search_types=[int, float])
# → ["revenue", "spends"]
```

`field_data_search()` возвращает **новый Dataset** с только найденными колонками. Критическая оптимизация для Spark: вместо создания пустого датасета с полным 60M-строчным индексом (что вызывает `toPandas()` на драйвере), выполняется **чистая проекция колонок** `self._data[cols]` — ленивая Spark-трансформация без материализации:

```python
def field_data_search(self, roles, tmp_role=False, search_types=None, space="all"):
    ...
    if not cols:
        return Dataset.create_empty(roles={}, backend=..., session=...)
    subset_ds = self._data[cols]  # O(1) Spark transformation
    for col, role in new_roles.items():
        subset_ds.roles[col] = role
    return subset_ds
```

### 4.6. `_clean_ds_for_iteration()` — изоляция итераций

Вызывается перед каждой итерацией `ParamsExperiment`, `CycledExperiment`, `GroupExperiment`:

```python
def _clean_ds_for_iteration(self) -> Dataset | SmallDataset:
    additional_cols = [
        col for col, role in self._data.roles.items()
        if isinstance(role, AdditionalRole)
    ]
    if not additional_cols:
        return self._data
    cleaned = self._data.drop(columns=additional_cols)
    if cleaned.backend_type == BackendsEnum.spark:
        cleaned.checkpoint(eager=True)
    return cleaned
```

Удаляет все синтетические колонки (сплит-метки, матчинг-индексы, CUPED-таргеты) и обрезает DAG. Без этого каждая итерация «видела» бы артефакты предыдущей.

### 4.7. `cleanup_additional()` — финальная очистка

Вызывается в конце `Output.extract()` для удаления внутренних колонок из пользовательского результата. Также восстанавливает роли `DisabledRole` → `initial_role`:

```python
def cleanup_additional(self) -> Self:
    cols_to_drop = [col for col, role in self._data.roles.items()
                    if isinstance(role, AdditionalRole)]
    cols_to_enable = self._data.search_columns(DisabledRole())
    if cols_to_drop:
        self._data = self._data.drop(columns=cols_to_drop)
    if cols_to_enable:
        self._data = self._data.replace_roles(
            {col: self._data.roles[col].initial_role for col in cols_to_enable}
        )
    return self
```

### 4.8. `copy()` — глубокое копирование с синхронизацией

```python
def copy(self, data=None) -> Self:
    result = deepcopy(self)
    if data is not None:
        result._data = data
        result._initial_cols = deepcopy(data.columns)  # синхронизация!
    return result
```

Без обновления `_initial_cols` колонки, удалённые из `data` (например, после CUPAC-очистки), оставались бы в `_initial_cols`, вызывая `KeyError` в `field_search(space="ds")`.

---

## 5. `GroupedDataset` — ленивая группировка

### 5.1. Назначение

`GroupedDataset` — обёртка над `groupby()` бэкенда, предоставляющая **единый интерфейс агрегации** и **итерации по группам**. Создаётся через `Dataset.groupby()`:

```python
grouped = ds.groupby("treat")
```

### 5.2. Агрегации

| Метод | Эквивалент |
|-------|-----------|
| `agg("mean")` | `groupby().agg("mean")` |
| `agg(["mean", "std"])` | Мультиагрегация с `NAME_BORDER_SYMBOL` в именах |
| `agg({col: "sum"})` | Поколоночная агрегация |
| `value_counts()` | Частоты категорий по группам |
| `count()`, `sum()`, `mean()`, `min()`, `max()`, `std()`, `var()`, `median()`, `prod()`, `first()`, `last()`, `size()` | Шорткаты |

### 5.3. Итерация по группам

```python
for group_key, group_ds in ds.groupby("treat"):
    print(group_key, len(group_ds))
```

`__iter__` делегирует в `backend_data.iter_groups()`:
- **Pandas**: `data.groupby(by, observed=False)` — группы отсортированы по ключу.
- **Spark**: собирает уникальные ключи на драйвер (`.drop_duplicates().to_pandas().sort_values()`), затем фильтрует по маске для каждой группы. Это гарантирует **детерминированный порядок** групп, что критично для выбора baseline (первая группа = контроль).

### 5.4. `value_counts()` по группам

Специализированный метод, возвращающий `Dataset` с колонками `{category: count}` по группам:

```python
grouped_vc = ds.groupby("treat").value_counts()
# → Dataset с ролями {col: StatisticRole()}
```

Для Spark реализован через `groupBy(*by, col).count()` → `map_from_entries(collect_list(struct(col, count)))` — одна агрегация на все категории.

### 5.5. Роли после агрегации

`_get_agg_roles()` сохраняет исходные роли для колонок, которые были в `groupby`, и назначает `StatisticRole()` для новых агрегатных колонок:

```python
def _get_agg_roles(self, result_columns):
    new_roles = {}
    for col in result_columns:
        if col in self.roles:
            new_roles[col] = deepcopy(self.roles[col])
        else:
            new_roles[col] = StatisticRole()
    return new_roles
```

---

## 6. Примеры использования

### 6.1. Создание датасета с ролями

```python
from hypex.dataset import Dataset, TargetRole, TreatmentRole, FeatureRole, InfoRole

ds = Dataset(
    roles={
        TargetRole(float): ["revenue", "spends"],
        TreatmentRole(): "treat",
        FeatureRole(): ["age", "gender", "industry"],
        InfoRole(): "user_id",
    },
    data=pd.read_csv("experiment_data.csv"),
)

print(ds.search_columns(TargetRole(), search_types=[int, float]))
# → ["revenue", "spends"]

print(ds.search_columns(FeatureRole(), search_types=[str]))
# → ["gender", "industry"]
```

### 6.2. Конвертация в Spark и обратно

```python
from hypex.utils import BackendsEnum

# Pandas → Spark
spark_ds = ds.to_backend(BackendsEnum.spark, session=spark)
spark_ds.persist()  # кэшируем перед многократным использованием

# ... выполняем пайплайн ...

# Spark → Pandas (только если строк ≤ 100_000)
pandas_result = spark_ds.to_backend(BackendsEnum.pandas)
```

### 6.3. Работа с `ExperimentData`

```python
from hypex.dataset import ExperimentData, ExperimentDataEnum

ed = ExperimentData(ds)

# Сохранить результат теста
ed.set_value(
    ExperimentDataEnum.analysis_tables,
    "TTest┆abc123┆revenue",
    SmallDataset.from_dict(
        {"p-value": [0.03], "statistic": [2.14], "pass": [True]},
        StatisticRole(),
    ),
)

# Найти все результаты TTest
ids = ed.get_ids("TTest", ExperimentDataEnum.analysis_tables)

# Найти таргет-колонки
targets = ed.field_search(TargetRole(), search_types=[int, float])

# Получить поднабор данных по ролям
subset = ed.field_data_search(TargetRole(), search_types=[int, float])
```

### 6.4. Итеративный пайплайн с `tmp_roles`

```python
# Внутри OnRoleExperiment.execute():
for field in target_fields:
    data.ds.tmp_roles = {field: TempTargetRole()}
    data = super().execute(data)  # Comparator видит только `field`
    data.ds.tmp_roles = {}        # сброс после итерации
```

---

## 7. Контракт для нового бэкенда

Для добавления третьего бэкенда (например, Polars) необходимо:

1. **Реализовать `DatasetBackendNavigation`** (~30 методов): `__getitem__`, `__len__`, `loc`, `iloc`, `create_empty`, `from_dict`, `to_dict`, `add_column`, `append`, `merge`, `drop`, `filter`, `astype`, `get_column_type`, `to_backend`, `random_split_labels`, ...

2. **Реализовать `DatasetBackendCalc`** (~40 методов): `mean`, `var`, `std`, `sum`, `count`, `agg`, `groupby`, `iter_groups`, `count_groups`, `value_counts`, `grouped_value_counts`, `corr`, `dot`, `fillna`, `dropna`, `isna`, `quantile`, `sample`, `sort_values`, `sort_index`, `merge`, `checkpoint`, ...

3. **Зарегистрировать в `DatasetBase._select_backend_from_data()`** и `BackendsEnum`.

4. **Добавить бэкенд-специфичные реализации** в `backend_factory` для каждого `Extension` и `Comparator`.

---

*Документация актуальна для версии **2.0.0a1**. Следующий раздел — «Конвейер исполнения: Executor и Experiment».*
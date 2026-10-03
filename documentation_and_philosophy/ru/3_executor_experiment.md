# Конвейер исполнения: Executor и Experiment

---

## 1. Иерархия Executor'ов

Конвейер построен на одном фундаментальном паттерне: **каждый блок принимает `ExperimentData` и возвращает `ExperimentData`**. Никаких побочных каналов, никаких глобальных переменных — всё состояние проходит через один контейнер.

```
Executor (ABC)
├── Calculator
│   ├── GroupsComparator      — сравнение на сырых данных (5 режимов)
│   │   └── GroupHypothesisTesting
│   │       ├── GroupTTest
│   │       ├── GroupKSTest
│   │       ├── GroupUTest
│   │       └── GroupChi2Test
│   ├── StatsComparator       — двухфазное: агрегация → сравнение
│   │   ├── StatsHypothesisTesting
│   │   │   ├── StatsTTest
│   │   │   ├── StatsChi2Test
│   │   │   ├── StatsZTest
│   │   │   ├── StatsKSTest
│   │   │   └── StatsUTest
│   │   ├── GroupDifference
│   │   └── GroupSizes
│   ├── MLExecutor            — fit/predict, группировка по TreatmentRole
│   │   ├── FaissNearestNeighbors
│   │   └── CUPACExecutor
│   ├── Transformer           — _is_transformer=True, копирует данные
│   │   ├── NaDropper / NaFiller
│   │   ├── TypeCaster / Float32Caster
│   │   ├── CUPEDTransformer
│   │   ├── CategoryAggregator
│   │   ├── CorrFilter / CVFilter / NanFilter / ConstFilter / OutliersFilter
│   │   └── Shuffle
│   ├── Encoder               — кодирование категориальных колонок
│   │   └── DummyEncoder
│   ├── GroupOperator         — оператор на двух таргетах
│   │   ├── MatchingMetrics
│   │   ├── Bias
│   │   └── SMD
│   ├── MahalanobisDistance   — расчёт ковариационной матрицы
│   └── MinSampleSize         — оценка размера выборки
├── IfExecutor                — условное исполнение
│   └── IfAAExecutor          — критерии остановки A/A
└── Experiment                — последовательный пайплайн из executors
    ├── OnRoleExperiment      — итерация по ролям
    └── ExperimentWithReporter
        ├── CycledExperiment
        ├── GroupExperiment
        ├── ParamsExperiment
        │   └── IfParamsExperiment
        └── (наследники с конкретными репортёрами)
```

### 1.1. Базовый контракт `Executor`

```python
class Executor(ABC):
    def __init__(self, key: Any = "", **calc_kwargs):
        self._id: str = ""
        self._params_hash = ""
        self.key: Any = key
        self._generate_id()
        self.calc_kwargs = calc_kwargs

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData: ...
```

Каждый `Executor`:
- Имеет **уникальный составной идентификатор** `id`.
- Хранит **параметры вычислений** `calc_kwargs` для передачи в `_inner_function`.
- Реализует `execute()` — единственную точку входа.

### 1.2. `Calculator` — чистое вычисление

`Calculator` добавляет разделение на **чистую функцию** (`_inner_function`) и **обёртку** (`calc`):

```python
class Calculator(Executor, ABC):
    @classmethod
    def calc(cls, data: Dataset, **kwargs):
        return cls._inner_function(data, **kwargs)

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Any: ...

    @property
    def search_types(self):
        raise AbstractMethodError
```

Это позволяет вызывать вычисление **вне пайплайна**:

```python
result = GroupTTest.calc(data=ds_revenue, reliability=0.05)
```

### 1.3. `MLExecutor` — машинное обучение

`MLExecutor` расширяет `Calculator` для задач с этапами `fit` / `predict`:

```python
class MLExecutor(Calculator, ABC):
    @abstractmethod
    def fit(self, X: Dataset, Y: Dataset | None = None) -> MLExecutor: ...

    @abstractmethod
    def predict(self, X: Dataset) -> Dataset: ...

    @property
    def search_types(self):
        return [int, float]
```

Особенность хранения результатов: каждый столбец предсказания сохраняется в `additional_fields` с ролью `AdditionalMatchingRole`:

```python
def _set_value(self, data, value, key=None):
    for i in range(value.shape[1]):
        data.set_value(
            ExperimentDataEnum.additional_fields,
            f"{self.id}{ID_SPLIT_SYMBOL}{i}",
            value=value.iloc[:, i],
            role=AdditionalMatchingRole(),
        )
    return data
```

### 1.4. `IfExecutor` — условное исполнение

```python
class IfExecutor(Executor, ABC):
    def __init__(self, if_executor=None, else_executor=None, key=""):
        ...

    @abstractmethod
    def check_rule(self, data, **kwargs) -> bool: ...

    def execute(self, data: ExperimentData) -> ExperimentData:
        if self.check_rule(data):
            return self.if_executor.execute(data) if self.if_executor else self._set_value(data, True)
        return self.else_executor.execute(data) if self.else_executor else self._set_value(data, False)
```

Конкретный пример — `IfAAExecutor` (критерий остановки в A/A-тесте):

```python
class IfAAExecutor(IfExecutor):
    def check_rule(self, data, **kwargs) -> bool:
        if self.all_features_passed:
            return self._count_feature_pass(data) == 0
        if self.sample_size is not None:
            return self._count_feature_pass(data) >= 1
        return False
```

Результат (`True` / `False`) сохраняется в `variables` пространстве.

### 1.5. `Transformer` — мутация данных

```python
class Transformer(Calculator):
    @property
    def _is_transformer(self):
        return True

    def execute(self, data: ExperimentData) -> ExperimentData:
        data = data.copy(data=self.calc(data=data.ds))
        return data
```

Ключевое отличие: `Transformer` **заменяет датасет** внутри `ExperimentData` через `copy(data=new_ds)`. Это приводит к **глубокому копированию** на уровне `Experiment`, если `transformer=True`:

```python
# Experiment.execute()
experiment_data = deepcopy(data) if self.transformer else data
```

---

## 2. Идентификация и параметры

### 2.1. Составной идентификатор

Каждый `Executor` получает уникальный `id` по формуле:

```
id = ClassName ┆ params_hash ┆ key
```

где `┆` — это `ID_SPLIT_SYMBOL` (`"\u2534"` — символ Unicode "BOX DRAWINGS LIGHT UP AND HORIZONTAL").

```python
def _generate_id(self):
    self._generate_params_hash()
    self._id = ID_SPLIT_SYMBOL.join([
        self.__class__.__name__,
        self._params_hash.replace(ID_SPLIT_SYMBOL, "|"),
        str(self._key).replace(ID_SPLIT_SYMBOL, "|"),
    ])
```

Примеры реальных ID:
```
StatsTTest┆┆revenue
GroupSizes┆┆treat
AASplitter┆cs 0.5|rs 42┆
FaissNearestNeighbors┆┆matched┆0
```

### 2.2. `set_params()` — изменение параметров

Поддерживает два формата:

```python
# Формат 1: {имя атрибута: dict}
executor.set_params({"random_state": 42, "control_size": 0.3})

# Формат 2: {класс: {имя: значение}}
experiment.set_params({
    TTest: {"calc_kwargs": {"equal_variance": True}},
    Comparator: {"grouping_role": [AdditionalTreatmentRole()]},
})
```

После каждого вызова `_generate_id()` пересобирает идентификатор.

### 2.3. `build_from_id()` — восстановление из строки

```python
@classmethod
def build_from_id(cls, executor_id: str):
    splitted_id = executor_id.split(ID_SPLIT_SYMBOL)
    if splitted_id[0] != cls.__name__:
        raise ValueError(f"{executor_id} is not a valid {cls.__name__} id")
    result = cls()
    result.init_from_hash(splitted_id[1])
    return result
```

Используется в `AAScoreAnalyzer.build_splitter_from_id()` для восстановления лучшего сплиттера из сохранённого результата.

### 2.4. `get_params()` — интроспекция параметров

Метод зеркалит `scikit-learn` API:

```python
def get_params(self, deep: bool = False):
    init_params = inspect.signature(self.__init__).parameters
    out = {}
    for name, param in init_params.items():
        if name == "self":
            continue
        value = getattr(self, name, param.default)
        out[name] = value
    return out
```

---

## 3. Типы Experiment'ов

### 3.1. `Experiment` — последовательный пайплайн

```python
class Experiment(Executor):
    def __init__(self, executors: Sequence[Executor], transformer: bool | None = None, key: Any = ""):
        self.executors = executors
        self.transformer = transformer if transformer is not None else self._detect_transformer()
        super().__init__(key)
        self.logger = HypExLogger(name="hypex.experiment")
```

`_detect_transformer()` сканирует список: если хотя бы один `executor._is_transformer == True`, весь пайплайн помечается как трансформер.

#### Механизм `execute()`:

```python
@timeit(level="PIPELINE", prefix="EXPERIMENT")
def execute(self, data: ExperimentData) -> ExperimentData:
    self.logger.log_spark_info()
    experiment_data = deepcopy(data) if self.transformer else data

    for executor in self.executors:
        with self.logger.process(
            name=executor.__class__.__name__,
            backend=experiment_data.ds.backend_type.value,
        ):
            cur_executor = self._get_executor_backend(executor, experiment_data.ds)
            cur_executor.key = self.key
            experiment_data = cur_executor.execute(experiment_data)

    return experiment_data
```

### 3.2. `OnRoleExperiment` — итерация по колонкам роли

Самый сложный тип: выполняет набор `Executor`'ов **для каждой колонки**, соответствующей заданной роли.

```python
class OnRoleExperiment(Experiment):
    def __init__(self, executors, role, transformer=None, key=""):
        self.role = [role] if isinstance(role, ABCRole) else list(role)
        super().__init__(executors, transformer, key)
```

#### Разделение на векторные и итеративные исполнители:

```python
def execute(self, data: ExperimentData) -> ExperimentData:
    target_fields = data.field_search(self.role)
    vector_executors = []
    iterative_executors = []
    _backend = data.ds.backend_type

    for ex in self.executors:
        if isinstance(ex, (StatsComparator, StatsHypothesisTesting)):
            vector_executors.append(ex)
        elif isinstance(ex, StatTestMasterAbstract):
            if _backend == BackendsEnum.spark:
                vector_executors.append(ex)      # мастер → Stats* на Spark
            else:
                iterative_executors.append(ex)   # мастер → Group* на Pandas
        else:
            iterative_executors.append(ex)
```

**Векторная ветка** — все таргеты за один проход:

```python
if vector_executors:
    tmp_roles_dict = {field: TempTargetRole() for field in target_fields if field in data.ds.columns}
    data.ds.tmp_roles = tmp_roles_dict
    self.executors = vector_executors
    data = super().execute(data)
    data.ds.tmp_roles = {}
```

**Итеративная ветка** — по одной колонке:

```python
if iterative_executors:
    self.executors = iterative_executors
    for field in target_fields:
        data.ds.tmp_roles = {field: TempTargetRole()}
        data = super().execute(data)
        data.ds.tmp_roles = {}
```

### 3.3. `ExperimentWithReporter` — базовый класс с отчётами

```python
class ExperimentWithReporter(Experiment):
    def __init__(self, executors, reporter, transformer=None, key=""):
        super().__init__(executors, transformer, key)
        self.reporter = reporter

    def one_iteration(self, data, key="", set_key_as_index=False):
        t_data = ExperimentData(data._clean_ds_for_iteration())
        self.key = key
        t_data = super().execute(t_data)
        result = self.reporter.report(t_data)
        if set_key_as_index:
            result.index = [key]
        return result
```

### 3.4. `ParamsExperiment` — sweep по декартову произведению параметров

```python
class ParamsExperiment(ExperimentWithReporter):
    def __init__(self, executors, reporter, params, transformer=None, stopping_criterion=None, key=""):
        super().__init__(executors, reporter, transformer, key)
        self._params = params
        self._update_flat_params()
        self.stopping_criterion = stopping_criterion
```

`_update_flat_params()` вычисляет **декартово произведение**:

```python
def _update_flat_params(self):
    classes = list(self._params)
    param_combinations = [
        list(product(*[product([parameter], values) for parameter, values in class_params.items()]))
        for class_params in self._params.values()
    ]
    self._flat_params = [
        {classes[i]: dict(param_combination[i]) for i in range(len(param_combination))}
        for param_combination in product(*param_combinations)
    ]
```

#### Цикл исполнения:

```python
@timeit(level="PIPELINE", prefix="PARAMS")
def execute(self, data: ExperimentData) -> ExperimentData:
    results = []
    self._update_flat_params()
    clean_ds = data._clean_ds_for_iteration()  # ← один раз вне цикла

    for flat_param in tqdm(self._flat_params):
        t_data = ExperimentData(clean_ds)
        for executor in self.executors:
            executor.set_params(flat_param)
            t_data = executor.execute(t_data)
        report = self.reporter.report(t_data)
        results.append(report)
        if self._stopping_criterion_met(t_data):
            break

    return self._set_result(data, results)
```

### 3.5. `IfParamsExperiment` — ранний выход

```python
class IfParamsExperiment(ParamsExperiment):
    def execute(self, data):
        clean_ds = data._clean_ds_for_iteration()
        for flat_param in tqdm(self._flat_params):
            t_data = ExperimentData(clean_ds)
            for executor in self.executors:
                cur_executer = self._get_executor_backend(executor, t_data)
                cur_executer.set_params(flat_param)
                t_data = cur_executer.execute(t_data)

            if_result = self.stopping_criterion.execute(t_data)
            if_executor_id = if_result.get_one_id(...)
            if if_result.variables[if_executor_id]["response"]:
                return self._set_result(data, [self.reporter.report(t_data)])
        return data
```

Отличие от `ParamsExperiment`: при срабатывании критерия **немедленно возвращает** результат текущей итерации, не продолжая цикл.

### 3.6. `GroupExperiment` — исполнение по группам

```python
class GroupExperiment(ExperimentWithReporter):
    def execute(self, data):
        group_field = data.ds.search_columns(self.searching_role)
        clean_ds = data._clean_ds_for_iteration()
        results = []
        for group, group_data in tqdm(clean_ds.groupby(group_field)):
            key = str(group[0] if isinstance(group, tuple) else group)
            self.key = key
            t_data = ExperimentData(group_data)
            t_data = super(ExperimentWithReporter, self).execute(t_data)
            report = self.reporter.report(t_data)
            results.append((key, report))
        return self._set_result(data, results)
```

Результаты каждой группы переименовываются с префиксом `{group} ` и объединяются через `merge()`.

### 3.7. `CycledExperiment` — N итераций

```python
class CycledExperiment(ExperimentWithReporter):
    @timeit(level="PIPELINE", prefix="CYCLED")
    def execute(self, data):
        clean_ds = data._clean_ds_for_iteration()
        result = []
        for i in tqdm(range(self.n_iterations)):
            self.key = str(i)
            t_data = ExperimentData(clean_ds)
            t_data = super(ExperimentWithReporter, self).execute(t_data)
            report = self.reporter.report(t_data)
            report.index = [str(i)]
            result.append(report)
        return self._set_result(data, result)
```

---

## 4. Механизм `_get_executor_backend`

### 4.1. Резолвинг мастер-класса в бэкенд-реализацию

```python
@staticmethod
def _get_executor_backend(executor: Executor, ds: Dataset):
    executor_cls = type(executor)                              # TTest
    backend_cls = backend_factory.resolve_backend(executor_cls, ds)  # GroupTTest или StatsTTest
    if backend_cls is None:
        return executor

    sig = inspect.signature(backend_cls.__init__)
    expected_params = {p.name for p in sig.parameters.values() if p.name != "self"}
    init_kwargs = {k: getattr(executor, k) for k in expected_params if hasattr(executor, k)}

    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword and hasattr(executor, "calc_kwargs"):
        init_kwargs.update(executor.calc_kwargs)

    new_executor = backend_cls(**init_kwargs)
    if hasattr(executor, "key"):
        new_executor.key = executor.key
    return new_executor
```

### 4.2. Что происходит шаг за шагом:

1. **Определение класса**: `type(executor)` → `TTest` (мастер-класс).
2. **Поиск реализации**: `backend_factory.resolve_backend(TTest, ds)` → ищет в реестре по `type(ds.backend_data)` (`PandasDataset` или `SparkDataset`).
3. **Если не найдено**: возвращает исходный `executor` без изменений.
4. **Перенос параметров**: через `inspect.signature()` извлекает ожидаемые параметры конструктора бэкенд-класса и копирует соответствующие атрибуты из мастер-экземпляра.
5. **Разворачивание `calc_kwargs`**: если конструктор принимает `**kwargs`, все ключи из `calc_kwargs` (например, `equal_variance=True`) передаются как именованные аргументы.
6. **Синхронизация `key`**: чтобы результаты сохранялись под тем же идентификатором.

### 4.3. Пример разрешения

| Мастер-класс | Бэкенд данных | Реализация |
|---|---|---|
| `TTest` | `PandasDataset` | `GroupTTest` (scipy t-test на сырых данных) |
| `TTest` | `SparkDataset` | `StatsTTest` (агрегация → t-статистика на драйвере) |
| `KSTest` | `PandasDataset` | `GroupKSTest` (scipy ks_2samp) |
| `KSTest` | `SparkDataset` | `StatsKSTest` (гистограммы → KS-статистика) |
| `Chi2Test` | `PandasDataset` | `GroupChi2Test` (scipy chi2_contingency) |
| `Chi2Test` | `SparkDataset` | `StatsChi2Test` (value_counts → contingency) |
| `UTest` | `PandasDataset` | `GroupUTest` (scipy mannwhitneyu) |
| `UTest` | `SparkDataset` | `StatsUTest` (гистограммы → U-статистика) |
| `ZTest` | `SparkDataset` | `StatsZTest` (пропорции из count/sum) |

---

## 5. Жизненный цикл данных в пайплайне

### 5.1. Когда происходит `deepcopy`

```
Experiment.execute(data)
│
├── self.transformer == True?
│   ├── ДА → experiment_data = deepcopy(data)
│   │        (копируется весь ExperimentData, включая ds, roles, tmp_roles)
│   └── НЕТ → experiment_data = data (ссылка)
│
└── для каждого executor:
    ├── executor._is_transformer == True?
    │   └── executor.execute() вызывает data.copy(data=new_ds)
    │       → создаёт новый ExperimentData с новым ds,
    │         но _initial_cols синхронизируется
    └── иначе → мутирует существующий ExperimentData
```

### 5.2. `tmp_roles`: установка и сброс

```
┌────────────────────────────────────────────────────────────────────────┐
│ OnRoleExperiment.execute(data)                                        │
│                                                                        │
│ 1. Поиск таргетов: target_fields = data.field_search(self.role)       │
│                                                                        │
│ 2. ВЕКТОРНАЯ ВЕТКА:                                                    │
│    data.ds.tmp_roles = {                                               │
│        "revenue": TempTargetRole(),                                    │
│        "spends":  TempTargetRole(),                                    │
│    }                                                                   │
│    → StatsComparator видит ОБА таргета за один проход                  │
│    → _compute_stats() агрегирует все колонки в одном джобе             │
│    data = super().execute(data)                                        │
│    data.ds.tmp_roles = {}   ← СБРОС                                   │
│                                                                        │
│ 3. ИТЕРАТИВНАЯ ВЕТКА:                                                  │
│    for field in ["revenue", "spends"]:                                 │
│        data.ds.tmp_roles = {field: TempTargetRole()}                   │
│        data = super().execute(data)                                    │
│        data.ds.tmp_roles = {}   ← СБРОС после каждой итерации         │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.3. `_clean_ds_for_iteration()` — изоляция итераций

Вызывается **один раз** перед циклом (оптимизация 2.0):

```python
# ParamsExperiment.execute()
clean_ds = data._clean_ds_for_iteration()  # ← вне цикла

for flat_param in tqdm(self._flat_params):
    t_data = ExperimentData(clean_ds)  # каждый раз из чистого среза
    ...
```

Механизм:

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
        cleaned.checkpoint(eager=True)  # обрезка DAG
    return cleaned
```

---

## 6. Пошаговое выполнение `AATest().execute(data)`

### Вход

```python
from hypex import AATest
from hypex.dataset import Dataset, TargetRole, TreatmentRole, FeatureRole

ds = Dataset(
    roles={
        TargetRole(): ["revenue", "spends"],
        FeatureRole(): ["age", "gender"],
        TreatmentRole(): "treat",
    },
    data=pd.read_csv("experiment.csv"),
)

aa = AATest(precision_mode=False, n_iterations=10)
result = aa.execute(ds)
```

### Шаг 1: `ExperimentShell.execute()`

```python
# ui/base.py → ExperimentShell.execute()
data = ExperimentData(deepcopy(ds))  # упаковка в контейнер
# (если Spark и auto_persist=True → persist)

result_experiment_data = self._experiment.execute(data)
self._out.extract(result_experiment_data)
return self._out  # → AAOutput
```

### Шаг 2: Внешний `Experiment.execute()`

```
Experiment(executors=[
    [Float32Caster?]         # если float32=True
    ParamsExperiment(...)     # основная итерация
    [IfParamsExperiment?]     # если sample_size задан
    [AADryTestAnalyzer?]      # если dry_test=True
    AAScoreAnalyzer()         # выбор лучшего сплита
], key="AATest")
```

### Шаг 3: `ParamsExperiment.execute()` — цикл по итерациям

```
clean_ds = data._clean_ds_for_iteration()

for flat_param in [
    {AASplitter: {random_state: 0, control_size: 0.5, sample_size: None, groups_sizes: None},
     Comparator: {grouping_role: [AdditionalTreatmentRole()], space: [SpaceEnum.additional]}},
    {AASplitter: {random_state: 1, ...}, ...},
    ...
]:
    t_data = ExperimentData(clean_ds)
    executor.set_params(flat_param)
    t_data = executor.execute(t_data)  # → базовый эксперимент
    report = self.reporter.report(t_data)
    results.append(report)
```

### Шаг 4: Базовый эксперимент (одна итерация)

```
Experiment(executors=[
    NaDropper(),                          # удаление NaN из таргетов
    AASplitter(),                         # рандомизированный сплит
    aa_metrics = Experiment(executors=[   # метрики
        GroupSizes(grouping_role=AdditionalTreatmentRole()),
        OnRoleExperiment(
            executors=[
                GroupDifference(...),
                TTest(...),
                KSTest(...),
                Chi2Test(...),
            ],
            role=[TargetRole(), FeatureRole()],
        ),
        OneAAStatAnalyzer(),              # агрегация
    ])
])
```

### Шаг 5: `NaDropper.execute()`

```python
# Transformer: удаляет строки с NaN в таргетах
target_cols = data.ds.search_columns(roles=self.target_roles)
result = data.copy(data=self.calc(data=data.ds, target_cols=target_cols, how=self.how))
```

### Шаг 6: `AASplitter.execute()`

```python
# Calculator: создаёт колонку split с метками групп
result = self.calc(
    data.ds,
    random_state=0,          # из flat_param
    control_size=0.5,
    sample_size=None,
    const_group_field=None,
)
data = self._set_value(data, result)
# → data.ds получает колонку "AASplitter┆rs 0┆" с AdditionalTreatmentRole
```

### Шаг 7: `GroupSizes.execute()` (StatsComparator)

```python
# Phase 1: одна агрегация
agg_data = data.ds[[group_col]]
group_col_stats = self._compute_stats(data=agg_data, group_cols=[group_col],
                                       target_columns=[group_col], stats=["count"])
# → {"0": {"AASplitter┆rs 0┆": {"count": 5000}}, "1": {"AASplitter┆rs 0┆": {"count": 5000}}}

# Phase 2: скалярное сравнение
result = self._inner_function(baseline_stats, compared_stats)
# → {"control size": 5000, "test size": 5000, "control size %": 50.0, "test size %": 50.0}

# Сохранение
data.set_value(ExperimentDataEnum.analysis_tables, self.id, result_dataset)
```

### Шаг 8: `OnRoleExperiment.execute()`

```python
target_fields = data.field_search([TargetRole(), FeatureRole()])
# → ["revenue", "spends", "age", "gender"]

# Векторные (если Spark):
#   TTest → StatsTTest: один _compute_stats на все таргеты

# Итеративные (если Pandas):
for field in ["revenue", "spends", "age", "gender"]:
    data.ds.tmp_roles = {field: TempTargetRole()}
    # GroupDifference.execute() → сравнение средних
    # GroupTTest.execute() → t-тест
    # GroupKSTest.execute() → KS-тест
    # GroupChi2Test.execute() → χ² (только для str-колонок)
    data = super().execute(data)
    data.ds.tmp_roles = {}
```

### Шаг 9: `OneAAStatAnalyzer.execute()`

```python
# Сбор всех результатов тестов
executor_ids = data.get_ids(self.ANALYSIS_TEST_CLASSES, ...)
analysis_data = self._collect_means(data, executor_ids)
# → {"mean┆StatsTTest┆p-value┆all": 0.42, "mean┆StatsTTest┆pass┆all": 0.03, ...}

analysis_data["mean test score"] = self._compute_composite_score(analysis_data)
# → взвешенный средний p-value (TTest вес=1, KS вес=2, Chi2 вес=2)

data.set_value(ExperimentDataEnum.analysis_tables, self.id, SmallDataset.from_dict(analysis_data))
```

### Шаг 10: `AAScoreAnalyzer.execute()`

```python
# 1. Получение таблицы результатов из ParamsExperiment
param_id = data.get_one_id(ParamsExperiment, ExperimentDataEnum.analysis_tables)
score_table = data.analysis_tables[param_id]

# 2. Расчёт AA-score для каждого (фича, тест, группа)
data = self._analyze_aa_score(data, score_table)
# → weight = 1 - |alpha - pass_rate|, pass = weight >= threshold

# 3. Выбор лучшего сплита
data = self._analyze_best_split(data, score_table)
# → score = weighted_pvalues × 2/3 + mean_test_score × 1/3
# → best_index = score_col.idxmax()
# → best_splitter_id = "AASplitter┆rs 7┆"

# 4. Применение лучшего сплита к исходным данным
best_splitter = self.build_splitter_from_id(best_splitter_id)
best_splitter.key = "best"
data = best_splitter.execute(data)
# → добавляет колонку "AASplitter┆rs 7┆best"
```

### Шаг 11: `AAOutput.extract()`

```python
super().extract(experiment_data)  # summary_reporter → AAPassedReporter
self._extract_experiments(experiment_data)         # таблица итераций
self._extract_aa_score(experiment_data)            # AA-score + best split statistics
self._extract_best_split_statistics(experiment_data)  # детальная таблица лучшего сплита
```

### Результат

```python
result.experiments        # DataFrame: каждая строка = итерация, колонки = тесты
result.aa_score           # DataFrame: score и pass для каждого (фича, тест)
result.best_split         # Dataset: колонка split с метками лучшего сплита
result.best_split_statistics  # SmallDataset: детальная статистика лучшего сплита
result.summary            # Dataset: итоговая таблица с pass/OK для лучшего сплита
```

---

## 7. Контракт для создания кастомного Executor'а

### 7.1. Наследование от `Calculator`

```python
from hypex.executor import Calculator
from hypex.dataset import Dataset

class MyCalculator(Calculator):
    @classmethod
    def _inner_function(cls, data: Dataset, **kwargs) -> Any:
        # Чистая логика вычисления
        return {"result": data.mean()}

    @property
    def search_types(self):
        return [int, float]

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = self.calc(data.ds)
        return self._set_value(data, result)

    def _set_value(self, data, value, key=None):
        return data.set_value(ExperimentDataEnum.variables, self.id, value, key="result")
```

### 7.2. Наследование от `Transformer`

```python
from hypex.transformers.abstract import Transformer

class MyFilter(Transformer):
    @staticmethod
    def _inner_function(data: Dataset, threshold: float = 0.5) -> Dataset:
        mask = data["score"] > threshold
        return data[mask]

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(data=self.calc(data=data.ds, threshold=0.7))
        return result
```

### 7.3. Регистрация в `backend_factory`

```python
from hypex.utils.registry import backend_factory
from hypex.dataset.backends import PandasDataset, SparkDataset

@backend_factory.register(MyMasterClass, PandasDataset)
class MyPandasImpl(MyMasterClass):
    ...

@backend_factory.register(MyMasterClass, SparkDataset)
class MySparkImpl(MyMasterClass):
    ...
```

---

## 8. Оптимизации производительности в конвейере

| Оптимизация | Где | Что делает |
|---|---|---|
| `deepcopy` только для трансформеров | `Experiment.execute()` | Не копирует данные, если пайплайн ничего не меняет |
| Векторные исполнители | `OnRoleExperiment` | Один `groupBy().agg()` вместо N×1 джобов |
| `_clean_ds_for_iteration()` вне цикла | `ParamsExperiment`, `CycledExperiment`, `GroupExperiment` | Один `checkpoint()` вместо N |
| `checkpoint(eager=True)` | `_clean_ds_for_iteration()` | Обрезка DAG в Spark |
| `persist()` перед итерациями | `ExperimentShell.execute()` | Кэширование исходных данных |
| `field_data_search()` без материализации | `ExperimentData` | `self._data[cols]` — ленивая проекция, без `toPandas()` |

---

*Документация актуальна для версии **2.0.0a1**. Следующий раздел — «Компараторы и статистическое тестирование».*
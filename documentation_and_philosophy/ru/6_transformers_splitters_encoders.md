# Предобработка: трансформеры, сплиттеры, энкодеры

---

## 1. Базовый контракт предобработки

Предобработка в HypEx построена на трёх семействах блоков, наследующих `Calculator`. Каждое семейство решает свою задачу:

| Семейство | Базовый класс | Что делает | Мутирует данные? |
|-----------|---------------|------------|------------------|
| **Трансформеры** | `Transformer` | Изменяют значения / структуру датасета | Да (`data.copy(data=new_ds)`) |
| **Сплиттеры** | `Calculator` | Создают колонку разбиения на группы | Нет (добавляет `AdditionalTreatmentRole`) |
| **Энкодеры** | `Encoder` | Кодируют категориальные признаки | Нет (добавляет `AdditionalFeatureRole`, дизейблит исходные) |

### 1.1. `Transformer` — мутация датасета

```python
class Transformer(Calculator):
    @property
    def _is_transformer(self):
        return True

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Dataset:
        raise AbstractMethodError

    @classmethod
    def calc(cls, data: Dataset, **kwargs):
        return cls._inner_function(data, **kwargs)

    def execute(self, data: ExperimentData) -> ExperimentData:
        data = data.copy(data=self.calc(data=data.ds))
        return data
```

Ключевое свойство — `_is_transformer = True`. Когда `Experiment` обнаруживает хотя бы один трансформер в пайплайне, он **глубоко копирует** входные данные перед исполнением:

```python
# Experiment.execute()
experiment_data = deepcopy(data) if self.transformer else data
```

Это гарантирует, что исходный `Dataset` пользователя не будет изменён.

### 1.2. Паттерн «поиск по роли → операция»

Все трансформеры следуют единому шаблону:

```
1. Найти колонки по роли:     target_cols = data.ds.search_columns(roles=self.target_roles, ...)
2. Применить операцию:        result = self._inner_function(data.ds, target_cols, ...)
3. Вернуть новый датасет:     data.copy(data=result)
```

Роли (`TargetRole`, `FeatureRole`, `PreTargetRole`) определяют, **какие колонки** обрабатываются, а не их имена. Это делает пайплайн независимым от конкретной схемы данных.

---

## 2. Трансформеры

### 2.1. Обработка пропусков

#### `NaDropper` — удаление строк с пропусками

Удаляет строки, где **любой** (или **все**) таргет содержит `NaN`. Используется в `AATest` как первый шаг пайплайна.

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `target_roles` | `ABCRole \| list` | `TargetRole()` | Роли колонок для проверки |
| `how` | `"any" \| "all"` | `"any"` | `"any"` — удалить если есть хотя бы один NaN; `"all"` — только если все NaN |

```python
@staticmethod
def _inner_function(data, target_cols, how="any"):
    if not target_cols:
        return data
    return data.dropna(subset=target_cols, how=how)
```

#### `NaFiller` — заполнение пропусков

Заполняет `NaN` заданным значением или методом (`ffill` / `bfill`).

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `target_roles` | `ABCRole \| list` | `FeatureRole()` | Роли колонок для заполнения |
| `values` | `ScalarType \| dict` | `None` | Значение для заполнения |
| `method` | `"bfill" \| "ffill"` | `None` | Метод заполнения |

### 2.2. Приведение типов

#### `TypeCaster` — универсальный кастер

Приводит колонки к заданным типам. Поддерживает два формата `dtype`:

```python
# Формат 1: {имя колонки: целевой тип}
TypeCaster(dtype={"age": float, "revenue": int})

# Формат 2: {исходный тип: целевой тип}
TypeCaster(dtype={int: float})  # все int → float
```

Дополнительно выполняет **даункастинг** `Decimal` / `float64` → `float` через `_downcast()`:

```python
@staticmethod
def _downcast(data):
    double_cols = [
        col for col, c_type in data.roles.items()
        if c_type.data_type is Decimal or c_type.data_type is float
    ]
    return data.astype({col: float for col in double_cols})
```

#### `Float32Caster` — даункастинг до `float32`

Специализированный трансформер для экономии памяти (~2×). Работает с обоими бэкендами:

```python
@staticmethod
def _inner_function(data, target_cols):
    if not target_cols:
        return data
    if data.backend_type == BackendsEnum.spark:
        dtype_map = {col: "float32" for col in target_cols}
    else:
        dtype_map = {col: np.float32 for col in target_cols}
    result = data.astype(dtype_map)
    # Восстанавливаем float в ролях, чтобы
    # search_columns(search_types=[float]) продолжал работать
    for col in target_cols:
        if col in result.roles:
            result.roles[col].data_type = float
    return result
```

**Важно:** после кастинга `data_type` в роли восстанавливается на `float`, иначе `search_columns(search_types=[float])` перестанет находить эти колонки.

### 2.3. `CUPEDTransformer` — редукция дисперсии

Реализует **CUPED** (Controlled-experiment Using Pre-Experiment Data). Для каждой пары `(target, pre_target)` вычисляет:

```
θ = Cov(Y, X) / Var(X)
Y_cuped = Y - θ · (X - E[X])
```

где `Y` — целевая метрика, `X` — пре-таргет (метрика до эксперимента).

| Параметр | Тип | Описание |
|----------|-----|----------|
| `cuped_features` | `dict[str, str]` | `{target: pre_target}` |

```python
@staticmethod
def _inner_function(data, cuped_features):
    result = deepcopy(data)
    for target_feature, pre_target_feature in cuped_features.items():
        theta = cuped_theta(
            result[target_feature].raw_data.values.flatten(),
            result[pre_target_feature].raw_data.values.flatten(),
        )
        pre_target_mean = result[pre_target_feature].mean()
        new_values_ds = (
            result[target_feature]
            - (result[pre_target_feature] - pre_target_mean) * theta
        )
        result = result.add_column(
            data=new_values_ds,
            role={f"{target_feature}_cuped": TargetRole()},
        )
    return result
```

Результат:
- В `ds` добавляется колонка `{target}_cuped` с ролью `TargetRole()` — она автоматически подхватывается `OnRoleExperiment`.
- В `analysis_tables` сохраняется отчёт о variance reduction (`feature`, `variance_reduction_pct`).

Функция `cuped_theta` использует **центрированную арифметику** для численной стабильности:

```python
def cuped_theta(y, x):
    # Исключаем NaN-пары
    valid = np.isfinite(y) & np.isfinite(x)
    y, x = y[valid], x[valid]
    x_centered = x - np.mean(x)
    var_x = float(np.mean(x_centered ** 2))
    if not np.isfinite(var_x) or var_x <= noise_floor:
        return 0.0
    cov_xy = float(np.mean((y - np.mean(y)) * x_centered))
    return cov_xy / var_x
```

### 2.4. `CategoryAggregator` — агрегация редких категорий

Объединяет категории с частотой ниже порога в одну группу.

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `target_roles` | `ABCRole \| list` | `FeatureRole()` | Роли категориальных колонок |
| `threshold` | `int` | `15` | Минимальная частота категории |
| `new_group_name` | `str \| None` | `None` | Имя объединённой группы |

```python
@staticmethod
def _inner_function(data, target_cols=None, threshold=15, new_group_name=None):
    for column in target_cols:
        categories_counts = data[column].value_counts()
        values_to_replace = categories_counts[
            categories_counts["count"] < threshold
        ][column].get_values(column=column)
        data[column] = data[column].replace(
            to_replace=values_to_replace, value=new_group_name
        )
    return data
```

### 2.5. Фильтры

Фильтры **«мягко» исключают** колонки из анализа, назначая им `InfoRole()` вместо удаления. Это сохраняет данные в датасете, но исключает их из поиска по ролям.

#### `CVFilter` — фильтр по коэффициенту вариации

Исключает колонки с слишком низким (константы) или высоким (шум) коэффициентом вариации.

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `lower_bound` | `float \| None` | `None` | Минимальный CV |
| `upper_bound` | `float \| None` | `None` | Максимальный CV |
| `search_types` | `list[type]` | `[float, int, bool]` | Только числовые колонки |

```python
for column in target_cols:
    cv = data[column].coefficient_of_variation()
    if (upper_bound and cv > upper_bound) or (lower_bound and cv < lower_bound):
        data.roles[column] = InfoRole()  # ← мягкое исключение
```

#### `ConstFilter` — фильтр констант

Исключает колонки, где одно значение занимает более `threshold` доли.

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `threshold` | `float` | `0.95` | Максимальная доля одного значения |

```python
value_counts = data[column].value_counts(normalize=True, sort=True)
if value_counts.get_values(0, "proportion") > threshold:
    data.roles[column] = InfoRole()
```

#### `NanFilter` — фильтр по доле пропусков

Исключает колонки с долей `NaN` выше порога.

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `threshold` | `float` | `0.8` | Максимальная доля NaN |

#### `CorrFilter` — фильтр корреляций

Исключает одну из пары сильно коррелированных колонок. Политика выбора (`drop_policy`):
- `"cv"` — оставить колонку с **большим** коэффициентом вариации (по умолчанию)
- `"corr"` — оставить колонку с **большей** корреляцией с `PreTargetRole`

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `target_roles` | `ABCRole \| list` | `FeatureRole()` | Колонки для проверки |
| `corr_space_roles` | `ABCRole \| list` | `[FeatureRole(), TargetRole()]` | Пространство для корреляции |
| `threshold` | `float` | `0.8` | Порог корреляции |
| `method` | `str` | `"pearson"` | Метод корреляции |

#### `OutliersFilter` — фильтр выбросов

Удаляет **строки**, где хотя бы один таргет выходит за пределы `[нижний перцентиль, верхний перцентиль]`.

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `lower_percentile` | `float` | `0` | Нижний перцентиль |
| `upper_percentile` | `float` | `1` | Верхний перцентиль |

В отличие от других фильтров, `OutliersFilter` удаляет **строки**, а не дизейблит колонки:

```python
mask = data[target_cols].apply(
    func=lambda x: (
        (x < x.quantile(lower_percentile)) | (x > x.quantile(upper_percentile))
    ),
    role={column: InfoRole() for column in target_cols},
    axis=0,
)
mask = mask.apply(func=lambda x: x.any(), role={"filter": InfoRole()}, axis=1)
drop_indexes = mask[mask["filter"]].dropna().index
data = data.drop(drop_indexes, axis=0)
```

### 2.6. `Shuffle` — перемешивание

`Shuffle` наследует `Calculator` (не `Transformer`), но по поведению аналогичен:

```python
class Shuffle(Calculator):
    def __init__(self, random_state=None, key=""):
        ...

    @staticmethod
    def _inner_function(data, random_state=None):
        return data.shuffle(random_state=random_state)

    def execute(self, data):
        result = data.copy(data=self.calc(data=data.ds, random_state=self.random_state))
        return result
```

### 2.7. Сводная таблица трансформеров

| Трансформер | Тип операции | Роли по умолчанию | Результат |
|-------------|-------------|-------------------|-----------|
| `NaDropper` | Удаление строк | `TargetRole()` | Меньше строк |
| `NaFiller` | Заполнение | `FeatureRole()` | Нет NaN |
| `TypeCaster` | Приведение типов | `FeatureRole()` | Новые типы |
| `Float32Caster` | Даункастинг | `[FeatureRole(), TargetRole()]` | `float32` |
| `CUPEDTransformer` | Редукция дисперсии | `TargetRole()` | `{target}_cuped` |
| `CategoryAggregator` | Агрегация категорий | `FeatureRole()` | Меньше категорий |
| `CVFilter` | Дизейбл колонок | `FeatureRole()` | `InfoRole()` |
| `ConstFilter` | Дизейбл колонок | `FeatureRole()` | `InfoRole()` |
| `NanFilter` | Дизейбл колонок | `FeatureRole()` | `InfoRole()` |
| `CorrFilter` | Дизейбл колонок | `FeatureRole()` | `InfoRole()` |
| `OutliersFilter` | Удаление строк | `FeatureRole()` | Меньше строк |
| `Shuffle` | Перемешивание | — | Новый порядок строк |

---

## 3. Сплиттеры

### 3.1. `AASplitter` — рандомизированный сплит

`AASplitter` создаёт колонку `split` с метками групп (`"control"`, `"test_1"`, `"test_2"`, ...). Результат сохраняется в `ds` как колонка с ролью `AdditionalTreatmentRole()`.

#### Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `control_size` | `float` | `0.5` | Доля контрольной группы (0–1) |
| `random_state` | `int \| None` | `None` | Сид для воспроизводимости |
| `sample_size` | `float \| None` | `None` | Доля данных для сэмплирования |
| `groups_sizes` | `list[float] \| None` | `None` | Кастомные пропорции групп |
| `constant_key` | `bool` | `True` | Не менять ключ при `key.setter` |
| `save_groups` | `bool` | `True` | Сохранять группы в `ExperimentData.groups` |

#### Идентификатор и хэш параметров

`AASplitter` переопределяет `_generate_params_hash()` для включения параметров в ID:

```python
def _generate_params_hash(self):
    hash_parts = []
    if self.control_size != 0.5:
        hash_parts.append(f"cs {self.control_size}")
    if self.random_state is not None:
        hash_parts.append(f"rs {self.random_state}")
    if self.groups_sizes is not None:
        hash_parts.append(f"gs {self.groups_sizes}")
    self._params_hash = "|".join(hash_parts)
```

Пример ID: `AASplitter┆cs 0.3|rs 42┆`

#### Механизм `random_split_labels`

Сплит выполняется через **детерминированное хэширование индекса**:

```
1. Индекс каждой строки конкатенируется с сидом
2. Вычисляется хэш:  hash = hash_array(index + seed) % MOD
3. MOD = 10_000_000
4. По хэшу определяется группа через CASE WHEN по порогам (edges)
```

Для **Pandas** используется `pandas.util.hash_array` (SipHash), для **Spark** — `F.hash()` (Murmur3). Детерминизм гарантирован **внутри** каждого бэкенда, но точное распределение может отличаться между бэкендами для одного и тогого сида.

#### `_const_group_plan()` — закреплённые группы

Когда в данных есть колонка с `ConstGroupRole()`, часть строк **закрепляется** за группами и не участвует в рандомизированном сплите:

```
┌────────────────────────────────────────────────────────────────────────┐
│  Данные с ConstGroupRole                                               │
│  ┌───────┬──────────────┐                                             │
│  │ user  │ const_group  │                                             │
│  ├───────┼──────────────┤                                             │
│  │ u1    │ control      │  ← закреплён в control                     │
│  │ u2    │ NaN          │  ← участвует в рандомизированном сплите    │
│  │ u3    │ test_1       │  ← закреплён в test_1                     │
│  │ u4    │ NaN          │  ← участвует в рандомизированном сплите    │
│  └───────┴──────────────┘                                             │
│                                                                        │
│  _const_group_plan():                                                  │
│  1. value_counts(const_group) → подсчёт закреплённых                   │
│  2. Пересчёт control_size:                                             │
│     target_control_total = (n_pinned + free × sample_size) × ctrl     │
│     adjusted_control = (target - pinned_control) / (free × sample)    │
│  3. translation: {label → split_label | _FREE_CONST_SENTINEL}         │
│                                                                        │
│  Результат:                                                            │
│  ├── Закреплённые строки → их метка из const_group                    │
│  └── Свободные строки → random_split_labels(adjusted_control_size)    │
└────────────────────────────────────────────────────────────────────────┘
```

Ключевая логика пересчёта `control_size`:

```python
# Целевое число контрольных наблюдений после сэмплирования
n_sampled_total = n_pinned + free_size * sample_size
target_control_total = n_sampled_total * control_size

# Сколько контрольных нужно из свободных строк
free_control_needed = max(0.0, target_control_total - n_pinned_control)
free_sampled = free_size * sample_size

# Скорректированная доля для рандомизированного сплита
adjusted_control_size = free_control_needed / free_sampled if free_sampled > 0 else 0.0
```

#### `_inner_function()` — полный алгоритм

```python
@staticmethod
def _inner_function(data, random_state=None, control_size=0.5,
                    groups_sizes=None, sample_size=1.0,
                    const_group_field=None, **kwargs):
    # 1. Определяем метки групп
    labels = ["control"] + [f"test_{i+1}" for i in range(len(groups_sizes)-1)]

    # 2. Если есть закреплённые группы — разрешаем их
    if const_group_field:
        translation, free_size, control_size = AASplitter._const_group_plan(...)

    # 3. Вычисляем пороги (edges) в масштабе MOD
    MOD = 10_000_000
    frac_limit = int(frac * MOD)
    edges = [int(control_size * frac_limit), frac_limit]

    # 4. Сплит свободных строк + присоединение закреплённых
    parts = []
    if const_group_field:
        # Свободные строки → random_split_labels
        parts.append(tagged[tagged == _FREE_CONST_SENTINEL].random_split_labels(...))
        # Закреплённые строки → их метка
        parts.append(pinned.rename({const_group_field: "split"}))
    else:
        parts.append(data.random_split_labels(...))

    # 5. Объединяем и возвращаем
    return parts[0].append(parts[1:])
```

#### `execute()` — сохранение результата

```python
def execute(self, data: ExperimentData) -> ExperimentData:
    const_group_fields = data.ds.search_columns(ConstGroupRole())
    const_group_fields = const_group_fields[0] if len(const_group_fields) > 0 else None
    result = self.calc(
        data.ds,
        random_state=self.random_state,
        control_size=self.control_size,
        sample_size=self.sample_size,
        const_group_field=const_group_fields,
        groups_sizes=self.groups_sizes,
    )
    data = self._set_value(data, result)
    return data
```

`_set_value()` сохраняет колонку `split` в `ds` с ролью `AdditionalTreatmentRole()` и опционально заполняет `ExperimentData.groups`:

```python
def _set_value(self, data, value, key=None):
    data = data.set_value(
        ExperimentDataEnum.additional_fields,
        self._id,
        value,
        role=AdditionalTreatmentRole(),
    )
    if self.save_groups:
        for group_key in unique_vals:
            group_data = data.ds[data.ds[splitter_col] == group_key]
            data.set_value(
                space=ExperimentDataEnum.groups,
                executor_id=self._id,
                value=group_data,
                key=str(group_key),
            )
    return data
```

### 3.2. `AASplitterWithStratification` — стратифицированный сплит

Применяет логику `AASplitter` **внутри каждой страты**, определённой колонками с `StratificationRole()`.

#### Алгоритм

```
1. Найти страты:  grouping_fields = data.ds.search_columns(StratificationRole())
2. Для каждой страты:
   a. Вырезать поднабор:  group_data
   b. Применить AASplitter._inner_function(group_data, ...)
   c. Сохранить результат
3. Объединить все страты через append()
```

```python
@staticmethod
def _inner_function(data, random_state=None, control_size=0.5,
                    grouping_fields=None, groups_sizes=None,
                    sample_size=1.0, const_group_field=None, **kwargs):
    if not grouping_fields:
        return AASplitter._inner_function(...)

    result_splits = []
    for _, group_data in data.groupby(grouping_fields):
        group_split = AASplitter._inner_function(
            group_data, random_state, control_size,
            groups_sizes=groups_sizes, sample_size=sample_size,
            const_group_field=const_group_field, **kwargs,
        )
        result_splits.append(group_split)

    # Один concat вместо цепочки append()
    return result_splits[0].append(result_splits[1:])
```

#### Оптимизация: `persist` + `checkpoint`

`execute()` переопределён для оптимизации Spark:

```python
@timeit(level="SPLIT", prefix="SPLITTER_STRAT")
def execute(self, data: ExperimentData) -> ExperimentData:
    # 1. Персистим данные перед итерацией по стратам
    if data.ds.backend_type == BackendsEnum.spark and not data.ds.is_persisted:
        data.ds.persist(storage_level="MEMORY_AND_DISK", action="count")
        persisted_locally = True

    try:
        result = self.calc(...)
        # 2. Checkpoint ДО unpersist, чтобы материализовать результат
        if data.ds.backend_type == BackendsEnum.spark:
            result.checkpoint(eager=True)
    finally:
        # 3. Анперсистим в finally для гарантии очистки
        if persisted_locally:
            data.ds.unpersist()

    # 4. Назначаем роль и сохраняем
    result = result.replace_roles({"split": AdditionalTreatmentRole()})
    data = self._set_value(data, result)
    return data
```

---

## 4. Энкодеры

### 4.1. Базовый класс `Encoder`

`Encoder` наследует `Calculator` и определяет общий поток:

```
1. Найти категориальные колонки по роли и типу
2. Закодировать через бэкенд-специфичное расширение
3. Дизейблить исходные колонки (DisabledRole)
4. Сохранить закодированные колонки в additional_fields
```

```python
class Encoder(Calculator):
    def __init__(self, target_roles=None, key=""):
        self.target_roles = target_roles or FeatureRole()

    @property
    def search_types(self):
        return [CategoricalTypes]  # [str]

    def _disable_target_cols(self, data, target_cols):
        disable_roles = {
            col: DisabledRole(initial_role=data.ds.roles[col])
            for col in target_cols
        }
        data.ds.replace_roles(new_roles_map=disable_roles)
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(
            roles=self.target_roles, search_types=self.search_types
        )
        if not target_cols:
            return data
        result = self.calc(data=data.ds, target_cols=target_cols)
        return self._set_value(data=data, value=result, key=self.key)
```

#### `DisabledRole` — «мягкое» удаление

Исходные категориальные колонки не удаляются, а помечаются `DisabledRole`, сохраняя ссылку на исходную роль:

```python
class DisabledRole(ABCRole):
    def __init__(self, initial_role=None, data_type=None):
        super().__init__(initial_role.data_type if initial_role else data_type)
        self._initial_role = initial_role
```

Восстановление происходит в `ExperimentData.cleanup_additional()`:

```python
if cols_to_enable:
    self._data = self._data.replace_roles(
        new_roles_map={
            col: self._data.roles[col].initial_role for col in cols_to_enable
        }
    )
```

### 4.2. `DummyEncoder` — one-hot кодирование

`DummyEncoder` делегирует кодирование бэкенд-специфичному расширению через `backend_factory`:

```python
class DummyEncoder(Encoder):
    @staticmethod
    def _inner_function(data, target_cols=None, **kwargs):
        if not target_cols:
            return Dataset.create_empty()
        encoder_cls = backend_factory.resolve_backend(DummyEncoderExtension, data)
        return encoder_cls.calc(data=data, target_cols=target_cols, **kwargs)
```

#### Pandas-реализация

```python
@backend_factory.register(DummyEncoderExtension, PandasDataset)
class PandasDummyEncoderExtension(DummyEncoderExtension):
    @staticmethod
    def calc(data, target_cols=None, **kwargs):
        dummies_df = pd.get_dummies(
            data=data[target_cols].raw_data,
            drop_first=True,
            dtype=int,
        )
        roles = DummyEncoderExtension._build_roles(
            dummies_columns=list(dummies_df.columns),
            source_cols=target_cols,
            source_roles=data.roles,
        )
        return DatasetAdapter.to_dataset(dummies_df, roles=new_roles, small=False)
```

#### Spark-реализация

```python
@backend_factory.register(DummyEncoderExtension, SparkDataset)
class SparkDummyEncoderExtension(DummyEncoderExtension):
    @staticmethod
    def calc(data, target_cols=None, **kwargs):
        dummies_df = ps.get_dummies(
            data=data[target_cols].raw_data,
            drop_first=True,
            dtype=int,
        )
        ...
```

#### Построение ролей для dummy-колонок

Каждая dummy-колонка `{source}_{category}` получает роль `AdditionalFeatureRole` с `data_type=bool`:

```python
@classmethod
def _build_roles(cls, dummies_columns, source_cols, source_roles):
    roles = {}
    for col in dummies_columns:
        src = cls._resolve_source_column(col, source_cols)
        roles[col] = source_roles[src].asadditional(int)
        roles[col].data_type = bool
    return roles
```

`asadditional()` ищет в `default_roles` класс, наследующий одновременно от исходной роли и от `AdditionalRole`. Для `FeatureRole` это `AdditionalFeatureRole`.

---

## 5. Предустановленный пайплайн `PREPROCESSING_DATA`

В модуле `hypex/preprocessing.py` определён готовый пайплайн предобработки:

```python
PREPROCESSING_DATA = Experiment(
    executors=[
        NaFiller(method="ffill"),
        CategoryAggregator(),
        CorrFilter(),
        CVFilter(),
        NanFilter(),
        ConstFilter(),
        OutliersFilter(lower_percentile=0.05, upper_percentile=0.95),
        DummyEncoder(),
    ]
)
```

### Порядок операций и обоснование

| Шаг | Трансформер | Что делает | Зачем именно здесь |
|-----|-------------|------------|-------------------|
| 1 | `NaFiller(method="ffill")` | Заполняет NaN методом прямого заполнения | До любых операций, требующих полных данных |
| 2 | `CategoryAggregator()` | Объединяет редкие категории | До кодирования и корреляций |
| 3 | `CorrFilter()` | Исключает коррелирующие колонки | До CV-фильтра (меньше данных для расчёта) |
| 4 | `CVFilter()` | Исключает колонки с экстремальным CV | До фильтра по константам |
| 5 | `NanFilter()` | Исключает колонки с >80% NaN | После заполнения (оставшиеся NaN — структурные) |
| 6 | `ConstFilter()` | Исключает квазиконстантные колонки | После всех статистических фильтров |
| 7 | `OutliersFilter(0.05, 0.95)` | Удаляет строки-выбросы | Последняя операция с данными |
| 8 | `DummyEncoder()` | One-hot кодирование | В конце, чтобы не кодировать исключённые колонки |

### Использование

```python
from hypex.preprocessing import PREPROCESSING_DATA
from hypex.dataset import ExperimentData

ed = ExperimentData(ds)
cleaned = PREPROCESSING_DATA.execute(ed)
# cleaned.ds содержит предобработанные данные
```

---

## 6. Примеры использования

### 6.1. Кастомный пайплайн предобработки

```python
from hypex.experiments.base import Experiment
from hypex.transformers import NaDropper, Float32Caster, CUPEDTransformer
from hypex.dataset import TargetRole, PreTargetRole

preprocessing = Experiment(executors=[
    NaDropper(target_roles=TargetRole()),
    CUPEDTransformer(cuped_features={"revenue": "revenue_lag_1"}),
    Float32Caster(target_roles=[TargetRole()]),
])
```

### 6.2. A/A-сплит с закреплёнными группами

```python
from hypex import AATest
from hypex.dataset import Dataset, TargetRole, TreatmentRole, FeatureRole, ConstGroupRole

ds = Dataset(
    roles={
        TargetRole(): ["revenue"],
        TreatmentRole(): "treat",
        FeatureRole(): ["age", "gender"],
        ConstGroupRole(): "fixed_group",  # закреплённые группы
    },
    data="experiment.csv",
)

aa = AATest(control_size=0.5, n_iterations=10)
result = aa.execute(ds)
# Строки с непустым "fixed_group" закреплены,
# остальные рандомизированы с учётом закреплённых
```

### 6.3. Стратифицированный сплит

```python
from hypex.dataset import StratificationRole

ds_strat = Dataset(
    roles={
        TargetRole(): ["revenue"],
        TreatmentRole(): "treat",
        FeatureRole(): ["age"],
        StratificationRole(): ["industry"],  # стратификация по индустрии
    },
    data="experiment.csv",
)

aa_strat = AATest(stratification=True, n_iterations=10)
result = aa_strat.execute(ds_strat)
# Сплит выполняется внутри каждой индустрии
```

### 6.4. Кодирование категориальных признаков для матчинга

```python
from hypex import Matching
from hypex.dataset import FeatureRole

ds_matching = Dataset(
    roles={
        TargetRole(): ["revenue"],
        TreatmentRole(): "treat",
        FeatureRole(): ["age", "gender", "industry"],  # gender и industry — категориальные
    },
    data="matching.csv",
)

# DummyEncoder автоматически применится внутри Matching
m = Matching(encode_categories=True)
result = m.execute(ds_matching)
```

---

## 7. Контракт для создания кастомного трансформера

```python
from hypex.transformers.abstract import Transformer
from hypex.dataset import Dataset, ExperimentData, TargetRole

class MyFilter(Transformer):
    def __init__(self, threshold: float = 0.5, key: str = ""):
        super().__init__(key=key)
        self.threshold = threshold
        self.target_roles = TargetRole()

    @staticmethod
    def _inner_function(data: Dataset, target_cols: list[str],
                        threshold: float = 0.5) -> Dataset:
        for col in target_cols:
            # Ваша логика
            mask = data[col] > threshold
            data = data[mask]
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        target_cols = data.ds.search_columns(roles=self.target_roles)
        result = data.copy(
            data=self.calc(data=data.ds, target_cols=target_cols,
                           threshold=self.threshold)
        )
        return result
```

Для **энкодера** наследуйтесь от `Encoder` и реализуйте `_inner_function` через `backend_factory`:

```python
from hypex.encoders.abstract import Encoder
from hypex.extensions.abstract import Extension
from hypex.utils.registry import backend_factory

class MyEncoderExtension(Extension):
    @staticmethod
    def calc(data, target_cols=None, **kwargs):
        # Ваша логика кодирования
        ...

@backend_factory.register(MyEncoderExtension, PandasDataset)
class PandasMyEncoder(MyEncoderExtension):
    ...

class MyEncoder(Encoder):
    @staticmethod
    def _inner_function(data, target_cols=None, **kwargs):
        encoder_cls = backend_factory.resolve_backend(MyEncoderExtension, data)
        return encoder_cls.calc(data=data, target_cols=target_cols, **kwargs)
```

---

*Документация актуальна для версии **2.0.0a1**. Следующий раздел — «Matching-пайплайн: FAISS, Bias, Treatment Effects».*
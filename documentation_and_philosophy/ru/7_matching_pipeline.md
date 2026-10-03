# Matching-пайплайн: поиск соседей и оценка эффектов

---

## 1. Общий поток данных

Matching-пайплайн решает задачу **оценки причинно-следственных связей** без рандомизированного эксперимента. Вместо случайного назначения в группы, алгоритм находит для каждого наблюдения в одной группе **наиболее похожего соседа** в другой группе по признакам (ковариатам), а затем сравнивает исходы.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         MATCHING PIPELINE                                    │
│                                                                              │
│  Входной Dataset                                                             │
│  ┌─────────┬─────────┬──────────┬──────────┬──────────┐                     │
│  │ treat   │ revenue │ age      │ gender   │ industry │                     │
│  │ (0/1)   │ (target)│ (feature)│ (feature)│ (feature)│                     │
│  └─────────┴─────────┴──────────┴──────────┴──────────┘                     │
│       │          │          │            │            │                       │
│       ▼          ▼          ▼            ▼            ▼                       │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 1. [DummyEncoder]                                                   │     │
│  │    One-hot кодирование категориальных признаков                     │     │
│  │    gender → gender_M, gender_F                                      │     │
│  │    industry → industry_Finance, industry_E-commerce, ...            │     │
│  │    Исходные колонки → DisabledRole                                  │     │
│  │    Закодированные → AdditionalFeatureRole                           │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 2. TypeCaster                                                       │     │
│  │    Приведение типов: {int: float} для FeatureRole и TargetRole      │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 3. [MahalanobisDistance]                                            │     │
│  │    Ковариация → Холецкий → Обратная матрица → Веса                  │     │
│  │    Результат: переменная в ExperimentData.variables                  │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 4. FaissNearestNeighbors                                            │     │
│  │    FAISS-индекс на контроле → поиск по тесту                        │     │
│  │    Результат: AdditionalMatchingRole колонки (индексы соседей)      │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 5. [Bias]                                                           │     │
│  │    Линейная регрессия на матченных данных                           │     │
│  │    Результат: bias + matched_target колонки                         │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 6. MatchingMetrics                                                  │     │
│  │    ITT/ITC → взвешенные средние → ATT, ATC, ATE                    │     │
│  │    Результат: переменные в ExperimentData.variables                  │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 7. MatchingAnalyzer                                                 │     │
│  │    Транспонирование: метрики → строки, статистики → колонки         │     │
│  │    Результат: таблица в analysis_tables                             │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 8. OnRoleExperiment(role=FeatureRole)                               │     │
│  │    Quality tests: TTest, KSTest, Chi2Test (matched_pairs)           │     │
│  │    Результат: p-values матчинга по признакам                        │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  Выход: MatchingOutput                                                       │
│  ┌─────────────────┬──────────────────┬──────────────────┐                  │
│  │ summary         │ quality_results  │ full_data        │                  │
│  │ ATT/ATC/ATE     │ p-values         │ matched dataset  │                  │
│  └─────────────────┴──────────────────┴──────────────────┘                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Условные блоки

Блоки в квадратных скобках `[...]` выполняются **опционально**:

| Блок | Условие активации |
|------|-------------------|
| `DummyEncoder` | `encode_categories=True` (по умолчанию) |
| `MahalanobisDistance` | `distance="mahalanobis"` (по умолчанию) |
| `Bias` | `bias_estimation=True` (по умолчанию) |

---

## 2. `MahalanobisDistance` — декорреляция признаков

### 2.1. Зачем нужен

Евклидово расстояние (L2) чувствительно к **корреляции** и **масштабу** признаков. Если два признака сильно коррелируют, их совместное влияние завышается. Махаланобис-расстояние учитывает ковариационную структуру данных:

$$d_M(x, y) = \sqrt{(x - y)^T \Sigma^{-1} (x - y)}$$

Вместо явного вычисления попарных расстояний, HypEx строит **линейное преобразование** $M = \Sigma^{-1/2}$, проецирует данные в пространство с единичной ковариацией, и затем использует обычный L2 в этом пространстве.

### 2.2. Алгоритм

```
1. Объединённая ковариация:
   Σ = (Σ_control + Σ_test) / 2

2. Регуляризация:
   Σ = Σ + εI  (ε = 1e-3, защита от вырожденности)

3. Разложение Холецкого:
   L = cholesky(Σ)

4. Обратная матрица:
   M = L⁻¹

5. [Опционально] Веса признаков:
   W = sqrt(diag(w / Σw))
   M = M · W

6. Транспонирование:
   Mᵀ  (для использования в dot())
```

### 2.3. Реализация

`MahalanobisDistance` наследует `Calculator` и использует три расширения из `scipy_linalg`:

```python
# comparators/distances.py → _inner_function()
cov = UniteCovExtension().calc(data, test_data)
cholesky = CholeskyExtension().calc(cov)
mahalanobis_transform = InverseExtension().calc(cholesky)

if weights is not None:
    features = data.columns
    w_list = np.array([weights[col] if col in weights else 1 for col in features])
    w_matrix = np.sqrt(np.diag(w_list / w_list.sum()))
    mahalanobis_transform = mahalanobis_transform.dot(w_matrix)

return mahalanobis_transform.transpose()
```

Результат сохраняется в `ExperimentData.variables`:

```python
def _set_value(self, data, value, key=None):
    data.set_value(ExperimentDataEnum.variables, self.id, value, self.key)
    return data
```

### 2.4. Как результат используется в FAISS

В `FaissNearestNeighbors.execute()` матрица извлекается из `variables` и передаётся в `calc()` как `mahalanobis`:

```python
mahalanobis = next(iter(
    data.variables[
        data.get_one_id(MahalanobisDistance, ExperimentDataEnum.variables)
    ].values()
)) if len(data.get_ids(MahalanobisDistance, ExperimentDataEnum.variables)[
    "MahalanobisDistance"]["variables"]
) > 0 else None
```

Внутри `FaissExtension._mahalanobis_transform()`:

```python
@staticmethod
def _mahalanobis_transform(data, mahalanobis):
    if mahalanobis is None:
        return data
    mahalanobis_index = list(mahalanobis.index)
    valid_cols = [col for col in data.columns if col in mahalanobis_index]
    if valid_cols and len(valid_cols) < len(data.columns):
        data = data[valid_cols]
    return data.dot(mahalanobis.raw_data)
```

---

## 3. `FaissNearestNeighbors` — оркестратор матчинга

### 3.1. Класс-обёртка

`FaissNearestNeighbors` наследует `MLExecutor` и определяет **стратегию матчинга**:

| Параметр | Тип | Описание |
|----------|-----|----------|
| `n_neighbors` | `int` | Число соседей (по умолчанию 1) |
| `two_sides` | `bool` | Двусторонний матчинг: `metric == "ate"` |
| `test_pairs` | `bool` | Матчинг только тест → контроль: `metric == "atc"` |
| `faiss_mode` | `"base" \| "fast" \| "auto"` | Режим FAISS-индекса |

### 3.2. Режимы матчинга

| `metric` | `two_sides` | `test_pairs` | Что матчится |
|----------|-------------|--------------|--------------|
| `"ate"` | `True` | `False` | Тест→Контроль **и** Контроль→Тест |
| `"atc"` | `False` | `True` | Тест→Контроль |
| `"att"` | `False` | `False` | Тест→Контроль |

### 3.3. `_execute_inner_function` — логика матчинга

```python
@classmethod
def _execute_inner_function(cls, grouping_data, tmp_roles, target_field=None,
                            n_neighbors=None, two_sides=None, test_pairs=None,
                            faiss_mode="auto", **kwargs):
    (_control_idx, _data), (_test_idx, _test_data), *_ = grouping_data
    _data.tmp_roles = tmp_roles

    if test_pairs is not True:
        # Тест → Контроль
        test_data = cls._inner_function(
            data=_data, test_data=_test_data,
            n_neighbors=n_neighbors or 1,
            faiss_mode=faiss_mode, **kwargs,
        )
        if two_sides is not True:
            return {"test": test_data}
        # Контроль → Тест (для ATE)
        control_data = cls._inner_function(
            data=_test_data, test_data=_data,
            n_neighbors=n_neighbors or 1,
            faiss_mode=faiss_mode, **kwargs,
        )
        return {"test": test_data, "control": control_data}

    # Только Контроль → Тест (для ATC)
    data = cls._inner_function(
        data=_test_data, test_data=_data,
        n_neighbors=n_neighbors or 1,
        faiss_mode=faiss_mode, **kwargs,
    )
    if two_sides is not True:
        return {"control": data}
    return {
        "control": data,
        "test": cls._inner_function(data=grouping_data[1][1],
                                     test_data=grouping_data[0][1], ...),
    }
```

### 3.4. Обработка результатов в `execute()`

После получения результатов матчинга:

1. **Замена NaN** на `-1` (метка «нет соседа»)
2. **Приведение типов** к `int`
3. **Предупреждение** если есть `-1`
4. **Переименование** колонок в `indexes_{i}`
5. **Назначение роли** `AdditionalMatchingRole`
6. **Объединение** результатов через `append()`
7. **Выравнивание** по индексу исходного датасета (`reindex`)
8. **Checkpoint** для обрезки DAG
9. **Unpersist** временных данных

```python
nans = 0
for group, result in compare_result.items():
    nans += sum(result.count_nulls().values())
result = result.fillna(-1).astype({col: int for col in result.columns})

if nans > 0:
    warn(f"Faiss returned {nans} nans, which were replaced with dummy matches.",
         UserWarning)
```

---

## 4. FAISS-расширения

### 4.1. `PandasFaissExtension` — in-memory матчинг

Для данных, помещающихся в память драйвера:

```
1. Извлечение признаков: X = data.raw_data.values
2. [Опционально] Махаланобис-проекция: X = X @ M
3. Создание индекса:
   ├── Если данных > 1M и faiss_mode="auto" или "fast":
   │   └── IndexIVFFlat(quantizer, d, nlist) + train(X)
   └── Иначе:
       └── IndexFlatL2(d)  (точный поиск)
4. Обёртка в IndexIDMap для хранения исходных индексов
5. Добавление данных: index.add_with_ids(X, ids)
6. Поиск: dist, indexes = index.search(X_query, k=n_neighbors)
7. Дедупликация при n_neighbors > 1: _prepare_indexes()
```

#### Дедупликация соседей

При `n_neighbors > 1` FAISS может вернуть дубликаты, если несколько точек находятся на одинаковом расстоянии. `_prepare_indexes()` сортирует кандидатов по расстоянию и выбирает топ-`k` уникальных:

```python
@staticmethod
def _prepare_indexes(index, dist, k):
    new = np.vstack([
        np.concatenate([
            val[np.where(dist[i] == d)[0]]
            for d in sorted(set(dist[i]))[:k]
        ])
        for i, val in enumerate(index)
    ])
    return new
```

### 4.2. `SparkFaissExtension` — распределённый матчинг

Для больших данных, распределённых по кластеру. Пайплайн состоит из трёх фаз:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ФАЗА 1: ВЕКТОРИЗАЦИЯ                              │
│                                                                      │
│  Входные признаки → VectorAssembler → _features (одна колонка)      │
│  Или: F.array(*cols) → _features (для shuffle mode)                 │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ФАЗА 2: FIT (построение индекса)                  │
│                                                                      │
│  Режим определяется MatchingConfig.FAISS_FIT_MODE:                  │
│                                                                      │
│  "sample"   → Обучение IVF на случайной выборке                     │
│  "cluster"  → Обучение на всём датасете (итеративный кластеринг)     │
│  "full"     → Точный поиск: свой IndexFlatL2 на каждой партиции     │
│  "shuffle"  → Ко-группировка по кластерам (по умолчанию)             │
│                                                                      │
│  Результат: RDD сериализованных индексов → _sharded_rdd              │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ФАЗА 3: PREDICT (поиск соседей)                   │
│                                                                      │
│  "shuffle" → _predict_shuffle():                                     │
│    1. Кластеризация тестовых данных                                  │
│    2. Ко-группировка по (cluster, c_bucket, t_bucket)               │
│    3. _crossover_search() на каждой паре                            │
│                                                                      │
│  Остальные → _predict_transmission():                                │
│    1. Сериализация индексов → файлы .index                          │
│    2. Распределение через SparkFiles                                 │
│    3. _per_partition_predict() на каждом экзекьюторе                │
│    4. LRU-кэш для избежания повторной десериализации                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3. Режимы fit детально

#### `"sample"` (по умолчанию для малых данных)

```
1. Случайная выборка: sample(fraction=FAISS_SAMPLE_TARGET/data_size, seed)
2. Сбор на драйвер: collect()
3. Обучение IVF: IndexIVFFlat(quantizer, d, nlist).train(X_sample)
4. Распределение: каждая партиция строит IndexIDMap(clone(index))
5. Сериализация и персист в RDD
```

#### `"cluster"` (полное обучение)

```
1. Итеративный кластеринг на драйвере:
   for batch in vectorized_data.rdd.mapPartitions(_partition_load):
       model.partial_fit(batch)  # MiniBatchKMeans или BIRCH
2. Центроиды → IVF-квантизатор
3. Распределение: каждая партиция строит IndexIDMap
4. Сериализация и персист в RDD
```

#### `"full"` (точный поиск)

```
Каждая партиция получает СВОЙ IndexFlatL2:
  → Нет обучения квантизатора
  → Нет общего индекса
  → Точный поиск, но медленнее
```

#### `"shuffle"` (по умолчанию, оптимизированный)

```
1. Обучение квантизатора через _prefit() (как в "cluster")
2. Кластеризация контрольных данных:
   → _clustered_control: (index, _features, _cluster)
3. Кластеризация тестовых данных:
   → _clustered_test: (index, _features, _cluster[])
4. Вычисление размеров бакетов:
   → _control_clusters_dict: {cluster_id: n_buckets}
   → _test_clusters_dict: {cluster_id: n_buckets}
5. Ко-группировка по (cluster, c_bucket, t_bucket):
   → Каждая задача обрабатывает только один кластер
   → Память ограничена BUCKET_SIZE
6. _crossover_search() на каждой паре:
   → IndexFlatL2 на контроле
   → search по тесту
7. Агрегация соседей:
   → n_neighbors=1: min(dists)
   → n_neighbors>1: top-k по dists
```

### 4.4. Конфигурация через `MatchingConfig`

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `FAISS_FIT_MODE` | `"shuffle"` | Режим построения индекса |
| `FAISS_SAMPLE_TARGET` | `5_000_000` | Размер выборки для `"sample"` |
| `FAISS_DRIVER_INDEX_LIMIT` | `5_000_000` | Лимит строк на драйвер в `"cluster"` |
| `FAISS_CHUNK_SIZE` | `4096` | Размер батча запросов на экзекьюторе |
| `FAISS_N_PROBES` | `8` | Число зондов в `"shuffle"` (аналог `nprobe`) |
| `BUCKET_SIZE` | `250_000` | Макс. размер ко-группированной задачи |
| `CACHING_INDEX_MAX_SIZE` | `5` | Размер LRU-кэша на экзекьюторе |
| `FAISS_PERSIST_POLITIC` | `MEMORY_AND_DISK` | Уровень персиста в Spark |

### 4.5. LRU-кэш на экзекьюторах

В режиме `"transmission"` (все режимы кроме `"shuffle"`) каждый экзекьютор должен загрузить **все** индексы партиций. Для избежания повторной десериализации используется `CachingIndex`:

```python
# extensions/faiss.py → _per_partition_predict()
cache = get_executor_cache()  # LRU-кэш в builtins

for ref in references:  # все .index файлы
    tmp_index = cache.get(ref, storage)  # из кэша или из файла
    dists, nids = tmp_index.search(batch, k)
    # ... агрегация кандидатов ...
```

`CachingIndex` — потокобезопасный `OrderedDict` с ограничением размера. При переполнении вытесняется **наименее недавно использованный** индекс.

---

## 5. `BiasExtension` — коррекция остаточного смещения

### 5.1. Зачем нужен

Матчинг по признакам не гарантирует **полного** выравнивания распределений. Остаточное смещение корректируется линейной регрессией:

```
Для контрольной группы (group_1):
    bias_c = (X - X_matched) · coefficients_control

Для тестовой группы (group_2):
    bias_t = (X - X_matched) · coefficients_test
```

где:
- `X` — исходные признаки наблюдения
- `X_matched` — признаки матчинг-партнёра (усреднённые по всем соседям)
- `coefficients` — коэффициенты линейной регрессии `target ~ features`

### 5.2. Pandas-реализация

```
1. Подготовка данных (_prepare_data):
   a. Извлечение индексов соседей (AdditionalMatchingRole колонки)
   b. Исключение меток -1 (нет соседа)
   c. Извлечение признаков соседей по индексам
   d. Агрегация: среднее по всем соседям для каждого наблюдения
   e. Переименование колонок: {col} → {col}_matched

2. Расчёт коэффициентов (_calc_coefs):
   a. Для каждой группы:
      - Фильтрация строк с NaN в матченных данных
      - Проверка достаточности данных (≥ n_features + 1)
      - np.linalg.lstsq(X_with_intercept, y)
      - Извлечение весов (без интерсепта)
   b. Если данных недостаточно → нулевые коэффициенты

3. Расчёт смещения (_calc_bias):
   a. diff = X_matched - X  (для каждой группы)
   b. bias = diff @ coefficients
   c. NaN → 0 (нет матчинга → нет коррекции)
   d. matched_target: сохраняем оригинал (с NaN для не-матченных)

4. Результат:
   DataFrame с колонками:
   - "bias": скорректированное смещение
   - "matched_target": целевая переменная матчинг-партнёра
```

### 5.3. Spark-реализация

Аналогичная логика, но с использованием распределённых операций:

```
1. _prepare_data:
   a. explode(array(*index_cols)) → длинные строки
   b. join с исходными данными по индексу
   c. groupBy("initial_index").agg(mean(col)) → агрегация
   d. persist(MEMORY_AND_DISK) + checkpoint

2. _calc_coefs:
   a. VectorAssembler(inputCols=features, outputCol="_features")
   b. LinearRegression(featuresCol, labelCol, regParam=0.01)
   c. repartition(group_field) → оптимизация джойнов
   d. fit на каждой группе отдельно
   e. model.coefficients → веса

3. _calc_bias:
   a. Spark SQL выражения:
      F.when(group == group_1, Σ (X_matched - X) * coef)
      F.when(group == group_2, Σ (X_matched - X) * coef)
   b. select(index, bias, matched_target)
```

### 5.4. Обработка не-матченных наблюдений

Если наблюдение не получило соседа (`-1` в индексах), его `matched_target` остаётся `NaN`. Это **не** заменяется нулём — `MatchingMetrics` исключает такие наблюдения из оценки эффектов:

```python
# PandasBisaExtesion._calc_bias()
nan_mask = ~np.isfinite(matched_target)
n_unmatched = int(nan_mask.sum())
if n_unmatched > 0:
    warnings.warn(
        f"BiasExtension: {n_unmatched} of {len(data)} observations "
        f"have no valid match. They will be excluded from the "
        f"treatment effect estimation.",
        UserWarning,
    )
```

---

## 6. `MatchingMetrics` — расчёт ATT, ATC, ATE

### 6.1. Индивидуальные эффекты

Для каждого наблюдения вычисляется **индивидуальный эффект**:

**Контрольная группа** (наблюдение не получало лечение, но имеем матчинг-партнёра из теста):

$$ITC_i = Y_i^{matched} - Y_i - bias_i$$

**Тестовая группа** (наблюдение получило лечение):

$$ITT_i = Y_i - Y_i^{matched} + bias_i$$

где $Y_i^{matched}$ — исход матчинг-партнёра, $bias_i$ — коррекция смещения.

### 6.2. Масштабирование весов

При `n_neighbors > 1` одно наблюдение может быть соседом для нескольких наблюдений. `scaled_counts` корректирует веса:

$$w_j = \frac{\text{число раз, когда } j \text{ был выбран как сосед}}{n\_neighbors}$$

Статистики `sum` и `sq_sum` вычисляются по `scaled_counts` на **полных** данных (до фильтрации по `valid_mask`), чтобы сохранить веса доноров.

### 6.3. Итоговые метрики

| Метрика | Формула | Стандартная ошибка |
|---------|---------|-------------------|
| **ATT** | $\bar{ITT}$ (среднее по тесту) | $\sqrt{f_s^c \cdot Var_{ITC}/n_c + Var_{ITT}/n_t}$ |
| **ATC** | $\bar{ITC}$ (среднее по контролю) | $\sqrt{Var_{ITC}/n_c + f_s^t \cdot Var_{ITT}/n_t}$ |
| **ATE** | $\frac{n_t \cdot \bar{ITT} + n_c \cdot \bar{ITC}}{n_t + n_c}$ | $\sqrt{\frac{Var_{ITC}(m + 2\Sigma w_c + \Sigma w_c^2) + Var_{ITT}(n + 2\Sigma w_t + \Sigma w_t^2)}{N^2}}$ |

где $f_s^c = m \cdot \Sigma w_c^2 / n^2$, $f_s^t = n \cdot \Sigma w_t^2 / m^2$.

### 6.4. P-values и доверительные интервалы

```
z = estimate / SE
p = 2 * (1 - Φ(|z|))        # двусторонний, нормальное приближение
CI = estimate ± 1.96 * SE   # 95% доверительный интервал
```

### 6.5. Обработка вырожденных случаев

Если одна из групп пуста после фильтрации не-матченных наблюдений:

```python
# matching_metric.py → _calc_metrics()
if m == 0 or n == 0:
    nan_row = [float("nan")] * 5
    return {"ATT": nan_row, "ATC": nan_row, "ATE": nan_row}
```

Это предотвращает `KeyError` в нижестоящих репортёрах.

### 6.6. Режимы `metric`

| Режим | Что возвращает |
|-------|---------------|
| `"att"` | Только ATT |
| `"atc"` | Только ATC |
| `"ate"` (по умолчанию) | ATT, ATC и ATE |

---

## 7. `MatchingAnalyzer` — форматирование результатов

`MatchingAnalyzer` транспонирует словарь переменных в табличный формат:

```python
class MatchingAnalyzer(Executor):
    def execute(self, data):
        variables = data.variables[
            data.get_one_id(MatchingMetrics, space=ExperimentDataEnum.variables)
        ]
        columns = ["Effect Size", "Standard Error", "P-value",
                   "CI Lower", "CI Upper"]
        return self._set_value(
            data,
            DatasetAdapter.to_dataset(
                variables,
                {field: StatisticRole(float) for field in variables},
            ).transpose(roles={column: StatisticRole(float) for column in columns}),
        )
```

Результат — таблица, где **строки** = метрики (ATT, ATC, ATE), **колонки** = статистики:

```
        Effect Size  Standard Error  P-value  CI Lower  CI Upper
ATT         12.34           2.11    0.001      8.20     16.48
ATC         10.87           1.95    0.002      7.05     14.69
ATE         11.61           1.43    0.000      8.81     14.41
```

---

## 8. Quality Tests — проверка качества матчинга

### 8.1. Идея

После матчинга распределения признаков в контроле и тесте должны **совпадать**. Если нет — матчинг некачественный.

### 8.2. Реализация

Quality tests выполняются как `OnRoleExperiment` с `compare_by="matched_pairs"`:

```python
executors += [
    OnRoleExperiment(
        executors=[
            TTest(compare_by="matched_pairs",
                  grouping_role=TreatmentRole(),
                  baseline_role=AdditionalMatchingRole()),
            KSTest(compare_by="matched_pairs",
                   grouping_role=TreatmentRole(),
                   baseline_role=AdditionalMatchingRole()),
            Chi2Test(compare_by="matched_pairs",
                     grouping_role=TreatmentRole(),
                     baseline_role=AdditionalMatchingRole()),
        ],
        role=FeatureRole(),
    )
]
```

### 8.3. Режим `matched_pairs` в `StatsComparator`

Для каждой пары (наблюдение, его матчинг-партнёр) выполняется самоджойн:

```
1. Объединение: group_col + targets + match-index
2. Self-join по match-index → колонки с суффиксом "_matched"
3. Агрегация обоих наборов колонок
4. Сравнение: _inner_function(stats_original, stats_matched)
```

### 8.4. Интерпретация результатов

| Результат | Интерпретация |
|-----------|---------------|
| `p-value > 0.05` | Распределения **не различаются** → матчинг хороший |
| `p-value < 0.05` | Распределения **различаются** → матчинг плохой |

---

## 9. `MatchingOutput` — извлечение результатов

### 9.1. Атрибуты

| Атрибут | Тип | Содержимое |
|---------|-----|------------|
| `summary` | `Dataset` | Таблица эффектов: строки = `ATT`, `ATC`, `ATE`; колонки = `Effect Size`, `Standard Error`, `P-value`, `CI Lower`, `CI Upper` |
| `quality_results` | `Dataset` | P-values quality tests по признакам |
| `full_data` | `Dataset` | Полный датасет с `*_matched_{i}` колонками (если `extract_full_data=True`) |
| `indexes` | `Dataset` | Индексы матчинга (если `compute_indexes=True`) |

### 9.2. Оптимизация для Spark

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
    return indexes.rename({
        col: f"indexes_{col.split(ID_SPLIT_SYMBOL)[-1]}"
        for col in indexes.columns
    })
```

### 9.3. Построение `full_data`

Если `extract_full_data=True`, для каждой позиции соседа выполняется:

```
1. Извлечение индексов: t_indexes = indexes.iloc[:, i]
2. Поиск матченных строк:
   ├── Pandas: label-based lookup по индексу
   └── Spark: ленивый джойн по _hypex_lookup
3. Переименование: {col} → {col}_matched_{i}
4. Left-join с исходным датасетом по индексу
5. [Только Spark] Checkpoint после каждого merge
```

**Важно:** для Spark `ds_reset` (результат `reset_index()`) вычисляется **один раз** вне цикла и чекапоинтится. Без этого каждый `merge` встраивал бы всю линейку `experiment_data.ds`, приводя к $O(N^2)$ росту плана.

### 9.4. Форматирование `summary`

```python
self.summary.raw_data = self.summary.raw_data.round(2)
```

Все значения округляются до 2 знаков для удобства отображения.

---

## 10. Диаграмма потока данных для полного Matching-пайплайна

```
                        ┌─────────────────────────────────────────┐
                        │         Входной Dataset                  │
                        │  treat | revenue | age | gender | ...   │
                        └──────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
            ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
            │ DummyEncoder  │     │ TypeCaster    │     │ (пропуск если │
            │ gender → dummies│    │ int → float   │     │  l2 distance) │
            │ industry → dum │     │               │     │               │
            └───────┬───────┘     └───────┬───────┘     └───────────────┘
                    │                      │
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │ MahalanobisDistance  │
                    │ Σ → L → M = L⁻¹     │
                    │ → variables[id]     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ FaissNearestNeighbors│
                    │                     │
                    │  fit:               │
                    │   контроль →        │
                    │   MAхаланобис →     │
                    │   FAISS-индекс      │
                    │                     │
                    │  predict:           │
                    │   тест →            │
                    │   Махаланобис →     │
                    │   search(k) →       │
                    │   индексы соседей   │
                    │                     │
                    │  → AdditionalMatch  │
                    │    Role колонки     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
            ┌───────────────┐     ┌───────────────┐
            │ [если         │     │ [если         │
            │  bias_est     │     │  bias_est     │
            │  = True]      │     │  = False]     │
            │               │     │               │
            │ BiasExtension │     │ matched_target│
            │ регрессия →   │     │ = target      │
            │ bias +        │     │ (без bias)    │
            │ matched_target│     │               │
            └───────┬───────┘     └───────┬───────┘
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ MatchingMetrics      │
                    │                     │
                    │ ITC = Y_matched     │
                    │     - Y - bias      │
                    │                     │
                    │ ITT = Y             │
                    │     - Y_matched     │
                    │     + bias          │
                    │                     │
                    │ scaled_counts       │
                    │ ATT, ATC, ATE       │
                    │ SE, p-value, CI     │
                    │                     │
                    │ → variables[id]     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ MatchingAnalyzer     │
                    │ transpose()          │
                    │ → analysis_tables   │
                    │                     │
                    │   ATT  | ES | SE | p│
                    │   ATC  | ES | SE | p│
                    │   ATE  | ES | SE | p│
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
            ┌───────────────┐     ┌───────────────┐
            │ OnRoleExper.  │     │ MatchingOutput│
            │ quality tests │     │               │
            │ (matched_pairs)│    │ summary       │
            │               │     │ quality_res   │
            │ TTest         │     │ [full_data]   │
            │ KSTest        │     │ [indexes]     │
            │ Chi2Test      │     │               │
            │ → p-values    │     └───────────────┘
            └───────────────┘
```

---

## 11. Пример использования

### 11.1. Базовый матчинг

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

# Махаланобис + FAISS + bias correction + ATE
m = Matching()
result = m.execute(ds)

print(result.summary)
#        Effect Size  Standard Error  P-value  CI Lower  CI Upper
# ATT         12.34           2.11    0.001      8.20     16.48
# ATC         10.87           1.95    0.002      7.05     14.69
# ATE         11.61           1.43    0.000      8.81     14.41

print(result.quality_results)
# p-values для каждого признака после матчинга
```

### 11.2. Расширенный: несколько соседей, L2, конкретные тесты

```python
m_adv = Matching(
    distance="l2",              # без Махаланобиса
    n_neighbors=3,              # 3 ближайших соседа
    bias_estimation=True,       # коррекция смещения
    quality_tests=["t-test", "ks-test"],  # конкретные тесты
    faiss_mode="fast",          # форсированный IVF-индекс
    extract_full_data=True,     # извлечь полный датасет
    compute_indexes=True,       # извлечь индексы матчинга
)
result = m_adv.execute(ds)

print(result.summary)
print(result.full_data)       # полный датасет с *_matched_{i} колонками
```

### 11.3. Групповой матчинг

```python
ds_grouped = Dataset(
    roles={
        TargetRole(): ["revenue"],
        TreatmentRole(): "treat",
        FeatureRole(): ["age", "gender"],
        GroupingRole(): ["segment"],  # группировка по сегментам
    },
    data="segmented_data.csv",
)

m_group = Matching(group_match=True)  # GroupExperiment
result = m_group.execute(ds_grouped)
# Матчинг выполняется отдельно внутри каждого сегмента
```

### 11.4. Матчинг с весами признаков

```python
m_weighted = Matching(
    distance="mahalanobis",
    weights={
        "age": 2.0,          # возраст в 2 раза важнее
        "gender": 0.5,       # пол в 2 раза менее важен
        # остальные признаки: вес = 1.0
    },
)
result = m_weighted.execute(ds)
```

---

## 12. Производительность и рекомендации

### 12.1. Выбор режима

| Размер данных | Рекомендуемый режим | Обоснование |
|---------------|-------------------|-------------|
| < 100K строк | `faiss_mode="base"` | Точный поиск, минимум накладных расходов |
| 100K – 1M строк | `faiss_mode="auto"` | Автоматический выбор по размеру |
| > 1M строк | `faiss_mode="fast"` или `"shuffle"` | IVF-индекс или ко-группировка |
| > 10M строк | `MatchingConfig.FAISS_FIT_MODE="shuffle"` | Минимальная память на задачу |

### 12.2. Настройка `MatchingConfig`

```python
from hypex.config import MatchingConfig

# Для очень больших данных: уменьшить размер бакетов
MatchingConfig.BUCKET_SIZE = 100_000

# Для повышения точности: увеличить зонды
MatchingConfig.FAISS_N_PROBES = 16

# Для экономии памяти на экзекьюторах: уменьшить кэш
MatchingConfig.CACHING_INDEX_MAX_SIZE = 3
```

### 12.3. Оптимизация извлечения

```python
# ПЛОХО для Spark: собирает все данные на драйвер
m = Matching(extract_full_data=True, compute_indexes=True)

# ХОРОШО для Spark: ленивые проекции, без материализации
m = Matching(extract_full_data=False, compute_indexes=False)
```

### 12.4. Очистка ресурсов

`SparkFaissExtension` реализует `__del__` и `unpersist()` для освобождения Spark-ресурсов:

```python
def unpersist(self):
    if self._sharded_rdd is not None:
        self._sharded_rdd.unpersist(blocking=True)
    if self._clustered_control is not None:
        self._clustered_control.unpersist(blocking=True)
    if self._clustered_test is not None:
        self._clustered_test.unpersist(blocking=True)
```

---

*Документация актуальна для версии **2.0.0a1**. Следующий раздел — «Репортёры, аналитики и извлечение результатов».*
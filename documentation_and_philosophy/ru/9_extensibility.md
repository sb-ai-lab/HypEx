# Расширяемость: создание кастомных экспериментов и блоков (Уровни 5–6)

---

## 1. Обзор расширяемой архитектуры

Уровни 5 и 6 определяют границы, на которых пользователь может **не использовать** готовые оболочки (`AATest`, `ABTest`, `Matching`, `HomogeneityTest`), а **собирать собственные пайплайны** из готовых блоков или **создавать новые блоки** наследованием.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       УРОВЕНЬ 5: Сборка пайплайна                        │
│                                                                          │
│  Пользователь комбинирует готовые Executor'ы в Experiment:               │
│  Experiment(executors=[GroupSizes(), TTest(), ABAnalyzer()])             │
│                                                                          │
│  Типы экспериментов:                                                     │
│  • Experiment          — последовательный                                │
│  • OnRoleExperiment    — итерация по ролям                               │
│  • ParamsExperiment    — sweep по параметрам                             │
│  • CycledExperiment    — N итераций                                      │
│  • GroupExperiment     — по группам                                      │
│  • IfParamsExperiment  — sweep с ранним выходом                          │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       УРОВЕНЬ 6: Создание блоков                         │
│                                                                          │
│  Пользователь наследует от базовых классов:                              │
│  • Calculator        → новая чистая функция                              │
│  • Transformer       → мутация датасета                                  │
│  • GroupsComparator  → сравнение на сырых данных                         │
│  • StatsComparator   → сравнение на агрегатах                            │
│  • MLExecutor        → fit/predict                                       │
│  • Extension         → бэкенд-специфичная реализация                     │
│                                                                          │
│  + Регистрация через @backend_factory.register                           │
│  + Создание кастомных ролей                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Уровень 5: Сборка кастомного пайплайна

### 2.1. Простейший пайплайн из готовых блоков

Минимальный кастомный эксперимент — это `Experiment` со списком `Executor`'ов:

```python
from hypex.comparators import GroupDifference, GroupSizes, TTest
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments.base import Experiment, OnRoleExperiment

my_experiment = Experiment(executors=[
    # 1. Размеры групп
    GroupSizes(grouping_role=TreatmentRole()),
    # 2. Сравнение для каждой таргет-колонки
    OnRoleExperiment(
        executors=[
            GroupDifference(grouping_role=TreatmentRole()),
            TTest(grouping_role=TreatmentRole(), reliability=0.05),
        ],
        role=TargetRole(),
    ),
])
```

Каждый `Executor` получает `ExperimentData`, мутирует его и возвращает. `Experiment.execute()` последовательно вызывает `execute()` каждого исполнителя.

### 2.2. Полный пример: A/B-тест с поправкой на множественное тестирование

```python
from hypex.analyzers.ab import ABAnalyzer
from hypex.comparators import GroupDifference, GroupSizes, KSTest, TTest
from hypex.dataset import TargetRole, TreatmentRole
from hypex.experiments.base import Experiment, OnRoleExperiment
from hypex.transformers.na_dropper import NaDropper
from hypex.utils import ABNTestMethodsEnum

ab_experiment = Experiment(executors=[
    # 0. Очистка данных
    NaDropper(target_roles=TargetRole()),
    # 1. Размеры групп
    GroupSizes(grouping_role=TreatmentRole()),
    # 2. Итерация по таргетам: разница средних + тесты
    OnRoleExperiment(
        executors=[
            GroupDifference(grouping_role=TreatmentRole()),
            TTest(grouping_role=TreatmentRole(), reliability=0.05),
            KSTest(grouping_role=TreatmentRole(), reliability=0.05),
        ],
        role=TargetRole(),
    ),
    # 3. Агрегация + поправка на множественное тестирование
    ABAnalyzer(
        multitest_method=ABNTestMethodsEnum.holm,
        alpha=0.05,
    ),
])
```

### 2.3. Добавление Output

Для извлечения результатов нужен `Output` с `Reporter`'ами:

```python
from hypex.dataset import ExperimentData
from hypex.reporters.abstract import DatasetReporter, DictReporter
from hypex.reporters.ab import ABTestReporter
from hypex.ui.base import Output

class MyOutput(Output):
    """Кастомный Output для нашего эксперимента."""
    multitest: str

    def __init__(self):
        super().__init__(
            summary_reporter=ABTestReporter(
                dict_reporter=DictReporter(),
                output_format="dataset",
                invert_pass=True,
            ),
        )

    def extract(self, experiment_data: ExperimentData):
        super().extract(experiment_data)
        # Извлекаем результат multitest
        multitest_id = experiment_data.get_one_id("ABAnalyzer", "analysis_tables")
        if "MultiTest" in multitest_id:
            self.multitest = experiment_data.analysis_tables[multitest_id]
        else:
            self.multitest = "Поправка не применена"
```

### 2.4. Обёртка в `ExperimentShell`

Для удобства использования оборачиваем в `ExperimentShell`:

```python
from hypex.ui.base import ExperimentShell

class MyABTest(ExperimentShell):
    def __init__(self):
        super().__init__(
            experiment=ab_experiment,
            output=MyOutput(),
        )

# Использование
result = MyABTest().execute(data)
print(result.summary)
print(result.multitest)
```

### 2.5. Использование `ParamsExperiment` — sweep по параметрам

`ParamsExperiment` выполняет декартово произведение параметров:

```python
from hypex.experiments.base_complex import ParamsExperiment
from hypex.reporters import AATestReporter, DatasetReporter, DictReporter
from hypex.splitters import AASplitter
from hypex.comparators.abstract import Comparator
from hypex.dataset import AdditionalTreatmentRole
from hypex.utils import SpaceEnum

params = {
    AASplitter: {
        "random_state": [0, 1, 2, 3, 4],  # 5 итераций
        "control_size": [0.5],
        "sample_size": [None],
        "groups_sizes": [None],
    },
    Comparator: {
        "grouping_role": [AdditionalTreatmentRole()],
        "space": [SpaceEnum.additional],
    },
}

aa_sweep = ParamsExperiment(
    executors=[base_experiment],  # ваш базовый пайплайн
    params=params,
    reporter=DatasetReporter(
        AATestReporter(dict_reporter=DictReporter(front=False), output_format="dict"),
        single_row=True,
    ),
)
```

### 2.6. Использование `GroupExperiment` — по группам

```python
from hypex.experiments.base_complex import GroupExperiment
from hypex.dataset import GroupingRole
from hypex.reporters.matching import MatchingDatasetReporter

grouped_experiment = GroupExperiment(
    executors=[
        # ... ваши исполнители ...
    ],
    reporter=MatchingDatasetReporter(),
    searching_role=GroupingRole(),
)
```

### 2.7. Использование `CycledExperiment` — N итераций

```python
from hypex.experiments.base_complex import CycledExperiment
from hypex.reporters import DatasetReporter

cycled = CycledExperiment(
    executors=[
        # ... ваши исполнители ...
    ],
    reporter=my_reporter,
    n_iterations=100,
)
```

### 2.8. Использование `OnRoleExperiment` — итерация по ролям

`OnRoleExperiment` разделяет исполнители на **векторные** (обрабатывают все таргеты за один проход) и **итеративные** (по одной колонке):

```python
from hypex.comparators import Chi2Test, KSTest, TTest
from hypex.comparators.comparators import GroupDifference
from hypex.dataset import FeatureRole, TargetRole
from hypex.experiments.base import OnRoleExperiment

on_role = OnRoleExperiment(
    executors=[
        GroupDifference(grouping_role=TreatmentRole()),
        TTest(grouping_role=TreatmentRole()),
        KSTest(grouping_role=TreatmentRole()),
        Chi2Test(grouping_role=TreatmentRole()),
    ],
    role=[TargetRole(), FeatureRole()],
)
```

На **Spark** мастер-классы (`TTest`, `KSTest`, `Chi2Test`) резолвятся в векторные `Stats*`-реализации. На **Pandas** — в итеративные `Group*`-реализации.

### 2.9. Сводная таблица типов `Experiment`

| Класс | Назначение | Ключевые параметры |
|-------|------------|-------------------|
| `Experiment` | Последовательный пайплайн | `executors`, `transformer`, `key` |
| `OnRoleExperiment` | Итерация по колонкам роли | `executors`, `role` |
| `ParamsExperiment` | Sweep по декартову произведению параметров | `executors`, `reporter`, `params`, `stopping_criterion` |
| `IfParamsExperiment` | Sweep с ранним выходом | `executors`, `reporter`, `params`, `stopping_criterion` |
| `CycledExperiment` | N итераций с репортёром | `executors`, `reporter`, `n_iterations` |
| `GroupExperiment` | Исполнение по группам | `executors`, `reporter`, `searching_role` |

---

## 3. Уровень 6: Создание кастомного Executor'а

### 3.1. Наследование от `Calculator` — чистая функция

`Calculator` — простейший блок. Реализует `_inner_function` и `search_types`:

```python
from __future__ import annotations
from typing import Any

from hypex.dataset import Dataset, ExperimentData, StatisticRole
from hypex.dataset.dataset import SmallDataset
from hypex.executor import Calculator
from hypex.utils import ExperimentDataEnum


class MyVarianceChecker(Calculator):
    """Проверяет дисперсию таргет-колонок и сохраняет результат."""

    def __init__(self, threshold: float = 1.0, key: str = ""):
        super().__init__(key=key)
        self.threshold = threshold

    @property
    def search_types(self) -> list[type] | None:
        """Типы колонок, которые обрабатывает этот блок."""
        return [int, float]

    @staticmethod
    def _inner_function(data: Dataset, threshold: float = 1.0, **kwargs) -> dict:
        """Чистая логика: проверяет дисперсию каждой колонки."""
        result = {}
        for col in data.columns:
            variance = float(data[col].var())
            result[col] = {
                "variance": variance,
                "is_high": variance > threshold,
            }
        return result

    def _set_value(self, data: ExperimentData, value: Any, key: Any = None) -> ExperimentData:
        """Сохраняем результат в variables."""
        return data.set_value(
            ExperimentDataEnum.variables,
            self.id,
            value,
            key="variance_check",
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        """Точка входа пайплайна."""
        result = self.calc(data.ds, threshold=self.threshold)
        return self._set_value(data, result)
```

**Ключевые моменты:**
- `_inner_function` — **статический** метод с чистой логикой
- `search_types` определяет, какие колонки обрабатываются
- `execute()` вызывает `calc()` и сохраняет результат через `_set_value()`

### 3.2. Наследование от `Transformer` — мутация датасета

`Transformer` **заменяет** датасет внутри `ExperimentData`:

```python
from __future__ import annotations

from hypex.dataset import Dataset, ExperimentData, TargetRole
from hypex.transformers.abstract import Transformer


class LogTransformer(Transformer):
    """Применяет логарифм к числовым таргетам."""

    def __init__(self, target_roles=None, key: str = ""):
        super().__init__(key=key)
        self.target_roles = target_roles or TargetRole()

    @property
    def search_types(self):
        return [int, float]

    @staticmethod
    def _inner_function(data: Dataset, target_cols: list[str], **kwargs) -> Dataset:
        """Применяет логарифм к указанным колонкам."""
        import numpy as np

        result = data  # deepcopy не нужен — Dataset иммутабелен при операциях
        for col in target_cols:
            # log(x + 1) для избежания -inf при нуле
            result[col] = (result[col] + 1).log()
        return result

    def execute(self, data: ExperimentData) -> ExperimentData:
        """Точка входа: ищем колонки, применяем трансформацию."""
        target_cols = data.ds.search_columns(
            roles=self.target_roles,
            search_types=self.search_types,
        )
        if not target_cols:
            return data
        # ВАЖНО: data.copy(data=new_ds) — заменяем датасет
        result = data.copy(data=self.calc(data=data.ds, target_cols=target_cols))
        return result
```

**Ключевые моменты:**
- `_is_transformer` автоматически `True`
- `execute()` вызывает `data.copy(data=new_ds)` — **замена** датасета
- `Experiment` обнаруживает трансформер и делает `deepcopy` перед пайплайном

### 3.3. Наследование от `GroupsComparator` — сравнение на сырых данных

`GroupsComparator` работает с **сырыми срезами** датасета. Реализует `_inner_function` и `search_types`:

```python
from __future__ import annotations
from typing import Literal

import numpy as np

from hypex.dataset import Dataset, ABCRole
from hypex.comparators.abstract import GroupHypothesisTesting
from hypex.utils.constants import NUMBER_TYPES_LIST


class GroupMedianTest(GroupHypothesisTesting):
    """Сравнение медиан двух групп (аналог t-теста для медиан)."""

    def __init__(
        self,
        compare_by: Literal["groups", "columns", "columns_in_groups", "cross", "matched_pairs"] = "groups",
        grouping_role: ABCRole | None = None,
        target_role: ABCRole | None = None,
        reliability: float = 0.05,
        key: str = "",
    ):
        super().__init__(
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_role=target_role,
            reliability=reliability,
            key=key,
        )

    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> dict:
        """Сравниваем медианы двух групп."""
        test_data = cls._check_test_data(test_data)

        median_a = float(data.median())
        median_b = float(test_data.median())

        # Простой бутстрэп для p-value
        reliability = kwargs.get("reliability", 0.05)
        diff = abs(median_a - median_b)

        # Упрощённый тест: используем нормальное приближение
        se_a = float(data.std() / np.sqrt(len(data)))
        se_b = float(test_data.std() / np.sqrt(len(test_data)))
        se = np.sqrt(se_a**2 + se_b**2)

        if se == 0:
            p_value = 1.0 if diff == 0 else 0.0
        else:
            from scipy.stats import norm
            z = diff / se
            p_value = float(2 * norm.sf(abs(z)))

        return {
            "p-value": p_value,
            "statistic": diff,
            "pass": p_value < reliability,
            "median_a": median_a,
            "median_b": median_b,
        }
```

**Ключевые моменты:**
- Наследуем от `GroupHypothesisTesting` (добавляет `reliability`)
- `_inner_function` принимает два `Dataset` — baseline и compared
- `search_types` определяет типы колонок для обработки
- Результат — словарь с `p-value`, `statistic`, `pass`

### 3.4. Наследование от `StatsComparator` — сравнение на агрегатах

`StatsComparator` работает с **предварительно агрегированными статистиками**:

```python
from __future__ import annotations
from typing import Any, ClassVar, Literal
import math

import numpy as np
from scipy.stats import t as t_dist

from hypex.dataset import ABCRole
from hypex.comparators.abstract import StatsHypothesisTesting
from hypex.utils.constants import NUMBER_TYPES_LIST


class StatsMedianTest(StatsHypothesisTesting):
    """Тест медиан на агрегированных статистиках (Spark-оптимизированный)."""

    REQUIRED_STATS: ClassVar[list[str]] = ["mean", "std", "count", "median"]

    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        compare_by: Literal["groups", "matched_pairs"] = "groups",
        target_roles: ABCRole | None = None,
        reliability: float = 0.05,
        key: str = "",
    ):
        super().__init__(
            stats=self.REQUIRED_STATS,
            compare_by=compare_by,
            grouping_role=grouping_role,
            target_roles=target_roles,
            key=key,
            reliability=reliability,
        )

    @property
    def search_types(self) -> list[type] | None:
        return NUMBER_TYPES_LIST

    @classmethod
    def _inner_function(
        cls,
        baseline_stats: dict[str, Any],
        compared_stats: dict[str, Any],
        reliability: float = 0.05,
        **kwargs,
    ) -> dict[str, Any]:
        """Сравнение на агрегированных статистиках."""
        n1 = baseline_stats["count"]
        n2 = compared_stats["count"]

        # Граничный случай
        if n1 < 2 or n2 < 2:
            return {"p-value": None, "statistic": None, "pass": None}

        # Используем медианы (если доступны) или средние как прокси
        median_1 = baseline_stats.get("median", baseline_stats["mean"])
        median_2 = compared_stats.get("median", compared_stats["mean"])

        # SE через дисперсии
        se = math.sqrt(
            baseline_stats["std"]**2 / n1 + compared_stats["std"]**2 / n2
        )

        if se == 0:
            if median_1 == median_2:
                return {"p-value": 1.0, "statistic": 0.0, "pass": False}
            return {"p-value": 0.0, "statistic": float("inf"), "pass": True}

        diff = abs(median_1 - median_2)
        z = diff / se
        df = n1 + n2 - 2
        p_value = float(2 * t_dist.sf(abs(z), df))

        return {
            "p-value": p_value,
            "statistic": float(z),
            "pass": p_value < reliability,
        }
```

**Ключевые моменты:**
- `REQUIRED_STATS` определяет, какие статистики нужны для фазы 1
- `_inner_function` получает **словари агрегатов**, а не сырые данные
- На Spark это **один** `groupBy().agg()` вместо N×M джобов

### 3.5. Наследование от `MLExecutor` — fit/predict

`MLExecutor` для задач с этапами обучения и предсказания:

```python
from __future__ import annotations
from typing import Any

from hypex.dataset import (
    ABCRole,
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    TargetRole,
)
from hypex.executor import MLExecutor
from hypex.utils import ExperimentDataEnum


class SimpleDistanceMatcher(MLExecutor):
    """Простейший матчинг по евклидову расстоянию."""

    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_role: ABCRole | None = None,
        n_neighbors: int = 1,
        key: str = "",
    ):
        super().__init__(
            grouping_role=grouping_role,
            target_role=target_role or FeatureRole(),
            key=key,
        )
        self.n_neighbors = n_neighbors
        self._index = None

    def fit(self, X: Dataset, Y: Dataset | None = None) -> "SimpleDistanceMatcher":
        """Строим индекс (сохраняем контрольную группу)."""
        self._index = X.raw_data.values
        self._index_labels = X.index.tolist()
        return self

    def predict(self, X: Dataset) -> Dataset:
        """Находим ближайших соседей."""
        import numpy as np
        import pandas as pd

        query = X.raw_data.values
        distances = np.sqrt(((query[:, None] - self._index[None, :])**2).sum(axis=-1))
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        result_df = pd.DataFrame(
            [[self._index_labels[j] for j in row] for row in nearest_indices],
            index=X.index,
            columns=[f"neighbor_{i}" for i in range(self.n_neighbors)],
        )
        return Dataset(
            roles={col: AdditionalMatchingRole() for col in result_df.columns},
            data=result_df,
        )

    @classmethod
    def _inner_function(cls, data: Dataset, test_data: Dataset | None = None,
                        target_data: Dataset | None = None, **kwargs) -> Any:
        """Основная логика матчинга."""
        matcher = cls(n_neighbors=kwargs.get("n_neighbors", 1))
        matcher.fit(data)
        return matcher.predict(test_data)

    @classmethod
    def _execute_inner_function(cls, grouping_data, tmp_roles, target_field=None, **kwargs):
        (_, _data), (_, _test_data), *_ = grouping_data
        _data.tmp_roles = tmp_roles
        return cls._inner_function(data=_data, test_data=_test_data, **kwargs)
```

### 3.6. Сводная таблица типов наследования

| Базовый класс | Что реализовать | Ключевые свойства | Когда использовать |
|---------------|----------------|-------------------|-------------------|
| `Calculator` | `_inner_function`, `search_types`, `execute` | `_is_transformer=False` | Чистое вычисление без мутации данных |
| `Transformer` | `_inner_function`, `search_types` | `_is_transformer=True` | Мутация датасета (фильтры, кодирование) |
| `GroupsComparator` | `_inner_function`, `search_types` | `compare_by`, `grouping_role` | Сравнение на сырых данных (Pandas) |
| `StatsComparator` | `_inner_function`, `REQUIRED_STATS`, `search_types` | `stats`, `compare_by` | Сравнение на агрегатах (Spark) |
| `MLExecutor` | `fit`, `predict`, `_inner_function` | `grouping_role`, `target_role` | Задачи с обучением |
| `IfExecutor` | `check_rule` | `if_executor`, `else_executor` | Условное исполнение |

---

## 4. Создание кастомной роли

### 4.1. Наследование от `ABCRole`

```python
from hypex.dataset.roles import ABCRole, AdditionalRole
from hypex.utils import RoleNameType


class SegmentationRole(ABCRole):
    """Роль для колонок сегментации пользователей."""

    _role_name: RoleNameType = "Segmentation"

    def __init__(self, segment_type: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.segment_type = segment_type

    def __repr__(self) -> str:
        return f"SegmentationRole(segment_type={self.segment_type}, data_type={self.data_type})"
```

### 4.2. Создание Additional-варианта

```python
class AdditionalSegmentationRole(AdditionalRole, SegmentationRole):
    """Additional-вариант роли сегментации."""

    _role_name: RoleNameType = "AdditionalSegmentation"
```

### 4.3. Регистрация в `default_roles`

```python
from hypex.dataset.roles import default_roles

# Добавляем в глобальный реестр
default_roles["segmentation"] = SegmentationRole()
default_roles["additionalsegmentation"] = AdditionalSegmentationRole()
```

### 4.4. Использование в `search_columns`

```python
from hypex.dataset import Dataset

ds = Dataset(
    roles={
        SegmentationRole(): "user_segment",
        TargetRole(): "revenue",
        TreatmentRole(): "treat",
    },
    data=my_dataframe,
)

# Поиск по кастомной роли
segments = ds.search_columns(SegmentationRole())
# → ["user_segment"]

# Использование в пайплайне
my_executor = MyExecutor(grouping_role=SegmentationRole())
```

### 4.5. Полная иерархия ролей для справки

```
ABCRole
├── IndexRole, InfoRole, DisabledRole, StratificationRole
├── GroupingRole, TreatmentRole, TargetRole, ConstGroupRole
├── StatisticRole, ResumeRole, FilterRole, ReportRole, DefaultRole
├── LagRole (parent, lag)
│   ├── FeatureRole
│   └── PreTargetRole
├── TempRole
│   ├── TempTargetRole, TempTreatmentRole, TempGroupingRole
└── AdditionalRole
    ├── AdditionalTreatmentRole, AdditionalGroupingRole
    ├── AdditionalTargetRole, AdditionalFeatureRole
    ├── AdditionalMatchingRole, AdditionalVarianceReductionRole
    └── AdditionalStatisticRole
```

---

## 5. Создание кастомного расширения (Extension)

### 5.1. Иерархия Extension'ов

```python
# hypex/extensions/abstract.py

class Extension(ABC):
    """Базовый класс всех расширений."""
    @staticmethod
    def result_to_dataset(result, roles, small=True):
        return DatasetAdapter.to_dataset(result, roles=roles, small=small)

    def calc(self, data: Dataset, **kwargs):
        raise NotImplementedError


class CompareExtension(Extension, ABC):
    """Расширение для сравнения двух датасетов."""
    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs):
        raise NotImplementedError


class MLExtension(Extension):
    """Расширение для ML-задач."""
    @abstractmethod
    def fit(self, X, Y=None, **kwargs): ...

    @abstractmethod
    def predict(self, X, **kwargs): ...

    def calc(self, data, mode=None, **kwargs):
        if mode in ["auto", "fit"]:
            return self.fit(data, **kwargs)
        return self.predict(data, **kwargs)
```

### 5.2. Пример: свой статистический тест как Extension

```python
from __future__ import annotations
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu

from hypex.dataset import Dataset, SmallDataset, StatisticRole
from hypex.dataset.backends import PandasDataset, SparkDataset
from hypex.extensions.abstract import CompareExtension
from hypex.utils.registry import backend_factory


class MyRobustTest(CompareExtension):
    """Мастер-класс: робастный тест на основе бутстрэпа."""

    def __init__(self, reliability: float = 0.05, n_bootstrap: int = 1000):
        super().__init__()
        self.reliability = reliability
        self.n_bootstrap = n_bootstrap

    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs):
        raise NotImplementedError("Используйте бэкенд-реализацию")


@backend_factory.register(MyRobustTest, PandasDataset)
class PandasMyRobustTest(MyRobustTest):
    """Pandas-реализация через scipy."""

    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs) -> SmallDataset:
        if other is None:
            raise ValueError("other dataset is required")

        x = data._to_numpy().flatten()
        y = other._to_numpy().flatten()

        # Убираем NaN
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        if len(x) < 2 or len(y) < 2:
            return SmallDataset.from_dict(
                {"p-value": [None], "statistic": [None], "pass": [None]},
                StatisticRole(),
            )

        # Бутстрэп-тест
        rng = np.random.default_rng(42)
        observed_diff = np.mean(x) - np.mean(y)
        combined = np.concatenate([x, y])
        n_x = len(x)

        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            perm = rng.permutation(combined)
            bootstrap_diffs.append(np.mean(perm[:n_x]) - np.mean(perm[n_x:]))

        p_value = float(np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff)))

        return SmallDataset.from_dict(
            {"p-value": [p_value], "statistic": [observed_diff],
             "pass": [p_value < self.reliability]},
            StatisticRole(),
        )


@backend_factory.register(MyRobustTest, SparkDataset)
class SparkMyRobustTest(MyRobustTest):
    """Spark-реализация: делегируем в Pandas после сбора."""

    def calc(self, data: Dataset, other: Dataset | None = None, **kwargs) -> SmallDataset:
        # Для малых данных собираем на драйвер
        # Для больших — нужна агрегация на Spark
        from hypex.utils import BackendsEnum

        if len(data) <= 100_000 and len(other) <= 100_000:
            pandas_data = data.to_backend(BackendsEnum.pandas)
            pandas_other = other.to_backend(BackendsEnum.pandas)
            return PandasMyRobustTest(self.reliability, self.n_bootstrap).calc(
                pandas_data, pandas_other
            )

        # Для больших данных: агрегация → аналитический расчёт
        stats_data = data.agg(["mean", "std", "count"])
        stats_other = other.agg(["mean", "std", "count"])
        # ... аналитический расчёт из агрегатов ...
        raise NotImplementedError("Агрегированный путь для больших данных")
```

### 5.3. Регистрация в `backend_factory`

```python
from hypex.utils.registry import backend_factory
from hypex.dataset.backends import PandasDataset, SparkDataset

# Декораторный синтаксис
@backend_factory.register(MyMasterClass, PandasDataset)
class PandasImpl(MyMasterClass):
    ...

@backend_factory.register(MyMasterClass, SparkDataset)
class SparkImpl(MyMasterClass):
    ...

# Явный синтаксис (для динамической регистрации)
backend_factory.register_explicit(MyMasterClass, PandasDataset, PandasImpl)
```

### 5.4. Резолвинг в рантайме

```python
# Внутри Experiment._get_executor_backend()
backend_cls = backend_factory.resolve_backend(MyMasterClass, data.ds)
# → PandasDataset → PandasImpl
# → SparkDataset  → SparkImpl
```

---

## 6. Паттерны проектирования для контрибьюторов

### 6.1. Паттерн «Мастер-класс → бэкенд-реализация»

Каждый статистический тест существует в **трёх формах**:

```python
# 1. Мастер-класс (пустая заглушка) — в comparators/comparators.py
class TTest(StatTestMasterAbstract):
    """Мастер-класс для t-теста."""
    pass

# 2. Pandas-реализация — в comparators/hypothesis_testing.py
@backend_factory.register(TTest, PandasDataset)
class GroupTTest(GroupHypothesisTesting):
    @classmethod
    def _inner_function(cls, data, test_data=None, **kwargs):
        return GroupTTestExtension(kwargs.get("reliability", 0.05)).calc(
            data, other=test_data, **kwargs
        )

# 3. Spark-реализация — в comparators/stats_hypothesis_testing.py
@backend_factory.register(TTest, SparkDataset)
class StatsTTest(StatsHypothesisTesting):
    REQUIRED_STATS = ["mean", "std", "count"]

    @classmethod
    def _inner_function(cls, baseline_stats, compared_stats, **kwargs):
        # Аналитический расчёт из агрегатов
        ...
```

### 6.2. Паттерн `_set_value` для сохранения результатов

Каждый `Executor` определяет, **куда** и **как** сохранять результаты:

```python
# В analysis_tables (результаты тестов)
def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
    return data.set_value(
        ExperimentDataEnum.analysis_tables,
        self.id,
        value,
    )

# В variables (скаляры, матрицы)
def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
    return data.set_value(
        ExperimentDataEnum.variables,
        self.id,
        value,
        key="result",
    )

# В additional_fields (синтетические колонки)
def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
    return data.set_value(
        ExperimentDataEnum.additional_fields,
        self.id,
        value,
        role=AdditionalMatchingRole(),
    )

# В groups (срезы по группам)
def _set_value(self, data: ExperimentData, value, key=None) -> ExperimentData:
    return data.set_value(
        ExperimentDataEnum.groups,
        self.id,
        value,
        key=str(group_name),
    )
```

### 6.3. Паттерн `search_types` для фильтрации колонок

`search_types` определяет, **какие типы данных** обрабатывает блок:

```python
# Числовые тесты
@property
def search_types(self) -> list[type] | None:
    return [int, float]  # NUMBER_TYPES_LIST

# Категориальные тесты
@property
def search_types(self) -> list[type] | None:
    return [str]  # CATEGORICAL_TYPES_LIST

# Любые типы
@property
def search_types(self) -> list[type] | None:
    return None  # без фильтрации
```

Используется в `OnRoleExperiment` и `BaseComparator._get_fields_data()`:

```python
target_fields_data = data.field_data_search(
    roles=self.target_roles,
    tmp_role=tmp_role,
    search_types=self.search_types,  # ← фильтрация по типу
)
```

### 6.4. Паттерн `key` для идентификации результатов

`key` — часть составного идентификатора `id`:

```
id = ClassName ┆ params_hash ┆ key
```

Примеры:
- `TTest┆┆revenue` — тест на колонке `revenue`
- `AASplitter┆rs 42┆` — сплиттер с `random_state=42`
- `GroupDifference┆┆["revenue", "spends"]` — разница для нескольких колонок

### 6.5. Паттерн `calc_kwargs` для передачи параметров

`calc_kwargs` разворачиваются в `_inner_function`:

```python
class StatsTTest(StatsHypothesisTesting):
    def __init__(self, equal_variance: bool | None = None, **kwargs):
        super().__init__(
            stats=self.REQUIRED_STATS,
            compare_by="groups",
            grouping_role=grouping_role,
            target_roles=target_roles,
            key=key,
            reliability=reliability,
            calc_kwargs={"equal_variance": equal_variance, **kwargs}
            if equal_variance is not None
            else kwargs,
        )
```

В `Experiment._get_executor_backend()`:

```python
if has_var_keyword and hasattr(executor, "calc_kwargs"):
    # Разворачиваем calc_kwargs в именованные аргументы
    init_kwargs.update(executor.calc_kwargs)
```

---

## 7. Полный пример: кастомный A/B-тест с бутстрэпом

```python
from __future__ import annotations
from typing import Any

import numpy as np

from hypex.comparators import GroupDifference, GroupSizes
from hypex.comparators.abstract import GroupHypothesisTesting
from hypex.dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    SmallDataset,
    StatisticRole,
    TargetRole,
    TreatmentRole,
)
from hypex.dataset.backends import PandasDataset, SparkDataset
from hypex.experiments.base import Experiment, OnRoleExperiment
from hypex.extensions.abstract import CompareExtension
from hypex.reporters.abstract import DatasetReporter, DictReporter
from hypex.ui.base import ExperimentShell, Output
from hypex.utils import ExperimentDataEnum
from hypex.utils.registry import backend_factory


# ─── 1. Мастер-класс ────────────────────────────────────────────────────────
class BootstrapTest(GroupHypothesisTesting):
    """Мастер-класс для бутстрэп-теста."""
    pass


# ─── 2. Pandas-реализация ───────────────────────────────────────────────────
@backend_factory.register(BootstrapTest, PandasDataset)
class GroupBootstrapTest(GroupHypothesisTesting):
    @property
    def search_types(self):
        return [int, float]

    @classmethod
    def _inner_function(cls, data, test_data=None, **kwargs):
        test_data = cls._check_test_data(test_data)
        reliability = kwargs.get("reliability", 0.05)
        n_bootstrap = kwargs.get("n_bootstrap", 1000)

        x = data._to_numpy().flatten()
        y = test_data._to_numpy().flatten()
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        observed_diff = np.mean(x) - np.mean(y)
        combined = np.concatenate([x, y])
        n_x = len(x)

        rng = np.random.default_rng()
        count = sum(
            1
            for _ in range(n_bootstrap)
            if abs(
                np.mean((perm := rng.permutation(combined))[:n_x])
                - np.mean(perm[n_x:])
            )
            >= abs(observed_diff)
        )
        p_value = count / n_bootstrap

        return {
            "p-value": p_value,
            "statistic": observed_diff,
            "pass": p_value < reliability,
        }


# ─── 3. Spark-реализация ────────────────────────────────────────────────────
@backend_factory.register(BootstrapTest, SparkDataset)
class StatsBootstrapTest(GroupHypothesisTesting):
    """Для Spark делегируем в Pandas (для малых данных)."""

    @property
    def search_types(self):
        return [int, float]

    @classmethod
    def _inner_function(cls, data, test_data=None, **kwargs):
        from hypex.utils import BackendsEnum

        pandas_data = data.to_backend(BackendsEnum.pandas)
        pandas_test = test_data.to_backend(BackendsEnum.pandas)
        return GroupBootstrapTest._inner_function(pandas_data, pandas_test, **kwargs)


# ─── 4. Сборка пайплайна ────────────────────────────────────────────────────
bootstrap_experiment = Experiment(executors=[
    GroupSizes(grouping_role=TreatmentRole()),
    OnRoleExperiment(
        executors=[
            GroupDifference(grouping_role=TreatmentRole()),
            BootstrapTest(grouping_role=TreatmentRole(), reliability=0.05),
        ],
        role=TargetRole(),
    ),
])


# ─── 5. Output ──────────────────────────────────────────────────────────────
class BootstrapOutput(Output):
    def __init__(self):
        super().__init__(
            summary_reporter=DatasetReporter(
                DictReporter(),
                output_format="dataset",
                invert_pass=True,
            ),
        )


# ─── 6. Оболочка ────────────────────────────────────────────────────────────
class BootstrapABTest(ExperimentShell):
    def __init__(self, n_bootstrap: int = 1000):
        experiment = Experiment(executors=[
            GroupSizes(grouping_role=TreatmentRole()),
            OnRoleExperiment(
                executors=[
                    GroupDifference(grouping_role=TreatmentRole()),
                    BootstrapTest(
                        grouping_role=TreatmentRole(),
                        reliability=0.05,
                        calc_kwargs={"n_bootstrap": n_bootstrap},
                    ),
                ],
                role=TargetRole(),
            ),
        ])
        super().__init__(
            experiment=experiment,
            output=BootstrapOutput(),
        )


# ─── Использование ──────────────────────────────────────────────────────────
from hypex.dataset import Dataset

ds = Dataset(
    roles={
        TargetRole(): ["revenue", "spends"],
        TreatmentRole(): "treat",
    },
    data="ab_data.csv",
)

result = BootstrapABTest(n_bootstrap=5000).execute(ds)
print(result.summary)
```

---

## 8. Чеклист для контрибьютора

### Создание нового статистического теста

- [ ] Определить мастер-класс в `comparators/comparators.py`
- [ ] Реализовать `Group*Test` в `comparators/hypothesis_testing.py` (Pandas)
- [ ] Реализовать `Stats*Test` в `comparators/stats_hypothesis_testing.py` (Spark)
- [ ] Зарегистрировать через `@backend_factory.register`
- [ ] Определить `REQUIRED_STATS` для `Stats*`
- [ ] Определить `search_types`
- [ ] Добавить в `TEST_NAME_NORMALIZATION` в `utils/constants.py`
- [ ] Добавить в `ANALYSIS_TEST_CLASSES` соответствующих анализаторов
- [ ] Написать юнит-тесты для обоих бэкендов

### Создание нового трансформера

- [ ] Наследовать от `Transformer` в `transformers/`
- [ ] Реализовать `_inner_function` (статический)
- [ ] Определить `search_types`
- [ ] Обработать пустой список колонок (ранний возврат)
- [ ] Использовать `data.copy(data=result)` в `execute()`

### Создание новой роли

- [ ] Наследовать от `ABCRole` в `dataset/roles.py`
- [ ] Определить `_role_name`
- [ ] Создать Additional-вариант (если нужен)
- [ ] Зарегистрировать в `default_roles`
- [ ] Экспортировать в `dataset/__init__.py`

---

*Документация актуальна для версии **2.0.0a1**. Следующий раздел — «Ядро и глубокая модификация (Уровни 7–8)».*
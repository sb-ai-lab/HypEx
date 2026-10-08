# `hypex.transformers` — Data Preparation

The only executor family allowed to **replace the working dataset**. Everything
else appends results; transformers change `ExperimentData.ds` itself.

## Role in the architecture

```
Transformer._is_transformer == True
        │
        └── the enclosing Experiment deep-copies ExperimentData before running,
            so the caller's dataset is never mutated in place.

Transformer.execute:  data.copy(data=self.calc(data.ds))
```

That flag is the contract. If you write a block that rewrites `ds` without it,
the copy does not happen and callers see their input mutated.

Typical position: first in the pipeline (cleaning, encoding, casting), or right
before the comparators (CUPED variance reduction).

## File map

| File | Contents |
|---|---|
| `abstract.py` | `Transformer` — the base class (12 lines; read it first). |
| `filters.py` | `CVFilter`, `ConstFilter`, `NanFilter`, `CorrFilter`, `OutliersFilter`. |
| `na_filler.py` | `NaFiller`. |
| `na_dropper.py` | `NaDropper` — **NEW**. |
| `category_agg.py` | `CategoryAggregator`. |
| `cuped.py` | `CUPEDTransformer`. |
| `type_caster.py` | `TypeCaster`. |
| `float32_caster.py` | `Float32Caster` — **NEW**. |
| `shuffle.py` | `Shuffle` (a `Calculator`, not a `Transformer` — see gotchas). |
| `__init__.py` | Exports all of the above plus `DummyEncoder`, re-exported from `hypex.encoders`. |

## Key classes

### `Transformer(Calculator)`

```python
class Transformer(Calculator):
    @property
    def _is_transformer(self): return True

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Dataset: ...

    @classmethod
    def calc(cls, data, **kwargs): return cls._inner_function(data, **kwargs)

    def execute(self, data): return data.copy(data=self.calc(data=data.ds))
```

Subclasses that need role resolution override `execute` to search columns first
and pass them into `calc` — that is what every concrete class here does.

### Filters (`filters.py`)

All take `target_roles` (default `FeatureRole()`) and drop columns or rows that
fail a criterion.

| Class | Signature | Drops |
|---|---|---|
| `CVFilter` | `(target_roles=None, lower_bound=None, upper_bound=None, key="")` | columns whose coefficient of variation falls outside the bounds |
| `ConstFilter` | `(target_roles=None, threshold=0.95, key="")` | columns where one value covers ≥ `threshold` of rows |
| `NanFilter` | `(target_roles=None, threshold=0.8, key="")` | columns whose NaN share exceeds `threshold` |
| `CorrFilter` | `(target_roles=None, corr_space_roles=None, threshold=0.8, method="pearson", numeric_only=True, key="")` | one of each pair of columns correlated above `threshold` |
| `OutliersFilter` | `(target_roles=None, lower_percentile=0, upper_percentile=1, key="")` | **rows** outside the percentile range |

Note `OutliersFilter` is the only row-wise filter in the set.

### `NaFiller` (`na_filler.py`)

`NaFiller(target_roles=None, values=None, method=None, key="")` — fills missing
values either with a constant / per-column dict (`values`) or by propagation
(`method="ffill" | "bfill"`).

### `NaDropper` (`na_dropper.py`) — **NEW**

`NaDropper(target_roles=None, how="any", key="")` — **NEW: drops rows with NaN values**
in target columns.

This is particularly useful for preparing data for statistical tests that cannot
handle missing values (e.g., `scipy.stats.ks_2samp`).

- `target_roles`: Roles to search for target columns (defaults to `TargetRole()`)
- `how`: Whether to drop rows if 'any' or 'all' targets are NaN (defaults to 'any')

### `CategoryAggregator` (`category_agg.py`)

`CategoryAggregator(target_roles=None, threshold=15, new_group_name=None, key="")` —
collapses rare categories (fewer than `threshold` occurrences) into
`new_group_name`. Operates on categorical columns only (`search_types ==
[CategoricalTypes]`). Keeps chi² tests and dummy encoding from exploding on
long-tailed categoricals.

### `CUPEDTransformer` (`cuped.py`)

`CUPEDTransformer(cuped_features: dict[str, str], key="")` where the mapping is
`{target_feature: pre_target_feature}`.

For each pair it computes `theta = cov(x, y) / (std(y) * std(x))` and adds a new
column `f"{target}_cuped" = target - (pre_target - mean(pre_target)) * theta`
with `TargetRole`. Zero or NaN variance sets `theta = 0`, so the adjustment
degrades to a no-op instead of producing NaNs.

The original target is kept; the adjusted column is added alongside, so both can
be tested and compared. Wired into `ABTest(cuped_features={...})`.

### `TypeCaster` (`type_caster.py`)

`TypeCaster(dtype, roles=None, key="")` where `dtype` is either
`{column: type}` or `{from_type: to_type}`. Used by `Matching` to force feature
columns into a numeric type FAISS can consume.

### `Float32Caster` (`float32_caster.py`) — **NEW**

**NEW: Memory optimization transformer**

`Float32Caster(target_roles=None, columns=None, key="")` — downcasts numeric 
columns from float64 to float32 to reduce memory footprint by ~2× with negligible
precision loss for most A/B-test metrics.

- Operates on columns identified by semantic roles, not by name
- Supports both Pandas and Spark backends
- Restores original `float` type in role metadata so downstream searches still work
- Can target specific roles (defaults to `[FeatureRole(), TargetRole()]`) or explicit columns

Example:
```python
from hypex.transformers import Float32Caster
from hypex.dataset import FeatureRole, TargetRole

caster = Float32Caster(target_roles=[FeatureRole(), TargetRole()])
result = caster.execute(experiment_data)
```

### `Shuffle` (`shuffle.py`)

`Shuffle(random_state=None, key="")` — row-permutes the dataset. It derives from
`Calculator`, not `Transformer`, but its `execute` does replace `ds`.

## How to work with it

### The ready-made preprocessing pipeline

`hypex/preprocessing.py` ships a tuned chain:

```python
from hypex.preprocessing import PREPROCESSING_DATA

PREPROCESSING_DATA = Experiment(executors=[
    NaFiller(method="ffill"),
    CategoryAggregator(),
    CorrFilter(),
    CVFilter(),
    NanFilter(),
    ConstFilter(),
    OutliersFilter(lower_percentile=0.05, upper_percentile=0.95),
    DummyEncoder(),
])

clean = PREPROCESSING_DATA.execute(ExperimentData(dataset))
```

### Standalone

```python
from hypex.transformers import CUPEDTransformer

adjusted = CUPEDTransformer.calc(dataset, cuped_features={"post_spends": "pre_spends"})
```

### Memory optimization example

```python
from hypex.transformers import Float32Caster
from hypex.dataset import FeatureRole, TargetRole

# Reduce memory usage for large datasets
optimizer = Float32Caster(target_roles=[FeatureRole(), TargetRole()])
optimized_data = optimizer.execute(experiment_data)
```

### In your own pipeline

```python
Experiment(executors=[
    NaFiller(method="ffill"),
    Float32Caster(),  # NEW: Add memory optimization
    CUPEDTransformer({"post_spends": "pre_spends"}),
    NaDropper(),  # NEW: Remove rows with NaN in targets
    GroupTTest(compare_by="groups", grouping_role=TreatmentRole()),
])
```

## How to add a transformer

```python
from hypex.dataset import Dataset
from hypex.transformers.abstract import Transformer


class MyTransformer(Transformer):
    def __init__(self, factor: float = 1.0, key=""):
        super().__init__(key=key)
        self.factor = factor

    @staticmethod
    def _inner_function(data: Dataset, factor: float = 1.0, **kwargs) -> Dataset:
        return data * factor   # must return a Dataset

    def execute(self, data):
        return data.copy(data=self.calc(data=data.ds, factor=self.factor))
```

Export it from `__init__.py`. If it is a cleaning step, consider adding it to
`PREPROCESSING_DATA`.

## Gotchas

* **Order matters and is not checked.** `NaFiller` before `CorrFilter` (NaNs skew
  correlations), `CategoryAggregator` before `DummyEncoder` (or you get a column
  per rare category), `OutliersFilter` late (it drops rows other filters may have
  needed).
* **Filters drop columns silently.** A comparator downstream simply finds fewer
  targets. If a result is missing, check whether a filter removed the column.
* **`CUPEDTransformer` adds, it does not replace.** After it runs there are two
  target columns per adjusted feature; role-based searches will pick up both.
* **`Shuffle` is not a `Transformer`** and therefore does not set
  `_is_transformer`. In an `Experiment` where it is the only mutating block, pass
  `transformer=True` explicitly or the input dataset will be modified.
* All of these operate on `ds` only — anything already written into
  `additional_fields` is not filtered along with it.
* **New memory optimization**: `Float32Caster` can significantly reduce memory usage
  for large datasets with many float columns.
* **New NaN handling**: `NaDropper` provides more control over row-level missing value
  handling compared to `NanFilter` which works at the column level.

## Related modules

`../encoders/README.md` (`DummyEncoder`, re-exported here) ·
`../executor/README.md` (the `Calculator` base) · `../experiments/README.md`
(the `transformer` flag and the deep copy) · `hypex/preprocessing.py`.

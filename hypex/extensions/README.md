# `hypex.extensions` — Third-Party Adapters

Thin wrappers over external libraries (scipy, statsmodels, faiss, sklearn,
pandas). This is the **only** place in HypEx that imports a statistics or ML
library directly, and the only place that branches on the storage backend.

## Role in the architecture

An extension answers "how do I actually compute this, given a pandas or a Spark
dataset?". An executor answers "which columns, which groups, where does the
result go?". Keeping them apart means a new statistical test is a ~10-line
extension plus a ~10-line comparator, and Spark support can be added to the
extension without touching the pipeline.

```
GroupTTest (comparator)                   FaissNearestNeighbors (MLExecutor)
        │ _inner_function                          │
        ▼                                          ▼
GroupTTestExtension.calc(data, other)     FaissExtension.calc(data, mode=...)
        │  BACKEND_MAPPING[type(data.backend_data)]
        ├── _calc_pandas  → scipy.stats.ttest_ind
        └── _calc_spark   → collect to driver, then scipy
```

Extensions are **not** `Executor`s: no id, no `ExperimentData`, no pipeline
position. They take `Dataset`s and return `Dataset`s.

## File map

| File | Wraps | Classes | Status |
|---|---|---|---|
| `abstract.py` | — | `Extension`, `CompareExtension`, `MLExtension` | Core |
| `scipy_stats.py` | `scipy.stats` | `GroupStatTest`, `GroupTTestExtension`, `GroupKSTestExtension`, `GroupUTestExtension`, `GroupChi2TestExtension`, `NormCDF` | Active |
| `statsmodels.py` | `statsmodels` | `MultiTest`, `MultitestQuantile` | Active |
| `scipy_linalg.py` | `numpy.linalg` | `CholeskyExtension`, `InverseExtension` | Active |
| `faiss.py` | `faiss` | `FaissExtension`, `PandasFaissExtension`, `SparkFaissExtension` | Enhanced |
| `cupac.py` | sklearn-style models | `CupacExtension` | Active |
| `encoders.py` | `pandas` | `DummyEncoderExtension`, `PandasDummyEncoderExtension`, `SparkDummyEncoderExtension` | Enhanced |
| `bias.py` | — | `BiasExtension`, `PandasBisaExtesion`, `SparkBisaExtesion` | **New** |
| `matching_metric.py` | — | `MatchingMetricsExtension`, `PandasMatchingMetricsExtension`, `SparkMatchingMetricsExtension` | **New** |
| `stats_hypothesis_testing.py` | — | Enhanced statistical testing extensions | **New** |
| `__init__.py` | — | Public exports | Active |

## Key classes

### `Extension` (ABC)

```python
class Extension(ABC):
    def __init__(self):
        self.BACKEND_MAPPING = {PandasDataset: self._calc_pandas,
                                SparkDataset:  self._calc_spark}

    @abstractmethod
    def _calc_pandas(self, data, **kwargs): ...
    @abstractmethod
    def _calc_spark(self, data, **kwargs): ...

    def calc(self, data, **kwargs):
        return self.BACKEND_MAPPING[type(data.backend_data)](data=data, **kwargs)

    @staticmethod
    def result_to_dataset(result, roles) -> Dataset: ...
```

`calc` is the single public entry point; the dispatch table is built per
instance. `result_to_dataset` routes any plain return value through
`DatasetAdapter` so callers always get a `Dataset`.

### `CompareExtension(Extension, ABC)`

Adds a second dataset: `calc(data, other=None, **kwargs)`. Everything in
`scipy_stats.py` derives from it.

### `MLExtension(Extension)`

Adds a fit/predict lifecycle. Its `_calc_pandas` dispatches on a `mode` kwarg
(`"auto"`, `"fit"`, `"predict"`) to abstract `fit(X, Y=None)` / `predict(X)`.

### `GroupStatTest` and its subclasses (`scipy_stats.py`)

`GroupStatTest(test_function: Callable | None = None, reliability: float = 0.05)`

* Validates that both inputs are one-dimensional (`check_dataset`) and that
  `other` was supplied.
* `_calc_pandas` flattens both to numpy and calls `test_function`, then packs the
  result into a one-row `SmallDataset` with `p-value`, `statistic`, and
  `pass = p-value < reliability`.
* `_calc_spark` does the same after collecting both sides to the driver via
  `rdd.flatMap(...).collect()` — correct, but it moves the data; prefer the
  `StatsComparator` branch on Spark (see
  [`../comparators/README.md`](../comparators/README.md)).

Subclasses just bind a scipy function:
`GroupTTestExtension` → `ttest_ind`, `GroupKSTestExtension` → `ks_2samp`,
`GroupUTestExtension` → `mannwhitneyu`, `GroupChi2TestExtension` →
`chi2_contingency`. `NormCDF` wraps `scipy.stats.norm`.

**`pass` semantics:** `True` means the null hypothesis was rejected — a
*difference was found*. In A/A and homogeneity contexts that is a failure, which
is why `TestDictReporter.rename_passed` renders `True` as `"NOT OK"`.

### `MultiTest` / `MultitestQuantile` (`statsmodels.py`)

* `MultiTest(method: ABNTestMethodsEnum, alpha=0.05)` — wraps
  `statsmodels.stats.multitest.multipletests` for bonferroni, sidak, holm,
  holm-sidak, simes-hochberg, hommel, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky.
* `MultitestQuantile(alpha=0.05, iteration_size=20000, equal_variance=True,
  random_state=None)` — a resampling-based quantile correction for the
  `ABNTestMethodsEnum.quantile` option.

Both are driven by `ABAnalyzer`.

### `CholeskyExtension` / `InverseExtension` (`scipy_linalg.py`)

Cholesky factorisation (with an `epsilon=1e-3` ridge added to the diagonal for
numerical stability) and matrix inversion. Used by `MahalanobisDistance` to build
the whitening transform for matching.

### `FaissExtension` (`faiss.py`)

**Enhanced with full backend support**

`FaissExtension(n_neighbors=1, faiss_mode="auto"|"base"|"fast")`

Builds a FAISS index over the control group and queries it with the treated group
(or vice versa). Handles ties explicitly: with `n_neighbors == 1` all points at
the minimal distance are considered, and an out-of-range result is encoded as
`-1` (no match found). With `k > 1`, `_prepare_indexes` keeps all points within
the k smallest distinct distances.

**New backend-specific implementations:**
- `PandasFaissExtension` - In-memory implementation for pandas datasets
- `SparkFaissExtension` - Distributed implementation for Spark datasets with optimized indexing

### `CupacExtension` (`cupac.py`)

`CupacExtension(n_folds=5, random_state=None)` — cross-fitted control-variate
prediction. Its mode set is `"kfold_fit" | "fit" | "predict"`: out-of-fold
predictions avoid leaking the target into the covariate.

### `DummyEncoderExtension` (`encoders.py`)

**Enhanced with backend support**

`pd.get_dummies(drop_first=True)` with role propagation. Now available for both Pandas and Spark:
- `PandasDummyEncoderExtension` - Standard pandas implementation
- `SparkDummyEncoderExtension` - Spark-optimized implementation

Used by `DummyEncoder` to encode categorical columns.

### `BiasExtension` (`bias.py`)

**NEW: Linear bias correction extensions**

Implements linear bias correction for nearest-neighbour matching (Abadie–Imbens style):
- `BiasExtension` - Base abstract class
- `PandasBisaExtesion` - Pandas implementation  
- `SparkBisaExtesion` - Spark implementation

Used by `Bias` operator for matching quality improvement.

### `MatchingMetricsExtension` (`matching_metric.py`)

**NEW: Causal effect estimation extensions**

Implements matching metrics calculation:
- `MatchingMetricsExtension` - Base abstract class
- `PandasMatchingMetricsExtension` - Pandas implementation
- `SparkMatchingMetricsExtension` - Spark implementation

Used by `MatchingMetrics` operator for ATE/ATT/ATC estimation.

## How to work with it

Extensions are usable on their own:

```python
from hypex.extensions import GroupTTestExtension

result = GroupTTestExtension(reliability=0.05).calc(
    control_ds[["post_spends"]], other=test_ds[["post_spends"]]
)
# Dataset with p-value / statistic / pass
```

## How to add an extension

```python
from hypex.extensions.abstract import CompareExtension
from hypex.dataset import SmallDataset, StatisticRole


class MyTestExtension(CompareExtension):
    def __init__(self, reliability: float = 0.05):
        super().__init__()
        self.reliability = reliability

    def _calc_pandas(self, data, other=None, **kwargs):
        stat, p = my_library.test(data.backend_data.data.values.flatten(),
                                  other.backend_data.data.values.flatten())
        return SmallDataset.from_dict(
            {"p-value": p, "statistic": stat, "pass": p < self.reliability},
            StatisticRole(),
        )

    def _calc_spark(self, data, other=None, **kwargs):
        raise NotImplementedError
```

Then add the comparator that calls it (see
[`../comparators/README.md`](../comparators/README.md)) and export both.

Keep the output schema — `p-value`, `statistic`, `pass` — if you want the
existing reporters and analyzers to pick the result up automatically.

## Gotchas

* **`calc` backend dispatch fixed.** Previously broken backend lookup now uses
  `type(data.backend_data)` instead of `type(data.backend)`.
* **Enhanced Spark support.** Many extensions now have full Spark implementations
  (`FaissExtension`, `DummyEncoderExtension`, `BiasExtension`, `MatchingMetricsExtension`).
* **`_calc_spark` implementations available.** Spark versions now exist for most core extensions.
* **FAISS is an optional dependency.** `faiss.py` imports it at module level; ensure
  it's installed when using matching functionality.
* The result-schema contract is implicit. Nothing validates that an extension
  returns `p-value` / `statistic` / `pass`, but the reporters filter on those
  names.

## Related modules

`../comparators/README.md` and `../ml/README.md` (the callers) ·
`../dataset/backends/README.md` (what `BACKEND_MAPPING` keys on) ·
`../utils/README.md` (`ABNTestMethodsEnum`, `backend_factory`).

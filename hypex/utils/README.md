# `hypex.utils` — Support Utilities

A collection of enums, type definitions, constants, error classes, and utility
functions that the rest of HypEx depends on.

## Role in the architecture

The things here are the **glue** between modules:

* **Enums** — the finite sets of options (`BackendsEnum`, `ABNTestMethodsEnum`, …)
* **Type aliases** — the type-hint vocabulary the codebase shares (`CategoricalTypes`,
  `FromDictTypes`, …)
* **Errors** — the exception hierarchy (`NotFoundInExperimentDataError`, …)
* **Constants** — symbols used for id building and parsing (`ID_SPLIT_SYMBOL`,
  `NAME_BORDER_SYMBOL`, …)
* **Adapters / helpers** — `DatasetAdapter`, timing helpers, data generators
* **New utilities** — backend factory, logging, indexing, Spark configuration

```
Every import into any other package    ┌──────────────────────────────┐
should be from `hypex.utils`, not      │   hypex.utils                   │
from a submodule directly.           │  (the only cross-package deps) │
                                        └──────────────────────────────┘
```

## File map

| File | Contents |
|---|---|
| `enums.py` | `BackendsEnum`, `SpaceEnum`, `ExperimentDataEnum`, `ABNTestMethodsEnum`, `ABTestTypesEnum`. |
| `typings.py` | `CategoricalTypes`, `FromDictTypes`, `GroupingDataType`, `SparkTypeMapper`, etc. |
| `errors.py` | The exception hierarchy used across the codebase. |
| `constants.py` | `ID_SPLIT_SYMBOL`, `NAME_BORDER_SYMBOL`, `UTILITY_*_COL_*`. |
| `adapter.py` | `DatasetAdapter` — static converters that turn plain values into `Dataset`s. |
| `naming.py` | Helpers for parsing composite metric column names. |
| `registry.py` | `BackendFactory` — **NEW: Automatic backend-specific implementation selection**. |
| `logger.py` | `HypExLogger`, `ProcessContext`, `logger` — **NEW: Enhanced logging framework**. |
| `index_utils.py` | `FaissIndexStorage`, `CachingIndex` — **NEW: FAISS indexing utilities**. |
| `spark_config.py` | `SparkSessionCalculator` — **NEW: Spark session configuration**. |
| `profiling.py` | `ProfilingContext`, `timeit`, `enable_profiling`, `disable_profiling` — **NEW: Performance profiling**. |
| `strict_abc.py` | `StrictABC`, `StrictABCMeta` — **NEW: Enhanced ABC implementation**. |
| `tutorial_data_creation.py` | Data generators for tutorials and testing. |
| `cuped_theta.py` | CUPED theta calculation utilities. |
| `models.py` | Model registry for ML executors. |
| `__init__.py` | Public re-exports of everything above. |

## Key components

### Enums (`enums.py`)

| Enum | Members | Used by |
|---|---|---|
| `BackendsEnum` | `pandas`, `pyspark` | Backend selection |
| `SpaceEnum` | `data`, `additional`, `analysis`, `variables`, `groups` | `ExperimentData` namespaces |
| `ExperimentDataEnum` | mirrors `SpaceEnum` for enum-style access | Data space identification |
| `ABNTestMethodsEnum` | `bonferroni`, `sidak`, `holm`, `holm-sidak`, `simes-hochberg`, `hommel`, `fdr_*`, `quantile` | Multi-testing correction in `ABAnalyzer` |
| `ABTestTypesEnum` | `t-test`, `u-test`, `chi2-test`, `z-test`, `ks-test` | Test type selection |

### Type aliases (`typings.py`)

Shorthand for type signatures shared across the codebase:

* `CategoricalTypes` — the union of string-like types treated as categoricals.
* `FromDictTypes` — `dict | Dataset | SmallDataset`.
* `GroupingDataType` — `tuple[tuple[str, Dataset], ...]` — the shape `GroupedDataset`
  returns when iterated.
* `SparkTypeMapper` — maps Python types to Spark SQL types.
* Various `*RoleTypes` — unions used in role-based column lookups.

### Exception hierarchy (`errors.py`)

```
HypExError
    ├── DataError
    │       ├── RoleColumnError           # role references missing column
    │       ├── ConcatDataError            # append / merge mismatch
    │       ├── DataTypeError              # incompatible dtype
    │       └── MergeOnError               # merge requires `on=`
    │
    ├── BackendTypeError                 # wrong backend passed
    ├── ConcatBackendError                # mismatched backends in append/merge
    │
    ├── ExperimentDataError
    │       ├── SpaceError                 # wrong space argument
    │       └── NotFoundInExperimentDataError # id lookup failed
    │
    ├── ExecutorError
    │       ├── NoColumnsError             # role search returned []
    │       ├── NoRequiredArgumentError    # missing mandatory arg
    │       ├── NotSuitableFieldError      # wrong number of groups / cols
    │       └── AbstractMethodError        # unimplemented abstract method
    │
    └── PairsNotFoundError                # FAISS matching found no pairs
```

### Constants (`constants.py`)

| Constant | Value | Purpose |
|---|---|---|
| `ID_SPLIT_SYMBOL` | `"┴"` | Joins `ClassName`, `params_hash`, `key` to build `executor.id` |
| `NAME_BORDER_SYMBOL` | `"┆"` | Joins name parts inside one id segment (e.g. `"group┆column"`) |
| `UTILITY_COL_SYMBOL` | `"⏣"` | Prefix for internal utility columns in Spark |
| `UTILITY_INDEX_COL_NAME` | `"⏣index"` | Row-index emulation column |
| `UTILITY_NEW_INDEX_COL_NAME` | `"⏣new_index"` | Temporary column used by the Spark `index` setter |
| `UTILITY_PHYSICAL_INDEX_COL_NAME` | `"⏣_physical_index"` | Physical index column |

### Adapters (`adapter.py`)

`DatasetAdapter` provides static methods that turn any plain Python value into a
`Dataset`, dispatching on type: `value_to_dataset`, `dict_to_dataset`,
`list_to_dataset`, `frame_to_dataset`, `ndarray_to_dataset`.

Blocks that return scalars or plain dicts route their return through this so the
pipeline only ever sees `Dataset`s.

### New Backend Factory (`registry.py`) — **NEW**

`BackendFactory` enables automatic selection of backend-specific implementations:

```python
from hypex.utils.registry import backend_factory

# Register backend-specific implementations
@backend_factory.register(MasterClass, PandasDataset)
class PandasImpl(MasterClass):
    def _calc_pandas(self, data, **kwargs): ...

@backend_factory.register(MasterClass, SparkDataset) 
class SparkImpl(MasterClass):
    def _calc_spark(self, data, **kwargs): ...

# Resolve at runtime based on dataset backend
resolved_cls = backend_factory.resolve_backend(MasterClass, dataset)
```

This powers the enhanced backend support in extensions and executors.

### Enhanced Logging (`logger.py`) — **NEW**

`HypExLogger` provides a structured logging framework:

- **`logger`** — Pre-configured logger instance for library-wide logging
- **`HypExLogger`** — Main logger class with configurable levels and formats
- **`ProcessContext`** — Context manager for structured process logging
- **Method logging** — Automatic logging of method calls via `@logger.log_methods` decorator

Example:
```python
from hypex.utils.logger import logger

logger.info("Processing started")
# Structured logging with context
with ProcessContext("data_processing"):
    # Operations are automatically logged
    result = expensive_operation()
```

### Indexing Utilities (`index_utils.py`) — **NEW**

FAISS-specific indexing utilities:

- **`FaissIndexStorage`** — Manages FAISS index storage and retrieval
- **`CachingIndex`** — Provides caching mechanisms for FAISS indexes

These utilities improve performance and memory management for large-scale matching.

### Spark Configuration (`spark_config.py`) — **NEW**

`SparkSessionCalculator` provides intelligent Spark session configuration:

- Automatic resource allocation based on available system resources
- Optimized configuration for different workload types
- Memory and parallelism tuning

### Performance Profiling (`profiling.py`) — **NEW**

Performance monitoring utilities:

- **`ProfilingContext`** — Context manager for profiling code blocks
- **`timeit`** — Decorator for timing method execution
- **`enable_profiling()` / `disable_profiling()`** — Global profiling controls

Example:
```python
from hypex.utils.profiling import timeit, ProfilingContext

@timeit
def expensive_operation():
    # This will be timed automatically
    pass

with ProfilingContext() as profiler:
    result = expensive_operation()
    print(profiler.get_stats())  # Get timing statistics
```

### Enhanced ABC (`strict_abc.py`) — **NEW**

`StrictABC` and `StrictABCMeta` provide enhanced abstract base class functionality:

- Ensures all abstract methods are implemented
- Better error messages for missing implementations
- Support for abstract properties and class methods

### Tutorial Data Creation (`tutorial_data_creation.py`)

Utility functions for creating test data:

- `create_test_data()` — Generate synthetic datasets for testing
- `gen_control_variates_df()` — Create control variate data
- `gen_oracle_df()` — Generate oracle data for validation
- `gen_special_medicine_df()` — Create specialized test datasets

### CUPED Utilities (`cuped_theta.py`)

`cuped_theta()` calculates the optimal theta parameter for CUPED variance reduction.

### Model Registry (`models.py`)

`CUPAC_MODELS` registry maps model names to model classes for CUPAC analysis.

## How to work with it

Import from the package, not from submodules:

```python
from hypex.utils import (
    BackendsEnum,           # enums
    CategoricalTypes,       # typings
    RoleColumnError,        # errors
    ID_SPLIT_SYMBOL,        # constants
    DatasetAdapter,         # adapters
    backend_factory,        # NEW: backend factory
    logger,                 # NEW: logging
    timeit,                 # NEW: profiling
)
```

## How to extend

* **New enum** — add it to `enums.py` and export from `__init__.py`.
* **New error** — subclass the closest existing error in `errors.py`.
* **New constant** — add to `constants.py` and document it here.
* **New type alias** — add to `typings.py` and export from `__init__.py`.
* **New utility** — create a new file and export from `__init__.py`.

## Gotchas

* **`ID_SPLIT_SYMBOL` / `NAME_BORDER_SYMBOL` are single unicode characters.**
  Never hardcode them as strings — always import from `hypex.utils` to avoid
  copy-paste drift.
* **Backend factory is new** — Use `backend_factory.resolve_backend()` for backend-specific
  implementation selection instead of manual type checking.
* **Logging is configurable** — Use the provided `logger` instance for consistent logging
  across the library.
* **Profiling affects performance** — Disable profiling in production or for 
  performance-critical code.

## Recent Additions and Improvements

### ✅ **New Backend Factory System**
- Automatic selection of backend-specific implementations
- Clean separation between abstract interfaces and concrete implementations
- Improved maintainability and extensibility

### ✅ **Enhanced Logging Framework**
- Structured logging with context management
- Method-level logging via decorators
- Configurable verbosity and formatting

### ✅ **Performance Profiling**
- Easy-to-use timing decorators and context managers
- Performance statistics collection
- Minimal overhead when disabled

### ✅ **Spark Configuration Utilities**
- Automatic resource allocation
- Optimized configuration presets
- Better memory management

### ✅ **Indexing Utilities**
- FAISS index management and caching
- Improved performance for large-scale matching

## Related modules

Every other module in `hypex/` imports from here. The tightest coupling is with
`dataset/` (uses the enums and typings) and `executor/` (uses the errors).

# HypEx — Package Architecture Overview

This file is the entry point for the per-module documentation. Every subpackage of
`hypex/` has its own `README.md` describing its classes, their place in the
architecture and how to work with them.

## What the library does

HypEx (Hypotheses and Experiments) is a causal-inference and A/B-testing toolkit.
It is built as a **pipeline of small, composable blocks** (`Executor`s) that read
from and write into a single shared state object (`ExperimentData`), on top of a
backend-agnostic tabular structure (`Dataset`, pandas or Spark).

## Current Version: 2.0.0a1

**Architecture Status**: The new architecture is now **ACTIVE** and fully functional. 
The library has successfully transitioned from the old 0.1.10 version to the new 
modular architecture with enhanced backend support.

## The one picture to keep in mind

```
                    ┌──────────────────────────────────────────────┐
 user code          │  ExperimentShell  (AATest / ABTest /         │
 (level 4)          │   Matching / HomogeneityTest)                │
                    └───────────────┬──────────────────────────────┘
                                    │ .execute(Dataset)
                    ┌───────────────▼──────────────────────────────┐
 pipeline           │  Experiment  = Sequence[Executor]            │
 (level 5)          │  (+ OnRoleExperiment / GroupExperiment /     │
                    │     CycledExperiment / ParamsExperiment)     │
                    └───────────────┬──────────────────────────────┘
                                    │ executes each block in order
     ┌──────────────────────────────┼──────────────────────────────┐
     │        │        │        │        │        │        │       │
 transformers splitters encoders comparators operators  ml     analyzers
 (mutate ds)  (assign   (encode  (compare   (compute   (fit/   (aggregate
              groups)   cats)    groups)    metrics)   predict) results)
     │        │        │        │        │        │        │       │
     └──────────────────────────┬──────────────────────────────────┘
                                │ every block reads/writes
                    ┌───────────▼──────────────────────────────────┐
 state              │  ExperimentData                              │
                    │   .ds  .additional_fields  .analysis_tables  │
                    │   .variables  .groups                        │
                    └───────────┬──────────────────────────────────┘
                                │
                    ┌───────────▼──────────────────────────────────┐
 data layer         │  Dataset  →  PandasDataset | SparkDataset    │
                    │  + roles (TargetRole, TreatmentRole, ...)    │
                    └──────────────────────────────────────────────┘
                                │
                    ┌───────────▼──────────────────────────────────┐
 output             │  Reporter  →  Output  (resume, full_data...) │
                    └──────────────────────────────────────────────┘

 extensions ── thin adapters over scipy / statsmodels / faiss / sklearn,
               called by comparators, operators and ml blocks.
 utils      ── enums, errors, typings, constants, Adapter, data generators.
```

## Module index

| Module | Layer | Purpose | Doc |
|---|---|---|---|
| `dataset/` | data | `Dataset`, `ExperimentData`, roles, pandas/Spark backends | [README](dataset/README.md) |
| `dataset/backends/` | data | Concrete pandas and Spark implementations | [README](dataset/backends/README.md) |
| `executor/` | core | `Executor`, `Calculator`, `MLExecutor`, `IfExecutor` base classes | [README](executor/README.md) |
| `experiments/` | core | `Experiment` containers that sequence and repeat executors | [README](experiments/README.md) |
| `comparators/` | analysis | Group comparison and statistical tests | [README](comparators/README.md) |
| `operators/` | analysis | Metric operators (SMD, matching metrics, bias correction) | [README](operators/README.md) |
| `analyzers/` | analysis | Aggregate raw test results into scores/verdicts | [README](analyzers/README.md) |
| `transformers/` | preprocessing | Filters, NA filling, CUPED, type casting, shuffling | [README](transformers/README.md) |
| `encoders/` | preprocessing | Categorical encoding executors | [README](encoders/README.md) |
| `splitters/` | preprocessing | A/A group assignment (with/without stratification) | [README](splitters/README.md) |
| `ml/` | analysis | ML-based executors: FAISS matching, CUPAC | [README](ml/README.md) |
| `extensions/` | integration | Thin adapters over scipy / statsmodels / faiss | [README](extensions/README.md) |
| `reporters/` | output | Turn `ExperimentData` into flat dicts / result tables | [README](reporters/README.md) |
| `ui/` | output | `ExperimentShell` and `Output` — the user-facing facade | [README](ui/README.md) |
| `forks/` | control flow | Conditional branching inside a pipeline | [README](forks/README.md) |
| `utils/` | support | Enums, errors, typings, constants, adapters, data generators | [README](utils/README.md) |
| `factory/` | **inactive** | Reserved for config-driven pipeline construction | [README](factory/README.md) |
| `hypotheses/` | **inactive** | Reserved for JSON-described experiments | [README](hypotheses/README.md) |

## Top-level modules (not folders)

| File | Contents |
|---|---|
| `aa.py` | `AATest` shell + `AA_TEST`, `AA_METRICS`, `ONE_AA_TEST` experiment presets |
| `ab.py` | `ABTest` shell; builds its experiment dynamically from constructor args |
| `matching.py` | `Matching` shell; builds a matching pipeline from constructor args |
| `homogeneity.py` | `HomogeneityTest` shell + `HOMOGENEITY_TEST` preset |
| `preprocessing.py` | `PREPROCESSING_DATA` — a ready-made cleaning `Experiment` |
| `__version__.py` | Package version |

## Abstraction levels

`schemes/architecture_levels.md` defines eight levels of use, from a no-code
platform UI (level 1) down to core architecture work (level 8). The levels that
matter when reading this code:

* **Level 4 — shells.** Use `AATest`, `ABTest`, `Matching`, `HomogeneityTest`.
  Start at [`ui/README.md`](ui/README.md).
* **Level 5 — compose your own pipeline** from existing blocks.
  Start at [`experiments/README.md`](experiments/README.md).
* **Level 6 — write a new block** by subclassing a typed executor.
  Start at [`executor/README.md`](executor/README.md), then the module whose
  base class you are extending (usually `comparators` or `transformers`).
* **Level 7+ — change the core.** Start at [`dataset/README.md`](dataset/README.md).

## Reading order for a newcomer

1. `dataset/README.md` — you cannot read anything else without `Dataset`, roles
   and `ExperimentData`.
2. `executor/README.md` — the contract every block implements.
3. `experiments/README.md` — how blocks are sequenced.
4. `comparators/README.md` — the largest and most representative block family.
5. `reporters/README.md` + `ui/README.md` — how results come back to the user.

## Recent Updates and Improvements

### ✅ **New Architecture is Active**
- The library has successfully transitioned to the new modular architecture
- 2.0 is substantially faster than 1.0.x and adds a Spark backend; it is not fully backward compatible
  (see "Migration from 1.0.7" below)

### 🚀 **Enhanced Backend Support**
- **Spark backend**: Significantly expanded coverage for core dataset operations
- **Pandas backend**: Reference implementation with full feature support
- **Backend factory**: Automatic selection of appropriate implementations

### 📈 **New Features and Improvements**

#### **Extensions Module** (`hypex/extensions/`)
- **NEW**: `BiasExtension` - Linear bias correction for matching
- **NEW**: `MatchingMetricsExtension` - Causal effect estimation
- **ENHANCED**: Full Spark support for `FaissExtension` and `DummyEncoderExtension`
- **IMPROVED**: Better error handling and backend dispatch

#### **Comparators Module** (`hypex/comparators/`)
- **ENHANCED**: Backend-specific implementations using `backend_factory`
- **NEW**: Enhanced support for Spark datasets in statistical testing
- **IMPROVED**: Better integration with extensions

#### **ML Module** (`hypex/ml/`)
- **ENHANCED**: `FaissNearestNeighbors` with full backend support
- **IMPROVED**: `CUPACExecutor` with better model validation and error handling
- **NEW**: Optimized error handling and performance improvements

#### **Transformers Module** (`hypex/transformers/`)
- **NEW**: `NaDropper` - Row-level NaN filtering
- **NEW**: `Float32Caster` - Memory optimization via float32 downcasting
- **ENHANCED**: Improved preprocessing pipeline

#### **Operators Module** (`hypex/operators/`)
- **ENHANCED**: `MatchingMetrics` with improved metric calculation
- **ENHANCED**: `Bias` with better error handling
- **IMPROVED**: Better integration with matching workflows

#### **Analyzers Module** (`hypex/analyzers/`)
- **NEW**: `AADryTestAnalyzer` - Dry test analysis for A/A validation
- **ENHANCED**: Support for additional test classes in `OneAAStatAnalyzer`
- **IMPROVED**: Better best split selection in `AAScoreAnalyzer`

#### **UI Module** (`hypex/ui/`)
- **ENHANCED**: `MatchingOutput` with significantly improved Spark support
- **ENHANCED**: Better DAG handling and memory management
- **IMPROVED**: Enhanced error messages and reporting

## Current Status Summary

| Module | Status | Notes |
|---|---|---|
| `dataset/` | ✅ Active | Core data layer with full backend support |
| `executor/` | ✅ Active | Base classes for all executors |
| `experiments/` | ✅ Active | Pipeline containers and experiments |
| `comparators/` | ✅ Active | Enhanced with backend-specific implementations |
| `operators/` | ✅ Active | Improved metric calculation and integration |
| `analyzers/` | ✅ Active | New analyzers and enhanced features |
| `transformers/` | ✅ Active | New transformers and improvements |
| `encoders/` | ✅ Active | Enhanced with Spark support |
| `splitters/` | ✅ Active | A/A group assignment |
| `ml/` | ✅ Active | Enhanced ML executors with better error handling |
| `extensions/` | ✅ Active | New extensions and improved backend support |
| `reporters/` | ✅ Active | Result formatting and reporting |
| `ui/` | ✅ Active | Enhanced user-facing facade with better Spark support |
| `forks/` | ✅ Active | Conditional branching |
| `utils/` | ✅ Active | Support utilities |
| `factory/` | ⏸️ Inactive | Planned feature, currently dormant |
| `hypotheses/` | ⏸️ Inactive | Planned feature, currently dormant |

## Conventions used across the package

* **Roles, not column names.** Blocks never hardcode column names; they search
  columns by role (`TargetRole`, `TreatmentRole`, `FeatureRole`, …).
* **Executor id as a key.** Every result is stored under `executor.id`, a string
  built from `ClassName + params_hash + key`, joined by `ID_SPLIT_SYMBOL` (`┴`).
  Reporters parse these ids back apart, so id format is load-bearing.
* **`NAME_BORDER_SYMBOL` (`┆`)** separates composite name parts inside one id
  segment (e.g. `group┆column`, `stat┆column`).
* **Immutability by convention.** Only `Transformer`s replace `ExperimentData.ds`;
  everything else appends to `additional_fields`, `analysis_tables`, `variables`
  or `groups`.
* **Notebooks in the repo root** (`ABTestTutorial.ipynb`, `MatchingTutorial.ipynb`,
  `DatasetTutorial.ipynb`, …) are the executable counterpart to these docs.

## Migration from 1.0.7

2.0.0a1 keeps the 1.0 architecture (roles, executors, `ExperimentData`), but a few defaults and
names changed. Full list: `RELEASE_NOTES_2.0.0a1.md` in the repository root.

### Breaking Changes:
- Python 3.13 is not supported (`python >=3.8, <3.13`); PySpark 3.5.1 is now a hard dependency.
- `Output.resume` / `resume_reporter` are renamed to `summary` / `summary_reporter`; the old names
  still work and emit `DeprecationWarning`.
- A/A splits use a different hash (SipHash on pandas): the same data and seed give different groups.
- `Matching` extracts full data and matched indexes only on request
  (`extract_full_data=True`, `compute_indexes=True`).
- The KS-test drops NaN by default (`nan_policy="omit"`).
- `Dataset.data` returns a `pyspark.sql.DataFrame` on the Spark backend; use `Dataset.raw_data` for
  the pandas-on-Spark object.

### What you get:
- **Speed**: substantially faster experiments, especially on large data and in A/A loops.
- **Spark backend**: the same API on pandas and Spark.
- **New blocks**: `StatsUTest`, `NaDropper`, `Float32Caster`, matching bias correction and metrics.

Version 0.1.x is no longer supported (`pip install hypex==0.1.10`).

## Getting Help

- **Documentation**: [ReadTheDocs](https://hypex.readthedocs.io/en/latest/)
- **Community**: [Telegram Chat](https://t.me/hypexchat)
- **Examples**: Check tutorials in the repo root and examples directory
- **Issues**: Report on GitHub with detailed reproduction steps

## Next Steps

The library continues to evolve with focus on:
- ✅ **Enhanced Spark support** - Better performance and memory management
- ✅ **New ML features** - Better matching and causal inference tools  
- ✅ **Improved usability** - Better error messages and user experience
- 🔄 **Config-driven construction** - Factory module (planned feature)
- 🔄 **Declarative experiments** - Hypotheses module (planned feature)

For the latest updates, check the [GitHub repository](https://github.com/sb-ai-lab/HypEx) and [Telegram chat](https://t.me/hypexchat).

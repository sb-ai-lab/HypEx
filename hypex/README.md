# HypEx — Package Architecture Overview

This file is the entry point for the per-module documentation. Every subpackage of
`hypex/` has its own `README.md` describing its classes, their place in the
architecture and how to work with them.

## What the library does

HypEx (Hypotheses and Experiments) is a causal-inference and A/B-testing toolkit.
It is built as a **pipeline of small, composable blocks** (`Executor`s) that read
from and write into a single shared state object (`ExperimentData`), on top of a
backend-agnostic tabular structure (`Dataset`, pandas or Spark).

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
| `factory/` | inactive | Reserved for config-driven pipeline construction | [README](factory/README.md) |
| `hypotheses/` | inactive | Reserved for JSON-described experiments | [README](hypotheses/README.md) |

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

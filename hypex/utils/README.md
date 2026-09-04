# `hypex.utils` — Shared Primitives

Cross-cutting definitions with no dependency on the rest of the package: enums,
exceptions, type aliases, magic constants, small adapters, and the tutorial data
generators.

## Role in the architecture

The bottom of the dependency graph. Every other module imports from here; this
module imports nothing from HypEx (except type-checking-only references to
`Dataset`). Keep it that way — a dependency from `utils` back into the pipeline
would create an import cycle.

## File map

| File | Contents |
|---|---|
| `enums.py` | `ExperimentDataEnum`, `BackendsEnum`, `SpaceEnum`, `ABNTestMethodsEnum`, `ABTestTypesEnum`, `RenameEnum`. |
| `errors.py` | All custom exceptions. |
| `constants.py` | The separator symbols, utility column names, dtype lists. |
| `typings.py` | Type aliases and `SparkTypeMapper`. |
| `adapter.py` | `Adapter` — `to_list` / `list_to_single`. |
| `decorator.py` | `inherit_docstring_from`. |
| `models.py` | `CUPAC_MODELS` — the sklearn/catboost model registry. |
| `tutorial_data_creation.py` | `DataGenerator`, `create_test_data`, and the scenario generators. |
| `__init__.py` | Public re-exports. Note `models.py` and `decorator.py` are **not** re-exported; import them from their submodules. |

## Enums (`enums.py`)

| Enum | Members | Used for |
|---|---|---|
| `ExperimentDataEnum` | `variables`, `additional_fields`, `analysis_tables`, `groups`, `ml` | Which namespace of `ExperimentData` a result goes to. Passed to `set_value` / `get_ids`. |
| `BackendsEnum` | `pandas`, `spark` | Backend selection in `Dataset(...)`. |
| `SpaceEnum` | `auto`, `additional`, `data` | Where a comparator should look for its columns. |
| `ABNTestMethodsEnum` | `bonferroni`, `sidak`, `holm_sidak`, `holm`, `simes_hochberg`, `hommel`, `fdr_bh`, `fdr_by`, `fdr_tsbh`, `fdr_tsbky`, `quantile` | Multiple-testing correction in `ABTest` / `ABAnalyzer`. |
| `ABTestTypesEnum` | `t_test`, `ks_test`, `u_test`, `chi2_test` | The `additional_tests` argument of `ABTest`. |
| `RenameEnum` | `all`, `columns`, `index` | Scope of `Output._replace_splitters`. |

## Constants (`constants.py`)

| Constant | Value | Meaning |
|---|---|---|
| `ID_SPLIT_SYMBOL` | `┴` (U+2534) | Joins the parts of an executor id: `Class ┴ params_hash ┴ key`. |
| `NAME_BORDER_SYMBOL` | `┆` (U+2506) | Delimits a composite name inside one id segment (`group┆column`, `stat┆column`, `┆colname┆`). |
| `MATCHING_INDEXES_SPLITTER_SYMBOL` | `╯` (U+256F) | Joins matched index lists into one string cell. |
| `UTILITY_COL_SYMBOL` | `⏣` (U+23E3) | Prefix marking internal columns. |
| `UTILITY_INDEX_COL_NAME` | `⏣index` | The emulated row index on the Spark backend. |
| `UTILITY_PHYSICAL_INDEX_COL_NAME` | `⏣_physical_index` | The positional index on the Spark backend. |
| `NUMBER_TYPES_LIST` | `[int, float]` | The usual `search_types` for numeric comparators. |
| `CATEGORICAL_TYPES_LIST` | `[str]` | The categorical counterpart. |

These are deliberately unusual glyphs so they cannot collide with a real column
name. Ids and result keys are parsed by splitting on them, so they are part of
the internal wire format — see `hypex/reporters/`.

## Errors (`errors.py`)

Every exception carries a formatted message; construct them with the offending
values rather than a string.

| Exception | Raised when |
|---|---|
| `RoleColumnError(roles, columns)` | A declared role names a column that does not exist. |
| `ConcatDataError`, `ConcatBackendError` | Appending a non-`Dataset`, or datasets on different backends. |
| `BackendTypeError`, `DataTypeError` | Operating across mismatched backends / types. |
| `SpaceError(space)` | Invalid `SpaceEnum` value. |
| `NoColumnsError(role)` | No column carries the required role. |
| `NotSuitableFieldError(field, "Grouping"\|"Target"\|"Baseline")` | A field cannot serve that role (e.g. only one group present). |
| `NotFoundInExperimentDataError(class_)` | `get_one_id` found no result for that executor. |
| `AbstractMethodError` | An abstract method was called; subclasses `NotImplementedError`. |
| `MergeOnError(on)` | Invalid merge key. |
| `NoRequiredArgumentError`, `NoneArgumentError`, `InvalidArgumentError` | Missing or wrong arguments in a `calc` call. |
| `PairsNotFoundError` | Matching produced no pairs — usually missing preprocessing. |

## Typings (`typings.py`)

Role-value aliases (`TargetRoleTypes`, `FeatureRoleTypes`, `CategoricalTypes`,
`StratificationRoleTypes`, `DefaultRoleTypes`, `ScalarType`), structural aliases
(`FromDictTypes`, `GroupingDataType`, `SetParamsDictTypes`, `MultiFieldKeyTypes`,
`SourceDataTypes`), and `SparkTypeMapper`, which maps pyspark `DataType`s to
Python types (`to_python`) — extend its `_SPARK_TO_PY` table when adding a
supported dtype.

## `Adapter` (`adapter.py`)

* `to_list(data)` — `None` → `[]`, `str` → `[str]`, any other sequence → `list`,
  scalar → `[scalar]`. Used everywhere a parameter may be one value or many.
* `list_to_single(data)` — the inverse; raises `ValueError` on more than one item.

## `inherit_docstring_from` (`decorator.py`)

Copies a docstring from another callable or property onto the decorated one.
Used in the dataset backends to inherit pandas' documentation for delegating
methods.

```python
@inherit_docstring_from(pd.DataFrame.mean)
def mean(self): return self._data.mean()
```

## `CUPAC_MODELS` (`models.py`)

`{model_name: {backend_name: estimator}}` for `"linear"`, `"ridge"`, `"lasso"`,
and `"catboost"` when catboost imports successfully (`CATBOOST_AVAILABLE`).
`CUPACExecutor` validates its `cupac_models` argument against this dict.

## Tutorial data (`tutorial_data_creation.py`)

Synthetic datasets for the notebooks and tests:

* `DataGenerator` — the configurable generator class.
* `create_test_data(...)` — the general-purpose A/B-shaped dataset.
* `gen_special_medicine_df(...)`, `gen_oracle_df(...)`,
  `gen_control_variates_df(...)` — scenario-specific frames with a known ground
  truth, useful for validating that an estimator recovers the true effect.
* `set_nans(...)`, `sigmoid(...)`, `sigmoid_division(...)` — helpers.

```python
from hypex.utils import create_test_data
df = create_test_data(rs=42, na_step=10, nan_cols=["pre_spends"])
```

## How to extend

* **New enum member** — add it and then handle it everywhere the enum is
  dispatched on; `ExperimentDataEnum` in particular is switched on inside
  `ExperimentData.set_value`, `check_hash` and `get_ids`.
* **New exception** — keep the message-formatting-in-`__init__` style so call
  sites stay short.
* **New constant** — if it is a separator, pick a glyph outside the range of
  realistic column names.

## Gotchas

* **Do not change the separator constants.** Executor ids, result keys and the
  reporters' parsing all depend on the exact characters.
* `constants.py` defines `UTILITY_INDEX_COL_NAME` and
  `UTILITY_PHYSICAL_INDEX_COL_NAME` several times over (harmless duplication, but
  do not read meaning into the repetition).
* `hypex/utils/__init__.py`'s `__all__` does not match the module contents
  exactly — for example `models` and `decorator` are absent, and
  `hypex/dataset/__init__.py` lists `"indexRole"` where the class is `IndexRole`.
  Import from the defining submodule if a name does not resolve.
* `typings.py` imports pyspark at module level, so pyspark is a hard dependency
  of the package.

## Related modules

Everything. Start from [`../README.md`](../README.md).

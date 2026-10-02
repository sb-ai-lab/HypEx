# `hypex.dataset.backends` — Execution Engines

Concrete implementations of the tabular storage that `Dataset` delegates to.
This is the only place in the library that knows whether the data lives in a
pandas DataFrame in local memory or in a distributed Spark DataFrame.

## Role in the architecture

```
Dataset / DatasetBase          ← public API, roles, pipeline integration
        │ delegates every operation to
        ▼
DatasetBackendNavigation  (indexing, columns, types, IO, operators)
        │
DatasetBackendCalc        (mean, var, agg, corr, quantile, ...)
        │
   ┌────┴─────┐
PandasDataset  SparkDataset
```

`DatasetBase._select_backend_from_data` / `_select_backend_from_str` pick the
backend; nothing above this package branches on backend type, with two
exceptions that do so deliberately: `Extension.BACKEND_MAPPING` (see
[`../../extensions/README.md`](../../extensions/README.md)) and the Spark cache
methods on `DatasetBase`.

## File map

| File | Contents |
|---|---|
| `abstract.py` | `DatasetBackendNavigation` and `DatasetBackendCalc` — the contract both engines must satisfy. |
| `pandas_backend.py` | `PandasNavigation`, `PandasDataset`. The reference implementation and the default. |
| `spark_backend.py` | `SparkNavigation`, `SparkDataset`. Distributed implementation built on `pyspark.pandas`. |
| `__init__.py` | Exports `PandasDataset`, `SparkDataset`. |

## The contract (`abstract.py`)

* **`DatasetBackendNavigation`** — structural operations: `name`, `index`,
  `columns`, `from_dict`, `to_dict`, `to_records`, `__getitem__`, `__len__`,
  the full operator set (comparisons, arithmetic, reflected and unary forms),
  `create_empty`, `get_column_type`, `astype`, `update_column_type`,
  `add_column`, `append`, `loc`, `iloc`, and the index-column helpers
  `add_index_col` / `remove_index_col`.
* **`DatasetBackendCalc`** — numeric reductions and aggregations: `mean`, `mode`,
  `var`, `std`, `max`, `idxmax`, `min`, `count`, `sum`, `agg`, `quantile`,
  `corr`, `cov`, `value_counts`, `na_counts`, `isna`, and the rest.

A backend class inherits both (`class SparkDataset(SparkNavigation, DatasetBackendCalc)`).

## `PandasDataset`

Constructed from a `pd.DataFrame`, `pd.Series`, `dict` of the form
`{"data": {...}, "index": [...]}`, a path to a `.csv` / `.xlsx` file, a
`spark.DataFrame` (collected via `toPandas()`), or `None` for an empty frame.

`data_compression` (`"downcasting" | "encoding" | "auto" | "disable"`) reduces
memory: numeric downcasting plus label encoding of categorical columns, with the
mapping kept in `_labels_dict` (exposed as `labels_dict`). Columns whose declared
role has `data_type == str` are passed as `non_compresion_cols` and left alone.

## `SparkDataset`

Wraps a `pyspark.pandas.DataFrame` plus a `SparkSession`. Points worth knowing:

* **`PANDAS_CONVERSION_LIMIT = 100_000`** — a guard on operations that would
  collect the frame to the driver. Exceeding it is an error rather than an OOM.
* **`_convert_agg_result`** — a 1×1 aggregation result is returned as a plain
  `float`, larger results stay distributed. Keeps the API identical to pandas'.
* **Caching** — `persist(storage_level, action)`, `unpersist(blocking)`,
  `is_persisted`, `get_storage_level`. `action="count"` forces materialisation
  immediately so the cache is warm; `"none"` leaves it lazy.
* **Index emulation.** Spark has no row index, so a utility column
  (`UTILITY_INDEX_COL_NAME` = `⏣index`, and `⏣_physical_index`) is added and
  removed around operations that need positional access — see `add_index_col`,
  `remove_index_col`, `loc`, `iloc`, `reindex`, `transpose`.
* **Group helpers** — `groupby`, `iter_groups(by)`, `grouped_value_counts(by,
  feature_cols)` and `agg(func)` are the ones used by `StatsComparator` to keep
  the number of Spark jobs down.
* `checkpoint()` intentionally raises `NotImplementedError`.

## How to work with it

You normally do **not** instantiate these directly — build a `Dataset` and let it
choose. Reach for the backend object only when you need engine-specific escape
hatches:

```python
ds.backend_data          # the PandasDataset / SparkDataset instance
ds.backend_data.data     # the raw pd.DataFrame / ps.DataFrame
ds.backend_type          # BackendsEnum.pandas | BackendsEnum.spark
ds.session               # SparkSession or None
```

## How to add a method

1. Add the abstract signature to `DatasetBackendNavigation` or
   `DatasetBackendCalc` in `abstract.py`.
2. Implement it in `pandas_backend.py` **and** `spark_backend.py`. Keep the
   return contract identical — same shape, same scalar-vs-frame behaviour.
3. Add the delegating wrapper on `DatasetBase` (`hypex/dataset/abstract.py`).
4. Add a test under `tests/` and, for the Spark path, under
   `tests/dataset_spark_backend/`.

## Gotchas

* **Ordering.** pandas preserves row order; Spark does not unless you sort.
  Anything that compares "first group vs the rest" sorts explicitly — see
  `GroupsComparator._split_for_groups_mode`.
* **Laziness.** A Spark result is a plan, not values. Prefer one `agg()` over a
  loop of reductions; that is exactly why `StatsComparator` exists.
* **Type mapping.** `SparkTypeMapper` in `hypex/utils/typings.py` translates
  Python types to Spark types; extend it when you add a supported dtype.

## Related modules

`../README.md` (the `Dataset` API above these classes) ·
`../../utils/README.md` (`BackendsEnum`, `SparkTypeMapper`, utility column names) ·
`../../extensions/README.md` (per-backend algorithm dispatch).

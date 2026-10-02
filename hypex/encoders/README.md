# `hypex.encoders` — Categorical Encoding

A small family: executors that turn categorical columns into numeric ones so
distance-based and regression-based blocks can consume them.

## Role in the architecture

Encoders sit between the transformers and the analytical blocks. They differ from
transformers in one important way:

> A transformer **replaces** `ExperimentData.ds`. An encoder **appends** its
> output to `ExperimentData.additional_fields`, leaving the original categorical
> columns intact.

That is why matching can use dummy columns for distance computation while quality
tests still run chi² on the original categorical column.

```
ds:  {"city": "Moscow", ...}
        │  DummyEncoder
        ▼
additional_fields:  {"DummyEncoder┴┴┆city┆_Moscow": 1, ...}
                     with AdditionalFeatureRole
```

## File map

| File | Contents |
|---|---|
| `abstract.py` | `Encoder` — the base class. |
| `encoders.py` | `DummyEncoder`. |
| `__init__.py` | Empty. `DummyEncoder` is re-exported from `hypex.transformers`, which is where most code imports it from. |

The actual encoding maths lives in `hypex/extensions/encoders.py`
(`DummyEncoderExtension`).

## Key classes

### `Encoder(Calculator)`

`Encoder(target_roles=FeatureRole(), key="")`

* `search_types` → `[CategoricalTypes]`, so only categorical columns are selected.
* `_get_ids(col_name)` / `_ids_to_names(cols)` — build the storage key for each
  produced column as `NAME_BORDER_SYMBOL + col_name + NAME_BORDER_SYMBOL`
  embedded in the executor id. Reporters and outputs parse the original column
  name back out of that.
* `_set_value` writes all produced columns into `additional_fields` in one call,
  carrying the roles the extension assigned.
* `execute` resolves the target columns; if none are categorical it returns the
  data untouched (no error).
* `_inner_function(data, **kwargs) -> Dataset` — the method a subclass implements.

### `DummyEncoder(Encoder)`

One-hot encoding. Delegates to `DummyEncoderExtension`, which:

* calls `pd.get_dummies(..., drop_first=True)` and casts to `int` — `drop_first`
  avoids the dummy-variable trap for regression-based blocks;
* derives each new column's role from the source column's role via
  `.asadditional(int)`, so a `FeatureRole` categorical becomes
  `AdditionalFeatureRole` dummies;
* returns an empty `Dataset` when there are no target columns.

Used in `PREPROCESSING_DATA` and in the `Matching` pipeline.

## How to work with it

```python
from hypex.transformers import DummyEncoder   # or hypex.encoders.encoders
from hypex.dataset import FeatureRole

Experiment(executors=[
    CategoryAggregator(threshold=15),   # collapse rare levels first
    DummyEncoder(target_roles=FeatureRole()),
    ...
])
```

After it runs, the dummy columns are in `experiment_data.additional_fields`, not
in `ds`.

## How to add an encoder

1. Write the maths as an `Extension` in `hypex/extensions/`, implementing
   `_calc_pandas` and `_calc_spark`. Assign roles to the produced columns with
   `source_role.asadditional(dtype)`.
2. Subclass `Encoder` here and implement `_inner_function` delegating to it:

```python
class TargetEncoder(Encoder):
    @staticmethod
    def _inner_function(data, target_cols=None, **kwargs):
        if not target_cols:
            return Dataset.create_empty()
        return TargetEncoderExtension().calc(data=data, target_cols=target_cols, **kwargs)
```

3. Re-export it from `hypex/transformers/__init__.py` if it belongs in the
   preprocessing chain.

## Gotchas

* **Spark is not implemented** for `DummyEncoderExtension` — only `_calc_pandas`
  exists. A Spark dataset will fail at `Extension.calc`'s backend lookup.
* **Aggregate rare categories first.** Without `CategoryAggregator`, a
  high-cardinality column produces one dummy column per level.
* `drop_first=True` means the number of dummy columns is `n_levels - 1`; the
  omitted level is the reference.
* The extension carries a `TODO` about role types being rewritten — the dummies'
  `data_type` is set to `bool` on the intermediate roles while the stored copy
  uses `int`. Do not rely on that dtype being stable.

## Related modules

`../extensions/README.md` (where the encoding is implemented) ·
`../transformers/README.md` (re-exports `DummyEncoder`; run
`CategoryAggregator` before it) · `../dataset/README.md` (`AdditionalRole`
family, `additional_fields`).

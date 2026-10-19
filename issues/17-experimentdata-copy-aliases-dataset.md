## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`ExperimentData.copy(data=...)` takes the replacement dataset **by reference**, contradicting its
docstring and creating shared mutable state between experiments.

`hypex/dataset/experiment_data.py:73-93`:

```python
def copy(self, data: Dataset | SmallDataset | None = None) -> Self:
    """Create a deep copy of this ExperimentData instance. ..."""
    result = deepcopy(self)
    if data is not None:
        result._data = data          # by reference, no copy
        result._initial_cols = deepcopy(data.columns)
    return result
```

Everything else is deep-copied, but the primary dataset is aliased. Dataset mutation is in-place
throughout this package (`add_column`, `__setitem__`, `update_column_type`, index setters), so two
experiments created via `ed.copy(data=shared_ds)` — or one experiment reusing a caller's dataset —
mutate each other's `ds`, columns, and roles.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import pandas as pd
from hypex.dataset import Dataset, InfoRole, ExperimentData

ds = Dataset(data=pd.DataFrame({"x": [1, 2, 3]}), roles={"x": InfoRole()})
ed = ExperimentData(ds)
new_ds = Dataset(data=pd.DataFrame({"x": [9, 9, 9]}), roles={"x": InfoRole()})

ed2 = ed.copy(data=new_ds)
print(ed2.ds is new_ds)        # True — aliased, despite the "deep copy" docstring

ed2.ds.add_column(data=[1.5, 2.5, 3.5], role={"y": InfoRole()})
print(new_ds.columns)          # ['x', 'y'] — caller's dataset was mutated
```

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

Either `copy(data=...)` deep-copies the replacement dataset (matching the docstring), or the
docstring explicitly documents that ownership of `data` is transferred to the copy and the caller
must not reuse it.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (backend-independent)

### Additional Context

<!-- Add any other context about the problem here. -->

- All current in-library call sites (`analyzers/aa.py:457`, `transformers/abstract.py:23`,
  `transformers/cuped.py:135`, `transformers/shuffle.py:26`, `transformers/filters.py:299`,
  `ml/cupac.py:355`) pass freshly built datasets, so the bug is latent — but it is public API and a
  trap for any user who reuses a dataset across experiments.
- `deepcopy` of `Dataset` is already used elsewhere and works (memo sharing between `DatasetBase`
  and its Lockers is handled), so a full copy is feasible.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

`result._data = deepcopy(data)` (keeping the `_initial_cols` sync), or rename the parameter to
`take_data=` and document the ownership transfer. Add a test asserting
`ed2.ds is not original_ds` after `copy(data=original_ds)`.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

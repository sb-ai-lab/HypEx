## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`ExperimentData._set_additional_fields` (`hypex/dataset/experiment_data.py:259-293`) unpersists the
Spark cache it just created, because `DatasetBase.add_column` mutates the dataset **in place** and
returns `self`:

```python
new_data.persist(storage_level=storage_level, action="none")
if was_persisted:
    self._data.unpersist()      # lines 291-292: new_data IS self._data here
self._data = new_data
```

For the single-column / raw-value branches, `new_data is self._data`, so the second `unpersist()`
removes exactly the cache `persist()` just established and resets `_is_persisted_flag` to `False`.
The persist-new / unpersist-old pattern only makes sense for the `merge` branch, where `new_data`
is a distinct object.

Verified on a persisted Spark dataset:
- `ed.ds.is_persisted` goes `True → False` across `set_value(additional_fields, ...)` — the
  documented caching contract around executor writes is inverted;
- a previously *un*persisted dataset ends up persisted (`True`), because the guard is skipped.

Net effect on Spark: executor results are re-computed on every subsequent access (performance), and
the `is_persisted`/`get_storage_level` state becomes misleading. Results are numerically unaffected.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
# PYSPARK_PYTHON / PYSPARK_DRIVER_PYTHON pointed at the venv interpreter
import pandas as pd, numpy as np
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false").getOrCreate()

from hypex.dataset import Dataset, TargetRole, ExperimentData
from hypex.dataset.roles import AdditionalRole
from hypex.utils import BackendsEnum, ExperimentDataEnum

df = pd.DataFrame({"y": np.arange(10, dtype=float), "idx": np.arange(10)})
ds = Dataset(data=df.set_index("idx"), roles={"y": TargetRole()},
             backend=BackendsEnum.spark, session=spark)
ds.persist(action="none")
ed = ExperimentData(ds)
print(ed.ds.is_persisted)                                   # True
ed._set_additional_fields("x1", [1.0]*10, AdditionalRole())
print("new backend flag:", ed.ds._backend_data._is_persisted_flag)   # False (expected True)
```

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

After `set_value` on a previously persisted dataset, the (new) `ds` remains persisted at the same
storage level; on an unpersisted dataset it remains unpersisted.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (Spark backend only)

### Additional Context

<!-- Add any other context about the problem here. -->

- The bug only manifests for the in-place `add_column` branches; the multi-column `merge` branch
  behaves as intended.
- Related asymmetry in the same method: when `was_persisted` is False, `new_data.persist(...)` is
  still called unconditionally (line 290), so unpersisted datasets silently gain a cache.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Branch on identity:

```python
if new_data is self._data:
    # in-place mutation: cache already persisted, nothing to swap
    if not was_persisted:
        new_data.persist(storage_level=storage_level, action="none")
else:
    new_data.persist(storage_level=storage_level, action="none")
    if was_persisted:
        self._data.unpersist()
self._data = new_data
```

Add a Spark test asserting `is_persisted` is preserved across `set_value`.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`Dataset.loc[single_label]` crashes on the Spark backend with
`AttributeError: 'Series' object has no attribute 'columns'`.

`SparkNavigation.loc` (`hypex/dataset/backends/spark_backend.py:1206-1217`) returns the raw
`pyspark.pandas` object unwrapped — for a scalar label that is a `ps.Series`. The `Locker` then
unconditionally accesses `.columns` (`hypex/dataset/abstract.py:55-60`):

```python
def __getitem__(self, item):
    t_data = self.backend.loc(item)
    return self.call_class(
        data=t_data,
        roles={k: v for k, v in self.roles.items() if k in t_data.columns},  # <- crash
    )
```

The pandas backend coerces the result to a frame before returning
(`pandas_backend.py:1081-1093`), so only Spark fails.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import pandas as pd
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()

from hypex.dataset import Dataset, InfoRole
from hypex.utils import BackendsEnum

pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
roles = {"x": InfoRole(), "y": InfoRole()}

spark_ds = Dataset(data=pdf, roles=roles, backend=BackendsEnum.spark, session=spark)
pandas_ds = Dataset(data=pdf, roles=roles)

pandas_ds.loc[0]   # returns a Dataset
spark_ds.loc[0]    # AttributeError: 'Series' object has no attribute 'columns'
```

(The index must be preserved — build the Spark dataset via `ps.from_pandas` or an index column — so
this is independent of the known `createDataFrame` index-loss bug.)

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

Scalar `loc` returns a one-row `Dataset` on both backends, or raises the same well-defined error on
both backends. (Ideally the row label should also be preserved as on pandas; currently even the
pandas scalar result is a frame whose column is named by the row label, which is its own oddity.)

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (Spark backend, local session)

### Additional Context

<!-- Add any other context about the problem here. -->

- Not currently exercised inside the library (`hypex/ui/matching.py:295` passes a list), but it is
  public API and a hard crash on one backend only.
- The list form `loc[[...]]` works on both backends (its ordering divergence is already xfail'd).

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Mirror the pandas backend: in `SparkNavigation.loc`, coerce a `ps.Series` result to a one-row /
one-column `ps.DataFrame` before returning (or wrap it at the `Locker` level for both backends), and
add a parity test for scalar `loc`.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

On the Spark backend, `mean()`, `sum()`, `min()` and `max()` **silently drop non-numeric columns**,
while pandas raises. The Spark `agg` filters columns through `get_numeric_columns()`
(`hypex/dataset/backends/spark_backend.py:1482-1489`), so a mixed-type frame returns numbers computed
only over the numeric subset; pandas raises
`TypeError: Could not convert string 'ab' to numeric`.

This is the same family as the xfail'd `test_count_ignores_nan_on_spark`
("Spark count() includes NaN and drops non-numeric columns"), but that xfail pins only `count` —
`mean`/`sum`/`min`/`max` share the silent-drop behavior and are not covered.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import pandas as pd
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()

from hypex.dataset import Dataset, InfoRole
from hypex.utils import BackendsEnum

pdf = pd.DataFrame({"num": [1.0, 2.0], "txt": ["ab", "cd"]})
roles = {"num": InfoRole(), "txt": InfoRole()}

pandas_ds = Dataset(data=pdf, roles=roles)
spark_ds = Dataset(data=pdf, roles=roles, backend=BackendsEnum.spark, session=spark)

pandas_ds.mean()   # TypeError: Could not convert string 'ab' to numeric
spark_ds.mean()    # returns {'num': 1.5} — 'txt' silently ignored
spark_ds.sum()     # same silent drop
```

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

Both backends agree: either both raise on non-numeric columns, or both exclude them with an
explicit warning. Silent divergence means a pipeline that works on pandas can produce
incomplete-but-plausible numbers on Spark.

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

- Statistical executors select their target columns by role and `search_types=[int, float]`, so
  library pipelines are mostly shielded; the divergence hits direct `Dataset` users.
- Fixing this together with the `count` xfail (same root cause in the agg numeric filter) would let
  several `xfail(strict=True)` markers be removed at once.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Align on one contract: raise a descriptive error when an aggregation receives non-numeric columns
(matching pandas), or warn and exclude consistently on both backends — then update the agg xfails
and `tests/KNOWN_LIBRARY_BUGS.md`.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

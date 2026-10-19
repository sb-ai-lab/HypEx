## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`SparkDataset.value_counts` (`hypex/dataset/backends/spark_backend.py:1725-1755`) diverges from the
pandas reference implementation in two ways:

1. **Multi-column input silently counts only the first column.** The Spark implementation does
   `col = next(iter(self.data.columns))` and counts that single column, while pandas aggregates row
   combinations over **all** columns. The result silently loses data — no error, no warning.
2. **`normalize=True` produces a misnamed column and then crashes.** The Spark path hardcodes
   `result.to_frame(name="count")`, but the abstract layer
   (`hypex/dataset/abstract.py:1141-1145`) then looks for a column named `"proportion"` when
   `normalize=True` (which is what pandas names it). The rename at line 1143 is a silent no-op, the
   role dict gets a key no column has, and `Dataset.__init__` role validation fails.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import pandas as pd
from hypex.dataset import Dataset, InfoRole
from hypex.utils import BackendsEnum

pdf = pd.DataFrame({"x": ["a", "a", "b"], "y": [1, 1, 2]})
roles = {"x": InfoRole(), "y": InfoRole()}

# 1. multi-column counts
spark_ds = Dataset(data=pdf, roles=roles, backend=BackendsEnum.spark, session=spark)
pandas_ds = Dataset(data=pdf, roles=roles)
print(pandas_ds.value_counts().columns)   # ['x', 'y', 'count']  (combination counts)
print(spark_ds.value_counts().columns)    # ['x', 'count']      — column 'y' silently dropped

# 2. normalize=True on Spark
spark_ds.value_counts(normalize=True)
# RoleColumnError: Check your roles. All of them must be names of data columns.
# Now roles have ['proportion'] values and columns have ['x', 'count'].
```

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

`value_counts` must behave identically on both backends: multi-column input counts row combinations
over all columns, and `normalize=True` returns a column named `"proportion"` with role wiring intact.

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

- `value_counts` is used by the chi2 machinery, so the single-column limitation currently constrains
  what the Spark path can support for multi-column categorical comparisons.
- The `normalize` column-name contract ("proportion" vs "count") is set by the pandas backend and
  the abstract layer; fixing the Spark side to emit `proportion` for `normalize=True` restores parity
  without touching the contract.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Implement multi-column counting in the Spark backend (group by all columns and aggregate counts,
matching pandas semantics), and name the normalized column `proportion` when `normalize=True`.
Add parity tests to `tests/unit/dataset/test_backend_parity.py` for both facets.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

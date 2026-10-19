## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`ABAnalyzer._build_row_index` (`hypex/analyzers/ab.py:239-247`, iterative mode) zips
`group_labels` taken from `data.groups` (natural `groupby` order, built at lines 337-338) onto
result rows whose order comes from `GroupsComparator._grouping_data_split`
(`hypex/comparators/abstract.py:218`), which sorts group keys **as strings** (`sorted(..., key=str)`).
When the two orders differ (numeric keys with different digit counts), each result row is assigned
the wrong group label — group statistics and p-values are published for the wrong groups end-to-end.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole
from hypex.ab import ABTest

rng = np.random.default_rng(0)
n = 1200
df = pd.DataFrame({"id": np.arange(n), "treat": rng.choice([1, 2, 10], n), "y": 0.0})
df["y"] = np.select([df["treat"] == 1, df["treat"] == 2, df["treat"] == 10],
                    [0.03, 8.0, -0.01]) + rng.normal(0, 0.1, n)
ds = Dataset(data=df, roles={"id": InfoRole(), "treat": TreatmentRole(), "y": TargetRole()})
print(df.groupby("treat")["y"].mean())   # true means: 1 -> 0.024, 2 -> 8.00, 10 -> -0.014
out = ABTest().execute(ds)
print(out.summary.raw_data[["group", "control mean", "test mean", "TTest p-value"]])
```

Observed output:

```
  group  control mean  test mean  TTest p-value
0    10      0.024336   8.001931   3.536246e-08
1     2      0.024336  -0.014353   0.000000e+00
```

The row labeled group **10** carries group **2**'s test mean (8.00), and the row labeled group **2**
carries group **10**'s mean (−0.01): means and p-values are published under swapped group labels.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

`group 10` row → test mean ≈ −0.01 (group 10's mean) with group 10's p-value; `group 2` row →
test mean ≈ 8.00 with group 2's p-value.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (pandas backend; the analyzer is shared with the Spark path)

### Additional Context

<!-- Add any other context about the problem here. -->

- The analyzer also mixes two orderings internally: `_split_for_groups_mode` (line 279-282) sorts by
  raw key while `_grouping_data_split` (line 218) sorts by `str`, so the two code paths can even pick
  different baseline groups for the same data.
- The existing guard tests (`tests/dataset_spark_backend/test_group_order.py`) pin the aggregation
  key order only, not the analyzer's label assignment.
- This is one of a family of string-vs-natural sort divergences; see the related issues for
  `GroupsComparator._grouping_data_split` and `StatsComparator.execute`.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Use one canonical ordering (ascending by value, nulls last) everywhere: in
`_grouping_data_split`/`_split_for_groups_mode` (so rows come out in a known order) and when building
`group_labels` in `ABAnalyzer.execute`/`_build_row_index`. Ideally derive labels from the parsed row
index instead of assuming positional correspondence.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

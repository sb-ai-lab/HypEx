## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`StatsComparator` produces the comparison **values** and the result **index labels** using two
different orderings of the group keys:

- `_execute_inner_function` (`hypex/comparators/abstract.py:921`) iterates `group_col_stats` in dict
  insertion order (pandas `groupby` natural sort; Spark `_sort_by_group_key` value sort) —
  baseline = first dict key;
- `execute` (`hypex/comparators/abstract.py:1096`) rebuilds the labels as
  `group_names = sorted(group_col_stats.keys(), key=str)`.

Whenever the two orders differ (numeric keys with different digit counts, e.g. `2` vs `10`), row *i*
of the result is labeled with the wrong compared group. The same `key=str` re-sort appears in
`StatsKSTest._execute_histograms` (line 631) and `StatsUTest._execute_histograms` (line 916),
re-sorting the already value-sorted Spark output lexicographically.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole, ExperimentData
from hypex.comparators.stats_hypothesis_testing import StatsTTest

rng = np.random.default_rng(0)
df = pd.DataFrame({"g": [2]*50 + [10]*50,
                   "y": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])})
ds = Dataset(data=df, roles={"g": TreatmentRole(int), "y": TargetRole()}, default_role=InfoRole())
StatsTTest(grouping_role=TreatmentRole()).execute(ExperimentData(ds))
print(ds.analysis_tables["StatsTTest┴┴y"].raw_data)
# index: ['2┆y']  statistic: -25.26
```

The row is labeled "compared group = 2", but the values were computed with group 2 as the
**baseline** and group 10 as the compared group (baseline = first dict key = 2). The label and the
control/test roles disagree; consumers that read the label to decide direction get the wrong answer.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

Row *i* of the result must correspond to entry *i* of the index: one canonical (value-ascending)
ordering for both the values and the labels, and the baseline must be that ordering's first key.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (pandas backend; the KS/U histogram paths affect Spark)

### Additional Context

<!-- Add any other context about the problem here. -->

- The `┆stats` tables are also built from `group_names` (str-sorted) while values are looked up
  **by key** (`for g in group_names`), so the stats table happens to be internally consistent — only
  the pairwise result index is misaligned relative to the value iteration order.
- On Spark, `StatsChi2TestExtension` additionally iterates per-column dicts in collect arrival order
  (see the dedicated issue) — this issue is the group-order facet only.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Use one ordering for both values and labels: sort the keys by value (nulls last) **before** calling
`_execute_inner_function`, and reuse that same list when building the index. Remove the `key=str`
re-sorts in `execute`, `StatsKSTest._execute_histograms` and `StatsUTest._execute_histograms`.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

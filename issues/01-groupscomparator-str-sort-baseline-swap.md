## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`GroupsComparator._grouping_data_split` (`hypex/comparators/abstract.py:218`) chooses the baseline
(control) group by sorting group keys **as strings**:

```python
compared_data = sorted(grouping_data.items(), key=lambda x: str(x[0]))
baseline_data = [compared_data.pop(0)]
```

For numeric group keys with different digit counts, `"10" < "2"` lexicographically, so the
**numerically largest** key becomes the control group. Control and test are swapped: the t/U
statistic sign is inverted, and the reported "compared group" label does not match the group whose
statistics were actually computed. Every other ordering in the codebase (pandas `groupby`,
`_split_for_groups_mode` at line 279, `_sort_by_group_key` on Spark) is a natural/value sort.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole, ExperimentData
from hypex.comparators import GroupTTest

rng = np.random.default_rng(0)
df = pd.DataFrame({"g": [2]*50 + [10]*50,
                   "y": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])})
ds = Dataset(data=df, roles={"g": TreatmentRole(int), "y": TargetRole()}, default_role=InfoRole())
GroupTTest(compare_by="groups", grouping_role=TreatmentRole()).execute(ExperimentData(ds))
table = list(ds.analysis_tables.values())[0].raw_data
print(table.index.tolist(), table["statistic"].iloc[0])
# -> ['2'], statistic = +25.26
```

`scipy.stats.ttest_ind(group_2, group_10)` (control = group 2, the smaller key) gives statistic
**−25.26**. The string sort picked group **10** as baseline.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

The baseline (control) group must be the smallest key by **value** (group 2), consistently with
pandas `groupby(sort=True)`, `_split_for_groups_mode`, and the Spark `_sort_by_group_key`. The
statistic should be −25.26 with the compared-group label `10`.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (pandas backend; affects any backend using `GroupsComparator`)

### Additional Context

<!-- Add any other context about the problem here. -->

- With string group keys (`"a"` vs `"b"`) the behavior is correct, which is why existing tests do not
  catch it: two-sided p-values are symmetric under the swap, only the statistic sign and the
  "which group is compared" label are wrong.
- End-to-end, this corrupts `ABTest` summaries when the treatment column has multi-digit numeric
  values. See the sibling issues about `StatsComparator` (`key=str` at `abstract.py:1096`) and
  `ABAnalyzer._build_row_index` for the same ordering family.
- The merge commit of the Spark backend (`7cdae829`) itself notes a "pre-existing control/test swap"
  that was never fixed.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Sort by value with nulls last (mirroring `_sort_by_group_key`), e.g. a shared helper used by
`_grouping_data_split`, `_split_for_groups_mode`, `StatsComparator.execute` and `ABAnalyzer`, so all
orderings agree. Do not re-sort with `key=str` after the values were produced.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

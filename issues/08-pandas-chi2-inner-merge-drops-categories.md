## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`PandasChi2TestExtension.matrix_preparation`
(`hypex/extensions/scipy_stats.py:271`) builds the contingency table with an **inner** merge:

```python
return data_vc.merge(other_vc, on=col_name)[["count_x", "count_y"]].fillna(0)
```

`Dataset.merge` defaults to `how="inner"`, so any category that occurs in only one of the two groups
is silently removed from the table before `chi2_contingency` runs. The trailing `.fillna(0)` is dead
code under an inner join — clear evidence an outer join was intended. A category present in only one
group is the strongest possible evidence against homogeneity, and it is exactly what gets discarded.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, InfoRole, TargetRole
from hypex.extensions.scipy_stats import PandasChi2TestExtension
from scipy.stats import chi2_contingency

ext = PandasChi2TestExtension()
d1 = Dataset(data=pd.DataFrame({"cat": ["u"]*30 + ["v"]*10}),
             roles={"cat": TargetRole(str)}, default_role=InfoRole())
d2 = Dataset(data=pd.DataFrame({"cat": ["u"]*20 + ["v"]*20 + ["rare"]*10}),
             roles={"cat": TargetRole(str)}, default_role=InfoRole())

m = ext.matrix_preparation(d1, d2)
print(m.raw_data.shape)                       # (2, 2) — 'rare' vanished
print(chi2_contingency([[30, 10, 0], [20, 20, 10]]))   # correct 2x3: p = 0.00075
print(ext.calc(d1, d2))                       # HypEx: p = 0.192
```

Degenerate case: control = 100×"x"+100×"y", test = 100×"y"+100×"z" → the inner merge leaves a
degenerate 2×1 table and HypEx returns **p = 1.0** where scipy gives ≈ 3.7e-44.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

The contingency table must span the **union** of categories (missing categories get zero counts),
so p-values match `chi2_contingency` on the full table (0.00075 in the example above).

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (pandas backend)

### Additional Context

<!-- Add any other context about the problem here. -->

- The proportion-scaling defect of this same method is already tracked by the strict xfail
  `test_chi2_matches_scipy_contingency` (`tests/unit/comparators/test_hypothesis_testing.py:131`);
  the inner-merge facet is **not** covered by that xfail and would survive a scaling-only fix.
- Affects `ABTest(additional_tests=["chi2-test"])`, homogeneity checks and Matching quality tests on
  the pandas backend.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Use `merge(other_vc, on=col_name, how="outer")` (making the existing `.fillna(0)` meaningful), or
build the table directly over the union of categories as the Spark implementation does. Fix together
with the scaling issue and update the xfail.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

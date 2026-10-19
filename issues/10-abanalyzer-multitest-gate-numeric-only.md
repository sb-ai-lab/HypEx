## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`ABAnalyzer.execute_multitest` (`hypex/analyzers/ab.py:125-131`) computes the multiple-testing family
size from **numeric** targets only:

```python
target_fields = data.ds.search_columns(TargetRole(), search_types=[int, float])
num_comparisons = (num_groups - 1) * len(target_fields)
if self.multitest_method and num_comparisons > 1:
```

but the p-value collection (`_add_pvalues`) includes categorical-target p-values as well (chi2 on
Spark is in the id list and its targets are `str`). With categorical targets the gate undercounts or
zeroes out, so collected chi2 p-values are silently left uncorrected; with mixed numeric +
categorical targets the correction call itself crashes.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

1. Categorical-only, gate undercounts (3 groups, 2 categorical targets → 4 comparisons collected,
   `num_comparisons = 2 × 0 = 0` → no correction; verified `out.multitest` reports
   "Multiple testing correction was not applied").
2. Mixed targets, crash (pandas, 3 groups, 1 numeric + 1 categorical target,
   `ABTest(additional_tests=["chi2-test"], multitest_method="holm")`):

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole, ExperimentData
from hypex.ab import ABTest

rng = np.random.default_rng(0)
n = 600
df = pd.DataFrame({"id": np.arange(n), "treat": rng.integers(0, 3, n),
                   "num_y": rng.normal(size=n), "purchased": rng.choice(["a", "b"], n)})
ds = Dataset(data=df, roles={"id": InfoRole(), "treat": TreatmentRole(),
                             "num_y": TargetRole(), "purchased": TargetRole(str)})
ABTest(additional_tests=["chi2-test"], multitest_method="holm").execute(ExperimentData(ds))
```

raises:

```
numpy.core._exceptions._UFuncNoLoopError: ufunc 'greater' did not contain a loop with signature
matching types (<class 'numpy.dtypes.StrDType'>, <class 'numpy.dtypes.Float64DType'>) -> None
```

in `statsmodels.stats.multitest.multipletests`.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

The correction family must include every collected p-value (numeric and categorical), and the gate
must count the actual number of collected comparisons. Mixed test families are already handled by
`MultiTest`'s per-test-family grouping, so no crash should occur.

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

- Related: `ABAnalyzer` currently does not even collect `GroupChi2Test` results on pandas (separate
  issue); this bug is masked until that one is fixed on pandas, but is live on Spark
  (`StatsChi2Test` is collected there).
- The `dtype` crash comes from string columns entering `multitest_pvalues` via the collected
  p-value table; the family-size gate then passes them into `multipletests` unfiltered.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Derive `num_comparisons` from the collected p-value table (its row count) instead of counting
numeric targets; ensure only float p-value cells are appended to `multitest_pvalues` (or coerce
dtype before `multipletests`).

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

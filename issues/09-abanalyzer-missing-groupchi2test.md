## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`ABAnalyzer.execute` (`hypex/analyzers/ab.py:300-310`) collects test results by executor class:

```python
executor_ids = data.get_ids([
    GroupTTest, GroupUTest, GroupKSTest,
    StatsTTest, StatsChi2Test, StatsKSTest, StatsUTest,
])
```

On the **pandas** backend, `ABTest(additional_tests=["chi2-test"])` dispatches to `GroupChi2Test`
(registered in `backend_factory` for `Chi2Test` + `PandasDataset`), and its results are stored under
an id prefixed `GroupChi2Test`. The class list only includes the Spark-side name `StatsChi2Test`, so
chi2 results are never merged into the analyzer output and never reach the multiple-testing
correction. The sibling `OneAAStatAnalyzer` (`hypex/analyzers/aa.py:96-108`) lists **both**
`GroupChi2Test` and `StatsChi2Test`, confirming the omission is unintentional.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole, ExperimentData
from hypex.ab import ABTest

rng = np.random.default_rng(0)
n = 600
df = pd.DataFrame({"id": np.arange(n), "treat": rng.integers(0, 3, n),
                   "purchased": rng.choice(["a", "b", "c"], n)})
ed = ExperimentData(Dataset(data=df, roles={"id": InfoRole(), "treat": TreatmentRole(),
                                            "purchased": TargetRole(str)}))
out = ABTest(additional_tests=["chi2-test"], multitest_method="holm").execute(ed)

print(list(out._experiment_data.analysis_tables.keys()))
# ['GroupSizes┴┴treat┆stats', 'GroupSizes┴┴treat', 'GroupChi2Test┴┴purchased', 'ABAnalyzer┴┴']
print(out._experiment_data.analysis_tables["ABAnalyzer┴┴"].raw_data)   # empty
print(out.multitest)
# "Multiple testing correction was not applied: total comparisons ((groups-1) × targets) ≤ 1 ..."
```

`GroupChi2Test┴┴purchased` exists with valid p-values, but the AB summary is empty and chi2
p-values receive no Holm/Bonferroni/FDR correction.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

chi2 results appear in the AB summary, and their p-values participate in the multiple-testing
correction, exactly as the other tests do (and as they do on the Spark backend via `StatsChi2Test`).

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

- The reporter (`hypex/reporters/ab.py:37`) is able to read chi2 results raw, so only the analyzer
  collection is missing.
- Related but separate defect: the multitest family gate in `execute_multitest` counts only numeric
  targets (see the dedicated issue), so even after collection, categorical-target p-values would
  remain uncorrected without a second fix.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Add `GroupChi2Test` to the `get_ids` class list in `ABAnalyzer.execute`, and add a scenario test
that runs `ABTest(additional_tests=["chi2-test"], multitest_method="holm")` end-to-end and asserts
non-empty analyzer output plus corrected p-values.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

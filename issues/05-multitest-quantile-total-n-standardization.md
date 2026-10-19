## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`MultitestQuantile._calc_pandas` (`hypex/extensions/statsmodels.py:178, 196-202`) standardizes the
observed difference of means by the **total** sample size:

```python
sample_size = len(data)                      # rows across ALL groups
...
t_value = np.sqrt(sample_size) * (means[j] - means[i]) / np.sqrt(variances[j] + variances[i])
```

but the simulated null quantiles produced by `quantile_of_marginal_distribution` correspond to a
statistic standardized by the **per-group** size (`sqrt(n_group)·Δ/√(v_j+v_i) ~ N(0,1)`). Using the
total N inflates the observed statistic by `sqrt(n_groups)`, so the method accepts a "winning" group
far more often than the nominal rate.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, InfoRole, TargetRole
from hypex.extensions.statsmodels import MultitestQuantile

mq = MultitestQuantile(0.05, 2000, equal_variance=True, random_state=0)
wins = 0; trials = 300; n = 100
np.random.seed(3)
for _ in range(trials):
    ys = np.random.normal(size=(2, n))                       # H0: identical distributions
    df = pd.DataFrame({"g": np.repeat([0, 1], n), "y": ys.flatten()})
    ds = Dataset(data=df, roles={"g": InfoRole(), "y": TargetRole()})
    res = mq.calc(ds, group_field="g", target_field="y", quantiles=None)
    wins += res.raw_data["accepted hypothesis"].iloc[0] != 0
print(wins / trials)
```

Observed false-"winner" rates under H0 (nominal α = 0.05):

- k=2 groups, n=100/group → **0.220**
- k=4 groups, n=80/group → **0.273**
- emulating the fix (`sample_size = len(data) // num_samples`) → **0.063**

End-to-end: `ABTest(multitest_method="quantile")` on pure-H0 data (3 groups × 2 targets, 20 trials)
declared a winner in **10 of 20** runs.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

Under H0 the probability of accepting any hypothesis must be ≈ α (5%). The standardizer must match
the simulated null: per-group size of the compared pair (or per-group sizes n_j, n_i with the exact
SE `sqrt(v_j/n_j + v_i/n_i)` plus a corresponding change to the simulated statistic).

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

- Distortion grows as `sqrt(k)` with the number of variants, so A/B/n tests are affected most.
- Reachable via `ABTest(multitest_method="quantile")` when `(num_groups - 1) × targets > 1`.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Set `sample_size` to the per-group size of the pair being compared (or use per-group sizes with the
exact SE formula and adjust `quantile_of_marginal_distribution` accordingly). Adding a statistical
calibration test (H0 false-positive rate ≈ α over a Monte-Carlo loop) would prevent regressions.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

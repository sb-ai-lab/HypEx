## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`UniteCovExtension.calc` (`hypex/extensions/scipy_linalg.py:15-27`), used by
`MahalanobisDistance._inner_function` (`hypex/comparators/distances.py:180`), combines the two
groups' covariances as a **plain average**:

```python
cov_data = data.raw_data.cov().to_numpy()
cov_test = test_data.raw_data.cov().to_numpy()
result = (cov_data + cov_test) / 2
```

The docstring in `MahalanobisDistance` says "pooled covariance matrix", but the standard pooled
within-group covariance (Rubin; used for Mahalanobis matching) is the **n-weighted**
`((n₁−1)·S₁ + (n₂−1)·S₂) / (n₁+n₂−2)`. The plain average gives each group equal weight regardless of
size, so with unbalanced groups (e.g. 10:1 control:treated) the distance metric is dominated by the
minority group's covariance structure, changing which matches are selected. The two formulas coincide
only for equal group sizes.

This may be a deliberate design choice (it is a valid positive-semidefinite metric either way), but
it deviates from the textbook definition the docstring claims and is undocumented.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, InfoRole, FeatureRole, TargetRole
from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import ExperimentData

rng = np.random.default_rng(0)
n_c, n_t = 1000, 10          # heavily unbalanced
cov_major = np.array([[1.0, 0.0], [0.0, 10.0]])   # majority group's covariance
control = rng.multivariate_normal([0, 0], cov_major, size=n_c)
treated = rng.multivariate_normal([0, 0], np.eye(2) * 50, size=n_t)  # very different, tiny

df = pd.DataFrame({"x1": np.r_[control[:, 0], treated[:, 0]],
                   "x2": np.r_[control[:, 1], treated[:, 1]],
                   "g": [0]*n_c + [1]*n_t, "y": rng.normal(size=n_c + n_t)})
ds = Dataset(data=df, roles={"x1": FeatureRole(), "x2": FeatureRole(),
                             "g": TargetRole(), "y": TargetRole()}, default_role=InfoRole())
MahalanobisDistance(grouping_role=TargetRole()).execute(ExperimentData(ds))
# compare the internal metric against:
S_pooled = ((n_c - 1) * np.cov(control.T) + (n_t - 1) * np.cov(treated.T)) / (n_c + n_t - 2)
S_plain  = (np.cov(control.T) + np.cov(treated.T)) / 2      # what UniteCovExtension computes
```

The two matrices differ substantially; distances and hence matches differ with them.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

Either the n-weighted pooled covariance is used (matching the docstring and the standard
Mahalanobis-matching definition), or the equal-weight averaging is documented as intentional.

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

- Related regularization concern in the same file: the Cholesky guard adds an **absolute**
  `epsilon = 1e-3` identity term unconditionally (`scipy_linalg.py:30-47`). Mahalanobis distance is
  supposed to be invariant to linear rescaling of features, but `1e-3` is negligible for
  large-variance features and dominates features with variance ≲ 1e-3 (effectively reducing them to
  plain standardized L2). A scale-relative epsilon (e.g. `ε·trace(cov)/d` or eigenvalue clipping)
  would preserve invariance. Same disposition: fix or document.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Implement the n-weighted pooled covariance (sizes are available from the input datasets) and a
scale-relative regularization; if the current behavior is kept intentionally, update the
`MahalanobisDistance` docstring and note the deviation in the module docs.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

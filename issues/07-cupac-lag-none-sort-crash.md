## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`CUPACExecutor._prepare_data` (`hypex/ml/cupac.py:143`) sorts (field, lag) pairs with a key that
crashes when any field has `lag=None`:

```python
sorted_fields_by_lag = sorted(searched_lags, key=lambda x: x[1])
# TypeError: '<' not supported between instances of 'int' and 'NoneType'
```

A current-period `FeatureRole()` column has `lag=None`, so mixing it with a lagged
`FeatureRole(parent=..., lag=1)` column raises `TypeError` — even though the very next block
(lines 144-150) explicitly handles `lag in [None, 0]` ("Empty dict means lag=0 or None
(current period)"). The sort is purely cosmetic here; the crash blocks a supported configuration.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, InfoRole, FeatureRole, TargetRole, ExperimentData
from hypex.ml.cupac import CUPACExecutor

np.random.seed(0)
n = 200
df = pd.DataFrame({
    "x1": np.random.normal(size=n),        # current-period feature  -> lag=None
    "x1_lag1": np.random.normal(size=n),   # lagged feature          -> lag=1
    "y": np.random.normal(size=n),
})
roles = {"x1": FeatureRole(), "x1_lag1": FeatureRole(parent="x1", lag=1), "y": TargetRole()}
ds = Dataset(data=df, roles=roles, default_role=InfoRole())
CUPACExecutor(cupac_models=["linear"]).execute(ExperimentData(ds))
# TypeError: '<' not supported between instances of 'int' and 'NoneType'
```

Works only if *all* features are lagged or *all* are lag-less.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

Current-period features (lag `None` or 0) coexist with lagged features; the fields are grouped into
`fields[field] = {}` for current-period and `fields[parent][lag] = field` for lagged ones, as the
function's own documentation and the following code block intend.

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

- The README/CUPED-CUPAC tutorial workflow (features at every lag) avoids the crash only because all
  features are lagged there.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Use a None-aware sort key, e.g. `sorted(searched_lags, key=lambda x: (x[1] is None, x[1] or 0))`.
A unit test with mixed lag/no-lag `FeatureRole` columns would pin the fix.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

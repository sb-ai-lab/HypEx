## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`PandasFaissExtension.calc(mode="predict")` (`hypex/extensions/faiss.py:361-377`) builds the query
matrix from the **wrong** argument:

```python
X = (
    self._mahalanobis_transform(test_data, self.mahalanobis).raw_data.values
    if mode == "auto"
    else self._mahalanobis_transform(data, self.mahalanobis).raw_data.values  # <- "predict" uses data
)
return self._predict(data, test_data, X)
```

The method's own docstring says `test_data` is "The query (treatment) dataset", and `_predict`
labels the results with `test_data.index`. In `mode="predict"` the neighbor search is executed with
vectors from `data` (the indexed baseline set) instead of `test_data`:

- if `data` and `test_data` have equal length, the neighbor columns are silently attached to the
  wrong rows;
- if lengths differ, `set_index(test_data.index)` raises.

Already tracked by the strict xfail `test_fit_then_predict_equals_auto`
(`tests/unit/test_ml_faiss.py:158`) for the shape symptom; this issue documents the root cause and
the silent wrong-row variant.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, FeatureRole
from hypex.extensions.faiss import PandasFaissExtension

Xa = Dataset(roles={"x": FeatureRole(), "y": FeatureRole()},
             data=pd.DataFrame({"x": [0., 1., 2., 3., 4.], "y": [0., 2., 4., 6., 8.]}))
Xb = Dataset(roles={"x": FeatureRole(), "y": FeatureRole()},
             data=pd.DataFrame({"x": [40., 41., 42., 43., 44.], "y": [80., 82., 84., 86., 88.]}))

ext = PandasFaissExtension(faiss_mode="fast")
ext.calc(data=Xa, test_data=Xb, mode="fit")            # index built on Xa
r = ext.calc(data=Xa, test_data=Xb, mode="predict")
print(r.raw_data.values.flatten().tolist())            # [0, 1, 2, 3, 4]
```

The query points are `Xb` rows (scale 10, far from everything in `Xa`), yet every row is matched to
its own positional twin from `Xa` — proof the search used `data` (Xa) as the query matrix while
labeling results with `test_data`'s index. If `Xa`/`Xb` lengths differed, the call would raise
instead.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

In `mode="predict"` the query vectors must come from `test_data`, matching the docstring and the
index labeling, so `fit` + `predict` produces the same result as `mode="auto"`.

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

- Not reachable through the current in-repo pipeline (`predict()` calls
  `calc(data=X, test_data=X, mode="predict")` with identical objects), but it is a live trap for
  direct users of `PandasFaissExtension` per its documented API.
- Related known issue: the executor-level `fit().predict()` path constructs a fresh unfitted
  extension (`FaissNearestNeighbors.predict` → "index is not created yet"), which is why this code
  path is only reachable via the extension directly today.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

In `calc`, build the query matrix from `test_data` in both `auto` and `predict` modes (only the
Mahalanobis transform anchor differs, if at all), then remove the xfail from
`test_fit_then_predict_equals_auto` and add an equal-length regression test asserting the
wrong-row variant is fixed.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

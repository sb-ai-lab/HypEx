## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`DatasetBase.merge` (`hypex/dataset/abstract.py:1242-1267`) declares the keyword arguments as
`left_index: bool = False, right_index: bool = False`, but both backends enable the automatic
index-join only when **both are `None`**:

- `pandas_backend.py:1811-1819` — `all([left_index is None, right_index is None])`
- `spark_backend.py:2146-2154` — same condition

Since the default is `False` (not `None`), the auto-index-join branch is **dead code**: a keyless
`d1.merge(d2)` raises on both backends — with *different* exception types (pandas `MergeError`,
Spark `ValueError`). The same dead-check pattern exists in `_set_additional_fields`'s merge call
site and anywhere defaults flow through.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import pandas as pd
from hypex.dataset import Dataset, InfoRole

d1 = Dataset(data=pd.DataFrame({"v": [1, 2]}, index=[10, 11]),
             roles={"v": InfoRole()}, default_role=InfoRole())
d2 = Dataset(data=pd.DataFrame({"w": [3, 4]}, index=[10, 12]),
             roles={"w": InfoRole()}, default_role=InfoRole())

d1.merge(d2)   # pandas: MergeError / Spark: ValueError — no keys, no index flags given
```

Only an explicit `left_index=True` (which in-library callers always pass) works.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

A keyless merge joins on the index on both backends (as the backend code clearly intends), or the
API raises the same, documented error on both backends. The `False` defaults must not disable the
backend-level `None` detection.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (both backends)

### Additional Context

<!-- Add any other context about the problem here. -->

- This is an API-contract bug rather than a silent-wrong-results bug: no in-library caller relies on
  keyless merges today.
- The pattern (`default False` vs backend check `is None`) is easy to reintroduce elsewhere; worth a
  convention note in `hypex/dataset/README.md`.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Change the abstract signature to `left_index: bool | None = None, right_index: bool | None = None`
so the backends' `is None` checks are reachable, or drop the backend auto-detection and require
explicit keys/flags with a uniform error. Add parity tests for keyless merge behavior.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`ExperimentData.check_hash(..., ExperimentDataEnum.groups)` causes an **infinite recursion**
(`RecursionError`).

`hypex/dataset/experiment_data.py:193-204` handles `additional_fields`, `variables` and
`analysis_tables` explicitly, but has no branch for `groups` — even though `groups` **is** a member
of `_SUPPORTED_SPACES` (lines 30-37). The "safe fallback" then re-enters `check_hash` with
`s == ExperimentDataEnum.groups`, which hits the same fallback with the same set:

```python
if space not in _SUPPORTED_SPACES:
    return False
return any(self.check_hash(executor_id, s) for s in _SUPPORTED_SPACES)  # groups -> groups -> ...
```

The comment "prevents recursion on unknown enum members" is wrong for the known-but-unhandled member.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import pandas as pd
from hypex.dataset import Dataset, InfoRole, ExperimentData
from hypex.utils import ExperimentDataEnum

ds = Dataset(data=pd.DataFrame({"x": [1, 2, 3]}), roles={"x": InfoRole()})
ed = ExperimentData(ds)
ed.check_hash("some_id", ExperimentDataEnum.groups)
# RecursionError: maximum recursion depth exceeded
```

(`ed.set_value(space=ExperimentDataEnum.groups, ...)` works fine, so a caller that stores into
`groups` and then checks with `check_hash` crashes.)

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

`check_hash("some_id", ExperimentDataEnum.groups)` returns whether the id is a key of
`self.groups` (True/False), like the other three spaces.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (backend-independent)

### Additional Context

<!-- Add any other context about the problem here. -->

- Note that `ExperimentDataEnum.ml` also exists and is not in `_SUPPORTED_SPACES`; after fixing
  `groups`, the fallback for genuinely unknown members (`ml`, ad-hoc strings) should return `False`
  rather than iterating a fixed set that no longer includes the queried space.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

```python
if space == ExperimentDataEnum.groups:
    return any(exec_id_str in groups for groups in self.groups.values())
if space not in _SUPPORTED_SPACES:
    return False
return any(self.check_hash(executor_id, s) for s in _SUPPORTED_SPACES if s != space)
```

(Adjust the membership check to however group result keys are nested; the key point is to exclude
`space` itself from the fallback loop and handle `groups` explicitly.)

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

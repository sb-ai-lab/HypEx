## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`ABCRole.__copy__` (`hypex/dataset/roles.py:34-44`) rebuilds the role through the base constructor
and therefore **silently drops every attribute beyond `data_type`**:

```python
def __copy__(self, data_type: DefaultRoleTypes | None = None):
    return type(self)(data_type or self.data_type)
```

Affected metadata: `LagRole.parent` / `.lag` (lines 107-122) and
`TargetRole` / `PreTargetRole` `.cofounders` (lines 215-228, 265-281). `copy.copy(role)` and
`role.astype(...)` (line 73: `role = copy(self)`) both route through this, as does `asadditional()`.

This is reachable from library code: `DummyEncoderExtension._build_roles`
(`hypex/extensions/encoders.py:48`) calls `source_roles[src].asadditional(int)`, so dummy columns
derived from lagged features or confounder-tagged targets lose their lineage — downstream
`roles[target].cofounders` lookups (`hypex/ml/cupac.py:197,201`) and lag logic then see empty
metadata.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import copy
from hypex.dataset.roles import PreTargetRole, TargetRole, FeatureRole

r = copy.copy(PreTargetRole(parent="y", lag=1, cofounders=["x"]))
print(r.parent, r.lag, r.cofounders)     # None None []

t = copy.copy(TargetRole(cofounders=["z"]))
print(t.cofounders)                      # []

f = FeatureRole(parent="y", lag=3).asadditional(int)
print(f.parent, f.lag)                   # None None
```

(`deepcopy` is unaffected; only `copy.copy` / `astype` / `asadditional` are broken.)

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

A copy preserves `parent`, `lag` and `cofounders`; only `data_type` may be replaced when explicitly
requested.

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

- CUPAC's CUPED/CUPAC wiring depends on role lineage (`PreTargetRole(parent=..., lag=...)`,
  `TargetRole(cofounders=[...])`), so encoding a lagged/confounder-tagged column before CUPAC
  currently breaks that wiring silently.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Implement `__copy__` to carry over subclass attributes, e.g.:

```python
def __copy__(self, data_type=None):
    new = type(self)(data_type or self.data_type)
    for attr in ("parent", "lag", "cofounders"):
        if hasattr(self, attr):
            setattr(new, attr, getattr(self, attr))
    return new
```

(or per-class overrides on `LagRole`/`TargetRole`). Add round-trip tests for
`copy.copy` / `astype` / `asadditional` on metadata-carrying roles.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

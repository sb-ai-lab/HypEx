## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`ABTest._make_experiment` (`hypex/ab.py:81-87`) validates `multitest_method` against the enum's
string-value map:

```python
if (
    multitest_method is not None
    and multitest_method in ABNTestMethodsEnum._value2member_map_
):
    multitest_method = ABNTestMethodsEnum(multitest_method)
else:
    multitest_method = ABNTestMethodsEnum.holm
```

`_value2member_map_` is keyed by the string **values**, so any enum **member**
(`ABNTestMethodsEnum.bonferroni`, …) fails the membership test and silently falls back to **holm**.
The class docstring (lines 35-39) advertises exactly this enum form, and the usage example passes
`ABNTestMethodsEnum.bonferroni`.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
from hypex.ab import ABTest
from hypex.utils.enums import ABNTestMethodsEnum as E

def resolve(mt):
    if mt is not None and mt in E._value2member_map_:
        return E(mt)
    return E.holm

print(resolve("bonferroni"))    # ABNTestMethodsEnum.bonferroni  (correct)
print(resolve(E.bonferroni))    # ABNTestMethodsEnum.holm        (WRONG)
```

Verified end-to-end: 3 groups × 2 targets (4 comparisons), p = 0.01993 —
`multitest_method="bonferroni"` yields corrected p = 0.07971, while
`multitest_method=ABNTestMethodsEnum.bonferroni` yields Holm's 0.06783.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

Passing the enum member documented in the docstring applies the requested correction method.
(`None` and unknown **string** names are separate, already-known issues tracked by strict xfails in
`tests/scenario/test_ab_scenario.py`; the valid-enum case is not covered there.)

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

- The same pattern likely applies to other enum-valued parameters resolved via
  `_value2member_map_` membership (worth an audit of `ab.py`/`aa.py` signatures).

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Accept enum members first:

```python
if multitest_method is None:
    multitest_method = ABNTestMethodsEnum.holm
elif isinstance(multitest_method, ABNTestMethodsEnum):
    pass
elif multitest_method in ABNTestMethodsEnum._value2member_map_:
    multitest_method = ABNTestMethodsEnum(multitest_method)
else:
    raise ValueError(f"Unknown multitest method: {multitest_method!r}")
```

(And consider raising on unknown names — the xfail'd test already expects that.)

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

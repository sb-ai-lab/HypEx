## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`Matching` documents and accepts `"smd"`, `"psi"` and `"repeats"` as `quality_tests` options, but
none of them is implemented in `_make_experiment` — calling `Matching(quality_tests=...)` with any of
them raises a bare `KeyError` at experiment-construction time.

`hypex/matching.py:126-143`:

```python
test_mapping = {
    "t-test": TTest(...),
    # "psi": PSI(...),          # commented out
    "ks-test": KSTest(...),
    "chi2-test": Chi2Test(...),
    # no "smd", no "repeats"
}
...
executors += [OnRoleExperiment(executors=[test_mapping[test] for test in quality_tests], ...)]
```

The class docstring (lines 38-40, 75-82, 209-215) advertises all three. Additionally, the `SMD`
operator that *would* serve `"smd"` has a wrong formula (tracked in a separate known-bug xfail),
and `"repeats"` has no implementation anywhere in `hypex/`.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole
from hypex import Matching

rng = np.random.default_rng(0)
n = 200
df = pd.DataFrame({"id": np.arange(n), "treat": rng.integers(0, 2, n),
                   "x": rng.normal(size=n), "y": rng.normal(size=n)})
ds = Dataset(data=df, roles={"id": InfoRole(), "treat": TreatmentRole(),
                             "x": TargetRole(), "y": TargetRole()})

for qt in ["smd", "psi", "repeats"]:
    Matching(quality_tests=qt).execute(ds)   # KeyError 's' / 'p' / 'r'
```

### Expected Behavior

<!-- A clear and concise description of what the expected behavior is. -->

Either the three quality tests run as documented, or they are rejected with a clear
`ValueError: unknown quality test 'psi'` at construction — and removed from the docstring until
implemented.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (pandas backend; backend-independent)

### Additional Context

<!-- Add any other context about the problem here. -->

- `Matching(quality_tests="auto")` works (it resolves to the implemented subset), so the crash hits
  only users who follow the docstring.
- `hypex/operators/operators.py::SMD._inner_function` currently computes
  `(mean1 + mean2) / std1` instead of the standardized mean difference — wiring `"smd"` without
  fixing that would publish garbage quality metrics (already tracked by the strict xfail
  `test_smd_is_standardized_mean_difference`).

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Short term: validate `quality_tests` against `test_mapping` keys and raise a descriptive
`ValueError`; fix the docstring. Longer term: implement PSI (binning exists in the codebase),
wire `SMD` after fixing its formula, and either implement `repeats` or drop it from the API.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

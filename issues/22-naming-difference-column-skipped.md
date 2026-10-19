## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

A feature (column) literally named `difference` is **silently excluded** from A/A scoring and
best-split selection.

`_parse_metric_col` (`hypex/utils/naming.py:55-74`) scans for metric suffixes in a fixed order and
returns on the first hit. For a column `"difference TTest pass test_1"` the token `"difference"`
matches the **metric** suffix at position 0, so the parse yields
`("", "", "difference", "TTest pass test_1")` — an empty feature name. `_resolve_column_parts`
(`hypex/analyzers/aa.py:68-82`) then returns `None`, and the column is skipped by both
`AAScoreAnalyzer._analyze_aa_score` and `_compute_weighted_pvalues`: the feature never receives an
AA score and does not contribute to best-split selection, without any warning.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
from hypex.utils.naming import _parse_metric_col

print(_parse_metric_col("y TTest difference test_1"))         # ok: feature 'y'
print(_parse_metric_col("difference TTest pass test_1"))      # ('', '', 'difference', 'TTest pass test_1')
```

Then run `AATest` on data with a `TargetRole` column named `difference`: it is absent from the
`aa score` table while other targets are scored.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

The column-name grammar is `{feature} {Test} {metric} {group}`; the parser must anchor the metric
match to the **tail** of the name (e.g. scan from the last known test-name token backwards), so any
feature name — including `difference` — parses correctly.

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

- `difference` is a plausible feature name for pre/post metrics (e.g. a pre-computed difference
  column), so this is not purely theoretical.
- The failure is silent: no warning distinguishes "column skipped because unparseable" from
  "column intentionally not scored".

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Match the metric suffix against the end of the column name (or locate the test-name token first and
split around it), and emit a warning when a `pass`-containing column cannot be parsed so silent
skips become visible.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)

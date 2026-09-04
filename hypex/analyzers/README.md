# `hypex.analyzers` — Result Aggregation

Analyzers are the second analytical pass. They do not touch the raw data: they
read the tables and variables that comparators, operators and ML executors have
already written, and condense them into a verdict, a score, or a formatted
result table.

## Role in the architecture

```
comparators / operators           analyzers                reporters / Output
 write per-test results   →   read them by executor   →   flatten into the
 into analysis_tables         class, aggregate,           user-facing resume
 and variables                write a summary table
```

An analyzer is a plain `Executor` — no `Calculator` base, because there is no
stateless `calc()` worth exposing: their input is the whole `ExperimentData`.
They find their inputs with `data.get_ids(SomeExecutorClass, ...)`, never by
holding references, which is what lets a pipeline be rearranged freely.

## File map

| File | Contents |
|---|---|
| `aa.py` | `OneAAStatAnalyzer`, `AAScoreAnalyzer`. |
| `ab.py` | `ABAnalyzer`. |
| `matching.py` | `MatchingAnalyzer`. |
| `__init__.py` | Empty — import from the submodules (`from hypex.analyzers.aa import ...`). |

## Key classes

### `OneAAStatAnalyzer` (`aa.py`)

Summarises **one** A/A iteration.

* Collects every `GroupTTest`, `GroupKSTest` and `GroupChi2Test` table for the
  run and averages their `p-value` and `pass` columns.
* Combines them into `mean test score` with fixed weights:
  t-test ×1, KS-test ×2, chi²-test ×2, divided by the total weight used.
* NaNs are coerced to 0 before weighting.
* Writes one row into `analysis_tables[self.id]`.

Used in `AA_METRICS` (`hypex/aa.py`) and in `HOMOGENEITY_TEST`
(`hypex/homogeneity.py`).

### `AAScoreAnalyzer` (`aa.py`)

Summarises **all** A/A iterations and picks the best split.

`AAScoreAnalyzer(alpha: float = 0.05, key: str = "")`; the acceptance threshold is
`1 - alpha * 1.2`.

* `_analyze_aa_score` — for each test column it computes a feature weight
  `1 - |alpha - mean(pass)|` (a split behaves well when the share of rejections
  matches the nominal alpha), stores `score` and `pass` per test, and writes an
  `"aa score"` table.
* `_get_best_split` — ranks iterations by `2/3 × weighted mean p-value +
  1/3 × mean test score` and takes the argmax.
* `build_splitter_from_id` — reconstructs the winning `AASplitter` /
  `AASplitterWithStratification` from its stored id (see
  `Executor.build_from_id`). This is why splitter parameter hashes must be
  stable.
* `_set_best_split` — records the winning splitter id in `variables` and
  **re-runs** that splitter with `key="best"`, `save_groups=False`,
  `constant_key=False`, so the final dataset carries the chosen split.

`AA_SPLITER_CLASS_MAPPING` is the name→class registry used for reconstruction;
add new splitter classes there if they should be selectable.

### `ABAnalyzer` (`ab.py`)

`ABAnalyzer(multitest_method=None, alpha=0.05, equal_variance=True,
quantiles=None, iteration_size=20000, random_state=None, key="")`

* Gathers `GroupTTest` and `GroupUTest` tables and averages `p-value` / `pass`
  per treatment group, producing one summary row.
* **Multiple-testing correction** (`execute_multitest`), applied only when there
  are more than two groups:
  * any `ABNTestMethodsEnum` except `quantile` → `MultiTest` (statsmodels
    `multipletests`: bonferroni, sidak, holm, holm-sidak, simes-hochberg,
    hommel, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky);
  * `quantile` → `MultitestQuantile`, a resampling procedure parameterised by
    `iteration_size`, `equal_variance` and `random_state`.
  * The result is stored under the analyzer id with key `"MultiTest"`.

### `MatchingAnalyzer` (`matching.py`)

Reads the `MatchingMetrics` output from `ExperimentData.variables`, converts it
to a `Dataset` and transposes it into the final five-column table:

`Effect Size · Standard Error · P-value · CI Lower · CI Upper`

`MatchingOutput` looks for this class by default (`searching_class=MatchingAnalyzer`).

## How to work with it

Place an analyzer after the blocks it consumes:

```python
Experiment(executors=[
    OnRoleExperiment(executors=[GroupTTest(...), GroupKSTest(...)], role=TargetRole()),
    OneAAStatAnalyzer(),        # reads the two tables above
])
```

## How to add an analyzer

```python
from hypex.executor import Executor
from hypex.utils import ExperimentDataEnum


class MyAnalyzer(Executor):
    def _set_value(self, data, value, key=None):
        return data.set_value(ExperimentDataEnum.analysis_tables, self.id, value)

    def execute(self, data):
        ids = data.get_ids(MyComparator,
                           searched_space=ExperimentDataEnum.analysis_tables)
        tables = [data.analysis_tables[i]
                  for i in ids[MyComparator]["analysis_tables"]]
        ...
        return self._set_value(data, summary)
```

If the result should reach the user, add a matching reporter
(`hypex/reporters/`) and wire it into an `Output` (`hypex/ui/`).

## Gotchas

* **Silent no-ops.** If the upstream comparator did not run, `get_ids` returns an
  empty list and the analyzer produces nothing — no exception. Check pipeline
  order first when a `resume` comes back empty.
* **Id parsing.** `AAScoreAnalyzer` and `ABAnalyzer` split ids on
  `ID_SPLIT_SYMBOL` / `NAME_BORDER_SYMBOL` and index into the parts. Changing an
  executor's `_generate_params_hash` or `key` format can break them.
* **`AAScoreAnalyzer` executes another executor** (the reconstructed splitter)
  from inside `execute`. Keep that in mind when reasoning about side effects.
* `alpha` in `AAScoreAnalyzer` is flagged `TODO: rename` — it is an expected
  rejection rate, not a test significance level.

## Related modules

`../comparators/README.md` and `../operators/README.md` (produce the inputs) ·
`../reporters/README.md` (consume the outputs) · `../splitters/README.md`
(reconstructed by `AAScoreAnalyzer`).

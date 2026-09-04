# `hypex.hypotheses` — Declarative Experiments (inactive)

> **Status: dormant.** `hypothesis.py` and `__init__.py` are fully commented out.
> The JSON schema in `schemes/scheme.json` is present on disk, but nothing reads
> it. Importing `hypex.hypotheses` yields no names.

## Intended role in the architecture

A `Hypothesis` was to be the **serialisable form of a whole experiment** —
dataset, pipeline and report described as JSON, so an experiment could be
authored in a UI, stored, versioned, shared and replayed without Python code.
That corresponds to levels 1–3 of `schemes/architecture_levels.md`.

```
scheme.json  ──validates──►  config (dict or path)
                                  │  Hypothesis(config)
                                  ├── .dataset     → Dataset(...)
                                  ├── .experiment  → Experiment([...])   via Factory
                                  └── .report      → Reporter / Output
```

## File map

| File | Contents |
|---|---|
| `hypothesis.py` | Commented-out `Hypothesis` class. |
| `schemes/scheme.json` | The JSON Schema an experiment description must satisfy. |
| `__init__.py` | Commented out. |

## What the commented code did

`Hypothesis(config: str | dict)`:

1. Loaded the config from a path or accepted a dict directly.
2. Loaded `schemes/scheme.json` and validated the config with
   `jsonschema.validate`.
3. Split the config into `dataset`, `experiment` and `report` sections.
4. Required that the dataset section carry either inline `data` or a `path`, and
   raised otherwise.
5. Called `_parse_dataset()` / `_parse_config()`, delegating class construction
   to `hypex.factory.base.Factory`.

## If you are reviving this

* **Fix the config-path handling first.** The old code hardcoded a
  Windows-style path (`"hypex\\hypotheses\\schemes\\scheme.json"`); use
  `importlib.resources` or a path relative to `__file__`.
* **`jsonschema` is an extra dependency.** Check whether it is declared in
  `pyproject.toml` before relying on it.
* **Re-derive the schema from the current class surface.** Both the commented
  code and, most likely, `scheme.json` predate the current comparator naming
  (`GroupTTest`, `StatsTTest`), the `ml/` package and the role changes.
* **Revive `hypex/factory` at the same time** — a `Hypothesis` cannot build
  anything without it. See [`../factory/README.md`](../factory/README.md).
* **The role-name mapping already exists:** `default_roles` in
  `hypex/dataset/roles.py`.

## The nearest working alternative today

Everything the JSON layer would do can be done in Python right now:

```python
from hypex.experiments import Experiment
from hypex.ui.base import ExperimentShell, Output

experiment = Experiment(executors=[...])          # the "experiment" section
shell = ExperimentShell(experiment, Output(...))  # the "report" section
result = shell.execute(Dataset(roles=..., data=...))  # the "dataset" section
```

`Experiment.set_params({SomeClass: {...}})` covers the parameterisation a
template config would express.

## Related modules

`../factory/README.md` (the construction half of this feature) ·
`../experiments/README.md` and `../ui/README.md` (the working Python equivalent) ·
`schemes/architecture_levels.md` in the repo root.

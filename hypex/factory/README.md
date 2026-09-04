# `hypex.factory` — Config-Driven Construction (inactive)

> **Status: dormant.** Every line of `base.py` and `__init__.py` is currently
> commented out. Nothing imports this module, and importing it has no effect.
> It is kept in the tree as the anchor for a planned feature.

## Intended role in the architecture

A `Factory` was meant to be the bridge between a **declarative description of an
experiment** and the object graph that `Experiment` needs — the missing piece
between levels 2–3 of `schemes/architecture_levels.md` (a scenario built in a
platform UI, or a template with tunable parameters) and level 5 (a pipeline
written in Python).

```
JSON / dict description          Factory              runnable objects
{"experiment": [                   │
   {"class": "GroupTTest",         ├──►  registry lookup by class name
    "params": {...}}, ...],        ├──►  role instantiation from role names
 "dataset": {...},                 └──►  Experiment([...]) + Dataset(...)
 "report": {...}}
```

The commented-out code shows the shape it was heading for: an `all_classes` list
enumerating every publicly constructible executor, role, experiment container and
reporter, so a name in a config could be resolved to a class via
`getattr(sys.modules[...], name)`.

## File map

| File | Contents |
|---|---|
| `base.py` | Commented-out `all_classes` registry and `Factory` sketch. |
| `__init__.py` | Commented out. |

## Relationship to `hypex/hypotheses`

`hypex/hypotheses/hypothesis.py` — also fully commented out — is the consumer
side: it reads a JSON config, validates it against
`hypex/hypotheses/schemes/scheme.json`, and calls `Factory` to build the objects.
The two modules are one feature split in half. See
[`../hypotheses/README.md`](../hypotheses/README.md).

## If you are reviving this

1. **Do not rebuild `all_classes` by hand.** The commented list is already stale
   — it references classes that no longer exist (`ATE`, `Arg1Role`, `Arg2Role`)
   and predates `StatsComparator`, `GroupHypothesisTesting`, the `Group*`
   renaming and the whole `ml/` package. Derive the registry from the packages'
   `__all__` lists, or from a decorator-based registration, so it cannot drift
   again.
2. **Reuse the existing role registry.** `default_roles` in
   `hypex/dataset/roles.py` already maps role-name strings to role instances.
3. **Match the parameter surface that already exists.** `Executor.set_params`
   accepts `{"attr": value}` and `{Class: {"attr": value}}` — a config format
   should map onto that rather than inventing a third convention.
4. **Update the JSON schema together with the code** —
   `hypex/hypotheses/schemes/scheme.json` is the contract a UI would generate
   against.

## Gotchas

* Do not import `hypex.factory` expecting names from it; there are none.
* The `hypex/hypotheses/schemes/scheme.json` file is live on disk even though the
  code that reads it is commented out — it may or may not reflect the current
  class names.

## Related modules

`../hypotheses/README.md` (the other half) · `../experiments/README.md` (what a
factory would build) · `schemes/architecture_levels.md` in the repo root (why
this layer was planned).

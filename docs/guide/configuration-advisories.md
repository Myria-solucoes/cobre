# Configuration advisories

## Training without an iteration limit

When training is enabled and `training.stopping_rules` omits `iteration_limit`,
setup emits a WARN tracing event identifying the implicit maximum of 100
iterations. Supply an explicit limit to choose the budget:

```json
{
  "training": {
    "stopping_rules": [
      { "type": "time_limit", "seconds": 30 },
      { "type": "iteration_limit", "limit": 7 }
    ],
    "stopping_mode": "any"
  }
}
```

The example stops when either configured rule fires. An omitted or empty rule
list resolves to the default iteration rule. A nonempty list without an iteration
rule retains its rules and `any`/`all` composition; the implicit maximum remains
the training loop's iteration budget. The warning does not insert another rule
into that list. Disabled training does not emit this advisory.

The CLI sends tracing diagnostics to stderr according to its logging settings.
Rust callers can capture the WARN event with a tracing subscriber.

## Reserved spillage discretization

`spillage_discretization_points` in an FPHA configuration is reserved and does
not affect the hyperplanes. Case validation emits a `ModelQuality` warning when
the parameter is explicitly supplied, including an explicit default value.
The diagnostic identifies the hydro and its stage-range or seasonal entry.
It is available in both CLI validation and Python validation reports.

Computed hyperplanes discretize volume and turbine flow at zero spillage.
Precomputed hyperplanes are read from their input table. Omit the unused key:

```json
{ "source": "computed", "volume_discretization_points": 5,
  "turbine_discretization_points": 5 }
```

This is the `fpha_config` object within a production-model entry, not a complete
case file. The warning does not remove existing input validation or add a
spillage axis to the fitting algorithm.

## Gap admissibility and validation error fields

The gap stopping rule requires enumerated forward selection and a uniform
effective risk measure across stages: expectation, or CVaR with matching
parameters. Sampled forwards and nonuniform effective risk measures remain
inadmissible.

On failure, `cobre validate --json` names the error category in `error.phase`.
Python's `cobre.io.validate` uses `errors[].kind`. These field names are distinct;
the shared category vocabulary does not make their response shapes identical.

# Parallel convergence readings

Cobre runtime `v0.15.0-myria.4` publishes two convergence readings side by
side. They answer different questions and must not be substituted for one
another:

- `gap_percent` preserves the legacy upper-bound/lower-bound comparison;
- `gap_regime = "lb_stability_v1"` describes recent movement of the best lower
  bound.

The second reading was added in parallel so maintainers can compare both series
without silently changing the meaning of an existing field or breaking
consumers of `gap_percent`.

## Why the parallel diagnostic exists

The legacy value is serialized as:

```text
100 * (current_upper_bound - current_lower_bound)
    / max(1, abs(current_lower_bound))
```

With enumerated forward selection and a compatible risk measure, the upper
bound is exact and this comparison can support the native gap stopping rule.
With sampled forward selection, however, the current upper bound is the mean
estimated by that iteration's forward sample. It is a noisy statistical
estimate, not an exact bound. It can move sharply between iterations or fall
below the lower bound, which makes the reported legacy gap negative. Cobre
therefore rejects the gap stopping rule for sampled training, although it keeps
publishing the historical `gap_percent` series.

`lb_stability_v1` provides a separate observation for these runs: is the best
lower bound still improving over a recent window? It deliberately does not
claim to repair or replace the legacy gap.

## Definition of `lb_stability_v1`

For observation `i`, define the monotone best lower bound as:

```text
best_lb_i = max(raw_lower_bound_1, ..., raw_lower_bound_i)
```

For each full rolling window, the diagnostic is:

```text
100 * (best_lb_last - best_lb_first) / max(1, abs(best_lb_last))
```

Runtime `v0.15.0-myria.4` fixes the window at three recorded observations.
Consequently, the first two observations in each output are null. This also
applies to a resumed attempt: iteration numbers may continue from an earlier
checkpoint, but the new output must first accumulate three observations of its
own.

Using the monotone best bound prevents a decrease in a raw lower-bound
observation from producing a negative stability value.

Do not confuse this output diagnostic with the configurable `bound_stalling`
stopping rule. `bound_stalling` compares raw lower-bound endpoints using a
user-selected window and tolerance and may terminate training. In contrast,
`lb_stability_v1` uses the monotone best lower bound, has a fixed
three-observation window, has no tolerance, and is display-only.

## How to interpret the value

| Value | Meaning |
| --- | --- |
| `null` | The output does not yet contain the three observations required by the window. It is not zero and does not indicate an error. |
| `0%` | The best lower bound did not advance between the window endpoints. This may be temporary stalling; it does not prove convergence. |
| Positive | The best lower bound advanced by that percentage over the window. A larger value means more recent LB movement, not greater error or distance to the optimum. |

For example, `29.44%` means that the best lower bound increased by `29.44%`
between the first and last observations in the window, normalized by the last
best lower bound. It does **not** mean that the solution is `29.44%` away from
the optimum.

Consider four raw lower-bound observations:

| Observation | Raw LB | Best LB through observation | `lb_stability_v1` |
| ---: | ---: | ---: | ---: |
| 1 | 100 | 100 | `null` |
| 2 | 110 | 110 | `null` |
| 3 | 105 | 110 | `(110 - 100) / 110 = 9.09%` |
| 4 | 121 | 121 | `(121 - 110) / 121 = 9.09%` |

The raw LB falls at observation 3, but the best LB remains 110. At observation
4, the three-observation window contains observations 2 through 4.

## Interpretation limits

`lb_stability_v1` has the explicit contract:

- `gap_regime_classification = "heuristic"`;
- `gap_regime_stop_eligible = false`.

It never terminates training and is not an optimality certificate. In
particular, a value near zero does not by itself establish that:

- the policy is sufficiently trained;
- the upper-bound estimate has stabilized;
- cuts or operational decisions have stabilized;
- simulated CMO, generation, storage, or other distributions are stable; or
- another window length would lead to the same assessment.

The three-observation window makes the experimental series responsive in short
runs, but also sensitive to short plateaus. No universal acceptance threshold
is defined for this version. Assess the series over multiple windows together
with the bounds, sampling uncertainty, policy outputs, and simulation results.

## Output contract

Each row of `training/convergence.parquet` contains:

| Field | Meaning |
| --- | --- |
| `gap_percent` | Unchanged legacy reading; nullable when the lower bound is non-positive. |
| `gap_regime` | Stable diagnostic identifier, `lb_stability_v1`. |
| `gap_regime_value_percent` | Diagnostic value in percent; nullable until the window is full. |
| `gap_regime_window_iterations` | Required observation count; `3` in this version. |
| `gap_regime_classification` | Evidence class, `heuristic`. |
| `gap_regime_stop_eligible` | Always `false` in this version. |

The final descriptor and value are repeated under `convergence.gap_regime` in
`training/metadata.json`.

Read both series without reinterpreting the legacy field:

```python
import pyarrow.parquet as pq

table = pq.read_table(
    "output/training/convergence.parquet",
    columns=[
        "iteration",
        "lower_bound",
        "upper_bound",
        "upper_bound_kind",
        "gap_percent",
        "gap_regime",
        "gap_regime_value_percent",
        "gap_regime_window_iterations",
        "gap_regime_classification",
        "gap_regime_stop_eligible",
    ],
)
print(table.to_pandas().tail())
```

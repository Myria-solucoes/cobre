# Convergence gap regimes

Cobre publishes two parallel convergence readings. They answer different
questions and must not be substituted for one another.

`gap_percent` remains the legacy comparison between the current forward value
and lower bound. Its formula and serialization are unchanged, including
negative values. Existing consumers can continue reading it without a semantic
migration.

`gap_regime = "lb_stability_v1"` is an experimental lower-bound stability
diagnostic. For each full rolling window, it reports the percentage increase
from the window's first monotone best lower bound to its last:

```text
100 * (best_lb_last - best_lb_first) / max(1, abs(best_lb_last))
```

where `best_lb_i` is the best lower bound observed through iteration `i`. The
value is null until the declared `gap_regime_window_iterations` observations
are available. A decrease in a raw lower-bound observation cannot produce a
negative stability value because the diagnostic uses the monotone best bound.

The companion fields state the interpretation contract:

- `gap_regime_classification = "heuristic"`
- `gap_regime_stop_eligible = false`

The diagnostic therefore describes policy-training stability; it is not an
optimality certificate and never terminates training. The same final descriptor
and value appear under `convergence.gap_regime` in
`training/metadata.json`.

Read both regimes without reinterpreting the legacy value:

```python
import pyarrow.parquet as pq

table = pq.read_table(
    "output/training/convergence.parquet",
    columns=[
        "iteration",
        "gap_percent",
        "gap_regime",
        "gap_regime_value_percent",
        "gap_regime_classification",
        "gap_regime_stop_eligible",
    ],
)
print(table.to_pandas().tail())
```

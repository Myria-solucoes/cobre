# Original-compatible runtime

The application label `0.16` selects the upstream solver with only the
automatic stationarity compatibility backport. It is not the corrected Myria
variant: the upstream CVaR mixture and training behavior remain unchanged.

On real NEWAVE inputs, automatically fitted PAR coefficients can have negative
implied residual variance. Without this backport the bridge cannot materialize
FPHA hyperplanes and conversion fails before training. The compatibility path
removes the fitted annual component for the affected hydro, then removes the
highest lag of the failing season until the periodic stationarity test passes.
It never repairs explicit user-supplied coefficients or unrelated load errors.

CLI and Python wheels must come from the same immutable original-compatible
release, with their published SHA-256 verified before installation. The
`original-runtime-release.yml` workflow tests and builds both Linux architectures.

```python
import cobre.io

report = cobre.io.validate("converted-case")
assert report["valid"], report["errors"]
for warning in report["warnings"]:
    if warning["kind"] == "StationarityRegularized":
        print(warning["entity"], warning["message"])
```

Regularization changes the automatically estimated stochastic model. Preserve
these warnings and inspect large reductions; successful execution is not a
claim of scientific equivalence with NEWAVE or convergence of a short run.

# Myria runtime releases

Myria runtime tags package Cobre for deployment without changing the public
input schema or Python API of their base Cobre release. Their release notes in
[`CHANGELOG.md`](../../CHANGELOG.md) are the authoritative record of behavioral
differences from that base release.

Runtime artifacts are attached to a versioned GitHub release. They are not a
replacement for the crates.io or PyPI distributions: use them when deploying
the corresponding Myria runtime tag.

## Install the CLI artifact

Choose a tag from the repository's GitHub Releases page and set it explicitly:

```bash
REPOSITORY=Myria-solucoes/cobre
RELEASE_TAG=v0.15.0-myria.1
ARTIFACT_DIR=$(mktemp -d)

case "$(uname -m)" in
  x86_64) ASSET=cobre-cli-x86_64-unknown-linux-gnu.tar.xz ;;
  aarch64|arm64) ASSET=cobre-cli-aarch64-unknown-linux-gnu.tar.xz ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

gh release download "$RELEASE_TAG" \
  --repo "$REPOSITORY" \
  --pattern "$ASSET" \
  --dir "$ARTIFACT_DIR"
tar -xJf "$ARTIFACT_DIR/$ASSET" -C "$ARTIFACT_DIR"
"$ARTIFACT_DIR/cobre" --version
```

Keep the tag in deployment configuration rather than resolving "latest" at
runtime. That makes rollback and reproduction deterministic.

## Install the Python runtime wheel

The runtime release contains CPython 3.12 stable-ABI wheels for supported Linux
architectures. Select exactly one architecture-specific wheel:

```bash
REPOSITORY=Myria-solucoes/cobre
RELEASE_TAG=v0.15.0-myria.1
ARTIFACT_DIR=$(mktemp -d)

case "$(uname -m)" in
  x86_64) WHEEL_PATTERN='cobre_python-*-x86_64.whl' ;;
  aarch64|arm64) WHEEL_PATTERN='cobre_python-*-aarch64.whl' ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

gh release download "$RELEASE_TAG" \
  --repo "$REPOSITORY" \
  --pattern "$WHEEL_PATTERN" \
  --dir "$ARTIFACT_DIR"
python -m pip install "$ARTIFACT_DIR"/*.whl
python -c 'import cobre; print(cobre.__version__)'
```

The GitHub release page publishes a SHA-256 digest beside each asset. Deployment
automation should verify that digest before installation.

## Automatic PAR stationarity regularization

Historical inflow fitting can produce a periodic autoregressive model that is
numerically non-stationary for one hydro and season. A Myria runtime based on
Cobre 0.15.0 handles this only for models estimated automatically from
`scenarios/inflow_history.parquet`:

1. remove the fitted annual component for the affected hydro, when present;
2. if necessary, reduce the failing season's order one lag at a time;
3. stop as soon as the periodic stationarity check passes; and
4. record every adjustment in the estimation report.

An autoregressive model supplied explicitly by the user is never rewritten by
this fallback. It retains the normal hard validation failure when it is not
stationary.

The Python validation result exposes each automatic adjustment as a structured
warning. Callers should preserve and display it rather than treating it as a
conversion failure:

```python
import cobre

report = cobre.io.validate("converted-case")
if not report["valid"]:
    raise RuntimeError(report["errors"])

for warning in report["warnings"]:
    if warning.get("kind") == "StationarityRegularized":
        print(
            warning["entity"],
            warning["message"],
            warning["file"],
            sep=" | ",
        )
```

A regularization warning means Cobre reduced model complexity to obtain a valid
automatic stochastic model. Review repeated or large order reductions as a data
or modeling signal, especially when a hydro's history is short, constant, or
nearly deterministic.

## Publishing a runtime tag

The release workflows are intentionally separate from the normal package
publication workflows:

- `.github/workflows/myria-runtime-release.yml` creates the release and uploads
  the x86_64 Linux CLI archive and wheel.
- `.github/workflows/myria-runtime-wheel-arm64.yml` adds the matching aarch64
  Linux artifacts to that existing release.

Before publishing a new runtime tag, update `CHANGELOG.md`, this guide when the
runtime contract changed, and any affected user guide. Document the base Cobre
release, behavioral delta, supported architectures, and validation surface.

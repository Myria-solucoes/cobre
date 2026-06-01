"""Cobre — Python bindings for the Cobre power systems solver.

This is the public ``cobre`` package. The actual solver is implemented in the
compiled extension module ``cobre._native`` (a PyO3/maturin ``cdylib``). This
``__init__`` re-exports the compiled module's public surface so that
``import cobre``, ``cobre.Study``, ``cobre.run.run(...)`` and friends resolve
exactly as they did before the crate adopted the maturin mixed layout.

The split into a private compiled ``_native`` plus a pure-Python ``cobre``
package exists so ergonomic wrappers can be authored in Python rather than
PyO3. This file is the single extension point for those additions.
"""

from __future__ import annotations

import importlib
import sys

# Pull the top-level public names (Study, Policy, version_info, __version__)
# from the compiled extension module.
from ._native import *  # noqa: F401,F403

# `from ._native import *` may skip dunders; re-bind __version__ explicitly so
# `cobre.__version__` is preserved byte-for-byte from the compiled module.
from ._native import __version__ as __version__

# Re-export the compiled submodules under their public `cobre.*` names. Binding
# them here exposes `cobre.run` etc. as attributes; registering each in
# `sys.modules` under the public name makes `import cobre.run` resolve to the
# very same compiled module object (no shadowing copy). This mirrors the Rust
# `register_submodule` sys.modules trick, but on the public side.
from ._native import errors as errors
from ._native import io as io
from ._native import model as model
from ._native import run as run
from ._native import schema as schema

sys.modules["cobre.errors"] = errors
sys.modules["cobre.io"] = io
sys.modules["cobre.model"] = model
sys.modules["cobre.run"] = run
sys.modules["cobre.schema"] = schema

# `results` is the pure-Python wrapper (`results.py`), not the bare compiled
# `_native.results`: it re-exports every compiled `load_*`/`report` and adds the
# `summary(dir) -> str` renderer. Importing `cobre._native` above eagerly bound
# the compiled `results` child both as a `cobre.results` attribute (via the
# `from ._native import *`) and as `sys.modules["cobre.results"]` (the child's
# unqualified `__name__` makes the import machinery register it under the parent
# package). A plain `from . import results` would short-circuit to that compiled
# child without ever loading the wrapper. Clear both shadows, then load the
# wrapper module explicitly and rebind it under the public name. The wrapper
# re-exports the compiled `load_*`/`report`, so no public name regresses.
sys.modules.pop("cobre.results", None)
if "results" in globals():
    del globals()["results"]
results = importlib.import_module("cobre.results")
sys.modules["cobre.results"] = results

# Top-level public classes/functions re-exported from `_native`.
from ._native import Policy as Policy
from ._native import Study as Study
from ._native import version_info as version_info

# --- Extension point ---------------------------------------------------------
# Pure-Python ergonomic wrappers are layered on top of the compiled surface
# here (e.g. a `summary(dir)` renderer). When such wrappers land, import them
# above and add their public names to `__all__` below. The typed `cobre.errors`
# exception hierarchy is registered from Rust (under `cobre._native.errors`) and
# re-exported above, so `import cobre.errors` and
# `from cobre.errors import ValidationError` resolve to the compiled classes.

__all__ = [
    "Study",
    "Policy",
    "version_info",
    "errors",
    "io",
    "model",
    "run",
    "results",
    "schema",
]

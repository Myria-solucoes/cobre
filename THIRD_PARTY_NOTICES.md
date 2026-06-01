# Third-Party Notices

Cobre is licensed under the Apache License 2.0 (see `LICENSE`). It bundles the
following third-party components as git submodules under
`crates/cobre-solver/vendor/`. Each component retains its original license;
bundling does not relicense it.

These components are C++ libraries compiled and statically linked by the
`cobre-solver` build script. They are **not** Rust crates and are therefore
**not** covered by cargo-deny's license checks (`deny.toml` governs only Cargo
dependencies). This file is the manual record of their license obligations.

## HiGHS — MIT

- Submodule: `crates/cobre-solver/vendor/HiGHS` (tag `v1.13.1`)
- Upstream: <https://github.com/ERGO-Code/HiGHS>
- Linked: statically, when the `highs` feature is enabled (enabled by default).
- License text: `crates/cobre-solver/vendor/HiGHS/LICENSE.txt`
- HiGHS bundles its own sub-dependencies; see
  `crates/cobre-solver/vendor/HiGHS/THIRD_PARTY_NOTICES.md` for the full list.

## Clp — Eclipse Public License 2.0 (EPL-2.0)

- Submodule: `crates/cobre-solver/vendor/Clp` (tag `releases/1.17.11`)
- Upstream: <https://github.com/coin-or/Clp>
- Linked: statically, **only** when the optional `clp` feature is enabled
  (disabled by default).
- License: <https://www.eclipse.org/legal/epl-2.0/>
- License text: `crates/cobre-solver/vendor/Clp/LICENSE`

EPL-2.0 source code is available via the upstream repository and via the
vendored submodule in this repository.

## CoinUtils — Eclipse Public License 2.0 (EPL-2.0)

- Submodule: `crates/cobre-solver/vendor/CoinUtils` (tag `releases/2.11.13`)
- Upstream: <https://github.com/coin-or/CoinUtils>
- Linked: statically, **only** when the optional `clp` feature is enabled
  (disabled by default).
- License: <https://www.eclipse.org/legal/epl-2.0/>
- License text: `crates/cobre-solver/vendor/CoinUtils/LICENSE`

EPL-2.0 source code is available via the upstream repository and via the
vendored submodule in this repository.

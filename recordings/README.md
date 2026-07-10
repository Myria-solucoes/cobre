# Terminal Recordings

This directory contains VHS tape files for generating terminal recordings (GIFs)
that demonstrate the Cobre CLI in action.

Generated GIFs are committed here as the source of truth. The documentation site
([cobre-docs](https://github.com/cobre-rs/cobre-docs)) vendors them from a tagged
release via `npm run refresh:recordings -- --ref <tag>` — so after regenerating a
GIF for a CLI change, cut a release and re-run that command in cobre-docs.
Temporary files (`demo/`) are gitignored.

## Prerequisites

### Cobre binary

Install the `cobre` binary from the workspace root:

```sh
cargo install --path crates/cobre-cli
```

Verify it is on your PATH:

```sh
cobre version
```

### VHS (for `.tape` files)

VHS generates GIF and SVG animations from `.tape` script files.

```sh
# macOS
brew install vhs

# Go toolchain (any platform)
go install github.com/charmbracelet/vhs@latest
```

### jq (for validation-error.tape)

```sh
# macOS
brew install jq

# Debian / Ubuntu
sudo apt-get install jq

# Fedora
sudo dnf install jq
```

## Brand Theme

All VHS tape files use the Cobre brand color palette instead of a named theme.
The colors are set with individual `Set` directives at the top of each tape:

| Directive            | Value              | Brand name |
| -------------------- | ------------------ | ---------- |
| `Set Background`     | `#0F1419`          | Midnight   |
| `Set Foreground`     | `#C8C6C2`          | Body       |
| `Set CursorColor`    | `#B87333`          | Copper     |
| `Set SelectionColor` | `#1A2028`          | Surface    |
| `Set FontFamily`     | `"JetBrains Mono"` | —          |
| `Set WindowBar`      | `"Colorful"`       | —          |
| `Set BorderRadius`   | `8`                | —          |
| `Set Padding`        | `12`               | —          |

The full brand palette is documented in `docs/internal/BRAND-GUIDELINES.md`.

## VHS Recordings

Run the tape files from the repository root. VHS writes output next to the tape file.

```sh
# Quick Start demo (init → run → report)
vhs recordings/quickstart.tape
# Output: recordings/quickstart.gif

# Validate demo (init → validate on a valid case)
vhs recordings/validation.tape
# Output: recordings/validation.gif

# Validation error demo (init → corrupt JSON with jq → validate showing errors)
vhs recordings/validation-error.tape
# Output: recordings/validation-error.gif

# Multi-threading speedup demo (--threads 1 vs --threads 4, side-by-side timing)
vhs recordings/multithreading.tape
# Output: recordings/multithreading.gif
```

The `validation-error.tape` uses `jq` to corrupt the 1dtoy case on the fly (no
pre-built broken directory is committed). It applies two mutations:

- The `reservoir` object is deleted from `hydros.json`, producing a missing
  required field error in the schema validation layer.
- `max_turbined_m3s` is set to a negative value, which would trip a constraint
  check in the semantic validation layer.

Validation runs the layers in strict dependency order and stops after the schema
layer when that layer collects errors, because the later layers consume parsed
data the schema layer failed to produce. As a result the recording surfaces the
schema error — a red `error:` label and a non-zero exit code — while the semantic
mutation is never reached. To observe the semantic error on its own, corrupt only
`max_turbined_m3s` and leave the schema intact.

The `multithreading.tape` runs the same 1dtoy case twice in sequence — first with
`--threads 1`, then with `--threads 4` — so the post-run summary timing lines appear
back-to-back in the recording for a direct comparison.

## Embedding

Reference GIF output in Markdown:

```markdown
![Quick Start](recordings/quickstart.gif)
```

For SVG output (smaller file size, no browser autoplay restrictions):

```markdown
![Quick Start](recordings/quickstart.svg)
```

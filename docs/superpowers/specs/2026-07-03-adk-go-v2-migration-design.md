# adk-go v2 migration — design

## Context

This repo (`github.com/Alcova-AI/adk-anthropic-go`) implements `model.LLM` for Anthropic
Claude models, for use with Google's Agent Development Kit for Go. It currently tracks
`google.golang.org/adk v1.0.0`. Google has released `google.golang.org/adk/v2 v2.0.0`
(tag `v2.0.0` on `google/adk-go`, module path `google.golang.org/adk/v2`, requires Go 1.25+).

Per Go's module rules, any module reaching major version 2+ must have its import path
suffixed with `/v2`. Since this package's public API changes (import path) whenever its
adk-go dependency's import path changes, this is a natural point to also cut this
package's own v2.

A scratch-copy spike (module path swapped to `/v2`, dependency bumped to
`google.golang.org/adk/v2 v2.0.0`, imports updated) compiled cleanly with `go vet` and
passed the full existing test suite with **no source changes** beyond import paths. The
`model.LLM` interface (`Name() string`, `GenerateContent(...)`) and `LLMRequest` are
byte-identical between v1.0.0 and v2.0.0. `LLMResponse` in v2 adds four new optional
fields: `ModelVersion`, `InputTranscription`, `OutputTranscription`,
`SessionResumptionHandle`. Only `ModelVersion` is relevant to this package (the other
three are for adk-go's live/bidi audio streaming, which Anthropic's API doesn't support).

## Goals

1. Ship a `v2` release of this package that depends on `adk-go v2.0.0`.
2. Keep the existing `adk-go v1.0.0`-based line alive and patchable going forward
   (some consumers won't be ready to move to adk-go v2 immediately).
3. Make the existing auto-tagging CI workflow understand both lines, so future PRs to
   either line get tagged correctly without manual intervention.
4. Populate the new `ModelVersion` field from Anthropic's resolved model string.

## Non-goals

- No behavior changes beyond `ModelVersion`. This is a dependency migration, not a
  feature release.
- No engagement with adk-go v2's higher-level features (graph workflow engine,
  collaboration agents, unified `agent.Context`) — none of them are reachable from a
  `model.LLM` implementation; they're runner/agent-level concerns in adk-go itself.
- No change to `anthropic-sdk-go` version (stays pinned at v1.43.0 — nothing forces it
  up, `go mod tidy` leaves it alone).

## Design

### 1. Branching model

Two permanent branches from this point forward:

- **`v1`** — new branch, cut from `main`'s current tip before this migration lands.
  Frozen adk-go v1.x baseline. Module path is unchanged
  (`github.com/Alcova-AI/adk-anthropic-go` — Go doesn't require a path suffix below
  major version 2). Tag its tip `v1.0.0` (promotes v0.1.18 to a stable release; no code
  changes needed — this is a version-number relabeling, not a breaking change) after
  adding a one-line `CHANGELOG.md` entry noting the promotion. Future fixes for adk-go
  v1.x consumers land here as ordinary PRs, tagged `v1.x.x`.

- **`main`** — becomes the v2 line permanently. This migration PR lands here. Module
  path becomes `github.com/Alcova-AI/adk-anthropic-go/v2`, dependency becomes
  `google.golang.org/adk/v2 v2.0.0`, tagged `v2.0.0`. All new feature work targets this
  branch going forward.

This generalizes: the next major migration (v3) would branch today's `main` off as a
new `v2` maintenance branch (name matches its tag prefix) and `main` becomes v3.

### 2. Code changes on `main` (the v2 PR)

Files touched, all mechanical except the `ModelVersion` addition:

- `go.mod` — module path → `.../v2`; `google.golang.org/adk v1.0.0` →
  `google.golang.org/adk/v2 v2.0.0`; `go mod tidy` (genai auto-bumps 1.40.0 → 1.57.0,
  forced transitively by adk v2; anthropic-sdk-go stays at v1.43.0)
- `anthropic.go` — import `google.golang.org/adk/v2/model`; self-import
  `github.com/Alcova-AI/adk-anthropic-go/v2/converters`
- `anthropic_test.go` — same import updates
- `converters/response.go` — import `google.golang.org/adk/v2/model`; in
  `MessageToLLMResponse`, set `resp.ModelVersion = string(msg.Model)`
- `converters/converters_test.go` — self-import update (`package converters_test`
  imports the module's own `converters` package by full path)
- `cache_test.go` — self-import update
- `doc.go` — update `go get`/import examples to the `/v2` path
- `README.md` — update `go get`/import examples to the `/v2` path; add a short
  "Versioning" section explaining the v1 (adk-go v1.x) / v2 (adk-go v2.x) split, with
  install commands for each and a link to adk-go's own
  [migration guide](https://github.com/google/adk-go/blob/main/README-v2.md) for
  changes that affect callers directly (e.g. `session.NewEvent` context argument,
  `ToolContext`/`CallbackContext` unification) — those affect ADK application code, not
  this adapter, but consumers upgrading adk-go itself will hit them.
- `CHANGELOG.md` — new `v2.0.0` entry describing the migration

Streaming responses: the accumulated `anthropic.Message` already carries `.Model` from
the SDK's event accumulation, so the same `MessageToLLMResponse` conversion picks it up
for both the streaming and non-streaming paths — no separate handling needed.

### 3. Release automation

`scripts/gh/determine_highest_version_prefix.py` currently finds the single highest
`vMAJOR.MINOR.PATCH` tag repo-wide and bumps its patch. That breaks once two major
series coexist (it would treat a `v1.x.x` tag as up-for-patch-bump even when the merge
target is the v2 line, and vice versa).

New logic:

- The workflow passes the PR's base branch (`github.event.pull_request.base.ref`) to
  the script as `BASE_REF`.
- If `BASE_REF` matches `^v(\d+)$` (e.g. `v1`), the target major is that number.
- Otherwise (e.g. `main`), the target major is a constant in the script,
  `CURRENT_MAIN_MAJOR = 2`, updated by hand at the next major migration.
- `next_version` filters existing tags to the target major, takes the highest, bumps
  the patch; if none exist yet in that series, falls back to `<major>.0.0`. This
  naturally bootstraps both `v1.0.0` and `v2.0.0` without special-casing.

`.github/workflows/tag-merged-pr.yml` changes:

- Pass `BASE_REF` (from `github.event.pull_request.base.ref`) as an env var to both the
  preview job and the tag job.
- Fix the changelog step's "previous tag" lookup to filter by the same major series
  before picking the second-highest tag (today it picks the repo-wide second-highest,
  which would be wrong once both series have tags).
- Add a `branches: [main, v1]` filter to the `pull_request` trigger so merges into
  unrelated feature branches don't get tagged.

### 4. Rollout sequence

1. Push `v1` branch (from current `main` tip) and tag `v1.0.0` on it directly —
   confirmed with the user before pushing, since this touches shared remote state.
2. Open the v2 migration PR against `main` with the changes above.
3. Merge; verify the updated workflow tags it `v2.0.0`.

## Testing / verification

- `go vet ./...` and `go test ./...` on the migrated code (already verified in a
  scratch spike against the real `adk-go v2.0.0` module — clean).
- Add/extend a test asserting `MessageToLLMResponse` sets `ModelVersion` from the
  Anthropic message's `Model` field.
- Manually validate the tagging script's new major-aware logic with a small local
  test (feed it a fake tag list + `BASE_REF` and check the output), since it's not
  covered by Go tests.

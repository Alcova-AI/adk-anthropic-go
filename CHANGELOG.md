# Changelog

## [v0.1.17] - Drop thinking under forced tool_choice

- Anthropic rejects extended thinking (manual or adaptive) when `tool_choice.type` is `"tool"` or `"any"`. Sent together, the API may either return a 400 or — worse — silently produce a text/thinking response with no `tool_use` block, which looks to callers like the model refused to use the tool.
- The genai-side shape that lands callers in this trap is easy to write by accident: `ToolConfig.FunctionCallingConfig.Mode = ModeAny` (with or without `AllowedFunctionNames`) maps to forced `tool_choice`, and `ThinkingConfig.ThinkingLevel ∈ {Low, Medium, High}` on adaptive-capable models (Sonnet 4.6+, Opus 4.6+, Mythos Preview) maps to adaptive thinking. Both happily go on the wire.
- `convertRequest` now drops `params.Thinking` and `params.OutputConfig.Effort` whenever `params.ToolChoice` is forced. The forced `tool_choice` is the load-bearing semantic — the caller has pinned the response shape — so downgrading thinking is the less-surprising of the two corrections.
- New exported helper `converters.IsForcedToolUse(tc anthropic.ToolChoiceUnionParam) bool` for callers that want the same check.

## [v0.1.16] - Model-aware adaptive thinking + effort mapping

- Upgrade `anthropic-sdk-go` from v1.28.0 to v1.43.0 — picks up `ModelClaudeOpus4_7`, `ModelClaudeMythosPreview`, and `OutputConfigEffortXhigh` constants.
- New model-aware mapper `ThinkingConfigToAnthropic(cfg, model)` returning `ThinkingMapping{Thinking, Effort}`:
  - On adaptive-capable models (Sonnet 4.6+, Opus 4.6+, Opus 4.7, Mythos Preview), `ThinkingLevel: Low/Medium/High` maps to **adaptive mode + matching `OutputConfig.Effort`** (low/medium/high) — instead of the old single-budget mapping. `IncludeThoughts: true` similarly maps to adaptive + high effort.
  - On non-adaptive models (Sonnet 4.5, Haiku 4.5, etc.), the same fields fall back to manual extended thinking (`type: "enabled", budget_tokens: N`), preserving v0.1.9 behaviour.
  - `nil` or empty `ThinkingConfig` on an adaptive-capable model maps to adaptive mode with Anthropic's default effort (high) — matches the "thinking on by default" behaviour Gemini Pro / Flash give for the equivalent tier. On manual-only Anthropic models it stays off, matching Gemini's Flash-Lite tier default.
  - `ThinkingLevel: Minimal` maps to "off" — Anthropic has no minimal tier.
  - Explicit `ThinkingBudget` always wins (manual mode with that exact budget).
  - Adaptive-capable models matched against the SDK's canonical unversioned constants — bump the SDK and add the constant when Anthropic ships a new adaptive-capable model or dated variant.
- `anthropic.go` now sets `params.OutputConfig.Effort` from the mapping when adaptive mode is selected.
- `ThinkingConfigToAnthropicThinking(cfg)` retained as a deprecated thin wrapper for the previous single-arg shape — preserves the manual-budget behaviour for callers that don't have a model handy.

## [v0.1.15] - Upgrade to ADK Go v1.0.0

- Upgrade `google.golang.org/adk` from v0.6.0 to v1.0.0
- Bump minimum Go version from 1.24.4 to 1.25.0 to match ADK v1.0.0's toolchain requirement
- Pulls in transitive bumps: `google.golang.org/grpc` v1.78.0 → v1.79.3
- No source changes required — this adapter only consumes `google.golang.org/adk/model`, which is API-compatible between v0.6.0 and v1.0.0

## [v0.1.14] - Configurable prompt caching

- **Configurable prompt caching** via `Config.PromptCaching`. Four independently optional breakpoints:
  - `Auto` — top-level `cache_control` for automatic breakpoint placement
  - `SystemInstruction` — breakpoint on the last system text block
  - `Tools` — breakpoint on the last tool definition
  - `ConversationHistory` — breakpoint on the penultimate message
  
  Each breakpoint has its own TTL (5m default or 1h). Caching is off by default.
- **Cache usage token mapping**: `CacheReadInputTokens` from Anthropic responses is now mapped to genai's `CachedContentTokenCount` for OTEL tracing
- Upgrade `anthropic-sdk-go` from v1.24.0 to v1.28.0 for top-level `CacheControl` support

## [v0.1.13] - Raise default max output tokens, custom BaseURL, dependency upgrades

- Raise default `MaxTokens` from 4096 to 16384 — the previous default was too low for agents producing large structured JSON outputs (e.g. meeting summaries), causing silent mid-JSON truncation. Callers can still override via `GenerateContentConfig.MaxOutputTokens`
- Add `Config.BaseURL` field to allow custom API endpoint addresses for proxy services or private Anthropic API deployments
- Upgrade `google.golang.org/adk` from v0.2.0 to v0.6.0
- Upgrade `google.golang.org/genai` from v1.36.0 to v1.40.0
- Upgrade `github.com/google/jsonschema-go` from v0.3.0 to v0.4.2

## [v0.1.11] - Restore Vertex AI prompt-based JSON fallback

- Restore prompt-based JSON fallback for Vertex AI — Vertex AI does not yet support `OutputConfig`, causing 400 "output_config: Extra inputs are not permitted" errors
- Skip `OutputConfig` in `convertRequest` when variant is Vertex AI
- Handle both streaming and non-streaming paths with prompt-based JSON + markdown fence stripping

## [v0.1.10] - Upgrade anthropic-sdk-go to v1.24.0

- Upgrade `anthropic-sdk-go` to v1.24.0 — structured outputs moved to GA, removing the need for the Beta API
- Remove Beta message, content, tool, and response converters
- Remove prompt-based JSON fallback for Vertex AI (no longer needed with GA structured outputs)
- Use the GA Messages API with `OutputConfig.Format` for structured outputs

## [v0.1.9] - Map genai.ThinkingConfig to Anthropic extended thinking

- Map `genai.ThinkingConfig` fields (`ThinkingLevel`, `ThinkingBudget`, `IncludeThoughts`) to Anthropic's `Thinking` parameter in both standard and Beta API requests
- `ThinkingLevel: HIGH` enables thinking with a 10,000 token budget; `LOW` uses the minimum 1,024 tokens
- An explicit `ThinkingBudget` always takes precedence over level defaults
- `IncludeThoughts: true` without a level or budget enables thinking with a 10,000 token default

## [v0.1.8] - Rename VertexRegion to VertexLocation and rename env var

### Breaking Changes

- **Renamed `VertexRegion` to `VertexLocation`** in `Config` struct to align with
  Google Cloud's official terminology. The Vertex AI SDK and documentation consistently
  use "location" rather than "region".
- **Renamed environment variable `GOOGLE_CLOUD_REGION` to `GOOGLE_CLOUD_LOCATION`**.
  Update your environment configuration accordingly.

### Migration

Replace in your code:

```go
// Before
&adkanthropic.Config{
    VertexRegion: os.Getenv("GOOGLE_CLOUD_REGION"),
}

// After
&adkanthropic.Config{
    VertexLocation: os.Getenv("GOOGLE_CLOUD_LOCATION"),
}
```

## [v0.1.7] - ToolConfig support for tool_choice

- Add `ToolConfig` support for the `tool_choice` parameter — maps `genai.FunctionCallingConfig` modes to Anthropic's auto/any/tool choices (#9)

## [v0.1.6] - Improve markdown fence stripping

- Add permissive fallback for extracting JSON from within markdown fences, handling preamble text and trailing content (#8)

## [v0.1.5] - Strip markdown fences from prompt-based JSON

- Strip markdown code fences from prompt-based JSON responses when models wrap output in ```json blocks (#7)

## [v0.1.4] - Prompt-based JSON fallback for Vertex AI

- Fall back to prompt-based JSON generation on Vertex AI, which doesn't support the `anthropic-beta` structured outputs header (#6)

## [v0.1.3] - Structured output support

- Add structured output support via `ResponseSchema` using the Beta API with `structured-outputs-2025-11-13` (#4)
- Add Beta message, content, tool, and response converters

## [v0.1.2] - Add adapter code

- Add full adapter implementation (#3)

## [v0.1.1] - Fix nil tool input

- Ensure `tool_use` input is always a valid dictionary (#2)
- Add auto-release tagging CI workflow

## [v0.1.0] - Initial release

- Initial adapter code with streaming, tool calling, extended thinking, multimodal inputs, PDF support, and Vertex AI backend

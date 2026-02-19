# Changelog

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

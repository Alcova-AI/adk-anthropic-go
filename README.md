# ADK Anthropic Go

Anthropic Messages API support for Google's [Agent Development Kit](https://github.com/google/adk-go).

## Installation

```bash
go get github.com/Alcova-AI/adk-anthropic-go/v3
```

## Migrating from v2

v3 is source-incompatible with v2. It still implements ADK's `model.LLM`
interface, but model construction and route behaviour are now explicit.

### Update the module path

```go
// v2
import adkanthropic "github.com/Alcova-AI/adk-anthropic-go/v2"

// v3
import adkanthropic "github.com/Alcova-AI/adk-anthropic-go/v3"
```

### Construct the Anthropic client

v2 accepted credentials and endpoint selection through the adapter config:

```go
model, err := adkanthropic.NewModel(ctx, anthropic.ModelClaudeSonnet4_6, &adkanthropic.Config{
	APIKey: os.Getenv("ANTHROPIC_API_KEY"),
})
```

In v3, the caller constructs and owns the Anthropic SDK client:

```go
client := anthropic.NewClient(option.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))

model, err := adkanthropic.NewModel(adkanthropic.Config{
	Client:         client,
	CanonicalModel: anthropic.ModelClaudeSonnet4_6,
})
```

For Vertex AI, replace the v2 variant, project, and location fields with a
client constructed using `vertex.WithGoogleAuth`, as shown in
[Anthropic on Vertex AI](#anthropic-on-vertex-ai).

### Move route behaviour to options

| v2 | v3 |
|---|---|
| `Config.DefaultMaxTokens` | `WithDefaultMaxTokens` |
| `Config.PromptCaching` | `WithPromptCaching` with an explicit cache mode |
| Model-name reasoning inference | `WithReasoning` with an explicit strategy |
| `WithGatewayEffortTranslation` | `ReasoningProviderNative` with a provider projector |
| `Config.APIKey` and `Config.BaseURL` | Anthropic SDK client options |
| `Config.Variant` and Vertex fields | `vertex.WithGoogleAuth` |
| `ThinkingBudget` | Unsupported; use `ThinkingLevel` |

v3 does not provide the v2 manual token-budget fallback for older Claude
models. Use adaptive-effort reasoning with supported Claude models, use a
provider-native projector for supported gateway models, or disable reasoning.

### Use Vercel AI Gateway with ADK

v3 can use Vercel AI Gateway as the endpoint behind ADK's `model.LLM`
interface. It sends Anthropic Messages-compatible requests to Vercel, so ADK
agents can use Claude and non-Claude models that Vercel exposes through that
endpoint.

Typed provider-native reasoning projectors are included for OpenAI, Google,
and Z.AI models. Vercel routes require zero-data-retention by default and can
return typed metadata for the resolved provider and actual request cost. See
[Vercel AI Gateway](#vercel-ai-gateway) for the complete configuration.

## Features

- Streaming and non-streaming responses
- Tool calling and structured output
- Text, image, and PDF input
- Signed thinking continuity
- Explicit adaptive-effort, provider-native, and disabled reasoning strategies
- Manual, gateway-automatic, and provider-default prompt caching
- Caller-owned Anthropic SDK clients for direct, Vertex AI, and compatible endpoints
- Typed Vercel routing, fail-closed ZDR policy, provider options, cost, and routing metadata

## Direct Anthropic API

The caller constructs and owns the SDK client. Authentication and endpoint selection therefore stay outside the adapter.

```go
client := anthropic.NewClient(option.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))

model, err := adkanthropic.NewModel(
	adkanthropic.Config{
		Client:         client,
		CanonicalModel: anthropic.ModelClaudeSonnet4_6,
	},
	adkanthropic.WithDefaultMaxTokens(16_384),
	adkanthropic.WithReasoning(adkanthropic.ReasoningConfig{
		Strategy:     adkanthropic.ReasoningAdaptiveEffort,
		DefaultLevel: genai.ThinkingLevelMedium,
	}),
)
```

## Anthropic on Vertex AI

```go
client := anthropic.NewClient(
	vertex.WithGoogleAuth(ctx, location, projectID),
)

model, err := adkanthropic.NewModel(adkanthropic.Config{
	Client:         client,
	CanonicalModel: anthropic.ModelClaudeSonnet4_6,
	RequestModel:   "claude-sonnet-4-6@20260217",
})
```

## Vercel AI Gateway

```go
client := anthropic.NewClient(
	option.WithAPIKey(os.Getenv("AI_GATEWAY_API_KEY")),
	option.WithBaseURL("https://ai-gateway.vercel.sh"),
)

model, err := adkanthropic.NewModel(
	adkanthropic.Config{
		Client:         client,
		CanonicalModel: "glm-5.3-flash",
		RequestModel:   "zai/glm-5.3-flash",
	},
	adkanthropic.WithReasoning(adkanthropic.ReasoningConfig{
		Strategy:     adkanthropic.ReasoningProviderNative,
		DefaultLevel: genai.ThinkingLevelMedium,
	}),
	adkanthropic.WithVercelGateway(vercel.Config{
		Routing: vercel.Routing{
			Only: []string{"zai", "baseten"},
			Sort: vercel.SortTTFT,
		},
		// ZDRRequired is the zero-value default.
		DataPolicy: vercel.ZDRRequired,
		Projector:  vercel.ZAIModelOptions{},
	}),
)
```

Set `DataPolicy: vercel.RetentionAllowed` only for requests that may use providers without zero-data-retention support. This sends `zeroDataRetention: false` explicitly.

Vercel routing metadata is available on the final ADK response:

```go
metadata, ok := vercel.MetadataFromResponse(response)
if ok {
	fmt.Println(metadata.ResolvedProvider, metadata.CostUSD)
}
```

## Reasoning

The route selects one strategy. The adapter does not infer a strategy from the model name.

| Strategy | Wire behaviour |
|---|---|
| `ReasoningDisabled` | Omits thinking and effort |
| `ReasoningAdaptiveEffort` | Uses `thinking.type: adaptive` and maps low, medium, and high to `output_config.effort` |
| `ReasoningProviderNative` | Emits no Anthropic reasoning fields and projects the resolved level into typed Vercel model options |

For Claude adaptive reasoning, `ThinkingLevel: MINIMAL` disables thinking for that request. Provider-native projectors define their model family's supported level mapping.

The v3 adapter accepts level-based reasoning only. A request with `ThinkingBudget` returns an error because the route strategy is the single source of wire behaviour.

## Prompt caching

Use `WithPromptCaching` to select one cache policy:

- `PromptCacheProviderDefault` sends no explicit cache controls.
- `PromptCacheGatewayAutomatic` lets Vercel AI Gateway manage breakpoints and requires `WithVercelGateway`.
- `PromptCacheManual` applies the configured Anthropic breakpoints.

Manual example:

```go
oneHour := &adkanthropic.CacheBreakpoint{
	TTL: anthropic.CacheControlEphemeralTTLTTL1h,
}

option := adkanthropic.WithPromptCaching(adkanthropic.PromptCachingConfig{
	Mode:              adkanthropic.PromptCacheManual,
	Tools:             oneHour,
	SystemInstruction: oneHour,
})
```

## Provider-native Vercel options

Provider-native settings can be supplied by namespace:

```go
vercel.Config{
	ProviderOptions: map[string]map[string]any{
		"anthropic": {
			"sendReasoning": false,
		},
	},
}
```

The adapter rejects the `gateway` namespace and provider options that can override adapter-owned reasoning, cache, or data-policy fields.

## License

Apache License 2.0

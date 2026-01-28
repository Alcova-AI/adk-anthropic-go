# ADK Anthropic Go

Anthropic Claude model support for Google's [Agent Development Kit (ADK)](https://github.com/google/adk-go).

This package implements the `model.LLM` interface for Anthropic Claude models, allowing you to use Claude with ADK agents.

## Installation

```bash
go get github.com/Alcova-AI/adk-anthropic-go
```

## Features

- Streaming and non-streaming responses
- Tool/function calling with `ToolConfig` support (tool_choice: auto, any, specific tool)
- Structured output via `ResponseSchema` (guaranteed schema-compliant JSON)
- Extended thinking (mapped to `genai.Part` with `Thought=true`)
- Multimodal inputs (text, images)
- PDF document processing (beta)
- System instructions
- Both direct Anthropic API and Vertex AI backends

## Supported Models

- `claude-sonnet-4-5` / `claude-sonnet-4-5-20250929` (Claude Sonnet 4.5)
- `claude-opus-4-5` / `claude-opus-4-5-20251101` (Claude Opus 4.5)
- `claude-sonnet-4-0` / `claude-sonnet-4-20250514` (Claude Sonnet 4)
- `claude-opus-4-0` / `claude-opus-4-20250514` (Claude Opus 4)
- `claude-opus-4-1-20250805` (Claude Opus 4.1)
- `claude-haiku-4-5` / `claude-haiku-4-5-20251001` (Claude Haiku 4.5)
- `claude-3-5-haiku-latest` / `claude-3-5-haiku-20241022` (Claude 3.5 Haiku)

## Usage

### Direct Anthropic API

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	adkanthropic "github.com/Alcova-AI/adk-anthropic-go"
	"google.golang.org/adk/agent/llmagent"
)

func main() {
	ctx := context.Background()

	model, err := adkanthropic.NewModel(ctx, anthropic.ModelClaudeSonnet4_20250514, &adkanthropic.Config{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	})
	if err != nil {
		log.Fatal(err)
	}

	agent, err := llmagent.New(llmagent.Config{
		Name:        "my_agent",
		Model:       model,
		Description: "A helpful assistant powered by Claude",
		Instruction: "You are a helpful assistant.",
	})
	if err != nil {
		log.Fatal(err)
	}

	// Use the agent...
	_ = agent
}
```

### Vertex AI

```go
model, err := adkanthropic.NewModel(ctx, "claude-sonnet-4@20250514", &adkanthropic.Config{
	Variant:         adkanthropic.VariantVertexAI,
	VertexProjectID: os.Getenv("GOOGLE_CLOUD_PROJECT"),
	VertexRegion:    os.Getenv("GOOGLE_CLOUD_REGION"),
})
```

Or set `ANTHROPIC_USE_VERTEX=1` to use Vertex AI without specifying the variant in code.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | API key for direct Anthropic API access |
| `ANTHROPIC_USE_VERTEX` | Set to `1` or `true` to use Vertex AI backend |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID for Vertex AI |
| `GOOGLE_CLOUD_REGION` | GCP region for Vertex AI (e.g., `us-central1`) |

### Configuration Options

```go
type Config struct {
	// APIKey for direct Anthropic API access
	APIKey string

	// Vertex AI configuration
	VertexProjectID string
	VertexRegion    string

	// Backend variant: VariantAnthropicAPI or VariantVertexAI
	Variant string

	// Default max tokens (default: 4096)
	DefaultMaxTokens int
}
```

### Structured Output

Set `ResponseSchema` on the request config to get guaranteed schema-compliant JSON responses. On the direct Anthropic API, this uses the Beta API with native structured outputs. On Vertex AI, it falls back to prompt-based JSON generation with automatic markdown fence stripping.

```go
config := &genai.GenerateContentConfig{
	ResponseSchema: &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"name", "age"},
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString},
			"age":  {Type: genai.TypeInteger},
		},
	},
}
```

### Tool Choice (ToolConfig)

Use `ToolConfig` to control how the model selects tools:

```go
config := &genai.GenerateContentConfig{
	Tools: tools,
	ToolConfig: &genai.ToolConfig{
		FunctionCallingConfig: &genai.FunctionCallingConfig{
			Mode: genai.FunctionCallingConfigModeAny, // must use a tool
		},
	},
}
```

Mode mapping:
| `genai` Mode | Anthropic `tool_choice` |
|---|---|
| `ModeAuto` | `auto` (model decides) |
| `ModeAny` | `any` (must use a tool) |
| `ModeAny` + single `AllowedFunctionNames` | `tool` (must use the named tool) |
| `ModeNone` | omitted (no tool use) |

## License

Apache License 2.0

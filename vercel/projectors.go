// Copyright 2025 Alcova AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package vercel

import (
	"fmt"

	"google.golang.org/genai"
)

// OpenAIModelOptions projects genai reasoning levels to OpenAI model options.
type OpenAIModelOptions struct{}

func (OpenAIModelOptions) ProjectProviderOptions(input ProviderOptionsInput) (map[string]map[string]any, error) {
	effort, err := directReasoningEffort(input.ThinkingLevel)
	if err != nil {
		return nil, err
	}
	options := map[string]any{"store": false}
	if effort != "" {
		options["reasoningEffort"] = effort
	}
	if input.IncludeThoughts {
		options["reasoningSummary"] = "auto"
	}
	return map[string]map[string]any{"openai": options}, nil
}

// GoogleModelOptions projects genai reasoning levels to Gemini 3 thinking
// configuration.
type GoogleModelOptions struct{}

func (GoogleModelOptions) ProjectProviderOptions(input ProviderOptionsInput) (map[string]map[string]any, error) {
	level, err := directReasoningEffort(input.ThinkingLevel)
	if err != nil {
		return nil, err
	}
	thinking := make(map[string]any, 2)
	if level != "" {
		thinking["thinkingLevel"] = level
	}
	if input.IncludeThoughts {
		thinking["includeThoughts"] = true
	}
	return map[string]map[string]any{
		"google": {"thinkingConfig": thinking},
	}, nil
}

// ZAIModelOptions projects genai reasoning levels to GLM 5.3's supported
// low, high, and max effort levels. Minimal and low map to low, medium maps
// to high, and high maps to max. GLM thinking remains enabled.
type ZAIModelOptions struct{}

func (ZAIModelOptions) ProjectProviderOptions(input ProviderOptionsInput) (map[string]map[string]any, error) {
	effort, err := zaiReasoningEffort(input.ThinkingLevel)
	if err != nil {
		return nil, err
	}
	options := map[string]any{
		"thinking": map[string]any{
			"type": "enabled",
		},
	}
	if effort != "" {
		options["reasoningEffort"] = effort
	}
	return map[string]map[string]any{"zai": options}, nil
}

func directReasoningEffort(level genai.ThinkingLevel) (string, error) {
	switch level {
	case "", genai.ThinkingLevelUnspecified:
		return "", nil
	case genai.ThinkingLevelMinimal:
		return "minimal", nil
	case genai.ThinkingLevelLow:
		return "low", nil
	case genai.ThinkingLevelMedium:
		return "medium", nil
	case genai.ThinkingLevelHigh:
		return "high", nil
	default:
		return "", fmt.Errorf("unsupported thinking level %q", level)
	}
}

func zaiReasoningEffort(level genai.ThinkingLevel) (string, error) {
	switch level {
	case "", genai.ThinkingLevelUnspecified:
		return "", nil
	case genai.ThinkingLevelMinimal, genai.ThinkingLevelLow:
		return "low", nil
	case genai.ThinkingLevelMedium:
		return "high", nil
	case genai.ThinkingLevelHigh:
		return "max", nil
	default:
		return "", fmt.Errorf("unsupported thinking level %q", level)
	}
}

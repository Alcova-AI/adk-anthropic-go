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

package adkanthropic

import (
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

// ReasoningStrategy controls how genai thinking levels are represented on the
// Anthropic Messages wire format.
type ReasoningStrategy uint8

const (
	// ReasoningDisabled omits thinking and effort fields.
	ReasoningDisabled ReasoningStrategy = iota
	// ReasoningAdaptiveEffort uses Claude adaptive thinking with output effort.
	ReasoningAdaptiveEffort
	// ReasoningProviderNative projects the resolved genai thinking level into
	// provider-specific Vercel model options. It emits no Anthropic reasoning
	// fields.
	ReasoningProviderNative
)

// ReasoningConfig configures route-level reasoning. DefaultLevel is used only
// when a request omits ThinkingConfig or leaves ThinkingLevel unspecified.
type ReasoningConfig struct {
	Strategy     ReasoningStrategy
	DefaultLevel genai.ThinkingLevel
}

func (c ReasoningConfig) validate() error {
	if c.Strategy > ReasoningProviderNative {
		return fmt.Errorf("unsupported reasoning strategy %d", c.Strategy)
	}
	if !validThinkingLevel(c.DefaultLevel) {
		return fmt.Errorf("unsupported default thinking level %q", c.DefaultLevel)
	}
	return nil
}

type resolvedReasoning struct {
	ThinkingLevel   genai.ThinkingLevel
	IncludeThoughts bool
}

func (c ReasoningConfig) resolve(cfg *genai.ThinkingConfig) (resolvedReasoning, error) {
	if cfg != nil && cfg.ThinkingBudget != nil {
		return resolvedReasoning{}, fmt.Errorf("ThinkingBudget is not supported; use ThinkingLevel with the route reasoning strategy")
	}

	level := c.DefaultLevel
	includeThoughts := false
	if cfg != nil {
		includeThoughts = cfg.IncludeThoughts
		if cfg.ThinkingLevel != "" && cfg.ThinkingLevel != genai.ThinkingLevelUnspecified {
			level = cfg.ThinkingLevel
		}
	}
	if !validThinkingLevel(level) {
		return resolvedReasoning{}, fmt.Errorf("unsupported thinking level %q", level)
	}
	return resolvedReasoning{ThinkingLevel: level, IncludeThoughts: includeThoughts}, nil
}

func (c ReasoningConfig) mapThinking(cfg *genai.ThinkingConfig) (thinkingMapping, error) {
	resolved, err := c.resolve(cfg)
	if err != nil {
		return thinkingMapping{}, err
	}

	switch c.Strategy {
	case ReasoningDisabled:
		return thinkingMapping{}, nil
	case ReasoningAdaptiveEffort:
		return adaptiveReasoning(resolved.ThinkingLevel, resolved.IncludeThoughts), nil
	case ReasoningProviderNative:
		return thinkingMapping{}, nil
	default:
		return thinkingMapping{}, fmt.Errorf("unsupported reasoning strategy %d", c.Strategy)
	}
}

type thinkingMapping struct {
	Thinking anthropic.ThinkingConfigParamUnion
	Effort   anthropic.OutputConfigEffort
}

func adaptiveReasoning(level genai.ThinkingLevel, includeThoughts bool) thinkingMapping {
	if level == genai.ThinkingLevelMinimal {
		return thinkingMapping{}
	}
	display := anthropic.ThinkingConfigAdaptiveDisplayOmitted
	if includeThoughts {
		display = anthropic.ThinkingConfigAdaptiveDisplaySummarized
	}
	mapping := thinkingMapping{
		Thinking: anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{Display: display},
		},
	}
	switch level {
	case genai.ThinkingLevelLow:
		mapping.Effort = anthropic.OutputConfigEffortLow
	case genai.ThinkingLevelMedium:
		mapping.Effort = anthropic.OutputConfigEffortMedium
	case genai.ThinkingLevelHigh:
		mapping.Effort = anthropic.OutputConfigEffortHigh
	}
	return mapping
}

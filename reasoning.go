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
	// ReasoningTokenBudget uses Anthropic enabled thinking with a token budget.
	// Gateways can translate that native Messages shape to non-Claude models.
	ReasoningTokenBudget
)

// ReasoningBudgets maps genai levels to Anthropic thinking token budgets.
type ReasoningBudgets struct {
	Low    int64
	Medium int64
	High   int64
}

// DefaultReasoningBudgets returns the adapter's standard level mapping.
func DefaultReasoningBudgets() ReasoningBudgets {
	return ReasoningBudgets{Low: 1024, Medium: 5000, High: 10000}
}

// ReasoningConfig configures route-level reasoning. DefaultLevel is used only
// when a request omits ThinkingConfig or leaves ThinkingLevel unspecified.
type ReasoningConfig struct {
	Strategy     ReasoningStrategy
	DefaultLevel genai.ThinkingLevel
	Budgets      ReasoningBudgets
}

func (c ReasoningConfig) validate() error {
	if c.Strategy > ReasoningTokenBudget {
		return fmt.Errorf("unsupported reasoning strategy %d", c.Strategy)
	}
	if !validThinkingLevel(c.DefaultLevel) {
		return fmt.Errorf("unsupported default thinking level %q", c.DefaultLevel)
	}
	if c.Strategy != ReasoningTokenBudget {
		return nil
	}
	budgets := c.withDefaults().Budgets
	if budgets.Low <= 0 || budgets.Medium <= 0 || budgets.High <= 0 {
		return fmt.Errorf("reasoning token budgets must be positive")
	}
	if budgets.Low > budgets.Medium || budgets.Medium > budgets.High {
		return fmt.Errorf("reasoning token budgets must be ordered low <= medium <= high")
	}
	return nil
}

func (c ReasoningConfig) withDefaults() ReasoningConfig {
	if c.Strategy == ReasoningTokenBudget && c.Budgets == (ReasoningBudgets{}) {
		c.Budgets = DefaultReasoningBudgets()
	}
	return c
}

func (c ReasoningConfig) mapThinking(cfg *genai.ThinkingConfig) (thinkingMapping, error) {
	if cfg != nil && cfg.ThinkingBudget != nil {
		return thinkingMapping{}, fmt.Errorf("ThinkingBudget is not supported; use ThinkingLevel with the route reasoning strategy")
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
		return thinkingMapping{}, fmt.Errorf("unsupported thinking level %q", level)
	}

	switch c.Strategy {
	case ReasoningDisabled:
		return thinkingMapping{}, nil
	case ReasoningAdaptiveEffort:
		return adaptiveReasoning(level, includeThoughts), nil
	case ReasoningTokenBudget:
		return budgetReasoning(level, includeThoughts, c.withDefaults().Budgets), nil
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

func budgetReasoning(level genai.ThinkingLevel, includeThoughts bool, budgets ReasoningBudgets) thinkingMapping {
	var budget int64
	switch level {
	case genai.ThinkingLevelLow:
		budget = budgets.Low
	case genai.ThinkingLevelMedium:
		budget = budgets.Medium
	case genai.ThinkingLevelHigh:
		budget = budgets.High
	}
	if budget == 0 {
		return thinkingMapping{}
	}
	display := anthropic.ThinkingConfigEnabledDisplayOmitted
	if includeThoughts {
		display = anthropic.ThinkingConfigEnabledDisplaySummarized
	}
	return thinkingMapping{
		Thinking: anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{
				BudgetTokens: budget,
				Display:      display,
			},
		},
	}
}

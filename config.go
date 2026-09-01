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

	"github.com/Alcova-AI/adk-anthropic-go/v3/vercel"
)

// Config contains the transport client and model identities used by the
// adapter. The caller owns authentication, endpoint selection, and the HTTP
// client by constructing Client with the Anthropic SDK.
type Config struct {
	Client anthropic.Client

	// CanonicalModel is exposed through Name and identifies model capabilities.
	CanonicalModel anthropic.Model

	// RequestModel is sent on the wire. Leave it empty to use CanonicalModel.
	// This is useful for gateways that require provider-qualified model IDs.
	RequestModel anthropic.Model
}

// Option configures model-route behaviour.
type Option func(*modelOptions) error

type modelOptions struct {
	defaultMaxTokens int
	reasoning        ReasoningConfig
	promptCaching    PromptCachingConfig
	vercel           *vercel.Config
}

// WithDefaultMaxTokens sets the route default for max_tokens. A request-level
// GenerateContentConfig.MaxOutputTokens value still takes precedence.
func WithDefaultMaxTokens(tokens int) Option {
	return func(opts *modelOptions) error {
		if tokens <= 0 {
			return fmt.Errorf("default max tokens must be positive")
		}
		opts.defaultMaxTokens = tokens
		return nil
	}
}

// WithReasoning selects the explicit wire strategy for genai thinking levels.
// The adapter never infers this strategy from a model name or endpoint.
func WithReasoning(cfg ReasoningConfig) Option {
	return func(opts *modelOptions) error {
		if err := cfg.validate(); err != nil {
			return err
		}
		opts.reasoning = cfg
		return nil
	}
}

// WithPromptCaching selects how cache controls are applied for this route.
func WithPromptCaching(cfg PromptCachingConfig) Option {
	return func(opts *modelOptions) error {
		if err := cfg.validate(); err != nil {
			return err
		}
		opts.promptCaching = cfg
		return nil
	}
}

// WithVercelGateway adds typed Vercel routing and data-policy options. The zero
// value of vercel.Config requires zero-data-retention routing.
func WithVercelGateway(cfg vercel.Config) Option {
	return func(opts *modelOptions) error {
		if err := cfg.Validate(); err != nil {
			return err
		}
		cfg = cloneVercelConfig(cfg)
		opts.vercel = &cfg
		return nil
	}
}

func cloneVercelConfig(cfg vercel.Config) vercel.Config {
	cfg.Routing.Only = append([]string(nil), cfg.Routing.Only...)
	cfg.Routing.Order = append([]string(nil), cfg.Routing.Order...)
	if cfg.ProviderOptions == nil {
		return cfg
	}
	providerOptions := make(map[string]map[string]any, len(cfg.ProviderOptions))
	for namespace, values := range cfg.ProviderOptions {
		copyValues := make(map[string]any, len(values))
		for key, value := range values {
			copyValues[key] = value
		}
		providerOptions[namespace] = copyValues
	}
	cfg.ProviderOptions = providerOptions
	return cfg
}

func validThinkingLevel(level genai.ThinkingLevel) bool {
	switch level {
	case "",
		genai.ThinkingLevelUnspecified,
		genai.ThinkingLevelMinimal,
		genai.ThinkingLevelLow,
		genai.ThinkingLevelMedium,
		genai.ThinkingLevelHigh:
		return true
	default:
		return false
	}
}

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

// Package vercel contains the optional Vercel AI Gateway extension for the
// Anthropic Messages adapter.
package vercel

import (
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// DataPolicy controls whether a request may use providers without verified
// zero-data-retention support.
type DataPolicy uint8

const (
	// ZDRRequired is the fail-closed default.
	ZDRRequired DataPolicy = iota
	// RetentionAllowed permits routes that do not support zero data retention.
	RetentionAllowed
)

// Sort selects Vercel's provider ranking signal.
type Sort string

const (
	SortDefault Sort = ""
	SortCost    Sort = "cost"
	SortTTFT    Sort = "ttft"
	SortTPS     Sort = "tps"
)

// Routing controls provider selection for a Vercel model route.
type Routing struct {
	Only  []string
	Order []string
	Sort  Sort
}

// ProviderOptionsInput contains the resolved request values that a model
// family can project into its native Vercel provider-options namespace.
type ProviderOptionsInput struct {
	ThinkingLevel   genai.ThinkingLevel
	IncludeThoughts bool
	MaxOutputTokens int64
}

// ProviderOptionsProjector converts resolved request values into deterministic
// model-family options. It cannot change gateway routing or data policy.
type ProviderOptionsProjector interface {
	ProjectProviderOptions(ProviderOptionsInput) (map[string]map[string]any, error)
}

// Config configures the Vercel AI Gateway extension. ProviderOptions holds
// provider-native settings by namespace, for example "anthropic" or "zai".
// The "gateway" namespace is reserved for the typed fields above.
type Config struct {
	Routing         Routing
	DataPolicy      DataPolicy
	ProviderOptions map[string]map[string]any
	Projector       ProviderOptionsProjector
}

// Validate checks that the configuration has an unambiguous and safe wire
// representation.
func (c Config) Validate() error {
	if c.DataPolicy > RetentionAllowed {
		return fmt.Errorf("unsupported Vercel data policy %d", c.DataPolicy)
	}
	switch c.Routing.Sort {
	case SortDefault, SortCost, SortTTFT, SortTPS:
	default:
		return fmt.Errorf("unsupported Vercel routing sort %q", c.Routing.Sort)
	}
	if err := validateProviders("only", c.Routing.Only); err != nil {
		return err
	}
	if err := validateProviders("order", c.Routing.Order); err != nil {
		return err
	}
	for namespace, values := range c.ProviderOptions {
		if strings.TrimSpace(namespace) == "" {
			return fmt.Errorf("Vercel provider option namespace must not be empty")
		}
		if strings.EqualFold(namespace, "gateway") {
			return fmt.Errorf("Vercel provider option namespace %q is reserved", namespace)
		}
		for key := range values {
			if isReservedProviderKey(key) {
				return fmt.Errorf("Vercel provider option %s.%s conflicts with adapter-owned reasoning, caching, or data policy", namespace, key)
			}
		}
	}
	if c.Projector != nil {
		projected, err := c.Projector.ProjectProviderOptions(ProviderOptionsInput{
			ThinkingLevel:   genai.ThinkingLevelHigh,
			IncludeThoughts: true,
			MaxOutputTokens: 1,
		})
		if err != nil {
			return fmt.Errorf("validate Vercel provider-options projector: %w", err)
		}
		if err := validateProjectedProviderOptions(c.ProviderOptions, projected); err != nil {
			return err
		}
	}
	return nil
}

func validateProjectedProviderOptions(static, projected map[string]map[string]any) error {
	for namespace, values := range projected {
		if strings.TrimSpace(namespace) == "" {
			return fmt.Errorf("projected Vercel provider option namespace must not be empty")
		}
		if strings.EqualFold(namespace, "gateway") {
			return fmt.Errorf("projected Vercel provider option namespace %q is reserved", namespace)
		}
		for key := range values {
			if strings.TrimSpace(key) == "" {
				return fmt.Errorf("projected Vercel provider option %s has an empty key", namespace)
			}
			if _, conflict := static[namespace][key]; conflict {
				return fmt.Errorf("projected Vercel provider option %s.%s conflicts with a static option", namespace, key)
			}
		}
	}
	return nil
}

func validateProviders(field string, providers []string) error {
	seen := make(map[string]struct{}, len(providers))
	for _, provider := range providers {
		provider = strings.TrimSpace(provider)
		if provider == "" {
			return fmt.Errorf("Vercel routing %s contains an empty provider", field)
		}
		if _, ok := seen[provider]; ok {
			return fmt.Errorf("Vercel routing %s contains duplicate provider %q", field, provider)
		}
		seen[provider] = struct{}{}
	}
	return nil
}

func isReservedProviderKey(key string) bool {
	normalized := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(key, "_", ""), "-", ""))
	switch normalized {
	case "gateway", "thinking", "thinkingconfig", "reasoning", "reasoningeffort", "reasoningsummary", "outputconfig", "cachecontrol", "zerodataretention", "store":
		return true
	default:
		return false
	}
}

// WireProviderOptions returns the JSON object expected by Vercel's
// providerOptions request field.
func (c Config) WireProviderOptions(input ProviderOptionsInput) (map[string]any, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	providerOptions := make(map[string]any, len(c.ProviderOptions)+1)
	for namespace, values := range c.ProviderOptions {
		copyValues := make(map[string]any, len(values))
		for key, value := range values {
			copyValues[key] = value
		}
		providerOptions[namespace] = copyValues
	}
	if c.Projector != nil {
		projected, err := c.Projector.ProjectProviderOptions(input)
		if err != nil {
			return nil, fmt.Errorf("project Vercel provider options: %w", err)
		}
		if err := validateProjectedProviderOptions(c.ProviderOptions, projected); err != nil {
			return nil, err
		}
		for namespace, values := range projected {
			existing, ok := providerOptions[namespace].(map[string]any)
			if !ok {
				existing = make(map[string]any, len(values))
				providerOptions[namespace] = existing
			}
			for key, value := range values {
				existing[key] = value
			}
		}
	}
	gateway := map[string]any{
		"zeroDataRetention": c.DataPolicy == ZDRRequired,
	}
	if len(c.Routing.Only) > 0 {
		gateway["only"] = append([]string(nil), c.Routing.Only...)
	}
	if len(c.Routing.Order) > 0 {
		gateway["order"] = append([]string(nil), c.Routing.Order...)
	}
	if c.Routing.Sort != SortDefault {
		gateway["sort"] = string(c.Routing.Sort)
	}
	providerOptions["gateway"] = gateway
	return providerOptions, nil
}

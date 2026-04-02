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

import "github.com/anthropics/anthropic-sdk-go"

// CacheBreakpoint configures a single cache control breakpoint.
type CacheBreakpoint struct {
	// TTL controls the cache time-to-live for this breakpoint.
	// Leave empty for the server default (5 minutes), or set to
	// anthropic.CacheControlEphemeralTTLTTL1h for 1-hour caching.
	//
	// Cost: 5m writes cost 1.25x base input; 1h writes cost 2x base input.
	// All cache reads cost 0.1x base input regardless of TTL.
	TTL anthropic.CacheControlEphemeralTTL
}

// PromptCachingConfig enables Anthropic prompt caching when provided.
// Each field controls a specific cache breakpoint position. All are
// independently optional — set only the breakpoints you need.
//
// Anthropic evaluates cache prefixes in order: tools → system → messages.
// When mixing TTLs, longer TTLs must appear before shorter ones in this order.
// Maximum 4 explicit breakpoints per request (auto does not count).
//
// See: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
type PromptCachingConfig struct {
	Auto                *CacheBreakpoint
	SystemInstruction   *CacheBreakpoint
	Tools               *CacheBreakpoint
	ConversationHistory *CacheBreakpoint
}

// Config holds configuration for creating an Anthropic Claude model.
type Config struct {
	// APIKey is the Anthropic API key for direct API access.
	// If not provided, it will be read from the ANTHROPIC_API_KEY environment variable.
	// This is only used when Variant is VariantAnthropicAPI.
	APIKey string

	// VertexProjectID is the Google Cloud project ID for Vertex AI access.
	// If not provided, it will be read from the GOOGLE_CLOUD_PROJECT environment variable.
	// This is only used when Variant is VariantVertexAI.
	VertexProjectID string

	// VertexLocation is the Google Cloud location for Vertex AI access.
	// If not provided, it will be read from the GOOGLE_CLOUD_LOCATION environment variable.
	// Common locations include "us-central1", "us-east5", "europe-west1", and "global".
	// This is only used when Variant is VariantVertexAI.
	VertexLocation string

	// Variant determines which backend to use for API calls.
	// Valid values are VariantAnthropicAPI and VariantVertexAI.
	// If empty, the variant is determined from the ANTHROPIC_USE_VERTEX environment variable.
	Variant string

	// DefaultMaxTokens is the default maximum number of tokens to generate.
	// Anthropic requires max_tokens to be explicitly set for all requests.
	// If not provided, defaults to 16384.
	DefaultMaxTokens int

	BaseURL string

	// PromptCaching configures optional prompt caching breakpoints.
	// When nil (the default), no cache control is applied.
	PromptCaching *PromptCachingConfig
}

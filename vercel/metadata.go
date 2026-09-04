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
	"encoding/json"
	"strconv"

	"google.golang.org/adk/v2/model"
)

// MetadataKey is the LLMResponse.CustomMetadata key used for Vercel data.
const MetadataKey = "vercel.gateway"

// Metadata contains the stable, allowlisted gateway facts exposed to callers.
type Metadata struct {
	GenerationID         string   `json:"generation_id,omitempty"`
	ResolvedProvider     string   `json:"resolved_provider,omitempty"`
	OriginalModelID      string   `json:"original_model_id,omitempty"`
	CanonicalModel       string   `json:"canonical_model,omitempty"`
	ModelAttemptCount    int      `json:"model_attempt_count,omitempty"`
	ProviderAttemptCount int      `json:"provider_attempt_count,omitempty"`
	CostUSD              *float64 `json:"cost_usd,omitempty"`
}

// MetadataFromResponse reads typed Vercel metadata from an ADK response.
func MetadataFromResponse(resp *model.LLMResponse) (Metadata, bool) {
	if resp == nil || resp.CustomMetadata == nil {
		return Metadata{}, false
	}
	metadata, ok := resp.CustomMetadata[MetadataKey].(Metadata)
	return metadata, ok
}

// ParseMetadata extracts Vercel provider metadata from the raw Messages API
// response. Unknown gateway fields are intentionally discarded.
func ParseMetadata(rawJSON string) (Metadata, bool) {
	if rawJSON == "" {
		return Metadata{}, false
	}
	var envelope struct {
		ProviderMetadata      gatewayProviderMetadata `json:"providerMetadata"`
		ProviderMetadataSnake gatewayProviderMetadata `json:"provider_metadata"`
		Message               *struct {
			ProviderMetadata      gatewayProviderMetadata `json:"providerMetadata"`
			ProviderMetadataSnake gatewayProviderMetadata `json:"provider_metadata"`
		} `json:"message"`
	}
	if err := json.Unmarshal([]byte(rawJSON), &envelope); err != nil {
		return Metadata{}, false
	}
	gateway := envelope.ProviderMetadata.Gateway
	if gateway.empty() {
		gateway = envelope.ProviderMetadataSnake.Gateway
	}
	if gateway.empty() && envelope.Message != nil {
		gateway = envelope.Message.ProviderMetadata.Gateway
		if gateway.empty() {
			gateway = envelope.Message.ProviderMetadataSnake.Gateway
		}
	}
	if gateway.empty() {
		return Metadata{}, false
	}
	metadata := Metadata{
		GenerationID:         gateway.GenerationID,
		ResolvedProvider:     gateway.Routing.ResolvedProvider,
		OriginalModelID:      gateway.Routing.OriginalModelID,
		CanonicalModel:       gateway.Routing.CanonicalSlug,
		ModelAttemptCount:    gateway.Routing.ModelAttemptCount,
		ProviderAttemptCount: gateway.Routing.TotalProviderAttemptCount,
		CostUSD:              parseCost(gateway.Cost),
	}
	return metadata, true
}

type gatewayProviderMetadata struct {
	Gateway gatewayMetadata `json:"gateway"`
}

type gatewayMetadata struct {
	GenerationID string          `json:"generationId"`
	Cost         json.RawMessage `json:"cost"`
	Routing      struct {
		OriginalModelID           string `json:"originalModelId"`
		ResolvedProvider          string `json:"resolvedProvider"`
		CanonicalSlug             string `json:"canonicalSlug"`
		ModelAttemptCount         int    `json:"modelAttemptCount"`
		TotalProviderAttemptCount int    `json:"totalProviderAttemptCount"`
	} `json:"routing"`
}

func (m gatewayMetadata) empty() bool {
	return m.GenerationID == "" && len(m.Cost) == 0 &&
		m.Routing.OriginalModelID == "" && m.Routing.ResolvedProvider == "" &&
		m.Routing.CanonicalSlug == "" && m.Routing.ModelAttemptCount == 0 &&
		m.Routing.TotalProviderAttemptCount == 0
}

func parseCost(raw json.RawMessage) *float64 {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var number float64
	if err := json.Unmarshal(raw, &number); err == nil {
		return &number
	}
	var text string
	if err := json.Unmarshal(raw, &text); err != nil {
		return nil
	}
	number, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return nil
	}
	return &number
}

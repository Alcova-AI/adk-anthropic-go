// Copyright 2026 Alcova AI
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
	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/adk/v2/model"
)

// ResponseMetadataKey is the LLMResponse.CustomMetadata key used for typed
// Anthropic response facts that ADK does not expose directly.
const ResponseMetadataKey = "anthropic.response"

// ResponseMetadata contains stable response facts that callers can use for
// tracing and cost reporting without depending on private metadata keys.
type ResponseMetadata struct {
	MessageID                  string `json:"message_id,omitempty"`
	CacheCreationInputTokens   int64  `json:"cache_creation_input_tokens,omitempty"`
	CacheCreation5mInputTokens int64  `json:"cache_creation_5m_input_tokens,omitempty"`
	CacheCreation1hInputTokens int64  `json:"cache_creation_1h_input_tokens,omitempty"`
}

// ResponseMetadataFromResponse reads typed Anthropic metadata from an ADK
// response.
func ResponseMetadataFromResponse(resp *model.LLMResponse) (ResponseMetadata, bool) {
	if resp == nil || resp.CustomMetadata == nil {
		return ResponseMetadata{}, false
	}
	metadata, ok := resp.CustomMetadata[ResponseMetadataKey].(ResponseMetadata)
	return metadata, ok
}

func attachAnthropicResponseMetadata(resp *model.LLMResponse, msg *anthropic.Message) {
	if resp == nil || msg == nil {
		return
	}
	if resp.CustomMetadata == nil {
		resp.CustomMetadata = make(map[string]any)
	}
	resp.CustomMetadata[ResponseMetadataKey] = ResponseMetadata{
		MessageID:                  msg.ID,
		CacheCreationInputTokens:   msg.Usage.CacheCreationInputTokens,
		CacheCreation5mInputTokens: msg.Usage.CacheCreation.Ephemeral5mInputTokens,
		CacheCreation1hInputTokens: msg.Usage.CacheCreation.Ephemeral1hInputTokens,
	}
}

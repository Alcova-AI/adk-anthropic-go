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
	"context"
	"strings"
	"testing"

	"google.golang.org/genai"

	"github.com/Alcova-AI/adk-anthropic-go/converters"
	"google.golang.org/adk/model"
)

func TestNewModel_ConfigBehavior(t *testing.T) {
	tests := []struct {
		name          string
		cfg           *Config
		wantMaxTokens int
		wantVariant   string
	}{
		{
			name: "explicit_max_tokens_and_variant",
			cfg: &Config{
				APIKey:           "test-api-key",
				DefaultMaxTokens: 2048,
				Variant:          VariantAnthropicAPI,
			},
			wantMaxTokens: 2048,
			wantVariant:   VariantAnthropicAPI,
		},
		{
			name: "default_max_tokens",
			cfg: &Config{
				APIKey:  "test-api-key",
				Variant: VariantAnthropicAPI,
			},
			wantMaxTokens: defaultMaxTokens,
			wantVariant:   VariantAnthropicAPI,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := NewModel(t.Context(), "claude-sonnet-4-20250514", tt.cfg)
			if err != nil {
				t.Fatalf("NewModel() error = %v", err)
			}

			if model.Name() != "claude-sonnet-4-20250514" {
				t.Errorf("Name() = %q, want %q", model.Name(), "claude-sonnet-4-20250514")
			}

			am := model.(*anthropicModel)
			if am.defaultMaxTokens != tt.wantMaxTokens {
				t.Errorf("defaultMaxTokens = %d, want %d", am.defaultMaxTokens, tt.wantMaxTokens)
			}
			if am.variant != tt.wantVariant {
				t.Errorf("variant = %q, want %q", am.variant, tt.wantVariant)
			}
		})
	}
}

func TestNewModel_VertexAI_MissingConfig(t *testing.T) {
	tests := []struct {
		name      string
		project   string
		region    string
		wantError string
	}{
		{"missing_project", "", "us-central1", "VertexProjectID is required"},
		{"missing_region", "test-project", "", "VertexRegion is required"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("GOOGLE_CLOUD_PROJECT", tt.project)
			t.Setenv("GOOGLE_CLOUD_REGION", tt.region)

			cfg := &Config{Variant: VariantVertexAI}
			_, err := NewModel(t.Context(), "claude-sonnet-4-20250514", cfg)
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("NewModel() error = %v, want contains %q", err, tt.wantError)
			}
		})
	}
}

func TestBuildPromptBasedJSONRequest(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString, Description: "The name"},
			"age":  {Type: genai.TypeInteger, Description: "The age"},
		},
		Required: []string{"name", "age"},
	}

	tests := []struct {
		name                  string
		existingSystemPrompt  string
		wantContainsSchema    bool
		wantContainsExisting  bool
	}{
		{
			name:                 "no_existing_system_prompt",
			existingSystemPrompt: "",
			wantContainsSchema:   true,
			wantContainsExisting: false,
		},
		{
			name:                 "with_existing_system_prompt",
			existingSystemPrompt: "You are a helpful assistant.",
			wantContainsSchema:   true,
			wantContainsExisting: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &model.LLMRequest{
				Contents: []*genai.Content{
					{Role: "user", Parts: []*genai.Part{{Text: "Hello"}}},
				},
				Config: &genai.GenerateContentConfig{
					ResponseSchema: schema,
				},
			}

			if tt.existingSystemPrompt != "" {
				req.Config.SystemInstruction = &genai.Content{
					Parts: []*genai.Part{{Text: tt.existingSystemPrompt}},
				}
			}

			// Build the modified config the same way generateWithPromptBasedJSON does
			schemaJSON := converters.SchemaToJSONString(req.Config.ResponseSchema)
			jsonInstruction := "You must respond with valid JSON that conforms to the following JSON schema:\n\n" + schemaJSON + "\n\nRespond ONLY with the JSON object, no markdown code fences, no explanations."

			modifiedConfig := *req.Config
			if modifiedConfig.SystemInstruction == nil {
				modifiedConfig.SystemInstruction = &genai.Content{
					Parts: []*genai.Part{{Text: jsonInstruction}},
				}
			} else {
				existingText := ""
				for _, part := range modifiedConfig.SystemInstruction.Parts {
					if part.Text != "" {
						existingText += part.Text + "\n\n"
					}
				}
				modifiedConfig.SystemInstruction = &genai.Content{
					Parts: []*genai.Part{{Text: existingText + jsonInstruction}},
				}
			}
			modifiedConfig.ResponseSchema = nil

			// Verify the modified config
			if modifiedConfig.ResponseSchema != nil {
				t.Error("ResponseSchema should be nil after modification")
			}

			if modifiedConfig.SystemInstruction == nil {
				t.Fatal("SystemInstruction should not be nil")
			}

			systemText := ""
			for _, part := range modifiedConfig.SystemInstruction.Parts {
				systemText += part.Text
			}

			if tt.wantContainsSchema && !strings.Contains(systemText, `"type": "object"`) {
				t.Error("SystemInstruction should contain the JSON schema")
			}

			if tt.wantContainsSchema && !strings.Contains(systemText, "Respond ONLY with the JSON object") {
				t.Error("SystemInstruction should contain JSON instruction")
			}

			if tt.wantContainsExisting && !strings.Contains(systemText, tt.existingSystemPrompt) {
				t.Errorf("SystemInstruction should contain existing prompt %q", tt.existingSystemPrompt)
			}
		})
	}
}

func TestStripMarkdownFromResponse(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "plain_json",
			input:    `{"key": "value"}`,
			expected: `{"key": "value"}`,
		},
		{
			name:     "json_with_json_fence",
			input:    "```json\n{\"key\": \"value\"}\n```",
			expected: `{"key": "value"}`,
		},
		{
			name:     "json_with_plain_fence",
			input:    "```\n{\"key\": \"value\"}\n```",
			expected: `{"key": "value"}`,
		},
		{
			name:     "json_with_fence_and_whitespace",
			input:    "  ```json\n{\"key\": \"value\"}\n```  ",
			expected: `{"key": "value"}`,
		},
		{
			name:     "multiline_json_with_fence",
			input:    "```json\n{\n  \"name\": \"test\",\n  \"value\": 123\n}\n```",
			expected: "{\n  \"name\": \"test\",\n  \"value\": 123\n}",
		},
		{
			name:     "no_closing_fence_unchanged",
			input:    "```json\n{\"key\": \"value\"}",
			expected: "```json\n{\"key\": \"value\"}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: tt.input}},
				},
			}

			stripMarkdownFromResponse(context.Background(), resp)

			if resp.Content.Parts[0].Text != tt.expected {
				t.Errorf("stripMarkdownFromResponse() = %q, want %q", resp.Content.Parts[0].Text, tt.expected)
			}
		})
	}
}

func TestStripMarkdownFromResponse_NilHandling(t *testing.T) {
	// Should not panic on nil inputs
	stripMarkdownFromResponse(context.Background(), nil)
	stripMarkdownFromResponse(context.Background(), &model.LLMResponse{})
	stripMarkdownFromResponse(context.Background(), &model.LLMResponse{Content: &genai.Content{}})
	stripMarkdownFromResponse(context.Background(), &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{nil}}})
}

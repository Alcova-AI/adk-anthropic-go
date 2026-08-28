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
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// testMaxTokens is an arbitrary non-zero max_tokens for constructing model
// fixtures in tests that don't exercise token-limit behaviour.
const testMaxTokens = 64000

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
			name: "default_max_tokens_is_safe_for_non_streaming",
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

func TestNewModel_DefaultMaxTokensSupportsNonStreaming(t *testing.T) {
	model, err := NewModel(t.Context(), anthropic.ModelClaudeHaiku4_5, &Config{
		APIKey:  "test-api-key",
		Variant: VariantAnthropicAPI,
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	maxTokens := model.(*anthropicModel).defaultMaxTokens
	if _, err := anthropic.CalculateNonStreamingTimeout(maxTokens, anthropic.ModelClaudeHaiku4_5, nil); err != nil {
		t.Fatalf("default max_tokens %d is incompatible with non-streaming requests: %v", maxTokens, err)
	}
}

func TestNewModel_VertexAI_MissingConfig(t *testing.T) {
	tests := []struct {
		name      string
		project   string
		location  string
		wantError string
	}{
		{"missing_project", "", "us-central1", "VertexProjectID is required"},
		{"missing_location", "test-project", "", "VertexLocation is required"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("GOOGLE_CLOUD_PROJECT", tt.project)
			t.Setenv("GOOGLE_CLOUD_LOCATION", tt.location)

			cfg := &Config{Variant: VariantVertexAI}
			_, err := NewModel(t.Context(), "claude-sonnet-4-20250514", cfg)
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("NewModel() error = %v, want contains %q", err, tt.wantError)
			}
		})
	}
}

func TestNewModel_RejectsUnknownVariant(t *testing.T) {
	_, err := NewModel(t.Context(), anthropic.ModelClaudeSonnet4_6, &Config{
		Variant: "UNKNOWN",
	})
	if err == nil || !strings.Contains(err.Error(), `unsupported Anthropic variant "UNKNOWN"`) {
		t.Fatalf("NewModel() error = %v, want unsupported variant error", err)
	}
}

func TestConvertRequest_RequestModelOverride(t *testing.T) {
	tests := []struct {
		name         string
		requestModel anthropic.Model
		wantModel    anthropic.Model
	}{
		{
			name:      "uses canonical model by default",
			wantModel: anthropic.ModelClaudeSonnet4_6,
		},
		{
			name:         "uses request model override",
			requestModel: "provider/claude-sonnet-4.6",
			wantModel:    "provider/claude-sonnet-4.6",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm, err := NewModel(t.Context(), anthropic.ModelClaudeSonnet4_6, &Config{
				APIKey:       "test-api-key",
				Variant:      VariantAnthropicAPI,
				RequestModel: tt.requestModel,
			})
			if err != nil {
				t.Fatalf("NewModel() error = %v", err)
			}
			if llm.Name() != string(anthropic.ModelClaudeSonnet4_6) {
				t.Errorf("Name() = %q, want canonical name %q", llm.Name(), anthropic.ModelClaudeSonnet4_6)
			}

			params, err := llm.(*anthropicModel).convertRequest(&model.LLMRequest{
				Contents: []*genai.Content{genai.NewContentFromText("Hello", "user")},
			})
			if err != nil {
				t.Fatalf("convertRequest() error = %v", err)
			}
			if params.Model != tt.wantModel {
				t.Errorf("request model = %q, want %q", params.Model, tt.wantModel)
			}
			if params.Thinking.OfAdaptive == nil {
				t.Error("expected canonical Sonnet name to retain adaptive thinking defaults")
			}
		})
	}
}

func TestNewModel_AnthropicCompatibleEndpointUsesConfiguredRequest(t *testing.T) {
	type capturedRequest struct {
		path   string
		apiKey string
		model  string
	}
	captured := make(chan capturedRequest, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		var requestBody struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(body, &requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		captured <- capturedRequest{
			path:   r.URL.Path,
			apiKey: r.Header.Get("x-api-key"),
			model:  requestBody.Model,
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"anthropic/claude-sonnet-4.6","content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := NewModel(t.Context(), anthropic.ModelClaudeSonnet4_6, &Config{
		APIKey:       "gateway-key",
		Variant:      VariantAnthropicAPI,
		BaseURL:      srv.URL,
		RequestModel: "provider/claude-sonnet-4.6",
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	for _, err := range llm.GenerateContent(t.Context(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Hello", "user")},
	}, false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
	}

	got := <-captured
	if got.path != "/v1/messages" {
		t.Errorf("request path = %q, want /v1/messages", got.path)
	}
	if got.apiKey != "gateway-key" {
		t.Errorf("x-api-key = %q, want gateway-key", got.apiKey)
	}
	if got.model != "provider/claude-sonnet-4.6" {
		t.Errorf("request model = %q, want %q", got.model, "provider/claude-sonnet-4.6")
	}
}

func TestFilterThoughtParts_PreservesProviderState(t *testing.T) {
	unsigned := &genai.Part{Thought: true, Text: "Vercel Gemini reasoning"}
	signed := &genai.Part{Thought: true, Text: "signed reasoning", ThoughtSignature: []byte("sig")}
	redacted := &genai.Part{
		Thought:      true,
		Text:         "[thinking redacted]",
		PartMetadata: map[string]any{"anthropic.redacted_thinking_data": "opaque"},
	}
	text := &genai.Part{Text: "answer"}

	got := filterThoughtParts([]*genai.Part{unsigned, signed, redacted, text})

	if len(got) != 3 {
		t.Fatalf("len(got) = %d, want 3", len(got))
	}
	if got[0] != signed || got[0].Text != "signed reasoning" {
		t.Errorf("got[0] = %+v, want signed thinking preserved unchanged", got[0])
	}
	if got[1] != redacted {
		t.Errorf("got[1] = %+v, want redacted provider state preserved", got[1])
	}
	if got[2] != text {
		t.Errorf("got[2] = %+v, want answer text", got[2])
	}
}

func TestFilterThoughtParts_PreservesThoughtOnlyTurn(t *testing.T) {
	got := filterThoughtParts([]*genai.Part{{
		Thought: true,
		Text:    "Vercel Gemini reasoning",
	}})

	if len(got) != 1 {
		t.Fatalf("len(got) = %d, want 1", len(got))
	}
	if !got[0].Thought || got[0].Text != "" {
		t.Errorf("got[0] = %+v, want empty thought marker", got[0])
	}
}

func TestGenerateContent_HonoursIncludeThoughts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"google/gemini-3.6-flash","content":[{"type":"thinking","thinking":"Vercel Gemini reasoning","signature":""},{"type":"text","text":"Hello"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := NewModel(t.Context(), anthropic.ModelClaudeSonnet4_6, &Config{
		APIKey:  "gateway-key",
		Variant: VariantAnthropicAPI,
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	for _, tc := range []struct {
		name            string
		includeThoughts bool
		wantParts       int
	}{
		{name: "hidden", includeThoughts: false, wantParts: 1},
		{name: "included", includeThoughts: true, wantParts: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := &model.LLMRequest{Config: &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: tc.includeThoughts},
			}}

			var got *model.LLMResponse
			for resp, err := range llm.GenerateContent(t.Context(), req, false) {
				if err != nil {
					t.Fatalf("GenerateContent() error = %v", err)
				}
				got = resp
			}

			if got == nil || got.Content == nil {
				t.Fatal("GenerateContent() returned no content")
			}
			if len(got.Content.Parts) != tc.wantParts {
				t.Fatalf("len(got.Content.Parts) = %d, want %d", len(got.Content.Parts), tc.wantParts)
			}
			if got.Content.Parts[len(got.Content.Parts)-1].Text != "Hello" {
				t.Errorf("last part = %+v, want answer text", got.Content.Parts[len(got.Content.Parts)-1])
			}
		})
	}
}

func TestGenerateContent_PreservesHiddenThoughtOnlyTurn(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"google/gemini-3.7-flash","content":[{"type":"thinking","thinking":"Vercel Gemini reasoning","signature":""}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := NewModel(t.Context(), anthropic.ModelClaudeSonnet4_6, &Config{
		APIKey:  "gateway-key",
		Variant: VariantAnthropicAPI,
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	var got *model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), &model.LLMRequest{}, false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		got = resp
	}

	if got == nil || got.Content == nil || len(got.Content.Parts) != 1 {
		t.Fatalf("GenerateContent() = %+v, want one content part", got)
	}
	if part := got.Content.Parts[0]; !part.Thought || part.Text != "" {
		t.Errorf("part = %+v, want empty thought marker", part)
	}
}

func TestConvertRequest_VertexAI_SetsOutputConfig(t *testing.T) {
	m := &anthropicModel{
		canonicalModel:   "claude-haiku-4-5-20251001",
		variant:          VariantVertexAI,
		defaultMaxTokens: testMaxTokens,
	}

	schema := &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"name"},
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString},
		},
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", "user"),
		},
		Config: &genai.GenerateContentConfig{
			ResponseSchema: schema,
		},
	}

	params, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}

	// Structured outputs are GA on Vertex AI, so OutputConfig must be set.
	if params.OutputConfig.Format.Schema == nil {
		t.Error("expected OutputConfig to be set for Vertex AI, but it was empty")
	}
}

func TestConvertRequest_OutputConfig_EnforcesAdditionalPropertiesFalse(t *testing.T) {
	m := &anthropicModel{
		canonicalModel:   "claude-haiku-4-5-20251001",
		variant:          VariantVertexAI,
		defaultMaxTokens: testMaxTokens,
	}

	// Top-level object, a nested object property, and an array of objects —
	// every object node must get additionalProperties:false.
	schema := &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"person"},
		Properties: map[string]*genai.Schema{
			"person": {
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"name": {Type: genai.TypeString},
				},
			},
			"tags": {
				Type: genai.TypeArray,
				Items: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"label": {Type: genai.TypeString},
					},
				},
			},
		},
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Hello", "user")},
		Config:   &genai.GenerateContentConfig{ResponseSchema: schema},
	}

	params, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}

	root := params.OutputConfig.Format.Schema
	if root == nil {
		t.Fatal("expected OutputConfig schema to be set")
	}

	assertAdditionalPropertiesFalse(t, root, "root")

	props, _ := root["properties"].(map[string]any)
	person, _ := props["person"].(map[string]any)
	assertAdditionalPropertiesFalse(t, person, "person")

	tags, _ := props["tags"].(map[string]any)
	items, _ := tags["items"].(map[string]any)
	assertAdditionalPropertiesFalse(t, items, "tags.items")
}

func TestConvertRequest_VertexAI_TransformsUnsupportedSchemaConstraints(t *testing.T) {
	m := &anthropicModel{
		canonicalModel:   "claude-haiku-4-5-20251001",
		variant:          VariantVertexAI,
		defaultMaxTokens: testMaxTokens,
	}

	minimum, maximum := 1.0, 100.0
	minLength, maxLength := int64(2), int64(50)
	minItems, maxItems := int64(2), int64(20)
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"score": {
				Type:        genai.TypeNumber,
				Description: "A score",
				Minimum:     &minimum,
				Maximum:     &maximum,
			},
			"name": {
				Type:      genai.TypeString,
				MinLength: &minLength,
				MaxLength: &maxLength,
			},
			"tags": {
				Type:     genai.TypeArray,
				Items:    &genai.Schema{Type: genai.TypeString},
				MinItems: &minItems,
				MaxItems: &maxItems,
			},
		},
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Hello", "user")},
		Config:   &genai.GenerateContentConfig{ResponseSchema: schema},
	}

	params, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}

	root := params.OutputConfig.Format.Schema
	properties, ok := root["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties = %T, want map[string]any", root["properties"])
	}

	score, ok := properties["score"].(map[string]any)
	if !ok {
		t.Fatalf("score = %T, want map[string]any", properties["score"])
	}
	for _, unsupported := range []string{"minimum", "maximum"} {
		if _, exists := score[unsupported]; exists {
			t.Errorf("score.%s should be removed from the Anthropic schema", unsupported)
		}
	}
	if description, _ := score["description"].(string); !strings.Contains(description, "maximum: 100") || !strings.Contains(description, "minimum: 1") {
		t.Errorf("score.description = %q, want preserved minimum and maximum guidance", description)
	}

	name, ok := properties["name"].(map[string]any)
	if !ok {
		t.Fatalf("name = %T, want map[string]any", properties["name"])
	}
	for _, unsupported := range []string{"minLength", "maxLength"} {
		if _, exists := name[unsupported]; exists {
			t.Errorf("name.%s should be removed from the Anthropic schema", unsupported)
		}
	}
	if description, _ := name["description"].(string); !strings.Contains(description, "maxLength: 50") || !strings.Contains(description, "minLength: 2") {
		t.Errorf("name.description = %q, want preserved length guidance", description)
	}

	tags, ok := properties["tags"].(map[string]any)
	if !ok {
		t.Fatalf("tags = %T, want map[string]any", properties["tags"])
	}
	for _, unsupported := range []string{"minItems", "maxItems"} {
		if _, exists := tags[unsupported]; exists {
			t.Errorf("tags.%s should be removed from the Anthropic schema", unsupported)
		}
	}
	if description, _ := tags["description"].(string); !strings.Contains(description, "maxItems: 20") || !strings.Contains(description, "minItems: 2") {
		t.Errorf("tags.description = %q, want preserved item-count guidance", description)
	}
}

func assertAdditionalPropertiesFalse(t *testing.T, schema map[string]any, label string) {
	t.Helper()
	if schema == nil {
		t.Fatalf("%s: schema missing", label)
	}
	v, ok := schema["additionalProperties"]
	if !ok {
		t.Errorf("%s: additionalProperties not set (Anthropic structured outputs require it to be false)", label)
		return
	}
	if b, isBool := v.(bool); !isBool || b {
		t.Errorf("%s: additionalProperties = %v, want false", label, v)
	}
}

func TestConvertRequest_OutputConfig_EnforcesAdditionalPropertiesFalse_AnyOf(t *testing.T) {
	m := &anthropicModel{
		canonicalModel:   "claude-haiku-4-5-20251001",
		variant:          VariantVertexAI,
		defaultMaxTokens: testMaxTokens,
	}

	// An object branch inside anyOf. SchemaToMap stores anyOf as
	// []map[string]any, so the enforcement walk must recurse into it.
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"choice": {
				AnyOf: []*genai.Schema{
					{
						Type:       genai.TypeObject,
						Properties: map[string]*genai.Schema{"name": {Type: genai.TypeString}},
					},
					{Type: genai.TypeString},
				},
			},
		},
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Hello", "user")},
		Config:   &genai.GenerateContentConfig{ResponseSchema: schema},
	}

	params, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}

	root := params.OutputConfig.Format.Schema
	if root == nil {
		t.Fatal("expected OutputConfig schema to be set")
	}

	props, _ := root["properties"].(map[string]any)
	choice, _ := props["choice"].(map[string]any)
	anyOf, ok := choice["anyOf"].([]any)
	if !ok {
		t.Fatalf("choice.anyOf: want []any, got %T", choice["anyOf"])
	}
	objectBranch, ok := anyOf[0].(map[string]any)
	if !ok {
		t.Fatalf("choice.anyOf[0]: want map[string]any, got %T", anyOf[0])
	}
	// The object branch under anyOf must get additionalProperties:false.
	assertAdditionalPropertiesFalse(t, objectBranch, "choice.anyOf[0]")
}

func TestConvertRequest_DirectAPI_SetsOutputConfig(t *testing.T) {
	m := &anthropicModel{
		canonicalModel:   "claude-haiku-4-5-20251001",
		variant:          VariantAnthropicAPI,
		defaultMaxTokens: testMaxTokens,
	}

	schema := &genai.Schema{
		Type:     genai.TypeObject,
		Required: []string{"name"},
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString},
		},
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", "user"),
		},
		Config: &genai.GenerateContentConfig{
			ResponseSchema: schema,
		},
	}

	params, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}

	if params.OutputConfig.Format.Schema == nil {
		t.Error("expected OutputConfig to be set for direct API, but it was empty")
	}
}

// TestConvertRequest_DefaultsToAdaptiveOnCapableModel guards against the
// regression spotted by Cursor Bugbot on PR #22: an earlier draft gated
// the thinking-config converter behind `if req.Config.ThinkingConfig != nil`,
// which made the converter's nil-handling path (return adaptive defaults
// for adaptive-capable models) unreachable from production. Unit tests
// that called the converter directly still passed, masking the integration
// gap. These cases lock in the contract that nil ThinkingConfig — whether
// inside a non-nil Config or via a nil Config entirely — produces adaptive
// thinking on a model that supports it.
func TestConvertRequest_DefaultsToAdaptiveOnCapableModel(t *testing.T) {
	cases := []struct {
		name string
		req  *model.LLMRequest
	}{
		{
			name: "nil_thinking_config_inside_non_nil_config",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("Hello", "user"),
				},
				Config: &genai.GenerateContentConfig{},
			},
		},
		{
			name: "nil_config",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("Hello", "user"),
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &anthropicModel{
				// Adaptive-capable model — unversioned SDK alias.
				canonicalModel:   "claude-sonnet-4-6",
				variant:          VariantAnthropicAPI,
				defaultMaxTokens: testMaxTokens,
			}

			params, err := m.convertRequest(tc.req)
			if err != nil {
				t.Fatalf("convertRequest() error = %v", err)
			}

			if params.Thinking.OfAdaptive == nil {
				t.Fatalf("expected adaptive thinking on %s, got Thinking=%+v", m.canonicalModel, params.Thinking)
			}
		})
	}
}

// TestConvertRequest_NilConfigLeavesThinkingOffOnNonAdaptive locks in the
// other half of the contract: on a model that doesn't support adaptive
// thinking (Haiku, older Sonnet/Opus), nil ThinkingConfig keeps thinking
// off rather than forcing a manual budget. The fall-through path through
// the converter must return an empty mapping for non-adaptive models.
func TestConvertRequest_NilConfigLeavesThinkingOffOnNonAdaptive(t *testing.T) {
	m := &anthropicModel{
		// Manual-only model — adaptive is not supported.
		canonicalModel:   "claude-haiku-4-5",
		variant:          VariantAnthropicAPI,
		defaultMaxTokens: testMaxTokens,
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", "user"),
		},
	}

	params, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}

	if params.Thinking.OfAdaptive != nil {
		t.Errorf("non-adaptive model %s should not default to adaptive thinking, got OfAdaptive=%+v", m.canonicalModel, params.Thinking.OfAdaptive)
	}
	if params.Thinking.OfEnabled != nil {
		t.Errorf("non-adaptive model %s should not default to manual thinking budget, got OfEnabled=%+v", m.canonicalModel, params.Thinking.OfEnabled)
	}
}

func TestConvertRequest_GatewayModelUsesEffortWithoutAdaptiveThinking(t *testing.T) {
	tests := []struct {
		name       string
		level      genai.ThinkingLevel
		wantEffort anthropic.OutputConfigEffort
	}{
		{name: "high", level: genai.ThinkingLevelHigh, wantEffort: anthropic.OutputConfigEffortHigh},
		{name: "low", level: genai.ThinkingLevelLow, wantEffort: anthropic.OutputConfigEffortLow},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &anthropicModel{
				canonicalModel:   "glm-5.3-flash",
				requestModel:     "zai/glm-5.3-flash",
				variant:          VariantAnthropicAPI,
				defaultMaxTokens: testMaxTokens,
			}
			req := &model.LLMRequest{
				Contents: []*genai.Content{genai.NewContentFromText("Hello", "user")},
				Config: &genai.GenerateContentConfig{
					ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: tt.level},
				},
			}

			params, err := m.convertRequest(req)
			if err != nil {
				t.Fatalf("convertRequest() error = %v", err)
			}
			if params.OutputConfig.Effort != tt.wantEffort {
				t.Errorf("OutputConfig.Effort = %q, want %q", params.OutputConfig.Effort, tt.wantEffort)
			}
			if params.Thinking.OfAdaptive != nil || params.Thinking.OfEnabled != nil {
				t.Errorf("expected no Anthropic Thinking config, got %+v", params.Thinking)
			}
			if params.Model != "zai/glm-5.3-flash" {
				t.Errorf("Model = %q, want %q", params.Model, "zai/glm-5.3-flash")
			}
		})
	}
}

// TestConvertRequest_ForcedToolUseDropsThinking locks in the workaround for
// Anthropic's "forced tool use cannot be combined with extended thinking"
// constraint. The combination is easy to land into via the genai shape:
//
//   - ToolConfig.FunctionCallingConfig.Mode = ModeAny (with or without an
//     AllowedFunctionNames whitelist) maps to tool_choice.type "any" or "tool".
//   - ThinkingConfig.ThinkingLevel ∈ {Low, Medium, High} maps to adaptive
//     thinking on Sonnet 4.6+ / Opus 4.6+ / Mythos.
//
// Sent together, Anthropic may ignore tool_choice and reply with text or
// thinking blocks — which looks to callers like the model refused to use the
// tool. The converter drops thinking when tool_choice is forced; this test
// guards that contract for both the specific-tool ("OfTool") and
// any-tool ("OfAny") shapes, plus the adaptive and manual thinking variants.
func TestConvertRequest_ForcedToolUseDropsThinking(t *testing.T) {
	toolDecl := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "save_thing",
				Description: "Save a thing.",
				Parameters: &genai.Schema{
					Type:     genai.TypeObject,
					Required: []string{"id"},
					Properties: map[string]*genai.Schema{
						"id": {Type: genai.TypeString},
					},
				},
			},
		},
	}

	cases := []struct {
		name       string
		modelName  string // adaptive Claude, manual-only Claude, or gateway model
		toolConfig *genai.ToolConfig
		thinking   *genai.ThinkingConfig
		wantEffort anthropic.OutputConfigEffort
	}{
		{
			name:      "adaptive_model_specific_tool",
			modelName: "claude-sonnet-4-6",
			toolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"save_thing"},
				},
			},
			thinking: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow},
		},
		{
			name:      "adaptive_model_any_tool",
			modelName: "claude-sonnet-4-6",
			toolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			},
			thinking: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
		},
		{
			name:      "adaptive_model_nil_thinking_config_defaults_to_adaptive",
			modelName: "claude-sonnet-4-6",
			toolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"save_thing"},
				},
			},
			thinking: nil, // adaptive-capable + nil → adaptive defaults; must still be dropped.
		},
		{
			name:      "manual_model_specific_tool",
			modelName: "claude-haiku-4-5",
			toolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"save_thing"},
				},
			},
			thinking: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
		},
		{
			name:      "manual_model_explicit_budget",
			modelName: "claude-haiku-4-5",
			toolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			},
			thinking: &genai.ThinkingConfig{ThinkingBudget: ptrInt32(5000)},
		},
		{
			name:      "gateway_model_keeps_effort",
			modelName: "glm-5.3-flash",
			toolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			},
			thinking:   &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
			wantEffort: anthropic.OutputConfigEffortHigh,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &anthropicModel{
				canonicalModel:   tc.modelName,
				variant:          VariantAnthropicAPI,
				defaultMaxTokens: testMaxTokens,
			}

			req := &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("Hello", "user"),
				},
				Config: &genai.GenerateContentConfig{
					Tools:          []*genai.Tool{toolDecl},
					ToolConfig:     tc.toolConfig,
					ThinkingConfig: tc.thinking,
				},
			}

			params, err := m.convertRequest(req)
			if err != nil {
				t.Fatalf("convertRequest() error = %v", err)
			}

			if params.Thinking.OfAdaptive != nil || params.Thinking.OfEnabled != nil {
				t.Errorf("expected Thinking to be cleared under forced tool_choice, got %+v", params.Thinking)
			}
			if params.OutputConfig.Effort != tc.wantEffort {
				t.Errorf("OutputConfig.Effort = %q, want %q", params.OutputConfig.Effort, tc.wantEffort)
			}
			if params.ToolChoice.OfAny == nil && params.ToolChoice.OfTool == nil {
				t.Errorf("expected forced ToolChoice (OfAny or OfTool) to be preserved, got %+v", params.ToolChoice)
			}
		})
	}
}

// TestConvertRequest_AutoToolUseKeepsThinking ensures the conflict resolution
// only fires when tool_choice is genuinely forced. ModeAuto, ModeNone, and no
// ToolConfig at all must all preserve the thinking parameter that the caller
// (or the model-aware defaults) selected.
func TestConvertRequest_AutoToolUseKeepsThinking(t *testing.T) {
	cases := []struct {
		name       string
		toolConfig *genai.ToolConfig
	}{
		{
			name: "auto_mode",
			toolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAuto,
				},
			},
		},
		{
			name: "none_mode",
			toolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeNone,
				},
			},
		},
		{
			name:       "no_tool_config",
			toolConfig: nil,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &anthropicModel{
				canonicalModel:   "claude-sonnet-4-6",
				variant:          VariantAnthropicAPI,
				defaultMaxTokens: testMaxTokens,
			}

			req := &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("Hello", "user"),
				},
				Config: &genai.GenerateContentConfig{
					ToolConfig:     tc.toolConfig,
					ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow},
				},
			}

			params, err := m.convertRequest(req)
			if err != nil {
				t.Fatalf("convertRequest() error = %v", err)
			}

			if params.Thinking.OfAdaptive == nil {
				t.Errorf("expected adaptive thinking to be preserved when tool_choice is not forced, got %+v", params.Thinking)
			}
		})
	}
}

func ptrInt32(v int32) *int32 { return &v }

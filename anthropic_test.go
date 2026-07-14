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

func TestConvertRequest_VertexAI_SetsOutputConfig(t *testing.T) {
	m := &anthropicModel{
		name:             "claude-haiku-4-5-20251001",
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
		name:             "claude-haiku-4-5-20251001",
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
		name:             "claude-haiku-4-5-20251001",
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
		name:             "claude-haiku-4-5-20251001",
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
		name:             "claude-haiku-4-5-20251001",
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
				name:             "claude-sonnet-4-6",
				variant:          VariantAnthropicAPI,
				defaultMaxTokens: testMaxTokens,
			}

			params, err := m.convertRequest(tc.req)
			if err != nil {
				t.Fatalf("convertRequest() error = %v", err)
			}

			if params.Thinking.OfAdaptive == nil {
				t.Fatalf("expected adaptive thinking on %s, got Thinking=%+v", m.name, params.Thinking)
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
		name:             "claude-haiku-4-5",
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
		t.Errorf("non-adaptive model %s should not default to adaptive thinking, got OfAdaptive=%+v", m.name, params.Thinking.OfAdaptive)
	}
	if params.Thinking.OfEnabled != nil {
		t.Errorf("non-adaptive model %s should not default to manual thinking budget, got OfEnabled=%+v", m.name, params.Thinking.OfEnabled)
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
		modelName  string // adaptive-capable vs manual-only
		toolConfig *genai.ToolConfig
		thinking   *genai.ThinkingConfig
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
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := &anthropicModel{
				name:             tc.modelName,
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
			if params.OutputConfig.Effort != "" {
				t.Errorf("expected OutputConfig.Effort to be cleared, got %q", params.OutputConfig.Effort)
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
				name:             "claude-sonnet-4-6",
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

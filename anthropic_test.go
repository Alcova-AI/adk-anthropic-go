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
	"github.com/anthropics/anthropic-sdk-go/option"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/Alcova-AI/adk-anthropic-go/v3/vercel"
)

func TestNewModel_RequiresConstructedClientAndCanonicalModel(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want string
	}{
		{name: "missing client", cfg: Config{CanonicalModel: "claude-sonnet-5"}, want: "Client must be constructed"},
		{name: "missing model", cfg: Config{Client: testClient()}, want: "CanonicalModel is required"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewModel(tt.cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("NewModel() error = %v, want contains %q", err, tt.want)
			}
		})
	}
}

func TestNewModel_UsesCanonicalAndRequestModels(t *testing.T) {
	llm, err := NewModel(Config{
		Client:         testClient(),
		CanonicalModel: "claude-sonnet-5",
		RequestModel:   "anthropic/claude-sonnet-5",
	}, WithDefaultMaxTokens(2048))
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}
	if got := llm.Name(); got != "claude-sonnet-5" {
		t.Fatalf("Name() = %q, want claude-sonnet-5", got)
	}
	m := llm.(*anthropicModel)
	if m.wireModel() != "anthropic/claude-sonnet-5" {
		t.Fatalf("wireModel() = %q", m.wireModel())
	}
	if m.defaultMaxTokens != 2048 {
		t.Fatalf("defaultMaxTokens = %d, want 2048", m.defaultMaxTokens)
	}
}

func TestNewModel_DefaultMaxTokensSupportsNonStreaming(t *testing.T) {
	m := mustTestModel(t, "claude-haiku-4-5")
	if _, err := anthropic.CalculateNonStreamingTimeout(m.defaultMaxTokens, "claude-haiku-4-5", nil); err != nil {
		t.Fatalf("default max_tokens %d is incompatible with non-streaming requests: %v", m.defaultMaxTokens, err)
	}
}

func TestReasoningStrategies(t *testing.T) {
	tests := []struct {
		name         string
		config       ReasoningConfig
		request      *genai.ThinkingConfig
		wantAdaptive bool
		wantBudget   int64
		wantEffort   anthropic.OutputConfigEffort
		wantDisplay  string
	}{
		{
			name:    "disabled ignores request level",
			config:  ReasoningConfig{Strategy: ReasoningDisabled},
			request: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
		},
		{
			name:         "adaptive uses request effort",
			config:       ReasoningConfig{Strategy: ReasoningAdaptiveEffort, DefaultLevel: genai.ThinkingLevelMedium},
			request:      &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow, IncludeThoughts: true},
			wantAdaptive: true,
			wantEffort:   anthropic.OutputConfigEffortLow,
			wantDisplay:  "summarized",
		},
		{
			name:         "adaptive uses route default",
			config:       ReasoningConfig{Strategy: ReasoningAdaptiveEffort, DefaultLevel: genai.ThinkingLevelMedium},
			wantAdaptive: true,
			wantEffort:   anthropic.OutputConfigEffortMedium,
			wantDisplay:  "omitted",
		},
		{
			name:    "minimal disables adaptive",
			config:  ReasoningConfig{Strategy: ReasoningAdaptiveEffort, DefaultLevel: genai.ThinkingLevelHigh},
			request: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMinimal},
		},
		{
			name:        "budget maps medium",
			config:      ReasoningConfig{Strategy: ReasoningTokenBudget},
			request:     &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMedium, IncludeThoughts: true},
			wantBudget:  5000,
			wantDisplay: "summarized",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := mustTestModel(t, "route-model", WithReasoning(tt.config))
			params, err := m.convertRequest(testRequest(tt.request))
			if err != nil {
				t.Fatalf("convertRequest() error = %v", err)
			}
			if got := params.Thinking.OfAdaptive != nil; got != tt.wantAdaptive {
				t.Fatalf("adaptive = %v, want %v", got, tt.wantAdaptive)
			}
			if params.Thinking.OfEnabled != nil {
				if params.Thinking.OfEnabled.BudgetTokens != tt.wantBudget {
					t.Fatalf("budget = %d, want %d", params.Thinking.OfEnabled.BudgetTokens, tt.wantBudget)
				}
			} else if tt.wantBudget != 0 {
				t.Fatalf("missing enabled thinking with budget %d", tt.wantBudget)
			}
			if params.OutputConfig.Effort != tt.wantEffort {
				t.Fatalf("effort = %q, want %q", params.OutputConfig.Effort, tt.wantEffort)
			}
			if tt.wantDisplay != "" {
				raw, err := json.Marshal(params.Thinking)
				if err != nil {
					t.Fatal(err)
				}
				if !strings.Contains(string(raw), `"display":"`+tt.wantDisplay+`"`) {
					t.Fatalf("thinking = %s, want display %s", raw, tt.wantDisplay)
				}
			}
		})
	}
}

func TestReasoningRejectsExplicitBudget(t *testing.T) {
	m := mustTestModel(t, "route-model", WithReasoning(ReasoningConfig{Strategy: ReasoningTokenBudget}))
	budget := int32(2048)
	_, err := m.convertRequest(testRequest(&genai.ThinkingConfig{ThinkingBudget: &budget}))
	if err == nil || !strings.Contains(err.Error(), "ThinkingBudget is not supported") {
		t.Fatalf("convertRequest() error = %v", err)
	}
}

func TestForcedToolUseDropsThinkingAndEffort(t *testing.T) {
	m := mustTestModel(t, "claude-sonnet-5", WithReasoning(ReasoningConfig{
		Strategy:     ReasoningAdaptiveEffort,
		DefaultLevel: genai.ThinkingLevelHigh,
	}))
	req := testRequest(&genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh})
	req.Config.ToolConfig = &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
		Mode:                 genai.FunctionCallingConfigModeAny,
		AllowedFunctionNames: []string{"save"},
	}}
	params, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}
	if params.Thinking.OfAdaptive != nil || params.Thinking.OfEnabled != nil || params.OutputConfig.Effort != "" {
		t.Fatalf("forced tool request kept reasoning: thinking=%+v effort=%q", params.Thinking, params.OutputConfig.Effort)
	}
}

func TestVercelGateway_RequestAndResponseMetadata(t *testing.T) {
	requestBody := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		var decoded map[string]any
		if err := json.Unmarshal(body, &decoded); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		requestBody <- decoded
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"msg_1","type":"message","role":"assistant","model":"zai/glm-5.3-flash",
			"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,
			"usage":{"input_tokens":4,"output_tokens":2},
			"providerMetadata":{"gateway":{"cost":"0.0123","generationId":"gen_1","routing":{
				"originalModelId":"zai/glm-5.3-flash","resolvedProvider":"baseten","canonicalSlug":"zai/glm-5.3-flash",
				"modelAttemptCount":1,"totalProviderAttemptCount":2
			}}}
		}`)
	}))
	defer srv.Close()

	client := anthropic.NewClient(option.WithAPIKey("gateway-key"), option.WithBaseURL(srv.URL))
	llm, err := NewModel(Config{
		Client:         client,
		CanonicalModel: "glm-5.3-flash",
		RequestModel:   "zai/glm-5.3-flash",
	},
		WithReasoning(ReasoningConfig{Strategy: ReasoningTokenBudget, DefaultLevel: genai.ThinkingLevelMedium}),
		WithVercelGateway(vercel.Config{Routing: vercel.Routing{Only: []string{"zai", "baseten"}}}),
	)
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	var response *model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), testRequest(nil), false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		response = resp
	}

	decoded := <-requestBody
	if decoded["model"] != "zai/glm-5.3-flash" {
		t.Fatalf("request model = %v", decoded["model"])
	}
	providerOptions := decoded["providerOptions"].(map[string]any)
	gateway := providerOptions["gateway"].(map[string]any)
	if gateway["zeroDataRetention"] != true {
		t.Fatalf("zeroDataRetention = %v, want true", gateway["zeroDataRetention"])
	}
	if got := decoded["thinking"].(map[string]any)["budget_tokens"]; got != float64(5000) {
		t.Fatalf("thinking budget = %v, want 5000", got)
	}

	metadata, ok := vercel.MetadataFromResponse(response)
	if !ok {
		t.Fatal("missing typed Vercel metadata")
	}
	if metadata.ResolvedProvider != "baseten" || metadata.CostUSD == nil || *metadata.CostUSD != 0.0123 {
		t.Fatalf("metadata = %+v", metadata)
	}
}

func TestVercelGateway_RetentionAllowedIsExplicit(t *testing.T) {
	m := mustTestModel(t, "route-model", WithVercelGateway(vercel.Config{DataPolicy: vercel.RetentionAllowed}))
	options := m.vercel.WireProviderOptions()
	gateway := options["gateway"].(map[string]any)
	if gateway["zeroDataRetention"] != false {
		t.Fatalf("zeroDataRetention = %v, want false", gateway["zeroDataRetention"])
	}
}

func TestPromptCachingModes(t *testing.T) {
	breakpoint := &CacheBreakpoint{}
	_, err := NewModel(Config{Client: testClient(), CanonicalModel: "route-model"}, WithPromptCaching(PromptCachingConfig{
		Mode:  PromptCacheGatewayAutomatic,
		Tools: breakpoint,
	}))
	if err == nil || !strings.Contains(err.Error(), "breakpoints require manual mode") {
		t.Fatalf("NewModel() error = %v", err)
	}

	m := mustTestModel(t, "route-model", WithPromptCaching(PromptCachingConfig{
		Mode: PromptCacheManual,
		Auto: breakpoint,
	}))
	params, err := m.convertRequest(testRequest(nil))
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}
	if params.CacheControl.Type == "" {
		t.Fatal("manual prompt cache breakpoint is missing")
	}
}

func TestConvertRequest_SetsStructuredOutput(t *testing.T) {
	m := mustTestModel(t, "route-model")
	req := testRequest(nil)
	req.Config.ResponseSchema = &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"answer": {Type: genai.TypeString}},
		Required:   []string{"answer"},
	}
	params, err := m.convertRequest(req)
	if err != nil {
		t.Fatalf("convertRequest() error = %v", err)
	}
	if params.OutputConfig.Format.Schema == nil {
		t.Fatal("structured output schema is missing")
	}
}

func TestFilterThoughtParts_PreservesProviderState(t *testing.T) {
	parts := []*genai.Part{
		{Text: "unsigned", Thought: true},
		{Text: "signed", Thought: true, ThoughtSignature: []byte("sig")},
		{Text: "redacted", Thought: true, PartMetadata: map[string]any{"provider": "state"}},
		{Text: "answer"},
	}
	got := filterThoughtParts(parts)
	if len(got) != 3 || got[0].Text != "signed" || got[1].Text != "redacted" || got[2].Text != "answer" {
		t.Fatalf("filterThoughtParts() = %+v", got)
	}
}

func TestFilterThoughtParts_PreservesThoughtOnlyTurn(t *testing.T) {
	got := filterThoughtParts([]*genai.Part{{Text: "hidden", Thought: true}})
	if len(got) != 1 || !got[0].Thought || got[0].Text != "" {
		t.Fatalf("filterThoughtParts() = %+v", got)
	}
}

func testClient(opts ...option.RequestOption) anthropic.Client {
	return anthropic.NewClient(append([]option.RequestOption{option.WithAPIKey("test-key")}, opts...)...)
}

func mustTestModel(t *testing.T, canonical anthropic.Model, options ...Option) *anthropicModel {
	t.Helper()
	llm, err := NewModel(Config{Client: testClient(), CanonicalModel: canonical}, options...)
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}
	return llm.(*anthropicModel)
}

func testRequest(thinking *genai.ThinkingConfig) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Hello", "user")},
		Config:   &genai.GenerateContentConfig{ThinkingConfig: thinking},
	}
}

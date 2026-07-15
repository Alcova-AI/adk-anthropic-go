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

package converters_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"github.com/Alcova-AI/adk-anthropic-go/v2/converters"
)

func TestContentsToMessages_SimpleText(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("Hello", "user"),
	}

	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	if messages[0].Role != "user" {
		t.Errorf("expected role 'user', got %q", messages[0].Role)
	}
}

func TestContentsToMessages_MultiTurn(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("Hello", "user"),
		genai.NewContentFromText("Hi there!", "model"),
		genai.NewContentFromText("How are you?", "user"),
	}

	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(messages))
	}

	expectedRoles := []string{"user", "assistant", "user"}
	for i, msg := range messages {
		if string(msg.Role) != expectedRoles[i] {
			t.Errorf("message %d: expected role %q, got %q", i, expectedRoles[i], msg.Role)
		}
	}
}

func TestContentsToMessages_MergesConsecutiveUserMessages(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("Hello", "user"),
		genai.NewContentFromText("How are you?", "user"),
	}

	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	// Should merge consecutive user messages
	if len(messages) != 1 {
		t.Fatalf("expected 1 merged message, got %d", len(messages))
	}

	// Should have 2 content blocks
	if len(messages[0].Content) != 2 {
		t.Errorf("expected 2 content blocks, got %d", len(messages[0].Content))
	}
}

func TestContentsToMessages_MergesConsecutiveAssistantMessagesWithoutThinking(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("First response", "model"),
		genai.NewContentFromText("Second response", "model"),
	}

	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 merged message, got %d", len(messages))
	}
	if messages[0].Role != anthropic.MessageParamRoleAssistant {
		t.Errorf("role = %q, want %q", messages[0].Role, anthropic.MessageParamRoleAssistant)
	}
	if len(messages[0].Content) != 2 {
		t.Errorf("expected 2 content blocks, got %d", len(messages[0].Content))
	}
}

func TestContentsToMessages_PreservesBoundaryWhenEitherAssistantMessageHasThinking(t *testing.T) {
	thought := &genai.Part{
		Text:             "Signed thought",
		Thought:          true,
		ThoughtSignature: []byte("signature"),
	}
	emptyThought := &genai.Part{
		Thought:          true,
		ThoughtSignature: []byte("signature"),
	}
	plain := &genai.Part{Text: "Plain response"}
	tests := []struct {
		name  string
		parts []*genai.Part
	}{
		{name: "thinking then plain", parts: []*genai.Part{thought, plain}},
		{name: "plain then thinking", parts: []*genai.Part{plain, thought}},
		{name: "empty thinking then plain", parts: []*genai.Part{emptyThought, plain}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages, err := converters.ContentsToMessages([]*genai.Content{
				{Role: "model", Parts: []*genai.Part{tt.parts[0]}},
				{Role: "model", Parts: []*genai.Part{tt.parts[1]}},
			})
			if err != nil {
				t.Fatalf("ContentsToMessages() error = %v", err)
			}

			wantRoles := []anthropic.MessageParamRole{
				anthropic.MessageParamRoleAssistant,
				anthropic.MessageParamRoleUser,
				anthropic.MessageParamRoleAssistant,
			}
			if len(messages) != len(wantRoles) {
				t.Fatalf("expected %d messages, got %d", len(wantRoles), len(messages))
			}
			for i, wantRole := range wantRoles {
				if messages[i].Role != wantRole {
					t.Errorf("message %d: role = %q, want %q", i, messages[i].Role, wantRole)
				}
			}
		})
	}
}

func TestContentsToMessages_PreservesRedactedThinkingBoundary(t *testing.T) {
	var responseBlock anthropic.ContentBlockUnion
	if err := responseBlock.UnmarshalJSON([]byte(`{
		"type": "redacted_thinking",
		"data": "opaque-redacted-data"
	}`)); err != nil {
		t.Fatalf("failed to unmarshal redacted thinking block: %v", err)
	}

	redactedPart, err := converters.ContentBlockToGenaiPart(responseBlock)
	if err != nil {
		t.Fatalf("ContentBlockToGenaiPart() error = %v", err)
	}
	partJSON, err := json.Marshal(redactedPart)
	if err != nil {
		t.Fatalf("failed to marshal redacted thinking part: %v", err)
	}
	var persistedPart genai.Part
	if err := json.Unmarshal(partJSON, &persistedPart); err != nil {
		t.Fatalf("failed to unmarshal redacted thinking part: %v", err)
	}
	messages, err := converters.ContentsToMessages([]*genai.Content{
		{Role: "model", Parts: []*genai.Part{&persistedPart}},
		genai.NewContentFromText("Plain response", "model"),
	})
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	wantRoles := []anthropic.MessageParamRole{
		anthropic.MessageParamRoleAssistant,
		anthropic.MessageParamRoleUser,
		anthropic.MessageParamRoleAssistant,
	}
	if len(messages) != len(wantRoles) {
		t.Fatalf("expected %d messages, got %d", len(wantRoles), len(messages))
	}
	for i, wantRole := range wantRoles {
		if messages[i].Role != wantRole {
			t.Errorf("message %d: role = %q, want %q", i, messages[i].Role, wantRole)
		}
	}

	redactedThinking := messages[0].Content[0].OfRedactedThinking
	if redactedThinking == nil {
		t.Fatal("first assistant message does not contain redacted thinking")
	}
	if redactedThinking.Data != "opaque-redacted-data" {
		t.Errorf("redacted thinking data = %q, want %q", redactedThinking.Data, "opaque-redacted-data")
	}
}

func TestContentsToMessages_DropsThoughtsFromUserMessages(t *testing.T) {
	var redactedBlock anthropic.ContentBlockUnion
	if err := redactedBlock.UnmarshalJSON([]byte(`{
		"type": "redacted_thinking",
		"data": "opaque-redacted-data"
	}`)); err != nil {
		t.Fatalf("failed to unmarshal redacted thinking block: %v", err)
	}
	redactedPart, err := converters.ContentBlockToGenaiPart(redactedBlock)
	if err != nil {
		t.Fatalf("ContentBlockToGenaiPart() error = %v", err)
	}

	tests := []struct {
		name    string
		thought *genai.Part
	}{
		{
			name: "empty signed thought",
			thought: &genai.Part{
				Thought:          true,
				ThoughtSignature: []byte("signature"),
			},
		},
		{
			name: "non-empty signed thought",
			thought: &genai.Part{
				Text:             "Hidden reasoning",
				Thought:          true,
				ThoughtSignature: []byte("signature"),
			},
		},
		{
			name:    "redacted thought",
			thought: redactedPart,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages, err := converters.ContentsToMessages([]*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "For context: [researcher] said: relevant result"},
						tt.thought,
					},
				},
			})
			if err != nil {
				t.Fatalf("ContentsToMessages() error = %v", err)
			}

			if len(messages) != 1 {
				t.Fatalf("expected 1 message, got %d", len(messages))
			}
			if messages[0].Role != anthropic.MessageParamRoleUser {
				t.Errorf("role = %q, want %q", messages[0].Role, anthropic.MessageParamRoleUser)
			}
			if len(messages[0].Content) != 1 {
				t.Fatalf("expected only the context text block, got %d blocks", len(messages[0].Content))
			}
			textBlock := messages[0].Content[0].OfText
			if textBlock == nil {
				t.Fatal("user message does not contain the context text block")
			}
			if textBlock.Text != "For context: [researcher] said: relevant result" {
				t.Errorf("text = %q, want the original context", textBlock.Text)
			}
		})
	}
}

func TestThinkingBlock_RoundTripsThroughPersistedPart(t *testing.T) {
	var responseBlock anthropic.ContentBlockUnion
	if err := responseBlock.UnmarshalJSON([]byte(`{
		"type": "thinking",
		"thinking": "",
		"signature": "c2lnbmF0dXJl"
	}`)); err != nil {
		t.Fatalf("failed to unmarshal thinking block: %v", err)
	}

	part, err := converters.ContentBlockToGenaiPart(responseBlock)
	if err != nil {
		t.Fatalf("ContentBlockToGenaiPart() error = %v", err)
	}
	partJSON, err := json.Marshal(part)
	if err != nil {
		t.Fatalf("failed to marshal thinking part: %v", err)
	}
	var persistedPart genai.Part
	if err := json.Unmarshal(partJSON, &persistedPart); err != nil {
		t.Fatalf("failed to unmarshal thinking part: %v", err)
	}

	requestBlock, err := converters.PartToContentBlock(&persistedPart)
	if err != nil {
		t.Fatalf("PartToContentBlock() error = %v", err)
	}
	if requestBlock == nil || requestBlock.OfThinking == nil {
		t.Fatal("persisted part did not round-trip to a thinking block")
	}
	if requestBlock.OfThinking.Thinking != "" {
		t.Errorf("thinking text = %q, want empty text", requestBlock.OfThinking.Thinking)
	}
	if requestBlock.OfThinking.Signature != "c2lnbmF0dXJl" {
		t.Errorf("thinking signature = %q, want %q", requestBlock.OfThinking.Signature, "c2lnbmF0dXJl")
	}
}

func TestContentsToMessages_RejectsThinkingBoundaryAfterToolUse(t *testing.T) {
	_, err := converters.ContentsToMessages([]*genai.Content{
		{
			Role: "model",
			Parts: []*genai.Part{
				{Text: "Signed thought", Thought: true, ThoughtSignature: []byte("signature")},
				{FunctionCall: &genai.FunctionCall{ID: "toolu_123", Name: "edit_document"}},
			},
		},
		{
			Role: "model",
			Parts: []*genai.Part{
				{Text: "Another thought", Thought: true, ThoughtSignature: []byte("signature-2")},
			},
		},
	})
	if err == nil {
		t.Fatal("expected missing tool result to return an error")
	}
	if !strings.Contains(err.Error(), "without an intervening tool result") {
		t.Errorf("error = %q, want missing tool result context", err)
	}
}

func TestContentBlockToGenaiPart_RejectsInvalidThinkingSignature(t *testing.T) {
	var responseBlock anthropic.ContentBlockUnion
	if err := responseBlock.UnmarshalJSON([]byte(`{
		"type": "thinking",
		"thinking": "Signed thought",
		"signature": "not-valid-base64"
	}`)); err != nil {
		t.Fatalf("failed to unmarshal thinking block: %v", err)
	}

	_, err := converters.ContentBlockToGenaiPart(responseBlock)
	if err == nil {
		t.Fatal("expected invalid thinking signature to return an error")
	}
	if !strings.Contains(err.Error(), "failed to decode thinking signature") {
		t.Errorf("error = %q, want thinking signature context", err)
	}
}

func TestContentsToMessages_PreservesConsecutiveSignedThinkingMessages(t *testing.T) {
	firstSignature := []byte("first-signature")
	secondSignature := []byte("second-signature")
	contents := []*genai.Content{
		genai.NewContentFromText("Start", "user"),
		{
			Role: "model",
			Parts: []*genai.Part{
				{Text: "First thought", Thought: true, ThoughtSignature: firstSignature},
			},
		},
		{
			Role: "model",
			Parts: []*genai.Part{
				{Text: "Second thought", Thought: true, ThoughtSignature: secondSignature},
				{FunctionCall: &genai.FunctionCall{
					ID:   "toolu_123",
					Name: "edit_document",
					Args: map[string]any{"path": "report.docx"},
				}},
			},
		},
	}

	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	wantRoles := []anthropic.MessageParamRole{
		anthropic.MessageParamRoleUser,
		anthropic.MessageParamRoleAssistant,
		anthropic.MessageParamRoleUser,
		anthropic.MessageParamRoleAssistant,
	}
	if len(messages) != len(wantRoles) {
		t.Fatalf("expected %d messages, got %d", len(wantRoles), len(messages))
	}
	for i, wantRole := range wantRoles {
		if messages[i].Role != wantRole {
			t.Errorf("message %d: role = %q, want %q", i, messages[i].Role, wantRole)
		}
	}
	if len(messages[2].Content) != 1 || messages[2].Content[0].OfText == nil {
		t.Fatal("inserted user continuation does not contain exactly one text block")
	}
	wantContinuation := "Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed."
	if messages[2].Content[0].OfText.Text != wantContinuation {
		t.Errorf("continuation text = %q, want %q", messages[2].Content[0].OfText.Text, wantContinuation)
	}

	if len(messages[1].Content) != 1 {
		t.Fatalf("first assistant message has %d blocks, want 1", len(messages[1].Content))
	}
	firstThinking := messages[1].Content[0].OfThinking
	if firstThinking == nil {
		t.Fatal("first assistant message does not contain a thinking block")
	}
	if firstThinking.Thinking != "First thought" {
		t.Errorf("first thinking text = %q, want %q", firstThinking.Thinking, "First thought")
	}
	if firstThinking.Signature != "Zmlyc3Qtc2lnbmF0dXJl" {
		t.Errorf("first thinking signature = %q, want original base64 signature", firstThinking.Signature)
	}

	if len(messages[3].Content) != 2 {
		t.Fatalf("second assistant message has %d blocks, want 2", len(messages[3].Content))
	}
	secondThinking := messages[3].Content[0].OfThinking
	if secondThinking == nil {
		t.Fatal("second assistant message does not contain a thinking block")
	}
	if secondThinking.Thinking != "Second thought" {
		t.Errorf("second thinking text = %q, want %q", secondThinking.Thinking, "Second thought")
	}
	if secondThinking.Signature != "c2Vjb25kLXNpZ25hdHVyZQ==" {
		t.Errorf("second thinking signature = %q, want original base64 signature", secondThinking.Signature)
	}

	toolUse := messages[3].Content[1].OfToolUse
	if toolUse == nil {
		t.Fatal("second assistant message does not contain the original tool use")
	}
	if toolUse.ID != "toolu_123" || toolUse.Name != "edit_document" {
		t.Errorf("tool use = (%q, %q), want (%q, %q)", toolUse.ID, toolUse.Name, "toolu_123", "edit_document")
	}
	if diff := cmp.Diff(map[string]any{"path": "report.docx"}, toolUse.Input); diff != "" {
		t.Errorf("tool input mismatch (-want +got):\n%s", diff)
	}
}

func TestContentsToMessages_Empty(t *testing.T) {
	messages, err := converters.ContentsToMessages(nil)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if messages != nil {
		t.Errorf("expected nil messages, got %v", messages)
	}
}

func TestSystemInstructionToSystem(t *testing.T) {
	tests := []struct {
		name        string
		instruction *genai.Content
		wantLen     int
	}{
		{
			name:        "nil instruction",
			instruction: nil,
			wantLen:     0,
		},
		{
			name:        "single text part",
			instruction: genai.NewContentFromText("You are a helpful assistant.", "system"),
			wantLen:     1,
		},
		{
			name: "multiple text parts",
			instruction: &genai.Content{
				Role: "system",
				Parts: []*genai.Part{
					{Text: "You are a helpful assistant."},
					{Text: "Be concise."},
				},
			},
			wantLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blocks := converters.SystemInstructionToSystem(tt.instruction)
			if len(blocks) != tt.wantLen {
				t.Errorf("SystemInstructionToSystem() returned %d blocks, want %d", len(blocks), tt.wantLen)
			}
		})
	}
}

func TestStopReasonToFinishReason(t *testing.T) {
	tests := []struct {
		name string
		stop anthropic.StopReason
		want genai.FinishReason
	}{
		{"end_turn", anthropic.StopReasonEndTurn, genai.FinishReasonStop},
		{"max_tokens", anthropic.StopReasonMaxTokens, genai.FinishReasonMaxTokens},
		{"stop_sequence", anthropic.StopReasonStopSequence, genai.FinishReasonStop},
		{"tool_use", anthropic.StopReasonToolUse, genai.FinishReasonStop},
		{"unknown", anthropic.StopReason("unknown"), genai.FinishReasonUnspecified},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := converters.StopReasonToFinishReason(tt.stop)
			if got != tt.want {
				t.Errorf("StopReasonToFinishReason(%q) = %v, want %v", tt.stop, got, tt.want)
			}
		})
	}
}

func TestUsageToMetadata(t *testing.T) {
	usage := anthropic.Usage{InputTokens: 10, OutputTokens: 20}
	want := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     10,
		CandidatesTokenCount: 20,
		TotalTokenCount:      30,
	}
	got := converters.UsageToMetadata(usage)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("UsageToMetadata() mismatch (-want +got):\n%s", diff)
	}
}

func TestToolsToAnthropicTools(t *testing.T) {
	tests := []struct {
		name    string
		tools   []*genai.Tool
		wantLen int
	}{
		{
			name:    "nil tools",
			tools:   nil,
			wantLen: 0,
		},
		{
			name: "single tool with one function",
			tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        "get_weather",
							Description: "Get the weather for a location",
							Parameters: &genai.Schema{
								Type: "object",
								Properties: map[string]*genai.Schema{
									"location": {Type: "string", Description: "The city name"},
								},
								Required: []string{"location"},
							},
						},
					},
				},
			},
			wantLen: 1,
		},
		{
			name: "tool with multiple functions",
			tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{Name: "func1", Description: "Function 1"},
						{Name: "func2", Description: "Function 2"},
					},
				},
			},
			wantLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := converters.ToolsToAnthropicTools(tt.tools)
			if len(result) != tt.wantLen {
				t.Errorf("ToolsToAnthropicTools() returned %d tools, want %d", len(result), tt.wantLen)
			}
		})
	}
}

func TestSchemaConversion(t *testing.T) {
	tool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "complex_func",
				Description: "A function with complex schema",
				Parameters: &genai.Schema{
					Type: "object",
					Properties: map[string]*genai.Schema{
						"name":  {Type: "string"},
						"count": {Type: "integer"},
						"items": {
							Type: "array",
							Items: &genai.Schema{
								Type: "string",
							},
						},
						"nested": {
							Type: "object",
							Properties: map[string]*genai.Schema{
								"inner": {Type: "boolean"},
							},
						},
					},
					Required: []string{"name"},
				},
			},
		},
	}

	result := converters.ToolsToAnthropicTools([]*genai.Tool{tool})
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
}

func TestPartToContentBlock_UnsupportedTypes(t *testing.T) {
	part := &genai.Part{
		ExecutableCode: &genai.ExecutableCode{
			Code:     "print('hello')",
			Language: "python",
		},
	}

	_, err := converters.PartToContentBlock(part)
	if err == nil {
		t.Error("expected error for ExecutableCode, got nil")
	}
}

func TestFunctionResponseToBlock(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					ID:       "call_123",
					Name:     "get_weather",
					Response: map[string]any{"temperature": 72, "condition": "sunny"},
				},
			},
		},
	}

	messages, err := converters.ContentsToMessages([]*genai.Content{content})
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	if len(messages[0].Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(messages[0].Content))
	}
}

func TestFunctionResponseToBlock_RequiresID(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					Name:     "get_weather",
					Response: map[string]any{"temperature": 72},
				},
			},
		},
	}

	_, err := converters.ContentsToMessages([]*genai.Content{content})
	if err == nil {
		t.Fatal("expected error for FunctionResponse without ID, got nil")
	}

	if !strings.Contains(err.Error(), "FunctionResponse.ID is required") {
		t.Errorf("expected error about missing ID, got: %v", err)
	}
}

func TestFunctionResponse_ForcesUserRole(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{
				FunctionResponse: &genai.FunctionResponse{
					ID:       "toolu_abc123",
					Name:     "get_weather",
					Response: map[string]any{"temperature": 22},
				},
			},
		},
	}

	messages, err := converters.ContentsToMessages([]*genai.Content{content})
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	if messages[0].Role != "user" {
		t.Errorf("expected role 'user' for tool result, got %q", messages[0].Role)
	}
}

func TestFunctionCall_ForcesAssistantRole(t *testing.T) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				FunctionCall: &genai.FunctionCall{
					ID:   "toolu_abc123",
					Name: "get_weather",
					Args: map[string]any{"location": "London"},
				},
			},
		},
	}

	messages, err := converters.ContentsToMessages([]*genai.Content{content})
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	if messages[0].Role != "assistant" {
		t.Errorf("expected role 'assistant' for tool call, got %q", messages[0].Role)
	}
}

func TestToolCallAndResult_Correlation(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("What's the weather in London?", "user"),
		{
			Role: "model",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "toolu_weather_123",
						Name: "get_weather",
						Args: map[string]any{"location": "London"},
					},
				},
			},
		},
		{
			Role: "user",
			Parts: []*genai.Part{
				{
					FunctionResponse: &genai.FunctionResponse{
						ID:       "toolu_weather_123",
						Name:     "get_weather",
						Response: map[string]any{"temperature": 15, "condition": "cloudy"},
					},
				},
			},
		},
	}

	messages, err := converters.ContentsToMessages(contents)
	if err != nil {
		t.Fatalf("ContentsToMessages() error = %v", err)
	}

	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(messages))
	}

	expectedRoles := []string{"user", "assistant", "user"}
	for i, msg := range messages {
		if string(msg.Role) != expectedRoles[i] {
			t.Errorf("message %d: expected role %q, got %q", i, expectedRoles[i], msg.Role)
		}
	}
}

var cmpOpts = []cmp.Option{
	cmp.AllowUnexported(),
}

var _ = cmpOpts

func TestFunctionDeclarationToTool_JsonSchemaNoRequired(t *testing.T) {
	fd := &genai.FunctionDeclaration{
		Name:        "test_func",
		Description: "A test function",
		ParametersJsonSchema: &jsonschema.Schema{
			Properties: map[string]*jsonschema.Schema{
				"name": {Type: "string"},
			},
			// Required intentionally omitted (nil)
		},
	}

	result := converters.FunctionDeclarationToTool(fd)
	if result.OfTool == nil {
		t.Fatal("expected OfTool to be non-nil")
	}

	if result.OfTool.InputSchema.Required != nil {
		t.Errorf("expected Required to be nil when jsonschema.Schema has no required fields, got %v",
			result.OfTool.InputSchema.Required)
	}

	props, ok := result.OfTool.InputSchema.Properties.(map[string]any)
	if !ok {
		t.Fatalf("expected Properties to be map[string]any, got %T", result.OfTool.InputSchema.Properties)
	}
	if _, ok := props["name"]; !ok {
		t.Error("expected 'name' property to be present")
	}
}

func TestFunctionDeclarationToTool_ParametersJsonSchemaMap(t *testing.T) {
	tests := []struct {
		name         string
		fd           *genai.FunctionDeclaration
		wantProps    map[string]any
		wantRequired []string
	}{
		{
			name: "map with []any required (JSON unmarshalled)",
			fd: &genai.FunctionDeclaration{
				Name:        "test_func",
				Description: "A test function",
				ParametersJsonSchema: map[string]any{
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city name",
						},
					},
					"required": []any{"location"},
				},
			},
			wantProps: map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "The city name",
				},
			},
			wantRequired: []string{"location"},
		},
		{
			name: "map with []string required (manually constructed)",
			fd: &genai.FunctionDeclaration{
				Name:        "test_func",
				Description: "A test function",
				ParametersJsonSchema: map[string]any{
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
						"age":  map[string]any{"type": "integer"},
					},
					"required": []string{"name", "age"},
				},
			},
			wantProps: map[string]any{
				"name": map[string]any{"type": "string"},
				"age":  map[string]any{"type": "integer"},
			},
			wantRequired: []string{"name", "age"},
		},
		{
			name: "map with no required field",
			fd: &genai.FunctionDeclaration{
				Name: "optional_func",
				ParametersJsonSchema: map[string]any{
					"properties": map[string]any{
						"optional": map[string]any{"type": "string"},
					},
				},
			},
			wantProps: map[string]any{
				"optional": map[string]any{"type": "string"},
			},
			wantRequired: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := converters.FunctionDeclarationToTool(tt.fd)

			if result.OfTool == nil {
				t.Fatal("expected OfTool to be non-nil")
			}

			is := result.OfTool.InputSchema

			if diff := cmp.Diff(tt.wantProps, is.Properties); diff != "" {
				t.Errorf("Properties mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.wantRequired, is.Required); diff != "" {
				t.Errorf("Required mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFunctionDeclarationToTool_ParametersPrecedence(t *testing.T) {
	fd := &genai.FunctionDeclaration{
		Name: "test_func",
		Parameters: &genai.Schema{
			Type: "object",
			Properties: map[string]*genai.Schema{
				"from_parameters": {Type: "string"},
			},
			Required: []string{"from_parameters"},
		},
		ParametersJsonSchema: map[string]any{
			"properties": map[string]any{
				"from_json_schema": map[string]any{"type": "string"},
			},
			"required": []any{"from_json_schema"},
		},
	}

	result := converters.FunctionDeclarationToTool(fd)

	if result.OfTool == nil {
		t.Fatal("expected OfTool to be non-nil")
	}

	is := result.OfTool.InputSchema
	props, ok := is.Properties.(map[string]any)
	if !ok {
		t.Fatalf("expected Properties to be map[string]any, got %T", is.Properties)
	}

	if _, ok := props["from_parameters"]; !ok {
		t.Error("expected 'from_parameters' property from Parameters (should take precedence)")
	}
	if _, ok := props["from_json_schema"]; ok {
		t.Error("unexpected 'from_json_schema' property - Parameters should take precedence")
	}
}

func TestSchemaToMap_AllFields(t *testing.T) {
	min, max := 1.0, 100.0
	minLen, maxLen := int64(1), int64(50)
	minItems, maxItems := int64(1), int64(10)
	nullable := true

	tool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "full_schema_func",
				Description: "Function with all schema fields",
				Parameters: &genai.Schema{
					Type:        "object",
					Description: "Root object",
					Properties: map[string]*genai.Schema{
						"name": {
							Type:        "STRING",
							Description: "User name",
							MinLength:   &minLen,
							MaxLength:   &maxLen,
							Pattern:     "^[a-zA-Z]+$",
						},
						"age": {
							Type:     "INTEGER",
							Minimum:  &min,
							Maximum:  &max,
							Nullable: &nullable,
						},
						"tags": {
							Type:     "ARRAY",
							MinItems: &minItems,
							MaxItems: &maxItems,
							Items:    &genai.Schema{Type: "STRING"},
						},
						"status": {
							Type: "STRING",
							Enum: []string{"active", "inactive"},
						},
						"metadata": {
							Type: "OBJECT",
							Properties: map[string]*genai.Schema{
								"created": {Type: "STRING", Format: "date-time"},
							},
						},
					},
					Required: []string{"name"},
				},
			},
		},
	}

	result := converters.ToolsToAnthropicTools([]*genai.Tool{tool})
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}

	is := result[0].OfTool.InputSchema
	props, ok := is.Properties.(map[string]any)
	if !ok {
		t.Fatalf("expected Properties to be map[string]any, got %T", is.Properties)
	}

	nameSchema, ok := props["name"].(map[string]any)
	if !ok {
		t.Fatal("expected 'name' property")
	}
	if nameSchema["type"] != "string" {
		t.Errorf("name.type = %v, want 'string'", nameSchema["type"])
	}
	if nameSchema["minLength"] != minLen {
		t.Errorf("name.minLength = %v, want %v", nameSchema["minLength"], minLen)
	}
	if nameSchema["pattern"] != "^[a-zA-Z]+$" {
		t.Errorf("name.pattern = %v, want '^[a-zA-Z]+$'", nameSchema["pattern"])
	}

	ageSchema, ok := props["age"].(map[string]any)
	if !ok {
		t.Fatal("expected 'age' property")
	}
	if ageSchema["nullable"] != true {
		t.Errorf("age.nullable = %v, want true", ageSchema["nullable"])
	}
	if ageSchema["minimum"] != min {
		t.Errorf("age.minimum = %v, want %v", ageSchema["minimum"], min)
	}

	tagsSchema, ok := props["tags"].(map[string]any)
	if !ok {
		t.Fatal("expected 'tags' property")
	}
	if tagsSchema["minItems"] != minItems {
		t.Errorf("tags.minItems = %v, want %v", tagsSchema["minItems"], minItems)
	}
	items, ok := tagsSchema["items"].(map[string]any)
	if !ok {
		t.Fatal("expected 'tags.items'")
	}
	if items["type"] != "string" {
		t.Errorf("tags.items.type = %v, want 'string'", items["type"])
	}

	statusSchema, ok := props["status"].(map[string]any)
	if !ok {
		t.Fatal("expected 'status' property")
	}
	if statusSchema["enum"] == nil {
		t.Error("expected 'status.enum' to be set")
	}

	metaSchema, ok := props["metadata"].(map[string]any)
	if !ok {
		t.Fatal("expected 'metadata' property")
	}
	metaProps, ok := metaSchema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected 'metadata.properties'")
	}
	createdSchema, ok := metaProps["created"].(map[string]any)
	if !ok {
		t.Fatal("expected 'metadata.properties.created'")
	}
	if createdSchema["format"] != "date-time" {
		t.Errorf("created.format = %v, want 'date-time'", createdSchema["format"])
	}
}

func TestSchemaToMap_AnyOf(t *testing.T) {
	tool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name: "anyof_func",
				Parameters: &genai.Schema{
					Type: "object",
					Properties: map[string]*genai.Schema{
						"value": {
							AnyOf: []*genai.Schema{
								{Type: "STRING"},
								{Type: "INTEGER"},
							},
						},
					},
				},
			},
		},
	}

	result := converters.ToolsToAnthropicTools([]*genai.Tool{tool})
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}

	props, ok := result[0].OfTool.InputSchema.Properties.(map[string]any)
	if !ok {
		t.Fatalf("expected Properties to be map[string]any, got %T", result[0].OfTool.InputSchema.Properties)
	}
	valueSchema, ok := props["value"].(map[string]any)
	if !ok {
		t.Fatal("expected 'value' property")
	}

	anyOf, ok := valueSchema["anyOf"].([]map[string]any)
	if !ok {
		t.Fatalf("expected 'anyOf' to be []map[string]any, got %T", valueSchema["anyOf"])
	}
	if len(anyOf) != 2 {
		t.Errorf("expected 2 anyOf entries, got %d", len(anyOf))
	}
}

func TestMessageToLLMResponse_WithCitations(t *testing.T) {
	msgJSON := `{
		"content": [{
			"type": "text",
			"text": "According to the documentation...",
			"citations": [
				{
					"type": "char_location",
					"document_title": "API Reference",
					"start_char_index": 0,
					"end_char_index": 50,
					"cited_text": "some text"
				},
				{
					"type": "web_search_result_location",
					"title": "Official Docs",
					"url": "https://example.com/docs",
					"cited_text": "other text"
				}
			]
		}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 20}
	}`

	var msg anthropic.Message
	if err := msg.UnmarshalJSON([]byte(msgJSON)); err != nil {
		t.Fatalf("failed to unmarshal message: %v", err)
	}

	resp, err := converters.MessageToLLMResponse(&msg)
	if err != nil {
		t.Fatalf("MessageToLLMResponse() error = %v", err)
	}

	if resp.CitationMetadata == nil {
		t.Fatal("expected CitationMetadata to be set")
	}
	if len(resp.CitationMetadata.Citations) != 2 {
		t.Fatalf("expected 2 citations, got %d", len(resp.CitationMetadata.Citations))
	}

	c0 := resp.CitationMetadata.Citations[0]
	if c0.Title != "API Reference" {
		t.Errorf("citation[0].Title = %q, want 'API Reference'", c0.Title)
	}
	if c0.StartIndex != 0 || c0.EndIndex != 50 {
		t.Errorf("citation[0] indices = (%d, %d), want (0, 50)", c0.StartIndex, c0.EndIndex)
	}

	c1 := resp.CitationMetadata.Citations[1]
	if c1.Title != "Official Docs" {
		t.Errorf("citation[1].Title = %q, want 'Official Docs'", c1.Title)
	}
	if c1.URI != "https://example.com/docs" {
		t.Errorf("citation[1].URI = %q, want 'https://example.com/docs'", c1.URI)
	}
}

func TestMessageToLLMResponse_SetsModelVersion(t *testing.T) {
	msgJSON := `{
		"model": "claude-sonnet-4-5-20250929",
		"content": [{"type": "text", "text": "Hello"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 20}
	}`

	var msg anthropic.Message
	if err := msg.UnmarshalJSON([]byte(msgJSON)); err != nil {
		t.Fatalf("failed to unmarshal message: %v", err)
	}

	resp, err := converters.MessageToLLMResponse(&msg)
	if err != nil {
		t.Fatalf("MessageToLLMResponse() error = %v", err)
	}

	if resp.ModelVersion != "claude-sonnet-4-5-20250929" {
		t.Errorf("ModelVersion = %q, want %q", resp.ModelVersion, "claude-sonnet-4-5-20250929")
	}
}

func TestContentBlockToGenaiPart_WebSearchToolResult(t *testing.T) {
	blockJSON := `{
		"type": "web_search_tool_result",
		"tool_use_id": "toolu_search_123",
		"content": [
			{
				"type": "web_search_result",
				"title": "Example Page",
				"url": "https://example.com",
				"page_age": "2 days ago",
				"encrypted_content": "abc123"
			},
			{
				"type": "web_search_result",
				"title": "Another Page",
				"url": "https://another.com",
				"page_age": "1 week ago",
				"encrypted_content": "def456"
			}
		]
	}`

	var block anthropic.ContentBlockUnion
	if err := block.UnmarshalJSON([]byte(blockJSON)); err != nil {
		t.Fatalf("failed to unmarshal block: %v", err)
	}

	part, err := converters.ContentBlockToGenaiPart(block)
	if err != nil {
		t.Fatalf("ContentBlockToGenaiPart() error = %v", err)
	}

	if part == nil {
		t.Fatal("expected part to be non-nil")
	}
	if part.FunctionResponse == nil {
		t.Fatal("expected FunctionResponse to be set")
	}
	if part.FunctionResponse.ID != "toolu_search_123" {
		t.Errorf("FunctionResponse.ID = %q, want 'toolu_search_123'", part.FunctionResponse.ID)
	}
	if part.FunctionResponse.Name != "web_search" {
		t.Errorf("FunctionResponse.Name = %q, want 'web_search'", part.FunctionResponse.Name)
	}

	results, ok := part.FunctionResponse.Response["results"].([]map[string]any)
	if !ok {
		t.Fatalf("expected results to be []map[string]any, got %T", part.FunctionResponse.Response["results"])
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0]["title"] != "Example Page" {
		t.Errorf("results[0].title = %q, want 'Example Page'", results[0]["title"])
	}
	if results[0]["url"] != "https://example.com" {
		t.Errorf("results[0].url = %q, want 'https://example.com'", results[0]["url"])
	}
}

func int32Ptr(v int32) *int32 { return &v }

func TestThinkingConfigToAnthropicThinking(t *testing.T) {
	tests := []struct {
		name         string
		cfg          *genai.ThinkingConfig
		wantNil      bool // expect zero value (no thinking)
		wantAdaptive bool // expect OfAdaptive populated
		wantBudget   int64
	}{
		{
			name:    "nil config",
			cfg:     nil,
			wantNil: true,
		},
		{
			name:       "HIGH level without budget",
			cfg:        &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
			wantBudget: 10000,
		},
		{
			name:       "LOW level without budget",
			cfg:        &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow},
			wantBudget: 1024,
		},
		{
			name:       "explicit budget",
			cfg:        &genai.ThinkingConfig{ThinkingBudget: int32Ptr(5000)},
			wantBudget: 5000,
		},
		{
			name:       "level with explicit budget - budget wins",
			cfg:        &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow, ThinkingBudget: int32Ptr(8000)},
			wantBudget: 8000,
		},
		{
			name:    "IncludeThoughts alone is ignored",
			cfg:     &genai.ThinkingConfig{IncludeThoughts: true},
			wantNil: true,
		},
		{
			name:    "unspecified level no budget no include",
			cfg:     &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelUnspecified},
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := converters.ThinkingConfigToAnthropicThinking(tt.cfg)
			if tt.wantNil {
				if got.OfEnabled != nil {
					t.Errorf("expected zero value, got OfEnabled with budget %d", got.OfEnabled.BudgetTokens)
				}
				if got.OfAdaptive != nil {
					t.Errorf("expected zero value, got OfAdaptive populated")
				}
				return
			}
			if tt.wantAdaptive {
				if got.OfAdaptive == nil {
					t.Fatal("expected OfAdaptive to be non-nil")
				}
				if got.OfEnabled != nil {
					t.Errorf("expected only OfAdaptive, got OfEnabled with budget %d", got.OfEnabled.BudgetTokens)
				}
				return
			}
			if got.OfEnabled == nil {
				t.Fatal("expected OfEnabled to be non-nil")
			}
			if got.OfEnabled.BudgetTokens != tt.wantBudget {
				t.Errorf("BudgetTokens = %d, want %d", got.OfEnabled.BudgetTokens, tt.wantBudget)
			}
		})
	}
}

func TestThinkingConfigToAnthropic_ModelAware(t *testing.T) {
	tests := []struct {
		name         string
		cfg          *genai.ThinkingConfig
		model        anthropic.Model
		wantNil      bool // expect zero-value thinking
		wantAdaptive bool
		wantBudget   int64
		wantEffort   anthropic.OutputConfigEffort
		wantDisplay  string
	}{
		// Model-aware level → adaptive + effort
		{
			name:         "HIGH on Sonnet 4.6 → adaptive + high effort",
			cfg:          &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
			model:        anthropic.ModelClaudeSonnet4_6,
			wantAdaptive: true,
			wantEffort:   anthropic.OutputConfigEffortHigh,
			wantDisplay:  "omitted",
		},
		{
			name:         "HIGH with IncludeThoughts on Sonnet 4.6 returns summaries",
			cfg:          &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh, IncludeThoughts: true},
			model:        anthropic.ModelClaudeSonnet4_6,
			wantAdaptive: true,
			wantEffort:   anthropic.OutputConfigEffortHigh,
			wantDisplay:  "summarized",
		},
		{
			name:         "MEDIUM on Opus 4.6 → adaptive + medium effort",
			cfg:          &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMedium},
			model:        anthropic.ModelClaudeOpus4_6,
			wantAdaptive: true,
			wantEffort:   anthropic.OutputConfigEffortMedium,
		},
		{
			name:         "LOW on Opus 4.7 → adaptive + low effort",
			cfg:          &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow},
			model:        anthropic.ModelClaudeOpus4_7,
			wantAdaptive: true,
			wantEffort:   anthropic.OutputConfigEffortLow,
		},
		{
			name:         "HIGH on Mythos Preview → adaptive + high effort",
			cfg:          &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
			model:        anthropic.ModelClaudeMythosPreview,
			wantAdaptive: true,
			wantEffort:   anthropic.OutputConfigEffortHigh,
		},
		{
			name:       "unknown dated variant (no SDK constant yet) falls back to manual",
			cfg:        &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
			model:      anthropic.Model("claude-sonnet-4-6-20251201"), // hypothetical future date
			wantBudget: 10000,
		},

		// Non-adaptive models fall back to manual budget
		{
			name:        "HIGH on Haiku 4.5 → manual budget 10000 (no effort)",
			cfg:         &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
			model:       anthropic.ModelClaudeHaiku4_5,
			wantBudget:  10000,
			wantDisplay: "omitted",
		},
		{
			name:       "MEDIUM on Haiku 4.5 → manual budget 5000 (no effort)",
			cfg:        &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMedium},
			model:      anthropic.ModelClaudeHaiku4_5,
			wantBudget: 5000,
		},
		{
			name:       "LOW on Sonnet 4.5 (non-adaptive) → manual budget 1024",
			cfg:        &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow},
			model:      anthropic.ModelClaudeSonnet4_5,
			wantBudget: 1024,
		},

		// IncludeThoughts controls output visibility without enabling thinking.
		{
			name:         "IncludeThoughts on Sonnet 4.6 → adaptive default with summaries",
			cfg:          &genai.ThinkingConfig{IncludeThoughts: true},
			model:        anthropic.ModelClaudeSonnet4_6,
			wantAdaptive: true,
			wantDisplay:  "summarized",
		},
		{
			name:    "IncludeThoughts alone on Haiku 4.5 → off",
			cfg:     &genai.ThinkingConfig{IncludeThoughts: true},
			model:   anthropic.ModelClaudeHaiku4_5,
			wantNil: true,
		},
		{
			name:        "manual thinking with IncludeThoughts returns summaries",
			cfg:         &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh, IncludeThoughts: true},
			model:       anthropic.ModelClaudeHaiku4_5,
			wantBudget:  10000,
			wantDisplay: "summarized",
		},

		// Explicit overrides remain authoritative
		{
			name:        "ThinkingBudget overrides level on any model",
			cfg:         &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh, ThinkingBudget: int32Ptr(2048)},
			model:       anthropic.ModelClaudeSonnet4_6,
			wantBudget:  2048,
			wantDisplay: "omitted",
		},

		// Empty model name (legacy single-arg behaviour) prefers manual budget
		{
			name:       "HIGH with empty model → manual budget 10000 (legacy fallback)",
			cfg:        &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
			model:      "",
			wantBudget: 10000,
		},

		// Minimal level → off (Anthropic has no minimal tier; matches Gemini's
		// "no thinking for most queries" intent)
		{
			name:    "MINIMAL on adaptive-capable model → off",
			cfg:     &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMinimal},
			model:   anthropic.ModelClaudeSonnet4_6,
			wantNil: true,
		},
		{
			name:    "MINIMAL on non-adaptive model → off",
			cfg:     &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMinimal},
			model:   anthropic.ModelClaudeHaiku4_5,
			wantNil: true,
		},

		// Nil config: per-tier default (adaptive on capable, off on manual-only)
		{
			name:         "nil config on adaptive-capable model → adaptive default",
			cfg:          nil,
			model:        anthropic.ModelClaudeSonnet4_6,
			wantAdaptive: true,
			wantDisplay:  "omitted",
		},
		{
			name:    "nil config on manual-only model → off",
			cfg:     nil,
			model:   anthropic.ModelClaudeHaiku4_5,
			wantNil: true,
		},
		{
			name:         "empty cfg on adaptive-capable model → adaptive default",
			cfg:          &genai.ThinkingConfig{},
			model:        anthropic.ModelClaudeSonnet4_6,
			wantAdaptive: true,
			wantDisplay:  "omitted",
		},
		{
			name:    "empty cfg on manual-only model → off",
			cfg:     &genai.ThinkingConfig{},
			model:   anthropic.ModelClaudeHaiku4_5,
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := converters.ThinkingConfigToAnthropic(tt.cfg, tt.model)

			if tt.wantNil {
				if got.Thinking.OfEnabled != nil || got.Thinking.OfAdaptive != nil {
					t.Errorf("expected zero Thinking, got %+v", got.Thinking)
				}
				if got.Effort != "" {
					t.Errorf("expected zero Effort, got %q", got.Effort)
				}
				return
			}

			if tt.wantAdaptive {
				if got.Thinking.OfAdaptive == nil {
					t.Fatal("expected OfAdaptive non-nil")
				}
				if got.Thinking.OfEnabled != nil {
					t.Errorf("expected only OfAdaptive, also got OfEnabled with budget %d", got.Thinking.OfEnabled.BudgetTokens)
				}
			} else {
				if got.Thinking.OfEnabled == nil {
					t.Fatal("expected OfEnabled non-nil")
				}
				if got.Thinking.OfEnabled.BudgetTokens != tt.wantBudget {
					t.Errorf("BudgetTokens = %d, want %d", got.Thinking.OfEnabled.BudgetTokens, tt.wantBudget)
				}
			}

			if got.Effort != tt.wantEffort {
				t.Errorf("Effort = %q, want %q", got.Effort, tt.wantEffort)
			}

			var display string
			if got.Thinking.OfAdaptive != nil {
				display = string(got.Thinking.OfAdaptive.Display)
			} else if got.Thinking.OfEnabled != nil {
				display = string(got.Thinking.OfEnabled.Display)
			}
			if tt.wantDisplay != "" && display != tt.wantDisplay {
				t.Errorf("Display = %q, want %q", display, tt.wantDisplay)
			}
		})
	}
}

func TestToolConfigToToolChoice(t *testing.T) {
	tests := []struct {
		name      string
		config    *genai.ToolConfig
		wantAuto  bool
		wantAny   bool
		wantTool  string
		wantZero  bool
		wantError bool
	}{
		{
			name:     "nil config",
			config:   nil,
			wantZero: true,
		},
		{
			name:     "nil FunctionCallingConfig",
			config:   &genai.ToolConfig{},
			wantZero: true,
		},
		{
			name: "ModeNone",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeNone,
				},
			},
			wantZero: true,
		},
		{
			name: "ModeAuto",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAuto,
				},
			},
			wantAuto: true,
		},
		{
			name: "ModeAny",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			},
			wantAny: true,
		},
		{
			name: "ModeAny with single AllowedFunctionNames",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"get_weather"},
				},
			},
			wantTool: "get_weather",
		},
		{
			name: "multiple AllowedFunctionNames returns error",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"func1", "func2"},
				},
			},
			wantError: true,
		},
		{
			name: "unknown mode returns error",
			config: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigMode("UNKNOWN"),
				},
			},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := converters.ToolConfigToToolChoice(tt.config)

			if tt.wantError {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.wantZero {
				if result.OfAuto != nil || result.OfAny != nil || result.OfTool != nil {
					t.Error("expected zero-value result")
				}
				return
			}
			if tt.wantAuto && result.OfAuto == nil {
				t.Error("expected OfAuto to be set")
			}
			if tt.wantAny && result.OfAny == nil {
				t.Error("expected OfAny to be set")
			}
			if tt.wantTool != "" {
				if result.OfTool == nil {
					t.Fatal("expected OfTool to be set")
				}
				if result.OfTool.Name != tt.wantTool {
					t.Errorf("OfTool.Name = %q, want %q", result.OfTool.Name, tt.wantTool)
				}
			}
		})
	}
}

func TestUsageToMetadata_CacheTokens(t *testing.T) {
	t.Run("cache read tokens mapped to CachedContentTokenCount", func(t *testing.T) {
		usage := anthropic.Usage{
			InputTokens:          100,
			OutputTokens:         50,
			CacheReadInputTokens: 80,
		}
		got := converters.UsageToMetadata(usage)
		if got.CachedContentTokenCount != 80 {
			t.Errorf("CachedContentTokenCount = %d, want 80", got.CachedContentTokenCount)
		}
	})

	t.Run("zero cache tokens stays zero", func(t *testing.T) {
		usage := anthropic.Usage{InputTokens: 10, OutputTokens: 20}
		got := converters.UsageToMetadata(usage)
		if got.CachedContentTokenCount != 0 {
			t.Errorf("CachedContentTokenCount = %d, want 0", got.CachedContentTokenCount)
		}
	})

	t.Run("PromptTokenCount includes cached and uncached tokens", func(t *testing.T) {
		usage := anthropic.Usage{
			InputTokens:              100,
			OutputTokens:             50,
			CacheReadInputTokens:     80,
			CacheCreationInputTokens: 20,
		}
		got := converters.UsageToMetadata(usage)
		// PromptTokenCount = InputTokens + CacheReadInputTokens + CacheCreationInputTokens
		if got.PromptTokenCount != 200 {
			t.Errorf("PromptTokenCount = %d, want 200", got.PromptTokenCount)
		}
		if got.TotalTokenCount != 250 {
			t.Errorf("TotalTokenCount = %d, want 250", got.TotalTokenCount)
		}
	})
}

func TestDefaultMaxTokensForModel(t *testing.T) {
	cases := []struct {
		name  string
		model anthropic.Model
		want  int
	}{
		{"sonnet_4_6", "claude-sonnet-4-6", 128000},
		{"sonnet_4_6_dated", "claude-sonnet-4-6-20250101", 128000},
		{"sonnet_4_6_vertex_suffix", "claude-sonnet-4-6@20250101", 128000},
		{"opus_4_6", "claude-opus-4-6", 128000},
		{"opus_4_7", "claude-opus-4-7", 128000},
		{"opus_4_8", "claude-opus-4-8", 128000},
		{"opus_4_8_vertex_suffix", "claude-opus-4-8@20260101", 128000},
		{"haiku_4_5", "claude-haiku-4-5", 64000},
		{"haiku_4_5_dated", "claude-haiku-4-5-20251001", 64000},
		{"uppercase_normalised", "Claude-Sonnet-4-6@20250101", 128000},
		{"unknown_sonnet_4", "claude-sonnet-4-20250514", 64000},
		{"unknown_future_model", "claude-something-9", 64000},
		{"empty", "", 64000},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := converters.DefaultMaxTokensForModel(tc.model); got != tc.want {
				t.Errorf("DefaultMaxTokensForModel(%q) = %d, want %d", tc.model, got, tc.want)
			}
		})
	}
}

// truncatedToolMessage builds an accumulated message whose trailing tool_use
// block was cut off mid-input: a signed thinking block, a completed text
// block, a completed tool call with valid input, then a tool call whose
// accumulated input JSON is incomplete. This mirrors the state the streaming
// loop holds when Accumulate fails on message_stop — the SDK keeps the
// flattened ContentBlockUnion.Input current through input_json_delta events but
// never refreshes it via content_block_stop for the truncated block, so we set
// the invalid input on the exported field directly.
func truncatedToolMessage(t *testing.T) *anthropic.Message {
	t.Helper()
	const msgJSON = `{
		"content": [
			{"type": "thinking", "thinking": "weighing options", "signature": "c2ln"},
			{"type": "text", "text": "Saving the file now."},
			{"type": "tool_use", "id": "toolu_ok", "name": "lookup", "input": {"q": "done"}},
			{"type": "tool_use", "id": "toolu_cut", "name": "save_file", "input": {}}
		],
		"stop_reason": "max_tokens",
		"usage": {"input_tokens": 5, "output_tokens": 50}
	}`

	var msg anthropic.Message
	if err := msg.UnmarshalJSON([]byte(msgJSON)); err != nil {
		t.Fatalf("failed to unmarshal message: %v", err)
	}
	// Overwrite the trailing tool call's input with truncated (invalid) JSON.
	msg.Content[3].Input = json.RawMessage(`{"path": "/reports/summ`)
	return &msg
}

func TestSalvageInterruptedMessage_TruncatedToolCall(t *testing.T) {
	msg := truncatedToolMessage(t)

	got := converters.SalvageInterruptedMessage(msg)

	if got.ToolName != "save_file" {
		t.Errorf("ToolName = %q, want %q", got.ToolName, "save_file")
	}
	if got.ToolID != "toolu_cut" {
		t.Errorf("ToolID = %q, want %q", got.ToolID, "toolu_cut")
	}
	if got.PartialInput != `{"path": "/reports/summ` {
		t.Errorf("PartialInput = %q, want the truncated fragment", got.PartialInput)
	}

	// Salvaged parts, in stream order: thinking, text, the completed tool call.
	// The truncated tool call must NOT appear as a part.
	if len(got.Parts) != 3 {
		t.Fatalf("len(Parts) = %d, want 3 (thinking, text, completed tool call)", len(got.Parts))
	}
	if !got.Parts[0].Thought || got.Parts[0].Text != "weighing options" {
		t.Errorf("Parts[0] = %+v, want thinking part 'weighing options'", got.Parts[0])
	}
	if got.Parts[1].Text != "Saving the file now." {
		t.Errorf("Parts[1].Text = %q, want the completed text", got.Parts[1].Text)
	}
	fc := got.Parts[2].FunctionCall
	if fc == nil || fc.Name != "lookup" || fc.ID != "toolu_ok" {
		t.Errorf("Parts[2] = %+v, want completed tool call 'lookup'", got.Parts[2])
	}
	for _, p := range got.Parts {
		if p.FunctionCall != nil && p.FunctionCall.ID == "toolu_cut" {
			t.Error("truncated tool call must not appear in salvaged Parts")
		}
	}
}

func TestHasIncompleteToolInput(t *testing.T) {
	t.Run("truncated_tool_call", func(t *testing.T) {
		if !converters.HasIncompleteToolInput(truncatedToolMessage(t)) {
			t.Error("HasIncompleteToolInput = false, want true for a truncated tool call")
		}
	})

	t.Run("valid_message_mid_thinking", func(t *testing.T) {
		// max_tokens truncation that landed mid-thinking: the message is valid
		// (no tool block), so it is NOT an interruption we salvage — it
		// converts normally with a max_tokens finish reason.
		const msgJSON = `{
			"content": [{"type": "thinking", "thinking": "still reasoning", "signature": "c2ln"}],
			"stop_reason": "max_tokens",
			"usage": {"input_tokens": 5, "output_tokens": 50}
		}`
		var msg anthropic.Message
		if err := msg.UnmarshalJSON([]byte(msgJSON)); err != nil {
			t.Fatalf("failed to unmarshal message: %v", err)
		}

		if converters.HasIncompleteToolInput(&msg) {
			t.Error("HasIncompleteToolInput = true, want false for a valid mid-thinking message")
		}

		resp, err := converters.MessageToLLMResponse(&msg)
		if err != nil {
			t.Fatalf("MessageToLLMResponse() error = %v", err)
		}
		if resp.FinishReason != genai.FinishReasonMaxTokens {
			t.Errorf("FinishReason = %v, want FinishReasonMaxTokens", resp.FinishReason)
		}
	})

	t.Run("completed_tool_call", func(t *testing.T) {
		const msgJSON = `{
			"content": [{"type": "tool_use", "id": "toolu_ok", "name": "lookup", "input": {"q": "done"}}],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 5, "output_tokens": 50}
		}`
		var msg anthropic.Message
		if err := msg.UnmarshalJSON([]byte(msgJSON)); err != nil {
			t.Fatalf("failed to unmarshal message: %v", err)
		}
		if converters.HasIncompleteToolInput(&msg) {
			t.Error("HasIncompleteToolInput = true, want false for a completed tool call")
		}
	})

	t.Run("nil_message", func(t *testing.T) {
		if converters.HasIncompleteToolInput(nil) {
			t.Error("HasIncompleteToolInput(nil) = true, want false")
		}
	})
}

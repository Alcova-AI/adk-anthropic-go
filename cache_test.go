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
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"

	"github.com/Alcova-AI/adk-anthropic-go/converters"
)

func makeSystemParams(t *testing.T, text string) []anthropic.TextBlockParam {
	t.Helper()
	return converters.SystemInstructionToSystem(&genai.Content{
		Parts: []*genai.Part{{Text: text}},
	})
}

func makeToolParams(t *testing.T) []anthropic.ToolUnionParam {
	t.Helper()
	tools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{Name: "get_weather", Description: "Get the current weather"},
				{Name: "send_email", Description: "Send an email"},
			},
		},
	}
	return converters.ToolsToAnthropicTools(tools)
}

func makeMessages(t *testing.T, count int) []anthropic.MessageParam {
	t.Helper()
	msgs := make([]anthropic.MessageParam, 0, count)
	roles := []anthropic.MessageParamRole{
		anthropic.MessageParamRoleUser,
		anthropic.MessageParamRoleAssistant,
	}
	for i := range count {
		block := anthropic.NewTextBlock("message " + string(rune('A'+i)))
		msgs = append(msgs, anthropic.MessageParam{
			Role:    roles[i%2],
			Content: []anthropic.ContentBlockParamUnion{block},
		})
	}
	return msgs
}

// TestApplyCacheBreakpoints_Auto verifies top-level CacheControl is set from the Auto field.
func TestApplyCacheBreakpoints_Auto(t *testing.T) {
	t.Run("sets auto cache control", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 1),
		}
		cfg := &PromptCachingConfig{
			Auto: &CacheBreakpoint{},
		}
		applyCacheBreakpoints(params, cfg)

		if string(params.CacheControl.Type) != "ephemeral" {
			t.Errorf("CacheControl.Type = %q, want %q", params.CacheControl.Type, "ephemeral")
		}
	})

	t.Run("sets auto cache control with 1h TTL", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 1),
		}
		cfg := &PromptCachingConfig{
			Auto: &CacheBreakpoint{TTL: anthropic.CacheControlEphemeralTTLTTL1h},
		}
		applyCacheBreakpoints(params, cfg)

		if string(params.CacheControl.Type) != "ephemeral" {
			t.Errorf("CacheControl.Type = %q, want %q", params.CacheControl.Type, "ephemeral")
		}
		if params.CacheControl.TTL != anthropic.CacheControlEphemeralTTLTTL1h {
			t.Errorf("CacheControl.TTL = %q, want %q", params.CacheControl.TTL, anthropic.CacheControlEphemeralTTLTTL1h)
		}
	})

	t.Run("nil auto does not set cache control", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 1),
		}
		cfg := &PromptCachingConfig{
			Auto: nil,
		}
		applyCacheBreakpoints(params, cfg)

		if string(params.CacheControl.Type) == "ephemeral" {
			t.Error("expected CacheControl.Type to be empty when Auto is nil")
		}
	})
}

// TestApplyCacheBreakpoints_SystemInstruction verifies the last system block gets cache_control.
func TestApplyCacheBreakpoints_SystemInstruction(t *testing.T) {
	t.Run("last system block gets cache control", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 1),
			System:   makeSystemParams(t, "You are a helpful assistant."),
		}
		cfg := &PromptCachingConfig{
			SystemInstruction: &CacheBreakpoint{},
		}
		applyCacheBreakpoints(params, cfg)

		last := params.System[len(params.System)-1]
		if string(last.CacheControl.Type) != "ephemeral" {
			t.Errorf("last system block CacheControl.Type = %q, want %q", last.CacheControl.Type, "ephemeral")
		}
	})

	t.Run("nil system instruction does not set cache control", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 1),
			System:   makeSystemParams(t, "You are a helpful assistant."),
		}
		cfg := &PromptCachingConfig{
			SystemInstruction: nil,
		}
		applyCacheBreakpoints(params, cfg)

		last := params.System[len(params.System)-1]
		if string(last.CacheControl.Type) == "ephemeral" {
			t.Error("expected system block CacheControl.Type to be empty when SystemInstruction is nil")
		}
	})

	t.Run("empty system is no-op", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 1),
		}
		cfg := &PromptCachingConfig{
			SystemInstruction: &CacheBreakpoint{},
		}
		// Should not panic with empty System slice.
		applyCacheBreakpoints(params, cfg)
	})
}

// TestApplyCacheBreakpoints_Tools verifies the last tool definition gets cache_control.
func TestApplyCacheBreakpoints_Tools(t *testing.T) {
	t.Run("last tool gets cache control", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 1),
			Tools:    makeToolParams(t),
		}
		cfg := &PromptCachingConfig{
			Tools: &CacheBreakpoint{},
		}
		applyCacheBreakpoints(params, cfg)

		last := params.Tools[len(params.Tools)-1]
		if last.OfTool == nil {
			t.Fatal("expected last tool to be OfTool variant")
		}
		if string(last.OfTool.CacheControl.Type) != "ephemeral" {
			t.Errorf("last tool CacheControl.Type = %q, want %q", last.OfTool.CacheControl.Type, "ephemeral")
		}
	})

	t.Run("nil tools config does not set cache control", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 1),
			Tools:    makeToolParams(t),
		}
		cfg := &PromptCachingConfig{
			Tools: nil,
		}
		applyCacheBreakpoints(params, cfg)

		last := params.Tools[len(params.Tools)-1]
		if last.OfTool != nil && string(last.OfTool.CacheControl.Type) == "ephemeral" {
			t.Error("expected last tool CacheControl.Type to be empty when Tools is nil")
		}
	})

	t.Run("empty tools is no-op", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 1),
		}
		cfg := &PromptCachingConfig{
			Tools: &CacheBreakpoint{},
		}
		// Should not panic with empty Tools slice.
		applyCacheBreakpoints(params, cfg)
	})
}

// TestApplyCacheBreakpoints_ConversationHistory verifies the penultimate message's last block gets cache_control.
func TestApplyCacheBreakpoints_ConversationHistory(t *testing.T) {
	t.Run("penultimate message last block gets cache control", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 3), // [user, assistant, user] — penultimate is assistant
		}
		cfg := &PromptCachingConfig{
			ConversationHistory: &CacheBreakpoint{},
		}
		applyCacheBreakpoints(params, cfg)

		penultimate := params.Messages[len(params.Messages)-2]
		last := penultimate.Content[len(penultimate.Content)-1]
		ccPtr := last.GetCacheControl()
		if ccPtr == nil {
			t.Fatal("expected GetCacheControl() to return a non-nil pointer")
		}
		if string(ccPtr.Type) != "ephemeral" {
			t.Errorf("penultimate message last block CacheControl.Type = %q, want %q", ccPtr.Type, "ephemeral")
		}
	})

	t.Run("single message is no-op", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 1),
		}
		cfg := &PromptCachingConfig{
			ConversationHistory: &CacheBreakpoint{},
		}
		// Should not panic when there is only one message (no penultimate).
		applyCacheBreakpoints(params, cfg)
	})

	t.Run("nil conversation history does not set cache control", func(t *testing.T) {
		params := &anthropic.MessageNewParams{
			Messages: makeMessages(t, 3),
		}
		cfg := &PromptCachingConfig{
			ConversationHistory: nil,
		}
		applyCacheBreakpoints(params, cfg)

		penultimate := params.Messages[len(params.Messages)-2]
		last := penultimate.Content[len(penultimate.Content)-1]
		ccPtr := last.GetCacheControl()
		if ccPtr != nil && string(ccPtr.Type) == "ephemeral" {
			t.Error("expected penultimate message block CacheControl.Type to be empty when ConversationHistory is nil")
		}
	})
}

// TestApplyCacheBreakpoints_MixedTTLs verifies multiple breakpoints can be set independently.
func TestApplyCacheBreakpoints_MixedTTLs(t *testing.T) {
	params := &anthropic.MessageNewParams{
		Messages: makeMessages(t, 3),
		System:   makeSystemParams(t, "You are a helpful assistant."),
		Tools:    makeToolParams(t),
	}
	cfg := &PromptCachingConfig{
		Tools:             &CacheBreakpoint{TTL: anthropic.CacheControlEphemeralTTLTTL1h},
		SystemInstruction: &CacheBreakpoint{TTL: anthropic.CacheControlEphemeralTTLTTL1h},
		Auto:              &CacheBreakpoint{}, // defaults to 5m
		// ConversationHistory deliberately nil
	}
	applyCacheBreakpoints(params, cfg)

	// Tools — last tool should have 1h TTL.
	lastTool := params.Tools[len(params.Tools)-1]
	if lastTool.OfTool == nil {
		t.Fatal("expected last tool to be OfTool variant")
	}
	if string(lastTool.OfTool.CacheControl.Type) != "ephemeral" {
		t.Errorf("tool CacheControl.Type = %q, want %q", lastTool.OfTool.CacheControl.Type, "ephemeral")
	}
	if lastTool.OfTool.CacheControl.TTL != anthropic.CacheControlEphemeralTTLTTL1h {
		t.Errorf("tool CacheControl.TTL = %q, want %q", lastTool.OfTool.CacheControl.TTL, anthropic.CacheControlEphemeralTTLTTL1h)
	}

	// System — last block should have 1h TTL.
	lastSystem := params.System[len(params.System)-1]
	if string(lastSystem.CacheControl.Type) != "ephemeral" {
		t.Errorf("system CacheControl.Type = %q, want %q", lastSystem.CacheControl.Type, "ephemeral")
	}
	if lastSystem.CacheControl.TTL != anthropic.CacheControlEphemeralTTLTTL1h {
		t.Errorf("system CacheControl.TTL = %q, want %q", lastSystem.CacheControl.TTL, anthropic.CacheControlEphemeralTTLTTL1h)
	}

	// Auto — top-level cache control set.
	if string(params.CacheControl.Type) != "ephemeral" {
		t.Errorf("top-level CacheControl.Type = %q, want %q", params.CacheControl.Type, "ephemeral")
	}

	// ConversationHistory — nil, so penultimate message should not have cache control.
	penultimate := params.Messages[len(params.Messages)-2]
	last := penultimate.Content[len(penultimate.Content)-1]
	ccPtr := last.GetCacheControl()
	if ccPtr != nil && string(ccPtr.Type) == "ephemeral" {
		t.Error("expected penultimate message block to have no cache control when ConversationHistory is nil")
	}
}

// TestApplyCacheBreakpoints_EmptyConfig verifies that all-nil config is a no-op.
func TestApplyCacheBreakpoints_EmptyConfig(t *testing.T) {
	params := &anthropic.MessageNewParams{
		Messages: makeMessages(t, 3),
		System:   makeSystemParams(t, "You are a helpful assistant."),
		Tools:    makeToolParams(t),
	}
	cfg := &PromptCachingConfig{} // all nil

	applyCacheBreakpoints(params, cfg)

	// Nothing should be set.
	if string(params.CacheControl.Type) == "ephemeral" {
		t.Error("expected top-level CacheControl to be empty")
	}

	lastSystem := params.System[len(params.System)-1]
	if string(lastSystem.CacheControl.Type) == "ephemeral" {
		t.Error("expected system CacheControl to be empty")
	}

	lastTool := params.Tools[len(params.Tools)-1]
	if lastTool.OfTool != nil && string(lastTool.OfTool.CacheControl.Type) == "ephemeral" {
		t.Error("expected tool CacheControl to be empty")
	}

	penultimate := params.Messages[len(params.Messages)-2]
	last := penultimate.Content[len(penultimate.Content)-1]
	ccPtr := last.GetCacheControl()
	if ccPtr != nil && string(ccPtr.Type) == "ephemeral" {
		t.Error("expected penultimate message block CacheControl to be empty")
	}
}

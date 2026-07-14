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
	"errors"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// accumulateEvents drives anthropic.Message.Accumulate over a sequence of
// fabricated stream-event payloads, returning the accumulated message and the
// first Accumulate error (if any). This reproduces the exact state the
// streaming loop holds when generation is cut off at the max_tokens ceiling
// mid-tool-call: the trailing tool_use block never receives a
// content_block_stop, so message_stop's re-marshal of the invalid input JSON
// fails.
func accumulateEvents(t *testing.T, payloads []string) (*anthropic.Message, error) {
	t.Helper()
	msg := &anthropic.Message{}
	for i, p := range payloads {
		var ev anthropic.MessageStreamEventUnion
		if err := json.Unmarshal([]byte(p), &ev); err != nil {
			t.Fatalf("event %d unmarshal: %v", i, err)
		}
		if err := msg.Accumulate(ev); err != nil {
			return msg, err
		}
	}
	return msg, nil
}

// interruptedToolCallStream is a stream that completes a thinking block, a text
// block, and one tool call, then is cut off partway through a second tool
// call's input — the shape that surfaced the production failure.
var interruptedToolCallStream = []string{
	`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-6","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}`,
	// Completed thinking block (signature is base64 for "sig").
	`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"weighing options"}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"c2ln"}}`,
	`{"type":"content_block_stop","index":0}`,
	// Completed text block.
	`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
	`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Saving the file now."}}`,
	`{"type":"content_block_stop","index":1}`,
	// Completed tool call with valid input.
	`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_ok","name":"lookup","input":{}}}`,
	`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\": \"done\"}"}}`,
	`{"type":"content_block_stop","index":2}`,
	// Tool call cut off mid-input: no content_block_stop.
	`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_cut","name":"save_file","input":{}}}`,
	`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"path\": \"/reports/summ"}}`,
	`{"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":50}}`,
	`{"type":"message_stop"}`,
}

func TestNewOutputInterruptedError_FromAccumulateFailure(t *testing.T) {
	msg, accErr := accumulateEvents(t, interruptedToolCallStream)
	if accErr == nil {
		t.Fatal("expected Accumulate to fail on the truncated tool input, got nil")
	}

	err := newOutputInterruptedError(msg, accErr)

	if err.StopReason != anthropic.StopReasonMaxTokens {
		t.Errorf("StopReason = %q, want %q", err.StopReason, anthropic.StopReasonMaxTokens)
	}
	if err.ToolName != "save_file" {
		t.Errorf("ToolName = %q, want %q", err.ToolName, "save_file")
	}
	if err.ToolID != "toolu_cut" {
		t.Errorf("ToolID = %q, want %q", err.ToolID, "toolu_cut")
	}
	if err.PartialInput != `{"path": "/reports/summ` {
		t.Errorf("PartialInput = %q, want the truncated fragment", err.PartialInput)
	}
	if err.Cause != accErr {
		t.Errorf("Cause = %v, want the accumulate error", err.Cause)
	}

	// Salvaged parts, in stream order: thinking, text, the completed tool call.
	// The truncated tool call must not appear as a part.
	if len(err.Parts) != 3 {
		t.Fatalf("len(Parts) = %d, want 3 (thinking, text, completed tool call)", len(err.Parts))
	}
	if !err.Parts[0].Thought || err.Parts[0].Text != "weighing options" {
		t.Errorf("Parts[0] = %+v, want thinking part 'weighing options'", err.Parts[0])
	}
	if err.Parts[1].Text != "Saving the file now." {
		t.Errorf("Parts[1].Text = %q, want the completed text", err.Parts[1].Text)
	}
	if fc := err.Parts[2].FunctionCall; fc == nil || fc.Name != "lookup" || fc.ID != "toolu_ok" {
		t.Errorf("Parts[2] = %+v, want completed tool call 'lookup'", err.Parts[2])
	}

	// The typed error is discoverable via errors.As and unwraps to its cause.
	var interrupted *OutputInterruptedError
	if !errors.As(error(err), &interrupted) {
		t.Error("errors.As failed to match *OutputInterruptedError")
	}
	if !errors.Is(err, accErr) {
		t.Error("errors.Is(err, accErr) = false, want true via Unwrap")
	}
}

func TestNewOutputInterruptedError_MidThinkingNoToolFields(t *testing.T) {
	// Cut off mid-thinking, before any tool call: the message is valid JSON so
	// Accumulate succeeds, but if a caller constructs the typed error from this
	// state the tool fields must stay empty.
	payloads := []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-6","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"still reasoning"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"c2ln"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":50}}`,
		`{"type":"message_stop"}`,
	}

	msg, accErr := accumulateEvents(t, payloads)
	if accErr != nil {
		t.Fatalf("Accumulate should succeed on a valid mid-thinking message, got %v", accErr)
	}

	err := newOutputInterruptedError(msg, nil)

	if err.ToolName != "" || err.ToolID != "" || err.PartialInput != "" {
		t.Errorf("tool fields should be empty for a mid-thinking cut, got name=%q id=%q input=%q", err.ToolName, err.ToolID, err.PartialInput)
	}
	if err.StopReason != anthropic.StopReasonMaxTokens {
		t.Errorf("StopReason = %q, want %q", err.StopReason, anthropic.StopReasonMaxTokens)
	}
	if len(err.Parts) != 1 || !err.Parts[0].Thought {
		t.Errorf("Parts = %+v, want a single salvaged thinking part", err.Parts)
	}
}

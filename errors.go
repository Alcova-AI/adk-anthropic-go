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
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

// OutputInterruptedError reports that the model's output was cut off before
// the response could complete — typically because generation hit the
// max_tokens ceiling partway through a tool call, leaving its input JSON
// incomplete. The adapter detects the interruption and preserves everything
// that survived; it makes no decision about how to continue. Callers own
// that policy: inspect the salvaged content with errors.As and decide
// whether to discard it, persist it into history with a notice, or resume
// the agent with a redirect.
type OutputInterruptedError struct {
	// StopReason is the Anthropic stop reason for the interrupted response
	// (typically anthropic.StopReasonMaxTokens). Empty if the stream ended
	// before a message_delta carried one.
	StopReason anthropic.StopReason

	// Parts holds the salvaged content that survived intact, converted to
	// genai parts in stream order: thinking parts (with signatures) and any
	// completed text or tool-call parts. The truncated tool call is NOT
	// included here — it is exposed as data via ToolName/PartialInput.
	Parts []*genai.Part

	// ToolName is the name of the tool call whose input was cut off, if the
	// interruption landed inside a tool call. Empty when the cut landed
	// elsewhere (e.g. mid-thinking).
	ToolName string

	// ToolID is the provider-assigned id of the truncated tool call, if any.
	ToolID string

	// PartialInput is the raw, incomplete input JSON accumulated for the
	// truncated tool call before the cut. It is not valid JSON.
	PartialInput string

	// Cause is the underlying error that surfaced the interruption, if any
	// (e.g. the SDK accumulator's marshal failure). May be nil when the
	// interruption was detected directly from the stop reason.
	Cause error
}

func (e *OutputInterruptedError) Error() string {
	switch {
	case e.ToolName != "":
		return fmt.Sprintf("model output interrupted (stop_reason=%s): tool call %q truncated after %d bytes of input", e.StopReason, e.ToolName, len(e.PartialInput))
	default:
		return fmt.Sprintf("model output interrupted (stop_reason=%s)", e.StopReason)
	}
}

func (e *OutputInterruptedError) Unwrap() error { return e.Cause }

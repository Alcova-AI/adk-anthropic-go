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
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/adk/v2/model"
)

// Exact shape Vertex delivers when overload arrives after the 200 OK —
// mirrors anthropic-sdk-go's error_type_test.go fixture.
const overloadedSSE = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"

const apiErrorSSE = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"Internal server error\"}}\n\n"

const messagePrefixSSE = "event: message_start\n" +
	"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-haiku-4-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n" +
	"event: content_block_start\n" +
	"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"

const successSSE = messagePrefixSSE +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: message_delta\n" +
	"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

// Overload after content has already streamed — must NOT retry.
const partialThenOverloadSSE = messagePrefixSSE +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n" +
	overloadedSSE

// Overload after a thinking delta has streamed — the thinking branch sets the
// yielded guard too, so this must not retry either.
const thinkingThenOverloadSSE = "event: message_start\n" +
	"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-haiku-4-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n" +
	"event: content_block_start\n" +
	"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n" +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"weighing options\"}}\n\n" +
	overloadedSSE

// newSSEServer answers the i-th request with bodies[i] (repeating the last
// body once exhausted) and counts requests.
func newSSEServer(t *testing.T, bodies ...string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i := int(requests.Add(1)) - 1
		if i >= len(bodies) {
			i = len(bodies) - 1
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, bodies[i])
	}))
	t.Cleanup(srv.Close)
	return srv, &requests
}

// newStreamTestModel builds a model against baseURL through the real SDK
// client (so mid-stream error events decode into *anthropic.Error exactly as
// in production) and stubs retrySleep to record delays without sleeping.
func newStreamTestModel(t *testing.T, baseURL string) (*anthropicModel, *[]time.Duration) {
	t.Helper()
	llm, err := NewModel(t.Context(), "claude-haiku-4-5", &Config{
		APIKey:  "test-key",
		Variant: VariantAnthropicAPI,
		BaseURL: baseURL,
	})
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	m := llm.(*anthropicModel)
	sleeps := &[]time.Duration{}
	m.retrySleep = func(_ context.Context, d time.Duration) error {
		*sleeps = append(*sleeps, d)
		return nil
	}
	return m, sleeps
}

type streamPair struct {
	resp *model.LLMResponse
	err  error
}

// collect drains a streaming GenerateContent call into its yielded pairs.
func collect(ctx context.Context, m *anthropicModel) []streamPair {
	var pairs []streamPair
	for resp, err := range m.GenerateContent(ctx, &model.LLMRequest{}, true) {
		pairs = append(pairs, streamPair{resp, err})
	}
	return pairs
}

// sseFromPayloads frames raw stream-event payloads (as used by errors_test.go
// fixtures) into SSE wire format, deriving each event: line from the
// payload's "type" field.
func sseFromPayloads(t *testing.T, payloads []string) string {
	t.Helper()
	var b strings.Builder
	for _, p := range payloads {
		var ev struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(p), &ev); err != nil {
			t.Fatalf("payload unmarshal: %v", err)
		}
		b.WriteString("event: " + ev.Type + "\ndata: " + p + "\n\n")
	}
	return b.String()
}

func TestGenerateStream_RetriesOverloadThenSucceeds(t *testing.T) {
	for _, tc := range []struct {
		name      string
		failCount int
	}{
		{"no_overload", 0},
		{"one_overload", 1},
		{"two_overloads", 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bodies := make([]string, 0, tc.failCount+1)
			for range tc.failCount {
				bodies = append(bodies, overloadedSSE)
			}
			bodies = append(bodies, successSSE)
			srv, requests := newSSEServer(t, bodies...)
			m, sleeps := newStreamTestModel(t, srv.URL)

			pairs := collect(t.Context(), m)

			for _, p := range pairs {
				if p.err != nil {
					t.Fatalf("unexpected error after %d overload(s): %v", tc.failCount, p.err)
				}
			}
			if len(pairs) != 2 {
				t.Fatalf("len(pairs) = %d, want 2 (partial + final)", len(pairs))
			}
			if !pairs[0].resp.Partial || pairs[0].resp.Content.Parts[0].Text != "Hello" {
				t.Errorf("pairs[0] = %+v, want partial 'Hello' delta", pairs[0].resp)
			}
			if !pairs[1].resp.TurnComplete {
				t.Errorf("final response TurnComplete = false, want true")
			}
			if got := int(requests.Load()); got != tc.failCount+1 {
				t.Errorf("requests = %d, want %d", got, tc.failCount+1)
			}
			if len(*sleeps) != tc.failCount {
				t.Fatalf("sleeps = %d, want %d", len(*sleeps), tc.failCount)
			}
			for i, d := range *sleeps {
				base := streamRetryBaseDelay << i
				if d < base || d >= base+base/4 {
					t.Errorf("sleeps[%d] = %v, want in [%v, %v)", i, d, base, base+base/4)
				}
			}
		})
	}
}

func TestGenerateStream_ExhaustsRetriesOnPersistentOverload(t *testing.T) {
	srv, requests := newSSEServer(t, overloadedSSE)
	m, sleeps := newStreamTestModel(t, srv.URL)

	pairs := collect(t.Context(), m)

	if len(pairs) != 1 || pairs[0].resp != nil {
		t.Fatalf("pairs = %+v, want exactly one error pair", pairs)
	}
	err := pairs[0].err
	// The wrap must stay identical to the pre-retry behaviour so caller-side
	// handling and error grouping are unchanged on exhaustion.
	if !strings.HasPrefix(err.Error(), "stream error: ") {
		t.Errorf("err = %q, want 'stream error: ' prefix", err)
	}
	var apierr *anthropic.Error
	if !errors.As(err, &apierr) || apierr.Type() != anthropic.ErrorTypeOverloadedError {
		t.Errorf("errors.As detection of overloaded_error failed through the wrap: %v", err)
	}
	if got := int(requests.Load()); got != streamMaxAttempts {
		t.Errorf("requests = %d, want %d", got, streamMaxAttempts)
	}
	if len(*sleeps) != streamMaxAttempts-1 {
		t.Errorf("sleeps = %d, want %d", len(*sleeps), streamMaxAttempts-1)
	}
}

func TestGenerateStream_NoRetryAfterPartialOutput(t *testing.T) {
	srv, requests := newSSEServer(t, partialThenOverloadSSE)
	m, sleeps := newStreamTestModel(t, srv.URL)

	pairs := collect(t.Context(), m)

	if len(pairs) != 2 {
		t.Fatalf("len(pairs) = %d, want 2 (partial then error)", len(pairs))
	}
	if pairs[0].err != nil || !pairs[0].resp.Partial || pairs[0].resp.Content.Parts[0].Text != "Hi" {
		t.Errorf("pairs[0] = %+v, want partial 'Hi' delta", pairs[0])
	}
	var apierr *anthropic.Error
	if !errors.As(pairs[1].err, &apierr) || apierr.Type() != anthropic.ErrorTypeOverloadedError {
		t.Errorf("pairs[1].err = %v, want wrapped overloaded_error", pairs[1].err)
	}
	if got := int(requests.Load()); got != 1 {
		t.Errorf("requests = %d, want 1 — an overload after yielded content must not retry", got)
	}
	if len(*sleeps) != 0 {
		t.Errorf("sleeps = %d, want 0", len(*sleeps))
	}
}

func TestGenerateStream_NoRetryAfterThinkingOutput(t *testing.T) {
	srv, requests := newSSEServer(t, thinkingThenOverloadSSE)
	m, sleeps := newStreamTestModel(t, srv.URL)

	pairs := collect(t.Context(), m)

	if len(pairs) != 2 {
		t.Fatalf("len(pairs) = %d, want 2 (thinking partial then error)", len(pairs))
	}
	if pairs[0].err != nil || !pairs[0].resp.Partial || !pairs[0].resp.Content.Parts[0].Thought {
		t.Errorf("pairs[0] = %+v, want partial thinking delta", pairs[0])
	}
	var apierr *anthropic.Error
	if !errors.As(pairs[1].err, &apierr) || apierr.Type() != anthropic.ErrorTypeOverloadedError {
		t.Errorf("pairs[1].err = %v, want wrapped overloaded_error", pairs[1].err)
	}
	if got := int(requests.Load()); got != 1 {
		t.Errorf("requests = %d, want 1 — an overload after yielded thinking must not retry", got)
	}
	if len(*sleeps) != 0 {
		t.Errorf("sleeps = %d, want 0", len(*sleeps))
	}
}

func TestGenerateStream_NoRetryOnNonOverloadedError(t *testing.T) {
	srv, requests := newSSEServer(t, apiErrorSSE)
	m, sleeps := newStreamTestModel(t, srv.URL)

	pairs := collect(t.Context(), m)

	if len(pairs) != 1 || pairs[0].resp != nil {
		t.Fatalf("pairs = %+v, want exactly one error pair", pairs)
	}
	var apierr *anthropic.Error
	if !errors.As(pairs[0].err, &apierr) || apierr.Type() != anthropic.ErrorTypeAPIError {
		t.Errorf("err = %v, want wrapped api_error", pairs[0].err)
	}
	if got := int(requests.Load()); got != 1 {
		t.Errorf("requests = %d, want 1 — api_error must not retry", got)
	}
	if len(*sleeps) != 0 {
		t.Errorf("sleeps = %d, want 0", len(*sleeps))
	}
}

func TestGenerateStream_AbortsWhenBackoffCancelled(t *testing.T) {
	srv, requests := newSSEServer(t, overloadedSSE)
	m, _ := newStreamTestModel(t, srv.URL)
	m.retrySleep = func(context.Context, time.Duration) error {
		return context.Canceled
	}

	pairs := collect(t.Context(), m)

	if len(pairs) != 1 || pairs[0].resp != nil {
		t.Fatalf("pairs = %+v, want exactly one error pair", pairs)
	}
	// Both identities must survive the wrap: the overload for detection, the
	// cancellation for callers that filter caller-initiated aborts.
	var apierr *anthropic.Error
	if !errors.As(pairs[0].err, &apierr) || apierr.Type() != anthropic.ErrorTypeOverloadedError {
		t.Errorf("err = %v, want the overload detectable via errors.As", pairs[0].err)
	}
	if !errors.Is(pairs[0].err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled detectable via errors.Is", pairs[0].err)
	}
	if got := int(requests.Load()); got != 1 {
		t.Errorf("requests = %d, want 1 — no attempt after a cancelled backoff", got)
	}
}

func TestGenerateStream_InterruptedOutputIsNotRetried(t *testing.T) {
	srv, requests := newSSEServer(t, sseFromPayloads(t, interruptedToolCallStream))
	m, sleeps := newStreamTestModel(t, srv.URL)

	pairs := collect(t.Context(), m)

	// The fixture streams a thinking delta and a text delta before the
	// truncated tool call, so those arrive as partials ahead of the error.
	if len(pairs) != 3 {
		t.Fatalf("len(pairs) = %d, want 3 (thinking, text, interruption)", len(pairs))
	}
	for _, p := range pairs[:2] {
		if p.err != nil || !p.resp.Partial {
			t.Fatalf("pair = %+v, want a partial delta", p)
		}
	}
	var interrupted *OutputInterruptedError
	if !errors.As(pairs[2].err, &interrupted) {
		t.Fatalf("err = %v (%T), want *OutputInterruptedError", pairs[2].err, pairs[2].err)
	}
	if got := int(requests.Load()); got != 1 {
		t.Errorf("requests = %d, want 1 — interruptions must never retry", got)
	}
	if len(*sleeps) != 0 {
		t.Errorf("sleeps = %d, want 0", len(*sleeps))
	}
}

func TestSleepWithContext(t *testing.T) {
	t.Run("cancel_aborts_promptly", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		go func() {
			time.Sleep(20 * time.Millisecond)
			cancel()
		}()
		start := time.Now()
		err := sleepWithContext(ctx, 5*time.Second)
		if elapsed := time.Since(start); elapsed >= time.Second {
			t.Errorf("sleepWithContext took %v, want prompt abort", elapsed)
		}
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	})

	t.Run("elapses_normally", func(t *testing.T) {
		if err := sleepWithContext(t.Context(), 5*time.Millisecond); err != nil {
			t.Errorf("err = %v, want nil", err)
		}
	})
}

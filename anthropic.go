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
	"errors"
	"fmt"
	"iter"
	"math/rand/v2"
	"net/http"
	"os"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"google.golang.org/genai"

	"github.com/Alcova-AI/adk-anthropic-go/v2/converters"
	"google.golang.org/adk/v2/model"
)

const defaultMaxTokens = 16384

// Mid-stream overload retry policy. Vertex AI can accept a streaming request
// (HTTP 200 at the header level) and then deliver overloaded_error as an SSE
// error event; the SDK's HTTP-level retries never see it because the request
// already "succeeded". Three attempts mirrors the SDK's own HTTP-level default
// (MaxRetries=2) that covers the non-streaming path.
const (
	streamMaxAttempts    = 3
	streamRetryBaseDelay = time.Second
)

type anthropicModel struct {
	client anthropic.Client
	// canonicalModel is exposed through Name and controls local capabilities.
	canonicalModel anthropic.Model
	// requestModel is the model identifier sent to the API.
	requestModel     anthropic.Model
	variant          string
	defaultMaxTokens int
	promptCaching    *PromptCachingConfig
	// gatewayEffortTranslation permits effort-only requests for non-Claude
	// models on a gateway that explicitly supports that translation.
	gatewayEffortTranslation bool

	// retrySleep waits between mid-stream overload retries. Overridable so
	// tests can drop the delay; production always gets sleepWithContext.
	retrySleep func(ctx context.Context, d time.Duration) error
}

// NewModel returns [model.LLM], backed by an Anthropic-compatible API.
//
// It creates an Anthropic client based on the provided configuration.
// If Variant is not specified, it checks the ANTHROPIC_USE_VERTEX environment variable.
//
// For direct Anthropic API, set APIKey in the config or the ANTHROPIC_API_KEY
// environment variable.
//
// For Vertex AI, set VertexProjectID and VertexLocation in the config or use
// GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.
//
// For an Anthropic-compatible API, set BaseURL and APIKey. Set RequestModel
// when the API expects a different model identifier from the canonical name.
// Name and capability checks continue to use the canonical model name.
func NewModel(ctx context.Context, modelName anthropic.Model, cfg *Config) (model.LLM, error) {
	return NewModelWithOptions(ctx, modelName, cfg)
}

// NewModelWithOptions returns [model.LLM] with optional, explicitly declared
// capabilities for Anthropic-compatible endpoints. NewModel remains available
// as the default constructor with no additional gateway capabilities.
func NewModelWithOptions(ctx context.Context, modelName anthropic.Model, cfg *Config, options ...ModelOption) (model.LLM, error) {
	if cfg == nil {
		cfg = &Config{}
	}
	modelOpts := &modelOptions{}
	for _, option := range options {
		option(modelOpts)
	}

	variant := cfg.Variant
	if variant == "" {
		variant = GetVariant()
	}

	var client anthropic.Client

	switch variant {
	case VariantVertexAI:
		projectID := cfg.VertexProjectID
		if projectID == "" {
			projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
		}
		if projectID == "" {
			return nil, fmt.Errorf("VertexProjectID is required for Vertex AI (set GOOGLE_CLOUD_PROJECT)")
		}

		location := cfg.VertexLocation
		if location == "" {
			location = os.Getenv("GOOGLE_CLOUD_LOCATION")
		}
		if location == "" {
			return nil, fmt.Errorf("VertexLocation is required for Vertex AI (set GOOGLE_CLOUD_LOCATION)")
		}

		client = newVertexClient(ctx, cfg)
	case VariantAnthropicAPI:
		client = newAPIClient(cfg)
	default:
		return nil, fmt.Errorf("unsupported Anthropic variant %q", variant)
	}

	// max_tokens precedence: a per-request GenerateContentConfig.MaxOutputTokens
	// override wins in convertRequest; a deployment-level Config.DefaultMaxTokens
	// wins here; otherwise use a conservative default that remains valid for
	// both streaming and non-streaming requests.
	maxTokens := cfg.DefaultMaxTokens
	if maxTokens == 0 {
		maxTokens = defaultMaxTokens
	}

	requestModel := cfg.RequestModel
	if requestModel == "" {
		requestModel = modelName
	}

	return &anthropicModel{
		client:                   client,
		canonicalModel:           modelName,
		requestModel:             requestModel,
		variant:                  variant,
		defaultMaxTokens:         maxTokens,
		promptCaching:            cfg.PromptCaching,
		gatewayEffortTranslation: modelOpts.gatewayEffortTranslation,
		retrySleep:               sleepWithContext,
	}, nil
}

// newAPIClient creates a client for the direct Anthropic API.
func newAPIClient(cfg *Config) anthropic.Client {
	opts := []option.RequestOption{}

	apiKey := cfg.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}

	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}

	return anthropic.NewClient(opts...)
}

// newVertexClient creates a client for Anthropic via Vertex AI.
// Note: The caller must validate that projectID and region are set before calling this.
func newVertexClient(ctx context.Context, cfg *Config) anthropic.Client {
	projectID := cfg.VertexProjectID
	if projectID == "" {
		projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
	}

	location := cfg.VertexLocation
	if location == "" {
		location = os.Getenv("GOOGLE_CLOUD_LOCATION")
	}

	return anthropic.NewClient(
		vertex.WithGoogleAuth(ctx, location, projectID),
	)
}

// Name returns the model name.
func (m *anthropicModel) Name() string {
	return string(m.canonicalModel)
}

func (m *anthropicModel) wireModel() anthropic.Model {
	if m.requestModel != "" {
		return m.requestModel
	}
	return m.canonicalModel
}

// GenerateContent calls the Anthropic model.
func (m *anthropicModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.maybeAppendUserContent(req)

	if stream {
		return m.generateStream(ctx, req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

// generate calls the model synchronously.
func (m *anthropicModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	params, err := m.convertRequest(req)
	if err != nil {
		return nil, fmt.Errorf("failed to convert request: %w", err)
	}

	msg, err := m.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}

	resp, err := converters.MessageToLLMResponse(msg)
	if err != nil {
		return nil, fmt.Errorf("failed to convert response: %w", err)
	}
	filterResponseThoughts(resp, requestIncludesThoughts(req))

	return resp, nil
}

func requestIncludesThoughts(req *model.LLMRequest) bool {
	return req != nil && req.Config != nil && req.Config.ThinkingConfig != nil &&
		req.Config.ThinkingConfig.IncludeThoughts
}

// filterResponseThoughts enforces IncludeThoughts on providers that do not.
// Vercel's Anthropic-compatible endpoint can return unsigned Gemini reasoning
// blocks even when thinking.display is "omitted". Their text is display-only
// and safe to discard. Signed thinking and redacted metadata are provider state
// that must be replayed unchanged on later requests, so they are always kept.
// If an unsigned thought is the complete response, retain an empty thought
// marker so ADK continues the thought-only turn instead of treating an empty
// response as the final answer.
func filterResponseThoughts(resp *model.LLMResponse, includeThoughts bool) {
	if includeThoughts || resp == nil || resp.Content == nil {
		return
	}
	resp.Content.Parts = filterThoughtParts(resp.Content.Parts)
}

func filterThoughtParts(parts []*genai.Part) []*genai.Part {
	filtered := make([]*genai.Part, 0, len(parts))
	removedUnsignedThought := false
	for _, part := range parts {
		if part == nil || !part.Thought || len(part.ThoughtSignature) > 0 || len(part.PartMetadata) > 0 {
			filtered = append(filtered, part)
			continue
		}
		removedUnsignedThought = true
	}
	if len(filtered) == 0 && removedUnsignedThought {
		return []*genai.Part{{Thought: true}}
	}
	return filtered
}

// generateStream returns a stream of responses from the model.
func (m *anthropicModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		params, err := m.convertRequest(req)
		if err != nil {
			yield(nil, fmt.Errorf("failed to convert request: %w", err))
			return
		}

		// Retry mid-stream overloads, but only while nothing has been yielded:
		// once a delta has reached the consumer, a retry would replay content
		// it already has, so streamOnce handles those failures terminally and
		// returns nil. This is a deliberate, narrow exception to the adapter's
		// "no continuation decisions" rule — a pre-content retry is invisible
		// to callers and carries no continuation semantics.
		includeThoughts := requestIncludesThoughts(req)
		for attempt := 1; ; attempt++ {
			streamErr := m.streamOnce(ctx, params, includeThoughts, yield)
			if streamErr == nil {
				return
			}
			if attempt == streamMaxAttempts || !isOverloadedStreamError(streamErr) {
				// Same wrap as before the retry existed, so caller-side
				// handling and error grouping stay identical on exhaustion.
				yield(nil, fmt.Errorf("stream error: %w", streamErr))
				return
			}
			if err := m.retrySleep(ctx, streamRetryDelay(attempt)); err != nil {
				// Cancelled during backoff: wrap the overload and the
				// cancellation together, so callers that filter caller
				// cancellations (errors.Is) and callers that detect overload
				// (errors.As) both still match.
				yield(nil, fmt.Errorf("stream error: %w (retry aborted: %w)", streamErr, err))
				return
			}
		}
	}
}

// streamOnce runs a single streaming attempt, yielding partial deltas and the
// final response. It returns a non-nil error only when the stream failed
// before any partial content reached the consumer — the one window in which
// generateStream may safely retry without duplicating output. Every other
// outcome (success, consumer stop, post-content failure, interruption) is
// fully handled here and signalled by a nil return.
func (m *anthropicModel) streamOnce(
	ctx context.Context,
	params anthropic.MessageNewParams,
	includeThoughts bool,
	yield func(*model.LLMResponse, error) bool,
) error {
	stream := m.client.Messages.NewStreaming(ctx, params)
	// Next() leaves the response body open on the SSE error-event and
	// consumer-stop paths; without this, each retried attempt would leak its
	// predecessor's connection. Close is nil-safe when the request itself
	// failed.
	defer stream.Close()

	message := anthropic.Message{}

	// True once any delta has been yielded — the point of no return for
	// retries.
	yielded := false

	for stream.Next() {
		event := stream.Current()

		// Accumulate the message. A failure here is almost always the
		// SDK's message_stop re-marshal choking on a tool call whose input
		// JSON was truncated at the max_tokens ceiling. Surface that as a
		// typed OutputInterruptedError carrying whatever survived; any other
		// accumulation failure keeps its original error so it isn't
		// misdiagnosed as an interruption.
		if err := message.Accumulate(event); err != nil {
			yield(nil, classifyAccumulateError(&message, err, includeThoughts))
			return nil
		}
		mergeMessageDeltaUsage(&message, event)

		// Handle different event types for streaming
		switch ev := event.AsAny().(type) {
		case anthropic.ContentBlockDeltaEvent:
			// Handle text deltas
			switch delta := ev.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				yielded = true
				resp := converters.StreamDeltaToPartialResponse(delta.Text)
				if !yield(resp, nil) {
					return nil
				}
			case anthropic.ThinkingDelta:
				if !includeThoughts {
					// The signature arrives in a later delta and is retained from the
					// final accumulated block. Do not expose provisional reasoning text.
					continue
				}
				yielded = true
				resp := converters.StreamThinkingDeltaToPartialResponse(delta.Thinking)
				if !yield(resp, nil) {
					return nil
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		if !yielded {
			// Pre-content failure: generateStream decides whether to retry.
			return err
		}
		yield(nil, fmt.Errorf("stream error: %w", err))
		return nil
	}

	// Belt-and-braces: the stream can complete without Accumulate erroring
	// yet still carry a tool call truncated at the ceiling (invalid input
	// JSON). Converting that normally would fail or emit a broken tool
	// call, so report the interruption instead. A max_tokens stop with an
	// otherwise-valid message (e.g. truncated mid-thinking) is NOT an
	// interruption for our purposes — it converts normally below and the
	// harness reacts off the mapped max_tokens FinishReason.
	if message.StopReason == anthropic.StopReasonMaxTokens && converters.HasIncompleteToolInput(&message) {
		yield(nil, newOutputInterruptedError(&message, nil, includeThoughts))
		return nil
	}

	// Yield the final complete response
	finalResp, err := converters.MessageToLLMResponse(&message)
	if err != nil {
		yield(nil, fmt.Errorf("failed to convert stream response: %w", err))
		return nil
	}
	filterResponseThoughts(finalResp, includeThoughts)
	finalResp.TurnComplete = true
	yield(finalResp, nil)
	return nil
}

// mergeMessageDeltaUsage preserves cumulative usage fields that compatible
// gateways can report on message_delta. The Anthropic SDK accumulator only
// copies output_tokens from that event because Anthropic normally reports the
// input fields on message_start. Gateways such as Vercel can report their final
// input count on message_delta instead. Taking the maximum keeps Anthropic and
// Vertex behaviour unchanged while accepting later cumulative totals.
func mergeMessageDeltaUsage(message *anthropic.Message, event anthropic.MessageStreamEventUnion) {
	if message == nil {
		return
	}
	delta, ok := event.AsAny().(anthropic.MessageDeltaEvent)
	if !ok {
		return
	}
	if delta.Usage.JSON.InputTokens.Valid() {
		message.Usage.InputTokens = max(message.Usage.InputTokens, delta.Usage.InputTokens)
	}
	if delta.Usage.JSON.CacheReadInputTokens.Valid() {
		message.Usage.CacheReadInputTokens = max(message.Usage.CacheReadInputTokens, delta.Usage.CacheReadInputTokens)
	}
	if delta.Usage.JSON.CacheCreationInputTokens.Valid() {
		message.Usage.CacheCreationInputTokens = max(message.Usage.CacheCreationInputTokens, delta.Usage.CacheCreationInputTokens)
	}
	if delta.Usage.JSON.OutputTokens.Valid() {
		message.Usage.OutputTokens = max(message.Usage.OutputTokens, delta.Usage.OutputTokens)
	}
}

// isOverloadedStreamError reports whether err is Anthropic's overloaded_error
// delivered mid-stream: an SSE error event arriving after the request already
// succeeded at the HTTP level, so the *anthropic.Error carries StatusCode 200.
// The status gate keeps this retry scoped to that gap — a direct-API 529 has
// already spent the SDK's own HTTP-level retries and is not retried again.
func isOverloadedStreamError(err error) bool {
	var apierr *anthropic.Error
	return errors.As(err, &apierr) &&
		apierr.Type() == anthropic.ErrorTypeOverloadedError &&
		apierr.StatusCode == http.StatusOK
}

// streamRetryDelay returns the backoff before retrying the given (1-based)
// failed attempt: ~1s then ~2s, each with up to 25% random jitter so
// concurrent streams hitting the same overloaded shard don't retry in
// lockstep.
func streamRetryDelay(attempt int) time.Duration {
	base := streamRetryBaseDelay << (attempt - 1)
	return base + rand.N(base/4)
}

// sleepWithContext blocks for d or until ctx is done, whichever comes first,
// returning ctx's error when cancelled so retries abort promptly instead of
// sleeping through a dead request.
func sleepWithContext(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// convertRequest converts an LLMRequest to Anthropic MessageNewParams.
func (m *anthropicModel) convertRequest(req *model.LLMRequest) (anthropic.MessageNewParams, error) {
	messages, err := converters.ContentsToMessages(req.Contents)
	if err != nil {
		return anthropic.MessageNewParams{}, fmt.Errorf("failed to convert contents: %w", err)
	}

	params := anthropic.MessageNewParams{
		Model:     m.wireModel(),
		Messages:  messages,
		MaxTokens: int64(m.defaultMaxTokens),
	}

	if req.Config != nil {
		// System instruction
		if req.Config.SystemInstruction != nil {
			params.System = converters.SystemInstructionToSystem(req.Config.SystemInstruction)
		}

		// Generation parameters
		if req.Config.Temperature != nil {
			params.Temperature = anthropic.Float(float64(*req.Config.Temperature))
		}
		if req.Config.TopP != nil {
			params.TopP = anthropic.Float(float64(*req.Config.TopP))
		}
		if req.Config.TopK != nil {
			params.TopK = anthropic.Int(int64(*req.Config.TopK))
		}
		if len(req.Config.StopSequences) > 0 {
			params.StopSequences = req.Config.StopSequences
		}
		if req.Config.MaxOutputTokens > 0 {
			params.MaxTokens = int64(req.Config.MaxOutputTokens)
		}

		// Tools
		if len(req.Config.Tools) > 0 {
			params.Tools = converters.ToolsToAnthropicTools(req.Config.Tools)
		}

		// Tool choice from ToolConfig
		if req.Config.ToolConfig != nil {
			toolChoice, err := converters.ToolConfigToToolChoice(req.Config.ToolConfig)
			if err != nil {
				return anthropic.MessageNewParams{}, err
			}
			params.ToolChoice = toolChoice
		}

		// Structured output format. Anthropic structured outputs are GA on both
		// the direct API and Vertex AI (output_config.format with a json_schema,
		// no beta header), so the same path serves both variants.
		if req.Config.ResponseSchema != nil {
			schemaMap, err := converters.SchemaToStructuredOutputMap(req.Config.ResponseSchema)
			if err != nil {
				return anthropic.MessageNewParams{}, fmt.Errorf("failed to transform response schema: %w", err)
			}
			params.OutputConfig = anthropic.OutputConfigParam{
				Format: anthropic.JSONOutputFormatParam{
					Schema: schemaMap,
				},
			}
		}
	}

	// Thinking config — call the converter unconditionally so its
	// nil-handling actually fires in production. The converter returns
	// adaptive defaults on adaptive-capable models when ThinkingConfig is
	// nil; gating this on a nil guard made that path unreachable, and the
	// PR's "nil config defaults to adaptive on Sonnet 4.6+/Opus 4.6+" claim
	// silently failed. Two nil shapes reach here:
	//   - req.Config == nil (no config object at all)
	//   - req.Config != nil but req.Config.ThinkingConfig == nil
	// Both should produce model-aware defaults, which ThinkingConfigToAnthropic
	// already handles when passed a nil pointer.
	var thinkingCfg *genai.ThinkingConfig
	if req.Config != nil {
		thinkingCfg = req.Config.ThinkingConfig
	}
	mapping := converters.ThinkingConfigToAnthropic(thinkingCfg, m.canonicalModel)
	if m.gatewayEffortTranslation {
		mapping = converters.ThinkingConfigToAnthropicWithGatewayEffortTranslation(thinkingCfg, m.canonicalModel)
	}
	params.Thinking = mapping.Thinking
	if mapping.Effort != "" {
		params.OutputConfig.Effort = mapping.Effort
	}

	// Anthropic rejects extended thinking (manual or adaptive) combined with
	// forced tool use (tool_choice.type = "tool" or "any"). When both are
	// requested, the API may either 400 or — worse — silently produce a
	// text/thinking response with no tool_use block, which looks to callers
	// like the model just refused to call the tool. The forced tool_choice
	// is the load-bearing semantic (the caller has pinned the response
	// shape), so drop the thinking parameter on this side of the wire.
	// Effort is tied to adaptive thinking for Claude, so clear it with that
	// mode. Gateway models can use effort without an Anthropic thinking field.
	if converters.IsForcedToolUse(params.ToolChoice) {
		adaptive := params.Thinking.OfAdaptive != nil
		params.Thinking = anthropic.ThinkingConfigParamUnion{}
		if adaptive {
			params.OutputConfig.Effort = ""
		}
	}

	if m.promptCaching != nil {
		applyCacheBreakpoints(&params, m.promptCaching)
	}

	return params, nil
}

// maybeAppendUserContent ensures the conversation ends with a user message.
// Anthropic requires strictly alternating user/assistant turns.
func (m *anthropicModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents,
			genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
		return
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents,
			genai.NewContentFromText("Continue processing previous requests as instructed.", "user"))
	}
}

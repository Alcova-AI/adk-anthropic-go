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

package converters

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// MessageToLLMResponse converts an Anthropic Message to a model.LLMResponse.
func MessageToLLMResponse(msg *anthropic.Message) (*model.LLMResponse, error) {
	if msg == nil {
		return nil, fmt.Errorf("nil message received")
	}

	content := &genai.Content{
		Role:  "model",
		Parts: make([]*genai.Part, 0, len(msg.Content)),
	}

	var allCitations []*genai.Citation
	for _, block := range msg.Content {
		part, err := ContentBlockToGenaiPart(block)
		if err != nil {
			return nil, fmt.Errorf("failed to convert content block: %w", err)
		}
		if part != nil {
			content.Parts = append(content.Parts, part)
		}
		// Collect citations from text blocks
		if textBlock, ok := block.AsAny().(anthropic.TextBlock); ok {
			if citations := textCitationsToSlice(textBlock.Citations); len(citations) > 0 {
				allCitations = append(allCitations, citations...)
			}
		}
	}

	resp := &model.LLMResponse{
		Content:       content,
		UsageMetadata: UsageToMetadata(msg.Usage),
		FinishReason:  StopReasonToFinishReason(msg.StopReason),
		ModelVersion:  string(msg.Model),
	}

	if len(allCitations) > 0 {
		resp.CitationMetadata = &genai.CitationMetadata{Citations: allCitations}
	}

	return resp, nil
}

// InterruptedContent holds what could be salvaged from a message whose
// generation was cut off mid-flight (typically at the max_tokens ceiling). The
// intact content blocks are converted to genai parts in stream order; the
// trailing tool call whose input JSON was truncated is reported separately as
// data rather than converted, because its input is not valid JSON.
type InterruptedContent struct {
	Parts        []*genai.Part
	ToolName     string
	ToolID       string
	PartialInput string
}

// SalvageInterruptedMessage extracts the content that survived an interrupted
// generation. Intact blocks (thinking with signatures, completed text, and
// completed tool_use blocks with valid JSON input) are converted to genai
// parts in stream order. A tool_use block whose Input is not valid JSON is the
// point of interruption: its name, id, and partial input are captured on the
// returned struct and the block itself is skipped. Blocks that fail conversion
// are skipped rather than aborting the salvage — this runs on an error path
// where preserving what is convertible matters more than completeness.
func SalvageInterruptedMessage(msg *anthropic.Message) InterruptedContent {
	var out InterruptedContent
	if msg == nil {
		return out
	}

	for _, block := range msg.Content {
		// Read the interruption off the flattened ContentBlockUnion fields, not
		// via AsToolUse(): the SDK accumulator keeps ContentBlockUnion.Input
		// current through input_json_delta events but only refreshes the
		// variant's backing JSON on content_block_stop — which never fires for
		// the block cut off at the ceiling.
		if isIncompleteToolUse(block) {
			out.ToolName = block.Name
			out.ToolID = block.ID
			out.PartialInput = string(block.Input)
			continue
		}

		part, err := ContentBlockToGenaiPart(block)
		if err != nil || part == nil {
			continue
		}
		out.Parts = append(out.Parts, part)
	}

	return out
}

// HasIncompleteToolInput reports whether any tool_use block in the message
// carries input that isn't valid JSON — the signature of a tool call cut off
// mid-generation. Used to detect an interruption even when the SDK accumulator
// didn't surface an error.
func HasIncompleteToolInput(msg *anthropic.Message) bool {
	if msg == nil {
		return false
	}
	for _, block := range msg.Content {
		if isIncompleteToolUse(block) {
			return true
		}
	}
	return false
}

// isIncompleteToolUse reports whether a content block is a tool_use block whose
// accumulated input JSON is truncated (not valid JSON). It reads the flattened
// ContentBlockUnion fields because those are what the SDK accumulator keeps
// current for an in-progress block.
func isIncompleteToolUse(block anthropic.ContentBlockUnion) bool {
	return block.Type == "tool_use" && !json.Valid(block.Input)
}

// ContentBlockToGenaiPart converts an Anthropic ContentBlockUnion to a genai.Part.
func ContentBlockToGenaiPart(block anthropic.ContentBlockUnion) (*genai.Part, error) {
	switch variant := block.AsAny().(type) {
	case anthropic.TextBlock:
		return &genai.Part{Text: variant.Text}, nil

	case anthropic.ThinkingBlock:
		// Map thinking blocks to genai.Part with Thought=true
		signature, err := base64.StdEncoding.DecodeString(variant.Signature)
		if err != nil {
			return nil, fmt.Errorf("failed to decode thinking signature: %w", err)
		}
		return &genai.Part{
			Text:             variant.Thinking,
			Thought:          true,
			ThoughtSignature: signature,
		}, nil

	case anthropic.RedactedThinkingBlock:
		// Keep the opaque data in provider-scoped metadata so the exact redacted
		// block can be passed back while retaining a useful display marker.
		return &genai.Part{
			Text:    "[thinking redacted]",
			Thought: true,
			PartMetadata: map[string]any{
				redactedThinkingDataMetadataKey: variant.Data,
			},
		}, nil

	case anthropic.ToolUseBlock:
		// Convert to FunctionCall
		args := make(map[string]any)
		if variant.Input != nil {
			// Input is json.RawMessage, unmarshal it
			if err := json.Unmarshal(variant.Input, &args); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tool input for %q (id=%s): %w", variant.Name, variant.ID, err)
			}
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   variant.ID,
				Name: variant.Name,
				Args: args,
			},
		}, nil

	case anthropic.ServerToolUseBlock:
		// Server-side tool use (web search, etc.)
		args := make(map[string]any)
		if variant.Input != nil {
			// Input is an any type, convert through JSON
			inputBytes, err := json.Marshal(variant.Input)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal server tool input for %q (id=%s): %w", variant.Name, variant.ID, err)
			}
			if err := json.Unmarshal(inputBytes, &args); err != nil {
				return nil, fmt.Errorf("failed to unmarshal server tool input for %q (id=%s): %w", variant.Name, variant.ID, err)
			}
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   variant.ID,
				Name: string(variant.Name),
				Args: args,
			},
		}, nil

	case anthropic.WebSearchToolResultBlock:
		// Web search results from Anthropic's built-in web search tool
		return webSearchResultToFunctionResponse(variant), nil

	default:
		// Unknown block type - skip
		return nil, nil
	}
}

// webSearchResultToFunctionResponse converts a WebSearchToolResultBlock to a FunctionResponse Part.
func webSearchResultToFunctionResponse(block anthropic.WebSearchToolResultBlock) *genai.Part {
	response := make(map[string]any)

	// Check if it's an error or results
	if results := block.Content.AsWebSearchResultBlockArray(); len(results) > 0 {
		searchResults := make([]map[string]any, 0, len(results))
		for _, result := range results {
			searchResults = append(searchResults, map[string]any{
				"title":   result.Title,
				"url":     result.URL,
				"pageAge": result.PageAge,
			})
		}
		response["results"] = searchResults
	} else if errBlock := block.Content.AsResponseWebSearchToolResultError(); errBlock.ErrorCode != "" {
		response["error"] = string(errBlock.ErrorCode)
	}

	return &genai.Part{
		FunctionResponse: &genai.FunctionResponse{
			ID:       block.ToolUseID,
			Name:     "web_search",
			Response: response,
		},
	}
}

// textCitationsToSlice converts Anthropic text citations to a slice of genai.Citation.
func textCitationsToSlice(citations []anthropic.TextCitationUnion) []*genai.Citation {
	if len(citations) == 0 {
		return nil
	}

	result := make([]*genai.Citation, 0, len(citations))
	for _, c := range citations {
		citation := &genai.Citation{
			Title: c.DocumentTitle,
		}

		// Map based on citation type
		switch c.Type {
		case "char_location":
			citation.StartIndex = int32(c.StartCharIndex)
			citation.EndIndex = int32(c.EndCharIndex)
		case "web_search_result_location":
			citation.Title = c.Title
			citation.URI = c.URL
		case "search_result_location":
			citation.Title = c.Title
		}

		result = append(result, citation)
	}

	return result
}

// UsageToMetadata converts Anthropic Usage to genai UsageMetadata.
//
// Anthropic reports input_tokens as non-cached tokens only, with cache tokens
// as separate additive fields. The OTEL GenAI convention expects input_tokens
// to be the total (cached + uncached), with cached as a subset. We normalise
// here so that downstream cost calculations work correctly.
func UsageToMetadata(usage anthropic.Usage) *genai.GenerateContentResponseUsageMetadata {
	totalInput := usage.InputTokens + usage.CacheReadInputTokens + usage.CacheCreationInputTokens
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        int32(totalInput),
		CandidatesTokenCount:    int32(usage.OutputTokens),
		TotalTokenCount:         int32(totalInput + usage.OutputTokens),
		CachedContentTokenCount: int32(usage.CacheReadInputTokens),
	}
}

// StopReasonToFinishReason maps Anthropic StopReason to genai FinishReason.
func StopReasonToFinishReason(sr anthropic.StopReason) genai.FinishReason {
	switch sr {
	case anthropic.StopReasonEndTurn:
		return genai.FinishReasonStop
	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	case anthropic.StopReasonStopSequence:
		return genai.FinishReasonStop
	case anthropic.StopReasonToolUse:
		return genai.FinishReasonStop
	default:
		return genai.FinishReasonUnspecified
	}
}

// StreamDeltaToPartialResponse converts a streaming content block delta to a partial LLMResponse.
// Used for streaming text updates.
func StreamDeltaToPartialResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{Text: text},
			},
		},
		Partial: true,
	}
}

// StreamThinkingDeltaToPartialResponse converts a streaming thinking delta to a partial LLMResponse.
func StreamThinkingDeltaToPartialResponse(thinking string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					Text:    thinking,
					Thought: true,
				},
			},
		},
		Partial: true,
	}
}

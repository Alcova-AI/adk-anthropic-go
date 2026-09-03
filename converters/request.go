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

// Package converters provides conversion functions between genai types and Anthropic SDK types.
package converters

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

const (
	continuationPrompt              = "Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed."
	redactedThinkingDataMetadataKey = "anthropic.redacted_thinking_data"
)

// ContentsToMessages converts genai Contents to Anthropic MessageParams. It
// merges ordinary same-role turns, but inserts a synthetic user continuation
// between assistant turns when merging would modify a thinking block. It
// returns an error when an unresolved tool use prevents inserting that boundary.
func ContentsToMessages(contents []*genai.Content) ([]anthropic.MessageParam, error) {
	if len(contents) == 0 {
		return nil, nil
	}

	var messages []anthropic.MessageParam
	for _, content := range contents {
		if content == nil {
			continue
		}

		msg, err := contentToMessage(content)
		if err != nil {
			return nil, fmt.Errorf("failed to convert content: %w", err)
		}
		if msg != nil {
			messages = append(messages, *msg)
		}
	}

	// Anthropic combines consecutive messages with the same role. Normalise them
	// explicitly so signed thinking blocks retain their original message boundary.
	messages, err := normalizeConsecutiveMessages(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to normalize messages: %w", err)
	}

	return messages, nil
}

// contentToMessage converts a single genai.Content to an Anthropic MessageParam.
func contentToMessage(content *genai.Content) (*anthropic.MessageParam, error) {
	if content == nil || len(content.Parts) == 0 {
		return nil, nil
	}

	// Check if this content contains tool results (FunctionResponse).
	// Anthropic requires tool results to be in user messages.
	hasFunctionResponse := false
	hasFunctionCall := false
	for _, part := range content.Parts {
		if part != nil {
			if part.FunctionResponse != nil {
				hasFunctionResponse = true
			}
			if part.FunctionCall != nil {
				hasFunctionCall = true
			}
		}
	}

	// Determine the role - tool results must be user, tool calls must be assistant
	var role anthropic.MessageParamRole
	if hasFunctionResponse {
		// Tool results MUST be in user messages per Anthropic API requirements
		role = anthropic.MessageParamRoleUser
	} else if hasFunctionCall {
		// Tool calls (from model) MUST be in assistant messages
		role = anthropic.MessageParamRoleAssistant
	} else {
		var err error
		role, err = mapRole(content.Role)
		if err != nil {
			return nil, err
		}
	}

	var blocks []anthropic.ContentBlockParamUnion
	for _, part := range content.Parts {
		if part == nil {
			continue
		}
		// TODO: Remove this compatibility guard once the minimum adk-go version
		// includes https://github.com/google/adk-go/pull/1104. Older versions can
		// retain a foreign agent's hidden thoughts while changing its role to user,
		// but Anthropic only accepts thinking blocks in assistant messages.
		if role == anthropic.MessageParamRoleUser && part.Thought {
			continue
		}
		block, err := PartToContentBlock(part)
		if err != nil {
			return nil, fmt.Errorf("failed to convert part: %w", err)
		}
		if block != nil {
			blocks = append(blocks, *block)
		}
	}

	if len(blocks) == 0 {
		return nil, nil
	}

	msg := anthropic.MessageParam{
		Role:    role,
		Content: blocks,
	}
	return &msg, nil
}

// mapRole maps genai role to Anthropic MessageParamRole.
func mapRole(role string) (anthropic.MessageParamRole, error) {
	switch strings.ToLower(role) {
	case "user":
		return anthropic.MessageParamRoleUser, nil
	case "model", "assistant":
		return anthropic.MessageParamRoleAssistant, nil
	default:
		return "", fmt.Errorf("unsupported role: %s", role)
	}
}

// PartToContentBlock converts a genai Part to an Anthropic ContentBlockParamUnion.
func PartToContentBlock(part *genai.Part) (*anthropic.ContentBlockParamUnion, error) {
	if part == nil {
		return nil, nil
	}

	if data, ok := redactedThinkingData(part); ok {
		block := anthropic.NewRedactedThinkingBlock(data)
		return &block, nil
	}

	// Thoughts from model responses must be passed back with their signature,
	// including when Anthropic returned an empty thinking summary.
	if part.Thought && len(part.ThoughtSignature) > 0 {
		block := anthropic.NewThinkingBlock(
			base64.StdEncoding.EncodeToString(part.ThoughtSignature),
			part.Text,
		)
		return &block, nil
	}

	// Text content
	if part.Text != "" {
		block := anthropic.NewTextBlock(part.Text)
		return &block, nil
	}

	// Inline binary data (images, PDFs)
	if part.InlineData != nil {
		return inlineDataToBlock(part.InlineData)
	}

	// File data (URI-based)
	if part.FileData != nil {
		return fileDataToBlock(part.FileData)
	}

	// Function response (tool result)
	if part.FunctionResponse != nil {
		return functionResponseToBlock(part.FunctionResponse)
	}

	// Function call - these should only appear in model responses, not requests
	// We return nil for these as they shouldn't be in user messages
	if part.FunctionCall != nil {
		return functionCallToBlock(part.FunctionCall)
	}

	// Executable code and CodeExecutionResult are Gemini-specific features
	// that don't have direct Anthropic equivalents
	if part.ExecutableCode != nil || part.CodeExecutionResult != nil {
		return nil, fmt.Errorf("ExecutableCode and CodeExecutionResult are not supported by Anthropic")
	}

	return nil, nil
}

func redactedThinkingData(part *genai.Part) (string, bool) {
	if !part.Thought || part.PartMetadata == nil {
		return "", false
	}
	data, ok := part.PartMetadata[redactedThinkingDataMetadataKey].(string)
	return data, ok
}

// inlineDataToBlock converts inline binary data to an Anthropic content block.
func inlineDataToBlock(blob *genai.Blob) (*anthropic.ContentBlockParamUnion, error) {
	if blob == nil {
		return nil, nil
	}

	mimeType := strings.ToLower(blob.MIMEType)

	// Handle images
	if strings.HasPrefix(mimeType, "image/") {
		mediaType, err := mapImageMediaType(mimeType)
		if err != nil {
			return nil, err
		}
		block := anthropic.ContentBlockParamUnion{
			OfImage: &anthropic.ImageBlockParam{
				Source: anthropic.ImageBlockParamSourceUnion{
					OfBase64: &anthropic.Base64ImageSourceParam{
						Data:      base64.StdEncoding.EncodeToString(blob.Data),
						MediaType: mediaType,
					},
				},
			},
		}
		return &block, nil
	}

	// Handle PDFs (beta feature)
	if mimeType == "application/pdf" {
		block := anthropic.ContentBlockParamUnion{
			OfDocument: &anthropic.DocumentBlockParam{
				Source: anthropic.DocumentBlockParamSourceUnion{
					OfBase64: &anthropic.Base64PDFSourceParam{
						Data: base64.StdEncoding.EncodeToString(blob.Data),
					},
				},
			},
		}
		return &block, nil
	}

	return nil, fmt.Errorf("unsupported MIME type for inline data: %s", mimeType)
}

// mapImageMediaType maps MIME types to Anthropic Base64ImageSourceMediaType.
func mapImageMediaType(mimeType string) (anthropic.Base64ImageSourceMediaType, error) {
	switch mimeType {
	case "image/jpeg":
		return anthropic.Base64ImageSourceMediaTypeImageJPEG, nil
	case "image/png":
		return anthropic.Base64ImageSourceMediaTypeImagePNG, nil
	case "image/gif":
		return anthropic.Base64ImageSourceMediaTypeImageGIF, nil
	case "image/webp":
		return anthropic.Base64ImageSourceMediaTypeImageWebP, nil
	default:
		return "", fmt.Errorf("unsupported image media type: %s", mimeType)
	}
}

// fileDataToBlock converts URI-based file data to an Anthropic content block.
func fileDataToBlock(fileData *genai.FileData) (*anthropic.ContentBlockParamUnion, error) {
	if fileData == nil {
		return nil, nil
	}

	mimeType := strings.ToLower(fileData.MIMEType)

	// Handle images via URL
	if strings.HasPrefix(mimeType, "image/") {
		block := anthropic.ContentBlockParamUnion{
			OfImage: &anthropic.ImageBlockParam{
				Source: anthropic.ImageBlockParamSourceUnion{
					OfURL: &anthropic.URLImageSourceParam{
						URL: fileData.FileURI,
					},
				},
			},
		}
		return &block, nil
	}

	// Handle PDFs via URL (beta feature)
	if mimeType == "application/pdf" {
		block := anthropic.ContentBlockParamUnion{
			OfDocument: &anthropic.DocumentBlockParam{
				Source: anthropic.DocumentBlockParamSourceUnion{
					OfURL: &anthropic.URLPDFSourceParam{
						URL: fileData.FileURI,
					},
				},
			},
		}
		return &block, nil
	}

	return nil, fmt.Errorf("unsupported MIME type for file data: %s", mimeType)
}

// functionResponseToBlock converts a FunctionResponse to an Anthropic tool result block.
func functionResponseToBlock(resp *genai.FunctionResponse) (*anthropic.ContentBlockParamUnion, error) {
	if resp == nil {
		return nil, nil
	}

	// The function ID is required for proper tool call correlation.
	// Without it, Anthropic cannot match tool results to their originating tool calls.
	if resp.ID == "" {
		return nil, fmt.Errorf("FunctionResponse.ID is required for tool call correlation (function: %s)", resp.Name)
	}

	var content []anthropic.ToolResultBlockParamContentUnion
	if resp.Response != nil {
		jsonBytes, err := json.Marshal(resp.Response)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal function response: %w", err)
		}
		content = append(content, anthropic.ToolResultBlockParamContentUnion{
			OfText: &anthropic.TextBlockParam{Text: string(jsonBytes)},
		})
	}

	for _, part := range resp.Parts {
		converted, err := functionResponsePartToBlock(part)
		if err != nil {
			return nil, fmt.Errorf("failed to convert function response part: %w", err)
		}
		if converted != nil {
			content = append(content, *converted)
		}
	}

	block := anthropic.ContentBlockParamUnion{
		OfToolResult: &anthropic.ToolResultBlockParam{
			ToolUseID: resp.ID,
			Content:   content,
			IsError:   anthropic.Bool(false),
		},
	}
	return &block, nil
}

func functionResponsePartToBlock(part *genai.FunctionResponsePart) (*anthropic.ToolResultBlockParamContentUnion, error) {
	if part == nil {
		return nil, nil
	}

	var block *anthropic.ContentBlockParamUnion
	var err error
	switch {
	case part.InlineData != nil:
		block, err = inlineDataToBlock(&genai.Blob{
			Data:     part.InlineData.Data,
			MIMEType: part.InlineData.MIMEType,
		})
	case part.FileData != nil:
		block, err = fileDataToBlock(&genai.FileData{
			FileURI:     part.FileData.FileURI,
			MIMEType:    part.FileData.MIMEType,
			DisplayName: part.FileData.DisplayName,
		})
	default:
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if block == nil {
		return nil, nil
	}

	switch {
	case block.OfImage != nil:
		return &anthropic.ToolResultBlockParamContentUnion{OfImage: block.OfImage}, nil
	case block.OfDocument != nil:
		return &anthropic.ToolResultBlockParamContentUnion{OfDocument: block.OfDocument}, nil
	default:
		return nil, fmt.Errorf("unsupported function response content block")
	}
}

// functionCallToBlock converts a FunctionCall to an Anthropic tool use block.
// This is used when passing model responses back (e.g., in conversation history).
func functionCallToBlock(call *genai.FunctionCall) (*anthropic.ContentBlockParamUnion, error) {
	if call == nil {
		return nil, nil
	}

	// Anthropic requires input to be a dictionary - ensure we have a valid map
	// After JSON round-trip, nil maps stay nil, so we must always provide a valid map
	var input any = call.Args
	if call.Args == nil || len(call.Args) == 0 {
		input = map[string]any{}
	}

	block := anthropic.NewToolUseBlock(call.ID, input, call.Name)
	return &block, nil
}

// SystemInstructionToSystem converts a genai SystemInstruction to Anthropic system text blocks.
func SystemInstructionToSystem(instruction *genai.Content) []anthropic.TextBlockParam {
	if instruction == nil || len(instruction.Parts) == 0 {
		return nil
	}

	var blocks []anthropic.TextBlockParam
	for _, part := range instruction.Parts {
		if part != nil && part.Text != "" {
			blocks = append(blocks, anthropic.TextBlockParam{
				Text: part.Text,
			})
		}
	}
	return blocks
}

// normalizeConsecutiveMessages makes roles alternate without modifying
// thinking-bearing messages. Anthropic rejects signed thinking blocks when
// their original assistant message is changed, including by combining it with
// an adjacent assistant message.
func normalizeConsecutiveMessages(messages []anthropic.MessageParam) ([]anthropic.MessageParam, error) {
	if len(messages) <= 1 {
		return messages, nil
	}

	var normalized []anthropic.MessageParam
	for i, msg := range messages {
		if i == 0 {
			normalized = append(normalized, msg)
			continue
		}

		last := &normalized[len(normalized)-1]
		if last.Role != msg.Role {
			normalized = append(normalized, msg)
			continue
		}

		if msg.Role == anthropic.MessageParamRoleAssistant &&
			(hasThinkingBlock(last.Content) || hasThinkingBlock(msg.Content)) {
			if hasToolUseBlock(last.Content) {
				return nil, fmt.Errorf("cannot preserve thinking boundary after tool use without an intervening tool result")
			}
			normalized = append(normalized,
				anthropic.NewUserMessage(anthropic.NewTextBlock(continuationPrompt)),
				msg,
			)
		} else {
			last.Content = append(last.Content, msg.Content...)
		}
	}
	return normalized, nil
}

func hasThinkingBlock(blocks []anthropic.ContentBlockParamUnion) bool {
	for _, block := range blocks {
		if block.OfThinking != nil || block.OfRedactedThinking != nil {
			return true
		}
	}
	return false
}

func hasToolUseBlock(blocks []anthropic.ContentBlockParamUnion) bool {
	for _, block := range blocks {
		if block.OfToolUse != nil {
			return true
		}
	}
	return false
}

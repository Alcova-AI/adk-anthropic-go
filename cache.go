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
)

// PromptCacheMode selects who manages prompt-cache breakpoints.
type PromptCacheMode uint8

const (
	// PromptCacheProviderDefault sends no explicit cache controls.
	PromptCacheProviderDefault PromptCacheMode = iota
	// PromptCacheManual applies the configured Anthropic cache breakpoints.
	PromptCacheManual
	// PromptCacheGatewayAutomatic lets a compatible gateway place breakpoints.
	PromptCacheGatewayAutomatic
)

// CacheBreakpoint configures one Anthropic cache control breakpoint.
type CacheBreakpoint struct {
	TTL anthropic.CacheControlEphemeralTTL
}

// PromptCachingConfig configures prompt caching for one model route.
type PromptCachingConfig struct {
	Mode                PromptCacheMode
	Auto                *CacheBreakpoint
	SystemInstruction   *CacheBreakpoint
	Tools               *CacheBreakpoint
	ConversationHistory *CacheBreakpoint
}

func (c PromptCachingConfig) validate() error {
	if c.Mode > PromptCacheGatewayAutomatic {
		return fmt.Errorf("unsupported prompt cache mode %d", c.Mode)
	}
	breakpoints := c.Auto != nil || c.SystemInstruction != nil || c.Tools != nil || c.ConversationHistory != nil
	if c.Mode != PromptCacheManual && breakpoints {
		return fmt.Errorf("prompt cache breakpoints require manual mode")
	}
	return nil
}

// applyCacheBreakpoints sets cache_control breakpoints on the request based
// on the provided configuration. Each breakpoint is independently optional.
//
// Anthropic evaluates cache prefixes in order: tools → system → messages.
// When mixing TTLs, longer TTLs must appear before shorter TTLs in this order.
func applyCacheBreakpoints(params *anthropic.MessageNewParams, cfg *PromptCachingConfig) {
	// 1. Tools — last tool definition
	if cfg.Tools != nil && len(params.Tools) > 0 {
		last := &params.Tools[len(params.Tools)-1]
		if last.OfTool != nil {
			last.OfTool.CacheControl = newCacheControl(cfg.Tools)
		}
	}

	// 2. System — last text block
	if cfg.SystemInstruction != nil && len(params.System) > 0 {
		params.System[len(params.System)-1].CacheControl = newCacheControl(cfg.SystemInstruction)
	}

	// 3. Conversation history — last content block of the penultimate message
	if cfg.ConversationHistory != nil && len(params.Messages) >= 2 {
		msg := &params.Messages[len(params.Messages)-2]
		if len(msg.Content) > 0 {
			last := &msg.Content[len(msg.Content)-1]
			if ccPtr := last.GetCacheControl(); ccPtr != nil {
				*ccPtr = newCacheControl(cfg.ConversationHistory)
			}
		}
	}

	// 4. Auto — top-level cache_control (Anthropic places breakpoint automatically)
	if cfg.Auto != nil {
		params.CacheControl = newCacheControl(cfg.Auto)
	}
}

// newCacheControl creates a CacheControlEphemeralParam from a breakpoint config.
func newCacheControl(bp *CacheBreakpoint) anthropic.CacheControlEphemeralParam {
	cc := anthropic.NewCacheControlEphemeralParam()
	if bp.TTL != "" {
		cc.TTL = bp.TTL
	}
	return cc
}

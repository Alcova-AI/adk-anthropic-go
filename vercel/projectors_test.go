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

package vercel

import (
	"reflect"
	"strings"
	"testing"

	"google.golang.org/genai"
)

func TestOpenAIModelOptions(t *testing.T) {
	for _, level := range []genai.ThinkingLevel{
		genai.ThinkingLevelMinimal,
		genai.ThinkingLevelLow,
		genai.ThinkingLevelMedium,
		genai.ThinkingLevelHigh,
	} {
		t.Run(string(level), func(t *testing.T) {
			options, err := (OpenAIModelOptions{}).ProjectProviderOptions(ProviderOptionsInput{ThinkingLevel: level})
			if err != nil {
				t.Fatalf("ProjectProviderOptions() error = %v", err)
			}
			want := map[string]map[string]any{
				"openai": {"reasoningEffort": strings.ToLower(string(level)), "store": false},
			}
			if !reflect.DeepEqual(options, want) {
				t.Fatalf("ProjectProviderOptions() = %#v, want %#v", options, want)
			}
		})
	}
}

func TestOpenAIModelOptions_ReasoningSummaryFollowsIncludeThoughts(t *testing.T) {
	options, err := (OpenAIModelOptions{}).ProjectProviderOptions(ProviderOptionsInput{
		ThinkingLevel: genai.ThinkingLevelMedium, IncludeThoughts: true,
	})
	if err != nil {
		t.Fatalf("ProjectProviderOptions() error = %v", err)
	}
	if got := options["openai"]["reasoningSummary"]; got != "auto" {
		t.Fatalf("reasoningSummary = %v, want auto", got)
	}
}

func TestGoogleModelOptions(t *testing.T) {
	for _, level := range []genai.ThinkingLevel{
		genai.ThinkingLevelMinimal,
		genai.ThinkingLevelLow,
		genai.ThinkingLevelMedium,
		genai.ThinkingLevelHigh,
	} {
		t.Run(string(level), func(t *testing.T) {
			options, err := (GoogleModelOptions{}).ProjectProviderOptions(ProviderOptionsInput{ThinkingLevel: level})
			if err != nil {
				t.Fatalf("ProjectProviderOptions() error = %v", err)
			}
			want := map[string]map[string]any{
				"google": {"thinkingConfig": map[string]any{"thinkingLevel": strings.ToLower(string(level))}},
			}
			if !reflect.DeepEqual(options, want) {
				t.Fatalf("ProjectProviderOptions() = %#v, want %#v", options, want)
			}
		})
	}
}

func TestGoogleModelOptions_IncludeThoughts(t *testing.T) {
	options, err := (GoogleModelOptions{}).ProjectProviderOptions(ProviderOptionsInput{
		ThinkingLevel:   genai.ThinkingLevelMinimal,
		IncludeThoughts: true,
	})
	if err != nil {
		t.Fatalf("ProjectProviderOptions() error = %v", err)
	}
	want := map[string]map[string]any{
		"google": {"thinkingConfig": map[string]any{"thinkingLevel": "minimal", "includeThoughts": true}},
	}
	if !reflect.DeepEqual(options, want) {
		t.Fatalf("ProjectProviderOptions() = %#v, want %#v", options, want)
	}
}

func TestZAIModelOptions_MapsSupportedEffortLevels(t *testing.T) {
	tests := []struct {
		level genai.ThinkingLevel
		want  string
	}{
		{genai.ThinkingLevelMinimal, "low"},
		{genai.ThinkingLevelLow, "low"},
		{genai.ThinkingLevelMedium, "high"},
		{genai.ThinkingLevelHigh, "max"},
	}
	for _, tt := range tests {
		t.Run(string(tt.level), func(t *testing.T) {
			options, err := (ZAIModelOptions{}).ProjectProviderOptions(ProviderOptionsInput{ThinkingLevel: tt.level})
			if err != nil {
				t.Fatalf("ProjectProviderOptions() error = %v", err)
			}
			zai := options["zai"]
			if zai["reasoningEffort"] != tt.want {
				t.Fatalf("reasoningEffort = %v, want %q", zai["reasoningEffort"], tt.want)
			}
			thinking := zai["thinking"].(map[string]any)
			if !reflect.DeepEqual(thinking, map[string]any{"type": "enabled"}) {
				t.Fatalf("thinking = %#v", thinking)
			}
		})
	}
}

func TestZAIModelOptions_OmitsUnspecifiedEffort(t *testing.T) {
	options, err := (ZAIModelOptions{}).ProjectProviderOptions(ProviderOptionsInput{})
	if err != nil {
		t.Fatalf("ProjectProviderOptions() error = %v", err)
	}
	if _, ok := options["zai"]["reasoningEffort"]; ok {
		t.Fatalf("reasoningEffort = %v, want omitted", options["zai"]["reasoningEffort"])
	}
}

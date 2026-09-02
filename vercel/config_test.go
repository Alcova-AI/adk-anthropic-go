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
	"testing"

	"google.golang.org/genai"
)

type testProjector map[string]map[string]any

func (p testProjector) ProjectProviderOptions(ProviderOptionsInput) (map[string]map[string]any, error) {
	return p, nil
}

func TestConfig_ZeroValueRequiresZDR(t *testing.T) {
	options, err := (Config{}).WireProviderOptions(ProviderOptionsInput{})
	if err != nil {
		t.Fatalf("WireProviderOptions() error = %v", err)
	}
	gateway := options["gateway"].(map[string]any)
	if gateway["zeroDataRetention"] != true {
		t.Fatalf("zeroDataRetention = %v, want true", gateway["zeroDataRetention"])
	}
}

func TestConfig_WireProviderOptionsRejectsInvalidDataPolicy(t *testing.T) {
	_, err := (Config{DataPolicy: DataPolicy(255)}).WireProviderOptions(ProviderOptionsInput{})
	if err == nil {
		t.Fatal("WireProviderOptions() error = nil")
	}
}

func TestConfig_WireProviderOptions(t *testing.T) {
	cfg := Config{
		Routing: Routing{
			Only:  []string{"anthropic", "vertexAnthropic"},
			Order: []string{"vertexAnthropic"},
			Sort:  SortTTFT,
		},
		DataPolicy: RetentionAllowed,
		ProviderOptions: map[string]map[string]any{
			"anthropic": {"sendReasoning": false},
		},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	options, err := cfg.WireProviderOptions(ProviderOptionsInput{})
	if err != nil {
		t.Fatalf("WireProviderOptions() error = %v", err)
	}
	gateway := options["gateway"].(map[string]any)
	if gateway["sort"] != "ttft" || gateway["zeroDataRetention"] != false {
		t.Fatalf("gateway = %#v", gateway)
	}
	provider := options["anthropic"].(map[string]any)
	if provider["sendReasoning"] != false {
		t.Fatalf("anthropic options = %#v", provider)
	}
}

func TestConfig_RejectsProjectedStaticConflict(t *testing.T) {
	cfg := Config{
		ProviderOptions: map[string]map[string]any{"openai": {"reasoningEffort": "low"}},
		Projector:       OpenAIModelOptions{},
	}
	if err := cfg.Validate(); err == nil {
		t.Fatal("Validate() error = nil")
	}
}

func TestConfig_MergesStaticAndProjectedOptions(t *testing.T) {
	cfg := Config{
		ProviderOptions: map[string]map[string]any{"openai": {"serviceTier": "priority"}},
		Projector:       OpenAIModelOptions{},
	}
	options, err := cfg.WireProviderOptions(ProviderOptionsInput{ThinkingLevel: genai.ThinkingLevelLow})
	if err != nil {
		t.Fatalf("WireProviderOptions() error = %v", err)
	}
	openai := options["openai"].(map[string]any)
	if openai["serviceTier"] != "priority" || openai["reasoningEffort"] != "low" || openai["store"] != false {
		t.Fatalf("OpenAI options = %#v", openai)
	}
}

func TestConfig_RejectsProjectedGatewayNamespace(t *testing.T) {
	cfg := Config{Projector: testProjector{"gateway": {"zeroDataRetention": false}}}
	if err := cfg.Validate(); err == nil {
		t.Fatal("Validate() error = nil")
	}
}

func TestConfig_RejectsAmbiguousOptions(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{name: "gateway namespace", cfg: Config{ProviderOptions: map[string]map[string]any{"gateway": {"only": []string{"x"}}}}},
		{name: "reasoning override", cfg: Config{ProviderOptions: map[string]map[string]any{"zai": {"reasoningEffort": "high"}}}},
		{name: "empty static key", cfg: Config{ProviderOptions: map[string]map[string]any{"zai": {" ": true}}}},
		{name: "static namespace leading whitespace", cfg: Config{ProviderOptions: map[string]map[string]any{" zai": {"temperature": 0}}}},
		{name: "static namespace trailing whitespace", cfg: Config{ProviderOptions: map[string]map[string]any{"zai ": {"temperature": 0}}}},
		{name: "static key leading whitespace", cfg: Config{ProviderOptions: map[string]map[string]any{"zai": {" temperature": 0}}}},
		{name: "static key trailing whitespace", cfg: Config{ProviderOptions: map[string]map[string]any{"zai": {"temperature ": 0}}}},
		{name: "reserved static key with whitespace", cfg: Config{ProviderOptions: map[string]map[string]any{"openai": {" store": true}}}},
		{name: "projected namespace leading whitespace", cfg: Config{Projector: testProjector{" zai": {"thinking": map[string]any{"type": "enabled"}}}}},
		{name: "projected namespace trailing whitespace", cfg: Config{Projector: testProjector{"zai ": {"thinking": map[string]any{"type": "enabled"}}}}},
		{name: "projected key leading whitespace", cfg: Config{Projector: testProjector{"zai": {" thinking": map[string]any{"type": "enabled"}}}}},
		{name: "projected key trailing whitespace", cfg: Config{Projector: testProjector{"zai": {"thinking ": map[string]any{"type": "enabled"}}}}},
		{name: "only provider leading whitespace", cfg: Config{Routing: Routing{Only: []string{" zai"}}}},
		{name: "only provider trailing whitespace", cfg: Config{Routing: Routing{Only: []string{"zai "}}}},
		{name: "order provider leading whitespace", cfg: Config{Routing: Routing{Order: []string{" zai"}}}},
		{name: "order provider trailing whitespace", cfg: Config{Routing: Routing{Order: []string{"zai "}}}},
		{name: "duplicate route", cfg: Config{Routing: Routing{Only: []string{"zai", "zai"}}}},
		{name: "invalid sort", cfg: Config{Routing: Routing{Sort: "latency"}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.cfg.Validate(); err == nil {
				t.Fatal("Validate() error = nil")
			}
		})
	}
}

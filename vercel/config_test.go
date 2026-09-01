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

import "testing"

func TestConfig_ZeroValueRequiresZDR(t *testing.T) {
	options := (Config{}).WireProviderOptions()
	gateway := options["gateway"].(map[string]any)
	if gateway["zeroDataRetention"] != true {
		t.Fatalf("zeroDataRetention = %v, want true", gateway["zeroDataRetention"])
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
	options := cfg.WireProviderOptions()
	gateway := options["gateway"].(map[string]any)
	if gateway["sort"] != "ttft" || gateway["zeroDataRetention"] != false {
		t.Fatalf("gateway = %#v", gateway)
	}
	provider := options["anthropic"].(map[string]any)
	if provider["sendReasoning"] != false {
		t.Fatalf("anthropic options = %#v", provider)
	}
}

func TestConfig_RejectsAmbiguousOptions(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{name: "gateway namespace", cfg: Config{ProviderOptions: map[string]map[string]any{"gateway": {"only": []string{"x"}}}}},
		{name: "reasoning override", cfg: Config{ProviderOptions: map[string]map[string]any{"zai": {"reasoningEffort": "high"}}}},
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

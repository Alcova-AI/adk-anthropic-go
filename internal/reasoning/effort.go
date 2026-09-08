// Copyright 2026 Alcova AI
// Licensed under the Apache License, Version 2.0.

// Package reasoning translates model-specific reasoning effort values.
package reasoning

import (
	"strings"
	"time"
)

// OpenAIEffort preserves supported effort values, translating minimal to none
// only for Luna, which rejects minimal. Unknown models retain their behaviour.
func OpenAIEffort(model, effort string) string {
	if effort != "minimal" {
		return effort
	}
	model = strings.TrimPrefix(model, "openai/")
	if model == "gpt-5.6-luna" {
		return "none"
	}
	if date, ok := strings.CutPrefix(model, "gpt-5.6-luna-"); ok {
		if _, err := time.Parse("2006-01-02", date); err == nil {
			return "none"
		}
	}
	return effort
}

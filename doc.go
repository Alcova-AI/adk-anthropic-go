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

// Package adkanthropic implements ADK's model.LLM interface on top of the
// Anthropic Messages API.
//
// The caller constructs the Anthropic SDK client, so the same adapter can use
// the direct Anthropic API, Anthropic on Vertex AI, or a compatible gateway
// without the adapter owning credentials or endpoint policy. Model routes also
// select explicit reasoning and prompt-cache strategies instead of relying on
// model-name or endpoint inference.
//
// The optional vercel subpackage adds typed AI Gateway routing, fail-closed
// zero-data-retention policy, provider options, and response metadata.
package adkanthropic

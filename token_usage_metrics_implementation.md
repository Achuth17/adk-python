# Token Usage Metrics Implementation Plan

## Goal Description
We implemented three new metrics explicitly defining token usage within the generative AI models for greater observability:
- `gen_ai.usage.system_instruction_tokens`
- `gen_ai.usage.reasoning_tokens`
- `gen_ai.usage.reasoning_tokens_limit`

These align with `new-metrics-to-be-added.md`. The metrics are captured from the model's response `usage_metadata` (and potentially `llm_request.config` for limits) and set as attributes on the LLM OpenTelemetry span.

## Proposed Changes

### Adding Multi-Provider Support (LiteLLM & ApigeeLLM)
To ensure the new metrics populate for non-Gemini models, we updated the usage metadata construction inside the provider integrations.

#### src/google/adk/models/lite_llm.py & src/google/adk/models/apigee_llm.py
- OpenAi structured responses nest reasoning tokens under `completion_tokens_details`. We updated the `usage_metadata` initialization logic in both adapters to parse `reasoning_tokens` from this dict (if it exists) and pass it into the `thoughts_token_count` keyword argument of `types.GenerateContentResponseUsageMetadata()`.
- *Note:* Providers typically don't separate `system_instruction_tokens`, so it is mapped as None if missing.

### Tracing Telemetry Attributes
We modified the core tracing logic to extract and attach the new token counts when an LLM is called using plain extraction style without `getattr`.

#### src/google/adk/telemetry/tracing.py
- **Function `trace_call_llm`**:
  - Extract system instructions using plain access: `if llm_response.usage_metadata.system_instruction_tokens is not None: span.set_attribute(...)`.
  - Extract reasoning tokens using plain access: `if llm_response.usage_metadata.thoughts_token_count is not None: span.set_attribute(...)` (the standard `google.genai` SDK field).
  - Extract reasoning tokens limit by securely checking for the presence of the fields on the request config trajectory (`llm_request.config.thinking_config.thinking_budget`).
  - Add `span.set_attribute` calls for `gen_ai.usage.system_instruction_tokens`, `gen_ai.usage.reasoning_tokens`, and `gen_ai.usage.reasoning_tokens_limit`.

## Tests
To ensure robustness, we test both the Provider layer (are the models correctly parsing the tokens from the network response?) and the Tracing layer (are the tokens correctly bridging to the OpenTelemetry span?).

#### tests/unittests/telemetry/test_spans.py
This file tests the core tracing logic and assumes the `LlmResponse` object is already correctly populated. It is provider-agnostic.
- **Function `test_trace_call_llm` (Presence case)**:
  - Update the mocked `usage_metadata` inside `LlmResponse` creation to explicitly include mock values, e.g., `thoughts_token_count=10`.
  - Add the new expected `mock.call` objects to `expected_calls` to assert the attributes are set correctly.
- **Function `test_trace_call_llm_with_no_usage_metadata` (Absence case)**:
  - Assert that the new trace attributes are NOT set when the tokens fields are missing, confirming the new plain attribute access is robust.
- Apply similar setup to other integration tests like `test_generate_content_span`.

#### tests/unittests/models/test_litellm.py
LiteLLM currently relies on mocking an HTTP payload containing a `usage` dictionary and asserting `.usage_metadata` translation.
- **Functions `test_generate_content_async_with_usage_metadata` & variants**:
  - Add `completion_tokens_details: {"reasoning_tokens": 5}` to the `usage` mocking dictionary.
  - Assert that `response.usage_metadata.thoughts_token_count == 5`.

#### tests/unittests/models/test_apigee_llm.py
`test_apigee_llm.py` doesn't currently appear to have heavy usage metadata parsing tests for its OpenAI-compatible format `CompletionsHTTPClient`.
- We added testing (e.g., `test_parse_response_usage_metadata`) explicitly testing the `_parse_response` mechanism of `CompletionsHTTPClient` to assert that OpenAI style dictionaries successfully populate `prompt_token_count`, `candidates_token_count` as well as the new `thoughts_token_count`.

# Implementation Plan: Tool Error Type Telemetry

## Goal Description
The goal is to allow tool developers to easily record standardized error classifications (like HTTP 500, 404, 401, timeout, etc.) on OpenTelemetry traces when a tool fails. We will achieve this by introducing an enum of common error types and a base exception `ToolExecutionError` that tools can raise. The ADK's telemetry tracer will catch this exception and populate the `error.type` attribute on the `execute_tool` span, compliant with OpenTelemetry GenAI/HTTP semantic conventions.

## Proposed Changes

### 1. `src/google/adk/errors/tool_execution_error.py` [NEW]
Create a new file to house the error types and the execution error exception.
*   **`ToolErrorType` (Enum)**: A string Enum containing standard `error.type` semantics:
    *   `TIMEOUT = "timeout"`
    *   `INTERNAL_SERVER_ERROR = "500"`
    *   `BAD_REQUEST = "400"`
    *   `UNAUTHORIZED = "401"`
    *   `NOT_FOUND = "404"`
    *   `UNKNOWN_HOST = "java.net.UnknownHostException"` (OpenTelemetry standard for DNS failures)
*   **`ToolExecutionError(Exception)`**: A custom exception class that takes an `error_type: ToolErrorType` and a `message`. 

### 2. `src/google/adk/telemetry/tracing.py` [MODIFY]
Update the telemetry logic to extract and record the `error.type` if it exists.
*   Update `trace_tool_call` or introduce a new helper to capture `error_type` from an exception.
*   Wait, `trace_tool_call` only runs *after/during* the span. It currently doesn't accept the `Exception` object. The exception is caught in `llm_flows/functions.py`.
*   We will modify `trace_tool_call` to accept an optional `error: Exception | None = None` argument.
*   Inside `trace_tool_call`, if `error` is provided and it has an `error_type` attribute (e.g., it is a `ToolExecutionError`), we will set `span.set_attribute(ERROR_TYPE, error.error_type.value)`. 
*   **Note:** OpenTelemetry Semantic Conventions define `error.type` as `ERROR_TYPE` in `opentelemetry.semconv._incubating.attributes.error_attributes`. The type is a `string`, but per the Semantic Conventions: "If a specific domain defines its own set of error identifiers (such as HTTP or RPC status codes), it’s RECOMMENDED to... Set `error.type` to capture all errors, regardless of whether they are defined within the domain-specific set or not." Therefore, using `500` or `timeout` is fully compliant.

### 3. `src/google/adk/flows/llm_flows/functions.py` [MODIFY]
*   Update `_execute_single_function_call_async` to pass the caught exception downward to the trace context so it can be recorded.
*   Since `trace_tool_call` is called in the `finally` block, we will capture the exception in a local variable `caught_error` within the `try/except` block and pass it into `trace_tool_call(..., error=caught_error)`.

### 4. Retrofitting Existing Tools [FUTURE/END]
Once the core capability is landed and tested, existing tools within the framework should be audited and retrofitted to utilize this new exception for standard error reporting.
*   **API/Network Tools**: Tools that make network requests (e.g., `openapi_tool`, `google_search_tool`, `bigquery_tool`) should catch underlying SDK/HTTP exceptions (`requests.exceptions.Timeout`, etc.) and re-raise them wrapped in a `ToolExecutionError` with the appropriate `error_type` (e.g., `TIMEOUT` or `500`).
*   **Validation**: Tools performing internal validation should raise `ToolExecutionError` with `400` (Bad Request) or similar types when inputs fail schema or logic checks.
*   **Proof of Concept**: We will pick at least one simple tool (e.g., `example_tool.py` or a straightforward search tool) to demonstrate this retrofit as part of the initial PR.

## Verification Plan

### Automated Tests
*   **`tests/unittests/telemetry/test_spans.py`**:
    *   Mock a tool throwing a `ToolExecutionError(error_type=ToolErrorType.INTERNAL_SERVER_ERROR)` and assert that `span.set_attribute(ERROR_TYPE, '500')` is called on the mock span object.
    *   Mock a tool throwing a `ToolExecutionError(error_type=ToolErrorType.TIMEOUT)` and assert that `span.set_attribute(ERROR_TYPE, 'timeout')` is called.
    *   Mock a tool throwing a standard built-in `ValueError` and assert that `span.set_attribute(ERROR_TYPE, ...)` is either not called or set to the exception type name ('ValueError') as a fallback, depending on ADK's desired behavior for unhandled python errors.
*   Run the specific unit tests:
```bash
pytest tests/unittests/telemetry/test_spans.py
pytest tests/unittests/flows/llm_flows/test_functions.py
```

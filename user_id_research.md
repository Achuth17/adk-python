# Adding `user.id` to ADK Telemetry

## Objective
The goal is to enrich the OpenTelemetry (OTel) spans within the `adk-python` framework with the `user.id` metric based on the `USER_ID` semantic convention. This will improve trace partitioning and allow observability tools to filter agent flows by their interacting user.

---

## 1. Where Does `user.id` Originate?
The `user.id` originates from the developer or application integrating the ADK. When an application invokes the agent (e.g., via a FastAPI endpoint or background worker), the developer must pass the user identifier into the runner:

```python
events = runner.run_async(
    user_id="user_12345",      # Provided by the integrating application
    session_id="session_abc", 
    new_message="Hello agent!"
)
```

1. **Session Retrieval**: Inside `runner.run_async` (`runners.py`), this ID is passed into `_get_or_create_session(user_id=user_id, session_id=session_id)`. The `SessionService` attaches this ID to the `Session` object in memory or in the database.
2. **Context Wrapping**: This populated `Session` is then wrapped inside an `InvocationContext` via `_setup_context_for_new_invocation`.
3. **Agent Flow**: This `InvocationContext` (`ctx`) is propagated throughout the entire Reason-Act loop, meaning `ctx.session.user_id` is globally accessible to agents, tools, and the LLM models.

---

## 2. Current State of Telemetry in ADK
ADK imports the standard generative AI attributes and standard `USER_ID` convention in `src/google/adk/telemetry/tracing.py`:
```python
from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID
```
However, currently, it only applies generative AI attributes like `GEN_AI_OPERATION_NAME`, `GEN_AI_AGENT_NAME`, and `GEN_AI_CONVERSATION_ID`, completely ignoring the imported `USER_ID`.

---

## 3. Implementation Plan

To correctly instrument the `user.id` across the lifecycle, we must attach the attribute securely at both the root invocation level (when the request enters the runner) and the agent invocation level (when the individual agent processes it).

### Step 1: Update the Root Invocation Span (`runners.py`)
Modify `_run_with_trace` inside the `run_async` method to explicitly set the `USER_ID` on the root span. Since `user_id` is an argument directly available here, no context extraction is needed.

**File: `src/google/adk/runners.py`**
```python
async def _run_with_trace(
    new_message: Optional[types.Content] = None,
    invocation_id: Optional[str] = None,
) -> AsyncGenerator[Event, None]:
    with tracer.start_as_current_span('invocation') as span:
        from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID
        span.set_attribute(USER_ID, user_id)
        # ... fetch session, set up context, and execute ...
```

### Step 2: Update Agent Spans (`tracing.py`)
Modify the global `trace_agent_invocation` helper so that every discrete agent span captures the user identifier as it takes over execution from the context.

**File: `src/google/adk/telemetry/tracing.py`**
```python
def trace_agent_invocation(
    span: trace.Span, agent: BaseAgent, ctx: InvocationContext
) -> None:
    # Required
    span.set_attribute(GEN_AI_OPERATION_NAME, 'invoke_agent')

    # Conditionally Required
    span.set_attribute(GEN_AI_AGENT_DESCRIPTION, agent.description)
    span.set_attribute(GEN_AI_AGENT_NAME, agent.name)
    span.set_attribute(GEN_AI_CONVERSATION_ID, ctx.session.id)
    
    # New Metric:
    span.set_attribute(USER_ID, ctx.session.user_id)
```

---

## 5. How local `adk web` `/run` endpoint parses `user.id`

If you are hosting the ADK application locally using the `adk web` CLI command, the local FastAPI server directly exposes a `/run` POST endpoint.

This endpoint maps the JSON payload you send over HTTP to a Pydantic object called `RunAgentRequest` (found in `cli/adk_web_server.py`):

```python
class RunAgentRequest(common.BaseModel):
  app_name: str
  user_id: str
  session_id: str
  new_message: Optional[types.Content] = None
  streaming: bool = False
  state_delta: Optional[dict[str, Any]] = None
  invocation_id: Optional[str] = None
```

When you perform the `curl` POST request:
```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
        "app_name": "your_agent_name",
        "user_id": "test_user",
        "session_id": "session_123",
        "new_message": {
          "role": "user",
          "parts": [{"text": "What is the weather today?"}]
        }
      }'
```

1. **FastAPI Route Validation**: The FastAPI route at `@app.post("/run")` intercepts the request body and validates it against `RunAgentRequest`.
2. **Object Creation**: It instantiates `req = RunAgentRequest(...)`, setting `req.user_id = "test_user"`.
3. **Runner Invocation**: It then directly executes the runner:
    ```python
    runner.run_async(
        user_id=req.user_id,
        session_id=req.session_id,
        new_message=req.new_message,
        # ...
    )
    ```
4. **Trace Injection**: Once again, because the entry point passes `req.user_id` down into the runner, the `"test_user"` identifier seamlessly becomes the OpenTelemetry `user.id` for the entire invocation trace exactly as it does for Reasoning Engine deployments.

---

## 6. Telemetry Span Design Decisions

In Agentic frameworks like ADK, OpenTelemetry traces are designed around nested granularity. This aligns with the [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/). We are capturing `user.id` at two distinct levels of the trace hierarchy:

### 1. The Root Invocation Span (`runners.py`)
This span (`tracer.start_as_current_span('invocation')`) acts as an outer container representing the **entire lifecycle of a user's single request** to the application. 
- Setting `USER_ID` here allows operators to easily filter top-level traces in observability dashboards. 
- If a user reports an error, an operator can search for `user.id: "..."` and instantly see the full duration and status of that user's overall request across the entire system.

### 2. The Agent Invocation Span (`tracing.py`)
Complex queries often delegate work to multiple different agents beneath the scenes (e.g., a root agent handing off to a specialized sub-agent). Each time an individual agent executes, `trace_agent_invocation` fires to create a child span.
- Setting `USER_ID` here ensures that if telemetry data from specific agents is aggregated or analyzed independently from the parent trace, the user context isn't lost.
- This allows for granular analytics, such as identifying if a specific agent is performing slowly for a particular user subset.

---

## 7. Verification Plan

### Automated Tests
To ensure the `user.id` attribute is correctly attached without regressions, we will add test cases targeting the two specific components we modified:

#### 1. testing `runners.py` (root invocation logic)
**File**: `tests/unittests/test_runners.py`
We need to verify that `runner.run_async` successfully creates an `invocation` span and assigns the `USER_ID`. Since `runners.py` uses its own imported `tracer`, we will:
1. Use `unittest.mock.patch` to mock `google.adk.runners.tracer`.
2. Execute a simple `runner.run_async(...)` with a `user_id`.
3. Assert that `tracer.start_as_current_span('invocation')` was called.
4. Assert that `span.set_attribute(USER_ID, "test_user")` was called on the mock span object returned by the tracer manager.

#### 2. testing `tracing.py` (agent invocation logic)
**File**: `tests/unittests/telemetry/test_spans.py`
This file already contains a robust set of tests for `trace_agent_invocation`. We will:
1. Locate the existing `test_trace_agent_invocation` function.
2. It already sets up an `InvocationContext` containing a populated `session` object with a dummy `user_id`.
3. We will add an extra `mock.call(USER_ID, invocation_context.session.user_id)` to the `expected_calls` list.
4. The test logic will implicitly assert that our updated `trace_agent_invocation` successfully pulls the user ID from the context and applies it to the span attributes.

### Manual Verification
1. Run local tests: `pytest tests/unittests/test_runners.py` and `pytest tests/unittests/telemetry/test_spans.py`
2. Confirm visually by running the `SqliteSpanExporter` locally (e.g., using `adk web` or a sample agent), querying the `spans` database trace attributes, and ensuring `user.id` corresponds appropriately to the instantiated user.

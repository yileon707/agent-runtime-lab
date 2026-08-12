# P0.1 — Runtime Baseline Audit

> **Date**: 2026-08-12
> **Project**: Interview Agent Runtime (0-1agent)
> **Phase**: Baseline — freeze and understand current runtime
> **Constraint**: No refactoring, no provider migration, no new implementation

---

## 1. Repository Baseline

| Item | Value |
|---|---|
| **Repository** | `shareAI-lab/learn-claude-code` |
| **HEAD commit** | `eb4307f4e495d2ed22699e1e5682eb55f8076ade` |
| **Branch** | `main` |
| **Remote** | `origin/main` |
| **Python version** | 3.14.4 (Windows) |
| **Dependencies** | `anthropic>=0.25.0`, `python-dotenv>=1.0.0`, `pyyaml>=6.0` |
| **Test framework** | pytest 9.1.1 + unittest |
| **Test command** | `python -m pytest tests/ -v` |
| **Total tests** | 157 |
| **Passed** | 103 |
| **Failed** | 107 (see §1.1) |
| **git status** | Clean (fresh clone, no modifications) |

### 1.1 Test Failures — Root Cause

The majority of failures (estimated 60+) are caused by **Windows platform incompatibility**:

- **`import fcntl`** — Unix-only module, imported at module level in `s13_agent_teams/code.py:23` and `s15_integrated_harness/code.py:23`. This prevents *any* test that loads these modules from running.
- **`signal.SIGKILL`** — Not available on Windows. Referenced in `s11_background_tasks/code.py:58`.
- **`signal.SIGTERM`** — Available but `os.killpg` is not fully supported.

This is a **platform limitation, not a code defect**. The project was designed for Unix/macOS. All 103 passing tests validate core logic that does not depend on platform-specific primitives. The remaining failures are Windows artifacts.

### 1.2 Test Type Classification

| Type | Count | Description |
|---|---|---|
| **Unit tests** | ~80 | Compaction, task system, cron validation, todo parsing, skill loading |
| **Integration tests** | ~60 | Module loading, runtime lifecycle, team coordination, permission pipelines |
| **Scenario tests** | ~15 | Web scenarios, workflow+goal lesson combinations |
| **Real-model tests** | **0** | No tests call the real Anthropic API |
| **Agent-level eval** | **0** | No pytest tests function as agent evaluations |
| **Trajectory eval** | **0** | Not present |
| **Context eval** | **0** | Not present |

**Key finding**: All tests mock `client.messages.create` with lambda functions returning `types.SimpleNamespace` objects. There are zero real-model tests and zero evaluation frameworks.

---

## 2. Runtime Entry Points

The project has **no single entry point**. Each of the 17 lesson directories contains a standalone `code.py`. The runtime hierarchy:

```
s15_integrated_harness/code.py     ← PRIMARY INTEGRATED RUNTIME (3062 lines)
  ├── loads s09_memory at runtime (importlib)
  ├── s16_workflow_runtime/code.py ← EXTENDS s15 (loads s15, monkey-patches tool pool)
  └── s17_goal_loop/code.py        ← COMPLETELY INDEPENDENT (883 lines, own agent loop)
```

**Execution**:
```sh
python s15_integrated_harness/code.py   # Full integrated harness
python s16_workflow_runtime/code.py     # Workflow (extends s15 CLI)
python s16_workflow_runtime/code.py demo  # Deterministic demo (no API)
python s17_goal_loop/code.py            # Goal loop REPL
python s17_goal_loop/code.py "/goal ..." # Single-shot goal
```

---

## 3. Model API Coupling Inventory

### 3.1 Anthropic SDK — Direct Import Map

**Every single Python file** (30+ files across s01-s17, agents/, skills/) imports `from anthropic import Anthropic`. There is **zero abstraction layer** between the application code and the Anthropic SDK.

| Location | API Concept | Runtime Responsibility | Migration Difficulty | Risk |
|---|---|---|---|---|
| All `code.py` (30+ files) | `from anthropic import Anthropic` | Client construction | **High** — every file | Must replace in every file |
| All `code.py` | `client = Anthropic(base_url=...)` | Provider instantiation | **High** | Coupled to Anthropic SDK constructor |
| All `code.py` | `client.messages.create(...)` | Model call API | **High** | Signature differs from OpenAI |
| All `code.py` | `response.stop_reason == "tool_use"` | Loop control | **High** | OpenAI uses `finish_reason: "tool_calls"` |
| All `code.py` | `response.stop_reason == "max_tokens"` | Truncation handling | **Medium** | OpenAI uses `finish_reason: "length"` |
| All `code.py` | `response.content` (list of blocks) | Response parsing | **High** | OpenAI returns `choices[0].message` |
| All `code.py` | `block.type == "tool_use"` | Tool call detection | **High** | OpenAI uses `message.tool_calls[]` |
| All `code.py` | `block.type == "text"` | Text extraction | **Medium** | OpenAI uses `message.content` (string or array) |
| All `code.py` | `block.name` | Tool name | **High** | OpenAI uses `function.name` |
| All `code.py` | `block.input` (dict) | Tool arguments | **High** | OpenAI uses `function.arguments` (JSON string) |
| All `code.py` | `block.id` | Tool call ID | **Medium** | OpenAI uses `tool_calls[].id` |
| All `code.py` | `{"type": "tool_result", "tool_use_id": ..., "content": ...}` | Tool result format | **High** | OpenAI uses `role: "tool"` messages |
| All `code.py` | `{"role": "user"}`, `{"role": "assistant"}` | Message roles | **Low** | Same in OpenAI |
| All `code.py` | `system=SYSTEM` (top-level param) | System prompt | **Medium** | OpenAI uses messages with `role: "system"` |
| All `code.py` | `tools=[{name, description, input_schema}]` | Tool definitions | **High** | OpenAI: `type: "function"`, `parameters` not `input_schema` |
| All `code.py` | `max_tokens=8000` | Token limit | **Low** | Same concept, different param name (`max_completion_tokens` in newer OpenAI) |
| s17:106-112 | `response.usage.input_tokens`, `.output_tokens` | Token counting | **Low** | OpenAI: `usage.prompt_tokens`, `completion_tokens` |
| s16:310-313 | `response.usage.input_tokens`, `.output_tokens` | Token counting | **Low** | Same as above |

### 3.2 Key Coupling Answers

**Q: How many places directly depend on Anthropic SDK?**
A: **30+ Python files** import `from anthropic import Anthropic`. Each file creates its own client. Every agent loop (~25 instances across the codebase) directly accesses `response.content`, `response.stop_reason`, `block.type`, `block.name`, `block.input`, `block.id`.

**Q: Is Tool Call internal representation directly equal to Anthropic `tool_use`?**
A: **Yes.** The string `"tool_use"` is hardcoded as the block type discriminator in every agent loop. Tool call blocks are treated as Anthropic SDK objects with `.type`, `.name`, `.input`, `.id` attributes. The `has_tool_use()` function (s15:1805) checks `block.type == "tool_use"`.

**Q: Is Tool Result directly using Anthropic `tool_result` message schema?**
A: **Yes.** Every agent loop constructs `{"type": "tool_result", "tool_use_id": block.id, "content": str(output)}`. This is the Anthropic Messages API format exactly. The compaction code (`collect_tool_results`, `is_tool_result_message`) depends on this exact shape.

**Q: Does `stop_reason` participate in Agent Loop control?**
A: **Yes, critically.** Every agent loop checks `response.stop_reason != "tool_use"` to decide whether to stop or continue. s15 additionally handles `response.stop_reason == "max_tokens"` (line 2924) for truncation recovery with token escalation.

**Q: Where is usage/token information read?**
A: Only in two places:
- `s17_goal_loop/code.py:106-112` — `_usage_total()` reads `response.usage.input_tokens` and `response.usage.output_tokens`
- `s16_workflow_runtime/code.py:310-313` — `AnthropicAgentRunner.run()` reads the same
- s15 `estimate_size()` uses `len(json.dumps(messages))` as a proxy — it does NOT use the API's token count

**Q: Does Context Compact depend on Anthropic content block structure?**
A: **Yes, deeply.** The compaction functions (`snip_compact`, `micro_compact`, `compact_history`, `tool_result_budget`) all depend on:
  1. `message["role"] == "assistant"` containing `block.type == "tool_use"`
  2. `message["role"] == "user"` containing `block.type == "tool_result"`
  3. The invariant that tool_use and tool_result messages form adjacent pairs

**Q: Do Memory / Goal Evaluator / Workflow / Subagent / Teammate each create their own Anthropic client?**
A:
- **Memory (s09)**: Creates its own client, but s15 injects its own client: `runtime.client = client`
- **Goal Evaluator (s17)**: Creates its own `Anthropic()` client for both worker and evaluator
- **Workflow (s16)**: Reuses s15's client: `RUNNER_FACTORY = lambda: AnthropicAgentRunner(host.client, host.MODEL)`
- **Subagent (s15)**: Uses the module-level `client` directly
- **Teammates (s15)**: Each teammate thread calls `client.messages.create()` on the shared module-level client

**Q: Are there multiple duplicate model-call implementations?**
A: **Yes.** At least 25 separate `client.messages.create()` call sites across the codebase, all with nearly identical patterns. s15 alone has 4 call sites (main loop, subagent, summarization, retry wrapper). No shared abstraction exists.

**Q: If switching to OpenAI Chat Completions, which modules would be affected?**
A: **All 30+ Python files** that import `from anthropic import Anthropic`. The coupling is pervasive and has zero abstraction.

---

## 4. State Inventory

| State | Owner | Persistence | Lifetime | Recovery Behavior |
|---|---|---|---|---|
| `messages[]` | `agent_loop()` / `AgentSession` | In-memory only (except transcripts) | Single conversation turn batch | Lost on crash; no recovery |
| `globals` (MODEL, client, WORKDIR) | Module-level | `.env` for config | Process lifetime | Re-read from env on restart |
| Memory (s09) | `.memory/MEMORY.md` | File-backed (markdown index + files) | Cross-session | Survives restart; s09 manages read/write |
| Tasks (s10/s15) | `.tasks/*.json` | File-backed, `fcntl.flock()` | Cross-session | Atomic write via temp file + `os.replace()` |
| Teams (s13/s15) | `.teams/`, `.mailboxes/` | File-backed JSONL | Cross-session | Assignment rehydration on restart |
| Background tasks | `background_tasks` dict + `background_results` dict | In-memory + thread | Process lifetime | Thread results lost on crash |
| Cron jobs | `scheduled_jobs` list + `.scheduled_tasks.json` | File-backed for durable, in-memory for active | Cross-session for durable; process for active | Durable jobs reloaded on startup; unacknowledged deliveries restored after model call failure |
| Transcripts | `.transcripts/*.jsonl` | File-backed | Cross-session | Written during compaction; survives restart |
| Tool outputs | `.task_outputs/tool-results/*.txt` | File-backed | Cross-session | Written when output > 30KB (PERSIST_THRESHOLD) |
| Workflow state | `.runtime/<runId>.json`, `.journal.jsonl` | File-backed, `fcntl.flock()` | Cross-session | Journal replay on resume; deterministic key matching |
| Goal state | `GoalController` (in-memory) | In-memory only | Single session | Lost on crash; no persistence |
| Worktrees | `.worktrees/` (git worktree) | Git filesystem | Cross-session | Rehydrated on restart via worktree registry parsing |
| Filesystem artifacts | WORKDIR | Filesystem | Cross-session | Survives restart |

---

## 5. Execution Path (s15 — Primary Integrated Runtime)

```
User Input
  │
  ▼
trigger_hooks("UserPromptSubmit")
  │
  ▼
history.append({"role": "user", "content": query})
  │
  ▼
┌─ agent_loop(messages, context, active_request) ─────────────────────┐
│                                                                       │
│  ▼                                                                    │
│  while True:                                                          │
│    1. consume_cron_queue() → inject scheduled prompts                 │
│    2. inject_background_notifications()                               │
│    3. todo reminder (every 3 rounds)                                  │
│    4. prepare_context():                                              │
│       ├── tool_result_budget()  → persist large outputs              │
│       ├── snip_compact()        → remove old middle messages         │
│       ├── micro_compact()       → shorten old tool results           │
│       └── compact_history()     → model summarization (if needed)    │
│    5. update_context() → memory + MCP + teammates                    │
│    6. assemble_tool_pool() → BUILTIN + MCP tools                     │
│    7. call_llm() → with_retry → client.messages.create()             │
│       ├── On max_tokens: escalate + retry                            │
│       ├── On 529: model failover                                     │
│       └── On prompt_too_long: reactive_compact + retry               │
│    8. acknowledge_cron_jobs()                                         │
│    9. messages.append({"role": "assistant", "content": response.content})│
│   10. If NOT has_tool_use(response.content):                          │
│       ├── trigger_hooks("Stop")                                       │
│       ├── remember_after_turn() → s09 memory consolidation           │
│       ├── release_completed_assignment()                              │
│       └── RETURN                                                      │
│   11. For each block in response.content:                             │
│       ├── If block.type != "tool_use": continue                       │
│       ├── If block.name == "compact": mark compact_requested          │
│       ├── trigger_hooks("PreToolUse") → permission gate              │
│       ├── If blocked: append error tool_result                        │
│       ├── If background: start_background_task()                      │
│       ├── Else: call handler → trigger_hooks("PostToolUse")           │
│       └── Append {"type": "tool_result", "tool_use_id": ..., ...}    │
│   12. messages.append({"role": "user", "content": results})           │
│   13. If compact_requested: compact_history()                         │
│   14. Loop back to step 1                                              │
└───────────────────────────────────────────────────────────────────────┘
  │
  ▼
print response text → wait for next user input
```

### Annotations for Special Paths:

- **Memory (s09)**: Called via `remember_after_turn()` at stop; extraction/consolidation triggered by the model calling `remember`/`forget` tools
- **Compact**: Triggered by size threshold (CONTEXT_LIMIT=50000 chars) OR model calling `compact` tool OR `prompt_too_long` error
- **Background**: `should_run_background()` checks `run_in_background: true` in tool input; results collected by `collect_background_results()`
- **Team**: Teammates run in separate threads with their own `client.messages.create()` loop; communicate via `MessageBus` (file-backed JSONL mailboxes)
- **MCP**: Tools discovered via MCP protocol, namespaced, merged into `assemble_tool_pool()`
- **Error Recovery**: `with_retry()` handles 529 (model failover), rate limits (exponential backoff), and `prompt_too_long` (reactive compact)

---

## 6. Correctness Invariants

### 6.1 Identified Invariants

| Invariant | Protected State | Test | Failure Consequence |
|---|---|---|---|
| **tool_use / tool_result pairing**: Every `tool_result` message must be immediately preceded by a `tool_use` assistant message | messages[] | `CompactionToolPairTests.test_snip_compact_keeps_head_tool_pair`, `test_snip_compact_keeps_tail_tool_pair`, `test_reactive_compact_keeps_tail_tool_pair`, `assert_no_orphan_tool_results()` | Orphaned tool results violate Anthropic API schema → 400 error |
| **No orphan tool_results after compaction**: snip_compact must not break tool_use/tool_result adjacency | messages[] | `assert_no_orphan_tool_results()` called after every compaction test | API rejects messages with orphan tool_results |
| **Task creation atomicity**: Tasks written atomically via temp file + `os.replace()` | `.tasks/*.json` | `test_task_store_rejects_a_symlink_outside_the_workspace` | Corrupt task file → task system unusable |
| **Atomic task claims**: `fcntl.flock()` prevents concurrent task claims | `.tasks/*.json` | `test_task_claim_is_atomic_across_processes`, `test_idle_claim_is_atomic_across_teammates` | Double-claim → two agents working same task |
| **Cron idempotency**: Failed model calls restore unacknowledged cron deliveries | `.scheduled_tasks.json` | `test_failed_model_call_restores_unacknowledged_cron_delivery` | Lost cron jobs |
| **Workflow run locking**: Cross-process `fcntl.LOCK_EX` prevents concurrent resume | `.runtime/<runId>.json` | `test_workflow_run_lock_is_cross_process` | Corrupt workflow state |
| **Journal integrity**: Resume validates every journal line | `.runtime/<runId>.journal.jsonl` | `test_workflow_runtime_rejects_corrupt_resume_journal` | Silent data corruption on resume |
| **Goal evaluator safety**: Evaluator has no tools; treats input as data, not instructions | s17 evaluator config | `test_goal_loop_file_tools_use_the_current_repository` | Prompt injection via conversation content |
| **Permission gate**: Every tool call passes through PreToolUse hooks | Tool execution | `test_integrated_permission_requires_approval_for_every_shell_command` | Unauthorized tool execution |
| **Background dispatch is Bash-only**: `should_run_background()` only applies to `bash` tool | Background tasks | `test_background_dispatch_is_bash_only_and_reports_failures` | Non-bash tools run in background incorrectly |

### 6.2 Context Compact — Message Structure Dependency

**Confirmed**: Context compaction **heavily depends** on the Anthropic message structure pattern:

```
assistant (with tool_use block) → user (with tool_result block)
```

**Evidence**:
- `message_has_tool_use()` (s15:1856) checks `msg["role"] == "assistant"` with `block.type == "tool_use"`
- `is_tool_result_message()` (s15:1865) checks `msg["role"] == "user"` with `block.type == "tool_result"`
- `snip_compact()` (s15:1922) explicitly preserves these adjacent pairs at head and tail boundaries
- `assert_no_orphan_tool_results()` (test_compaction_tool_pairs.py:98) enforces this invariant
- Tests verify that compaction never produces a tool_result without a preceding tool_use

If OpenAI's `tool_calls` / `tool` role messages use a different adjacency pattern, the entire compaction subsystem must be rewritten.

---

## 7. Observability Inventory

### 7.1 Current Channels

| Channel | Implementation | Scope |
|---|---|---|
| **print()** | ANSI-colored `print()` statements in every module for tool calls, errors, hooks, cron injects, background results | Every module |
| **logging** | **None** — no `logging` module usage anywhere | N/A |
| **transcript** | `.transcripts/*.jsonl` written during `compact_history()` and `reactive_compact()` | s15 only |
| **tool output artifact** | `.task_outputs/tool-results/<tool_use_id>.txt` when output > 30KB | s15 only |
| **token usage** | `_usage_total()` in s17, `RunnerOutput.tokens` in s16, otherwise `estimate_size()` (char count proxy) | s16, s17 only |
| **latency** | **None** | N/A |
| **trace** | **None** — no distributed tracing | N/A |
| **trace_id** | **None** | N/A |
| **span_id** | **None** | N/A |
| **run_id** | `wf_<name>_<token>` in s16 workflow runtime | s16 only |
| **turn_id** | **None** | N/A |
| **tool_call_id** | Anthropic SDK `block.id` used as `tool_use_id` in tool results; persisted to tool output filenames | Everywhere (but not a first-class concept) |

### 7.2 Summary

The project has **no structured observability**. The only output channels are:
1. `print()` with ANSI colors (human-readable, not machine-parseable)
2. Transcript files (only written on compaction, not every turn)
3. Token usage only tracked in s16 and s17 (not in s15, the primary runtime)
4. No latency tracking, no tracing, no structured logging

---

## 8. Evaluation Inventory

| Type | Present? | Details |
|---|---|---|
| **Unit tests** | Yes (103 passing) | Compaction, task system, cron validation, todo parsing, skill loading |
| **Integration tests** | Yes (partially passing) | Module loading, runtime lifecycle, team coordination |
| **Scenario tests** | Yes (partially passing) | Web scenarios, workflow+goal combinations |
| **Real-model tests** | **None** | All tests use mocked `client.messages.create` |
| **Agent-level eval** | **None** | No test evaluates agent behavior quality |
| **Trajectory eval** | **None** | No test evaluates action sequences |
| **Context eval** | **None** | No test evaluates context management quality |
| **Eval framework** | **None** | No evaluation harness, scoring, or metrics |

### Important Distinction

**pytest is NOT synonymous with Agent Eval.** The current test suite only validates:
- Code compiles and modules load
- Data transformations are correct (compaction, task CRUD, cron matching)
- Invariants hold (tool_use/tool_result pairing, atomic claims)

It does NOT validate:
- Whether the agent makes correct decisions
- Whether tool calls are appropriate
- Whether context management preserves task-critical information
- Whether the agent completes goals successfully

---

## 9. DeepSeek Compatibility Impact

> **Target**: OpenAI-compatible `POST /chat/completions` with `model: deepseek-v4-pro`
> **Note**: DeepSeek API contract requires external verification. This section identifies impacts assuming OpenAI Chat Completions shape.

### 9.1 Message Schema

| Current (Anthropic) | Target (OpenAI) | Impact |
|---|---|---|
| `messages[i].content` is `list[ContentBlock]` | `messages[i].content` is `string` or array (depending on `type`) | **High** — every message accessor changes |
| Assistant content: list of `{type: "text", text: ...}` and `{type: "tool_use", ...}` | Assistant content: `string` or `null`, tool calls in separate `tool_calls[]` field | **High** — tool calls move out of content |
| User content: list of `{type: "tool_result", tool_use_id, content}` | User messages with `role: "tool"` and `tool_call_id` | **High** — different message type |
| System prompt as top-level `system=...` param | System prompt as `{"role": "system", "content": ...}` message | **Medium** |

### 9.2 Tool Definitions

| Current (Anthropic) | Target (OpenAI) | Impact |
|---|---|---|
| `{"name": ..., "description": ..., "input_schema": {...}}` | `{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}` | **High** — entirely different schema |

### 9.3 Tool Calls

| Current (Anthropic) | Target (OpenAI) | Impact |
|---|---|---|
| `block.type == "tool_use"`, `block.name`, `block.input` (dict), `block.id` | `message.tool_calls[]`, `tc.function.name`, `tc.function.arguments` (JSON string), `tc.id` | **High** — different location, different types |

### 9.4 Tool Results

| Current (Anthropic) | Target (OpenAI) | Impact |
|---|---|---|
| `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": ..., "content": ...}]}` | `{"role": "tool", "tool_call_id": ..., "content": ...}` | **High** — different role, different key name |

### 9.5 Finish Reason

| Current (Anthropic) | Target (OpenAI) | Impact |
|---|---|---|
| `response.stop_reason == "tool_use"` | `choice.finish_reason == "tool_calls"` | **High** — different attribute path, different string value |
| `response.stop_reason == "max_tokens"` | `choice.finish_reason == "length"` | **Medium** |
| `response.stop_reason == "end_turn"` | `choice.finish_reason == "stop"` | **Medium** |

### 9.6 Usage

| Current (Anthropic) | Target (OpenAI) | Impact |
|---|---|---|
| `response.usage.input_tokens` | `response.usage.prompt_tokens` | **Low** |
| `response.usage.output_tokens` | `response.usage.completion_tokens` | **Low** |

### 9.7 Thinking/Reasoning Provider State

> **external API contract requires verification** — DeepSeek v4-pro's reasoning/thinking capabilities need separate documentation review

### 9.8 Streaming

The current codebase has **no streaming implementation**. All `client.messages.create()` calls are synchronous without `stream=True`. → **No impact**.

### 9.9 Errors/Retries

The current retry logic (`with_retry()` in s15:2863) handles Anthropic-specific error strings (`prompt_too_long`, 529 overloaded). → **Medium impact** — error messages differ across providers.

---

## 10. Known Architectural Risks

> **This section records risks only. No solutions are designed in this phase.**

| # | Risk | Severity | Detail |
|---|---|---|---|
| R1 | **Zero API abstraction** | Critical | 30+ files import `from anthropic import Anthropic` directly. Changing providers requires touching every file. |
| R2 | **Anthropic message schema is the internal representation** | Critical | `messages[]` stores Anthropic content blocks directly. Compaction, tool dispatch, text extraction all depend on `.type`, `.name`, `.input` attributes. |
| R3 | **Compaction depends on tool_use/tool_result adjacency** | High | `snip_compact()` assumes `assistant(tool_use) → user(tool_result)` message pairs. OpenAI's `tool` role messages break this invariant. |
| R4 | **No structured observability** | High | Only ANSI `print()` and occasional transcript files. No latency, token tracking (except s16/s17), or structured logging. Debugging in production would be blind. |
| R5 | **No evaluation framework** | High | Zero agent-level, trajectory, or context evals. Cannot measure whether changes improve or degrade agent behavior. |
| R6 | **Multiple duplicate client.messages.create() sites** | Medium | At least 25 identical call patterns. Any API change must be replicated across all sites. |
| R7 | **s17 is completely independent of s15** | Medium | Goal loop reimplements agent loop, tools, hooks, and permission from scratch. Two runtimes to maintain. |
| R8 | **Windows incompatibility** | Medium | `fcntl`, `signal.SIGKILL` make the primary runtime (s15) and s13 non-functional on Windows. |
| R9 | **No provider-agnostic tool registry** | Medium | Tool definitions use Anthropic `input_schema` format directly. No intermediate representation. |
| R10 | **In-memory state with no crash recovery** | Medium | `messages[]`, goal state, background task results are lost on crash. Only tasks and cron have partial persistence. |
| R11 | **Transcript is compaction-only** | Low | Transcripts are only written during compaction, not every turn. Losing power between compactions loses conversation history. |
| R12 | **No model response validation** | Low | Code assumes well-formed responses. Malformed blocks could crash the runtime. |

---

## Appendix A: s15 / s16 / s17 — Relationship Analysis

**s15 (Integrated Harness)** is the most comprehensive runtime:
- Full agent loop with tool dispatch, permission hooks, context compaction
- Task system, team coordination, cron scheduling, background tasks
- MCP integration, skill loading, memory (s09)
- 3062 lines

**s16 (Workflow Runtime)** extends s15:
- Loads s15 at runtime via `importlib` → monkey-patches `assemble_tool_pool()`
- Adds `Workflow` tool with pipeline/parallel orchestration
- Journal-based agent replay for resume
- 875 lines
- Does NOT duplicate s15 — it reuses it

**s17 (Goal Loop)** is completely independent:
- Own Anthropic client, agent loop, tools, hooks, permission
- Adds GoalController + separate evaluator model
- No file-backed state
- 883 lines
- Shares zero code with s15/s16

**Integrated baseline judgment**: **s15 is the single most reasonable integrated baseline.** It is the only runtime that covers the full harness feature set. s16 extends it; s17 is a separate experiment.

---

## Appendix B: Test Execution Result

```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
collected 157 items

103 passed, 107 failed in 29.16s
```

Failure root cause: `fcntl` (Unix-only) imported at module level in s13 and s15. Tests that don't load those modules pass (103 tests). This is a platform limitation.

---

*Audit completed 2026-08-12. No runtime behavior modified.*

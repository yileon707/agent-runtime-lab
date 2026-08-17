"""Focused tests for TRACE V0.1a — the observer lifecycle event layer.

Covers the spec's A–G cases:

  A. correct lifecycle order (Run -> Cycle -> Iteration -> ... -> RunEnd)
  B. two cycles share a run_id but have distinct cycle_ids
  C. one cycle, multiple iterations -> unique, cycle-scoped iteration_ids
  D. a raising observer never breaks the loop
  E. an observer's return value is ignored (cannot block/stop)
  F. no observers registered -> behavior identical to an observer-less run
  G. Control Hooks (PreToolUse / Stop) semantics are unchanged

The Run/Cycle cases (B, D, E, F) exercise ``agent_operator.run_goal_loop``
offline with fake worker/evaluator, so they run on any platform. The Iteration
cases (A, C, G) drive ``s15.agent_loop`` with a scripted fake client; s15
imports ``fcntl`` (Linux only), so those are skipped elsewhere.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import fcntl  # noqa: F401  (s15 hard-depends on it)
    HAVE_FCNTL = True
except ImportError:
    HAVE_FCNTL = False

requires_fcntl = pytest.mark.skipif(
    not HAVE_FCNTL, reason="s15 imports fcntl (Linux only)"
)


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the observer first so both the operator and s15 share this one instance
# via sys.modules["agent_observer"].
agent_observer = _load_by_path("agent_observer", REPO_ROOT / "agent_observer.py")
operator = _load_by_path("agent_operator_under_test", REPO_ROOT / "agent_operator.py")


@pytest.fixture(autouse=True)
def _reset_observer():
    """Isolate observer subscriptions and identity state between tests."""
    agent_observer.clear_observers()
    agent_observer.begin_run()
    agent_observer.end_run()
    yield
    agent_observer.clear_observers()
    agent_observer.begin_run()
    agent_observer.end_run()


def _record(events):
    """Register an observer that appends every event to ``events``."""

    def callback(event):
        events.append(event)

    agent_observer.register_observer(callback)
    return events


# --- offline fakes for the operator control loop -----------------------------


def _fake_worker():
    calls = {"count": 0}

    def worker(history, context, active_request):
        calls["count"] += 1
        history.append({"role": "assistant", "content": f"work-{calls['count']}"})

    return worker, calls


def _fake_set_goal():
    state = {"goals": []}

    def set_goal(goal):
        state["goals"].append(goal)

    return set_goal, state


def _fake_evaluate(actions):
    queue = list(actions)

    def evaluate(history):
        action = queue.pop(0)
        return types.SimpleNamespace(action=action, reason=f"reason-{action}")

    return evaluate


# --- offline s15 loader (Linux only) ----------------------------------------


def _load_s15_offline(tmp: Path):
    """Load s15 with faked anthropic/dotenv/yaml and no network.

    ``remember_after_turn`` is stubbed so the Stop path does not issue the
    s09 memory LLM call; iteration lifecycle is independent of memory
    extraction, which is what these tests observe.
    """
    fake_anthropic = types.ModuleType("anthropic")

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = types.SimpleNamespace(create=None)

    fake_dotenv = types.ModuleType("dotenv")
    fake_yaml = types.ModuleType("yaml")
    setattr(fake_anthropic, "Anthropic", FakeAnthropic)
    setattr(fake_dotenv, "load_dotenv", lambda override=True: None)
    setattr(fake_yaml, "safe_load", lambda text: {})
    setattr(fake_yaml, "YAMLError", Exception)

    previous = {
        "anthropic": sys.modules.get("anthropic"),
        "dotenv": sys.modules.get("dotenv"),
        "yaml": sys.modules.get("yaml"),
    }
    previous_cwd = Path.cwd()
    previous_model = os.environ.get("MODEL_ID")
    previous_key = os.environ.get("ANTHROPIC_API_KEY")

    spec = importlib.util.spec_from_file_location(
        "s15_trace_under_test", REPO_ROOT / "s15_integrated_harness" / "code.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load s15")
    module = importlib.util.module_from_spec(spec)
    sys.modules["anthropic"] = fake_anthropic
    sys.modules["dotenv"] = fake_dotenv
    sys.modules["yaml"] = fake_yaml
    try:
        os.chdir(tmp)
        os.environ["MODEL_ID"] = "test-model"
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        spec.loader.exec_module(module)
        module.remember_after_turn = lambda _messages: None
        return module
    finally:
        os.chdir(previous_cwd)
        if previous_model is None:
            os.environ.pop("MODEL_ID", None)
        else:
            os.environ["MODEL_ID"] = previous_model
        if previous_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = previous_key
        for name, prev in previous.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _tool_use_response(name: str, tool_id: str):
    return types.SimpleNamespace(
        stop_reason="tool_use",
        content=[
            types.SimpleNamespace(type="tool_use", id=tool_id, name=name, input={})
        ],
    )


def _end_turn_response():
    return types.SimpleNamespace(stop_reason="end_turn", content=[])


# --- A. lifecycle order -----------------------------------------------------


@requires_fcntl
def test_a_full_lifecycle_order():
    with tempfile.TemporaryDirectory() as tmp:
        s15 = _load_s15_offline(Path(tmp))
        # One tool_use turn then a final end_turn -> two s15 iterations in one
        # cycle. "noop" is an unregistered tool, so the turn executes no real
        # handler (call_tool_handler returns "Unknown tool") and makes no LLM
        # call of its own.
        responses = [_tool_use_response("noop", "t1"), _end_turn_response()]
        s15.client.messages.create = lambda **kwargs: responses.pop(0)

        events = []
        _record(events)

        def worker(history, context, active_request):
            with s15.agent_lock:
                s15.agent_loop(history, context, active_request)

        result = operator.run_goal_loop(
            worker, _fake_set_goal()[0], _fake_evaluate(["achieved"]), "task", "goal"
        )

        assert result.status == "success"
        assert [e.name for e in events] == [
            "RunStart", "CycleStart",
            "IterationStart", "IterationEnd",
            "IterationStart", "IterationEnd",
            "CycleEnd", "RunEnd",
        ]

        run_ids = {e.run_id for e in events}
        assert None not in run_ids and len(run_ids) == 1  # fixed per run
        assert [e.iteration_id for e in events if e.name == "IterationStart"] == [
            "cycle-1.1", "cycle-1.2"
        ]


# --- B. two cycles, same run, distinct cycle ids ----------------------------


def test_b_two_cycles_share_run_id_but_not_cycle_id():
    events = []
    _record(events)

    worker, calls = _fake_worker()
    result = operator.run_goal_loop(
        worker, _fake_set_goal()[0], _fake_evaluate(["block", "achieved"]), "t", "g"
    )

    assert result.status == "success"
    assert result.worker_cycles == 2
    assert calls["count"] == 2

    starts = [e for e in events if e.name == "CycleStart"]
    ends = [e for e in events if e.name == "CycleEnd"]
    assert [e.cycle_id for e in starts] == ["cycle-1", "cycle-2"]
    assert [e.cycle_id for e in ends] == ["cycle-1", "cycle-2"]

    run_ids = {e.run_id for e in events}
    assert None not in run_ids and len(run_ids) == 1  # same run


# --- C. multiple iterations, unique cycle-scoped ids ------------------------


def test_iteration_ids_reset_each_cycle_and_do_not_cross():
    agent_observer.begin_run()
    agent_observer.begin_cycle()
    ids_cycle1 = [agent_observer.next_iteration_id() for _ in range(3)]
    agent_observer.begin_cycle()
    ids_cycle2 = [agent_observer.next_iteration_id() for _ in range(2)]
    agent_observer.end_run()

    assert ids_cycle1 == ["cycle-1.1", "cycle-1.2", "cycle-1.3"]
    assert ids_cycle2 == ["cycle-2.1", "cycle-2.2"]
    assert set(ids_cycle1).isdisjoint(set(ids_cycle2))


@requires_fcntl
def test_c_one_cycle_multiple_iterations_unique_ids():
    with tempfile.TemporaryDirectory() as tmp:
        s15 = _load_s15_offline(Path(tmp))
        responses = [_tool_use_response("noop", f"t{i}") for i in range(3)]
        responses.append(_end_turn_response())  # 3 tool turns + 1 final = 4 iterations
        s15.client.messages.create = lambda **kwargs: responses.pop(0)

        events = []
        _record(events)

        agent_observer.begin_run()
        agent_observer.begin_cycle()
        s15.agent_loop([], {}, "task")

        starts = [e for e in events if e.name == "IterationStart"]
        ends = [e for e in events if e.name == "IterationEnd"]
        assert len(starts) == 4
        assert len(ends) == 4

        ids = [e.iteration_id for e in starts]
        assert ids == ["cycle-1.1", "cycle-1.2", "cycle-1.3", "cycle-1.4"]
        assert len(set(ids)) == 4  # unique within the cycle
        assert {e.cycle_id for e in starts} == {"cycle-1"}  # all same cycle
        assert [e.iteration_id for e in ends] == ids  # ends pair with starts


# --- D. observer exceptions are isolated ------------------------------------


def test_emit_observer_isolates_exceptions_and_ignores_returns():
    def bad(event):
        raise RuntimeError("observer bug")

    def good(event):
        good.called += 1
        return 42  # arbitrary return, must be dropped

    good.called = 0
    agent_observer.register_observer(bad)
    agent_observer.register_observer(good)

    agent_observer.emit_observer(agent_observer.make_event("X"))  # must not raise
    assert good.called == 1  # a later observer still ran after an earlier raised


def test_d_raising_observer_does_not_break_loop():
    def boom(event):
        raise RuntimeError("observer bug")

    agent_observer.register_observer(boom)

    worker, calls = _fake_worker()
    result = operator.run_goal_loop(
        worker, _fake_set_goal()[0], _fake_evaluate(["block", "achieved"]), "t", "g"
    )

    assert result.status == "success"
    assert result.worker_cycles == 2
    assert calls["count"] == 2


# --- E. observer return value cannot block/stop -----------------------------


def test_e_observer_return_value_is_ignored():
    def returns_garbage(event):
        return "STOP"  # must be discarded, never acted on

    agent_observer.register_observer(returns_garbage)

    worker, calls = _fake_worker()
    result = operator.run_goal_loop(
        worker, _fake_set_goal()[0], _fake_evaluate(["achieved"]), "t", "g"
    )

    assert result.status == "success"
    assert result.worker_cycles == 1
    assert calls["count"] == 1


# --- F. no observer -> behavior identical -----------------------------------


def test_f_no_observer_behavior_unchanged():
    no_obs = operator.run_goal_loop(
        _fake_worker()[0], _fake_set_goal()[0], _fake_evaluate(["block", "achieved"]),
        "t", "g",
    )

    events = []
    _record(events)
    with_obs = operator.run_goal_loop(
        _fake_worker()[0], _fake_set_goal()[0], _fake_evaluate(["block", "achieved"]),
        "t", "g",
    )

    fields = ("status", "worker_cycles", "evaluations", "blocks", "iterations")
    assert tuple(getattr(no_obs, f) for f in fields) == tuple(
        getattr(with_obs, f) for f in fields
    )
    assert no_obs.status == "success"
    assert events  # an observer still sees events when registered


# --- G. Control Hooks semantics unchanged -----------------------------------


@requires_fcntl
def test_g_control_hooks_unchanged_with_observer():
    with tempfile.TemporaryDirectory() as tmp:
        s15 = _load_s15_offline(Path(tmp))
        stopped = []
        s15.register_hook(
            "PreToolUse", lambda block: ("DENIED" if block.name == "bash" else None)
        )
        s15.register_hook("Stop", lambda messages: stopped.append(1) or None)

        responses = [
            types.SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    types.SimpleNamespace(
                        type="tool_use", id="b1", name="bash",
                        input={"command": "ls"},
                    )
                ],
            ),
            _end_turn_response(),
        ]
        s15.client.messages.create = lambda **kwargs: responses.pop(0)

        events = []
        _record(events)

        history = []
        s15.agent_loop(history, {}, "test")

        assert stopped == [1]  # Stop hook fired
        # PreToolUse blocked the bash tool -> a DENIED tool_result is recorded.
        assert any(
            isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(
                isinstance(b, dict)
                and b.get("type") == "tool_result"
                and "DENIED" in str(b.get("content"))
                for b in m["content"]
            )
            for m in history
        ), history
        # Iteration events still flowed alongside the hooks.
        assert [e.name for e in events].count("IterationStart") == 2
        assert [e.name for e in events].count("IterationEnd") == 2

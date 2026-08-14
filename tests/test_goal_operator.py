from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATOR_PATH = REPO_ROOT / "agent_operator.py"
SPEC = importlib.util.spec_from_file_location("agent_operator_under_test", OPERATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {OPERATOR_PATH}")
operator = importlib.util.module_from_spec(SPEC)
sys.modules["agent_operator_under_test"] = operator
SPEC.loader.exec_module(operator)


# --- fakes -------------------------------------------------------------------


def make_set_goal():
    state = {"goals": []}

    def set_goal(goal: str) -> None:
        state["goals"].append(goal)

    return set_goal, state


def make_worker():
    """Fake s15.agent_loop: appends one assistant message per call, records
    call count + history object identity to prove history is reused in place."""
    calls = {"count": 0, "ids": [], "snapshots": []}

    def worker(history: list, context: dict, active_request: str) -> None:
        calls["count"] += 1
        calls["ids"].append(id(history))
        calls["snapshots"].append(list(history))
        history.append({"role": "assistant", "content": f"work-turn-{calls['count']}"})
        context["seen"] = calls["count"]

    return worker, calls


def make_evaluate(actions):
    """Fake evaluator returning scripted StopDecision-like objects."""
    queue = list(actions)
    calls = {"count": 0}

    def evaluate(history: list):
        calls["count"] += 1
        action = queue.pop(0)
        return SimpleNamespace(action=action, reason=f"reason-for-{action}")

    return evaluate, calls


# --- CASE A: achieved after first stop --------------------------------------


def test_case_a_achieved_after_first_turn():
    worker, wcalls = make_worker()
    evaluate, ecalls = make_evaluate(["achieved"])
    set_goal, gstate = make_set_goal()

    result = operator.run_goal_loop(worker, set_goal, evaluate, "do it", "goal met")

    assert result.status == "success"
    assert result.worker_cycles == 1
    assert result.evaluations == 1
    assert result.blocks == 0
    assert result.iterations == 1
    assert wcalls["count"] == 1
    assert ecalls["count"] == 1
    assert gstate["goals"] == ["goal met"]


# --- CASE B: block once then achieved ---------------------------------------


def test_case_b_block_then_achieved_reuses_same_history():
    worker, wcalls = make_worker()
    evaluate, ecalls = make_evaluate(["block", "achieved"])
    set_goal, gstate = make_set_goal()

    result = operator.run_goal_loop(worker, set_goal, evaluate, "task", "goal")

    assert result.status == "success"
    assert result.worker_cycles == 2
    assert result.evaluations == 2
    assert result.blocks == 1
    assert result.iterations == 2
    assert gstate["goals"] == ["goal"]

    # SAME history object reused across both worker calls.
    assert len(set(wcalls["ids"])) == 1

    # The 2nd worker call saw: [task, work-turn-1, continuation].
    snapshot = wcalls["snapshots"][1]
    assert snapshot[-2] == {"role": "assistant", "content": "work-turn-1"}
    assert snapshot[-1]["role"] == "user"
    assert "[Goal still active]" in snapshot[-1]["content"]
    assert "reason-for-block" in snapshot[-1]["content"]


# --- CASE C: impossible stops immediately (nonzero) -------------------------


def test_case_c_impossible_stops_immediately():
    worker, wcalls = make_worker()
    evaluate, ecalls = make_evaluate(["failed"])

    result = operator.run_goal_loop(worker, make_set_goal()[0], evaluate, "task", "goal")

    assert result.status == "failure"
    assert result.worker_cycles == 1
    assert result.evaluations == 1
    assert result.blocks == 0
    assert wcalls["count"] == 1  # stopped immediately, no 2nd worker call


# --- CASE D: max iterations bounded (no infinite loop) ----------------------


def test_case_d_max_iterations_bounded():
    worker, wcalls = make_worker()
    evaluate, ecalls = make_evaluate(["block", "block", "block"])

    result = operator.run_goal_loop(
        worker, make_set_goal()[0], evaluate, "task", "goal", max_iterations=3
    )

    assert result.status == "failure"
    assert result.iterations == 3
    assert result.worker_cycles == 3
    assert result.evaluations == 3
    assert result.blocks == 3
    assert "after 3 iteration" in result.reason
    assert wcalls["count"] == 3


# --- extra: evaluator error -> error status (nonzero) -----------------------


def test_evaluator_error_stops_with_error_status():
    worker, wcalls = make_worker()
    evaluate, ecalls = make_evaluate(["error"])

    result = operator.run_goal_loop(worker, make_set_goal()[0], evaluate, "task", "goal")

    assert result.status == "error"
    assert result.worker_cycles == 1
    assert result.evaluations == 1
    assert "reason-for-error" in result.reason


def test_limit_action_maps_to_failure():
    worker, wcalls = make_worker()
    evaluate, ecalls = make_evaluate(["limit"])

    result = operator.run_goal_loop(worker, make_set_goal()[0], evaluate, "task", "goal")

    assert result.status == "failure"
    assert result.worker_cycles == 1


def test_zero_iterations_rejected():
    with pytest.raises(ValueError):
        operator.run_goal_loop(make_worker()[0], make_set_goal()[0], make_evaluate(["achieved"])[0],
                               "task", "goal", max_iterations=0)

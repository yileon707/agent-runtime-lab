"""Focused tests for the structured evaluator adapter (CORE V0.1.1).

These exercise parse_report_goal (pure response validation) and the
StructuredGoalEvaluator._evaluate_sync wiring (forced tool_choice + thinking
disabled) with fakes — no network, no s15/s17 import needed.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
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


def tool_use_block(name: str = "report_goal_status", input_data=None) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input_data)


def text_block(text: str = "some plain text") -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def response(*blocks, stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


VALID_INPUT = {"ok": True, "reason": "goal satisfied", "impossible": False}


# --- valid ----------------------------------------------------------------


def test_valid_forced_tool_returns_structured_dict():
    parsed = operator.parse_report_goal(response(tool_use_block(input_data=VALID_INPUT)))
    assert parsed == {"ok": True, "reason": "goal satisfied", "impossible": False}


# --- malformed (no tool_use block) ----------------------------------------


def test_malformed_no_tool_use_raises():
    with pytest.raises(operator.StructuredEvaluatorError):
        operator.parse_report_goal(response(text_block("not json"), stop_reason="end_turn"))


def test_malformed_empty_content_raises():
    with pytest.raises(operator.StructuredEvaluatorError):
        operator.parse_report_goal(response())


# --- wrong tool -----------------------------------------------------------


def test_wrong_tool_name_raises():
    with pytest.raises(operator.StructuredEvaluatorError):
        operator.parse_report_goal(
            response(tool_use_block(name="do_something_else", input_data=VALID_INPUT))
        )


# --- invalid fields -------------------------------------------------------


@pytest.mark.parametrize(
    "bad_input",
    [
        {"ok": "yes", "reason": "x", "impossible": False},  # ok not bool
        {"ok": True, "reason": "", "impossible": False},  # empty reason
        {"ok": True, "reason": 123, "impossible": False},  # reason not str
        {"reason": "x", "impossible": False},  # missing ok
        {"ok": True, "reason": "x"},  # missing impossible -> default False (valid)
        {"ok": True, "reason": "x", "impossible": "no"},  # impossible not bool
        {"ok": True, "reason": "x", "impossible": True},  # ok and impossible
    ],
)
def test_invalid_fields_raise(bad_input):
    # "missing impossible" defaults to False and is valid — skip that one here.
    if "impossible" not in bad_input:
        assert operator.parse_report_goal(
            response(tool_use_block(input_data=bad_input))
        )["impossible"] is False
        return
    with pytest.raises(operator.StructuredEvaluatorError):
        operator.parse_report_goal(response(tool_use_block(input_data=bad_input)))


def test_input_not_a_dict_raises():
    with pytest.raises(operator.StructuredEvaluatorError):
        operator.parse_report_goal(response(tool_use_block(input_data="not an object")))


def test_ok_and_impossible_both_true_raises():
    with pytest.raises(operator.StructuredEvaluatorError):
        operator.parse_report_goal(
            response(
                tool_use_block(
                    input_data={"ok": True, "reason": "x", "impossible": True}
                )
            )
        )


# --- wiring: forced tool_choice + thinking disabled -> GoalEvaluation ------


@dataclass(frozen=True)
class FakeGoalEvaluation:
    ok: bool
    reason: str
    impossible: bool = False


def test_evaluate_sync_forces_tool_and_returns_goal_evaluation():
    calls: dict = {}

    class FakeClient:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return response(tool_use_block(input_data=VALID_INPUT))

    fake_s17 = SimpleNamespace(
        transcript_text=lambda msgs: "TRANSCRIPT",
        GoalEvaluation=FakeGoalEvaluation,
    )
    evaluator = operator.StructuredGoalEvaluator(
        client=FakeClient(), model="deepseek-v4-pro", s17=fake_s17
    )

    result = evaluator._evaluate_sync("condition", [{"role": "user", "content": "hi"}])

    assert isinstance(result, FakeGoalEvaluation)
    assert result.ok is True
    assert result.reason == "goal satisfied"
    assert result.impossible is False

    kw = calls["kwargs"]
    assert kw["tool_choice"] == {"type": "tool", "name": "report_goal_status"}
    assert kw["thinking"] == {"type": "disabled"}
    assert kw["tools"] == [operator.REPORT_GOAL_TOOL]
    assert "completion_condition" in kw["messages"][0]["content"]

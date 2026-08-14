#!/usr/bin/env python3
"""Autonomous Goal Runner (CORE V0.1).

Thin composition layer over the frozen Stable Agent V0 (s15) worker and the
s17 goal evaluator. The operator owns ONLY the control loop:

    initialize runtime
    history = [user task]
    goal_controller.set_goal(goal)
    repeat up to MAX_GOAL_ITERATIONS:
        s15.agent_loop(history, context, active_request=task)
        evaluate goal against the SAME history
        achieved   -> print, exit 0
        impossible -> print reason, exit nonzero
        error      -> print reason, exit nonzero
        unfinished -> append ONE continuation, continue
    iteration limit -> exit nonzero

Tools, compaction, permissions, memory, and the provider remain s15's
responsibility. This module makes zero changes to s15_integrated_harness/code.py
and s17_goal_loop/code.py: it imports both via importlib (neither package has an
__init__.py) and reuses s15's module-level client/model for the evaluator.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DEFAULT_MAX_GOAL_ITERATIONS = 8

REPO_ROOT = Path(__file__).resolve().parent

CONTINUATION_TEMPLATE = (
    "[Goal still active]\n"
    "Evaluator: {reason}\n"
    "Continue the original task and produce the missing evidence."
)


@dataclass
class OperatorResult:
    """Outcome of one goal-runner run, for CLI mapping and offline tests."""

    status: str  # "success" | "failure" | "error"
    iterations: int = 0
    worker_cycles: int = 0
    evaluations: int = 0
    blocks: int = 0
    reason: str = ""
    last_action: str = ""


def _load_module(name: str, path: Path):
    """Load a code.py as a module by path (no __init__.py in chapter dirs)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_worker():
    """Load s15 (Stable Agent V0). Auto-initializes its client/MODEL globals."""
    return _load_module(
        "_operator_s15", REPO_ROOT / "s15_integrated_harness" / "code.py"
    )


def load_goal():
    """Load s17 (goal evaluator). Import-safe: no client created at import."""
    return _load_module(
        "_operator_s17", REPO_ROOT / "s17_goal_loop" / "code.py"
    )


def run_goal_loop(
    worker: Callable[[list, dict, str], None],
    set_goal: Callable[[str], None],
    evaluate: Callable[[list], Any],
    task: str,
    goal: str,
    context: dict | None = None,
    max_iterations: int = DEFAULT_MAX_GOAL_ITERATIONS,
) -> OperatorResult:
    """Minimal control loop, decoupled from s15/s17 for offline testing.

    worker(history, context, active_request): mutates history in place (like
        s15.agent_loop) and returns None.
    set_goal(goal): arms the goal controller.
    evaluate(history) -> object with `.action` and `.reason` (StopDecision).
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    context = {} if context is None else context

    set_goal(goal)
    history = [{"role": "user", "content": task}]

    result = OperatorResult(status="failure")
    for iteration in range(1, max_iterations + 1):
        worker(history, context, task)
        result.worker_cycles += 1

        decision = evaluate(history)
        result.evaluations += 1
        result.last_action = decision.action

        if decision.action == "achieved":
            result.status = "success"
            result.iterations = iteration
            result.reason = decision.reason
            return result

        if decision.action in ("failed", "limit"):
            result.status = "failure"
            result.iterations = iteration
            result.reason = decision.reason or (
                f"goal not achieved (action={decision.action})"
            )
            return result

        if decision.action == "error":
            result.status = "error"
            result.iterations = iteration
            result.reason = decision.reason or "goal evaluator raised an error"
            return result

        # "block" (unfinished) — and defensively "allow"/"defer" — continue.
        if decision.action == "block":
            result.blocks += 1
        history.append(
            {
                "role": "user",
                "content": CONTINUATION_TEMPLATE.format(
                    reason=decision.reason or "(no evaluator reason)"
                ),
            }
        )

    result.iterations = max_iterations
    result.reason = f"goal not achieved after {max_iterations} iteration(s)"
    return result


def build_worker_and_evaluator(s15, s17, goal: str):
    """Wire the real s15 worker to the s17 evaluator (shared client/model)."""
    evaluator = s17.PromptGoalEvaluator(client=s15.client, model=s15.MODEL)
    controller = s17.GoalController(evaluator=evaluator)

    s15.CLI_ACTIVE = True
    s15.start_runtime_services()

    def worker(history: list, context: dict, active_request: str) -> None:
        turn_start = len(history)
        with s15.agent_lock:
            s15.agent_loop(history, context, active_request)
        context.update(s15.update_context(context, history))
        s15.print_turn_assistants(history, turn_start)

    def evaluate(history: list):
        return asyncio.run(controller.evaluate_after_turn(history))

    return worker, controller.set_goal, evaluate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Goal Runner")
    parser.add_argument("--task", required=True, help="the user task")
    parser.add_argument("--goal", required=True, help="the completion condition")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_GOAL_ITERATIONS,
        help="max worker/evaluator cycles (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    s15 = load_worker()
    s17 = load_goal()
    worker, set_goal, evaluate = build_worker_and_evaluator(s15, s17, args.goal)

    context = s15.update_context({}, [])
    result = run_goal_loop(
        worker,
        set_goal,
        evaluate,
        args.task,
        args.goal,
        context=context,
        max_iterations=args.max_iterations,
    )

    if result.status == "success":
        print(
            f"[goal runner] SUCCESS: goal achieved after "
            f"{result.worker_cycles} worker cycle(s), "
            f"{result.evaluations} evaluation(s), {result.blocks} block(s)"
        )
        print(f"[goal runner] {result.reason}")
        return 0

    print(
        f"[goal runner] {result.status.upper()}: {result.reason} "
        f"({result.worker_cycles} worker cycle(s), "
        f"{result.evaluations} evaluation(s))",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

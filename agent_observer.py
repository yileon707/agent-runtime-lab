"""Minimal observer event layer for agent runtime tracing (TRACE V0.1a).

A tiny, dependency-free event surface that lets an external observer watch the
core runtime lifecycle (Run -> Cycle -> Iteration) without being able to affect
it. Observer callbacks are fire-and-forget: their return values are discarded
and any exception they raise is swallowed, so a misbehaving observer can never
change runtime behavior.

This module is deliberately NOT a span tree, a queue, a journal, a worker
thread, or an OpenTelemetry-style framework. It exposes exactly three
primitives:

  * RuntimeEvent              -- the event value (name, timestamp, identity)
  * register_observer / clear_observers -- subscribe / unsubscribe
  * emit_observer             -- fan an event out to current observers

The identity helpers (begin_run / begin_cycle / next_iteration_id) are used by
the operator (Run/Cycle) and s15 (Iteration) so all events carry consistent
run_id / cycle_id / iteration_id without the runtime itself knowing about
tracing. Scope is one Goal Runner run at a time; the state is module-global and
single-run, matching the operator's single-threaded control loop.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class RuntimeEvent:
    name: str
    timestamp: float
    run_id: Optional[str] = None
    cycle_id: Optional[str] = None
    iteration_id: Optional[str] = None
    correlation_id: Optional[str] = None
    attributes: dict = field(default_factory=dict)


_observers: list[Callable[[RuntimeEvent], None]] = []

_current_run_id: Optional[str] = None
_current_cycle_id: Optional[str] = None
_cycle_counter = 0
_iteration_counter = 0


def register_observer(callback: Callable[[RuntimeEvent], None]) -> None:
    """Subscribe a callback. Duplicate callbacks are ignored."""
    if callback is not None and callback not in _observers:
        _observers.append(callback)


def clear_observers() -> None:
    """Unsubscribe every observer (used for test isolation)."""
    _observers.clear()


def emit_observer(event: RuntimeEvent) -> None:
    """Fan an event out to current observers.

    Iterates over a snapshot so an observer that registers/unregisters during
    emit cannot mutate the iteration mid-loop. Return values are discarded and
    exceptions are isolated, so observers cannot alter runtime behavior.
    """
    for callback in tuple(_observers):
        try:
            callback(event)
        except Exception:
            pass


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def current_run_id() -> Optional[str]:
    return _current_run_id


def current_cycle_id() -> Optional[str]:
    return _current_cycle_id


def begin_run(run_id: Optional[str] = None) -> str:
    """Start a new run, resetting cycle/iteration identity. Returns the run_id."""
    global _current_run_id, _current_cycle_id, _cycle_counter, _iteration_counter
    _current_run_id = run_id or new_run_id()
    _current_cycle_id = None
    _cycle_counter = 0
    _iteration_counter = 0
    return _current_run_id


def end_run() -> Optional[str]:
    """Close the current run. Returns the closed run_id (or None)."""
    global _current_run_id, _current_cycle_id, _iteration_counter
    previous = _current_run_id
    _current_run_id = None
    _current_cycle_id = None
    _iteration_counter = 0
    return previous


def begin_cycle() -> Optional[str]:
    """Start a new cycle within the current run. Returns the cycle_id."""
    global _current_cycle_id, _cycle_counter, _iteration_counter
    _cycle_counter += 1
    _current_cycle_id = f"cycle-{_cycle_counter}"
    _iteration_counter = 0
    return _current_cycle_id


def end_cycle() -> Optional[str]:
    """Close the current cycle. Returns the closed cycle_id (or None)."""
    global _current_cycle_id, _iteration_counter
    previous = _current_cycle_id
    _current_cycle_id = None
    _iteration_counter = 0
    return previous


def next_iteration_id() -> Optional[str]:
    """Allocate the next iteration_id within the active cycle (or None).

    Iteration ids are ``f"{cycle_id}.{n}"`` so they are unique and orderable
    within a cycle; the counter resets on each begin_cycle, so ids never bleed
    across cycles.
    """
    global _iteration_counter
    if _current_cycle_id is None:
        return None
    _iteration_counter += 1
    return f"{_current_cycle_id}.{_iteration_counter}"


def make_event(
    name: str,
    *,
    run_id: Optional[str] = None,
    cycle_id: Optional[str] = None,
    iteration_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    attributes: Optional[dict] = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        name=name,
        timestamp=time.time(),
        run_id=run_id,
        cycle_id=cycle_id,
        iteration_id=iteration_id,
        correlation_id=correlation_id,
        attributes=dict(attributes or {}),
    )

"""Tests of the time limit helper."""
from __future__ import annotations

from sympy_extras._timeout import attempt


def test_attempt_contains_a_recursion_error() -> None:
    # sympy-extras#49: attempt() is the package's way of containing a
    # SymPy call that gives up, and is used around exactly the calls that
    # can exhaust the stack, but it let RecursionError through -- one
    # equation deep in dsolve ended four benchmark sweeps out of four.
    def bottomless() -> int:
        return bottomless() + 1

    assert attempt(bottomless, 5) is None


def test_attempt_returns_the_value_otherwise() -> None:
    assert attempt(lambda: 42, 5) == 42

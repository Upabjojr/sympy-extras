"""The last operations the core sent to SymPy through the bridge, now
native: each test records the calls of the bridge (SymPy's, registered by
``sympy_extras.interop``) while the operation runs, checks that there are
none, and compares the result with SymPy's."""
from __future__ import annotations

from typing import Callable, Iterator, Optional
from contextlib import contextmanager

import mpmath
import sympy

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.bridge import bridge, register_bridge
from sympy_extras.interop import SymPyBridge, expr_from_sympy, to_sympy

s = sympy


class _RecordingBridge(SymPyBridge):
    """SymPy's bridge, recording the operations which reach it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def rewrite_node(self, e: Basic, args: tuple[Basic, ...], target: type, piecewise: bool) -> Optional[Basic]:
        self.calls.append('rewrite_node %s' % type(e).__name__)
        return SymPyBridge.rewrite_node(self, e, args, target, piecewise)

    def numeric(self, e: Expr, dps: int) -> Optional[mpmath.mpc]:
        self.calls.append('numeric %s' % type(e).__name__)
        return SymPyBridge.numeric(self, e, dps)


@contextmanager
def _recording() -> Iterator[_RecordingBridge]:
    previous = bridge()
    recorder = _RecordingBridge()
    register_bridge(recorder)
    try:
        yield recorder
    finally:
        register_bridge(previous)


def _same(theirs: sympy.Basic, ours: Callable[[], Basic]) -> None:
    with _recording() as recorder:
        result = ours()
    assert recorder.calls == [], recorder.calls
    assert str(to_sympy(result, evaluate=False)) == str(theirs), (result, theirs)


def test_rewrite_of_nodes_without_rules_needs_no_bridge() -> None:
    # hyper, meijerg, the pairs of Piecewise and IndexedRoot were sent to
    # SymPy's rewrite one by one (it has no rule for them and answered None)
    from sympy_extras.functions.elementary.trigonometric import Cos
    from sympy_extras.functions.special.gamma_functions import Gamma
    from sympy_extras.polys.cad.indexedroot_sympy import IndexedRoot
    a, u, n, x = s.symbols('a u n x')
    for e, theirs, ours in [
            (s.hyper((a / 2,), (s.Rational(1, 2),), -s.factorial(a)), s.cos, Cos),
            (s.hyper((s.Rational(1, 6),), (s.Rational(1, 2),), s.factorial(a)), s.gamma, Gamma),
            (s.meijerg(((-1, n + 1), ()), ((), (-1, 0)), u / a * s.factorial(n)), s.gamma, Gamma),
            (s.Piecewise((s.factorial(a), a > 0), (0, True)), s.gamma, Gamma),
            (IndexedRoot(x**3 - x - 1, x, 0) + s.factorial(a), s.gamma, Gamma)]:
        native = expr_from_sympy(e)
        _same(e.rewrite(theirs), lambda: native.rewrite(ours))


def test_divergent_sum_value_needs_no_bridge() -> None:
    # numeric_value(Sum(1, (i, 1, oo))) went to SymPy's evalf for its ValueError
    from sympy.testing.pytest import raises
    from sympy_extras.core.evalf import numeric_value
    i = s.Symbol('i', integer=True)
    native = expr_from_sympy(s.Sum(1, (i, 1, s.oo)))
    assert isinstance(native, Expr)
    with _recording() as recorder:
        raises(ValueError, lambda: numeric_value(native, 80))
    assert recorder.calls == [], recorder.calls
    raises(ValueError, lambda: s.Sum(1, (i, 1, s.oo)).evalf())

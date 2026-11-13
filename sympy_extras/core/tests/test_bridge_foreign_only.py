"""The generic fallbacks of the core decide natively for native nodes:
``equals``, ``is_constant``, the first solution of ``Order``, ``integrate``,
``invert`` with several variables, ``simplify`` of a node which is not an
expression, the numerical value of a number node; the bridge serves only
the wrapped foreign (SymPy) objects. Each test records the calls of the
bridge (SymPy's, registered by ``sympy_extras.interop``) and compares the
result with SymPy's."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Union

import mpmath
import sympy

from sympy.testing.pytest import raises

from sympy_extras._numeric import reliable_complex
from sympy_extras.core import Integer, Rational, Symbol, oo, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.core.bridge import bridge, register_bridge
from sympy_extras.core.expr import (Expr, _constant_by_simplified_derivatives, _remaining_zero_test,
                                    _surds_zero_test)
from sympy_extras.core.order import _solve_for
from sympy_extras.functions import acos, asin, cos, exp, log, sin
from sympy_extras.interop import SymPyBridge, SymPyFunction, expr_from_sympy, to_sympy

s = sympy


class _RecordingBridge(SymPyBridge):
    """SymPy's bridge, recording the generic operations which reach it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def equals_zero(self, e: Expr) -> Optional[bool]:
        self.calls.append('equals_zero')
        return SymPyBridge.equals_zero(self, e)

    def is_constant(self, e: Expr, wrt: Sequence[Symbol]) -> Optional[bool]:
        self.calls.append('is_constant')
        return SymPyBridge.is_constant(self, e, wrt)

    def solve_first(self, equation: Expr, x: Symbol) -> Optional[Expr]:
        self.calls.append('solve_first')
        return SymPyBridge.solve_first(self, equation, x)

    def integrate(self, e: Expr, specs: Sequence[Union[Symbol, tuple[Symbol, Expr, Expr]]]) -> Expr:
        self.calls.append('integrate')
        return SymPyBridge.integrate(self, e, specs)

    def invert(self, e: Expr, g: Expr, gens: Sequence[Symbol]) -> Expr:
        self.calls.append('invert')
        return SymPyBridge.invert(self, e, g, gens)

    def simplify(self, e: Basic, ratio: float = 1.7, rational: Optional[bool] = False, inverse: bool = False,
                 doit: bool = True) -> Basic:
        self.calls.append('simplify')
        return SymPyBridge.simplify(self, e, ratio, rational, inverse, doit)

    def numeric(self, e: Expr, dps: int) -> Optional[mpmath.mpc]:
        self.calls.append('numeric')
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


x, y, k, d = Symbol('x'), Symbol('y'), Symbol('k'), Symbol('d')


def test_equals_and_is_constant_need_no_bridge() -> None:
    # the differences the simplified form did not decide went to SymPy's
    # equals; the expressions the values at points did not decide to
    # SymPy's is_constant
    with _recording() as recorder:
        assert (asin(x) + acos(x)).equals(pi / 2) is None  # SymPy: None too
        assert (sin(x)**2 + cos(x)**2 + y).is_constant(x) is True
        assert (exp(x) * exp(-x)).is_constant() is True
        assert (log(x) + y).is_constant(x) is False
    assert recorder.calls == [], recorder.calls
    sx = s.Symbol('x')
    assert (s.asin(sx) + s.acos(sx) - s.pi / 2).equals(0) is None


def test_remaining_steps_of_equals() -> None:
    # SymPy's steps after the simplification: the minimal polynomial and the
    # surds of a number, the derivative and the value at zero otherwise
    assert _remaining_zero_test(sqrt(2) + sqrt(3) - sqrt(5 + 2 * sqrt(6))) is True
    assert _remaining_zero_test(sqrt(2) - 1) is False
    assert _surds_zero_test(sqrt(2) * sqrt(3) - sqrt(6)) in (True, None)
    assert _surds_zero_test(sqrt(5) - 2) is False   # the solution 2 is an integer
    assert _remaining_zero_test(x * exp(x) + 1) is False
    assert _remaining_zero_test(x * exp(x)) is None
    assert _constant_by_simplified_derivatives(sin(x)**2 + cos(x)**2, [x]) is True
    assert _constant_by_simplified_derivatives(x + 1 / x, [x]) is False


def test_first_solution_of_order_needs_no_bridge() -> None:
    # several solutions, or a ConditionSet of the native solveset, went to
    # SymPy's solveset; the first one is SymPy's
    with _recording() as recorder:
        first_log = _solve_for(log(d) - k, d)
        first_cubic = _solve_for(d**3 - k * d, d)
        nothing = _solve_for(k - sin(d), d)  # a union of images: SymPy's Order takes none
    assert recorder.calls == [], recorder.calls
    sk, sd = s.symbols('k d')
    assert first_log is not None and to_sympy(first_log) == s.exp(sk)
    assert first_cubic is not None and to_sympy(first_cubic) == list(s.solveset(sd**3 - sk * sd, sd))[0]
    assert nothing is None


def test_integrate_needs_no_bridge() -> None:
    # integrate went to SymPy's (the native one only took sums of powers)
    with _recording() as recorder:
        indefinite = (1 / (x + 1)).integrate(x)
        definite = (exp(-x**2)).integrate((x, 0, oo))
        double = (x * y).integrate(x, (y, 0, 1))
    assert recorder.calls == [], recorder.calls
    assert to_sympy(indefinite) == s.log(s.Symbol('x') + 1)
    assert to_sympy(definite) == s.sqrt(s.pi) / 2
    assert to_sympy(double) == s.Symbol('x')**2 / 4


def test_invert_with_parameters_needs_no_bridge() -> None:
    # invert with a generator and other symbols went to SymPy's invert
    sx, sy = s.symbols('x y')
    cases = [(x + y, x**2 + 1, sx + sy, sx**2 + 1), (x**3 + y, x**2 + y * x + 1, sx**3 + sy, sx**2 + sy * sx + 1),
             (x, x**2 + y, sx, sx**2 + sy)]
    for f, g, sf, sg in cases:
        with _recording() as recorder:
            ours = f.invert(g, x)
        assert recorder.calls == [], recorder.calls
        theirs = expr_from_sympy(s.invert(sf, sg, sx))
        value = reliable_complex(ours - theirs, 20, {y: Rational(3, 7), x: Rational(5, 11)})
        assert value is not None and abs(value) < 1e-15, (f, g, ours, theirs)
    # several generators: SymPy's DMP.invert raises
    raises(ValueError, lambda: x.invert(x**2 + 1, x, y))
    raises(ValueError, lambda: s.invert(sx, sx**2 + 1, sx, sy))


def test_simplify_of_other_nodes_needs_no_bridge() -> None:
    # a node which is neither an expression nor a formula went to SymPy's
    # simplify: rebuilt with its arguments simplified
    from sympy_extras.sets.sets import FiniteSet, Interval
    with _recording() as recorder:
        a = FiniteSet(sin(x)**2 + cos(x)**2, x).simplify()
        b = Interval(Integer(0), exp(x) * exp(-x)).simplify()
    assert recorder.calls == [], recorder.calls
    assert a == FiniteSet(Integer(1), x)
    assert b == Interval(Integer(0), Integer(1))


def test_foreign_objects_still_go_to_the_bridge() -> None:
    # by design: what only SymPy knows is asked to SymPy
    f = SymPyFunction(s.Function('f'))
    with _recording() as recorder:
        (f(x) + x).is_constant(x)
        (f(x) * x).integrate(x)
    assert 'integrate' in recorder.calls and 'is_constant' in recorder.calls


def test_sum_and_integral_values_need_no_bridge() -> None:
    from sympy_extras.concrete.expr_with_limits import Sum
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.integrals.integral import Integral
    i = Symbol('i', integer=True)
    with _recording() as recorder:
        total = numeric_value(Sum(1 / i**Rational(3, 2), (i, 1, oo)) + 1, 80)
        area = numeric_value(Integral(exp(-x**2), x, Integer(0), oo) * 2, 80)
    assert recorder.calls == [], recorder.calls
    with mpmath.workprec(80):
        assert total is not None and abs(total - mpmath.zeta(1.5) - 1) < mpmath.mpf(10)**-20
        assert area is not None and abs(area - mpmath.sqrt(mpmath.pi)) < mpmath.mpf(10)**-20

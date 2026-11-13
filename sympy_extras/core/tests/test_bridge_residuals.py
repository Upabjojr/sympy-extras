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
from sympy_extras.core.symbol import Symbol
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

    def rewrite_named_node(self, e: Basic, args: tuple[Basic, ...], name: str, limitvar: Optional[Symbol]
                           ) -> Optional[Basic]:
        self.calls.append('rewrite_named_node %s' % type(e).__name__)
        return SymPyBridge.rewrite_named_node(self, e, args, name, limitvar)

    def nseries(self, node: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        self.calls.append('nseries %s' % type(node).__name__)
        return SymPyBridge.nseries(self, node, x, n, logx, cdir)

    def equals_zero(self, e: Expr) -> Optional[bool]:
        self.calls.append('equals_zero %s' % type(e).__name__)
        return SymPyBridge.equals_zero(self, e)

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
    assert str(theirs) in (str(to_sympy(result, evaluate=False)), str(to_sympy(result))), (result, theirs)


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


def test_tractable_rewrites_need_no_bridge() -> None:
    # erf, erfc, erfi, Ei, li, Li, uppergamma, besseli, besselk and
    # fibonacci went to SymPy's _eval_rewrite_as_tractable, whose helpers
    # _erfs, _eis, _besseli, _besselk came back wrapped
    from sympy_extras.core.rewriting import rewrite_tractable
    x, y = s.Symbol('x', positive=True), s.Symbol('y')
    for e in [s.erf(x), -s.erf(x - 1), s.erfc(x), s.erfi(x), s.Ei(x), s.li(x), s.Li(x), s.uppergamma(y, x),
              s.besseli(y, x), s.besselk(2, x), s.besseli(1, y), s.fibonacci(x), s.erf(s.sinh(x)), s.Ei(s.erf(x))]:
        native, variable = expr_from_sympy(e), expr_from_sympy(x)
        assert isinstance(variable, Symbol)
        theirs = e.rewrite('tractable', deep=True, limitvar=x)
        _same(theirs, lambda: rewrite_tractable(native, variable))
        back = expr_from_sympy(theirs)
        _same(theirs.rewrite('intractable', deep=True), lambda: rewrite_tractable(back, None, True))


def test_asymptotic_series_need_no_bridge() -> None:
    # the _eval_aseries of erf, erfc, erfi, Ei, Si, Ci, besseli, besselk and
    # of the helpers of the Gruntz algorithm went to SymPy's nseries
    from sympy.functions.special.bessel import _besseli, _besselk
    from sympy.functions.special.error_functions import _eis, _erfs
    w = s.Symbol('w', positive=True)
    variable = expr_from_sympy(w)
    assert isinstance(variable, Symbol)
    for e in [s.erf(1 / w), s.erfc(1 / w), s.erfi(1 / w), s.Ei(1 / w), s.Ei(-1 / w), s.Si(1 / w), s.Ci(1 / w),
              s.besseli(2, 1 / w), s.besselk(s.Rational(1, 3), 1 / w), _erfs(1 / w), _eis(1 / w), _eis(-1 / w),
              _besseli(1, 1 / w), _besselk(0, 1 / w), s.Si((2 * w + 1) / w), s.erf(1 / w + 1 / w**2), _erfs(s.I / w),
              _eis(w), _erfs(w)]:
        native = expr_from_sympy(e)
        assert isinstance(native, Expr)
        _same(e.nseries(w, n=4), lambda: native.nseries(variable, n=4))


def test_limits_with_the_special_functions_match_sympy() -> None:
    from sympy_extras.core.limits import limit
    x = s.Symbol('x', positive=True)
    for e in [(s.erf(x) - 1) * s.exp(x**2) * x, s.erfc(x) * s.exp(x**2) * x, s.erfi(x) * s.exp(-x**2) * x,
              s.Ei(x) * s.exp(-x) * x, s.li(x) * s.log(x) / x, s.Li(x) * s.log(x) / x,
              s.besselk(1, x) * s.exp(x) * s.sqrt(x), s.Si(x), s.Ei(-x) * x * s.exp(x),
              s.fibonacci(x + 1) / s.fibonacci(x), s.erf(x)**x]:
        native, variable, point = expr_from_sympy(e), expr_from_sympy(x), expr_from_sympy(s.oo)
        assert isinstance(variable, Symbol)
        _same(s.limit(e, x, s.oo), lambda: limit(native, variable, point))


def test_expint_series_through_a_polar_ei_is_not_sympys() -> None:
    # SymPy 1.14 expands Ei(exp_polar(I*pi)/w) without its branch term
    # I*pi: expint(2, 1/w).series(w) is -I*pi/w + O(w**3), and
    # limit(x*exp(x)*expint(2, x), x, oo) is -oo*I (it is 1); the native
    # asymptotic series of Ei declines a polar argument
    from sympy.testing.pytest import raises
    from sympy_extras.core.coreerrors import PoleError
    from sympy_extras.core.limits import limit
    w = s.Symbol('w', positive=True)
    assert str(s.expint(2, 1 / w).series(w, 0, 3)) == '-I*pi/w + O(w**3)'
    native, variable = expr_from_sympy(s.expint(2, 1 / w)), expr_from_sympy(w)
    assert isinstance(native, Expr) and isinstance(variable, Symbol)
    raises(PoleError, lambda: native.nseries(variable, n=4))
    x = s.Symbol('x', positive=True)
    e, v = expr_from_sympy(x * s.exp(x) * s.expint(2, x)), expr_from_sympy(x)
    assert isinstance(v, Symbol)
    assert str(limit(e, v, expr_from_sympy(s.oo))) != '-oo*I'

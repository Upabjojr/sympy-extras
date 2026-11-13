"""The series of the nodes whose expansion went to SymPy through the
bridge: the functions expanded by SymPy's generic ``Function._eval_nseries``
(``factorial``, ``catalan``, ``exp_polar``, ``elliptic_pi``, ``appellf1``,
``LeviCivita``, ``motzkin``, ``QPochhammer``), the ports of
``Piecewise``, ``Derivative``, ``Subs`` and ``Integral._eval_nseries``,
and the nodes without one (``Max``, ``Sum``, ...), which raise
``NotImplementedError`` as SymPy's ``Expr._eval_nseries`` does. The series
are checked by the values: at a small ``x`` the truncated series is the
function up to the order term."""
from __future__ import annotations

from sympy.testing.pytest import raises

from sympy_extras._numeric import reliable_complex
from sympy_extras.core import Function, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import Derivative, Subs, diff
from sympy_extras.functions import cos, exp, factorial, sin
from sympy_extras.functions.elementary.piecewise import piecewise

x, y = Symbol('x'), Symbol('y')
SMALL = Rational(1, 64)


def _matches(f: Expr, n: int, tolerance: float = 1e-3) -> None:
    """The series of ``f`` to ``O(x**n)`` agrees with ``f`` at a small
    ``x`` up to a multiple of ``x**n``."""
    series = f.nseries(x, n=n)
    assert series.getO() is not None, series
    v0 = reliable_complex(f, 15, {x: SMALL})
    v1 = reliable_complex(series.removeO(), 15, {x: SMALL})
    assert v0 is not None and v1 is not None, (f, series)
    assert abs(v0 - v1) <= tolerance * float(SMALL)**n * max(1.0, abs(v0)), (f, series, v0, v1)


def test_generic_function_series_are_native() -> None:
    from sympy_extras.functions.combinatorial.numbers import catalan
    from sympy_extras.functions.special.elliptic_integrals import elliptic_pi
    for f in (factorial(x), factorial(2 * x + Rational(1, 2)), factorial(sin(x) + 2), catalan(x + 1),
              catalan(x + Rational(1, 2)), elliptic_pi(x + Rational(1, 5), Rational(1, 4))):
        _matches(f, 4, 10.0)


def test_exp_polar_series_expands_the_sum_first() -> None:
    # SymPy's exp_polar is an ExpBase: exp_polar(2*x + 1) expands to
    # exp_polar(1)*exp_polar(2*x) before its derivatives are taken
    from sympy_extras.functions.elementary.exponential import exp_polar
    e = exp_polar(2 * x + 1).expand()
    assert e == exp_polar(Integer(1)) * exp_polar(2 * x)


def test_piecewise_derivative_subs_series() -> None:
    pw = piecewise((sin(x), y > 0), (cos(x), True))
    s = pw.nseries(x, n=4)
    assert s.subs(y, 1) == sin(x).nseries(x, n=4) and s.subs(y, -1) == cos(x).nseries(x, n=4)
    d = Derivative(sin(x) * y, x).nseries(x, n=4)
    # the derivative of each term, the order divided by x
    assert d.doit() == (y * cos(x)).nseries(x, n=3)
    sb = Subs(sin(x * y), y, Integer(2)).nseries(x, n=4)
    assert sb.doit() == sin(2 * x).nseries(x, n=4)
    sb2 = Subs(cos(y), y, x).nseries(x, n=4)
    assert sb2.doit() == cos(x).nseries(x, n=4)
    f = Function('f')
    assert diff(Derivative(f(x) * exp(x), x).nseries(x, n=3).removeO().doit(), x, 3) == 0


def test_integral_series() -> None:
    from sympy_extras.integrals.integral import Integral
    from sympy_extras.integrals.risch.risch import NonElementaryIntegral
    # a bound in x: the integrand expanded in the variable of integration
    s = Integral(exp(y), y, Integer(0), x).nseries(x, n=4)
    assert (s.removeO() - (x + x**2 / 2 + x**3 / 6 + x**4 / 24)).expand() == 0 and s.getO() is not None
    # the integral of sin(x*y) over (0, 1) is (1 - cos(x))/x
    u = Integral(sin(x * y), y, Integer(0), Integer(1)).nseries(x, n=5)
    assert (u.removeO() - (x / 2 - x**3 / 24)).expand() == 0
    t = NonElementaryIntegral(exp(-x**2), x).nseries(x, n=6)
    assert t.removeO() == x - x**3 / 3 + x**5 / 10


def test_nodes_without_a_series_raise() -> None:
    from sympy_extras.concrete.expr_with_limits import Sum
    from sympy_extras.functions.elementary.miscellaneous import max_, min_
    k = Symbol('k', integer=True)
    # SymPy's Expr._eval_nseries raises NotImplementedError (they went to
    # SymPy through the bridge, which raised it)
    raises(NotImplementedError, lambda: max_(x, Integer(1)).nseries(x, n=3))
    raises(NotImplementedError, lambda: min_(x, y).nseries(x, n=3))
    raises(NotImplementedError, lambda: Sum(x**k, (k, Integer(0), Integer(3))).nseries(x, n=3))

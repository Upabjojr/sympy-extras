"""``sympy_extras.solvers.solvers`` (``solve_equation``, ``invert``,
``solve_system``) and ``sympy_extras.solvers.bivariate``
(``solve_lambert``) against their Rust port; the solutions compared as
strings, in order (``None`` as ``NONE``)."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('solve', [Symbol(op), *args])


def _shown(r: Any) -> Any:
    return 'None' if r is None else [str(v) for v in r]


def _outcome(f: Any) -> Any:
    try:
        return ('ok', f())
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _rust_list(r: Any) -> Any:
    return 'None' if str(r) == 'NONE' else [str(v) for v in r.args]


def _equations() -> list[tuple[Any, Any]]:
    from sympy_extras.core import Function, Integer, Rational, Symbol, sqrt
    from sympy_extras.functions import atan, cos, exp, log, sin, tan
    x, y, a, b, C1, C2 = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b'), Symbol('C1'), Symbol('C2')
    p, v, u, m = Symbol('p'), Symbol('v'), Symbol('u'), Symbol('m')
    k = Symbol('k', integer=True, positive=True)
    n = Symbol('n', integer=True)
    return [(x**2 - a, x), (x**3 - 1, x), (x**3 - x**3*sin(a) + 2, x), (x**4 - (4 + 2*sqrt(Integer(2)))*x**2 + 1, x),
            (x - x + 1, x), (1/x, x), (-p/(p*y + y**3) - v, p), ((x**2 - 1)/(x - 1), x), (log(y) - C1*x - C2, y),
            (exp(2*x) - 3*exp(x) + 2, x), (x**Rational(2, 3) - 4, x),
            (-C1 + log(x) - 2*sqrt(Integer(3))*atan(2*sqrt(Integer(3))*x*y/3 + sqrt(Integer(3))/3)/3, y), (sqrt(x) + 1, x),
            (2*sqrt(x) - a, x), (x - sqrt(x**2 - 1), x), (2*u/(3*sqrt(1/(u*a))) - a*y - b - x, u),
            (-2*u/(3*sqrt(1/(u*a))) - a*y - b - x, u), (5*sqrt(x)/3 + 3 + 1/(3*(x - 1)), x), (sin(x) - Rational(3, 2), x),
            (cos(2*x) - Rational(1, 3), x), (x*exp(x) - 1, x), (x - exp(x) + 2, x), ((-1)**k/(4*k**3) - 1/(4*(k + m)**3), m),
            (x**2 - 4, n), (2**x - 8, x), (tan(x) - 1, x), (x**5 - x - 1, x), (x**2 + x + 1, x), (exp(x) + exp(-x) - 4, x),
            (log(x) + log(x - 1), x), (x**4 - 1, x), (a*x**2 + b*x + 1, x), (x*log(x) - 1, x)]


@pytest.mark.parametrize("index", range(len(_equations())))
def test_solve_equation(rust: Any, index: int) -> None:
    from sympy_extras.solvers.solvers import solve_equation
    f, v = _equations()[index]
    got = _outcome(lambda: _rust_list(_call('equation', f, v)))
    expected = _outcome(lambda: _shown(solve_equation(f, v)))
    assert got == expected, (f, v)


def test_lambert_and_invert(rust: Any) -> None:
    from sympy_extras.core import Symbol, Integer
    from sympy_extras.functions import exp, log
    from sympy_extras.solvers.bivariate import solve_lambert
    from sympy_extras.solvers.solvers import invert
    x, y = Symbol('x'), Symbol('y')
    for f in [x + log(x) - 2, x**2 - 2, x*exp(x) - 1, x - exp(x) + 2, x**x - 2, 2**x - x - 2, x*log(x) - 1, exp(x)*x**2 - 1,
              3*x - exp(-x)]:
        assert _outcome(lambda: _rust_list(_call('lambert', f, x))) == _outcome(lambda: _shown(solve_lambert(f, x))), f
    for f in [2*exp(x + 1) - y, x*log(x) - 1, log(2*x - 1) + 3, x**3 - y, exp(x**2) - 5]:
        assert _outcome(lambda: tuple(str(t) for t in _call('invert', f, x).args)) == \
            _outcome(lambda: tuple(str(t) for t in invert(f, x))), f


def test_systems(rust: Any) -> None:
    from sympy_extras.core import Symbol, Tuple
    from sympy_extras.solvers.solvers import solve_system
    x, y, z, a = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('a')
    a0, a1, a2 = Symbol('a0'), Symbol('a1'), Symbol('a2')
    cases = [([x + y - 2, x - y], [x, y]), ([x*y - 1, x - y], [x, y]), ([x + y - a], [x, y]), ([x**2 + 1, x - 1], [x]),
             ([-a2**3, -3*a1*a2**2, -3*a0*a2**2 - 3*a1**2*a2, -6*a0*a1*a2 - a1**3, -3*a0**2*a2 - 3*a0*a1**2,
               -3*a0**2*a1 + 2*a2 - 1, -a0**3 + a1], [a0, a1, a2]),
             ([x*z - y, x - y*z], [x, y, z]), ([x**2 + y**2 - 5, x*y - 2], [x, y]), ([x**2 - a*y, y - 1], [x, y])]
    for equations, unknowns in cases:
        r = _call('system', Tuple(*equations), Tuple(*unknowns))
        got = 'None' if str(r) == 'NONE' else [[(str(k), str(v)) for k, v in (p.args for p in s.args)] for s in r.args]
        found = solve_system(equations, unknowns)
        expected = 'None' if found is None else [[(str(k), str(v)) for k, v in s.items()] for s in found]
        assert got == expected, equations

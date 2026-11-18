"""``sympy_extras.calculus.util`` (``periodicity``, ``lcim``) against its
Rust port, on the cases of its tests, with and without ``check``; the
outputs compared as strings (``NotImplementedError`` on the Rust side,
from the parts of ``simplify`` not ported, is a skip)."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('periodicity', [Symbol(op), *args])


def _outcome(f: Any) -> Any:
    try:
        r = f()
        return ('ok', 'None' if r is None or (hasattr(r, 'args') and str(r) == '()') else str(r))
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _cases() -> list[tuple[Any, Any]]:
    from sympy_extras.core import I, Integer, Mod, Rational, Symbol
    from sympy_extras.functions import Abs, cos, cot, csc, exp, log, re, sec, sin, sqrt, tan
    x, y, a = Symbol('x'), Symbol('y'), Symbol('a', real=True)
    cases = [sin(2*x), (-2)*tan(4*x), sin(x)**2, 3**tan(3*x), tan(x)*cos(x), sin(x)**(tan(x)), tan(x)*sec(x),
             sin(2*x)*cos(2*x) - y, tan(x) + cot(x), sin(x) - cos(2*x), sin(x) - 1, sin(4*x) + sin(x)*cos(x), exp(sin(x)),
             log(cot(2*x)) - sin(cos(2*x)), sin(2*x)*exp(tan(x) - csc(2*x)), cos(sec(x) - csc(2*x)), tan(sin(2*x)),
             2*tan(x)**2, sin(re(x)), sin(x)**2 + cos(x)**2, sin(x) + I*cos(x), sin(Mod(x, 4)), Mod(sin(x), 4),
             tan(Mod(3*x - 2, 4)), Mod(sqrt(2)*(x + 1) + x, 3), Mod(x**2 + 1, x), exp(x), exp(I*x), log(x), exp(x)**sin(x),
             Abs(sin(Abs(sin(x)))), Abs(sin(tan(x))), Abs(sin(sin(x) + tan(x))), sin(x) > Rational(1, 2), x > 2,
             x**3 - x**2 + 1, Abs(x), Abs(x**2 - 1), Mod(x**2 + 4, 2), Mod(exp(x), 3)]
    cases += [f(x) for f in (lambda t: Abs(cos(t)), lambda t: Abs(sin(t)), lambda t: Abs(sec(t)), lambda t: Abs(csc(t)),
                             lambda t: Abs(tan(t)), lambda t: Abs(cot(t)))]
    out = [(c, x) for c in cases]
    out += [(tan(x), y), (x - sin(2*y), y), (sin(x)**y, y)]
    out += [(exp(I*a), a), (exp(a), a), (exp(sin(a)), a), (exp(2*I*a), a), (exp(a + I*sin(a)), a), (exp(cos(a/2) + sin(a)), a)]
    return out


@pytest.mark.parametrize("index", range(len(_cases())))
def test_periodicity(rust: Any, index: int) -> None:
    from sympy_extras.core.logic import false, true
    from sympy_extras.calculus.util import periodicity
    f, v = _cases()[index]
    for check in (False, True):
        got = _outcome(lambda: _call('periodicity', f, v, true if check else false))
        if got[0] == 'SKIP':
            continue
        assert got == _outcome(lambda: periodicity(f, v, check=check)), (f, v, check)


def test_lcim(rust: Any) -> None:
    from sympy_extras.core import Integer, Rational, pi, sqrt
    from sympy_extras.calculus.util import lcim
    for numbers in [[Rational(1, 2), Integer(2), Integer(3)], [pi/2, pi/4, pi], [2*pi, pi/2], [Integer(1), 2*pi],
                    [Rational(1, 2), Rational(3, 4), Rational(5, 6)], [sqrt(Integer(2)), 3*sqrt(Integer(2))/2], [pi, sqrt(Integer(2))]]:
        assert _outcome(lambda: _call('lcim', *numbers)) == _outcome(lambda: lcim(numbers)), numbers

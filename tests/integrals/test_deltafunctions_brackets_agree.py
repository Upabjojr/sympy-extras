"""The Rust ports of ``integrals/deltafunctions.py`` and
``integrals/brackets.py`` against the Python ones, on the calls of the
Python tests and on a list of integrands (read with the core's parser:
plain symbols; the numbers of the dummies are not compared)."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.parsing import parse
import sympy_extras.integrals.brackets as br
import sympy_extras.integrals.deltafunctions as df

from ..conftest import load

rc = load("sympy_extras_rs.core")


def as_tuple(r: object) -> object:
    if isinstance(r, br.SeriesCoefficient):
        return Tuple(r.coefficient, r.index, Integer(r.period), r.offset)
    if isinstance(r, br.FormalSeriesTerms):
        return Tuple(r.formula, r.index, r.start, r.independent)
    if r.__class__.__name__ == 'MellinTransform':
        return Tuple(r.transform, r.strip[0], r.strip[1])  # type: ignore[attr-defined]
    if isinstance(r, tuple):
        return Tuple(*[Symbol('NONE') if v is None else v for v in r])
    return r


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    if r is None or str(r) == 'NONE':
        return 'None'
    return re.sub(r'\b_?([a-zA-Z]+)\d*\b', r'\1', str(as_tuple(r)))


CASES: list[tuple[str, ...]] = [
    ('deltafunctions', 'change_mul', 'x', 'x'),
    ('deltafunctions', 'change_mul', 'x*y', 'x'),
    ('deltafunctions', 'change_mul', 'x*y*DiracDelta(x)', 'x'),
    ('deltafunctions', 'change_mul', 'x*y*DiracDelta(x)*DiracDelta(y)', 'x'),
    ('deltafunctions', 'change_mul', 'DiracDelta(x)**2', 'x'),
    ('deltafunctions', 'integrate', 'x', 'x'),
    ('deltafunctions', 'integrate', 'x + DiracDelta(x)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x, 1)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x, 2)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x, 3)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x, 4)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x, 5)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x, 6)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x, 7)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x, 8)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x, 9)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x, 10)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x - y)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(-x + y)', 'x'),
    ('deltafunctions', 'integrate', 'x*DiracDelta(x)', 'x'),
    ('deltafunctions', 'integrate', '(x - y)*DiracDelta(x - y)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x)**2', 'x'),
    ('deltafunctions', 'integrate', 'y*DiracDelta(x)**2', 'x'),
    ('deltafunctions', 'integrate', 'y*DiracDelta(x, 1)', 'x'),
    ('deltafunctions', 'integrate', 'x*sin(x)*cos(x)*DiracDelta(x)', 'x'),
    ('deltafunctions', 'integrate', 'x*sin(x)*cos(x)*DiracDelta(x - 1)', 'x'),
    ('deltafunctions', 'integrate', 'y**2*DiracDelta(x - z)*DiracDelta(y - z)', 'y'),
    ('deltafunctions', 'expand', 'DiracDelta(x**2 - 1)', 'x'),
    ('deltafunctions', 'expand', 'DiracDelta(x**2)', 'x'),
    ('deltafunctions', 'integrate', 'exp(x)*DiracDelta(x**2 - 4)', 'x'),
    ('deltafunctions', 'integrate', 'exp(x)*DiracDelta(x - 1)', 'x'),
    ('deltafunctions', 'integrate', 'x**2*DiracDelta(x - 1/2)', 'x'),
    ('deltafunctions', 'integrate', 'x*DiracDelta(x - 3)', 'x'),
    ('brackets', 'taylor', 'exp(-x)', 'x'),
    ('brackets', 'taylor', 'sin(x)', 'x'),
    ('brackets', 'taylor', 'log(x + 1)', 'x'),
    ('brackets', 'taylor', 'besselj(0, x)', 'x'),
    ('brackets', 'taylor', 'exp(x) + sin(x)', 'x'),
    ('brackets', 'taylor', 'log(x)', 'x'),
    ('brackets', 'taylor', 'x**2 + 1', 'x'),
    ('brackets', 'taylor', 'exp(x)', 'x'),
    ('brackets', 'taylor', '(exp(4*x) - exp(-4*x))**2', 'x'),
    ('brackets', 'taylor', '1/(x**2 + x + 1)', 'x'),
    ('brackets', 'taylor', 'log(1 - x)', 'x'),
    ('brackets', 'ramanujan', 'exp(-x)', 'x', 's'),
    ('brackets', 'ramanujan', '1/(x + 1)', 'x', 's'),
    ('brackets', 'ramanujan', 'sin(x)', 'x', 's'),
    ('brackets', 'ramanujan', 'cos(x)', 'x', 's'),
    ('brackets', 'ramanujan', 'erf(x)', 'x', 's'),
    ('brackets', 'ramanujan', 'log(x + 1)', 'x', 's'),
    ('brackets', 'ramanujan', 'atan(x)', 'x', 's'),
    ('brackets', 'ramanujan', 'exp(-x**2)', 'x', 's'),
    ('brackets', 'ramanujan', 'cos(sqrt(x))', 'x', 's'),
    ('brackets', 'ramanujan', 'exp(x)', 'x', 's'),
    ('brackets', 'mellin', 'x**2*exp(-3*x)', 'x', 's'),
    ('brackets', 'mellin', 'log(2*x**2 + 1)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-x)*sin(x)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-x**2)*cos(2*x)', 'x', 's'),
    ('brackets', 'mellin', 'exp(-x)*sin(x)*cos(x)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-x)*besselj(0, x)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-x)/(x + 1)', 'x', 's'),
    ('brackets', 'taylor', 'exp(y)', 'y'),
    ('brackets', 'taylor', 'sin(y)', 'y'),
    ('brackets', 'taylor', 'cos(y)', 'y'),
    ('brackets', 'taylor', 'sinh(y)', 'y'),
    ('brackets', 'taylor', 'cosh(y)', 'y'),
    ('brackets', 'mellin', 'exp(-x)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-x)', 'x', 's'),
    ('brackets', 'formal', 'exp(-x)', 'x'),
    ('brackets', 'mellin', 'exp(-x**2)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-x**2)', 'x', 's'),
    ('brackets', 'taylor', 'exp(-x**2)', 'x'),
    ('brackets', 'formal', 'exp(-x**2)', 'x'),
    ('brackets', 'mellin', 'sin(x)', 'x', 's'),
    ('brackets', 'brackets', 'sin(x)', 'x', 's'),
    ('brackets', 'formal', 'sin(x)', 'x'),
    ('brackets', 'mellin', 'cos(x)', 'x', 's'),
    ('brackets', 'brackets', 'cos(x)', 'x', 's'),
    ('brackets', 'taylor', 'cos(x)', 'x'),
    ('brackets', 'formal', 'cos(x)', 'x'),
    ('brackets', 'mellin', 'atan(x)', 'x', 's'),
    ('brackets', 'brackets', 'atan(x)', 'x', 's'),
    ('brackets', 'taylor', 'atan(x)', 'x'),
    ('brackets', 'formal', 'atan(x)', 'x'),
    ('brackets', 'mellin', '1/(1 + x)', 'x', 's'),
    ('brackets', 'ramanujan', '1/(1 + x)', 'x', 's'),
    ('brackets', 'brackets', '1/(1 + x)', 'x', 's'),
    ('brackets', 'taylor', '1/(1 + x)', 'x'),
    ('brackets', 'formal', '1/(1 + x)', 'x'),
    ('brackets', 'mellin', '1/(1 + x)**2', 'x', 's'),
    ('brackets', 'ramanujan', '1/(1 + x)**2', 'x', 's'),
    ('brackets', 'brackets', '1/(1 + x)**2', 'x', 's'),
    ('brackets', 'taylor', '1/(1 + x)**2', 'x'),
    ('brackets', 'formal', '1/(1 + x)**2', 'x'),
    ('brackets', 'mellin', 'log(1 + x)', 'x', 's'),
    ('brackets', 'ramanujan', 'log(1 + x)', 'x', 's'),
    ('brackets', 'brackets', 'log(1 + x)', 'x', 's'),
    ('brackets', 'taylor', 'log(1 + x)', 'x'),
    ('brackets', 'formal', 'log(1 + x)', 'x'),
    ('brackets', 'mellin', 'exp(-x)*sin(x)', 'x', 's'),
    ('brackets', 'ramanujan', 'exp(-x)*sin(x)', 'x', 's'),
    ('brackets', 'taylor', 'exp(-x)*sin(x)', 'x'),
    ('brackets', 'formal', 'exp(-x)*sin(x)', 'x'),
    ('brackets', 'mellin', 'exp(-x)*cos(x)', 'x', 's'),
    ('brackets', 'ramanujan', 'exp(-x)*cos(x)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-x)*cos(x)', 'x', 's'),
    ('brackets', 'taylor', 'exp(-x)*cos(x)', 'x'),
    ('brackets', 'formal', 'exp(-x)*cos(x)', 'x'),
    ('brackets', 'ramanujan', 'x**2*exp(-3*x)', 'x', 's'),
    ('brackets', 'brackets', 'x**2*exp(-3*x)', 'x', 's'),
    ('brackets', 'taylor', 'x**2*exp(-3*x)', 'x'),
    ('brackets', 'formal', 'x**2*exp(-3*x)', 'x'),
    ('brackets', 'mellin', 'cos(sqrt(x))', 'x', 's'),
    ('brackets', 'brackets', 'cos(sqrt(x))', 'x', 's'),
    ('brackets', 'taylor', 'cos(sqrt(x))', 'x'),
    ('brackets', 'formal', 'cos(sqrt(x))', 'x'),
    ('brackets', 'mellin', '1/(1 + x**2)', 'x', 's'),
    ('brackets', 'ramanujan', '1/(1 + x**2)', 'x', 's'),
    ('brackets', 'brackets', '1/(1 + x**2)', 'x', 's'),
    ('brackets', 'taylor', '1/(1 + x**2)', 'x'),
    ('brackets', 'formal', '1/(1 + x**2)', 'x'),
    ('brackets', 'mellin', 'exp(-x)*besselj(0, x)', 'x', 's'),
    ('brackets', 'ramanujan', 'exp(-x)*besselj(0, x)', 'x', 's'),
    ('brackets', 'taylor', 'exp(-x)*besselj(0, x)', 'x'),
    ('brackets', 'formal', 'exp(-x)*besselj(0, x)', 'x'),
    ('brackets', 'mellin', 'besselj(0, x)', 'x', 's'),
    ('brackets', 'ramanujan', 'besselj(0, x)', 'x', 's'),
    ('brackets', 'brackets', 'besselj(0, x)', 'x', 's'),
    ('brackets', 'formal', 'besselj(0, x)', 'x'),
    ('brackets', 'mellin', 'x*exp(-x**2)', 'x', 's'),
    ('brackets', 'ramanujan', 'x*exp(-x**2)', 'x', 's'),
    ('brackets', 'brackets', 'x*exp(-x**2)', 'x', 's'),
    ('brackets', 'taylor', 'x*exp(-x**2)', 'x'),
    ('brackets', 'formal', 'x*exp(-x**2)', 'x'),
    ('brackets', 'mellin', 'sin(x)/x', 'x', 's'),
    ('brackets', 'ramanujan', 'sin(x)/x', 'x', 's'),
    ('brackets', 'brackets', 'sin(x)/x', 'x', 's'),
    ('brackets', 'taylor', 'sin(x)/x', 'x'),
    ('brackets', 'formal', 'sin(x)/x', 'x'),
    ('brackets', 'mellin', 'exp(-2*x)*cos(3*x)', 'x', 's'),
    ('brackets', 'ramanujan', 'exp(-2*x)*cos(3*x)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-2*x)*cos(3*x)', 'x', 's'),
    ('brackets', 'taylor', 'exp(-2*x)*cos(3*x)', 'x'),
    ('brackets', 'formal', 'exp(-2*x)*cos(3*x)', 'x'),
    ('brackets', 'mellin', '1/(1 + x)**3', 'x', 's'),
    ('brackets', 'ramanujan', '1/(1 + x)**3', 'x', 's'),
    ('brackets', 'brackets', '1/(1 + x)**3', 'x', 's'),
    ('brackets', 'taylor', '1/(1 + x)**3', 'x'),
    ('brackets', 'formal', '1/(1 + x)**3', 'x'),
    ('brackets', 'mellin', 'exp(-x)/(1 + x)', 'x', 's'),
    ('brackets', 'ramanujan', 'exp(-x)/(1 + x)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-x)/(1 + x)', 'x', 's'),
    ('brackets', 'taylor', 'exp(-x)/(1 + x)', 'x'),
    ('brackets', 'formal', 'exp(-x)/(1 + x)', 'x'),
    ('brackets', 'mellin', 'sinh(x)*exp(-2*x)', 'x', 's'),
    ('brackets', 'ramanujan', 'sinh(x)*exp(-2*x)', 'x', 's'),
    ('brackets', 'brackets', 'sinh(x)*exp(-2*x)', 'x', 's'),
    ('brackets', 'taylor', 'sinh(x)*exp(-2*x)', 'x'),
    ('brackets', 'formal', 'sinh(x)*exp(-2*x)', 'x'),
    ('brackets', 'mellin', 'erfc(x)', 'x', 's'),
    ('brackets', 'ramanujan', 'erfc(x)', 'x', 's'),
    ('brackets', 'brackets', 'erfc(x)', 'x', 's'),
    ('brackets', 'taylor', 'erfc(x)', 'x'),
    ('brackets', 'formal', 'erfc(x)', 'x'),
    ('brackets', 'mellin', 'exp(-x**3)', 'x', 's'),
    ('brackets', 'ramanujan', 'exp(-x**3)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-x**3)', 'x', 's'),
    ('brackets', 'taylor', 'exp(-x**3)', 'x'),
    ('brackets', 'formal', 'exp(-x**3)', 'x'),
    ('brackets', 'mellin', 'atan(x)/x', 'x', 's'),
    ('brackets', 'ramanujan', 'atan(x)/x', 'x', 's'),
    ('brackets', 'brackets', 'atan(x)/x', 'x', 's'),
    ('brackets', 'taylor', 'atan(x)/x', 'x'),
    ('brackets', 'formal', 'atan(x)/x', 'x'),
    ('brackets', 'mellin', 'x/(1 + x**2)', 'x', 's'),
    ('brackets', 'ramanujan', 'x/(1 + x**2)', 'x', 's'),
    ('brackets', 'brackets', 'x/(1 + x**2)', 'x', 's'),
    ('brackets', 'taylor', 'x/(1 + x**2)', 'x'),
    ('brackets', 'formal', 'x/(1 + x**2)', 'x'),
    ('brackets', 'mellin', 'exp(-x)*log(1 + x)', 'x', 's'),
    ('brackets', 'ramanujan', 'exp(-x)*log(1 + x)', 'x', 's'),
    ('brackets', 'brackets', 'exp(-x)*log(1 + x)', 'x', 's'),
    ('brackets', 'taylor', 'exp(-x)*log(1 + x)', 'x'),
    ('brackets', 'formal', 'exp(-x)*log(1 + x)', 'x'),
    ('brackets', 'mellin', 'cos(x)*exp(-x**2)', 'x', 's'),
    ('brackets', 'ramanujan', 'cos(x)*exp(-x**2)', 'x', 's'),
    ('brackets', 'brackets', 'cos(x)*exp(-x**2)', 'x', 's'),
    ('brackets', 'taylor', 'cos(x)*exp(-x**2)', 'x'),
    ('brackets', 'formal', 'cos(x)*exp(-x**2)', 'x'),
    ('brackets', 'mellin', 'sin(x)**2/x**2', 'x', 's'),
    ('brackets', 'ramanujan', 'sin(x)**2/x**2', 'x', 's'),
    ('brackets', 'brackets', 'sin(x)**2/x**2', 'x', 's'),
    ('brackets', 'taylor', 'sin(x)**2/x**2', 'x'),
    ('brackets', 'formal', 'sin(x)**2/x**2', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x - 1)*exp(x)', 'x'),
    ('deltafunctions', 'change_mul', 'DiracDelta(x - 1)*exp(x)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x**2 - 1)', 'x'),
    ('deltafunctions', 'change_mul', 'DiracDelta(x**2 - 1)', 'x'),
    ('deltafunctions', 'change_mul', 'y*DiracDelta(x, 1)', 'x'),
    ('deltafunctions', 'integrate', 'x**2*DiracDelta(x, 2)', 'x'),
    ('deltafunctions', 'change_mul', 'x**2*DiracDelta(x, 2)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(2*x - 3)*sin(x)', 'x'),
    ('deltafunctions', 'change_mul', 'DiracDelta(2*x - 3)*sin(x)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x)*DiracDelta(y)', 'x'),
    ('deltafunctions', 'change_mul', 'DiracDelta(x)*DiracDelta(y)', 'x'),
    ('deltafunctions', 'change_mul', 'x*DiracDelta(x)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x - y)*x', 'x'),
    ('deltafunctions', 'change_mul', 'DiracDelta(x - y)*x', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(3*x**2 - 3*x - 6)*x', 'x'),
    ('deltafunctions', 'change_mul', 'DiracDelta(3*x**2 - 3*x - 6)*x', 'x'),
    ('deltafunctions', 'integrate', 'cos(x)*DiracDelta(x, 3)', 'x'),
    ('deltafunctions', 'change_mul', 'cos(x)*DiracDelta(x, 3)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(cos(x))', 'x'),
    ('deltafunctions', 'change_mul', 'DiracDelta(cos(x))', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x**2 + 1)', 'x'),
    ('deltafunctions', 'change_mul', 'DiracDelta(x**2 + 1)', 'x'),
    ('deltafunctions', 'integrate', 'DiracDelta(x - 1) + x', 'x'),
    ('deltafunctions', 'change_mul', 'DiracDelta(x - 1) + x', 'x'),
    ('deltafunctions', 'integrate', 'exp(-x)*DiracDelta(x + 2, 1)', 'x'),
    ('deltafunctions', 'change_mul', 'exp(-x)*DiracDelta(x + 2, 1)', 'x'),
]

PYTHON: dict[tuple[str, str], Callable[..., object]] = {
    ('deltafunctions', 'integrate'): df.deltaintegrate,
    ('deltafunctions', 'expand'): df.expand_diracdelta,
    ('deltafunctions', 'change_mul'): df.change_mul,
    ('brackets', 'mellin'): br.mellin_transform_series,
    ('brackets', 'ramanujan'): br.ramanujan_master_theorem,
    ('brackets', 'brackets'): br.method_of_brackets,
    ('brackets', 'taylor'): br.taylor_coefficient,
    ('brackets', 'formal'): br.formal_series_terms,
}


@pytest.mark.parametrize('case', CASES, ids=[' '.join(c)[:80] for c in CASES])
def test_agree(case: tuple[str, ...]) -> None:
    args = [parse(a) for a in case[2:]]
    rust = outcome(lambda: rc.logic_call(case[0], [Symbol(case[1])] + args))
    if rust == 'SKIP':
        pytest.skip('not ported')
    assert rust == outcome(lambda: PYTHON[(case[0], case[1])](*args))

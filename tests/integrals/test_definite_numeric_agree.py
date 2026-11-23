"""The numerical verification of :mod:`sympy_extras.integrals.definite`
(the quadrature, ``quadosc`` for an oscillatory half-line, the crossings
of the branch cuts of a complex segment, the check of a claimed
divergence, ``verify_numerically`` with the branches of a ``Piecewise``
and the isolated values of the parameters) against the Rust port."""
from __future__ import annotations

from fractions import Fraction

import pytest

from sympy_extras.core import Dummy, I, Integer, Rational, Symbol, Tuple, oo, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.core.relational import ne as Ne
from sympy_extras.functions import Abs, Piecewise, Si, cos, exp, log, sin
from sympy_extras.integrals import definite as native
from ..conftest import load

rc = load("sympy_extras_rs.core")

x, y, z, k = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('k')
a = Symbol('a', positive=True)

VERIFY: list[tuple[Basic, Basic, Basic, Basic, list[Basic]]] = [
    (Integer(1), exp(-x), Integer(0), oo, []),
    (Integer(2), exp(-x), Integer(0), oo, []),
    (1/a, exp(-a*x), Integer(0), oo, []),
    (1/a**2, exp(-a*x), Integer(0), oo, []),
    (pi/2, sin(x)/x, Integer(0), oo, []),
    (pi/2, sin(x)/x, Integer(2), oo, []),
    (pi/2 - Si(2), sin(x)/x, Integer(2), oo, []),
    (-2*I/3, z**2, -I, I, []),
    (sqrt(6)*pi/36, x**2/(3 + 2*x**6), Integer(0), oo, []),
    (Integer(-1), log(x), Integer(0), Integer(1), []),
    (Piecewise(((y**2 - 1)/log(y), (y > 0) & Ne(y, 1)), (Integer(2), True)), y**x, Integer(0), Integer(2), [y > 0]),
    (Piecewise(((y**2 - 1)/log(y), (y > 0) & Ne(y, 1)), (Integer(3), True)), y**x, Integer(0), Integer(2), [y > 0]),
    (1/(k + 1), x**k, Integer(0), Integer(1), [k > -1]),
    (pi/(2*k), 1/(k**2 + x**2), Integer(0), oo, [k > 0]),
    (Integer(0), sin(k*x), -pi, pi, []),
]


@pytest.mark.parametrize('case', range(len(VERIFY)))
def test_verify_numerically(case: int) -> None:
    value, f, lo, hi, assumptions = VERIFY[case]
    p = native.verify_numerically(value, f, x if f.has(x) else z, lo, hi, assumptions)
    r = rc.logic_call('definite', [Symbol('verify'), value, f, x if f.has(x) else z, lo, hi, Tuple(*assumptions),
                                   Integer(2), Tuple()])
    assert str(r) == {True: 'True', False: 'False', None: 'NONE'}[p]


QUADRATURE: list[tuple[Basic, Basic, Basic, list[Basic]]] = [
    (exp(-x**2), -oo, oo, []),
    (log(Abs(x - Rational(1, 2))), Integer(0), Integer(1), [Rational(1, 2)]),
    (1/x, Integer(0), Integer(1), []),
    (sin(x)/x, Integer(2), oo, []),
    (cos(x)/(1 + x**2), Integer(0), oo, []),
    (2**log(x), -I, I, []),
    (sqrt(x)*exp(-x), Integer(0), oo, []),
]


@pytest.mark.parametrize('case', range(len(QUADRATURE)))
def test_quadrature(case: int) -> None:
    f, lo, hi, inside = QUADRATURE[case]
    p = native._quadrature(f, x, lo, hi, inside)
    r = rc.logic_call('definite', [Symbol('quadrature'), f, x, lo, hi, Tuple(*inside)])
    if p is None:
        assert str(r) == 'NONE'
        return
    re, im = (complex(Fraction(int(v.p), int(v.q))) for v in r.args)
    ours = re + 1j*im.real
    assert abs(ours - p) <= 1e-10*(1 + abs(p))


def test_converges_crossings_regions() -> None:
    for f, lo, hi in [(1/x, Integer(0), Integer(1)), (exp(x), Integer(0), oo), (exp(-x), Integer(0), oo),
                      (x**2*log(x + 1), Integer(0), oo), (x**k, Integer(0), Integer(1))]:
        r = rc.logic_call('definite', [Symbol('converges'), f, x, lo, hi, Tuple()])
        assert bool(r) == native._quadrature_converges(f, x, lo, hi)
    t = Dummy('t', real=True)
    for g in [2*I*Integer(2)**log(2*I*t - I), sqrt(t**2 - Rational(1, 4)) + log(3*t - 1)]:
        r = rc.logic_call('definite', [Symbol('crossings'), g, t])
        # (Python's atoms are a set: the order is the hashes'; the nodes are sorted before use)
        assert sorted(str(c) for c in r.args) == sorted(str(c) for c in native._axis_crossings(g, t))
    value = Piecewise(((y**2 - 1)/log(y), (y > 0) & Ne(y, 1)), (Integer(2), True))
    r = rc.logic_call('definite', [Symbol('regions'), value, Tuple(y)])
    assert [str(c) for c in r.args] == [str(c) for c in native._branch_regions(value, [y])]

"""The module-level helpers of the integrator of
:mod:`sympy_extras.integrals.definite` against the Rust port, as strings."""
from __future__ import annotations

from typing import Any, Callable

import pytest

from sympy_extras.core import Dummy, I, Integer, Rational, Symbol, Tuple, oo, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import false, true
from sympy_extras.core.relational import Eq, ne
from sympy_extras.functions import (Abs, Piecewise, abs_, arg, asinh, cos, cosh, exp, hermite, laguerre, log,
                                    sign, sin, tan, zeta)
from sympy_extras.integrals import definite as d
from sympy_extras.integrals.integral import Integral
from sympy_extras.sets import FiniteSet, Interval, Intersection, Union, EmptySet
from ..conftest import load

rc = load("sympy_extras_rs.core")

x, a, b, c, y, n = (Symbol(s) for s in 'xabcyn')
mu, s = Symbol('mu'), Symbol('sigma')
u = Symbol('u', positive=True)
t = Symbol('t')
B = lambda v: true if v else false  # noqa: E731


def _none(v: object) -> object:
    return 'NONE' if v is None else v


CASES: list[tuple[str, list[Basic], Callable[[], Any]]] = [
    ('positions', [Integer(0), a, b, true], lambda: Tuple(*[Tuple(c_, p) for c_, p in d._positions(Integer(0), a, b, True)])),
    ('positions', [Integer(0), a, oo, false], lambda: Tuple(*[Tuple(c_, p) for c_, p in d._positions(Integer(0), a, oo, False)])),
    ('positions', [c, -oo, b, true], lambda: Tuple(*[Tuple(c_, p) for c_, p in d._positions(c, -oo, b, True)])),
    ('merged', [((a > c) & (b > c)) | ((a > c) & (b < c)) | ((a < c) & (b > c)) | ((a < c) & (b < c))],
     lambda: d._merged_disjunction(((a > c) & (b > c)) | ((a > c) & (b < c)) | ((a < c) & (b > c)) | ((a < c) & (b < c)))),
    ('merged', [((a > 1) & (b > 1)) | ((a < -1) & (b < -1))],
     lambda: d._merged_disjunction(((a > 1) & (b > 1)) | ((a < -1) & (b < -1)))),
    ('strict', [c <= 0, c*log(-c) - 1], lambda: d._strict_where_undefined(c <= 0, c*log(-c) - 1)),
    ('strict', [c <= 0, c - 1], lambda: d._strict_where_undefined(c <= 0, c - 1)),
    ('odd_roots', [abs_(8*c**3)**Rational(1, 3)*sign(8*c**3)], lambda: d._real_odd_roots(abs_(8*c**3)**Rational(1, 3)*sign(8*c**3))),
    ('odd_roots', [abs_(-c**3)**Rational(1, 3)*sign(-c**3)], lambda: d._real_odd_roots(abs_(-c**3)**Rational(1, 3)*sign(-c**3))),
    ('odd_roots', [abs_(c + 1)**Rational(1, 3)*sign(c + 1)], lambda: d._real_odd_roots(abs_(c + 1)**Rational(1, 3)*sign(c + 1))),
    ('nowhere', [(abs_(arg(mu/s**2)) < pi/2) & (abs_(arg(-mu/s**2)) < pi/2), Tuple(mu, s), Tuple()],
     lambda: B(d._holds_nowhere((abs_(arg(mu/s**2)) < pi/2) & (abs_(arg(-mu/s**2)) < pi/2), [mu, s], None))),
    ('nowhere', [abs_(arg(mu/s**2)) < pi/2, Tuple(mu, s), Tuple()],
     lambda: B(d._holds_nowhere(abs_(arg(mu/s**2)) < pi/2, [mu, s], None))),
    ('outside', [1/(n + 1), n > -1, x**n, x, Integer(0), Integer(1), Tuple()],
     lambda: B(d._holds_outside(1/(n + 1), n > -1, x**n, x, Integer(0), Integer(1), None))),
    ('real_for_real', [c], lambda: B(d._real_for_real_parameters(c))),
    ('real_for_real', [I*y], lambda: B(d._real_for_real_parameters(I*y))),
    ('real_for_real', [sqrt(-c)], lambda: B(d._real_for_real_parameters(sqrt(-c)))),
    ('complex_for_real', [c*(-Rational(1, 2) + sqrt(3)*I/2)], lambda: B(d._complex_for_real_parameters(c*(-Rational(1, 2) + sqrt(3)*I/2)))),
    ('arcsine', [sin(x)**Rational(1, 3)*cos(x)**Rational(1, 3), x, u],
     lambda: d._arcsine_substitution(sin(x)**Rational(1, 3)*cos(x)**Rational(1, 3), x, u)),
    ('squares', [a**2*(1 - cos(t))**2 + a**2*sin(t)**2], lambda: d._trigonometric_squares(a**2*(1 - cos(t))**2 + a**2*sin(t)**2)),
    ('squares', [3 - 3*cos(2*x)], lambda: d._trigonometric_squares(3 - 3*cos(2*x))),
    ('squares', [tan(x)**2 + 1], lambda: d._trigonometric_squares(tan(x)**2 + 1)),
    ('combined', [exp(-a*t)*exp(-s*t)*t + exp(t)*exp(2*t)], lambda: d._combined_exponentials(exp(-a*t)*exp(-s*t)*t + exp(t)*exp(2*t))),
    ('combined', [exp(a)*exp(t)*exp(2*t), t], lambda: d._combined_exponentials(exp(a)*exp(t)*exp(2*t), t)),
    ('sums_to_products', [(sin(19*x) + sin(20*x))/(cos(19*x) + cos(20*x)), x],
     lambda: d._sums_to_products((sin(19*x) + sin(20*x))/(cos(19*x) + cos(20*x)), x)),
    ('generator', [exp(-2*x) + exp(-3*x), x, false], lambda: _none(d._exponential_generator(exp(-2*x) + exp(-3*x), x, False))),
    ('generator', [exp(x/2) + exp(-x/3), x, true], lambda: _none(d._exponential_generator(exp(x/2) + exp(-x/3), x, True))),
    ('generator', [exp(-a*x), x, true], lambda: _none(d._exponential_generator(exp(-a*x), x, True))),
    ('trig_zeros', [sin(2*x + 1), x, Integer(0), Integer(5)], lambda: _none(Tuple(*d._trigonometric_zeros(sin(2*x + 1), x, Integer(0), Integer(5))))),
    ('trig_zeros', [cos(x), x, -pi, 2*pi], lambda: _none(Tuple(*d._trigonometric_zeros(cos(x), x, -pi, 2*pi)))),
    ('singular', [1/((x - a)*(x - b)), x, Integer(0), Integer(1)], lambda: _none(Tuple(*d._singular_points(1/((x - a)*(x - b)), x, Integer(0), Integer(1))))),
    ('singular', [1/(x**3 - c**3), x, Integer(0), oo], lambda: _none(Tuple(*d._singular_points(1/(x**3 - c**3), x, Integer(0), oo)))),
    ('singular', [tan(x), x, Integer(0), Integer(4)], lambda: _none(Tuple(*d._singular_points(tan(x), x, Integer(0), Integer(4))))),
    ('candidates', [Union(FiniteSet(a), FiniteSet(b)), x, Integer(0), Integer(1)],
     lambda: _none(Tuple(*d._candidate_points(Union(FiniteSet(a), FiniteSet(b)), x, Integer(0), Integer(1))))),
    ('candidates', [EmptySet, x, Integer(0), Integer(1)], lambda: Tuple(*d._candidate_points(EmptySet, x, Integer(0), Integer(1)))),
    ('flattened', [((-sin(x))**a)**(1/a) + ((x + 1)**Rational(1, 2))**Rational(2, 3), x, Integer(0), Integer(1)],
     lambda: d._flattened_powers(((-sin(x))**a)**(1/a) + ((x + 1)**Rational(1, 2))**Rational(2, 3), x, Integer(0), Integer(1))),
    ('algebraic', [sqrt(x**2 + 1)/(x + 1), x], lambda: B(d._algebraic(sqrt(x**2 + 1)/(x + 1), x))),
    ('algebraic', [sqrt(exp(x) + 1), x], lambda: B(d._algebraic(sqrt(exp(x) + 1), x))),
    ('radical_base', [sqrt(1 - x**2)*x + (1 - x**2)**Rational(3, 2), x],
     lambda: _none((lambda r: None if r is None else Tuple(r[0], r[1]))(d._radical_base(sqrt(1 - x**2)*x + (1 - x**2)**Rational(3, 2), x)))),
    ('denest', [sqrt(c**2*x**2), Tuple(c < 0)], lambda: d._denest(sqrt(c**2*x**2), [c < 0])),
    ('denest', [sqrt(c**2*x**2), Tuple()], lambda: d._denest(sqrt(c**2*x**2), None)),
    ('signed_nodes', [sqrt(x - 1)*log(x + a) + x**2 + log(a), x], lambda: Tuple(*d._signed_nodes(sqrt(x - 1)*log(x + a) + x**2 + log(a), x))),
    ('log_inside', [log(log(1/x)), x], lambda: B(d._logarithm_inside(log(log(1/x)), x))),
    ('log_inside', [log(x)**2*log(1 + x), x], lambda: B(d._logarithm_inside(log(x)**2*log(1 + x), x))),
    ('log_inside', [1/log(x), x], lambda: B(d._logarithm_inside(1/log(x), x))),
    ('zeta_one', [zeta(y)*(y - 1) + y, y], lambda: d._zeta_about_one(zeta(y)*(y - 1) + y, y)),
    ('finite_at', [sin(x)/x, x, Integer(0)], lambda: B(d._finite_at(sin(x)/x, x, Integer(0)))),
    ('finite_at', [exp(x), x, Integer(1)], lambda: B(d._finite_at(exp(x), x, Integer(1)))),
    ('powers_of_logs', [exp(-y*(-log(u)) + 2), u], lambda: d._powers_of_logarithms(exp(-y*(-log(u)) + 2), u)),
    ('log_powers_only', [log(u)**2 + log(u + 1), u], lambda: B(d._log_powers_only(log(u)**2 + log(u + 1), u))),
    ('hermite', [hermite(2*n, x), Tuple(Eq(n, n))], lambda: d._hermite_as_kummer(hermite(2*n, x), None)),
    ('quarter', [3*pi/2], lambda: _none(d._quarter(3*pi/2))),
    ('quarter', [pi/3], lambda: _none(d._quarter(pi/3))),
    ('tangent', [pi/2], lambda: d._tangent_of(pi/2)),
    ('tangent', [pi/4], lambda: d._tangent_of(pi/4)),
    ('endpoint', [log(x)*x, x, Integer(0), Symbol('+')], lambda: _none(d._endpoint(log(x)*x, x, Integer(0), '+'))),
    ('endpoint', [exp(-x), x, oo, Symbol('-')], lambda: _none(d._endpoint(exp(-x), x, oo, '-'))),
    ('from_integrate', [Piecewise((1/a, abs_(arg(a)) < pi/2), (Integral(exp(-a*x), x, Integer(0), oo), True)), x],
     lambda: (lambda r: 'NONE' if r is None else Tuple(r.value, r.condition))(
         d._from_integrate(Piecewise((1/a, abs_(arg(a)) < pi/2), (Integral(exp(-a*x), x, Integer(0), oo), True)), x))),
    ('even', [Integer(4)], lambda: B(d._even(Integer(4)))),
    ('factored', [x**2 - 2*x + 1, x], lambda: (lambda r: 'NONE' if r is None else Tuple(r[0], Tuple(*[Tuple(f, m) for f, m in r[1]]), B(r[2])))(d._factored(x**2 - 2*x + 1, x))),
    ('factored', [(1 - cos(x))*a, x], lambda: (lambda r: 'NONE' if r is None else Tuple(r[0], Tuple(*[Tuple(f, m) for f, m in r[1]]), B(r[2])))(d._factored((1 - cos(x))*a, x))),
    ('factored', [tan(x) + 1, x], lambda: (lambda r: 'NONE' if r is None else Tuple(r[0], Tuple(*[Tuple(f, m) for f, m in r[1]]), B(r[2])))(d._factored(tan(x) + 1, x))),
    ('forms', [sin(x)**2*exp(-a*x)*exp(-x), x, Tuple()], lambda: Tuple(*d._forms(sin(x)**2*exp(-a*x)*exp(-x), x, None))),
    ('forms', [asinh(x)/cosh(x), x, Tuple()], lambda: Tuple(*d._forms(asinh(x)/cosh(x), x, None))),
    ('forms', [laguerre(n, x)*exp(-x), x, Tuple()], lambda: Tuple(*d._forms(laguerre(n, x)*exp(-x), x, None))),
]


@pytest.mark.parametrize('case', range(len(CASES)))
def test_helper(case: int) -> None:
    name, args, python = CASES[case]
    try:
        expected = str(python())
    except NotImplementedError:
        pytest.skip("not implemented in Python")
    try:
        found = str(rc.logic_call('definite', [Symbol('helper'), Symbol(name), *args]))
    except NotImplementedError as err:
        if 'not ported' in str(err):
            pytest.skip(str(err))
        raise
    if name in UNORDERED:
        # Python lists these points in the order of a set of strings,
        # which changes with PYTHONHASHSEED; the placement orders them
        found, expected = (sorted(v.strip('()').split(', ')) for v in (found, expected))
    assert found == expected


UNORDERED = {'singular', 'candidates'}

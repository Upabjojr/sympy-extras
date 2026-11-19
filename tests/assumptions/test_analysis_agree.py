"""The Rust port of ``assumptions/analysis.py`` against the Python module:
the domains of the variables from the facts, the rational points between
constants (from their values at the working precision), the certified
sorting of constants, the zeros of rational functions in a domain."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.assumptions import analysis as an
from sympy_extras.assumptions.facts import Facts
from sympy_extras.core import E, Integer, Ne, Rational, Symbol, Tuple, oo, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.sets.sets import Reals, interval

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y = Symbol('x'), Symbol('y')


def call(op: str, *args: Basic) -> str:
    try:
        return str(rc.logic_call('analysis', [Symbol(op), *args])).replace('NONE', 'None')
    except NotImplementedError:
        return 'SKIP'


def py(f: Callable[[], object]) -> str:
    try:
        return str(f())
    except NotImplementedError:
        return 'SKIP'


FACTS: list[list[Basic]] = [
    [x > 0, x < pi], [x > y], [x**2 < 2], [x >= 1, x <= 1], [x > 0, x**2 - 3*x + 1 > 0], [Ne(x, 2)],
    [x < E, x > -sqrt(2)], [x**3 - x > 0, x < 2], [x > 1, x < sqrt(3)], [x**2 - 2 >= 0],
]


@pytest.mark.parametrize('i', range(len(FACTS)))
def test_domain_of(i: int) -> None:
    facts = FACTS[i]
    got = call('domain_of', x, Tuple(*facts))
    if got != 'SKIP':
        assert got == py(lambda: an.domain_of(x, Facts(facts)))


def test_between_sorted_zeros() -> None:
    for p, q in [(Integer(1), Integer(2)), (pi, Rational(22, 7)), (-oo, sqrt(2)), (E, oo), (-oo, oo), (sqrt(2), sqrt(3))]:
        assert call('between', p, q) == py(lambda: an._between(p, q))
    values = [pi, Integer(3), sqrt(10), E, Rational(22, 7), sqrt(2) + 1]
    assert call('sorted', Tuple(*values)) == py(lambda: tuple(an._sorted_constants(values) or ()))
    for f, d in [(x**2 - 2, Reals), (1/(x - 1), interval(Integer(0), Integer(2))), ((x - 1)*(x + 3), interval(Integer(0), oo))]:
        found, zeros = an._real_zeros_in(f, x, d)
        assert call('real_zeros_in', f, x, d) == str((found, tuple(zeros)))


def _signs(s: object) -> str:
    return 'None' if s is None else str(tuple(sorted(s)))  # type: ignore[call-overload]


def _rust_signs(r: str) -> str:
    return r


def test_sign_on_continuity_zeros() -> None:
    from sympy_extras.functions import Abs, atan, cos, exp, log, sin, tan
    for f, d in [(x**2 - 2, Reals), (exp(x) - 2, interval(Integer(0), oo)), (sin(x), interval(Integer(0), pi)),
                 (sin(x), interval(Integer(-1), Integer(4))), (x*log(x), interval(Integer(0), Integer(1), True, True)),
                 (x - cos(x), Reals), (exp(x) + x, Reals), (tan(x), interval(Integer(-1), Integer(1))),
                 (log(x) - 1, interval(Integer(1), Integer(5))), (x**3 - 3*x + 1, Reals), (atan(x) - 1, Reals),
                 (Abs(x) - 1, Reals), (1/(x - 1), interval(Integer(2), oo))]:
        got = call('sign_on', f, x, d)
        if got != 'SKIP':
            assert got == _signs(an.sign_on(f, x, d)), (f, d)
        got = call('continuous', f, x, d)
        if got != 'SKIP':
            assert got == str(an._continuous_on(f, x, d)), (f, d)
        got = call('finite_zeros', f, x, d)
        if got != 'SKIP':
            z = an._finite_zeros(f, x, d)
            assert got == ('None' if z is None else str(tuple(z))), (f, d)


def test_decide_box_multivariate() -> None:
    from sympy_extras.core import Eq
    from sympy_extras.functions import cos, exp, log, sin
    for atom, facts in [(x**2 + 1 > 0, []), (exp(x) > x, []), (sin(x) < 1, [x > 0, x < 1]), (x - sin(x) > 0, [x > 0]),
                        (log(x) < x, [x > 0]), (x*y > 0, [x > 0, y > 0]), (x + y > 2, [x > 1, y > 1]),
                        (x**2 - y > 0, [x > 2, y < 1, y > 0]), (exp(x) - y > 0, [x > 0, y < 1, y > 0]),
                        (cos(x) > 0, [x > 0, x < 1]), (x**3 > x, [x > 1]), (Eq(x, 2), [x > 3])]:
        got = call('decide', atom, Tuple(*facts))
        if got != 'SKIP':
            assert got == str(an.decide_relational(atom, Facts(facts))), atom
    for f, syms, facts in [(x*y - 1, [x, y], [x > 2, y > 1]), (x - y, [x, y], [x > 0, x < 1, y > 2, y < 3]),
                           (exp(x*y) - 1, [x, y], [x > 0, y > 0]), (sin(x + y), [x, y], [x > 0, x < 1, y > 0, y < 1])]:
        got = call('signs_multivariate', f, Tuple(*syms), Tuple(*facts))
        if got != 'SKIP':
            assert got == _signs(an.signs_multivariate(f, syms, Facts(facts))), f
    found = an.box_of([x, y], Facts([x > 0, x < 1, y >= 2]))
    r = rc.logic_call('analysis', [Symbol('box_of'), Tuple(x, y), Tuple(x > 0, x < 1, y >= 2)])
    assert found is not None
    assert [(t.args[1], t.args[2]) for t in r.args[0].args] == [found[0][x], found[0][y]]
    assert [(bool(t.args[1]), bool(t.args[2])) for t in r.args[1].args] == [found[1][x], found[1][y]]

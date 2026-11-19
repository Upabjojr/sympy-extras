"""The Rust interval arithmetic (``assumptions/ivarith.rs``, mpmath's
``iv`` at 53 bits) and the enclosures of ``assumptions/intervals.py``
against the Python ones: the exact endpoints."""
from __future__ import annotations

import random

import mpmath
import pytest

from sympy_extras.assumptions.intervals import evaluate
from sympy_extras.core import E, Integer, Rational, Symbol, Tuple, false, oo, pi, true
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import mpf_to_fraction
from sympy_extras.core.numbers import number_from_fraction
from sympy_extras.functions import Abs, atan, cos, cosh, exp, log, sin, sinh, sqrt, tan, tanh

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y = Symbol('x'), Symbol('y')


def _endpoint(m: object) -> str:
    v = mpmath.mpf(m)
    if v == mpmath.inf:
        return 'oo'
    if v == -mpmath.inf:
        return '-oo'
    return str(number_from_fraction(mpf_to_fraction(v)))


def _random_cases(seed: int, count: int) -> list[tuple[Basic, tuple[Basic, Basic]]]:
    rng = random.Random(seed)
    functions = [exp, log, sin, cos, tan, atan, sinh, cosh, tanh, Abs, sqrt]

    def leaf() -> Basic:
        return rng.choice([x, y, x, Integer(rng.randint(-5, 5)), Rational(rng.randint(-9, 9), rng.randint(1, 7)), pi, E])

    def gen(d: int) -> Basic:
        if d == 0:
            return leaf()
        k = rng.random()
        if k < 0.25:
            return gen(d - 1) + gen(d - 1)
        if k < 0.5:
            return gen(d - 1)*gen(d - 1)
        if k < 0.65:
            return gen(d - 1)**rng.choice([Integer(2), Integer(3), Integer(-1), Integer(-2), Rational(1, 3), Rational(3, 2)])
        if k < 0.9:
            return rng.choice(functions)(gen(d - 1))
        return gen(d - 1) - gen(d - 1)

    cases = []
    for _ in range(count):
        lo = Rational(rng.randint(-20, 20), rng.randint(1, 5))
        cases.append((gen(rng.randint(1, 2)), (lo, lo + Rational(rng.randint(0, 20), rng.randint(1, 5)))))
    return cases


CASES: list[tuple[Basic, tuple[Basic, Basic]]] = [
    (exp(x), (Integer(0), Integer(1))), (sin(x), (Integer(0), Integer(4))), (cos(x), (Integer(-1), Integer(1))),
    (tan(x), (Integer(1), Integer(2))), (atan(x), (-oo, oo)), (log(x), (Integer(1), Integer(3))),
    (sqrt(x), (Integer(0), Integer(2))), (sinh(x)*cosh(x), (Integer(-1), Integer(2))), (tanh(x), (Integer(-3), Integer(3))),
    (Abs(x - 1), (Integer(-2), Integer(2))), (pi*x + E, (Integer(0), Integer(1))), (x**3 - 2*x, (Integer(-1), Integer(1))),
    (1/x, (Integer(1), Integer(2))), (1/x, (Integer(-1), Integer(1))), (exp(1/x), (Integer(1), oo)),
    (sin(x), (Integer(100), Integer(101))), (cos(x), (Rational(1, 3), Rational(1, 2))), (atan(E)*x, (Integer(-3), Integer(4))),
] + _random_cases(7, 300)


@pytest.mark.parametrize('i', range(len(CASES)))
def test_evaluate(i: int) -> None:
    e, (lo, hi) = CASES[i]
    box = {x: (lo, hi), y: (Integer(1), Integer(2))}
    v = evaluate(e, box)
    expected = 'NONE' if v is None else str((_endpoint(v.a), _endpoint(v.b)))
    r = rc.logic_call('intervals', [Symbol('evaluate'), e, Tuple(*[Tuple(s, a, b) for s, (a, b) in box.items()])])
    got = 'NONE' if str(r) == 'NONE' else str((str(r.args[0]), str(r.args[1])))
    assert got == expected, e


def _signs(s: object) -> str:
    return 'NONE' if s is None else str(tuple(sorted(s)))  # type: ignore[call-overload]


def _call(op: str, *args: Basic) -> Basic:
    return rc.logic_call('intervals', [Symbol(op), *args])


def _box(box: dict[Basic, tuple[Basic, Basic]]) -> Basic:
    return Tuple(*[Tuple(s, a, b) for s, (a, b) in box.items()])


def _out(r: Basic) -> str:
    return 'NONE' if str(r) == 'NONE' else (str(tuple(r.args)) if isinstance(r, Tuple) else str(r))


SIGN_CASES: list[tuple[Basic, dict[Basic, tuple[Basic, Basic]]]] = [
    (sin(x) - 2 + y, {x: (Integer(0), Integer(1)), y: (Integer(0), Integer(1))}),
    (exp(x) + 1, {x: (Integer(-1), Integer(1))}), (x**2 - 2, {x: (Integer(0), Integer(3))}),
    (x*y - 1, {x: (Integer(2), Integer(3)), y: (Integer(1), oo)}), (x - y, {x: (Integer(0), Integer(1)), y: (Integer(2), Integer(5))}),
    (log(x) + x, {x: (Integer(1), oo)}), (sin(x), {x: (Integer(0), Integer(3))}), (x**2, {x: (-oo, oo)}),
]


@pytest.mark.parametrize('i', range(len(SIGN_CASES)))
def test_signs_and_ranges(i: int) -> None:
    from sympy_extras.assumptions import intervals as iv
    e, box = SIGN_CASES[i]
    for weak in (False, True):
        assert _out(_call('signs_on_box', e, _box(box), Integer(120), (true if weak else false))) \
            == _signs(iv.signs_on_box(e, box, 120, weak))
    r = iv.range_on_box(e, box)
    assert _out(_call('range_on_box', e, _box(box))) == ('NONE' if r is None else str(r))
    openness = {s: (True, False) for s in box}
    r2 = iv.monotone_range(e, box, openness)
    assert _out(_call('monotone_range', e, _box(box), Tuple(*[Tuple(s, true, false) for s in box]))) \
        == ('NONE' if r2 is None else str(r2))
    s3 = iv.corner_signs(e, box, openness)
    assert _out(_call('corner_signs', e, _box(box), Tuple(*[Tuple(s, true, false) for s in box]))) \
        == _signs(s3)


def test_isolate_and_certified() -> None:
    from sympy_extras.assumptions import analysis, intervals as iv
    from sympy_extras.sets.sets import interval
    for f, piece in [(x**2 - 2, interval(Integer(0), Integer(3))), (sin(x), interval(Integer(-1), Integer(4))),
                     (x**3 - x, interval(Integer(-2), Integer(2), True, False)), (exp(x) - 3, interval(Integer(0), Integer(2)))]:
        assert _out(_call('isolate_signs', f, Tuple(), x, piece, Integer(400))) == _signs(iv.isolate_signs(f, x, piece))
    for v in [E - 2, sin(3) - Rational(1, 7), pi - Rational(355, 113), sqrt(2) + sqrt(3) - sqrt(5 + 2*sqrt(6)), x, Integer(0),
              exp(-50) - Rational(1, 10**22), cos(1)**2 + sin(1)**2 - 1]:
        got = _out(_call('certified_sign', v, Tuple()))
        assert got == str(analysis.certified_sign(v)).replace('None', 'NONE'), v
    for a_, b_ in [(pi, Integer(3)), (oo, oo), (Integer(2), -oo), (E, E)]:
        assert _out(_call('certified_compare', a_, Tuple(), b_)) == str(analysis.certified_compare(a_, b_)).replace('None', 'NONE')
    corner_box = {x: (Integer(0), oo), y: (Integer(1), Integer(2))}
    for e, corner in [(x/(x + y), {x: oo, y: Integer(1)}), (exp(-x)*y, {x: oo, y: Integer(2)}), (x + y, {x: Integer(0), y: Integer(1)})]:
        expected = iv.corner_value(e, corner, corner_box)
        got = _out(_call('corner_value', e, _box(corner_box), Tuple(*[Tuple(s, c) for s, c in corner.items()])))
        assert got == ('NONE' if expected is None else str(expected))

"""The Rust port of ``numberfields.qqbar`` (exact arithmetic on algebraic
numbers) against the Python module: the docstring examples on both
implementations, the roots of random integer polynomials, random sums,
products, quotients, powers, polynomial values, roots, real and imaginary
parts, the decisions (equality, order, sign) and primitive elements. The
numbers are compared through their reprs, minimal polynomials and
decisions, which do not depend on how far their regions were refined."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any, Callable

import pytest

from ..conftest import IMPLEMENTATIONS, load, stable_seed

F = Fraction


def qb(impl: str) -> Any:
    return load(impl + ".polys.numberfields.qqbar")


def test_known(impl: str) -> None:
    m = qb(impl)
    Algebraic, algebraic_roots = m.Algebraic, m.algebraic_roots
    sqrt2, = [r for r in algebraic_roots([1, 0, -2]) if r > 0]
    sqrt3, = [r for r in algebraic_roots([1, 0, -3]) if r > 0]
    s = sqrt2 + sqrt3
    assert repr(s) == 'Algebraic([1, 0, -10, 0, 1], 3.14626437)'
    assert s*s == 5 + 2*sqrt2*sqrt3
    assert ((sqrt2*sqrt2).minpoly, sqrt2*sqrt2 == 2) == ([1, -2], True)
    assert 1/(sqrt3 - sqrt2) == s
    i = Algebraic.imaginary_unit()
    assert repr((1 + i)**8) == 'Algebraic([1, -16], 16)'
    roots = algebraic_roots([1, 0, 0, -2])
    assert repr(roots) == ('[Algebraic([1, 0, 0, -2], 1.25992105), Algebraic([1, 0, 0, -2], -0.6299605249 - '
                           '1.091123636*I), Algebraic([1, 0, 0, -2], -0.6299605249 + 1.091123636*I)]')
    assert roots[0] + roots[1] + roots[2] == 0 and roots[0]*roots[1]*roots[2] == 2
    assert (roots[1]*roots[2]).is_real and roots[1].conjugate() == roots[2]
    low, high = algebraic_roots([1, -1, -1])
    assert high**2 == high + 1 and low*high == -1 and low < 0 < high
    assert Algebraic.from_rational(F(3, 4)).minpoly == [4, -3]
    assert repr(sqrt2.polynomial_value([1, 0, 0, 1])) == 'Algebraic([1, -2, -7], 3.828427125)'
    two = Algebraic.from_rational(2)
    assert repr((two + two.nth_root(2)).nth_root(2)) == 'Algebraic([1, 0, -4, 0, 2], 1.847759065)'
    assert Algebraic.from_rational(-1).nth_root(2) == Algebraic.imaginary_unit()
    assert repr(Algebraic.from_rational(-8).nth_root(3)) == 'Algebraic([1, -2, 4], 1 + 1.732050808*I)'
    assert repr(algebraic_roots([1, 0, -3, 0, 2])) == ('[Algebraic([1, 0, -2], -1.414213562), Algebraic([1, 1], -1), '
                                                     'Algebraic([1, -1], 1), Algebraic([1, 0, -2], 1.414213562)]')
    assert repr(algebraic_roots([1, 0, 1])) == '[Algebraic([1, 0, 1], 0 - 1*I), Algebraic([1, 0, 1], 0 + 1*I)]'
    theta, p, q = m.primitive_element(sqrt2, sqrt3)
    assert (repr(theta), p, q) == ('Algebraic([1, 0, -10, 0, 1], 3.14626437)', [F(1, 2), 0, F(-9, 2), 0],
                                   [F(-1, 2), 0, F(11, 2), 0])
    assert theta.polynomial_value(p) == sqrt2 and theta.polynomial_value(q) == sqrt3
    with pytest.raises(ZeroDivisionError):
        Algebraic.from_rational(0).inverse()
    with pytest.raises(TypeError):
        i < 1
    with pytest.raises(TypeError):
        float(i)
    with pytest.raises(ValueError):
        algebraic_roots([0, 0])


def random_polynomial(rng: random.Random) -> list[int]:
    while True:
        f = [rng.randint(1, 3)] + [rng.randint(-4, 4) for _ in range(rng.randint(1, 3))]
        if any(f[1:]):
            return f


def describe(value: Any) -> Any:
    """What the comparison looks at: the repr and the decisions."""
    if isinstance(value, list):
        return [describe(v) for v in value]
    if type(value).__name__ == 'Algebraic':
        return (repr(value), value.minpoly, value.is_real, value.sign() if value.is_real else None)
    return value


def outcome(function: Callable[[], Any]) -> Any:
    try:
        return describe(function())
    except (ValueError, ZeroDivisionError, TypeError, ArithmeticError) as e:
        return (type(e).__name__, str(e))


@pytest.mark.parametrize("seed", range(40))
def test_arithmetic_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("qqbar", seed))
    for _ in range(2):
        f, g = random_polynomial(rng), random_polynomial(rng)
        choice = rng.random(), rng.random(), rng.randint(-3, 4), rng.randint(2, 3)
        coefficients = [F(rng.randint(-3, 3), rng.choice([1, 2])) for _ in range(rng.randint(1, 4))]
        results = []
        for impl in IMPLEMENTATIONS:
            m = qb(impl)
            roots_f, roots_g = m.algebraic_roots(f), m.algebraic_roots(g)
            a = roots_f[int(choice[0]*len(roots_f))]
            b = roots_g[int(choice[1]*len(roots_g))]
            exponent, index = choice[2], choice[3]
            results.append([
                describe(roots_f), describe(roots_g),
                outcome(lambda: a + b), outcome(lambda: a - b), outcome(lambda: a*b), outcome(lambda: a/b),
                outcome(lambda: a**exponent), outcome(lambda: a.polynomial_value(coefficients)),
                outcome(lambda: a.nth_root(index)), outcome(lambda: a.real_part()), outcome(lambda: a.imaginary_part()),
                outcome(lambda: a.conjugate()), outcome(lambda: a == b), outcome(lambda: a < b),
                outcome(lambda: a == a.conjugate()), outcome(lambda: a*a == a**2),
            ])
        assert results[0] == results[1], (f, g, choice)


@pytest.mark.parametrize("seed", range(15))
def test_primitive_element_agrees(rust: Any, seed: int) -> None:
    rng = random.Random(stable_seed("primitive element", seed))
    for _ in range(2):
        f = [1, 0, rng.choice([-2, -3, -5, 1, 2])]
        g = [1, 0, 0, rng.choice([-2, -3, 2])] if rng.random() < 0.3 else [1, 0, rng.choice([-6, -7, 3, 5])]
        choice = rng.random(), rng.random()
        results = []
        for impl in IMPLEMENTATIONS:
            m = qb(impl)
            roots_f, roots_g = m.algebraic_roots(f), m.algebraic_roots(g)
            a = roots_f[int(choice[0]*len(roots_f))]
            b = roots_g[int(choice[1]*len(roots_g))]

            def element() -> Any:
                theta, p, q = m.primitive_element(a, b)
                return (repr(theta), p, q, theta.polynomial_value(p) == a, theta.polynomial_value(q) == b)
            results.append(outcome(element))
        assert results[0] == results[1], (f, g)

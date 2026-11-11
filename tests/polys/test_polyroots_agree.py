"""``roots`` of the Rust port against ``sympy_extras.polys.polyroots``:
products of random linear and quadratic factors with multiplicities,
random cubics and quartics (Cardano, Ferrari, Euler's resolvent),
binomials, cyclotomic polynomials, compositions, parametric
coefficients, the filters and ``multiple``, compared through ``repr``
and the class of the error; the cases which are ``NotImplementedError``
in the port (a quartic needing ``Piecewise``, ``EX``) are skipped."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed


def outcome(f: Any) -> str:
    try:
        return repr(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def check(rc: Any, e: Any, gens: list[Any], **options: Any) -> bool:
    from sympy_extras.polys.polyroots import roots
    got = outcome(lambda: rc.roots(e, gens, **options))
    if got == 'SKIP':
        return False
    expected = outcome(lambda: roots(e, *gens, **options))
    if got != expected and options.get("multiple"):
        # the order of the terms of a sum of numbers depends on the
        # rounding noise of their imaginary parts (_coefficient_rank: the
        # Python's -1.5e-23 is the port's exact 0): the same roots
        from collections import Counter
        assert Counter(map(repr, rc.roots(e, gens, **options))) == \
            Counter(map(repr, roots(e, *gens, **options))), (e, gens, options)
        return True
    assert got == expected, (e, gens, options)
    return True


def random_polynomial(rng: random.Random, x: Any, a: Any) -> Any:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    kind = rng.random()
    if kind < 0.4:
        factors: list[Any] = [Integer(rng.choice([1, 1, 2, -3]))]
        for _ in range(rng.randint(1, 3)):
            degree = rng.choice([1, 1, 2])
            coefficients: list[Any] = [Integer(rng.randint(-4, 4)) for _ in range(degree)]
            if a is not None and rng.random() < 0.3:
                coefficients[-1] = mul(coefficients[-1], a)
            f = add(pow(x, Integer(degree)), *[mul(c, pow(x, Integer(i))) for i, c in enumerate(coefficients)])
            factors.append(pow(f, Integer(rng.choice([1, 1, 2]))))
        return mul(*factors)
    if kind < 0.65:
        degree = rng.choice([3, 3, 4])
        coefficients = [Integer(rng.randint(-5, 5)) for _ in range(degree)]
        return add(pow(x, Integer(degree)), *[mul(c, pow(x, Integer(i))) for i, c in enumerate(coefficients)])
    if kind < 0.8:
        n = rng.randint(2, 6)
        c: Any = rng.choice([Integer(rng.randint(-9, 9)) or Integer(2), Rational(rng.randint(1, 5), 3)])
        if a is not None and rng.random() < 0.4:
            c = mul(c, pow(a, Integer(rng.randint(1, 2))))
        return add(pow(x, Integer(n)), c)
    if kind < 0.9:
        inner = add(pow(x, Integer(2)), Integer(rng.randint(-3, 3)))
        return add(pow(inner, Integer(2)), Integer(rng.randint(-3, 3)))
    return add(pow(x, Integer(rng.choice([4, 6, 8, 10, 12]))), Integer(rng.choice([-1, 1])))


@pytest.mark.parametrize("seed", range(40))
def test_roots(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("roots", seed))
    x, a = Symbol('x'), Symbol('a')
    compared = 0
    for _ in range(4):
        e = random_polynomial(rng, x, a if rng.random() < 0.3 else None)
        compared += check(rc, e, [x])
        option: dict[str, Any] = rng.choice([{"multiple": True}, {"filter": "Q"}, {"filter": "R"}, {"filter": "I"}, {"cubics": False},
                             {"quartics": False}, {"filter": "Z", "multiple": True}])
        compared += check(rc, e, [x], **option)
    assert compared >= 3


def test_edge_cases(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Symbol, sqrt
    from sympy_extras.functions import sin
    rc = load("sympy_extras_rs.core")
    x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
    cases: list[tuple[Any, list[Any]]] = [
        (Integer(3), [x]), (x, [x]), (x**5, [x]), (x**3*(x - 1), [x]), (x**2 - a**2, [x]),
        (x**2 + a*x, [x]), (a*x**2 + 1, [x]), (x**2*y - y, [x]), (x*y - 1, [x, y]), (sin(x) - 1, [sin(x)]),
        (x**2 - sqrt(2), [x]), (x**3 - I, [x]), ((x + 1)**3 - 8, [x]), (x**4 - 4*x**3 + 6*x**2 - 4*x + 1, [x]),
        (x**2 - 2*a*x + a**2 - 1, [x]), (x**6 + x**3 + 1, [x]), (x**4 + 4, [x]), (x**3 - a**3, [x]),
    ]
    for e, gens in cases:
        check(rc, e, gens)
        check(rc, e, gens, multiple=True)
    # the Python raises IndexError on the zero polynomial (_integer_basis of
    # no terms); the port a ValueError
    with pytest.raises(ValueError):
        rc.roots(Integer(0), [x])

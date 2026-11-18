"""``puiseux_expansions`` of ``sympy_extras.polys.puiseux`` against its Rust
port: the docstring curves (a node, a cusp, infinity, ramified and
conjugate branches), curves at algebraic points and over a number field,
and random curves at random points, compared on every branch
(ramification, conjugates, the series, its field, order and scale)."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def expansions_py(F: Any, x: Any, y: Any, point: Any, order: int, domain: Any) -> Any:
    from sympy_extras.polys.puiseux import puiseux_expansions
    return [(b.ramification, b.conjugates, str(b.as_expr(x)), [Fraction(c) for c in b.field.minpoly],
             str(b.field.alpha_expr), b.order, [Fraction(c) for c in b.scale])
            for b in puiseux_expansions(F, x, y, point, order=order, domain=domain)]


def expansions_rs(rc: Any, F: Any, x: Any, y: Any, point: Any, order: int, domain: Any) -> Any:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.containers import Tuple
    minpoly = Tuple(*[Rational(Fraction(c).numerator, Fraction(c).denominator) for c in domain.minpoly])
    r = rc.logic_call('puiseux_expansions', [F, x, y, point, Integer(order), minpoly, domain.alpha_expr])
    return [(int(b.args[0]), int(b.args[1]), rc.sstr(b.args[2]), [Fraction(str(c)) for c in b.args[3].args],
             str(b.args[4]), None if str(b.args[5]) == 'False' else int(b.args[5]),
             [Fraction(str(c)) for c in b.args[6].args]) for b in r.args]


def cases() -> list[tuple[Any, Any, int, Any]]:
    from sympy_extras.core import Integer, oo, sqrt, symbols
    from sympy_extras.polys.puiseux import RATIONALS, field_of
    x, y = symbols('x y')
    Q5 = field_of(sqrt(5))[0]
    return [(y**2 - x**3 - x**2, Integer(0), 4, RATIONALS), (y**3 - x**2, Integer(0), 2, RATIONALS),
            (y**2 - x**3 - 1, oo, 3, RATIONALS), (y**4 - x, Integer(0), 2, RATIONALS),
            (y**2 - 2 - x, Integer(0), 2, RATIONALS), (y**2 - x, Integer(1), 3, RATIONALS),
            (y**3 - 3*x*y + x**3, Integer(0), 3, RATIONALS), (y**2 - x**2 + 2, sqrt(2), 3, RATIONALS),
            ((y**2 - x)**2 - x**5, Integer(0), 3, RATIONALS), (y**2 - sqrt(5)*x, Integer(0), 2, Q5),
            (y**3 - x*y - x**4, oo, 2, RATIONALS), (y**2 + y - x**3, Integer(0), 4, RATIONALS)]


@pytest.mark.parametrize("index", range(12))
def test_fixed_curves(rust: Any, index: int) -> None:
    from sympy_extras.core import symbols
    rc = load("sympy_extras_rs.core")
    x, y = symbols('x y')
    F, point, order, domain = cases()[index]
    assert outcome(lambda: expansions_rs(rc, F, x, y, point, order, domain)) == \
        outcome(lambda: expansions_py(F, x, y, point, order, domain)), (F, point)


@pytest.mark.parametrize("seed", range(30))
def test_random_curves(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, oo, symbols
    from sympy_extras.polys.puiseux import RATIONALS
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("puiseux", seed))
    x, y = symbols('x y')
    n = rng.randint(2, 3)
    F = y**n + sum((rng.randint(-2, 2) * x**rng.randint(0, 3) * y**i for i in range(n)), Integer(0))
    F = F - rng.choice([x, x**2, x**3, Integer(1) + x])
    point = rng.choice([Integer(0), Integer(1), Integer(-1), oo])
    order = rng.randint(1, 3)
    assert outcome(lambda: expansions_rs(rc, F, x, y, point, order, RATIONALS)) == \
        outcome(lambda: expansions_py(F, x, y, point, order, RATIONALS)), (F, point, order)

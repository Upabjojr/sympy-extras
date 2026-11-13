"""The CAD projection factor sets of the Rust port against
``sympy_extras.polys.cad.projection``: McCallum's and Hong's operators,
the reduction by an equational constraint and the augmented sets, on
random polynomials in two and three variables with integer and rational
coefficients, compared through the printed ``Poly`` and the class of the
error."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed


def outcome(f: Callable[[], Any]) -> str:
    try:
        return repr(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def random_polynomial(rng: random.Random, gens: list[Any]) -> Any:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    terms = []
    for _ in range(rng.randint(1, 4)):
        c: Any = rng.choice([Integer(rng.randint(-3, 3)), Rational(rng.randint(-2, 2), 2)])
        terms.append(mul(c, *[pow(g, Integer(rng.randint(0, 2))) for g in gens if rng.random() < 0.6]))
    return add(*terms)


def check(rc: Any, polys: list[Any], gens: list[Any], method: str, equational: Any = None,
          extra: Any = None) -> bool:
    from sympy_extras.polys.cad import projection as pr
    got = outcome(lambda: rc.cad_projection_sets(polys, gens, method, equational, extra))
    if got == 'SKIP':
        return False

    def python() -> list[list[str]]:
        sets = pr.projection_sets(polys, gens, method, equational)
        if extra is not None:
            sets = pr.augmented_projection_sets(sets, gens, extra, method)
        return [[str(p) for p in level] for level in sets]
    expected = outcome(python)
    assert got == expected, (polys, gens, method, equational, extra)
    return True


@pytest.mark.parametrize("seed", range(30))
def test_projection_sets(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("cad projection", seed))
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    gens = rng.choice([[x, y], [x, y], [x, y, z]])
    polys = [random_polynomial(rng, gens) for _ in range(rng.randint(1, 3))]
    compared = 0
    for method in ("mccallum", "hong"):
        compared += check(rc, polys, gens, method)
    compared += check(rc, polys, gens, "mccallum", equational=polys[0])
    compared += check(rc, polys, gens, rng.choice(["mccallum", "hong"]), extra=[random_polynomial(rng, gens)])
    assert compared >= 1


def test_edge_cases(rust: Any) -> None:
    from sympy_extras.core import Integer, Symbol, sqrt
    rc = load("sympy_extras_rs.core")
    x, y = Symbol('x'), Symbol('y')
    check(rc, [x**2 + y**2 - 1, x - y], [x, y], "mccallum")
    check(rc, [x**2 + y**2 - 1, x - y], [x, y], "mccallum", equational=x**2 + y**2 - 1)
    check(rc, [x - y**2], [x, y], "hong", extra=[y - 1])
    check(rc, [Integer(3)], [x, y], "mccallum")
    check(rc, [x**2 - 1], [x], "mccallum")
    check(rc, [sqrt(2)*x - y], [x, y], "mccallum")
    check(rc, [x*y - 1], [x, y], "nonsense")
    check(rc, [x*y - 1], [], "mccallum")

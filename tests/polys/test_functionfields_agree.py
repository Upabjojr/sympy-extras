"""``integral_basis`` and the bases of ``function_field`` (integral, local
at infinity, normal at infinity with its pole orders) of the Rust port
against ``sympy_extras.polys.functionfields``: the docstring curves, a
curve with a parameter, and random curves ``y**n = P(x)`` and with mixed
terms."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed


def curves() -> list[Any]:
    from sympy_extras.core import Symbol
    x, y = Symbol('x'), Symbol('y')
    return [y**3 - x**2*(x + 1), y**2 - x**3, y**4 + (x**2 + x)*y - x**2, x*y**2 - 1, y**2 - x**3 - 1,
            y**3 - x**7, (x - 1)*y**2 - x**3, y**2 - x**2*(x - 1)**3]


def check(rc: Any, F: Any) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.polys.domains import QQ
    from sympy_extras.polys import functionfields as ff
    x, y = Symbol('x'), Symbol('y')
    assert [repr(e) for e in rc.integral_basis(F, x, y)] == [repr(e) for e in ff.integral_basis(F, x, y)], F
    if ff.parameters_of(F, x, y):
        return
    E = ff.function_field(F, x, y, QQ)
    integral, local, normal, orders = rc.function_field_bases(F, x, y)
    assert [repr(e) for e in integral] == [repr(w.as_expr()) for w in E.integral_basis()], F
    assert [repr(e) for e in local] == [repr(w.as_expr()) for w in E.local_basis_at_infinity()], F
    assert [repr(e) for e in normal] == [repr(w.as_expr()) for w in E.normal_basis()], F
    assert orders == E.pole_orders_at_infinity(E.normal_basis()), F


def test_known_curves(rust: Any) -> None:
    from sympy_extras.core import Symbol
    rc = load("sympy_extras_rs.core")
    x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
    for F in curves() + [y**2 - (x - a)**2*(x - 1)]:
        check(rc, F)


@pytest.mark.parametrize("seed", range(20))
def test_random_curves(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    rc = load("sympy_extras_rs.core")
    x, y = Symbol('x'), Symbol('y')
    rng = random.Random(stable_seed("function fields", seed))
    n = rng.randint(2, 3)
    roots = [Integer(rng.randint(-2, 2)) for _ in range(rng.randint(1, 3))]
    P = mul(*[pow(add(x, mul(Integer(-1), r)), Integer(rng.randint(1, 4))) for r in roots])
    F = add(pow(y, Integer(n)), mul(Integer(-1), P))
    if rng.random() < 0.4:
        F = add(F, mul(Integer(rng.randint(1, 2)), x, y))
    check(rc, F)

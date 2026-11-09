"""The native first solution of ``Order``'s change of variables against
SymPy's ``solveset`` (its first element): equal values at sample points,
and the same ``Order.subs`` results."""

from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import Symbol
from sympy_extras.core.order import _solve_rational
from sympy_extras.interop import expr_from_sympy, solveset_first_through_sympy, to_sympy

sx, sd, sa = sympy.symbols('x d a')
x, d = Symbol('x'), Symbol('d')

EQUATIONS = [sx - sd**2, sx - 1/sd, sx - sd**2 - sd, sx - 1/(sd + 1), sx - (sd + 1)/(sd - 1), sx - sd**3,
             sx - 2*sd**2 - 1, sx - sd**4, sx - 1/sd**2, sx - sd/(sd**2 + 1), sx - sa*sd**2, sx - sd**2 + 3*sd,
             sx - (sd**2 + 1)/sd, sx - sd**6, sx - 1/(sd**2 - 4), 2 - sd**2, sx**2 - sd**2 - 1, sx - sd**3 - sd]


@pytest.mark.parametrize('e', EQUATIONS, ids=str)
def test_first_solution(e: sympy.Expr) -> None:
    native = _solve_rational(expr_from_sympy(e), d)
    reference = solveset_first_through_sympy(expr_from_sympy(e), d)
    if native is None:
        return   # falls back to the bridge
    assert reference is not None, e
    n, r = to_sympy(native), to_sympy(reference)
    assert isinstance(n, sympy.Expr) and isinstance(r, sympy.Expr)
    difference = n - r
    for point in [sympy.Rational(7, 3), sympy.Rational(-5, 2), sympy.Rational(1, 9) + sympy.I]:
        value = complex(difference.subs({sx: point, sa: sympy.Rational(3, 5)}).evalf(30))
        assert abs(value) < 1e-20, (e, native, reference, point)


def test_order_subs() -> None:
    from sympy_extras.core.order import order
    from sympy_extras.core import Integer, oo
    y = Symbol('y')
    for point, new in [(Integer(0), y**2), (oo, y**2), (Integer(0), 1/y), (oo, 1/(y + 1)), (Integer(0), y**2 + y),
                       (oo, y**3), (oo, y**2 - 3*y), (Integer(0), 2*y**2 + 1)]:
        got = order(x, (x, point)).subs(x, new)
        want = sympy.Order(sx, (sx, to_sympy(point))).subs(sx, to_sympy(new))
        assert to_sympy(got) == want, (point, new, got, want)


def _sympy_solutions(e: sympy.Expr) -> list[sympy.Basic]:
    solutions = sympy.solveset(e, sd)
    if isinstance(solutions, sympy.Complement):
        kept, removed = solutions.args
        return [s for s in kept.args if s not in removed.args]
    if solutions is sympy.S.EmptySet:
        return []
    assert isinstance(solutions, sympy.FiniteSet), solutions
    return list(solutions.args)


def test_random_rational_equations() -> None:
    from sympy_extras.polys.rationalsolve import solveset_rational
    rng = random.Random(20261109)
    native = 0
    for _ in range(120):
        num = sum(rng.randint(-3, 3)*rng.choice([1, sx, sa])*sd**i for i in range(rng.randint(1, 4)))
        den = sum(rng.randint(-2, 2)*sd**i for i in range(rng.randint(1, 3)))
        if den == 0 or not (num/den).has(sd):
            continue
        e = rng.choice([sx, sympy.Integer(2), sx**2 + 1]) - num/den
        got = solveset_rational(expr_from_sympy(e), d)
        if got is None:
            continue
        native += 1
        assert [to_sympy(s) for s in got[0]] == _sympy_solutions(e), e
    assert native >= 60

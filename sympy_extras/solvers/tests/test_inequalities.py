from __future__ import annotations

from sympy_extras.core import Eq, Ne, Rational, Symbol, sqrt
from sympy_extras.core.logic import Boolean, false, true
from sympy_extras.solvers.inequalities import solve_inequality

x = Symbol('x')


def test_linear() -> None:
    assert solve_inequality(2*x - 4 > 0, x) == (x > 2)
    assert solve_inequality(3 - x <= 1, x) == (x >= 2)
    p = Symbol('p', positive=True)
    assert solve_inequality(p*x < 1, x) == (x < 1/p)


def test_denominators_of_a_linear_relational() -> None:
    # x < x*(2/x - 1) came back x < 1: the point x = 0, where the
    # relational is not defined, was a solution (SymPy: (x < 1) & Ne(x, 0))
    solved = solve_inequality(x < x*(2/x - 1), x)
    assert solved == ((x < 1) & Ne(x, 0))
    assert solved.subs(x, 0) is false


def _agree(ie: Boolean, solved: Boolean) -> bool:
    """The solved form has the truth values of ``ie`` at sample points
    (where ``ie`` is defined)."""
    points = [Rational(n, 4) for n in range(-24, 25)] + [sqrt(2), -sqrt(2), sqrt(3)]
    for p in points:
        try:
            expected = ie.subs(x, p)
        except (ZeroDivisionError, TypeError):
            continue
        if expected is not true and expected is not false:
            continue
        if solved.subs(x, p) is not expected:
            return False
    return True


def test_rational_by_the_cad() -> None:
    # SymPy's reduce_rational_inequalities: the same set (checked at
    # sample points), the restrictions at the infinities dropped
    for ie in (x**2 - 4 < 0, x**2 > 2, x**2 + 4 < 0, Eq(x**2, 4), Ne(x**2, 4), 1/(x - 1) <= 2, 1/x > 0,
               x**2 - 2*x + 1 > 0, x**3 - x > 0, (x**2 - 1)/(x - 3) >= 0, x**4 - 5*x**2 + 4 < 0):
        solved = solve_inequality(ie, x)
        assert solved.free_symbols <= {x}, (ie, solved)
        assert _agree(ie, solved), (ie, solved)
    assert solve_inequality(x**2 + 4 < 0, x) is false
    assert solve_inequality(Eq(x**2, 4), x) == (Eq(x, -2) | Eq(x, 2))
    assert str(solve_inequality(x**2 > 2, x)) == '(sqrt(2) < x) | (x < -sqrt(2))'
    assert str(solve_inequality(x**2 - 2*x + 1 > 0, x)) == '(x < oo) & Ne(x, 1)'

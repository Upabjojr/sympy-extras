"""Charpit's method on equations of the core (the other tests, in
``test_charpit.py``, go through the SymPy face)."""
from __future__ import annotations

from typing import Callable

from sympy_extras.core import Eq, Function, Symbol, diff, expand
from sympy_extras.core.expr import Expr
from sympy_extras.functions import log
from sympy_extras.simplify.simplify import simplify

from sympy_extras.solvers.charpit import (_polynomial_solutions, _power_antiderivative, check_complete_integral,
    complete_integral)

x, y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')
u = Function('u')(x, y)
p, q = u.diff(x), u.diff(y)


def raises(exception: type[BaseException], f: Callable[[], object]) -> None:
    try:
        f()
    except exception:
        return
    raise AssertionError('%s not raised' % exception.__name__)


def _residual(F: Callable[[Expr, Expr, Expr], Expr], solution: Expr) -> Expr:
    """``F(u, u_x, u_y)`` at the solution, computed independently of the
    module (no symbols for the derivatives)."""
    return simplify(F(solution, diff(solution, x), diff(solution, y)))


def test_standard_forms_native() -> None:
    cases: list[tuple[Expr, Callable[[Expr, Expr, Expr], Expr]]] = [
        (p*q - 1, lambda U, P, Q: P*Q - 1),
        (p**2 + q**2 - 1, lambda U, P, Q: P**2 + Q**2 - 1),
        (p**2 - q, lambda U, P, Q: P**2 - Q),
        (p*x - q*y, lambda U, P, Q: P*x - Q*y),
        (p*q - x, lambda U, P, Q: P*Q - x),
        (u - p*x - q*y - p*q, lambda U, P, Q: U - P*x - Q*y - P*Q),
    ]
    for equation, F in cases:
        solution = complete_integral(equation, u)
        assert solution is not None, equation
        assert isinstance(solution, Eq) and solution.lhs == u
        value = solution.rhs
        assert isinstance(value, Expr) and value.has(a) and value.has(b)
        assert _residual(F, value) == 0, (equation, value)
        assert check_complete_integral(equation, u, solution)


def test_known_complete_integrals() -> None:
    # Clairaut: the planes u = a x + b y + a b
    assert str(complete_integral(Eq(u, p*x + q*y + p*q), u)) == 'Eq(u(x, y), a*b + a*x + b*y)'
    # p q = 1: u = a x + y/a + b
    assert str(complete_integral(p*q - 1, u)) == 'Eq(u(x, y), a*x + b + y/a)'
    c, d = Symbol('c'), Symbol('d')
    assert str(complete_integral(p*q - 1, u, (c, d))) == 'Eq(u(x, y), c*x + d + y/c)'


def test_checks_and_errors() -> None:
    assert check_complete_integral(p*q - 1, u, a*x + y/a + b)
    assert not check_complete_integral(p*q - 1, u, Eq(u, a*x + y + b))
    assert complete_integral(p*q - u*x*y, u) is None
    raises(ValueError, lambda: complete_integral(u.diff(x, 2) - q, u))
    raises(ValueError, lambda: complete_integral(p - 1, Function('v')(x)))


def test_polynomial_solutions() -> None:
    t = Symbol('t')
    # simplified, in the order of SymPy's solve (default_sort_key)
    assert str(_polynomial_solutions(t**2 + a**2 - 1, t)) == '[-sqrt(1 - a**2), sqrt(1 - a**2)]'
    roots = _polynomial_solutions(t**3 - 2, t)
    assert roots is not None and len(roots) == 3 and all(expand(r**3) == 2 for r in roots)
    # a root which cancels the denominator is not a solution
    assert _polynomial_solutions((t**2 - 1)/(t - 1), t) == [-1]
    # not polynomial in the unknown: SymPy's solve is called instead
    assert _polynomial_solutions(log(t) - 1, t) is None
    assert _polynomial_solutions(a - 1, t) is None


def test_constant_factor_stays_out() -> None:
    # (a + 1)/u was integrated as a*log(u) + log(u), for which SymPy's
    # solve finds no root (solve(a*log(u) + log(u) - 1, u) == []): the
    # complete integral exp((a*y + b + x)/(a + 1)) of p + q = u was lost
    v = Symbol('v')
    assert _power_antiderivative((a + 1)/v, v) == (a + 1)*log(v)
    assert _power_antiderivative(3*v**2 + a, v) == expand(v**3 + a*v)
    assert _power_antiderivative(v**Symbol('n'), v) is None

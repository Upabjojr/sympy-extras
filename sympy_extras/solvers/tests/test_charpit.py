from __future__ import annotations

from sympy import Function, symbols, Eq, sqrt, S
from sympy.testing.pytest import raises

from sympy_extras.solvers.charpit import complete_integral, check_complete_integral

x, y, a, b = symbols('x y a b')
u = Function('u')(x, y)
p, q = u.diff(x), u.diff(y)


def _check(equation: object, f: object = u) -> None:
    from sympy_extras._typing import as_expr
    from sympy.core.function import AppliedUndef
    assert isinstance(f, AppliedUndef)
    solution = complete_integral(as_expr(equation) if not isinstance(equation, Eq) else equation, f)
    assert solution is not None, equation
    assert check_complete_integral(as_expr(equation) if not isinstance(equation, Eq) else equation, f, solution)
    assert solution.has(a) and solution.has(b)


def test_standard_forms() -> None:
    _check(p*q - 1)                              # F(p, q) = 0
    _check(p**2 + q**2 - 1)
    _check(Eq(u, p*x + q*y + p*q))               # Clairaut
    _check(Eq(u, p*x + q*y + sqrt(1 + p**2 + q**2)))
    _check(p**2 + q**2 - u)                      # F(u, p, q) = 0
    _check(u*p*q - 1)
    _check(p**2 + x - q - y**2)                  # separable
    _check(p*x - q*y)
    _check(p**2 - x*q)                           # p = a is a first integral
    _check(p*y - q**2)                           # q = a is a first integral
    _check(q - p**2 - x)


def test_unsolved() -> None:
    assert complete_integral(p*q - u*x*y, u) is None
    raises(ValueError, lambda: complete_integral(u.diff(x, 2) - q, u))
    raises(ValueError, lambda: complete_integral(p - 1, Function('v')(x)))
    assert check_complete_integral(p*q - 1, u, Eq(u, a*x + y/a + b))
    assert not check_complete_integral(p*q - 1, u, Eq(u, a*x + y + b))
    assert complete_integral(p*q - 1, u, (symbols('c'), symbols('d'))) is not None
    assert S.true

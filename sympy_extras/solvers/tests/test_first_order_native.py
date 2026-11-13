"""The first order solvers and the Abel equations on the objects of the
core (:mod:`sympy_extras.solvers.first_order`,
:mod:`sympy_extras.solvers.abel`); the SymPy faces are tested in
``test_first_order.py`` and ``test_abel.py``. The solutions are checked
by substitution into the equations: explicit ones directly, implicit ones
``F(x, y) = G(x) + C1`` by implicit differentiation along the equation,
parametric ones on the curve and on its slope."""
from __future__ import annotations

from typing import Optional

from sympy_extras.core import Eq, Function, Integer, Rational, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.functions import exp
from sympy_extras.core.evalf import evalf, to_complex
from sympy_extras.simplify.simplify import simplify
from sympy_extras.solvers.abel import (REPRESENTATIVES, Coefficients, _defines_solution, _transformed,
    abel_by_invariants, abel_coefficients, abel_equivalence, abel_invariants, air_solution, constant_invariant,
    particular_solution)
from sympy_extras.solvers.first_order import (_derivative_polynomial, _explicit_rhs, _replaced, abel_ode, chini_ode,
    dsolve_first_order, lagrange_ode, riccati_ode)

x, t = Symbol('x'), Symbol('t')
a, b = Symbol('a'), Symbol('b')
C1, p = Symbol('C1'), Symbol('p')
y = Function('y')(x)
Y = Symbol('Y')


def _value(e: Expr, values: dict[Symbol, Expr]) -> Optional[complex]:
    value = evalf(e, 20, subs=values)
    if value.free_symbols:
        return None
    return to_complex(value)


def _slope(equation: Expr) -> Expr:
    """``y'`` from an equation linear in it."""
    F, _, u, q = _derivative_polynomial(equation, y)
    rhs = _explicit_rhs(F, q)
    assert rhs is not None
    return _replaced(rhs, {u: Y})


_POINTS = ((Rational(3, 5), Rational(2, 5)), (Rational(11, 10), Rational(9, 10)), (Rational(17, 10), Rational(3, 10)))


def _implicit_holds(equation: Expr, solution: Optional[Basic], points: int = 3) -> bool:
    """``d/dx (F - G) = 0`` along the equation for ``Eq(F, G + C1)`` (``C1``
    in ``G`` only additively), at the first ``points`` points ``(x, y)``."""
    assert isinstance(solution, Eq)
    relation = _replaced(solution.lhs - solution.rhs, {y: Y})
    slope = _slope(equation)
    total = relation.diff(x) + relation.diff(Y)*slope
    checked = 0
    for x0, y0 in _POINTS[:points]:
        value = _value(total, {x: x0, Y: y0, C1: Integer(0)})
        if value is None:
            continue
        if abs(value) > 1e-10:
            return False
        checked += 1
    return checked > 0


def _explicit_holds(equation: Expr, solution: Optional[Basic]) -> bool:
    assert isinstance(solution, Eq) and solution.lhs == y
    residual = equation.subs(y, solution.rhs).doit()
    assert isinstance(residual, Expr)
    values: dict[Symbol, Expr] = {s: Rational(3, 10) + Rational(k, 7)
                                  for k, s in enumerate(sorted(residual.free_symbols - {x, C1}, key=str))}
    values[C1] = Rational(1, 3)
    for x0 in (Rational(7, 10), Rational(13, 10)):
        values[x] = x0
        value = _value(residual, values)
        if value is None or abs(value) > 1e-10:
            return False
    return True


def _equation(coefficients: Coefficients) -> Expr:
    f3, f2, f1, f0 = coefficients
    return y.diff(x) - (f3*y**3 + f2*y**2 + f1*y + f0)


def test_derivative_polynomial() -> None:
    F, x_, u, q = _derivative_polynomial(Eq(y.diff(x), x*y**2), y)
    assert x_ == x and F == q - x*u**2
    assert _explicit_rhs(F, q) == x*u**2
    assert _explicit_rhs(q**2 - u, q) is None
    # the second derivative became zero when y was replaced by a symbol, so
    # y'' - y**3 was read as the first order equation -y**3 = 0
    try:
        _derivative_polynomial(y.diff(x, 2) - y**3, y)
    except ValueError:
        pass
    else:
        raise AssertionError("a second order equation was accepted")


def test_chini() -> None:
    equation = y.diff(x) - x**2*y**3 + x**2
    assert _implicit_holds(equation, chini_ode(equation, y))
    equation = y.diff(x) - y**3 - x**Rational(-3, 2)
    assert _implicit_holds(equation, chini_ode(equation, y))
    # Chini's invariant is not constant
    assert chini_ode(y.diff(x) - y**3 - x, y) is None


def test_abel_first_and_second_kind() -> None:
    equation = y.diff(x) - y**3 - 3*y**2 - 3*y
    assert _implicit_holds(equation, abel_ode(equation, y))
    # second kind: y y' = 3 y**2 + 3 y + 1, with w = 1/y
    equation = y*y.diff(x) - 3*y**2 - 3*y - 1
    assert _implicit_holds(equation, abel_ode(equation, y))
    assert abel_ode(y.diff(x) - y**3 - x, y) is None
    # the relation returned for y y' = y + x was 0 = C1 - 11**(2/3)*log(x)/9,
    # which has no y in it and solves nothing
    found = abel_ode(y*y.diff(x) - y - x, y)
    assert found is None or _implicit_holds(y*y.diff(x) - y - x, found)


def test_riccati() -> None:
    equation = y.diff(x) + y**2 - 2/x**2
    solution = riccati_ode(equation, y)
    assert _explicit_holds(equation, solution)
    assert isinstance(solution, Eq) and simplify(solution.rhs - (2*C1*x**3 - 1)/(C1*x**4 + x)) == 0
    # Airy functions: no rational particular solution
    equation = y.diff(x) + y**2 + a*x + b
    assert _explicit_holds(equation, riccati_ode(equation, y, special_values=False))
    assert riccati_ode(y.diff(x) - y**3, y) is None


def test_lagrange() -> None:
    equation = Eq(y, 2*x*y.diff(x) + y.diff(x)**2)
    solution = lagrange_ode(equation, y)
    assert solution is not None and len(solution) == 2
    first, second = solution
    assert isinstance(first, Eq) and isinstance(second, Eq)
    X, Yp = first.rhs, second.rhs
    # y = 2 x p + p**2 along the curve, and dy/dx = p
    assert simplify(Yp - 2*X*p - p**2) == 0
    assert simplify(Yp.diff(p) - p*X.diff(p)) == 0
    # Clairaut's equation is SymPy's
    assert lagrange_ode(Eq(y, x*y.diff(x) + y.diff(x)**2), y) is None
    assert isinstance(dsolve_first_order(equation, y), Tuple)


def test_dsolve_first_order() -> None:
    # Chini's equation with n = 1/2 (constant invariant)
    equation = y.diff(x) - y**Rational(1, 2) - x
    assert _implicit_holds(equation, dsolve_first_order(equation, y))
    equation = y.diff(x) + y**2 - 2/x**2
    assert _explicit_holds(equation, dsolve_first_order(equation, y))


def test_coefficients_and_invariants() -> None:
    assert abel_coefficients(y.diff(x) - x*y**3 - y**2 + 1, y) == ((x, Integer(1), Integer(0), Integer(-1)), x)
    coefficients: Coefficients = (x, Integer(1), -x, x**2 + 1)
    invariants = abel_invariants(coefficients, x)
    assert simplify(invariants.I1 - invariants.s5**3/invariants.s3**5) == 0
    # invariant under y = P u + Q, x = xi(t)
    moved = abel_invariants(_transformed(coefficients, x, t**2 + 1, t, t**3, t), t)
    assert simplify(moved.I1 - invariants.I1.subs(x, t**2 + 1)) == 0
    assert simplify(moved.I2 - invariants.I2.subs(x, t**2 + 1)) == 0
    assert abel_invariants((1, 0, 0, x**Rational(-3, 2)), x).I1 == Rational(-27, 8)
    assert constant_invariant((1, 0, 0, x**Rational(-3, 2)), x) is True
    assert constant_invariant((-1, -2*x, 0, 0), x) is False


def test_equivalence_and_particular_solutions() -> None:
    airy: Coefficients = (Integer(-1), -2*t, Integer(0), Integer(0))
    found = abel_equivalence((-1, -2*(x - 1), 0, 0), x, airy, t)
    assert found is not None and found.xi == x - 1 and found.P == 1 and found.Q == 0
    target = _transformed(airy, t, x**2, x, Integer(1), x)
    found = abel_equivalence(target, x, airy, t)
    assert found is not None and found.xi == x**2 and found.P == 1/x and found.Q == -1/x
    assert abel_equivalence((1, 0, 0, x**Rational(-3, 2)), x, airy, t) is None
    assert particular_solution((-1, -2*x, 0, 0), x) == 0
    assert particular_solution((1, 0, -x**2, 1), x) == x
    assert particular_solution((1, 0, 0, exp(x)), x) is None


def test_air_class_and_representatives() -> None:
    for representative, _ in REPRESENTATIVES:
        f3, f2, f1, f0 = (c.subs(t, x) for c in representative)
        assert _implicit_holds(_equation((f3, f2, f1, f0)), air_solution((f3, f2, f1, f0), x, y))
    assert air_solution((1, 0, -x**2, 1), x, y) is None
    equation = _equation((Integer(-1), 2 - 2*x, Integer(0), Integer(0)))
    # (the pulled-back Airy solutions are large: one point each)
    assert _implicit_holds(equation, abel_by_invariants(equation, y), 1)
    assert _implicit_holds(equation, abel_ode(equation, y), 1)
    assert abel_by_invariants(y.diff(x) - y**3 - exp(x), y) is None


def test_a_relation_implying_zero_derivative_is_rejected() -> None:
    # sympy-extras#38: a relation without x implies y' = 0, which is no
    # solution of y' = x y**3 + y**2; the numerical check rejects it and
    # keeps a true one
    coefficients: Coefficients = (x, Integer(1), Integer(0), Integer(0))
    assert not _defines_solution(y - C1, coefficients, x, y, C1)
    # y' = y**3 has the solutions 1/y**2 + 2 x = C1
    assert _defines_solution(1/y**2 + 2*x - C1, (Integer(1), Integer(0), Integer(0), Integer(0)), x, y, C1)
    # a relation without y defines no solution at all
    assert not _defines_solution(C1 - x, (Integer(1), Integer(0), Integer(0), Integer(0)), x, y, C1)

"""Tests of the native heuristic Risch integrator
(:mod:`sympy_extras.integrals.heurisch`) on the objects of the core: every
antiderivative checked by differentiation (exactly, or at sample points
by mpmath, an independent evaluation), the steps of the port that replace
SymPy's domains (the placeholders of the constants, the number field of
the algebraic ones, the linear systems over the fractions of the
parameters, the tangents of the half angle, the solutions of the
denominators), and the agreement with the SymPy face; the integrators
test the face."""
from __future__ import annotations

import sympy

from sympy_extras._typing_sympy import as_expr as as_sympy_expr
from sympy_extras.core import Float, I, Integer, Rational, S, Symbol, pi, sqrt
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import diff
from sympy_extras.functions import Piecewise, abs_, atan, besselj, cos, cosh, erf, exp, log, sin, sinh, tan, tanh
from sympy_extras.integrals import heurisch_sympy
from sympy_extras.integrals.heurisch import (_half_angle_tangents, _NumberField, _Placeholders, _power_coefficients,
                                             _real_forms, _root_factors, _solutions, _solve_coefficients, _algebraic,
                                             components, heurisch_antiderivative, heurisch_cases)
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.polys.polytools import cancel

x, y, n, a = Symbol('x'), Symbol('y'), Symbol('n'), Symbol('a')
p = Symbol('p', positive=True)


def _value(e: Expr, point: Expr) -> complex:
    found = numeric_value(e.subs(x, point), 200)
    assert found is not None, e
    return complex(found)


def _is_antiderivative(F: Expr, f: Expr) -> bool:
    """``F' = f``, exactly or at three sample points (the parameters set to
    rational values)."""
    difference = diff(F, x) - f
    if cancel(difference) == S.Zero:
        return True
    values = {y: Rational(3, 7), n: Rational(5, 3), a: Rational(2, 5), p: Rational(7, 4)}
    difference = difference.subs(values)
    return all(abs(_value(difference, point)) < 1e-9 for point in (Rational(1, 3), Rational(6, 5), Rational(5, 2)))


def test_elementary_and_special_antiderivatives() -> None:
    for f in (y * x * cos(x**2), exp(-x**2) * x, 1 / (x**2 + 1), exp(x) / x, y * tan(x), log(x)**2 / x**3,
              exp(x**2) * exp(x), x * exp(x), sin(x) * cos(x), 1 / (x**4 + 1), 1 / (sin(x) + cos(x)),
              exp(-x**2) * erf(x), x * besselj(0, x), sinh(x) * cosh(x), 1 / (x**3 + 1), pi * x / (x**2 + pi),
              sqrt(2) * x / (x**2 + 2), exp(I * x), 1 / (x + I), exp(p * x) * cos(x)):
        F = heurisch_antiderivative(f, x)
        assert F is not None, f
        assert _is_antiderivative(F, f), (f, F)


def test_known_values() -> None:
    # (hand computations)
    assert heurisch_antiderivative(1 / (x**2 + 1), x) == atan(x)
    assert heurisch_antiderivative(exp(-x**2) * erf(x), x) == sqrt(pi) * erf(x)**2 / 4
    assert heurisch_antiderivative(y, x) == x * y
    # refused: a float, a jump
    assert heurisch_antiderivative(x + Float(0.5), x) is None
    assert heurisch_antiderivative(abs_(x), x) is None


def test_cases_over_the_parameters() -> None:
    found = heurisch_cases(cos(n * x), x)
    assert isinstance(found, Piecewise)
    assert found.args[0].args[0] == sin(n * x) / n and found.args[1].args[0] == x
    # no case where the integrand itself is undefined (1/(x + a) at no a)
    found = heurisch_cases(1 / (x + a), x)
    assert found is not None and not isinstance(found, Piecewise)
    assert heurisch_cases(y, x) == x * y


def test_solutions_of_the_denominators() -> None:
    # SymPy's solve([d], dict=True, exclude=(x,)) for one equation
    assert _solutions(n, x) == [{n: S.Zero}]
    assert _solutions(n**2 - 1, x) == [{n: Integer(-1)}, {n: Integer(1)}]
    assert _solutions(a - n, x) == [{a: n}]
    # an assumption of the symbol refutes the solution
    assert _solutions(p, x) == []


def test_placeholders() -> None:
    placeholders = _Placeholders([x])
    abstracted = placeholders.abstract(pi * x**2 + sqrt(2) * x / y + exp(y))
    assert not abstracted.has(pi) and abstracted.has(x)
    assert placeholders.restore(abstracted) == pi * x**2 + sqrt(2) * x / y + exp(y)
    # the negative powers of the parameters are kept for the linear systems only
    assert _Placeholders([x], rational=True).abstract(x / y) == x / y
    assert _Placeholders([x]).abstract(x / y) != x / y


def test_number_field_of_the_constants() -> None:
    atoms = [sqrt(2), I, Integer(2)**Rational(1, 3)]
    values = [_algebraic(atom) for atom in atoms]
    assert all(v is not None for v in values)
    field = _NumberField(atoms, [v for v in values if v is not None])
    assert field.field.degree == 12
    # each constant is the polynomial in theta of its form
    for atom in atoms:
        form = field.forms[atom]
        assert isinstance(form, Expr)
        value = numeric_value(form.subs(field.theta, field.theta_expr), 200)
        assert value is not None and abs(complex(value) - complex(atom.evalf(30))) < 1e-10
    assert _algebraic(pi) is None and _algebraic(sqrt(pi)) is None


def test_linear_systems() -> None:
    c0, c1, v = Symbol('c0'), Symbol('c1'), Symbol('v')
    # (c0 - 1)*v + c1*sqrt(2) - 2: c0 = 1, c1 = sqrt(2)
    solution = _solve_coefficients((c0 - 1) * v + c1 * sqrt(2) - 2, [v], [c0, c1], {sqrt(2), Integer(2)})
    assert solution == {c0: Integer(1), c1: sqrt(2)}
    # over QQ(y, pi): c0*y*pi - 1
    solution = _solve_coefficients(c0 * y * pi * v - v, [v], [c0], {y, pi})
    assert solution == {c0: 1 / (pi * y)}
    # inconsistent
    assert _solve_coefficients(c0 * v + 1, [v], [c0], set()) is None
    # free unknowns are zero: c0 + c1 = 1 gives c0 = 1
    assert _solve_coefficients(c0 + c1 - 1, [v], [c0, c1], set()) == {c0: Integer(1)}


def test_half_angle_tangents() -> None:
    assert _half_angle_tangents(sin(x)) == 2 * tan(x / 2) / (tan(x / 2)**2 + 1)
    assert _half_angle_tangents(cos(x)) == (1 - tan(x / 2)**2) / (tan(x / 2)**2 + 1)
    assert _half_angle_tangents(cosh(x)) == (tanh(x / 2)**2 + 1) / (1 - tanh(x / 2)**2)
    assert _half_angle_tangents(x**2) == x**2


def test_polynomial_steps() -> None:
    assert _power_coefficients((x + y)**2, x) == {2: Integer(1), 1: 2 * y, 0: y**2}
    assert _power_coefficients(exp(x) + x, x) is None
    assert _root_factors(x**2 - 2, x, None) == [x - sqrt(2), x + sqrt(2)]
    assert _root_factors(x**2 - 2, x, 'Q') == [x**2 - 2]


def test_real_forms_and_components() -> None:
    assert _real_forms(-I * log(x - I) / 2 + I * log(x + I) / 2) == atan(x)
    assert components(sin(x) * cos(x)**2, x) == {sin(x), cos(x), x}
    assert components(sqrt(x + 1)**3, x) == {sqrt(x + 1), x}


def test_the_face_agrees() -> None:
    sx = sympy.Symbol('x')
    for f in (y * x * cos(x**2), 1 / (x**4 + 1), exp(-x**2) * erf(x), 1 / (sin(x) + cos(x))):
        native = heurisch_antiderivative(f, x)
        face = heurisch_sympy.heurisch_antiderivative(as_sympy_expr(to_sympy(f)), sx)
        assert native is not None and face is not None
        assert from_sympy(face) == native

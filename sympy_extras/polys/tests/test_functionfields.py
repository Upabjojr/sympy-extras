"""Tests of the function fields: arithmetic, Trager's integral bases
against the literature and against an independent maximality test by
Puiseux expansions, the local bases at infinity and the normalization
against the valuations of the branches at infinity."""
from __future__ import annotations

from fractions import Fraction
from math import ceil

from sympy import symbols, Poly, CRootOf, Rational, Expr, S, cancel
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.matrices import DomainMatrix

from sympy_extras._typing import as_expr
from sympy_extras.polys.functionfields import FunctionField, Element, integral_basis
from sympy_extras.polys.puiseux import puiseux_expansions

x, y, z, a, b = symbols('x y z a b')


def _is_maximal_at(E: FunctionField, basis: list[Element], point: Expr) -> bool:
    """Whether no element ``sum(c_i w_i)/(x - point)`` with constant
    ``c_i`` in the field of the point is integral: the order generated
    by ``basis`` is then maximal at the point (van Hoeij's criterion),
    checked through the Puiseux expansions above it."""
    F = E.F
    Kx = E.Kx
    Y = symbols('Y')
    curve = as_expr(sum(Kx.to_sympy(c) * Y**(E.n - k) for k, c in enumerate(F)))
    branches = puiseux_expansions(curve, x, Y, point, order=3)
    d = 1 if isinstance(point, Rational) else _degree(QQ.algebraic_field(point))
    # unknowns: the rational coordinates of c_i in the powers of the point
    rows: list[list[object]] = []
    for branch in branches:
        L = branch.field
        D = _degree(L)
        alpha = L.from_sympy(point) if d > 1 else L.one
        series = []
        for w in basis:
            numerator = Poly(as_expr(sum(Kx.to_sympy(c) * Y**i for i, c in enumerate(w.coefficients))).as_numer_denom()[0], Y, x)
            denominator = Poly(as_expr(sum(Kx.to_sympy(c) * Y**i for i, c in enumerate(w.coefficients))).as_numer_denom()[1], x)
            s_num = branch.evaluate(numerator)
            s_den = branch.evaluate(denominator)
            series.append(s_num * s_den.inverse(8))
        e = branch.ramification
        for j in range(-8, e):
            # the coefficient of t^j of sum_i c_i w_i must vanish for j < e
            for coordinate in range(D):
                row: list[object] = []
                for s_i in series:
                    power = L.one
                    for k in range(d):
                        value = (s_i.scaled(power)).coefficient(j)
                        coefficients = list(reversed(value.to_list())) if L != QQ else [value]
                        coefficients += [QQ.zero] * (D - len(coefficients))
                        row.append(coefficients[coordinate])
                        power = power * alpha
                rows.append(row)
    matrix = DomainMatrix([[QQ.convert(v) for v in row] for row in rows], (len(rows), E.n * d), QQ)
    return matrix.nullspace().shape[0] == 0


def _degree(K: object) -> int:
    """The degree of a number field over the rationals."""
    return int(K.mod.degree()) if isinstance(K, AlgebraicField) else 1


def _check_integral_basis(F: Expr) -> FunctionField:
    E = FunctionField(F, x, y)
    basis = E.integral_basis()
    assert len(basis) == E.n
    for w in basis:
        assert E.is_integral(w), (F, w)
    # the discriminant of the curve is the discriminant of the basis times
    # the square of the index
    power = [Element(E, [E.Kx.zero] * k + [E.Kx.one]) for k in range(E.n)]
    quotient = E.discriminant(power) / E.discriminant(basis)
    assert quotient.denom == 1 or quotient.denom.degree() == 0, quotient
    root, remainder = Poly(E.Kx.to_sympy(quotient), x).sqf_list(), None
    del remainder
    for factor, multiplicity in root[1]:
        assert multiplicity % 2 == 0, (F, quotient)
    # maximal at every root of the discriminant of the basis with multiplicity
    disc = Poly(E.Kx.to_sympy(E.discriminant(basis)), x)
    for factor, multiplicity in disc.sqf_list()[1]:
        if multiplicity < 2:
            continue
        for irreducible, _ in factor.factor_list()[1]:
            point = as_expr(CRootOf(irreducible, 0)) if irreducible.degree() > 1 else as_expr(-irreducible.all_coeffs()[1] / irreducible.all_coeffs()[0])
            assert _is_maximal_at(E, basis, point), (F, irreducible)
    return E


def _same(found: list[Expr], expected: list[Expr]) -> bool:
    return len(found) == len(expected) and all(cancel(u - v) == 0 for u, v in zip(found, expected))


def test_bases_of_the_literature() -> None:
    # Kauers 2008, example 6
    assert integral_basis(y**3 - x**2 * (x + 1), x, y) == [1, y, y**2 / x]
    # Bronstein's tutorial, examples 1 and 2
    assert integral_basis(y**4 + (x**2 + x) * y - x**2, x, y) == [1, y, y**2, y**3 / x]
    # simple radical extensions, formula (11) of the tutorial: y^n = H with
    # H = H_1 H_2^2 ... gives y^(i-1)/prod H_j^floor((i-1) j/n)
    assert integral_basis(y**2 - x**3, x, y) == [1, y / x]
    assert _same(integral_basis(y**4 - x**2 * (x + 1)**3, x, y),
                 [S.One, y, y**2 / (x * (x + 1)), y**3 / (x * (x + 1)**2)])
    assert integral_basis(y**2 - x**8 - 1, x, y) == [1, y]
    assert _same(integral_basis(y**2 - (x**2 + 1)**2 * (x - 3), x, y), [S.One, y / (x**2 + 1)])


def test_bases_are_maximal_by_puiseux_expansions() -> None:
    for F in [y**3 - x**2 * (x + 1), y**4 + (x**2 + x) * y - x**2, y**2 - x**3, (y**2 - x)**2 - x**3,
              y**4 - x**2 * (x + 1)**3, y**3 - 3 * y - x**2, y**4 - 4 * x * y**2 + 4, y**6 - x**2 * (x + 1)**3,
              y**2 - (x**2 + 1)**2 * (x - 3)]:
        _check_integral_basis(F)


def test_curves_which_are_not_monic() -> None:
    assert integral_basis(x * y**2 - 1, x, y) == [1, x * y]
    E = _check_integral_basis((x + 1) * y**2 - x**3)
    assert _same([w.as_expr() for w in E.integral_basis()], [S.One, (x + 1) * y / x])


def test_parameters_in_the_coefficients() -> None:
    E = FunctionField(y**2 - x**3 - a * x - b, x, y)
    assert [w.as_expr() for w in E.integral_basis()] == [1, y]
    E = FunctionField(y**2 - (x - a)**2 * (x - b), x, y)
    assert _same([w.as_expr() for w in E.integral_basis()], [S.One, y / (x - a)])


def test_arithmetic() -> None:
    E = FunctionField(y**3 - x**2 * (x + 1), x, y)
    u = E.element(y**2 / x)
    assert (u * E.element(y)).as_expr() == x**2 + x
    assert (u * u.inverse()).as_expr() == 1
    assert E.element(y).derivative() == E.element((3 * x**2 + 2 * x) / (3 * y**2))
    assert E.element(y).trace() == 0 and E.element(y**3).trace() == E.Kx.from_sympy(3 * x**2 * (x + 1))
    assert E.is_integral(E.element(y)) and E.is_integral(u) and not E.is_integral(E.element(y / x))


def _valuation_at_infinity(E: FunctionField, w: Element) -> Fraction:
    Y = symbols('Y')
    Kx = E.Kx
    curve = as_expr(sum(Kx.to_sympy(c) * Y**(E.n - k) for k, c in enumerate(E.F)))
    expression = as_expr(sum(Kx.to_sympy(c) * Y**i for i, c in enumerate(w.coefficients)))
    numerator, denominator = expression.as_numer_denom()
    values: list[Fraction] = []
    for branch in puiseux_expansions(curve, x, Y, S.Infinity, order=6):
        s = branch.evaluate(Poly(numerator, Y, x)) * branch.evaluate(Poly(denominator, x)).inverse(12)
        v = s.valuation()
        assert v is not None
        values.append(Fraction(v, branch.ramification))
    return min(values)


def test_local_bases_at_infinity_and_pole_orders() -> None:
    for F in [y**2 - x**3 - 1, y**3 - x**2 * (x + 1), y**4 + (x**2 + x) * y - x**2, y**4 - 4 * x * y**2 + 4,
              (x + 1) * y**2 - x**3, y**2 - x**8 - 1]:
        E = FunctionField(F, x, y)
        local = E.local_basis_at_infinity()
        for w in local:
            assert _valuation_at_infinity(E, w) >= 0, (F, w)
        for w in E.integral_basis():
            assert E.pole_order_at_infinity(w) == ceil(-_valuation_at_infinity(E, w)), (F, w)


def test_normal_bases() -> None:
    E = FunctionField(y**2 - x**3 - 1, x, y)
    assert [w.as_expr() for w in E.normal_basis()] == [1, y]
    assert E.pole_orders_at_infinity(E.normal_basis()) == [0, 2]
    for F in [y**3 - x**2 * (x + 1), y**4 + (x**2 + x) * y - x**2, y**4 - 4 * x * y**2 + 4, (x + 1) * y**2 - x**3]:
        E = FunctionField(F, x, y)
        normal = E.normal_basis()
        orders = E.pole_orders_at_infinity(normal)
        # the module is the integral closure still
        for w in normal:
            assert E.is_integral(w)
        change = DomainMatrix([list(E.coordinates(w, E.integral_basis())) for w in normal], (E.n, E.n), E.Kx).det()
        assert change.numer.degree() == 0 and change.denom.degree() == 0
        # normal: the pole order of a combination is the maximum of the terms
        combination = E.zero()
        for k, w in enumerate(normal):
            combination = combination + w.scaled(E.Kx.from_sympy(x**(k + 1) + 1))
        assert E.pole_order_at_infinity(combination) == max(k + 1 + orders[k] for k in range(E.n))
        assert E.pole_order_at_infinity(combination) == ceil(-_valuation_at_infinity(E, combination))

"""Tests of the native Trager algorithm (:mod:`sympy_extras.integrals.trager`)
on the objects of the core; the old API on SymPy objects is tested in
test_trager.py through :mod:`sympy_extras.integrals.trager_sympy`."""
from __future__ import annotations

from fractions import Fraction

from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import diff
from sympy_extras.functions import sqrt
from sympy_extras.integrals import trager
from sympy_extras.integrals.trager import (hermite_reduce_algebraic, is_nonelementary_algebraic,
                                           trager_antiderivative, trager_reduce)
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.polyclasses import DensePoly
from sympy_extras.polys.polyerrors import NotInvertible
from sympy_extras.polys.polytools import cancel
from sympy_extras.simplify.simplify import simplify

x, a, b, c = Symbol('x'), Symbol('a'), Symbol('b'), Symbol('c')


def _is_antiderivative(F: Expr, f: Expr) -> bool:
    return cancel(diff(F, x) - f) == 0 or simplify(diff(F, x) - f) == 0


def test_square_roots() -> None:
    cases = [
        (1 / (x * sqrt(x**2 + 1)), '-log((sqrt(x**2 + 1) + 1)/x)'),
        (x / sqrt(x**4 + 1), 'log(x**2 + sqrt(x**4 + 1))/2'),
        (1 / sqrt(x**2 - 1), 'log(x + sqrt(x**2 - 1))'),
        (1 / (x * sqrt(x**3 + 1)), '-log((x**3/2 + sqrt(x**3 + 1) + 1)/x**3)/3'),
    ]
    for f, expected in cases:
        F = trager_antiderivative(f, x)
        assert F is not None and str(F) == expected, (f, F)
        assert _is_antiderivative(F, f)
    # residues in QQ(sqrt(2), I) and QQ<I>: logarithms with complex coefficients
    for f in [(x**2 - 1) / ((x**2 + 1) * sqrt(x**4 + 1)), 1 / ((x**2 + 1) * sqrt(x**2 + 2))]:
        F = trager_antiderivative(f, x)
        assert F is not None and F.has(I) and _is_antiderivative(F, f)


def test_hermite_reduction_and_the_remainder() -> None:
    assert str(hermite_reduce_algebraic(x**2 / sqrt(x**2 + 1), x)) == '(x*sqrt(x**2 + 1)/2, -1/(2*sqrt(x**2 + 1)))'
    assert str(trager_reduce(1 / sqrt(x**3 + 1), x)) == '(0, 1/sqrt(x**3 + 1))'
    f = (2 * x**2 + 1) / ((x**2 + 1) * sqrt(x**4 + x**2 + 1))
    found = trager_reduce(f, x)
    assert found is not None
    F, h = found
    assert str(h) == '3/(2*sqrt(x**4 + x**2 + 1))' and _is_antiderivative(F, f - h)
    assert is_nonelementary_algebraic(f, x) is True
    assert is_nonelementary_algebraic(x / sqrt(x**4 + 1), x) is False
    assert is_nonelementary_algebraic(1 / ((x - 1) * sqrt(x**3 + 1)), x) is None


def test_parameters() -> None:
    # the logarithmic part over QQ(a)
    F = trager_antiderivative(1 / sqrt(x**2 + a), x)
    assert F is not None and _is_antiderivative(F, 1 / sqrt(x**2 + a))
    # the parameters of the typed polynomials are known by their names:
    # the answer had the plain symbol p in place of the positive one
    p = Symbol('p', positive=True)
    F = trager_antiderivative(1 / sqrt(x**2 + p), x)
    assert F is not None and F.free_symbols == {x, p}
    # FriCAS's test 295: residues algebraic over QQ(a, b, c), the parameters
    # made rational and the algorithm rerun (the radicand of the new
    # parameters has rational coefficients in them, QQ(delta), not EX:
    # it was left undecided)
    f = sqrt(a + b * x**2 + c * x**4) / (a - c * x**4)
    F = trager_antiderivative(f, x)
    assert F is not None and F.has(sqrt(a + b * x**2 + c * x**4))


def test_rational_parameters() -> None:
    found = trager._rational_parameters([sqrt(b - 2 * sqrt(a) * sqrt(c)) / 4, sqrt(c)], [a, b, c])
    assert found is not None
    substitution, back = found
    assert set(substitution) <= {a, b, c}
    assert all(isinstance(value, Expr) and not any(trager._half_integer(node) for node in trager._pows(value))
               for value in substitution.values())
    assert all(isinstance(value, Expr) and any(trager._half_integer(node) for node in trager._pows(value))
               for value in back.values())
    # a root linear in no parameter: no reparametrization
    assert trager._rational_parameters([sqrt(a**2 + 1)], [a]) is None


def test_roots_of_index_n() -> None:
    cases = [x**2 * (x**3 + 1)**Rational(1, 3), x**2 * (x**3 + a)**Rational(1, 3), 1 / (x * (x**3 + 1)**Rational(1, 3)),
             1 / (x**3 + 1)**Rational(1, 3), 1 / (x * (x**4 + 1)**Rational(1, 4))]
    for f in cases:
        F = trager_antiderivative(f, x)
        assert F is not None, f
        assert not F.has(I), f
    assert trager_antiderivative(x / (x**3 + 1)**Rational(1, 3), x) is None
    parsed = trager._root_components(1 / (1 + (x**3 + 1)**Rational(1, 3)), x)
    assert parsed is not None
    P, n, components = parsed
    assert n == 3 and [cancel(component * (x**3 + 2)) for component in components] == [1, -1, 1]
    # not squarefree, a square root, two radicands: not this route
    assert trager._root_components((x**2 * (x + 1))**Rational(1, 3), x) is None
    assert trager._root_components(sqrt(x**3 + 1), x) is None
    assert trager._root_components((x + 1)**Rational(1, 3) * (x + 2)**Rational(1, 3), x) is None


def test_hermite_reduction_and_residues_on_the_curve_of_index_n() -> None:
    P = DensePoly.from_expr(x**3 + 1, x, QQ)
    y = (x**3 + 1)**Rational(1, 3)
    A = 1 / (x**4 * (x**3 + 1))
    reduced = trager._hermite_index_n(A, P, 3, 2, x)
    assert reduced is not None
    terms, M, D = reduced
    assert D.to_expr() == x and all(T.to_expr() != 1 for _, T in terms)
    exact: Expr = Integer(0)
    for S_, T_ in terms:
        exact = exact + S_.to_expr() * y**2 / T_.to_expr()
    assert simplify(diff(exact, x) + M.to_expr() * y**2 / (D.to_expr() * (x**3 + 1)) - A * y**2) == 0
    for A_ in [x / (x**3 + 1), 1 / (x**2 * (x**3 + 1)), 1 / ((x + 1) * (x**3 + 1))]:
        assert trager._hermite_index_n(A_, P, 3, 2, x) is None, A_
    one = DensePoly.from_expr(Integer(1), x, QQ)
    assert trager._residue_at_infinity(one, one, P, 3, 2) == 1
    assert trager._residue_at_infinity(one, DensePoly.from_expr(x, x, QQ), P, 3, 2) == 0
    N = trager._NumberField([Rational(-1, 2) + sqrt(3) * I / 2])
    K = N.field
    zeta = N.coordinates[Rational(-1, 2) + sqrt(3) * I / 2]
    basis = trager._rational_basis([K.one, zeta, K.mul(zeta, zeta)], K)
    assert basis is not None and basis[0] == [0, 1]
    assert basis[1] == [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)], [Fraction(-1), Fraction(-1)]]


def test_the_inverse_modulo_a_polynomial_over_the_gaussian_parametric_field() -> None:
    F = trager._algebraic_parametric(trager._gaussian(), [a, b])
    G = F.from_expr(x**2 - (a**2 + b**2) / 4, x)
    p = F.from_expr(4 * x**3, x).mul_ground(F.element(I * b / (a**2 + b**2)))
    inverse = trager._inverse(p, G)
    assert F.poly_expr((p * inverse).rem(G)) == 1
    try:
        trager._inverse(F.from_expr(x, x), F.from_expr(x**2, x))
        assert False
    except NotInvertible:
        pass


def test_number_fields_of_the_constants() -> None:
    N = trager._NumberField([sqrt(2), I])
    assert N.degree == 4
    L = trager._algebraic(N)
    assert L.expr(L.element(sqrt(2) * I + 1)) == 1 + sqrt(2) * I
    assert L.expr(L.domain.mul(L.element(sqrt(2)), L.element(sqrt(2)))) == 2


def test_the_general_curve() -> None:
    F = trager_antiderivative(sqrt(x) + sqrt(x + 1), x)
    assert F is not None and _is_antiderivative(F, sqrt(x) + sqrt(x + 1))
    F = trager_antiderivative(sqrt(x) * sqrt(x + 1), x)
    assert F is not None and F.has(sqrt(x) * sqrt(x + 1))

"""Tests of the univariate view of multivariate polynomials (native)."""
from __future__ import annotations

import random

from sympy import Matrix, Poly as SymPyPoly, expand, sympify
from sympy.core.expr import Expr as SymPyExpr
from sympy.polys.domains import ZZ as SymPyZZ
from sympy.polys.rings import PolyElement as SymPyPolyElement, ring
from sympy.testing.pytest import raises

from sympy_extras._typing_sympy import as_expr, as_symbol

from sympy_extras.core import Integer, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.polys.domains import ZZ
from sympy_extras.polys.regularchains.recursive import (ZPoly, coefficient, degree, distinct_factors, exact_quotient,
    initial, main_variable, normalize, prem, primitive_part, pseudo_divide, pseudo_remainder, subresultant_chain,
    subresultants, tail)
from sympy_extras.polys.rings import PolyRing

R = PolyRing(ZZ, ['x', 'y', 'z'])
X, Y, Z = Symbol('x'), Symbol('y'), Symbol('z')


def P(e: Expr) -> ZPoly:
    return R.from_expr(e)


def _random(generator: random.Random, d: int, leading: Expr) -> ZPoly:
    p: Expr = leading*X**d
    for k in range(d):
        p = p + generator.randint(-3, 3)*Y**generator.randint(0, 2)*Z**generator.randint(0, 1)*X**k
    return P(p)


def _sympy(p: ZPoly) -> SymPyExpr:
    """The polynomial as a SymPy expression (an independent oracle)."""
    return as_expr(sympify(str(p)))


def test_main_variable_initial_tail() -> None:
    p = P((Y + 1)*X**2 + Y*X + 3)
    assert main_variable(p) == 0 and main_variable(P(Y*Z + 1)) == 1 and main_variable(P(Z**3)) == 2
    assert main_variable(R.from_int(5)) == 3 and main_variable(R.zero) == 3
    assert degree(p, 0) == 2 and degree(p, 1) == 1 and degree(p, 2) == 0 and degree(R.zero, 0) == -1
    assert initial(p) == P(Y + 1) and tail(p) == P(Y*X + 3)
    assert initial(P(Y*Z**2 + Z)) == P(Z**2) and tail(P(Y*Z**2 + Z)) == P(Z)
    assert initial(R.from_int(7)) == R.from_int(7) and tail(R.from_int(7)) == R.zero
    assert coefficient(p, 0, 1) == P(Y) and coefficient(p, 1, 1) == P(X**2 + X) and coefficient(p, 0, 5) == R.zero
    # the initial and the tail rebuild the polynomial
    for q in (p, P(X*Y*Z - Z**2 + 1), P(Y**3*Z + Y)):
        v = main_variable(q)
        assert initial(q)*R.gens[v]**degree(q, v) + tail(q) == q


def test_normalize_and_primitive_part() -> None:
    assert normalize(P(-6*X*Y + 4*Z)) == P(3*X*Y - 2*Z)
    assert normalize(R.zero) == R.zero
    p = P(2*(Y**2 - 1)*X**2 + 4*(Y + 1)*X)
    assert primitive_part(p, 0) == P((Y - 1)*X**2 + 2*X)
    # the content depends on the variable
    assert primitive_part(p, 1) == P(X*Y**2 + 2*Y - X + 2)
    assert primitive_part(P(X*Y + Z), 0) == P(X*Y + Z)
    assert primitive_part(R.from_int(-4), 0) == R.one


def test_pseudo_division() -> None:
    generator = random.Random(5)
    for _ in range(30):
        f = _random(generator, generator.randint(1, 5), Y*Z + generator.randint(1, 3))
        g = _random(generator, generator.randint(1, 3), generator.choice([Y + 1, Z**2 - Y, Integer(2), Y*Z]))
        m, q, r = pseudo_divide(f, g, 0)
        assert m*f == q*g + r and degree(r, 0) < degree(g, 0)
        assert r == pseudo_remainder(f, g, 0)
        # the multiplier divides a power of the leading coefficient of the divisor
        lead = coefficient(g, 0, degree(g, 0))
        assert exact_quotient(lead**(degree(f, 0) + 1), m)*m == lead**(degree(f, 0) + 1)
        # the true pseudo-remainder is a multiple of the lazy one
        assert prem(f, g, 0)*m == r*lead**max(degree(f, 0) - degree(g, 0) + 1, 0)
    # in another variable
    f, g = P(X*Y**3 + Z), P(Z*Y**2 - X)
    m, q, r = pseudo_divide(f, g, 1)
    assert m*f == q*g + r and degree(r, 1) < 2
    assert pseudo_divide(P(Y + 1), P(X**2 + Y), 0) == (R.one, R.zero, P(Y + 1))
    raises(ZeroDivisionError, lambda: pseudo_divide(P(X), R.zero, 0))
    raises(ZeroDivisionError, lambda: pseudo_remainder(P(X), R.zero, 0))


def test_prem_and_subresultants_are_sympys() -> None:
    # the ports of PolyElement.prem and PolyElement.subresultants against SymPy 1.14
    S, x, y, z = ring('x y z', SymPyZZ)

    def converted(p: ZPoly) -> SymPyPolyElement:
        result: SymPyPolyElement = S.from_dict({m: SymPyZZ(c) for m, c in p.terms.items()}) if p.terms else S.zero
        return result

    generator = random.Random(11)
    for _ in range(40):
        f = _random(generator, generator.randint(1, 4), Y + generator.randint(0, 2))
        g = _random(generator, generator.randint(1, 3), Z - Y)
        for i in range(3):
            if degree(g, i) >= 0:
                assert converted(prem(f, g, i)) == converted(f).prem(converted(g), i)
            assert [converted(s) for s in subresultants(f, g, i)] == converted(f).subresultants(converted(g), i)


def _determinant_subresultant(f: ZPoly, g: ZPoly, j: int) -> SymPyExpr:
    """The `j`-th subresultant as the determinant polynomial of the rows
    `x^k f` and `x^k g` of the Sylvester matrix (with SymPy)."""
    sx = as_symbol(sympify('x'))
    F, G = SymPyPoly(_sympy(f), sx), SymPyPoly(_sympy(g), sx)
    m, n = F.degree(), G.degree()
    rows = [[0]*k + F.all_coeffs() + [0]*(n - j - 1 - k) for k in range(n - j)]
    rows += [[0]*k + G.all_coeffs() + [0]*(m - j - 1 - k) for k in range(m - j)]
    size = m + n - 2*j
    result: SymPyExpr = expand(sum(
        Matrix([[row[c] for c in list(range(size - 1)) + [m + n - j - 1 - i]] for row in rows]).det()*sx**i
        for i in range(j + 1)))
    return result


def test_subresultant_chain_against_determinants() -> None:
    generator = random.Random(3)
    cases = []
    for _ in range(8):
        f = _random(generator, generator.randint(2, 4), Y + 1)
        g = _random(generator, generator.randint(1, 3), Z + Y)
        cases.append((f, g) if degree(f, 0) >= degree(g, 0) else (g, f))
    # common factors and gaps in the degrees: the defective cases
    cases += [(P((X - Y)**2*(X + 1)), P((X - Y)*(X - 2))), (P(X**5 + Y), P(X**2 + Y)),
              (P(X**4 + X*Y + 1), P(X**4 + Y)), (P(X**6 + Y*X + 1), P(X**2*Y + 1)), (P(X**5 + Z), P(X**3 + Y*X**3 + 1))]
    for f, g in cases:
        chain = subresultant_chain(f, g, 0)
        assert len(chain) == degree(g, 0)
        for j, S in enumerate(chain):
            expected = _determinant_subresultant(f, g, j)
            assert expand(_sympy(S) - expected) == 0 or expand(_sympy(S) + expected) == 0
            assert degree(S, 0) <= j


def test_subresultant_chain_properties() -> None:
    f, g = P(X**2 + Y**2 - 1), P(X*Y - 1)
    chain = subresultant_chain(f, g, 0)
    assert chain == [P(Y**4 - Y**2 + 1)]
    sx = as_symbol(sympify('x'))
    assert _sympy(chain[0]) == SymPyPoly(_sympy(f), sx).resultant(SymPyPoly(_sympy(g), sx)).as_expr()
    # with respect to another variable
    assert subresultant_chain(P(Y**2 + Z*Y + X), P(Y - X), 1) == [P(X**2 + X*Z + X)]
    # a common factor of degree two: the subresultants below it vanish
    common = P(X**2 + Y)
    chain = subresultant_chain(common*P(X**2 + Z), common*P(X - 1), 0)
    assert chain[0] == R.zero and chain[1] == R.zero and degree(chain[2], 0) == 2
    assert exact_quotient(chain[2], common)*common == chain[2]
    raises(ValueError, lambda: subresultant_chain(P(X + 1), P(X**2 + Y), 0))
    raises(ValueError, lambda: subresultant_chain(P(X + 1), P(Y), 0))


def test_distinct_factors() -> None:
    assert distinct_factors(P(-2*(X*Y - 1)**2*(Z + 3*Y)*Y**3)) == [P(Y), P(3*Y + Z), P(X*Y - 1)]
    assert distinct_factors(R.from_int(6)) == [] and distinct_factors(P(X**2 + Y**2)) == [P(X**2 + Y**2)]


def test_exact_quotient() -> None:
    p = P(X*Y + Z)
    q = exact_quotient(p*P(X), P(X))
    assert q == p and hash(q) == hash(p) and len({p, q}) == 1
    assert len({primitive_part(p*P(Y + 1), 0), p}) == 1
    assert exact_quotient(P((X - Y)*(X**2 + Z)), P(X - Y)) == P(X**2 + Z)
    raises(ValueError, lambda: exact_quotient(P(X**2 + 1), P(X - 1)))
    raises(ValueError, lambda: exact_quotient(P(3*X), P(2*X)))
    raises(ZeroDivisionError, lambda: exact_quotient(P(X), R.zero))

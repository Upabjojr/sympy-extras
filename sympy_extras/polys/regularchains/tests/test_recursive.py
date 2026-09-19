"""Tests of the univariate view of multivariate polynomials."""
from __future__ import annotations

import random

from sympy import Matrix, Poly, ZZ, expand, symbols
from sympy.core.expr import Expr
from sympy.polys.rings import PolyElement, ring
from sympy.testing.pytest import raises

from sympy_extras.polys.regularchains.recursive import (coefficient, degree, distinct_factors, exact_quotient,
    initial, main_variable, normalize, primitive_part, pseudo_divide, pseudo_remainder, subresultant_chain, tail)

R, x, y, z = ring('x y z', ZZ)
X, Y, Z = symbols('x y z')


def _random(generator: random.Random, d: int, leading: PolyElement) -> PolyElement:
    p: PolyElement = leading*x**d
    for k in range(d):
        p = p + generator.randint(-3, 3)*y**generator.randint(0, 2)*z**generator.randint(0, 1)*x**k
    return p


def test_main_variable_initial_tail() -> None:
    p = (y + 1)*x**2 + y*x + 3
    assert main_variable(p) == 0 and main_variable(y*z + 1) == 1 and main_variable(z**3) == 2
    assert main_variable(R(5)) == 3 and main_variable(R.zero) == 3
    assert degree(p, 0) == 2 and degree(p, 1) == 1 and degree(p, 2) == 0 and degree(R.zero, 0) == -1
    assert initial(p) == y + 1 and tail(p) == y*x + 3
    assert initial(y*z**2 + z) == z**2 and tail(y*z**2 + z) == z
    assert initial(R(7)) == 7 and tail(R(7)) == 0
    assert coefficient(p, 0, 1) == y and coefficient(p, 1, 1) == x**2 + x and coefficient(p, 0, 5) == 0
    # the initial and the tail rebuild the polynomial
    for q in (p, x*y*z - z**2 + 1, y**3*z + y):
        v = main_variable(q)
        assert initial(q)*R.gens[v]**degree(q, v) + tail(q) == q


def test_normalize_and_primitive_part() -> None:
    assert normalize(-6*x*y + 4*z) == 3*x*y - 2*z
    assert normalize(R.zero) == 0
    p = 2*(y**2 - 1)*x**2 + 4*(y + 1)*x
    assert primitive_part(p, 0) == (y - 1)*x**2 + 2*x
    # the content depends on the variable
    assert primitive_part(p, 1) == x*y**2 + 2*y - x + 2
    assert primitive_part(x*y + z, 0) == x*y + z
    assert primitive_part(R(-4), 0) == 1


def test_pseudo_division() -> None:
    generator = random.Random(5)
    for _ in range(30):
        f = _random(generator, generator.randint(1, 5), y*z + generator.randint(1, 3))
        g = _random(generator, generator.randint(1, 3), generator.choice([y + 1, z**2 - y, R(2), y*z]))
        m, q, r = pseudo_divide(f, g, 0)
        assert m*f == q*g + r and degree(r, 0) < degree(g, 0)
        assert r == pseudo_remainder(f, g, 0)
        # the multiplier divides a power of the leading coefficient of the divisor
        lead = coefficient(g, 0, degree(g, 0))
        assert not (lead**(degree(f, 0) + 1)).rem(m)
        # the true pseudo-remainder is a multiple of the lazy one
        assert f.prem(g, 0)*m == r*lead**max(degree(f, 0) - degree(g, 0) + 1, 0)
    # in another variable
    m, q, r = pseudo_divide(x*y**3 + z, z*y**2 - x, 1)
    assert m*(x*y**3 + z) == q*(z*y**2 - x) + r and degree(r, 1) < 2
    assert pseudo_divide(y + 1, x**2 + y, 0) == (1, 0, y + 1)
    raises(ZeroDivisionError, lambda: pseudo_divide(x, R.zero, 0))
    raises(ZeroDivisionError, lambda: pseudo_remainder(x, R.zero, 0))


def _determinant_subresultant(f: PolyElement, g: PolyElement, j: int) -> Expr:
    """The `j`-th subresultant as the determinant polynomial of the rows
    `x^k f` and `x^k g` of the Sylvester matrix."""
    F, G = Poly(f.as_expr(), X), Poly(g.as_expr(), X)
    m, n = F.degree(), G.degree()
    rows = [[0]*k + F.all_coeffs() + [0]*(n - j - 1 - k) for k in range(n - j)]
    rows += [[0]*k + G.all_coeffs() + [0]*(m - j - 1 - k) for k in range(m - j)]
    size = m + n - 2*j
    result: Expr = expand(sum(
        Matrix([[row[c] for c in list(range(size - 1)) + [m + n - j - 1 - i]] for row in rows]).det()*X**i
        for i in range(j + 1)))
    return result


def test_subresultant_chain_against_determinants() -> None:
    generator = random.Random(3)
    cases = []
    for _ in range(8):
        f = _random(generator, generator.randint(2, 4), y + 1)
        g = _random(generator, generator.randint(1, 3), z + y)
        cases.append((f, g) if degree(f, 0) >= degree(g, 0) else (g, f))
    # common factors and gaps in the degrees: the defective cases
    cases += [((x - y)**2*(x + 1), (x - y)*(x - 2)), (x**5 + y, x**2 + y), (x**4 + x*y + 1, x**4 + y),
        (x**6 + y*x + 1, x**2*y + 1), (x**5 + z, x**3 + y*x**3 + 1)]
    for f, g in cases:
        chain = subresultant_chain(f, g, 0)
        assert len(chain) == degree(g, 0)
        for j, S in enumerate(chain):
            expected = _determinant_subresultant(f, g, j)
            assert expand(S.as_expr() - expected) == 0 or expand(S.as_expr() + expected) == 0
            assert degree(S, 0) <= j


def test_subresultant_chain_properties() -> None:
    f, g = x**2 + y**2 - 1, x*y - 1
    chain = subresultant_chain(f, g, 0)
    assert chain == [y**4 - y**2 + 1]
    assert chain[0].as_expr() == f.as_expr().as_poly(X).resultant(g.as_expr().as_poly(X)).as_expr()
    # with respect to another variable
    assert subresultant_chain(y**2 + z*y + x, y - x, 1) == [x**2 + x*z + x]
    # a common factor of degree two: the subresultants below it vanish
    common = x**2 + y
    chain = subresultant_chain(common*(x**2 + z), common*(x - 1), 0)
    assert chain[0] == 0 and chain[1] == 0 and not chain[2].rem(common) and degree(chain[2], 0) == 2
    raises(ValueError, lambda: subresultant_chain(x + 1, x**2 + y, 0))
    raises(ValueError, lambda: subresultant_chain(x + 1, y, 0))


def test_distinct_factors() -> None:
    assert distinct_factors(-2*(x*y - 1)**2*(z + 3*y)*y**3) == [y, 3*y + z, x*y - 1]
    assert distinct_factors(R(6)) == [] and distinct_factors(x**2 + y**2) == [x**2 + y**2]


def test_quotients_have_the_hash_of_their_terms() -> None:
    # SymPy 1.14: (p*x).exquo(x) == p but its cached hash is not hash(p)
    # (issue #25), so that equal chains were returned twice by a
    # decomposition; exact_quotient returns a copy without the cache
    p = x*y + z
    q = exact_quotient(p*x, x)
    assert q == p and hash(q) == hash(p) and len({p, q}) == 1
    assert len({primitive_part(p*(y + 1), 0), p}) == 1

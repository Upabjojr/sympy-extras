"""The native Darboux polynomials and Prelle–Singer procedure
(:mod:`sympy_extras.solvers.darboux`) on the objects of the core: every
result is checked by its defining property (``D f = g f``, ``D(A/B) =
L``, the exactness of ``R (Q dy - P dx)``, ``D I = 0``)."""
from __future__ import annotations

from fractions import Fraction

from sympy_extras.core import Function, I, Integer, Rational, Symbol, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import expand
from sympy_extras.functions import exp, log
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.polytools import cancel
from sympy_extras.polys.rings import PolyRing, lex
from sympy_extras.solvers.darboux import (DarbouxPolynomial, _NumberField, _Parameters, _Rationals,
    _field_of, _number_field, darboux_first_integral, darboux_integrating_factor, darboux_polynomials,
    exponential_factors, prelle_singer, vector_field)

x, y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')


def _D(P: Expr, Q: Expr, e: Expr) -> Expr:
    """The vector field ``Q d/dx + P d/dy`` applied to ``e``."""
    return cancel(Q*e.diff(x) + P*e.diff(y))


def _darboux(P: Expr, Q: Expr, found: list[DarbouxPolynomial]) -> bool:
    return all(expand(Q*d.polynomial.diff(x) + P*d.polynomial.diff(y) - d.cofactor*d.polynomial) == 0
               for d in found)


def _exact(P: Expr, Q: Expr, R: Expr) -> bool:
    return cancel((R*Q).diff(x) + (R*P).diff(y)) == 0


def _first_integral(P: Expr, Q: Expr, F: Expr) -> bool:
    return _D(P, Q, F) == 0 and (F.has(x) or F.has(y))


def test_lotka_volterra() -> None:
    P, Q = y*(x - 1), x*(1 - y)
    found = darboux_polynomials(P, Q, x, y, degree=3)
    assert _darboux(P, Q, found)
    assert {(d.polynomial, d.cofactor) for d in found} == {(x, 1 - y), (y, x - 1)}
    R = darboux_integrating_factor(P, Q, x, y)
    assert R is not None and _exact(P, Q, R) and R == 1/(x*y)
    F = darboux_first_integral(P, Q, x, y)
    assert F is not None and _first_integral(P, Q, F) and F == -x - y + log(x) + log(y)


def test_conjugate_lines() -> None:
    # over the rationals the product of the conjugate lines x +- I y,
    # over Q(i) the lines with conjugate cofactors
    P, Q = x + y, x - y
    assert darboux_polynomials(P, Q, x, y, degree=2) == [DarbouxPolynomial(x**2 + y**2, Integer(2))]
    over_i = darboux_polynomials(P, Q, x, y, degree=1, extension=I)
    assert _darboux(P, Q, over_i) and {d.polynomial for d in over_i} == {x - I*y, x + I*y}
    # the complex first integral of degree one, found over Q(2 i): the
    # square root of the discriminant of x**2 + y**2
    F = darboux_first_integral(P, Q, x, y, degree=1)
    assert F is not None and F.has(I) and _first_integral(P, Q, F)


def test_algebraic_coefficients() -> None:
    P: Expr = sqrt(2)*y
    Q: Expr = x
    found = darboux_polynomials(P, Q, x, y, degree=1)
    assert _darboux(P, Q, found)
    assert {d.cofactor for d in found} == {Integer(1), sqrt(2)}
    P, Q = -(2*x**2 + x*y + y**2), x*(y - x)
    found = darboux_polynomials(P, Q, x, y, degree=1, extension=I)
    assert _darboux(P, Q, found) and {d.polynomial for d in found} == {x, x - I*y, x + I*y}
    extension = _field_of(P, Q, x, y).quadratic_extension()
    assert extension is not None and expand(extension**2 + 4) == 0
    # several algebraic numbers: a primitive element (sqrt(2) + sqrt(3),
    # I + sqrt(2)); the invariant line has irrational coefficients
    for P in (sqrt(2)*y + sqrt(3)*x, I*y + sqrt(2)*x):
        found = darboux_polynomials(P, x, x, y, degree=1)
        assert _darboux(P, x, found) and len(found) == 2
        F = darboux_first_integral(P, x, x, y, degree=1)
        assert F is not None and _first_integral(P, x, F)
    # a cube root and its square
    P = Integer(2)**Rational(1, 3)*y + Integer(2)**Rational(2, 3)*x
    assert _darboux(P, x, darboux_polynomials(P, x, x, y, degree=1))


def test_pencil() -> None:
    # the level sets of (x**2 + y)/(y**2 + x) share one cofactor; the
    # rational first integral comes from the cofactors alone
    f1, f2 = x**2 + y, y**2 + x
    P = -(f1.diff(x)*f2 - f1*f2.diff(x))
    Q = f1.diff(y)*f2 - f1*f2.diff(y)
    found = darboux_polynomials(P, Q, x, y, degree=2)
    assert _darboux(P, Q, found)
    assert {f1, f2, x - y, x + y - 1} <= {d.polynomial for d in found}
    F = darboux_first_integral(P, Q, x, y, degree=2)
    assert F is not None and _first_integral(P, Q, F)
    assert cancel(F.diff(x)*(f1/f2).diff(y) - F.diff(y)*(f1/f2).diff(x)) == 0


def test_dicritical() -> None:
    found = darboux_polynomials(y, x, x, y, degree=1)
    assert _darboux(y, x, found) and {x, y} <= {d.polynomial for d in found}
    # x was lost when a row was divided by a content vanishing at the
    # cofactor (b**3 = 0 became 1 = 0)
    assert any(d.polynomial == x for d in darboux_polynomials(x*y - 1, x**2, x, y, degree=1))


def test_parameters() -> None:
    found = darboux_polynomials(a*y, x, x, y, degree=1)
    assert _darboux(a*y, x, found) and {d.cofactor for d in found} == {a, Integer(1)}
    F = darboux_first_integral(a*y, x, x, y, degree=1)
    assert F is not None and _first_integral(a*y, x, F)
    # the old module raised ValueError("0**0") from SymPy's
    # PolyElement.evaluate over ZZ(a, b) at a zero cofactor coefficient
    # (sympy-extras#25); the native evaluation finds y**2 - b/a
    P, Q = b - a*y**2, Integer(1)
    found = darboux_polynomials(P, Q, x, y)
    assert _darboux(P, Q, found) and [d.polynomial for d in found] == [y**2 - b/a]
    R = darboux_integrating_factor(P, Q, x, y)
    assert R is not None and _exact(P, Q, R)
    # pi is a parameter, as for SymPy's Poly
    found = darboux_polynomials(pi*y, x, x, y, degree=1)
    assert _darboux(pi*y, x, found) and {d.cofactor for d in found} == {pi, Integer(1)}


def test_exponential_factors() -> None:
    found = exponential_factors(x*y + 1, Integer(1), x, y, degree=1)
    assert found and all(_D(x*y + 1, Integer(1), e.argument) == e.cofactor for e in found)
    assert darboux_integrating_factor(x*y + 1, Integer(1), x, y, degree=1) == exp(-x**2/2)
    found = exponential_factors(y, x**2, x, y, degree=1)
    assert any(cancel(e.argument + 1/x) == 0 for e in found)


def test_no_darboux_polynomial_of_low_degree() -> None:
    # Airy's Riccati equation and Jouanolou's field of degree three
    assert darboux_polynomials(y**2 + x, Integer(1), x, y, degree=3) == []
    assert darboux_first_integral(y**2 + x, Integer(1), x, y, degree=3) is None
    assert darboux_polynomials(1 - y*x**2, y**2 - x**3, x, y, degree=2) == []


def test_vector_field_and_prelle_singer() -> None:
    f = Function('y')(x)
    parsed = vector_field((x - f)*f.diff(x) - x - f, f)
    assert parsed is not None and parsed[:3] == (x + parsed[3], x - parsed[3], x)
    assert vector_field(f.diff(x)**2 - f, f) is None
    solution = prelle_singer(x*(1 - f)*f.diff(x) - f*(x - 1), f)
    assert solution is not None and solution.rhs == Symbol('C1')
    # explicit for the rational first integral x/y
    solution = prelle_singer(f.diff(x) - (x*f**2 - f)/(x**2*f - x), f)
    assert solution is not None and solution.lhs == f
    assert cancel(solution.rhs.diff(x) - solution.rhs/x) == 0


def test_explicit_despite_an_excluded_point() -> None:
    # (x y + 1)/(x - y) = C1 has the root (C1 x - 1)/(C1 + x) and the
    # excluded point y = x (the root of the denominator, missed for a
    # generic C1): a first version of the port took any excluded point
    # as a failure and left the solution implicit (SymPy's solve, in the
    # old module, gave it)
    f = Function('y')(x)
    solution = prelle_singer((1 + x**2)*f.diff(x) - 1 - f**2, f)
    assert solution is not None and solution.lhs == f
    s = solution.rhs
    assert cancel((1 + x**2)*s.diff(x) - 1 - s**2) == 0


def test_factorisation_over_the_fields() -> None:
    R = PolyRing(QQ, ['u', 'v'], lex)
    u, v = R.gens
    assert _Rationals().factor((u*u - v*v).mul_ground(Fraction(3))) == [u - v, u + v]
    K = _number_field(I)
    S = PolyRing(K, ['u', 'v'], lex)
    U, V = S.gens
    i = S.ground(K.alpha)
    assert len(_NumberField(K).factor(U*U + V*V)) == 2
    assert all(g.degree() == 1 for g in _NumberField(K).factor((U - i*V)*(U + i*V)))
    C = _Parameters([a])
    T = PolyRing(C.K, ['u'], lex)
    t = T.gens[0]
    A = T.ground(C.K.from_expr(C.rename(a)))
    assert [g.degree() for g in C.factor(t*t - A*A)] == [1, 1]
    assert C.factor(t*t - A) == [t*t - A]

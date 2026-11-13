"""Most of these tests come from the examples in Bronstein's book (the
Risch differential equation on the typed core)."""
from __future__ import annotations

from typing import Optional, Union

from pytest import raises

from sympy_extras.core import I, Rational
from sympy_extras.core.expr import Expr
from sympy_extras.core.symbol import Symbol
from sympy_extras.integrals.risch.risch import (DifferentialExtension, derivation,
    NonElementaryIntegralException)
from sympy_extras.integrals.risch.rde import (OO, order_at, order_at_oo, weak_normalizer,
    normal_denom, special_denom, _special_denom_cancel_bound, bound_degree,
    spde, solve_poly_rde, no_cancel_equal, cancel_primitive, cancel_exp,
    cancel_tan, rischDE)
from sympy_extras.integrals.risch.tower import TowerPoly
from sympy_extras.polys.polytools import cancel

x, t, z, k = Symbol('x'), Symbol('t'), Symbol('z'), Symbol('k')
t0, t1, t2 = Symbol('t0'), Symbol('t1'), Symbol('t2')
oo = OO


def P(e: Union[Expr, int], gen: Optional[Symbol] = None) -> TowerPoly:
    """SymPy's ``Poly(e, gen)`` (the only free symbol by default)."""
    if gen is None:
        free = sorted(e.free_symbols, key=str) if isinstance(e, Expr) else []
        if len(free) != 1:
            raise ValueError('a generator is needed for %s' % (e,))
        gen = free[0]
    return TowerPoly.from_expr(e, gen)


def test_order_at() -> None:
    a = P(t**4, t)
    b = P((t**2 + 1)**3*t, t)
    c = P((t**2 + 1)**6*t, t)
    d = P((t**2 + 1)**10*t**10, t)
    e = P((t**2 + 1)**100*t**37, t)
    p1 = P(t, t)
    p2 = P(1 + t**2, t)
    assert order_at(a, p1, t) == 4
    assert order_at(b, p1, t) == 1
    assert order_at(c, p1, t) == 1
    assert order_at(d, p1, t) == 10
    assert order_at(e, p1, t) == 37
    assert order_at(a, p2, t) == 0
    assert order_at(b, p2, t) == 3
    assert order_at(c, p2, t) == 6
    assert order_at(d, p1, t) == 10
    assert order_at(e, p2, t) == 100
    assert order_at(P(0, t), P(t, t), t) == oo
    assert order_at_oo(P(t**2 - 1, t), P(t + 1), t) == \
        order_at_oo(P(t - 1, t), P(1, t), t) == -1
    assert order_at_oo(P(0, t), P(1, t), t) == oo

def test_weak_normalizer() -> None:
    a = P((1 + x)*t**5 + 4*t**4 + (-1 - 3*x)*t**3 - 4*t**2 + (-2 + 2*x)*t, t)
    d = P(t**4 - 3*t**2 + 2, t)
    DE = DifferentialExtension.manual([P(1, x), P(t, t)])
    r = weak_normalizer(a, d, DE, z)
    assert r == (P(t**5 - t**4 - 4*t**3 + 4*t**2 + 4*t - 4, t),
        (P((1 + x)*t**2 + x*t, t),
         P(t + 1, t)))
    assert weak_normalizer(r[1][0], r[1][1], DE) == (P(1, t), r[1])
    # The residue of (1 + t**2)/(t**2 - 1) at t == 1 and at t == -1 is 1:
    # q == gcd(a - Dd1, d1)**1 == t**2 - 1 (Bronstein's WeakNormalizer);
    # SymPy's (t**2 - 1)**2 takes the gcd once per root of the resultant
    # counted with multiplicity (also weakly normalized here, residue -1)
    r = weak_normalizer(P(1 + t**2), P(t**2 - 1, t), DE, z)
    assert r == (P(t**2 - 1, t), (P(-1, t), P(1, t)))
    assert weak_normalizer(r[1][0], r[1][1], DE, z) == (P(1, t), r[1])
    # SymPy's weak_normalizer(Poly(3, t), Poly(t - 1, t)) returns q == t - 1
    # and (3 - t)/(t - 1), whose residue 2 at t == 1 is a positive integer
    # (not weakly normalized): the gcd must be raised to the power n == 3
    q, (sa, sd) = weak_normalizer(P(3, t), P(t - 1, t), DE, z)
    assert q == P((t - 1)**3, t) and (sa, sd) == (P(-3, t), P(1, t))
    assert weak_normalizer(sa, sd, DE, z) == (P(1, t), (sa, sd))
    DE = DifferentialExtension.manual([P(1, x), P(1 + t**2)])
    r = weak_normalizer(P(1 + t**2), P(t, t), DE, z)
    assert r == (P(t, t), (P(0, t), P(1, t)))
    assert weak_normalizer(r[1][0], r[1][1], DE, z) == (P(1, t), r[1])


def test_normal_denom() -> None:
    DE = DifferentialExtension.manual([P(1, x)])
    raises(NonElementaryIntegralException, lambda: normal_denom(P(1, x), P(1, x),
    P(1, x), P(x, x), DE))
    fa, fd = P(t**2 + 1, t), P(1, t)
    ga, gd = P(1, t), P(t**2, t)
    DE = DifferentialExtension.manual([P(1, x), P(t**2 + 1, t)])
    assert normal_denom(fa, fd, ga, gd, DE) == \
        (P(t, t), (P(t**3 - t**2 + t - 1, t), P(1, t)), (P(1, t),
        P(1, t)), P(t, t))


def test_special_denom() -> None:
    # TODO: add more tests here
    DE = DifferentialExtension.manual([P(1, x), P(t, t)])
    assert special_denom(P(1, t), P(t**2, t), P(1, t), P(t**2 - 1, t),
    P(t, t), DE) == \
        (P(1, t), P(t**2 - 1, t), P(t**2 - 1, t), P(t, t))
#    assert special_denom(P(1, t), P(2*x, t), P((1 + 2*x)*t, t), DE) == 1

    # issue 3940
    # Note, this isn't a very good test, because the denominator is just 1,
    # but at least it tests the exp cancellation case
    DE = DifferentialExtension.manual([P(1, x), P(-2*x*t0, t0),
        P(I*k*t1, t1)])
    DE.decrement_level()
    assert special_denom(P(1, t0), P(I*k, t0), P(1, t0), P(t0, t0),
    P(1, t0), DE) == \
        (P(1, t0), P(I*k, t0),
                P(t0, t0), P(1, t0))


    assert special_denom(P(1, t), P(t**2, t), P(1, t), P(t**2 - 1, t),
    P(t, t), DE, case='tan') == \
           (P(1, t), P(t**2, t), P(t, t), P(1, t0))  # SymPy's Poly(1, t, t0), Poly(t**2, t0, t), ...

    raises(ValueError, lambda: special_denom(P(1, t), P(t**2, t), P(1, t), P(t**2 - 1, t),
    P(t, t), DE, case='unrecognized_case'))

    # Example 6.2.1
    DE = DifferentialExtension.manual([P(1, x), P(t, t)])
    a = P(t**2 + 2*x*t + x**2, t)
    b = P((1 + 1/x**2)*t**2 + (2*x - 1 + 2/x)*t + x**2, t)
    c = P(t/x**2 - 1 + 2/x, t)
    assert special_denom(a, b, P(1, t), c, P(1, t), DE) == \
        (a, P(t**2/x**2 + (2/x - 1)*t, t), P(t**2/x**2 + (2/x - 1)*t, t), P(t, t))

    # Example 6.2.2, adding a hypertangent case where the special denominator
    # is not required in the reduction of the system, hence returning h = 1
    DE = DifferentialExtension.manual([P(1, x), P(1 + t**2, t)])
    a = P(t, t)
    ba = P((t - 1)*(t**2 + 1), t)
    bd = P(1, t)
    ca = P(1, t)
    cd = P(1, t)
    expected_a = P(t, t)
    expected_b = P(t**3 - t**2 + t - 1, t)
    expected_c = P(1, t)
    expected_h = P(1, t)
    assert special_denom(a, ba, bd, ca, cd, DE) == \
        (expected_a, expected_b, expected_c, expected_h)

    # A hypertangent case with a nontrivial special denominator:
    # t*Dq + t*q == (t - 2*t**2)/(t**2 + 1) has the solution
    # q == 1/(t**2 + 1), so with h == t**2 + 1, r == q*h == 1 must satisfy
    # A*Dr + B*r == C.
    A, B, C, h = special_denom(P(t, t), P(t, t), P(1, t),
        P(-2*t**2 + t, t), P(t**2 + 1, t), DE)
    assert (A, B, C, h) == (P(t, t), P(-2*t**2 + t, t),
        P(-2*t**2 + t, t), P(t**2 + 1, t))
    # Verify the contract explicitly (r == q*h == 1, so Dr == 0)
    assert C == B  # A*D(1) + B*1 == C

    # The possible-cancellation sharpening itself: a == 1, b == 2*t over
    # Dt == t**2 + 1 admits the solution q == 1/(t**2 + 1) of
    # Dq + 2*t*q == 0, so the bound must drop from 0 to -1 (the earlier
    # hypertangent cases enter the helper with n already at -1, which
    # cannot detect a broken adjustment).
    assert _special_denom_cancel_bound(P(1, t), P(2*t, t),
        P(1, t), 0, DE, 'tan') == -1

def test_bound_degree_fail() -> None:
    # Primitive
    DE = DifferentialExtension.manual([P(1, x),
        P(t0/x**2, t0), P(1/x, t)])
    assert bound_degree(P(t**2, t), P(-(1/x**2*t**2 + 1/x), t),
        P((2*x - 1)*t**4 + (t0 + x)/x*t**3 - (t0 + 4*x**2)/2*x*t**2 + x*t,
        t), DE) == 3


def test_bound_degree() -> None:
    # Base
    DE = DifferentialExtension.manual([P(1, x)])
    assert bound_degree(P(1, x), P(-2*x, x), P(1, x), DE) == 0

    # Primitive (see above test_bound_degree_fail)
    # TODO: Add test for when the degree bound becomes larger after limited_integrate
    # TODO: Add test for db == da - 1 case

    # Exp
    # TODO: Add tests
    # TODO: Add test for when the degree becomes larger after parametric_log_deriv()

    # Nonlinear
    DE = DifferentialExtension.manual([P(1, x), P(t**2 + 1, t)])
    assert bound_degree(P(t, t), P((t - 1)*(t**2 + 1), t), P(1, t), DE) == 0


def test_bound_degree_rational_z() -> None:
    # Primitive case with deg(a) == deg(b) and alpha == 1/(x*(x + 1)) ==
    # Dz/z for z == x/(x + 1) -- a proper ratio in k*, so derivation(z, DE)
    # inside bound_degree() needs the derivation of an element (z is not polynomial in the
    # tower generators; this used to crash with SympifyError).
    DE = DifferentialExtension.manual([P(1, x), P(1/x, t)], exts=['log'], extargs=[x])
    assert bound_degree(P(t, t),
        P(-t/(x*(x + 1)), t), P(1, t), DE) == 0
    # The same alpha end-to-end: Dy - y/(x*(x + 1)) == (x + 1 - t)/(x*(x + 1))
    # has the solution y == t (plus c*x/(x + 1) for any constant c, so check
    # the defining equation rather than the exact result).
    ya, yd = rischDE(P(-1, t), P(x*(x + 1), t),
        P(x + 1 - t, t), P(x*(x + 1), t), DE)
    y = ya.as_expr()/yd.as_expr()
    assert cancel(derivation(P(y, t), DE).as_expr() -
        y/(x*(x + 1)) - (x + 1 - t)/(x*(x + 1))) == 0


def test_bound_degree_undecidable() -> None:
    # Exp case with deg(a) == deg(b) and alpha == -lc(b)/lc(a) == 1/(x + 1):
    # deciding whether alpha == m*Dt/t + Dz/z requires log(x + 1), which is
    # not in the tower, so parametric_log_deriv() cannot decide and
    # bound_degree() must raise rather than return an unsound bound.
    # param_rischDE() propagates this (it used to guess n = 5 instead,
    # which could truncate the parametric solution basis and turn into a
    # false proof of nonelementarity downstream in is_deriv_in_field()).
    DE = DifferentialExtension.manual([P(1, x), P(1/x, t0),
        P(t1, t1)], exts=['log', 'exp'], extargs=[x, x])
    raises(NotImplementedError, lambda: bound_degree(P(t1, t1),
        P(-t1/(x + 1), t1), P(1, t1), DE))
    raises(NotImplementedError, lambda: bound_degree(P(t1, t1),
        P(-t1/(x + 1), t1), [P(1, t1)], DE, parametric=True))


def test_spde() -> None:
    DE = DifferentialExtension.manual([P(1, x), P(t**2 + 1, t)])
    raises(NonElementaryIntegralException, lambda: spde(P(t, t), P((t - 1)*(t**2 + 1), t), P(1, t), 0, DE))
    DE = DifferentialExtension.manual([P(1, x), P(t, t)])
    assert spde(P(t**2 + x*t*2 + x**2, t), P(t**2/x**2 + (2/x - 1)*t, t),
        P(t**2/x**2 + (2/x - 1)*t, t), 0, DE) == \
        (P(0, t), P(0, t), 0, P(0, t), P(1, t))
    DE = DifferentialExtension.manual([P(1, x), P(t0/x**2, t0), P(1/x, t)])
    assert spde(P(t**2, t), P(-t**2/x**2 - 1/x, t),
    P((2*x - 1)*t**4 + (t0 + x)/x*t**3 - (t0 + 4*x**2)/(2*x)*t**2 + x*t, t), 3, DE) == \
        (P(0, t), P(0, t), 0, P(0, t),
        P(t0*t**2/2 + x**2*t**2 - x**2*t, t))
    DE = DifferentialExtension.manual([P(1, x)])
    assert spde(P(x**2 + x + 1, x), P(-2*x - 1, x), P(x**5/2 +
    3*x**4/4 + x**3 - x**2 + 1, x), 4, DE) == \
        (P(0, x), P(x/2 - Rational(1, 4), x), 2, P(x**2 + x + 1, x), P(x*Rational(5, 4), x))
    # SymPy's test passes a symbolic bound n (the result -2 + n); the
    # bounds are integers here
    assert spde(P(x**2 + x + 1, x), P(-2*x - 1, x), P(x**5/2 +
    3*x**4/4 + x**3 - x**2 + 1, x), 10, DE) == \
        (P(0, x), P(x/2 - Rational(1, 4), x), 8, P(x**2 + x + 1, x), P(x*Rational(5, 4), x))
    DE = DifferentialExtension.manual([P(1, x), P(1, t)])
    raises(NonElementaryIntegralException, lambda: spde(P((t - 1)*(t**2 + 1)**2, t), P((t - 1)*(t**2 + 1), t), P(1, t), 0, DE))
    DE = DifferentialExtension.manual([P(1, x)])
    assert spde(P(x**2 - x, x), P(1, x), P(9*x**4 - 10*x**3 + 2*x**2, x), 4, DE) == \
        (P(0, x), P(0, x), 0, P(0, x), P(3*x**3 - 2*x**2, x))
    assert spde(P(x**2 - x, x), P(x**2 - 5*x + 3, x), P(x**7 - x**6 - 2*x**4 + 3*x**3 - x**2, x), 5, DE) == \
        (P(1, x), P(x + 1, x), 1, P(x**4 - x**3, x), P(x**3 - x**2, x))
    # n == oo (from rischDE() when bound_degree() cannot decide) must not
    # loop forever when deg(a) > 0: a == t, b == 1, c == x has no
    # solution and gcd(a, b) == 1 stays trivial, so no other exit is
    # ever taken.  This used to spin indefinitely.
    DE = DifferentialExtension.manual([P(1, x), P(t, t)])
    raises(NotImplementedError, lambda: spde(P(t, t), P(1, t),
        P(x, t), oo, DE))
    # ...while a solvable positive-degree case with n == oo still reduces
    # and returns (q == 1 here; c becomes zero after one pass)...
    assert spde(P(t, t), P(1, t), P(1, t), oo, DE) == \
        (P(0, t), P(0, t), 0, P(0, t), P(1, t))
    # an integer bound
    raises(NonElementaryIntegralException, lambda: spde(P(t, t),
        P(1, t), P(x, t), 1, DE))
    # ...and with deg(a) == 0 it returns immediately, so n == oo is fine.
    assert spde(P(2, t), P(t, t), P(t, t), oo, DE) == \
        (P(t/2, t), P(t/2, t), oo,
         P(1, t), P(0, t))

def test_solve_poly_rde_no_cancel() -> None:
    # deg(b) large
    DE = DifferentialExtension.manual([P(1, x), P(1 + t**2, t)])
    assert solve_poly_rde(P(t**2 + 1, t), P(t**3 + (x + 1)*t**2 + t + x + 2, t),
    oo, DE) == P(t + x, t)
    # deg(b) small
    DE = DifferentialExtension.manual([P(1, x)])
    assert solve_poly_rde(P(0, x), P(x/2 - Rational(1, 4), x), oo, DE) == \
        P(x**2/4 - x/4, x)
    DE = DifferentialExtension.manual([P(1, x), P(t**2 + 1, t)])
    assert solve_poly_rde(P(2, t), P(t**2 + 2*t + 3, t), 1, DE) == \
        P(t + 1, t)
    # The reduced equation Dy + b0*y == c0 is solved over the field k:
    # here y == 1/x
    assert solve_poly_rde(P(x, t), P(t**2 + x*t + 2 - 1/x**2, t), 1,
        DE) == \
        P(t + 1/x, t)
    # deg(b) == deg(D) - 1
    DE = DifferentialExtension.manual([P(1, x), P(t**2 + 1, t)])
    assert no_cancel_equal(P(1 - t, t),
    P(t**3 + t**2 - 2*x*t - 2*x, t), oo, DE) == \
        (P(t**2, t), 1, P((-2 - 2*x)*t - 2*x, t))
    # The m == 0 branch: Dq + t*q == t has the solution q == 1 (this used
    # to return an Expr instead of a Poly)
    assert no_cancel_equal(P(t, t), P(t, t), 5, DE) == P(1, t)
    # Immediate u == 0 return: -lc(b)/lc(Dt) == 3 is hit on the first
    # iteration, reducing to a smaller problem of degree at most 3
    assert no_cancel_equal(P(-3*t, t), P(t**2 + 1, t), oo, DE) == \
        (P(0, t), 3, P(t**2 + 1, t))


def test_solve_poly_rde_cancel() -> None:
    # exp
    DE = DifferentialExtension.manual([P(1, x), P(t, t)])
    assert cancel_exp(P(2*x, t), P(2*x, t), 0, DE) == \
        P(1, t)
    assert cancel_exp(P(2*x, t), P((1 + 2*x)*t, t), 1, DE) == \
        P(t, t)
    # The is_deriv_in_field() branch: b == (2*x + 1)/x == D(x)/x + 2*Dt/t
    # (parametric_log_deriv() gives (1, 2, x)), and Dq + b*q == c has the
    # solution q == 1 for c == (2*x + 1)/x.
    assert cancel_exp(P((2*x + 1)/x, t), P((2*x + 1)/x, t), 0, DE) == \
        P(1, t)
    # ... but 1/(2*x) also works for c == 1/x (the code finds solutions
    # with denominators via p/(z*t**m)):
    assert cancel_exp(P((2*x + 1)/x, t), P(1/x, t), 0, DE) == \
        P(1/(2*x), t)
    # c == 1/x**2 requires a' + 2*a == 1/x over QQ(x), which has no
    # rational solution (pole order mismatch at 0), so no solution exists
    raises(NonElementaryIntegralException, lambda: cancel_exp(
        P((2*x + 1)/x, t), P(1/x**2, t), 0, DE))

    # The t**(-m) stripping in the m < 0 branch is only valid when the
    # stripped coefficient is C/z for a constant C.  Here b == 1/x - 2 ==
    # D(x)/x - 2*Dt/t (z == x, m == -2), and the candidate q == 1 + t**2
    # has t**2-coefficient 1, with z*1 == x nonconstant, so no choice of
    # the integration constant removes the term: there is no solution
    # with n == 0.
    raises(NonElementaryIntegralException, lambda: cancel_exp(
        P(1/x - 2, t),
        P(1/x - 2 + t**2/x, t), 0, DE))
    # ...while the same b with c == b has the genuine solution q == 1.
    assert cancel_exp(P(1/x - 2, t),
        P(1/x - 2, t), 0, DE) == P(1, t)

    # primitive
    DE = DifferentialExtension.manual([P(1, x), P(1/x, t)])

    # If the DecrementLevel context manager is working correctly, this shouldn't
    # cause any problems with the further tests.
    raises(NonElementaryIntegralException, lambda: cancel_primitive(P(1, t), P(t, t), oo, DE))

    # b == 1/(2*x) has radical index 2 (2*b == Dx/x, so
    # is_log_deriv_k_t_radical_in_field() returns (2, x)); the polynomial
    # loop must keep the caller's degree bound n == 3 rather than
    # clobbering it with the radical index (which would wrongly raise
    # NonElementaryIntegralException for the solution q == t**3).
    b = P(1/(2*x), t)
    q0 = P(t**3, t)
    c = (derivation(q0, DE) + b*q0).to_field()
    assert cancel_primitive(b, c, 3, DE).as_expr() == t**3

    assert cancel_primitive(P(1, t), P(t + 1/x, t), 2, DE) == \
        P(t, t)
    assert cancel_primitive(P(4*x, t), P(4*x*t**2 + 2*t/x, t), 3, DE) == \
        P(t**2, t)

    # The is_deriv_in_field() branch: b == 1/x == D(x)/x
    # (is_log_deriv_k_t_radical_in_field() gives (1, x)), and
    # Dq + b*q == c has the solution q == t for c == (t + 1)/x.
    assert cancel_primitive(P(1/x, t), P((t + 1)/x, t), 1, DE) == \
        P(t, t)
    # z*c == t/(x + 1) has antiderivative t*log(x + 1) - log(x + 1) + x,
    # which is not in QQ(x, log(x)), so no solution exists
    raises(NonElementaryIntegralException, lambda: cancel_primitive(
        P(1/x, t), P(t/(x*(x + 1)), t), 1, DE))

    # The degree bound n must not be clobbered by the index returned from
    # is_log_deriv_k_t_radical_in_field(): b == 1/(2*x) has index 2
    # (2*b == D(x)/x), which used to replace n == 3 and wrongly reject
    # the degree 3 solution q == t**3 of Dq + b*q == c.
    b = P(Rational(1, 2)/x, t)
    q = P(t**3, t)
    c = derivation(q, DE) + b*q
    assert cancel_primitive(b, c, 3, DE) == q

    # The b == 0 cancellation case (Dq == c by in-field integration):
    # primitive (t == log(x)): D(t**2) == 2*t/x
    assert solve_poly_rde(P(0, t), P(2*t/x, t), 2, DE) == P(t**2, t)
    # (note Dq == 1/x has the in-field solution q == t == log(x); use an
    # integrand whose antiderivative needs log(x + 1), which is not in
    # the field)
    raises(NonElementaryIntegralException, lambda: solve_poly_rde(
        P(0, t), P(t/(x + 1), t), 2, DE))
    # exp (t == exp(x)): D(x*t) == (x + 1)*t
    DE = DifferentialExtension.manual([P(1, x), P(t, t)])
    assert solve_poly_rde(P(0, t), P((x + 1)*t, t), 1, DE) == \
        P(x*t, t)
    raises(NonElementaryIntegralException, lambda: solve_poly_rde(
        P(0, t), P(t/x, t), 1, DE))


def test_cancel_tan() -> None:
    # t = tan(x)
    DE = DifferentialExtension.manual([P(1, x), P(t**2 + 1, t)])
    # Example 6.6.1: Dq + (1 - t)*q == -2*(x + 1)*t - 2*x with n == 1
    # (b0 == 1) has the solution q == -2*x*t
    assert cancel_tan(P(1, t), P(-2*(x + 1)*t - 2*x, t), 1, DE) == \
        P(-2*x*t, t)
    # ... reached from solve_poly_rde() via no_cancel_equal()'s
    # hand-off (Example 6.5.3), and from rischDE(): equation (6.21),
    # Dy + (1 - t)*y == t**3 + t**2 - 2*x*t - 2*x, has the solution
    # y == t**2 - 2*x*t
    assert solve_poly_rde(P(1 - t, t), P(t**3 + t**2 - 2*x*t - 2*x, t),
        2, DE) == P(t**2 - 2*x*t, t)
    ya, yd = rischDE(P(1 - t, t), P(1, t),
        P(t**3 + t**2 - 2*x*t - 2*x, t), P(1, t), DE)
    assert cancel(ya.as_expr()/yd.as_expr()) == t**2 - 2*x*t
    # n == 0 with b0 != 0 is a Risch differential equation over k
    assert cancel_tan(P(1, t), P(x + 1, t), 0, DE) == P(x, t)
    # n == 0 with b0 == 0 is in-field integration in k
    assert cancel_tan(P(0, t), P(2*x, t), 0, DE) == P(x**2, t)
    raises(NonElementaryIntegralException, lambda: cancel_tan(
        P(0, t), P(1/x, t), 0, DE))
    # The book's pseudocode returns u + v*t unverified when n == 1,
    # which is a wrong answer here: the t**2 coefficient of
    # Dq + (1 - t)*q vanishes identically for deg(q) <= 1, so
    # c == t**2 + 1 has no solution, while the projected coupled
    # system is solvable (by (u, v) == (0, 0), so the book returns
    # q == 0 with Dq + (1 - t)*q == 0 != c)
    raises(NonElementaryIntegralException, lambda: cancel_tan(
        P(1, t), P(t**2 + 1, t), 1, DE))
    # the degree bound appears in the equation itself, so n == oo is
    # not allowed
    raises(ValueError, lambda: cancel_tan(P(1, t), P(1, t), oo, DE))
    # A nonconstant eta (Dt == x*(t**2 + 1)) with a constant
    # -lc(b)/lc(Dt) must pass the symbolic-ratio guard in
    # solve_poly_rde() (which used to test lc(b) itself):
    # Dq + (1 - x*t)*q == x*t**3 + t**2 + 2*x*t has the solution t**2
    DE = DifferentialExtension.manual([P(1, x),
        P(x*(t**2 + 1), t)])
    assert solve_poly_rde(P(1 - x*t, t), P(x*t**3 + t**2 + 2*x*t, t),
        2, DE) == P(t**2, t)
    # ... including when the ratio only becomes a number after
    # cancellation ((2*x + 2)/(x + 1) does not auto-simplify, so the
    # uncancelled dispatch comparison used to raise TypeError)
    DE = DifferentialExtension.manual([P(1, x),
        P((x + 1)*(t**2 + 1), t)])
    b = P(1 - (2*x + 2)*t, t)
    q0 = P(t**2, t)
    c = derivation(q0, DE) + b*q0
    assert solve_poly_rde(b, c, 2, DE) == q0
    # A ratio that is a nonconstant element of k ((x**2 - 1)/(x - 1)
    # == x + 1) never equals an integer degree, so no cancellation
    # occurs and no_cancel_equal() decides the equation (this also
    # used to raise TypeError from the dispatch comparison)
    DE = DifferentialExtension.manual([P(1, x),
        P((x - 1)*(t**2 + 1), t)])
    b = P(1 - (x**2 - 1)*t, t)
    q0 = P(t, t)
    c = derivation(q0, DE) + b*q0
    assert solve_poly_rde(b, c, 1, DE) == q0
    # A numeric ratio that is not a positive integer (here 3/2) can
    # never be a cancellation degree, so no_cancel_equal() decides
    # the equation even with n below the ratio (this used to fall
    # into the cancellation dispatch and raise NotImplementedError)
    DE = DifferentialExtension.manual([P(1, x),
        P(t**2 + 1, t)])
    b = P(1 - Rational(3, 2)*t, t)
    q0 = P(t, t)
    c = derivation(q0, DE) + b*q0
    assert solve_poly_rde(b, c, 1, DE).as_expr() == t
    # A positive-integer ratio above the degree bound is likewise
    # unattainable: Dq + (1 - 2*t)*q == -t**2 + t + 1 with n == 1 has
    # the solution t (this used to raise NotImplementedError from the
    # cancellation dispatch, and no_cancel_equal() used to turn it
    # into a false no-solution proof)
    b = P(1 - 2*t, t)
    assert solve_poly_rde(b, P(-t**2 + t + 1, t), 1, DE).as_expr() == t
    # A non-real numeric ratio (here sqrt(-1)) admits no ordered
    # comparison with n but can never be a cancellation degree
    b = P(1 - I*t, t)
    q0 = P(t, t)
    c = derivation(q0, DE) + b*q0
    assert solve_poly_rde(b, c, 1, DE).as_expr().expand() == t
    # A parameter-mixed ratio whose tower-dependent part survives
    # every specialization (x + k: the coefficient of x is 1) is
    # accepted...
    b = P(1 - (x + k)*t, t)
    q0 = P(t, t)
    c = derivation(q0, DE) + b*q0
    assert solve_poly_rde(b, c, 1, DE).as_expr() == t
    # ... while a ratio depending only on a parameter of the constant
    # field is still refused (the cancellation degree would be
    # parameter-dependent, making no_cancel_equal()'s divisions only
    # generically valid)
    raises(TypeError, lambda: solve_poly_rde(P(1 - k*t, t),
        P(1, t), 2, DE))


def test_rischDE() -> None:
    # TODO: Add more tests for rischDE, including ones from the text
    DE = DifferentialExtension.manual([P(1, x), P(t, t)])
    DE.decrement_level()
    assert rischDE(P(-2*x, x), P(1, x), P(1 - 2*x - 2*x**2, x),
    P(1, x), DE) == \
        (P(x + 1, x), P(1, x))

    # See issue 28407
    DE = DifferentialExtension.manual([P(1, x), P(t, t)])
    fa = P(-t + 1, t)
    fd = P(1, t)
    ga = P(1, t)
    gd = P(1, t)
    assert rischDE(fa, fd, ga, gd, DE) == (P(-1, t), P(t, t))

    # A derivation with an 'other_linear' monomial raises a graceful
    # NotImplementedError (it used to be an uncaught ValueError)
    DE = DifferentialExtension.manual([P(1, x), P(t + 1, t)])
    raises(NotImplementedError, lambda: rischDE(P(1, t), P(1, t),
        P(t, t), P(1, t), DE))

    # Dy + y/x == 1.  f == 1/x is not weakly normalized; the weak normalizer
    # q == x must be applied to g and divided back out of the solution.
    # rischDE() used to discard it and return y == x, which solves
    # Dy + y/x == 2, not 1.  The correct answer here is y == x/2 (plus any
    # multiple of the homogeneous solution 1/x).
    DE = DifferentialExtension.manual([P(1, x)])
    ya, yd = rischDE(P(1, x), P(x, x), P(1, x), P(1, x), DE)
    assert (ya, yd) == (P(x**2/2, x), P(x, x))
    # Explicitly verify Dy + f*y == g for the returned solution
    y = ya.as_expr()/yd.as_expr()
    assert cancel(y.diff(x) + y/x - 1) == 0

from __future__ import annotations

import mpmath

from sympy import (mobius, totient, divisor_sigma, log, zeta, pi, re, S, Derivative, N, Rational,
    Symbol, Float)
from sympy.core.expr import Expr
from sympy.abc import n, s

from sympy_extras.concrete.dirichlet import dirichlet_series


def _partial(term: Expr, count: int) -> Float:
    """The partial sum with ``count`` terms, exactly then as a float."""
    total = S.Zero
    for k in range(1, count + 1):
        total += term.subs(n, k)
    return Float(N(total, 20), 15)


def test_closed_forms() -> None:
    assert dirichlet_series(1/n**s, n) == (zeta(s), re(s) > 1)
    assert dirichlet_series(mobius(n)/n**s, n) == (1/zeta(s), re(s) > 1)
    assert dirichlet_series(mobius(n)**2/n**s, n) == (zeta(s)/zeta(2*s), re(s) > 1)
    assert dirichlet_series(totient(n)/n**s, n) == (zeta(s - 1)/zeta(s), re(s) > 2)
    assert dirichlet_series(divisor_sigma(n)/n**s, n) == (zeta(s)*zeta(s - 1), re(s) > 2)
    assert dirichlet_series(divisor_sigma(n, 2)/n**s, n) == (zeta(s)*zeta(s - 2), re(s) > 3)
    assert dirichlet_series(divisor_sigma(n, 0)/n**s, n) == (zeta(s)**2, re(s) > 1)
    assert dirichlet_series((-1)**(n + 1)/n**s, n) == ((1 - 2**(1 - s))*zeta(s), re(s) > 0)
    assert dirichlet_series((-1)**n/n**s, n) == (-(1 - 2**(1 - s))*zeta(s), re(s) > 0)
    assert dirichlet_series(log(n)/n**s, n) == (-Derivative(zeta(s), s), re(s) > 1)
    assert dirichlet_series(log(n)**2/n**s, n) == (Derivative(zeta(s), (s, 2)), re(s) > 1)
    assert dirichlet_series(3*mobius(n)*n**(-s), n) == (3/zeta(s), re(s) > 1)
    # numerical exponents and a later start
    assert dirichlet_series((-1)**(n + 1)/n**2, n) == (pi**2/12, S.true)
    assert dirichlet_series(mobius(n)/n**2, n, 2) == (6/pi**2 - 1, S.true)
    assert dirichlet_series(1/(n**s + 1), n) is None
    assert dirichlet_series(mobius(n)*totient(n)/n**s, n) is None
    assert dirichlet_series(Symbol('c')**n/n**s, n) is None


def test_against_partial_sums() -> None:
    # the tails are O(N**-3) or smaller: the partial sums with 400 terms
    # agree with the closed forms to about 6 digits
    cases = [(mobius(n)/n**4, 1/zeta(4)), (mobius(n)**2/n**4, zeta(4)/zeta(8)),
             (totient(n)/n**5, zeta(4)/zeta(5)), (divisor_sigma(n)/n**5, zeta(5)*zeta(4)),
             (divisor_sigma(n, 0)/n**4, zeta(4)**2), ((-1)**(n + 1)/n**3, Rational(3, 4)*zeta(3))]
    for term, expected in cases:
        found = dirichlet_series(term, n)
        assert found is not None and found[0] == expected
        assert abs(_partial(term, 400) - N(expected, 15)) < 1e-6
    # the derivative of zeta is returned for a symbolic exponent only
    # (SymPy cannot evaluate it numerically); the identity is checked
    # against mpmath
    assert dirichlet_series(log(n)/n**4, n) is None
    expected = -Float(mpmath.zeta(4, derivative=1), 15)
    assert abs(_partial(log(n)/n**4, 400) - expected) < 1e-5

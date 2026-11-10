from __future__ import annotations

import mpmath

from sympy_extras.core import Basic, Derivative, Eq, Expr, Integer, Rational, Symbol, add, gt, pi, true
from sympy_extras.functions import divisor_sigma, log, mobius, re, totient, zeta
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.concrete.dirichlet import dirichlet_series

n, s = Symbol('n'), Symbol('s')


def _ex(b: Basic) -> Expr:
    assert isinstance(b, Expr)
    return b


def _float(e: Expr) -> float:
    return float(complex(e.evalf(20)).real)


def _partial(term: Expr, count: int) -> float:
    """The partial sum with ``count`` terms, exactly then as a float."""
    return _float(add(*[_ex(term.subs(n, Integer(k))) for k in range(1, count + 1)]))


def test_closed_forms() -> None:
    assert dirichlet_series(1/n**s, n) == (zeta(s), gt(re(s), 1))
    assert dirichlet_series(mobius(n)/n**s, n) == (1/zeta(s), gt(re(s), 1))
    assert dirichlet_series(mobius(n)**2/n**s, n) == (zeta(s)/zeta(2*s), gt(re(s), 1))
    assert dirichlet_series(totient(n)/n**s, n) == (zeta(s - 1)/zeta(s), gt(re(s), 2))
    assert dirichlet_series(divisor_sigma(n)/n**s, n) == (zeta(s)*zeta(s - 1), gt(re(s), 2))
    assert dirichlet_series(divisor_sigma(n, 2)/n**s, n) == (zeta(s)*zeta(s - 2), gt(re(s), 3))
    assert dirichlet_series(divisor_sigma(n, 0)/n**s, n) == (zeta(s)**2, gt(re(s), 1))
    assert dirichlet_series((-1)**(n + 1)/n**s, n, special_values=False) == ((1 - 2**(1 - s))*zeta(s),
                                                                             gt(re(s), 0))
    assert dirichlet_series((-1)**n/n**s, n, special_values=False) == (-(1 - 2**(1 - s))*zeta(s), gt(re(s), 0))
    assert dirichlet_series(log(n)/n**s, n) == (-Derivative(zeta(s), s), gt(re(s), 1))
    assert dirichlet_series(log(n)**2/n**s, n) == (Derivative(zeta(s), (s, 2)), gt(re(s), 1))
    assert dirichlet_series(3*mobius(n)*n**(-s), n) == (3/zeta(s), gt(re(s), 1))
    # numerical exponents and a later start
    assert dirichlet_series((-1)**(n + 1)/n**2, n) == (pi**2/12, true)
    assert dirichlet_series(mobius(n)/n**2, n, 2) == (6/pi**2 - 1, true)
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
        assert abs(_partial(term, 400) - _float(expected)) < 1e-6
    # the derivative of zeta is returned for a symbolic exponent only; the
    # identity is checked against mpmath
    assert dirichlet_series(log(n)/n**4, n) is None
    derivative = -float(mpmath.zeta(4, derivative=1))
    assert abs(_partial(log(n)/n**4, 400) - derivative) < 1e-5


def test_isolated_values_of_the_parameters() -> None:
    # the bug: the alternating series was (1 - 2**(1 - s))*zeta(s), 0*zoo
    # at s = 1, where it converges to log(2)
    found = dirichlet_series((-1)**(n + 1)/n**s, n)
    assert found is not None
    value, condition = found
    assert condition == gt(re(s), 0)
    assert isinstance(value, Piecewise) and value.args[0].args == (log(2), Eq(s, 1))
    expected = float(mpmath.nsum(lambda j: (-1)**(j + 1)/j, [1, mpmath.inf]))
    assert abs(_float(_ex(value.subs(s, 1))) - expected) < 1e-12
    found = dirichlet_series((-1)**n/n**s, n, 2)
    assert found is not None
    assert found[0].subs(s, 1) == 1 - log(2)
    # the points outside the region of convergence get no case: zeta(1) in
    # 1/zeta(s) (at s = 1, re(s) > 1 is false) and in zeta(s - 1)/zeta(s)
    assert dirichlet_series(mobius(n)/n**s, n) == (1/zeta(s), gt(re(s), 1))
    assert dirichlet_series(totient(n)/n**s, n) == (zeta(s - 1)/zeta(s), gt(re(s), 2))

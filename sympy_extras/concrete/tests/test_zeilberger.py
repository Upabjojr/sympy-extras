from __future__ import annotations

from sympy import symbols, binomial, factorial, Function, Eq, simplify, combsimp, oo, N
from sympy.testing.pytest import raises

from sympy_extras.concrete.zeilberger import zeilberger, wz_certificate, wz_prove, zeilberger_sum, Telescoper
from sympy_extras._testing import untyped
from sympy_extras._typing import as_expr

n, k = symbols('n k', integer=True)
x = symbols('x')


def _same(a: object, b: object) -> bool:
    difference = as_expr(a) - as_expr(b)
    if simplify(combsimp(difference)) == 0:
        return True
    # numerically at a few points (the symbolic simplifiers miss some gamma identities)
    return all(abs(N(difference.subs(n, i), 30)) < 1e-20 for i in range(0, 6))


def test_zeilberger() -> None:
    Z = zeilberger(binomial(n, k), n, k)
    assert isinstance(Z, Telescoper) and Z.order == 1
    assert Z.coefficients == [-2, 1] and Z.check()
    Z = zeilberger(binomial(n, k)**2, n, k)
    assert Z is not None and Z.check()
    assert _same(Z.coefficients[0]/Z.coefficients[1], -2*(2*n + 1)/(n + 1))
    Z = zeilberger((-1)**k*binomial(2*n, k)**3, n, k)
    assert Z is not None and Z.check()
    Z = zeilberger(binomial(n, k)**2*binomial(n + k, k)**2, n, k)
    assert Z is not None and Z.order == 2 and Z.check()
    # a term with a free parameter
    Z = zeilberger(binomial(n, k)*x**k, n, k)
    assert Z is not None and Z.check()
    assert _same(Z.coefficients[0]/Z.coefficients[1], -(x + 1))
    assert zeilberger(binomial(n, k), n, k, max_order=0) is None
    raises(ValueError, lambda: zeilberger(binomial(n, k)**k, n, k))
    raises(TypeError, lambda: untyped(zeilberger)(n > k, n, k))


def test_wz() -> None:
    R = wz_certificate(binomial(n, k), 2**n, n, k)
    assert R is not None and _same(R, k/(2*(k - n - 1)))
    assert wz_prove(binomial(n, k), 2**n, n, k) is True
    assert wz_prove(binomial(n, k)**2, binomial(2*n, n), n, k) is True
    assert wz_prove(binomial(n, k)*binomial(x, k), binomial(n + x, n), n, k) is True
    assert wz_prove((-1)**k*binomial(2*n, k)**3, (-1)**n*factorial(3*n)/factorial(n)**3, n, k) is True
    # a wrong closed form has no certificate
    assert wz_certificate(binomial(n, k), 3**n, n, k) is None
    assert wz_prove(binomial(n, k), 2**n + 1, n, k) is None


def test_zeilberger_sum() -> None:
    assert zeilberger_sum(binomial(n, k), (k, 0, n)) == 2**n
    assert zeilberger_sum(binomial(n, k)**2, (k, 0, n)) == binomial(2*n, n)
    assert _same(zeilberger_sum((-1)**k*binomial(2*n, k)**3, (k, 0, 2*n)), (-1)**n*factorial(3*n)/factorial(n)**3)
    assert _same(zeilberger_sum(k*binomial(n, k), (k, 0, n)), n*2**(n - 1))
    assert _same(zeilberger_sum(binomial(n, k)*x**k, (k, 0, n)), (x + 1)**n)
    # sums with no hypergeometric closed form: the recurrence
    result = zeilberger_sum(binomial(n, k)**2*binomial(n + k, k)**2, (k, 0, n))
    assert isinstance(result, Eq) and result.has(Function('S')(n + 2))
    # infinite upper limit with vanishing terms
    assert zeilberger_sum(binomial(n, k), (k, 0, oo)) == 2**n
    assert zeilberger_sum(1/(k*(k + 1)), (k, 1, n)) is not None


def test_polynomial_times_factorial_does_not_crash() -> None:
    # sympy-extras#43: SymPy's rsolve raised AttributeError on the
    # recurrence for this summand, and it escaped zeilberger_sum; the
    # sum has no hypergeometric closed form, so None (or a recurrence)
    # is the acceptable answer, a crash is not.
    from sympy import symbols, factorial, Sum
    from sympy.core.function import AppliedUndef
    n, k = symbols('n k', integer=True)
    result = zeilberger_sum((2*k**2 + k)*factorial(k), (k, 0, n), n)
    # a recurrence in S(n) is a legitimate answer; a closed form must be right
    if result is not None and not result.has(Sum) and not result.atoms(AppliedUndef):
        for value in range(0, 6):
            direct = sum((2*j**2 + j)*factorial(j) for j in range(0, value + 1))
            assert result.subs(n, value) == direct


def test_zeilberger_sum_never_returns_a_boolean() -> None:
    # sympy-extras#33: for this summand zeilberger_sum returned the
    # Boolean False -- a rejected check re-asserted as an answer -- where
    # the sum is a well defined integer for every n.
    from sympy import binomial, symbols, Sum
    n, k = symbols('n k', integer=True)
    term = (-1)**k*binomial(n, k)*binomial(2*n - 2*k, n)
    from sympy.core.function import AppliedUndef
    from sympy import nan
    result = zeilberger_sum(term, (k, 0, n), n)
    assert result not in (True, False)
    assert result is None or not result.has(nan)
    # a recurrence in S(n) is a legitimate answer; a closed form must be right
    if result is not None and not result.has(Sum) and not result.atoms(AppliedUndef):
        for value in range(1, 6):
            assert result.subs(n, value) == Sum(term.subs(n, value), (k, 0, value)).doit(), value

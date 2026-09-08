from __future__ import annotations

from sympy import symbols, Rational, cancel, simplify, factor, Sum, N, S
from sympy.testing.pytest import raises

from sympy_extras.concrete.qhyper import (QPochhammer, qpochhammer, qbinomial, q_ratio, qgosper_term,
    qgosper_sum, qzeilberger, normal_in)
from sympy_extras._typing import as_expr

q, k, n, x, a = symbols('q k n x a')


def test_qpochhammer_and_qbinomial() -> None:
    assert QPochhammer(a, q, 0) == 1
    assert QPochhammer(a, q, 3) == (1 - a)*(1 - a*q)*(1 - a*q**2)
    assert qpochhammer(a, q, k) == QPochhammer(a, q, k)
    assert factor(qbinomial(S(4), S(2), q)) == (q**2 + 1)*(q**2 + q + 1)
    assert cancel(qbinomial(S(5), S(2), q) - qbinomial(S(5), S(3), q)) == 0


def test_q_ratio() -> None:
    data = q_ratio(q**k, k, q)
    assert data is not None and data[0] == q
    data = q_ratio(q**(k*(k - 1)/2), k, q)
    assert data is not None and data[0] == data[1]
    data = q_ratio(qpochhammer(a, q, k), k, q)
    assert data is not None and data[0] == 1 - a*data[1]
    data = q_ratio(1/qpochhammer(q, q, n - k), k, q)
    assert data is not None and cancel(data[0] - (1 - q**n/data[1])) == 0
    assert q_ratio(k*q**k, k, q) is None
    assert q_ratio(qpochhammer(a, q, k**2), k, q) is None


def _check_sum(term: object, lo: int, hi_values: list[int], q_value: Rational) -> None:
    t = as_expr(term)
    closed = qgosper_sum(t, (k, lo, n), q)
    assert closed is not None
    for value in hi_values:
        direct = Sum(t, (k, lo, value)).doit().subs(q, q_value)
        assert simplify(closed.subs({n: value, q: q_value}) - direct) == 0, (term, value)


def test_qgosper() -> None:
    z = qgosper_term(q**k, k, q)
    assert z is not None and normal_in(as_expr(z.subs(k, k + 1) - z - q**k), k, q) == 0
    _check_sum(q**k, 0, [0, 1, 4], Rational(1, 3))
    _check_sum(q**(2*k), 1, [1, 3], Rational(2, 5))
    # a telescoping sum with a q-Pochhammer symbol, checked numerically
    t = q**k*qpochhammer(a, q, k)
    closed = qgosper_sum(t, (k, 0, n), q)
    assert closed is not None
    for value in (0, 1, 3):
        point = {q: Rational(1, 3), a: Rational(1, 2)}
        direct = Sum(t, (k, 0, value)).doit().subs(point)
        assert simplify(closed.subs({n: value}).subs(point) - direct) == 0
    raises(ValueError, lambda: qgosper_term(k*q**k, k, q))


def test_qzeilberger() -> None:
    # the q-binomial theorem: sum_k [n k] q**(k(k-1)/2) x**k = (-x; q)_n
    t = qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k
    Z = qzeilberger(t, n, k, q)
    assert Z is not None and Z.order == 1
    assert normal_in(as_expr(Z.coefficients[0]/Z.coefficients[1] + 1 + q**n*x), n, q) == 0
    # the recurrence and the certificate, checked numerically at a point
    values = {q: Rational(1, 3), x: Rational(2, 7), n: 3}
    G = Z.certificate*t
    lhs = sum((a_j*t.subs(n, n + j) for j, a_j in enumerate(Z.coefficients)), as_expr(0))
    for kv in range(0, 3):
        difference = (lhs - G.subs(k, k + 1) + G).subs(values).subs(k, kv)
        assert abs(N(difference, 20)) < 1e-15
    # sum_k [n k] (-1)**k q**(k(k-1)/2) = (1; q)_n = 0 for n >= 1: the summand itself
    # telescopes, which the algorithm reports as the relation with a_1 = 0
    t2 = qbinomial(n, k, q)*(-1)**k*q**(k*(k - 1)/2)
    Z2 = qzeilberger(t2, n, k, q, max_order=1)
    assert Z2 is not None and Z2.coefficients == [1, 0]
    G2 = Z2.certificate*t2
    for kv in range(0, 3):
        difference = (t2 - G2.subs(k, k + 1) + G2).subs({q: Rational(1, 3), n: 3}).subs(k, kv)
        assert abs(N(difference, 20)) < 1e-15

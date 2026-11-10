from __future__ import annotations

import sympy
from sympy.testing.pytest import raises

from sympy_extras._typing import as_expr, as_symbol
from sympy_extras.concrete.qhyper import (QPochhammer, qpochhammer, qbinomial, q_ratio, qgosper_term,
    qgosper_sum, qzeilberger, normal_in)
from sympy_extras.core import Basic, Dummy, Eq, Expr, Integer, Rational, Symbol, add
from sympy_extras.functions import Piecewise
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.polys.polytools import cancel, factor

q, k, n, x, a, b = (Symbol(s) for s in 'qknxab')
ZERO = Integer(0)


def _ex(e: Basic) -> Expr:
    assert isinstance(e, Expr), e
    return e


def _sympy(e: Expr) -> sympy.Expr:
    return as_expr(to_sympy(e))


def _direct(t: Expr, lo: int, hi: int, point: dict[Basic, Basic]) -> Expr:
    """``sum_{k=lo}^{hi} t`` at ``point``, term by term."""
    return add(*[_ex(_ex(t.subs(k, Integer(i))).subs(point)) for i in range(lo, hi + 1)])


def test_qpochhammer_and_qbinomial() -> None:
    assert qpochhammer(a, q, 0) == 1
    assert qpochhammer(a, q, 3) == (1 - a)*(1 - a*q)*(1 - a*q**2)
    assert qpochhammer(a, q, k) == QPochhammer(a, q, k)
    # the node never evaluates, the function does
    assert isinstance(QPochhammer(a, q, 2), QPochhammer) and QPochhammer(a, q, 2).doit() == (1 - a)*(1 - a*q)
    assert cancel(qpochhammer(a, q, -2)*(1 - a/q)*(1 - a/q**2)) == 1
    assert factor(qbinomial(4, 2, q)) == (q**2 + 1)*(q**2 + q + 1)
    assert cancel(qbinomial(5, 2, q) - qbinomial(5, 3, q)) == 0
    # printed as the SymPy function was
    assert str(QPochhammer(a, q, n - k)) == 'QPochhammer(a, q, -k + n)'


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


def _check_sum(t: Expr, lo: int, hi_values: list[int], q_value: Rational) -> None:
    closed = qgosper_sum(t, (k, lo, n), q)
    assert closed is not None
    for value in hi_values:
        direct = _direct(t, lo, value, {q: q_value})
        assert cancel(_ex(closed.subs({n: Integer(value), q: q_value})) - direct) == 0, (t, value)


def test_qgosper() -> None:
    z = qgosper_term(q**k, k, q, special_values=False)
    assert z is not None and normal_in(_ex(z.subs(k, k + 1)) - z - q**k, k, q) == 0
    _check_sum(q**k, 0, [0, 1, 4], Rational(1, 3))
    _check_sum(q**(2*k), 1, [1, 3], Rational(2, 5))
    # a telescoping sum with a q-Pochhammer symbol, checked at a point
    t = q**k*qpochhammer(a, q, k)
    closed = qgosper_sum(t, (k, 0, n), q)
    assert closed is not None
    point: dict[Basic, Basic] = {q: Rational(1, 3), a: Rational(1, 2)}
    for value in (0, 1, 3):
        direct = _direct(t, 0, value, point)
        assert cancel(_ex(_ex(closed.subs({n: Integer(value)})).subs(point)) - direct) == 0
    raises(ValueError, lambda: qgosper_term(k*q**k, k, q))


def test_qzeilberger() -> None:
    # the q-binomial theorem: sum_k [n k] q**(k(k-1)/2) x**k = (-x; q)_n
    t = qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k
    Z = qzeilberger(t, n, k, q)
    assert Z is not None and Z.order == 1
    assert normal_in(Z.coefficients[0]/Z.coefficients[1] + 1 + q**n*x, n, q) == 0
    # the recurrence and the certificate, checked exactly at a point
    values: dict[Basic, Basic] = {q: Rational(1, 3), x: Rational(2, 7), n: Integer(3)}
    G = Z.certificate*t
    lhs = add(*[a_j*_ex(t.subs(n, n + j)) for j, a_j in enumerate(Z.coefficients)])
    for kv in range(0, 3):
        difference = _ex(_ex((lhs - _ex(G.subs(k, k + 1)) + G).subs(values)).subs(k, Integer(kv)))
        assert cancel(difference) == 0
    # sum_k [n k] (-1)**k q**(k(k-1)/2) = (1; q)_n = 0 for n >= 1: the summand itself
    # telescopes, which the algorithm reports as the relation with a_1 = 0
    t2 = qbinomial(n, k, q)*(-1)**k*q**(k*(k - 1)/2)
    Z2 = qzeilberger(t2, n, k, q, max_order=1)
    assert Z2 is not None and Z2.coefficients == [1, 0]
    G2 = Z2.certificate*t2
    for kv in range(0, 3):
        difference = _ex(_ex((t2 - _ex(G2.subs(k, k + 1)) + G2).subs({q: Rational(1, 3), n: Integer(3)}))
                         .subs(k, Integer(kv)))
        assert cancel(difference) == 0
    assert str(Z.recurrence()) == '(-q**n*x - 1)*S(n) + S(n + 1)'


def test_isolated_values_of_the_parameters() -> None:
    from sympy_extras.concrete.tests.cases import check_antidifference_cases, check_sum_cases
    qs, ks, ns = as_symbol(to_sympy(q)), as_symbol(to_sympy(k)), as_symbol(to_sympy(n))
    # the bug: the sum of q**(2*k) was (q**(2*n + 2) - 1)/((q - 1)*(q + 1)),
    # undefined at q = 1 and q = -1, where the sum is n + 1
    value = qgosper_sum(q**(2*k), (k, 0, n), q)
    assert value is not None
    assert check_sum_cases(_sympy(value), _sympy(q**(2*k)), ks, 0, ns, ns) == 2
    assert value.subs(q, 1) == n + 1 and value.subs(q, -1) == n + 1
    term = qgosper_term(q**k, k, q)
    assert term is not None and check_antidifference_cases(_sympy(term), _sympy(q**k), ks) == 1
    assert not isinstance(qgosper_sum(q**k, (k, 0, n), q, special_values=False), Piecewise)
    # a case inside a case: (1 - (a; q)_{n+1})/a at a = 0, the sum of q**k,
    # which has its own at q = 1
    t = q**k*qpochhammer(a, q, k)
    value = qgosper_sum(t, (k, 0, n), q)
    assert value is not None
    assert check_sum_cases(_sympy(value), _sympy(t), ks, 0, ns, ns) == 2
    assert isinstance(value, Piecewise) and value.args[0].args == (n + 1, Eq(a, 0) & Eq(q, 1))
    # the closed form at q = 1, (1 - (a; 1)_{n+1})/a, is defined: no case
    # q = 1 for a generic a
    assert len(value.args) == 3 and value.args[1].args[1] == Eq(a, 0)
    assert qs in _sympy(value).free_symbols


def test_undefined_antidifference_at_a_limit() -> None:
    # the bug: nan was returned as the sum of qbinomial(k, 2, q)*q**k from
    # k = 0, the antidifference being 0*zoo at k = 0 ((q; q)_{k - 2} is
    # infinite there)
    assert qgosper_sum(qbinomial(k, 2, q)*q**k, (k, 0, n), q) is None
    # (0; q)_k is 1, which the case a = 0 above needs
    assert qpochhammer(0, q, k) == 1


def test_no_unknown_left_in_the_sum() -> None:
    # the bug: the sum of q**k/((1 - q**k)*(1 - q**(k + 1))) contained the
    # unknown _f0 of the q-Gosper equation, a coefficient of f left free by
    # the linear system (a solution of the homogeneous equation), which was
    # not set to 0
    t = q**k/((1 - q**k)*(1 - q**(k + 1)))
    value = qgosper_sum(t, (k, 1, n), q, special_values=False)
    assert value is not None and not value.atoms(Dummy)
    for hi in (1, 2, 4):
        point: dict[Basic, Basic] = {q: Rational(1, 3)}
        assert cancel(_ex(_ex(value.subs(n, Integer(hi))).subs(point)) - _direct(t, 1, hi, point)) == 0


def test_interop_of_the_symbol() -> None:
    from sympy_extras.concrete import qhyper_sympy
    converted = to_sympy(QPochhammer(a, q, k))
    assert isinstance(converted, qhyper_sympy.QPochhammer)
    assert from_sympy(converted) == QPochhammer(a, q, k)
    # the SymPy function evaluates on conversion, as SymPy's functions do
    assert to_sympy(QPochhammer(a, q, 2)) == (1 - sympy.Symbol('a'))*(1 - sympy.Symbol('a')*sympy.Symbol('q'))


def test_sympy_face() -> None:
    from sympy_extras.concrete import qhyper_sympy as face
    qs, ks, ns, xs = sympy.symbols('q k n x')
    assert str(face.qgosper_sum(qs**ks, (ks, 0, ns), qs)) == \
        'Piecewise((n + 1, Eq(q, 1)), (q**(n + 1)/(q - 1) - 1/(q - 1), True))'
    Z = face.qzeilberger(face.qbinomial(ns, ks, qs)*qs**(ks*(ks - 1)/2)*xs**ks, ns, ks, qs)
    assert Z is not None and Z.coefficients == [-qs**ns*xs - 1, 1]
    assert str(face.qgosper_term(qs**ks, ks, qs, special_values=False)) == 'q**k/(q - 1)'
    assert sympy.factor(face.qbinomial(4, 2, qs)) == (qs**2 + 1)*(qs**2 + qs + 1)
    assert face.QPochhammer(0, qs, ks) == 1

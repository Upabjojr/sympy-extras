from __future__ import annotations

import random
from typing import Optional, Sequence

from sympy import (Expr, Integer, Piecewise, Rational, Sum, Symbol, binomial, expand_func, harmonic, simplify,
    symbols)

from sympy_extras._typing import as_expr
from sympy_extras.concrete import creative_telescoping, definite_sum, summation

n, k, a, b, x = symbols('n k a b x')


def _direct(f: Expr, lower: Expr, upper: Expr, value: int) -> Expr:
    """The sum at n = value, term by term."""
    lo, hi = int(lower.subs(n, value)), int(upper.subs(n, value))
    return as_expr(sum((f.subs(n, value).subs(k, i).doit() for i in range(lo, hi + 1)), Integer(0)))


def _assert_ordered_sums(value: Expr, at: int) -> None:
    """Every unevaluated ``Sum`` in ``value`` has an upper limit at least
    its lower limit minus one (an empty sum at most)."""
    for node in value.atoms(Sum):
        assert isinstance(node, Sum)
        for limit in node.limits:
            length = as_expr(limit[2] - limit[1] + 1)
            assert not (length.is_number and length < 0), (node, at)


def _check(result: Optional[Expr], f: Expr, lower: Expr, upper: Expr, values: Sequence[int] = range(0, 9),
           parameters: Optional[dict[Symbol, Expr]] = None) -> None:
    """The closed form agrees with the direct sum for several n (at
    rational values of the other parameters), and its unevaluated sums
    have ordered limits there: SymPy's ``doit`` reads ``Sum(t, (j, 2,
    0))`` by Karr's convention, as ``-t(1)``, where the empty-sum reading
    (Mathematica's) gives 0."""
    assert result is not None, f
    for value in values:
        _assert_ordered_sums(as_expr(result.subs(n, value)), value)
        direct = _direct(f, lower, upper, value)
        closed = as_expr(result.subs(n, value).doit())
        difference = as_expr(direct - closed)
        if parameters:
            difference = as_expr(difference.xreplace(parameters))
        assert simplify(expand_func(difference)) == 0, (f, value)


def test_creative_telescoping_hypergeometric() -> None:
    Z = creative_telescoping(binomial(n, k), k, n)
    assert Z is not None and Z.order == 1 and Z.coefficients == [-2, 1] and Z.check()
    Z = creative_telescoping(binomial(n, k)**2, k, n)
    assert Z is not None and Z.order == 1 and Z.check()
    assert simplify(Z.coefficients[0]/Z.coefficients[1] + 2*(2*n + 1)/(n + 1)) == 0
    assert creative_telescoping(binomial(n, k), k, n, order=0) is None
    Z = creative_telescoping(binomial(n, k)**2*binomial(n + k, k)**2, k, n, order=2)
    assert Z is not None and Z.order == 2 and Z.check()


def test_hypergeometric_identities() -> None:
    cases: list[tuple[Expr, Expr, Expr]] = [
        (binomial(n, k), Integer(0), n),
        (binomial(n, k)**2, Integer(0), n),
        (k*binomial(n, k), Integer(0), n),
        (binomial(n, k)*x**k, Integer(0), n),
        ((-1)**k*binomial(2*n, k)**2, Integer(0), 2*n),
        ((-1)**(k + 1)*binomial(n, k)/k, Integer(1), n),
        (binomial(2*n, k), Integer(0), n),
    ]
    for f, lower, upper in cases:
        _check(definite_sum(f, (k, lower, upper)), f, lower, upper, parameters={x: Rational(3, 7)})
    assert definite_sum(binomial(n, k), (k, 0, n)) == 2**n
    assert definite_sum(binomial(n, k)**2, (k, 0, n)) == binomial(2*n, n)
    assert simplify(as_expr(definite_sum(k*binomial(n, k), (k, 0, n))) - n*2**(n - 1)) == 0
    assert definite_sum((-1)**(k + 1)*binomial(n, k)/k, (k, 1, n)) == harmonic(n)


def test_vandermonde() -> None:
    f = binomial(a, k)*binomial(b, n - k)
    result = definite_sum(f, (k, 0, n))
    assert result == binomial(a + b, n)
    _check(result, f, Integer(0), n, range(0, 6), {a: Rational(7, 3), b: Rational(-5, 11)})


def test_alternating_squares_by_residue_classes() -> None:
    # S(n + 2) = -4 (n + 1)/(n + 2) S(n): one recurrence for the even and
    # one for the odd n
    f = (-1)**k*binomial(n, k)**2
    result = definite_sum(f, (k, 0, n))
    assert isinstance(result, Piecewise)
    _check(result, f, Integer(0), n, range(0, 10))
    assert result.subs(n, 7) == 0 and result.subs(n, 6) == -20


def test_harmonic_number_identities() -> None:
    # sum binomial(n, k) H_k = 2**n (H_n - sum_{j=1}^n 1/(j 2**j))
    f = binomial(n, k)*harmonic(k)
    result = definite_sum(f, (k, 0, n))
    _check(result, f, Integer(0), n)
    j = Symbol('j', integer=True)
    assert result is not None and result.has(Sum)
    expected = 2**n*(harmonic(n) - Sum(1/(j*2**j), (j, 1, n)))
    for m in range(8):
        assert (result - expected).subs(n, m).doit() == 0
    # sum H_k/(n + 1 - k) = H_{n+1}**2 - H_{n+1}^(2)
    f = harmonic(k)/(n + 1 - k)
    result = definite_sum(f, (k, 1, n))
    _check(result, f, Integer(1), n)
    assert result == harmonic(n + 1)**2 - harmonic(n + 1, 2)
    # sum binomial(n, k)**2 H_k = binomial(2n, n) (2 H_n - H_{2n})
    f = binomial(n, k)**2*harmonic(k)
    result = definite_sum(f, (k, 0, n))
    _check(result, f, Integer(0), n, range(0, 7))
    assert result == (2*harmonic(n) - harmonic(2*n))*binomial(2*n, n)
    # sum (-1)**k binomial(n, k) H_k = -1/n for n >= 1
    f = (-1)**k*binomial(n, k)*harmonic(k)
    result = definite_sum(f, (k, 0, n))
    _check(result, f, Integer(0), n)
    assert result is not None and result.subs(n, 5) == Rational(-1, 5)
    # sum (-1)**(k - 1) binomial(n, k)/k**2 = (H_n**2 + H_n^(2))/2
    f = (-1)**(k - 1)*binomial(n, k)/k**2
    result = definite_sum(f, (k, 1, n))
    _check(result, f, Integer(1), n)
    assert result is not None and simplify(result - (harmonic(n)**2 + harmonic(n, 2))/2) == 0
    # sum (-1)**k binomial(n, k) H_k**2 = (n H_n - 2)/n**2 for n >= 1
    f = (-1)**k*binomial(n, k)*harmonic(k)**2
    _check(definite_sum(f, (k, 0, n)), f, Integer(0), n)


def test_unevaluated_sums_have_ordered_limits() -> None:
    # the bug (an audit against Mathematica 12.2): sum k binomial(n, k) H_k
    # came out as 2**n*(4*n*H_n + 4*n*Sum(1/(2**j*j*(j + 1)), (j, 2, n - 1))
    # - 3*n + 4)/8 for every n >= 1, the Sum from 2 to 0 at n = 1; its
    # value, 1, holds by Karr's convention for reversed limits only, the
    # empty sum giving 3/2
    f = k*binomial(n, k)*harmonic(k)
    result = definite_sum(f, (k, 0, n))
    assert result is not None and result.has(Sum)
    _check(result, f, Integer(0), n)
    closed = as_expr(result.subs(n, 1))
    assert closed.doit() == 1


def test_no_certified_closed_form() -> None:
    # Apéry's numbers satisfy a recurrence of order two without
    # hypergeometric solutions
    assert definite_sum(binomial(n, k)**2*binomial(n + k, k)**2, (k, 0, n), order=2) is None
    # limits which are not of the form m*n + s
    assert definite_sum(binomial(n, k), (k, 0, n**2)) is None
    # harmonic numbers with the parameter in the argument are not in the field
    assert definite_sum(binomial(n, k)*harmonic(n + k), (k, 0, n)) is None


def test_random_certificates() -> None:
    # the telescoping relation, checked exactly at integer points
    rng = random.Random(20260925)
    hypergeometric: list[Expr] = [binomial(n, k), binomial(n, k)**2, binomial(n + k, k)/2**k,
                                  (-1)**k*binomial(n, k), binomial(2*n, k), 3**k*binomial(n, k)]
    rest: list[Expr] = [Integer(1), k, 1/(k + 1), harmonic(k), k*harmonic(k)]
    found = 0
    for _ in range(6):
        f = as_expr(rng.choice(hypergeometric)*rng.choice(rest))
        Z = creative_telescoping(f, k, n, order=2)
        if Z is None:
            continue
        found += 1
        assert Z.check()
        g = Z.certificate
        for n0 in (6, 9):
            for k0 in range(1, 5):
                lhs = sum((c.subs(n, n0)*f.subs({n: n0 + i, k: k0}) for i, c in enumerate(Z.coefficients)),
                          Integer(0))
                upper = g.subs({n: n0, k: k0 + 1})
                lower = g.subs({n: n0, k: k0})
                if not (upper.is_finite and lower.is_finite):
                    continue
                assert lhs - (upper - lower) == 0, (f, n0, k0)
    assert found >= 4


def test_summation_tries_creative_telescoping() -> None:
    f = binomial(n, k)**2*harmonic(k)
    result = summation(f, (k, 0, n))
    assert not result.has(Sum)
    _check(result, f, Integer(0), n, range(0, 6))
    # a summand which does not depend on the limit is left to Karr's
    # algorithm alone
    assert summation(2**k/k, (k, 1, n)) == Sum(2**k/k, (k, 1, n))

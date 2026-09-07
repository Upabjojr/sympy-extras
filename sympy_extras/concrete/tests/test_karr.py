from __future__ import annotations

from typing import Iterable, Optional

from sympy import (Sum, Product, Expr, harmonic, factorial, binomial, Rational,
    RisingFactorial, sin, sqrt, simplify, cancel)
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy.abc import k, n, j, x

from sympy_extras.concrete import (karr_term, karr_sum, summation,
    build_pisigma_field, PiSigmaField)


def _check(f: Expr, closed: Optional[Expr], lower: int = 1, values: Iterable[int] = range(1, 7)) -> None:
    """The closed form agrees with the direct sum for several n."""
    assert closed is not None
    for m in values:
        direct = sum(f.subs({k: i, n: m}).doit() for i in range(lower, m + 1))
        assert simplify(closed.subs(n, m).doit() - direct) == 0, (f, m)


def test_harmonic_sums() -> None:
    s = karr_sum(harmonic(k), (k, 1, n))
    assert s == n*harmonic(n) - n + harmonic(n)
    _check(harmonic(k), s)
    s = karr_sum(k*harmonic(k), (k, 1, n))
    _check(k*harmonic(k), s)
    s = karr_sum(harmonic(k)**2, (k, 1, n))
    assert s == n*harmonic(n)**2 - 2*n*harmonic(n) + 2*n + harmonic(n)**2 - harmonic(n)
    _check(harmonic(k)**2, s)
    s = karr_sum(k**2*harmonic(k), (k, 1, n))
    _check(k**2*harmonic(k), s)
    s = karr_sum(harmonic(k, 2), (k, 1, n))
    _check(harmonic(k, 2), s)
    s = karr_sum(harmonic(k)*harmonic(k, 2), (k, 1, n))
    _check(harmonic(k)*harmonic(k, 2), s)
    # sum(H_k**2/k) needs the nested sum of H_j**(2)/j, which is not a
    # harmonic number: correctly no closed form in harmonic numbers
    assert karr_sum(harmonic(k)**2/k, (k, 1, n)) is None
    # shifted arguments
    s = karr_sum(harmonic(k + 1), (k, 0, n))
    _check(harmonic(k + 1), s, lower=0)
    s = karr_sum(harmonic(k - 1)/k, (k, 2, n))
    _check(harmonic(k - 1)/k, s, lower=2, values=range(2, 8))
    s = karr_sum(harmonic(k)/(k + 1), (k, 1, n))
    assert cancel(s - (harmonic(n + 1)**2 - harmonic(n + 1, 2))/2).subs(n, 5).doit() == 0
    _check(harmonic(k)/(k + 1), s)


def test_no_closed_form() -> None:
    # provably no closed form in the field built from the summand
    assert karr_sum(harmonic(k)/k, (k, 1, n), auto=False) is None
    assert karr_term(harmonic(k, 2), k, auto=False) is None
    assert karr_sum(2**k/k, (k, 1, n)) is None
    assert karr_sum(factorial(k)/k, (k, 1, n)) is None
    assert karr_sum(binomial(n, k), (k, 0, n)) is None
    assert karr_sum(harmonic(k)*2**k, (k, 1, n)) is None
    assert karr_sum(1/k, (k, 1, n)) is None
    assert karr_term(1/k**2, k) is None
    # with the right extension there is one
    s = karr_sum(harmonic(k)/k, (k, 1, n), extensions=[harmonic(k, 2)])
    assert s is not None
    assert s.expand() == \
        harmonic(n)**2/2 + harmonic(n, 2)/2


def test_hypergeometric_and_rational() -> None:
    assert karr_sum(k*factorial(k), (k, 1, n)) == n*factorial(n) + factorial(n) - 1
    s2 = karr_sum(k**2*2**k, (k, 0, n))
    assert s2 is not None
    assert s2.expand() == 2*2**n*n**2 - 4*2**n*n + 6*2**n - 6
    assert karr_sum(1/(k*(k + 1)), (k, 1, n)) == n/(n + 1)
    assert karr_sum(k**2, (k, 1, n)) == n*(2*n**2 + 3*n + 1)/6
    assert karr_sum(binomial(2*k, k)/4**k, (k, 0, n)) == (2*n + 1)*binomial(2*n, n)/4**n
    s = karr_sum(k*binomial(n, k)/(n - k + 1), (k, 0, n))
    if s is not None:
        _check(k*binomial(n, k)/(n - k + 1), s, lower=0)
    assert karr_sum(RisingFactorial(x, k), (k, 0, n)) is None
    s = karr_sum(RisingFactorial(x, k)/factorial(k), (k, 0, n))
    _check(RisingFactorial(x, k)/factorial(k), s, lower=0)
    assert karr_sum(2**(-k), (k, 0, n)) == (2*2**n - 1)/2**n
    s = karr_sum(k*3**k, (k, 1, n))
    _check(k*3**k, s)
    # mixed products and sums
    s = karr_sum(harmonic(k)/2**k, (k, 1, n))
    assert s is None
    # sum(k*k!*H_k) differs from a telescoping term by sum(k!), which has
    # no closed form
    assert karr_sum(harmonic(k)*k*factorial(k), (k, 1, n)) is None
    s = karr_sum(harmonic(k)*k*factorial(k) + factorial(k), (k, 1, n))
    _check(harmonic(k)*k*factorial(k) + factorial(k), s)
    s = karr_sum(2**k*(harmonic(k) - 2/(k + 1)), (k, 1, n))
    if s is not None:
        _check(2**k*(harmonic(k) - 2/(k + 1)), s)
    # parameters
    s = karr_sum(n*k + 1/(k*(k + 1)), (k, 1, n))
    _check(n*k + 1/(k*(k + 1)), s)
    s = karr_sum(x**k, (k, 0, n))
    assert simplify(s - (x**(n + 1) - 1)/(x - 1)) == 0


def test_nested_sums_and_products() -> None:
    inner = Sum(1/j**2, (j, 1, k))
    s = karr_sum(inner, (k, 1, n))
    _check(inner, s)
    s = karr_sum(Sum(harmonic(j), (j, 1, k)), (k, 1, n))
    _check(Sum(harmonic(j), (j, 1, k)), s)
    # a nested sum which is a known sequence is recognised
    F, f = build_pisigma_field(Sum(1/j, (j, 1, k)) + harmonic(k), k)
    assert len(F.extensions) == 1 and F.to_expr(f) == 2*harmonic(k) or F.to_expr(f) == 2*Sum(1/j, (j, 1, k))
    p = Product(2*j, (j, 1, k))
    s = karr_sum((2*k + 1)*p, (k, 1, n))
    _check((2*k + 1)*p, s)
    assert karr_sum(k*p, (k, 1, n)) is None


def test_karr_term_and_field() -> None:
    assert karr_term(harmonic(k), k) == k*(harmonic(k) - 1)
    assert karr_term(k*factorial(k), k) == factorial(k)
    g = karr_term(k, k)
    assert g is not None
    assert (g.subs(k, k + 1) - g).expand() == k
    assert karr_sum(harmonic(k), k) == karr_term(harmonic(k), k)
    F, f = build_pisigma_field(harmonic(k + 1)*factorial(k), k)
    assert [e.kind for e in F.extensions] == ['pi', 'sigma']
    assert F.to_expr(f) == (k*harmonic(k) + harmonic(k) + 1)*factorial(k)/(k + 1)
    # factorial(k + 1) is recognised as (k + 1)*factorial(k)
    F, f = build_pisigma_field(factorial(k + 1) + factorial(k), k)
    assert len(F.extensions) == 1
    assert F.to_expr(f) == (k + 2)*factorial(k)
    F, f = build_pisigma_field(harmonic(k + 2) - harmonic(k), k)
    assert len(F.extensions) == 1 and F.to_expr(f) == (2*k + 3)/(k**2 + 3*k + 2)
    F, f = build_pisigma_field(2**k + 4**k, k)
    assert len(F.extensions) == 1 and F.to_expr(f) == 2**k*(2**k + 1)
    raises(ValueError, lambda: build_pisigma_field(sin(k), k))
    raises(ValueError, lambda: build_pisigma_field(sqrt(k), k))
    raises(TypeError, lambda: untyped(build_pisigma_field)(k, 2))


def test_pisigma_field() -> None:
    F = PiSigmaField(k, params=[n])
    assert F.level == 0 and F.params == (n,)
    kk = F.gens[0]
    assert F.sigma(kk) == kk + 1 and F.sigma(kk, -1) == kk - 1 and F.sigma(kk, 2) == kk + 2
    h = F.add_sigma(1/(k + 1), harmonic(k))
    t = F.from_expr(h)
    kk = F.gens[0]
    assert F.level == 1 and F.level_of(t) == 1 and F.level_of(kk) == 0 and F.level_of(F.from_expr(n)) == -1
    assert F.sigma(F.sigma(t), -1) == t
    assert F.as_poly_in(t**2*kk + 1, 1) == {2: kk, 0: F.one}
    assert F.as_poly_in(1/t, 1) is None
    assert F.to_constant(F.from_expr(n/2)) == F.C.from_sympy(n/2)
    raises(ValueError, lambda: F.to_constant(kk))
    p = F.add_pi(2, 2**k)
    u = F.from_expr(p)
    kk, t = F.gens[0], F.from_expr(h)
    assert F.as_poly_in(kk/u, 2, laurent=True) == {-1: kk}
    assert F.as_poly_in(kk/u, 2) is None
    assert F.sigma(u, -1) == u/2
    # the parameterized equation: sigma(g) - g = c1*H + c2*(k*H) has a two
    # dimensional solution space plus the constants
    sols = F.solve(F.one, [t, kk*t])
    assert sols is not None
    assert len(sols) == 3
    for c, g in sols:
        assert F.sigma(g) - g == F.field(c[0])*t + F.field(c[1])*kk*t
    # homogeneous equations
    assert F.solve(F.from_expr(k + 1), []) == []
    hom = F.solve(F.from_expr((k + 1)/k), [])
    assert hom is not None
    [(c, g)] = hom
    assert c == [] and F.to_expr(g) in (k, 2*k) or F.sigma(g)*kk == (kk + 1)*g
    raises(ValueError, lambda: F.add_pi(0, 1))
    raises(TypeError, lambda: untyped(PiSigmaField)(2))
    assert repr(F.extensions[0]) == "Extension(sigma, harmonic(k), 1/(k + 1))"


def test_summation() -> None:
    assert summation(k**2, (k, 1, n)) == n**3/3 + n**2/2 + n/6
    assert summation(harmonic(k), (k, 1, n)) == n*harmonic(n) - n + harmonic(n)
    assert summation(harmonic(k)/k, (k, 1, n)).expand() == harmonic(n)**2/2 + harmonic(n, 2)/2
    assert summation(2**k/k, (k, 1, n)) == Sum(2**k/k, (k, 1, n))
    assert summation(sin(k), (k, 1, n)).has(Sum)
    assert summation(harmonic(k), (k, 1, 3)) == Rational(13, 3)
    assert summation(Sum(1/j**2, (j, 1, k)), (k, 1, n)) == n*harmonic(n, 2) - harmonic(n) + harmonic(n, 2)

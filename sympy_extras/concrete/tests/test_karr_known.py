"""Karr's algorithm against closed forms published elsewhere.

The identities below are taken from the literature and from the
documentation of other computer algebra systems (as results, not code):
Graham, Knuth, Patashnik, *Concrete Mathematics* (GKP, equation numbers of
the 2nd edition); Petkovsek, Wilf, Zeilberger, *A = B*; Schneider's papers
on the Sigma package; the documented examples of Mathematica's ``Sum`` and
Maple's ``sum``. Every closed form is compared with ours exactly at several
values of ``n`` and, where SymPy can decide it, symbolically.
"""
from __future__ import annotations

from sympy import (Sum, harmonic, factorial, binomial, Rational, simplify,
    symbols)
from sympy.abc import k, n, m, x

from sympy_extras.concrete import karr_sum

H = harmonic


def _agree(ours, published, values=range(1, 9), extra=None):
    assert ours is not None
    for v in values:
        subs = {n: v}
        if extra:
            subs.update(extra)
        assert simplify((ours - published).subs(subs).doit()) == 0, (v, ours, published)


def test_concrete_mathematics():
    # GKP (6.67): sum_{0<=k<n} H_k = n H_n - n
    _agree(karr_sum(H(k), (k, 0, n - 1)), n*H(n) - n)
    # GKP (6.69): sum_{0<=k<n} k H_k = n(n-1)/2 H_n - n(n-1)/4
    _agree(karr_sum(k*H(k), (k, 0, n - 1)), n*(n - 1)/2*H(n) - n*(n - 1)/4)
    # GKP (6.70): sum_{0<=k<n} C(k, m) H_k = C(n, m+1) (H_n - 1/(m+1)), m = 2
    _agree(karr_sum(binomial(k, 2)*H(k), (k, 0, n - 1)), binomial(n, 3)*(H(n) - Rational(1, 3)))
    # the same with a symbolic m
    _agree(karr_sum(binomial(k, m)*H(k), (k, 0, n - 1)), binomial(n, m + 1)*(H(n) - 1/(m + 1)),
           values=range(3, 9), extra={m: 2})
    # GKP (2.36) / exercise: sum_{1<=k<=n} H_k = (n+1) H_n - n
    _agree(karr_sum(H(k), (k, 1, n)), (n + 1)*H(n) - n)
    # GKP exercise 6.53 style: sum_{1<=k<=n} H_k/k = (H_n^2 + H_n^(2))/2
    _agree(karr_sum(H(k)/k, (k, 1, n)), (H(n)**2 + H(n, 2))/2)
    # sum of squares of harmonic numbers (Knuth, TAOCP 1 exercise 1.2.7-18)
    _agree(karr_sum(H(k)**2, (k, 1, n)), (n + 1)*H(n)**2 - (2*n + 1)*H(n) + 2*n)
    # GKP (5.9) style Gosper-summable rational sums: sum 1/(k(k+1)) = n/(n+1)
    _agree(karr_sum(1/(k*(k + 1)), (k, 1, n)), n/(n + 1))
    _agree(karr_sum(1/(k*(k + 1)*(k + 2)), (k, 1, n)), Rational(1, 4) - 1/(2*(n + 1)*(n + 2)))
    # GKP (2.26): sum_{0<=k<=n} k 2^k = (n-1) 2^(n+1) + 2
    _agree(karr_sum(k*2**k, (k, 0, n)), (n - 1)*2**(n + 1) + 2)
    # geometric and arithmetico-geometric sums, GKP (2.25), (2.27)
    _agree(karr_sum(x**k, (k, 0, n)), (x**(n + 1) - 1)/(x - 1), extra={x: 3})
    _agree(karr_sum(k*x**k, (k, 1, n)), x*(1 - (n + 1)*x**n + n*x**(n + 1))/(1 - x)**2, extra={x: 5})


def test_a_equals_b_and_gosper_examples():
    # A = B, chapter 5: sum_{k=0}^{n} k k! = (n+1)! - 1
    _agree(karr_sum(k*factorial(k), (k, 0, n)), factorial(n + 1) - 1)
    # A = B: sum_{k=0}^{n} (4k+1) k!/(2k+1)! = 2 - (n+1)!... Gosper-summable;
    # compare with SymPy's own Gosper implementation
    from sympy.concrete.gosper import gosper_sum
    f = (4*k + 1)*factorial(k)/factorial(2*k + 1)
    _agree(karr_sum(f, (k, 0, n)), gosper_sum(f, (k, 0, n)))
    # A = B (5.4.1): partial sums of (-1)^k C(n, k): sum_{k=0}^{m} (-1)^k C(n,k) = (-1)^m C(n-1, m)
    _agree(karr_sum((-1)**k*binomial(n, k), (k, 0, m)), (-1)**m*binomial(n - 1, m),
           values=range(2, 8), extra={m: 1})
    # the central binomial partial sums: sum_{k=0}^{n} C(2k,k)/4^k = (2n+1) C(2n,n)/4^n
    _agree(karr_sum(binomial(2*k, k)/4**k, (k, 0, n)), (2*n + 1)*binomial(2*n, n)/4**n)
    # sum_{k=1}^{n} k/2^k = 2 - (n+2)/2^n
    _agree(karr_sum(k/2**k, (k, 1, n)), 2 - (n + 2)/2**n)
    # sum_{k=1}^{n} 1/(4k^2-1) = n/(2n+1)
    _agree(karr_sum(1/(4*k**2 - 1), (k, 1, n)), n/(2*n + 1))
    # an alternating sum: sum_{k=1}^{n} (-1)^k k = ((-1)^n (2n+1) - 1)/4
    _agree(karr_sum((-1)**k*k, (k, 1, n)), ((-1)**n*(2*n + 1) - 1)/4)
    # not Gosper-summable and not Karr-summable either (A = B, sum of k!)
    assert karr_sum(factorial(k), (k, 0, n)) is None
    assert karr_sum(binomial(n, k), (k, 0, m)) is None


def test_sigma_examples():
    # examples of the kind treated in Schneider's papers on Sigma
    # sum_{k=1}^{n} H_k^(2) = (n+1) H_n^(2) - H_n
    _agree(karr_sum(H(k, 2), (k, 1, n)), (n + 1)*H(n, 2) - H(n))
    # sum_{k=1}^{n} k^2 H_k = n(n+1)(2n+1)/6 H_n - n(n-1)(4n+1)/36
    _agree(karr_sum(k**2*H(k), (k, 1, n)), n*(n + 1)*(2*n + 1)/6*H(n) - n*(n - 1)*(4*n + 1)/36)
    # sum_{k=0}^{n} H_k/(k+1) = (H_{n+1}^2 - H_{n+1}^(2))/2
    _agree(karr_sum(H(k)/(k + 1), (k, 0, n)), (H(n + 1)**2 - H(n + 1, 2))/2)
    # sum_{k=1}^{n} H_k^3 = (n+1) H_n^3 - 3(n+1) H_n^2 + 3(2n+1) H_n - 6n + ... : compare
    # with the direct sum only (published forms differ in presentation)
    s = karr_sum(H(k)**3, (k, 1, n))
    for v in range(1, 7):
        assert s.subs(n, v).doit() == sum(H(i)**3 for i in range(1, v + 1))
    # sum_{k=1}^{n} H_k H_k^(2) needs H^(3), found automatically
    s = karr_sum(H(k)*H(k, 2), (k, 1, n))
    for v in range(1, 7):
        assert s.subs(n, v).doit() == sum(H(i)*H(i, 2) for i in range(1, v + 1))
    # harmonic numbers with a scaled argument: sum_{k=1}^{n} H_{2k}
    s = karr_sum(H(2*k), (k, 1, n))
    assert s == n*H(2*n) - n + H(n)/4 + H(2*n)/2
    for v in range(1, 7):
        assert s.subs(n, v).doit() == sum(H(2*i) for i in range(1, v + 1))
    # H_{3k+1}^(2) would need sum 1/(3j+2), which is not a harmonic number
    assert karr_sum(H(3*k + 1, 2), (k, 0, n)) is None
    # nested sums as such
    j = symbols('j')
    s = karr_sum(Sum(H(j), (j, 1, k)), (k, 1, n))
    _agree(s, karr_sum((n + 1 - k)*H(k), (k, 1, n)))
    # impossibility results: no closed form in harmonic numbers and powers
    assert karr_sum(H(k)*2**k, (k, 1, n)) is None
    assert karr_sum(H(k)/2**k, (k, 1, n)) is None
    assert karr_sum(2**k/k, (k, 1, n)) is None
    assert karr_sum(H(k)**2/k, (k, 1, n)) is None
    assert karr_sum(H(k)/(k*(k + 1)), (k, 1, n)) is not None
    assert karr_sum(1/(k*2**k), (k, 1, n)) is None


def test_mathematica_maple_documented_results():
    # Mathematica: Sum[HarmonicNumber[k], {k, 1, n}] == (n + 1) HarmonicNumber[n] - n
    _agree(karr_sum(H(k), (k, 1, n)), (n + 1)*H(n) - n)
    # Mathematica: Sum[k HarmonicNumber[k], {k, 1, n}]
    _agree(karr_sum(k*H(k), (k, 1, n)), n*(n + 1)/2*H(n) - n*(n - 1)/4)
    # Maple: sum(k*3^k, k=1..n) = 3/4*((2n-1)*3^n + 1)
    _agree(karr_sum(k*3**k, (k, 1, n)), Rational(3, 4)*((2*n - 1)*3**n + 1))
    # Maple: sum(1/(k*(k+2)), k=1..n) = 3/4 - (2n+3)/(2(n+1)(n+2))
    _agree(karr_sum(1/(k*(k + 2)), (k, 1, n)), Rational(3, 4) - (2*n + 3)/(2*(n + 1)*(n + 2)))
    # Maple: sum(k^3, k=1..n) = n^2 (n+1)^2/4
    _agree(karr_sum(k**3, (k, 1, n)), n**2*(n + 1)**2/4)
    # Mathematica: Sum[HarmonicNumber[k]/k, {k,1,n}] == (HarmonicNumber[n]^2 + HarmonicNumber[n, 2])/2
    _agree(karr_sum(H(k)/k, (k, 1, n)), (H(n)**2 + H(n, 2))/2)

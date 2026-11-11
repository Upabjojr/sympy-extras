"""Tests of the native modular Gröbner bases and of the native Buchberger
algorithm on the typed rings (no SymPy). The oracles are F4
(:mod:`sympy_extras.polys.f4`), an independent algorithm, the bases being
compared after reduction (a reduced Gröbner basis is unique), and hand
computations."""
from __future__ import annotations

import random
from fractions import Fraction
from itertools import chain
from typing import Iterator, TypeVar

from sympy_extras.ntheory.generate import primerange
from sympy_extras.polys.buchberger import buchberger, s_polynomial
from sympy_extras.polys.domains import GF, QQ, ZZ
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.f4 import f4
from sympy_extras.polys.modulargroebner import (
    GRADED_FACTOR, ModularTrace, default_primes, direct_groebner, groebner, homogeneous_order, modular_groebner)
from sympy_extras.polys.rings import (
    MonomialOrder, PolyElement, PolyRing, WeightOrder, elimination_order, grevlex, grlex, lex)
from sympy_extras.settings import configure, settings

T = TypeVar('T')


def tiny_primes() -> Iterator[int]:
    """2, 3, 5, 7, ...: the primes which are unlucky for small inputs."""
    return chain(primerange(2, 10**4), default_primes())


def strs(G: list[PolyElement[T]]) -> list[str]:
    return [str(g) for g in G]


def c(n: int, d: int = 1) -> Fraction:
    return Fraction(n, d)


# ----------------------------------------------------------------------
# the classical families

def katsura(n: int, order: MonomialOrder) -> tuple[list[PolyElement[Fraction]], PolyRing[Fraction]]:
    R = PolyRing(QQ, ['u%d' % i for i in range(n + 1)], order)
    u = R.gens

    def U(i: int) -> PolyElement[Fraction]:
        return u[abs(i)] if abs(i) <= n else R.zero
    eqs: list[PolyElement[Fraction]] = []
    for m in range(n):
        e = R.zero
        for l in range(-n, n + 1):
            e = e + U(l)*U(m - l)
        eqs.append(e - U(m))
    e = R.zero
    for l in range(-n, n + 1):
        e = e + U(l)
    eqs.append(e - R.one)
    return eqs, R


def cyclic(n: int, order: MonomialOrder) -> tuple[list[PolyElement[Fraction]], PolyRing[Fraction]]:
    R = PolyRing(QQ, ['x%d' % i for i in range(n)], order)
    x = R.gens
    eqs: list[PolyElement[Fraction]] = []
    for k in range(1, n):
        e = R.zero
        for i in range(n):
            t = R.one
            for j in range(k):
                t = t*x[(i + j) % n]
            e = e + t
        eqs.append(e)
    t = R.one
    for v in x:
        t = t*v
    eqs.append(t - R.one)
    return eqs, R


def test_classical_families() -> None:
    for build, n in ((katsura, 2), (katsura, 3), (cyclic, 3), (cyclic, 4)):
        for order in (lex, grevlex, grlex):
            F, R = build(n, order)
            expected = f4(F, R)
            assert modular_groebner(F, R) == expected
            assert modular_groebner(F, R, primes=tiny_primes()) == expected
            assert buchberger(F, R) == expected
    F, R = katsura(3, lex)
    # the triangular lexicographic basis of Katsura-3: the last element is
    # a univariate polynomial of degree 8, the number of solutions
    G = modular_groebner(F, R)
    assert len(G) == 4 and G[-1].LM == (0, 0, 0, 8) and all(g.is_monic for g in G)


# ----------------------------------------------------------------------
# known results by hand

def test_known_bases() -> None:
    R = PolyRing(QQ, ['x', 'y', 'z'], lex)
    x, y, z = R.gens
    F = [x**2 + y**2 + z**2 - R.one, x*y - z.mul_ground(c(1, 3)), x + y - z.mul_ground(c(2))]
    expected = ['x + y - 2*z', 'y**2 - 2*y*z + 1/3*z', 'z**2 - 2/15*z - 1/5']
    assert strs(modular_groebner(F, R)) == strs(buchberger(F, R)) == strs(groebner(F, R)) == expected
    # x**2 = y and x*y = 1 give y**3 = x**2*y**2 = 1 and x = x*(x*y) ... = y**2
    R2 = PolyRing(QQ, ['x', 'y'], lex)
    X, Y = R2.gens
    assert strs(modular_groebner([X*X - Y, X*Y - R2.one], R2)) == ['x - y**2', 'y**3 - 1']
    # the S-polynomial of x**2 - y and x*y - 1 for lex
    assert str(s_polynomial(X*X - Y, X*Y - R2.one)) == 'x - y**2'


def test_the_theorem_fails_without_homogenisation() -> None:
    # modulo 2, 3, 5 and 7 the ideal of F is <x>, and [x] passes the checks
    # on the non-homogeneous input although <F> = <x*z, x*(210*y + 1)> is
    # smaller: on the homogenisation the candidate of these primes fails
    # and the first good prime, 11, discards them
    R = PolyRing(QQ, ['x', 'z', 'y'], lex)
    x, z, y = R.gens
    F = [x*(z + y.mul_ground(c(210)) + R.one), x*(z + y.mul_ground(c(420)) + R.from_int(2))]
    trace = ModularTrace()
    G = modular_groebner(F, R, primes=tiny_primes(), trace=trace)
    assert strs(G) == ['x*z', 'x*y + 1/210*x'] and G == f4(F, R)
    assert trace.homogenized and trace.unlucky == [2, 3, 5, 7] and trace.lucky[0] == 11
    assert trace.failed_verifications == 1
    for order in (lex, grevlex, elimination_order(1, 3)):
        Ro = R.with_order(order)
        Fo = [PolyElement(dict(f.terms), Ro) for f in F]
        assert modular_groebner(Fo, Ro) == modular_groebner(Fo, Ro, primes=tiny_primes()) == f4(Fo, Ro)


def test_unlucky_and_skipped_primes() -> None:
    R = PolyRing(QQ, ['x', 'y'], lex)
    x, y = R.gens
    # modulo 2 and 3 the generators coincide: a larger Hilbert function
    trace = ModularTrace()
    assert strs(modular_groebner([x**2 + y, x**2 + y.mul_ground(c(7))], R, primes=tiny_primes(), trace=trace)) == \
        ['x**2', 'y']
    assert trace.unlucky == [2, 3] and trace.lucky == [5, 7] and trace.skipped == [] and trace.homogenized
    # a prime which divides a leading coefficient of the input is not used
    R3 = PolyRing(QQ, ['x', 'y', 'z'], lex)
    x, y, z = R3.gens
    F = [(x*y).mul_ground(c(5)) + y*z, (x*z).mul_ground(c(5)) + y**2 + z**2]
    trace = ModularTrace()
    assert modular_groebner(F, R3, primes=tiny_primes(), trace=trace) == f4(F, R3)
    assert trace.skipped == [5] and 5 not in trace.used


def test_primes_run_out() -> None:
    R = PolyRing(QQ, ['x', 'y'], lex)
    x, y = R.gens
    F = [x**2 + y.mul_ground(c(1, 1000003)), x*y - R.one]
    try:
        modular_groebner(F, R, primes=[2, 3, 5])
    except ValueError:
        pass
    else:
        assert False, "the primes did not run out"
    assert modular_groebner(F, R, primes=tiny_primes()) == f4(F, R)


def test_trivial_inputs() -> None:
    R = PolyRing(QQ, ['x', 'y'], grevlex)
    x, y = R.gens
    assert modular_groebner([], R) == [] and modular_groebner([R.zero], R) == []
    assert groebner([], R) == [] and buchberger([R.zero], R) == []
    assert modular_groebner([R.zero, x], R) == groebner([R.zero, x], R) == [x]
    assert modular_groebner([x, x + R.one], R) == [R.one]
    assert modular_groebner([R.from_int(3)], R) == [R.one]
    assert strs(modular_groebner([x.mul_ground(c(2)) - R.from_int(3)], R)) == ['x - 3/2']
    assert modular_groebner([x*y, x*y], R) == [x*y]


def test_integer_ring() -> None:
    # over ZZ the basis over QQ with its denominators cleared: primitive,
    # positive leading coefficients
    R = PolyRing(ZZ, ['x', 'y', 'z'], lex)
    x, y, z = R.gens
    F = [(x**2).mul_ground(3) + y*z - R.from_int(2), x*y - (z**2).mul_ground(5) + R.one, (y**2).mul_ground(7) - z]
    G = modular_groebner(F, R)
    Q = PolyRing(QQ, ['x', 'y', 'z'], lex)
    expected = f4([PolyElement({m: Fraction(a) for m, a in f.terms.items()}, Q) for f in F], Q)
    assert len(G) == len(expected)
    for g, h in zip(G, expected):
        assert g.ring == R and g.LC > 0 and g.primitive()[0] == 1
        common = Fraction(g.LC)
        assert {m: Fraction(a)/common for m, a in g.terms.items()} == h.terms
    assert groebner(F, R) == G


def test_other_domains_are_refused() -> None:
    R = PolyRing(GF(7), ['x', 'y'], lex)
    x, y = R.gens
    try:
        modular_groebner([x**2 + y], R)
    except ValueError:
        pass
    else:
        assert False, "GF(7) was accepted"
    # the dispatcher computes directly over other fields
    F = [x**2 + y.mul_ground(3), x*y - R.one]
    assert groebner(F, R) == f4(F, R) == buchberger(F, R)


def test_orders_without_sympy_counterpart() -> None:
    # the weight orders of the typed layer (which the SymPy face could not
    # convert: the modular algorithm was skipped for them before the port)
    for order in (WeightOrder((1, 2, 3)), WeightOrder((2, 1, 0), grevlex), elimination_order(2, 3, lex)):
        R = PolyRing(QQ, ['x', 'y', 'z'], order)
        x, y, z = R.gens
        F = [x**2 + y*z - R.from_int(2), x*y - z**2 + R.one, y**2 - z.mul_ground(c(1, 7))]
        assert modular_groebner(F, R) == f4(F, R) == buchberger(F, R)


def test_homogeneous_order() -> None:
    H = homogeneous_order(lex, 2)
    assert H.is_graded and repr(H) == 'HomogeneousOrder(lex, 2)'
    # the degree first, then lex on the first two exponents only
    assert H((0, 2, 0)) > H((1, 0, 0)) and H((1, 0, 1)) > H((0, 2, 0)) and H((0, 1, 2)) > H((1, 0, 1))
    assert H((1, 1, 0)) > H((1, 0, 1)) and H((1, 0, 0)) > H((0, 1, 0))


# ----------------------------------------------------------------------
# random audits against F4

def _random_system(rng: random.Random, K: Domain[T], one: T) -> tuple[list[PolyElement[T]], PolyRing[T]]:
    n = rng.randint(2, 3)
    order = rng.choice([lex, grevlex, grlex, elimination_order(1, n), WeightOrder(tuple(rng.randint(0, 3) for _ in range(n)))])
    R = PolyRing(K, list('xyz'[:n]), order)
    homogeneous = rng.random() < 0.25
    polys: list[PolyElement[T]] = []
    for _ in range(rng.randint(1, n + 1)):
        degree = rng.randint(1, 3)
        f = R.zero
        for _ in range(rng.randint(1, 4)):
            while True:
                m = tuple(rng.randint(0, degree) for _ in range(n))
                if (sum(m) == degree) if homogeneous else (sum(m) <= degree):
                    break
            f = f + R.term(m, K.mul(one, K.from_int(rng.choice([1, 2, 3, 5, 7, 210, -1, -6, 2310, 10**12 + 39]))))
        polys.append(f)
    return polys, R


def test_random_audit_over_the_rationals() -> None:
    rng = random.Random(20261111)
    unlucky = homogenized = 0
    for i in range(120):
        third = Fraction(1, rng.choice([1, 1, 2, 3, 7, 30]))
        F, R = _random_system(rng, QQ, third)
        expected = f4(F, R)
        trace = ModularTrace()
        assert modular_groebner(F, R, primes=tiny_primes(), trace=trace) == expected
        assert modular_groebner(F, R) == expected
        if i % 4 == 0:
            assert buchberger(F, R) == expected
        unlucky += bool(trace.unlucky)
        homogenized += trace.homogenized
    assert unlucky > 10 and homogenized > 50


def test_buchberger_over_finite_fields() -> None:
    rng = random.Random(11)
    for p in (2, 3, 7, 32003, 2**61 - 1):
        K = GF(p)
        for _ in range(15):
            F, R = _random_system(rng, K, K.one)
            assert buchberger(F, R) == f4(F, R)


# ----------------------------------------------------------------------
# the dispatcher and the settings

def test_groebner_dispatch() -> None:
    F, R = katsura(3, lex)
    expected = f4(F, R)
    assert groebner(F, R) == expected
    with configure(groebner_direct_time=1e-6):
        # the direct computation is interrupted and the modular one answers
        assert groebner(F, R) == expected
    with configure(modular_groebner=False, groebner_direct_time=1e-6):
        assert groebner(F, R) == expected
    with configure(groebner_algorithm='buchberger'):
        assert groebner(F, R) == direct_groebner(F, R) == expected
        with configure(groebner_direct_time=0):
            # the images modulo the primes by Buchberger's algorithm
            assert groebner(F, R) == expected
    G, Rg = katsura(3, grevlex)
    assert GRADED_FACTOR > 1
    with configure(groebner_direct_time=1e-7):
        assert groebner(G, Rg) == f4(G, Rg)
    assert settings.groebner_algorithm == 'f4' and settings.modular_groebner is True

"""Tests of the modular Gröbner bases. The oracle is SymPy's own
``groebner`` over ``QQ``: the results must be equal, not only equivalent
(for the two largest lexicographic bases, which take SymPy a minute, the
oracle is SymPy's FGLM conversion of its ``grevlex`` basis)."""
from __future__ import annotations

import random
from itertools import chain
from math import gcd, isqrt
from typing import Iterator, Sequence

from sympy import GF, QQ, ZZ, Add, Expr, Integer, Mul, Symbol, isprime, primerange, sqrt, symbols
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as sympy_groebner
from sympy.polys.orderings import MonomialOrder, grevlex, grlex, lex
from sympy.polys.rings import PolyElement, PolyRing, ring
from sympy.testing.pytest import raises

from sympy_extras._typing import Monomial
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.modulargroebner import (
    GRADED_FACTOR, IntPoly, ModularTrace, _HomogeneousOrder, _critical_pairs, _image, _is_groebner, _luckiness, _normal_form,
    _s_polynomial, _verified,
    chinese_remainder, default_primes, groebner, modular_groebner, rational_reconstruction)
from sympy_extras.polys.orderings import elimination_order
from sympy_extras.settings import configure, settings


def tiny_primes() -> Iterator[int]:
    """2, 3, 5, 7, ...: the primes which are unlucky for small inputs."""
    return chain(primerange(2, 10**4), default_primes())


# ----------------------------------------------------------------------
# integers

def test_rational_reconstruction() -> None:
    for m in (101, 2*3*5*7*11, 32749*32719, 2**61 - 1):
        bound = isqrt((m - 1)//2)
        assert 2*bound**2 < m
        step = max(1, bound//12)
        for n in chain(range(-bound, bound + 1, step), (-bound, -1, 0, 1, bound)):
            for d in chain(range(1, bound + 1, step), (1, bound)):
                if gcd(n, d) != 1 or gcd(d, m) != 1:
                    continue
                assert rational_reconstruction(n*pow(d, -1, m) % m, m) == (n, d)
    # every residue modulo 101 is a fraction within the bound or is refused,
    # and what is returned is congruent, in lowest terms and within the bound
    found = 0
    for a in range(101):
        fraction = rational_reconstruction(a, 101)
        if fraction is not None:
            n, d = fraction
            assert (n - a*d) % 101 == 0 and gcd(n, d) == 1 and abs(n) <= 7 and 0 < d <= 7
            found += 1
    assert found == len({n*pow(d, -1, 101) % 101 for n in range(-7, 8) for d in range(1, 8)})
    assert rational_reconstruction(10, 101) is None
    # residues are taken modulo m
    assert rational_reconstruction(-2, 101) == rational_reconstruction(99, 101) == (-2, 1)


def test_rational_reconstruction_bound() -> None:
    m = 10007
    a = 3*pow(50, -1, m) % m
    assert rational_reconstruction(a, m) == (3, 50)
    assert rational_reconstruction(a, m, 50) == (3, 50)
    assert rational_reconstruction(a, m, 49) is None
    # a denominator which is not invertible modulo m is never returned
    assert rational_reconstruction(5, 15) is None
    # modulo 2 nothing is unique (1 = -1): the bound is zero
    assert rational_reconstruction(1, 2) is None and rational_reconstruction(0, 3) == (0, 1)
    raises(ValueError, lambda: rational_reconstruction(1, 1))
    raises(ValueError, lambda: rational_reconstruction(1, 101, 8))


def test_chinese_remainder() -> None:
    for m, p in ((3, 5), (4, 9), (35, 11), (2**61 - 1, 2**31 - 1)):
        for a in (0, 1, m//2, m - 1):
            for b in (0, 1, p//3, p - 1):
                c = chinese_remainder(a, m, b, p)
                assert 0 <= c < m*p and c % m == a and c % p == b
    # a fraction through several primes and back
    primes = [10007, 10009, 10037]
    residue, modulus = 0, 1
    for p in primes:
        residue = chinese_remainder(residue, modulus, -355*pow(113, -1, p) % p, p)
        modulus *= p
    assert rational_reconstruction(residue, modulus) == (-355, 113)


def test_default_primes() -> None:
    primes = default_primes()
    first = [next(primes) for _ in range(3)]
    assert first[0] == 2**1024 - 105 and not any(isprime(2**1024 - k) for k in range(1, 105, 2))
    assert all(isprime(p) for p in first)
    assert first[0] > first[1] > first[2] > 2**1023
    # the sequence is fixed: a second iterator gives the same primes
    again = default_primes()
    assert [next(again) for _ in range(3)] == first


# ----------------------------------------------------------------------
# the verification

def _int_polys(polys: Sequence[PolyElement], order: MonomialOrder) -> list[tuple[Monomial, IntPoly]]:
    result: list[tuple[Monomial, IntPoly]] = []
    for f in polys:
        g: IntPoly = {tuple(m): int(c) for m, c in f.items()}
        result.append((max(g, key=order), g))
    return result


def _is_groebner_all_pairs(basis: Sequence[tuple[Monomial, IntPoly]], order: MonomialOrder) -> bool:
    for i in range(len(basis)):
        for j in range(i):
            s = _s_polynomial(basis[i][0], basis[i][1], basis[j][0], basis[j][1])
            if _normal_form(s, basis, order, True):
                return False
    return True


def test_critical_pairs_against_all_pairs() -> None:
    # the criteria of Gebauer and Möller must not change the answer: sets
    # which are Gröbner bases (with redundant elements, in any position) and
    # sets which are not, against the reduction of all the S-polynomials
    rng = random.Random(7)
    R, x, y, z = ring("x,y,z", ZZ, grevlex)
    gens = (x, y, z)
    yes = no = saved = 0
    for _ in range(200):
        polys: list[PolyElement] = []
        for _ in range(rng.randint(2, 4)):
            f = R.zero
            for _ in range(rng.randint(1, 3)):
                f += rng.randint(-3, 3)*gens[rng.randrange(3)]**rng.randint(0, 2)*gens[rng.randrange(3)]**rng.randint(0, 1)
            if f:
                polys.append(f)
        if not polys:
            continue
        if rng.random() < 0.3:
            polys = polys + [g for g in sympy_groebner(polys, R)]
            rng.shuffle(polys)
        basis = _int_polys(polys, grevlex)
        expected = _is_groebner_all_pairs(basis, grevlex)
        assert _is_groebner(basis, grevlex) == expected
        yes += expected
        no += not expected
        saved += len(basis)*(len(basis) - 1)//2 - len(_critical_pairs([lm for lm, _ in basis]))
    assert yes > 30 and no > 30 and saved > 100


def test_the_theorem_fails_without_homogenisation() -> None:
    # Arnold's theorem 7.1 needs a homogeneous ideal. For F below the ideal
    # modulo 2, 3, 5 and 7 is <x>, and {x} passes the two tests (F reduces to
    # zero, {x} is a Gröbner basis) with the leading monomials of every one
    # of these primes, although <F> = <x*z, x*(210*y + 1)> is smaller: a
    # modular algorithm which verifies the candidate of the non-homogeneous
    # ideal returns [x].
    R, x, z, y = ring("x,z,y", QQ, lex)
    F = [x*(z + 210*y + 1), x*(z + 420*y + 2)]
    wrong: IntPoly = {(1, 0, 0): 1}
    assert _verified([{(1, 0, 0): (1, 1)}], [(1, 0, 0)], [g for _, g in _int_polys(F, lex)], lex) == [wrong]
    for p in (2, 3, 5, 7):
        Rp = R.clone(domain=GF(p))
        assert sympy_groebner([f.set_ring(Rp) for f in F], Rp) == [Rp.gens[0]]
    # on the homogenisation the candidate of these primes fails the first
    # test (F does not reduce to zero) and the first good prime, 11,
    # discards them
    trace = ModularTrace()
    G = modular_groebner(F, R, primes=tiny_primes(), trace=trace)
    assert G == sympy_groebner(F, R) == [x*z, x*y + x/210]
    assert trace.homogenized and trace.unlucky == [2, 3, 5, 7] and trace.lucky[0] == 11
    assert trace.failed_verifications == 1
    # with the default primes as well, and in the other orders
    for order in (lex, grevlex, elimination_order(1, 3)):
        Ro = R.clone(order=order)
        Fo = [f.set_ring(Ro) for f in F]
        assert modular_groebner(Fo, Ro) == modular_groebner(Fo, Ro, primes=tiny_primes()) == sympy_groebner(Fo, Ro)


# ----------------------------------------------------------------------
# unlucky primes

def test_unlucky_primes_are_discarded() -> None:
    R, x, y = ring("x,y", QQ, lex)
    # modulo 2 and 3 the two generators coincide: fewer leading monomials, a
    # larger Hilbert function
    trace = ModularTrace()
    assert modular_groebner([x**2 + y, x**2 + 7*y], R, primes=tiny_primes(), trace=trace) == [x**2, y]
    assert trace.unlucky == [2, 3] and trace.lucky == [5, 7] and trace.skipped == []
    # the same Hilbert function and other leading monomials: the difference
    # of the generators is 5*x*z + y*z - z**2, whose leading monomial is y*z
    # modulo 5; the image with the smaller monomials is the unlucky one
    R, x, y, z = ring("x,y,z", QQ, lex)
    F = [x*y + z**2, x*y + 5*x*z + y*z]
    order = _HomogeneousOrder(lex, 3)
    integer = [g for _, g in _int_polys(F, lex)]
    at5, at7 = ([lm for lm, _ in _image(integer, R.symbols, order, order, p)] for p in (5, 7))
    assert at5 == [(0, 1, 1), (1, 1, 0), (1, 0, 2)] and at7 == [(1, 0, 1), (1, 1, 0), (0, 2, 1)]
    assert [n for n, _ in _luckiness(at5, order)] == [n for n, _ in _luckiness(at7, order)] == [0, 0, 2, 1]
    assert _luckiness(at5, order) < _luckiness(at7, order)
    trace = ModularTrace()
    assert modular_groebner(F, R, primes=tiny_primes(), trace=trace) == sympy_groebner(F, R)
    assert trace.unlucky == [5] and trace.lucky == [2, 3, 7, 11, 13] and not trace.homogenized
    # a prime which divides a leading coefficient of the input is not used
    F = [5*x*y + y*z, 5*x*z + y**2 + z**2]
    trace = ModularTrace()
    assert modular_groebner(F, R, primes=tiny_primes(), trace=trace) == sympy_groebner(F, R)
    assert trace.skipped == [5] and 5 not in trace.used


def test_primes_run_out() -> None:
    R, x, y = ring("x,y", QQ, lex)
    F = [x**2 + y/1000003, x*y - 1]
    raises(ValueError, lambda: modular_groebner(F, R, primes=[2, 3, 5]))
    assert modular_groebner(F, R, primes=tiny_primes()) == sympy_groebner(F, R)


# ----------------------------------------------------------------------
# the interface

def test_trivial_inputs() -> None:
    R, x, y = ring("x,y", QQ, grevlex)
    assert modular_groebner([], R) == []
    assert modular_groebner([R.zero], R) == []
    # SymPy's ring level groebner raises ZeroDivisionError on [0, x] (it
    # divides by the zero polynomial while interreducing the input); zero
    # polynomials are dropped here
    raises(ZeroDivisionError, lambda: sympy_groebner([R.zero, x], R))
    assert modular_groebner([R.zero, x], R) == groebner([R.zero, x], R) == [x]
    assert modular_groebner([x, x + 1], R) == [R.one]
    assert modular_groebner([R(3)], R) == [R.one]
    assert modular_groebner([2*x - 3], R) == [x - QQ(3, 2)]
    assert modular_groebner([x*y, x*y], R) == [x*y]


def test_integer_ring() -> None:
    # over ZZ SymPy returns the basis over QQ with the denominators cleared
    R, x, y, z = ring("x,y,z", ZZ, lex)
    F = [3*x**2 + y*z - 2, x*y - 5*z**2 + 1, 7*y**2 - z]
    G = modular_groebner(F, R)
    assert G == sympy_groebner(F, R)
    assert all(g.ring == R for g in G)


def test_other_domains_use_sympy() -> None:
    R, x, y = ring("x,y", GF(7), lex)
    F = [x**2 + 3*y, x*y - 1]
    assert modular_groebner(F, R) == sympy_groebner(F, R)
    K = QQ.algebraic_field(sqrt(2))
    R, x, y = ring("x,y", K, lex)
    F = [x**2 - K.from_sympy(sqrt(2))*y, x*y - 1]
    assert modular_groebner(F, R) == sympy_groebner(F, R)
    a = Symbol('a')
    R, x, y = ring("x,y", QQ.frac_field(a), lex)
    F = [x**2 - R.domain.from_sympy(a)*y, x*y - 1]
    assert modular_groebner(F, R) == sympy_groebner(F, R)


# ----------------------------------------------------------------------
# the classical families

def katsura(n: int) -> tuple[list[Expr], tuple[Symbol, ...]]:
    u = symbols('u0:%d' % (n + 1))

    def U(i: int) -> Expr:
        return u[abs(i)] if abs(i) <= n else Integer(0)
    eqs: list[Expr] = [Add(*[U(l)*U(m - l) for l in range(-n, n + 1)]) - U(m) for m in range(n)]
    eqs.append(Add(*[U(l) for l in range(-n, n + 1)]) - 1)
    return [e.expand() for e in eqs], u


def cyclic(n: int) -> tuple[list[Expr], tuple[Symbol, ...]]:
    x = symbols('x0:%d' % n)
    eqs: list[Expr] = [Add(*[Mul(*[x[(i + j) % n] for j in range(k)]) for i in range(n)]) for k in range(1, n)]
    eqs.append(Mul(*x) - 1)
    return eqs, x


def _in_ring(system: tuple[list[Expr], tuple[Symbol, ...]], order: MonomialOrder) -> tuple[list[PolyElement], PolyRing]:
    eqs, gens = system
    R = ring(gens, QQ, order)[0]
    return [R.from_expr(e) for e in eqs], R


def test_classical_families() -> None:
    for system in (katsura(2), katsura(3), cyclic(3), cyclic(4)):
        for order in (grevlex, lex):
            F, R = _in_ring(system, order)
            assert modular_groebner(F, R) == sympy_groebner(F, R)
            assert modular_groebner(F, R, primes=tiny_primes()) == sympy_groebner(F, R)
    for system in (katsura(4), cyclic(5)):
        F, R = _in_ring(system, grevlex)
        G = sympy_groebner(F, R)
        assert modular_groebner(F, R) == G
        # the lexicographic bases take SymPy a minute each: they are
        # converted from the grevlex ones with FGLM (the ideals are
        # zero-dimensional), an independent algorithm
        Flex, Rlex = _in_ring(system, lex)
        expected = sorted(matrix_fglm(G, R, lex), key=lambda g: lex(g.LM), reverse=True)
        assert modular_groebner(Flex, Rlex) == [g.set_ring(Rlex) for g in expected]


# ----------------------------------------------------------------------
# the random audit

def _random_system(rng: random.Random) -> tuple[list[PolyElement], PolyRing]:
    n = rng.randint(2, 4)
    orders = [lex, grevlex, grlex, elimination_order(rng.randint(1, n - 1), n)]
    domain = ZZ if rng.random() < 0.1 else QQ
    R = ring(list('xyzw'[:n]), domain, rng.choice(orders))[0]
    homogeneous = rng.random() < 0.25
    style = rng.choice(['plain', 'plain', 'dense', 'multiples', 'near'])
    polys: list[PolyElement] = []
    for _ in range(rng.randint(1, n + 1)):
        degree = rng.randint(1, 3)
        terms: dict[Monomial, int] = {}
        denominator = 1 if domain == ZZ else rng.choice([1, 1, 1, 2, 3, 5, 6, 7])
        for _ in range(rng.randint(3, 6) if style == 'dense' else rng.randint(1, 4)):
            while True:
                m = tuple(rng.randint(0, degree) for _ in range(n))
                if (sum(m) == degree) if homogeneous else (sum(m) <= degree):
                    break
            if style == 'multiples':
                # coefficients divisible by the first primes
                terms[m] = rng.choice([1, 1, 2, 3, 5, 6, 7, 10, 30, 210, 2310])*rng.choice([-1, 1])
            else:
                terms[m] = rng.randint(-9, 9)
        polys.append(R.from_dict({m: domain(c) for m, c in terms.items()})/denominator if domain == QQ
                     else R.from_dict(terms))
    if style == 'near' and len(polys) >= 2 and polys[0]:
        # two generators which coincide modulo a small prime
        lm = polys[0].LM
        m = lm if homogeneous else tuple(rng.randint(0, 1) for _ in range(n))
        polys[1] = polys[0] + R.from_dict({m: domain(rng.choice([2, 3, 5, 7, 11, 13])*rng.randint(1, 5))})
    return polys, R


def test_random_audit() -> None:
    rng = random.Random(20260921)
    done = unlucky = homogenized = failed = rational = 0
    while done < 300:
        F, R = _random_system(rng)
        F = [f for f in F if f]
        with configure(timeout=2):
            from sympy_extras._timeout import attempt
            expected = attempt(lambda: sympy_groebner(F, R), settings.timeout)
        if expected is None:
            continue
        trace = ModularTrace()
        assert modular_groebner(F, R, primes=tiny_primes(), trace=trace) == expected
        assert modular_groebner(F, R) == expected
        done += 1
        unlucky += bool(trace.unlucky)
        homogenized += trace.homogenized
        failed += trace.failed_verifications
        rational += any(QQ.denom(c) != 1 for g in expected for c in g.values()) if R.domain == QQ else 0
    # the audit meets what it is meant to meet
    assert unlucky > 50 and homogenized > 150 and rational > 100 and failed > 0


# ----------------------------------------------------------------------
# the wiring

def test_ideal_uses_the_modular_algorithm() -> None:
    eqs, gens = katsura(3)
    results = []
    for modular, seconds in ((False, 0.25), (True, 0.25), (True, 0.0)):
        with configure(modular_groebner=modular, groebner_direct_time=seconds):
            I = Ideal(eqs, *gens)
            results.append((I.groebner_basis('lex'), I.groebner_basis(), I.eliminate(gens[:2]).gens,
                            I.dimension(), I.degree()))
    assert results[0] == results[1] == results[2]
    assert settings.modular_groebner is True and settings.groebner_direct_time == 0.25


def test_groebner_dispatch() -> None:
    F, R = _in_ring(katsura(3), lex)
    expected = sympy_groebner(F, R)
    assert groebner(F, R) == expected
    with configure(groebner_direct_time=1e-6):
        # the direct computation is interrupted and the modular one answers
        assert groebner(F, R) == expected
    with configure(modular_groebner=False, groebner_direct_time=1e-6):
        assert groebner(F, R) == expected
    # the graded orders have a longer budget and the same results
    F, R = _in_ring(katsura(3), grevlex)
    assert GRADED_FACTOR > 1
    with configure(groebner_direct_time=1e-7):
        assert groebner(F, R) == sympy_groebner(F, R)

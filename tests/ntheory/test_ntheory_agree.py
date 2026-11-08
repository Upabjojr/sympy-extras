"""The Rust port of ``sympy_extras.ntheory`` against the Python package:
the docstring examples on both implementations, and primality, Jacobi
symbols, prime generation and factorisations of random integers compared
(strong pseudoprimes, Carmichael numbers and prime powers among them)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load


def nt(impl: str) -> Any:
    return load(impl + ".ntheory")


def test_known(impl: str) -> None:
    n = nt(impl)
    pt, ge, fa = load(impl + ".ntheory.primetest"), load(impl + ".ntheory.generate"), load(impl + ".ntheory.factor_")
    assert [k for k in range(30) if n.isprime(k)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert (n.isprime(2**61 - 1), n.isprime(2**61 + 1), n.isprime(3215031751)) == (True, False, False)
    assert (pt.is_strong_probable_prime(3215031751, 2), n.jacobi_symbol(5, 21)) == (True, 1)
    assert (n.nextprime(100), n.nextprime(2), n.prevprime(100), n.prevprime(3)) == (101, 3, 97, 2)
    assert list(ge.primerange(10, 30)) == [11, 13, 17, 19, 23, 29]
    assert 2**1024 - n.prevprime(2**1024) == 105
    big = 2**4*3*1000003*(2**61 - 1)
    assert n.factorint(big) == {2: 4, 3: 1, 1000003: 1, 2305843009213693951: 1}
    assert n.factorint(big, limit=100) == {2: 4, 3: 1, 2305849926742721592081853: 1}
    assert (n.factorint(-12), n.factorint(0), n.factorint(1), n.factorint(-1)) == \
        ({-1: 1, 2: 2, 3: 1}, {0: 1}, {}, {-1: 1})
    assert n.factorint(10403) == {101: 1, 103: 1} and fa.pollard_brent(10403) in (101, 103)
    assert (n.is_carmichael(561), n.is_carmichael(563)) == (True, False)
    assert n.find_carmichael_numbers_in_range(1, 2000) == [561, 1105, 1729]
    assert pt.SMALL_PRIMES[:5] == [2, 3, 5, 7, 11] and pt.SMALL_PRIMES[-1] == 997
    with pytest.raises(ValueError):
        n.jacobi_symbol(3, 10)
    with pytest.raises(ValueError):
        n.prevprime(2)


#: strong pseudoprimes to base 2, Carmichael numbers, Mersenne numbers
SPECIAL = [2047, 3277, 4033, 4681, 8321, 3215031751, 2152302898747, 3474749660383, 341550071728321,
           561, 1105, 1729, 2465, 2821, 6601, 8911, 41041, 825265, 321197185, 2**61 - 1, 2**89 - 1, 2**64 - 59,
           2**64 + 13, 2**127 - 1, (2**31 - 1)**2, 1000003**3]


def test_special_values_agree(rust: Any) -> None:
    py, rs = nt("sympy_extras"), nt("sympy_extras_rs")
    for n in SPECIAL:
        assert rs.isprime(n) == py.isprime(n), n
        assert rs.factorint(n) == py.factorint(n), n
        assert rs.is_carmichael(n) == py.is_carmichael(n), n
        assert rs.perfect_power(n) == py.perfect_power(n), n


@pytest.mark.parametrize("seed", range(10))
def test_random_values_agree(rust: Any, seed: int) -> None:
    py, rs = nt("sympy_extras"), nt("sympy_extras_rs")
    pt_py, pt_rs = load("sympy_extras.ntheory.primetest"), load("sympy_extras_rs.ntheory.primetest")
    rng = random.Random(seed)
    for _ in range(40):
        n = rng.choice([rng.randrange(-10**6, 10**6), rng.randrange(10**12), rng.randrange(2**80)])
        assert rs.isprime(n) == py.isprime(n), n
        if n > 2:
            assert rs.prevprime(n) == py.prevprime(n), n
        assert rs.nextprime(n) == py.nextprime(n), n
        odd = abs(n) | 1
        a = rng.randrange(-10**9, 10**9)
        assert rs.jacobi_symbol(a, odd) == py.jacobi_symbol(a, odd), (a, odd)
        if odd > 2:
            assert pt_rs.is_strong_probable_prime(odd, 2) == pt_py.is_strong_probable_prime(odd, 2), odd
    for _ in range(15):
        # products of primes of every size, with repeated factors
        n = rng.choice([1, -1])
        for _ in range(rng.randint(0, 5)):
            p = py.nextprime(rng.randrange(2, rng.choice([100, 10**4, 10**8, 2**36])))
            n *= p**rng.randint(1, 3)
        assert rs.factorint(n) == py.factorint(n), n
        limit = rng.choice([10, 500, 2000])
        assert rs.factorint(n, limit) == py.factorint(n, limit), (n, limit)
        assert rs.perfect_power(abs(n)) == py.perfect_power(abs(n)), n
    a = rng.randrange(1000)
    assert list(load("sympy_extras_rs.ntheory.generate").primerange(a, a + 500)) == \
        list(load("sympy_extras.ntheory.generate").primerange(a, a + 500))


def test_carmichael_agree(rust: Any) -> None:
    py, rs = nt("sympy_extras"), nt("sympy_extras_rs")
    assert rs.find_carmichael_numbers_in_range(0, 30000) == py.find_carmichael_numbers_in_range(0, 30000)
    assert rs.find_first_n_carmichaels(8) == py.find_first_n_carmichaels(8)

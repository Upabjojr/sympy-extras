"""Primality and factorisation against SymPy (the oracle), known
pseudoprimes and the defining identities."""
from __future__ import annotations

import random

from sympy import factorint as sympy_factorint, isprime as sympy_isprime, nextprime as sympy_nextprime, \
    prevprime as sympy_prevprime

from sympy_extras.ntheory import factorint, isprime, jacobi_symbol, nextprime, perfect_power, prevprime, primerange
from sympy_extras.ntheory.primetest import is_strong_lucas_probable_prime, is_strong_probable_prime


def test_isprime_against_sympy() -> None:
    assert [n for n in range(-5, 5000) if isprime(n)] == [n for n in range(-5, 5000) if sympy_isprime(n)]
    rng = random.Random(1)
    for _ in range(3000):
        n = rng.randrange(1, 2**rng.choice([20, 40, 64, 100, 200]))
        assert isprime(n) == sympy_isprime(n), n


def test_pseudoprimes_are_rejected() -> None:
    # Carmichael numbers and strong pseudoprimes to base 2
    for n in [561, 1105, 1729, 2047, 3277, 4033, 4681, 8321, 3215031751, 2152302898747, 3474749660383,
              341550071728321, 3825123056546413051, 318665857834031151167461]:
        assert not isprime(n), n
    # the first strong base-2 pseudoprimes pass Miller-Rabin and fail Lucas
    for n in [2047, 3277, 4033, 4681, 8321]:
        assert is_strong_probable_prime(n, 2) and not is_strong_lucas_probable_prime(n)
    # strong Lucas pseudoprimes (Selfridge parameters) fail the base-2 test
    for n in [5459, 5777, 10877, 16109, 18971]:
        assert is_strong_lucas_probable_prime(n) and not is_strong_probable_prime(n, 2)
    # Mersenne primes and their neighbours
    for e in [61, 89, 107, 127, 521, 607]:
        assert isprime(2**e - 1) and not isprime(2**e + 1)


def test_jacobi() -> None:
    def euler(a: int, p: int) -> int:
        r = pow(a, (p - 1)//2, p)
        return -1 if r == p - 1 else r
    for p in [3, 5, 7, 11, 101, 1009]:
        for a in range(-20, 40):
            assert jacobi_symbol(a, p) == euler(a % p, p)
    assert jacobi_symbol(2, 15) == 1 and jacobi_symbol(7, 15) == -1 and jacobi_symbol(5, 15) == 0


def test_next_and_previous() -> None:
    for n in list(range(0, 300)) + [10**12, 2**64, 10**30]:
        assert nextprime(n) == sympy_nextprime(n), n
        if n > 2:
            assert prevprime(n) == sympy_prevprime(n), n
    assert 2**1024 - prevprime(2**1024) == 105
    assert list(primerange(90, 110)) == [97, 101, 103, 107, 109]


def test_factorint_against_sympy() -> None:
    rng = random.Random(2)
    for _ in range(300):
        n = rng.randrange(-10**6, 2**rng.choice([16, 32, 48, 64, 80]))
        expected = sympy_factorint(n)
        assert factorint(n) == dict(sorted(expected.items())), n
    for n in [0, 1, -1, 2, 4, 2**64, 3**40, 10**20 + 39, (2**31 - 1)*(2**61 - 1), 1000003**3*999983**2,
              600851475143]:
        assert factorint(n) == dict(sorted(sympy_factorint(n).items())), n
    # the limit convention: trial division only, the cofactor kept whole
    n = 2**4*3*1000003*(2**61 - 1)
    assert factorint(n, limit=100) == dict(sorted(sympy_factorint(n, limit=100).items()))
    for _ in range(100):
        m = rng.randrange(2, 2**40)
        product = 1
        for p, k in factorint(m).items():
            assert isprime(p)
            product *= p**k
        assert product == m


def test_perfect_power() -> None:
    assert perfect_power(2**10) == (2, 10) and perfect_power(6**6) == (6, 6) and perfect_power(10) is None
    assert perfect_power(3**3*5**3) == (15, 3) and perfect_power(1) is None

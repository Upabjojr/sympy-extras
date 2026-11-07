r"""Integer factorisation: trial division, then Pollard's rho method with
Brent's cycle detection, the factors tested by Baillie–PSW
(:func:`~sympy_extras.ntheory.primetest.isprime`).

``factorint(n)`` returns the prime factorisation as a dictionary from
primes to exponents; a negative ``n`` has the factor ``-1``, ``0`` gives
``{0: 1}``. With ``limit``, only the primes up to it are removed by trial
division and the cofactor which remains is returned as one entry (it may
be composite), the convention of SymPy's ``factorint``.

Examples
========

>>> from sympy_extras.ntheory.factor_ import factorint, pollard_brent
>>> factorint(2**4*3*1000003*(2**61 - 1))
{2: 4, 3: 1, 1000003: 1, 2305843009213693951: 1}
>>> factorint(2**4*3*1000003*(2**61 - 1), limit=100)
{2: 4, 3: 1, 2305849926742721592081853: 1}
>>> factorint(-12), factorint(0), factorint(1), factorint(-1)
({-1: 1, 2: 2, 3: 1}, {0: 1}, {}, {-1: 1})
>>> factorint(10403), pollard_brent(10403) in (101, 103)
({101: 1, 103: 1}, True)

References
==========

.. [1] R. P. Brent, *An improved Monte Carlo factorization algorithm*,
       BIT 20 (1980), 176-184.
.. [2] R. Crandall, C. Pomerance, *Prime Numbers: A Computational
       Perspective*, 2nd ed., Springer, 2005, Section 5.2.
"""
from __future__ import annotations

from math import gcd, isqrt
from typing import Optional

from .primetest import SMALL_PRIMES, isprime

__all__ = ['factorint', 'pollard_brent', 'perfect_power', 'is_carmichael', 'find_carmichael_numbers_in_range',
           'find_first_n_carmichaels']


def perfect_power(n: int) -> Optional[tuple[int, int]]:
    """``(b, e)`` with ``b**e = n`` and ``e >= 2`` maximal, for ``n > 1``;
    ``None`` when ``n`` is not a perfect power."""
    if n < 4:
        return None
    best: Optional[tuple[int, int]] = None
    for e in range(2, n.bit_length() + 1):
        b = _integer_root(n, e)
        if b < 2:
            break
        if b**e == n:
            best = (b, e)
    return best


def _integer_root(n: int, e: int) -> int:
    """The integer part of ``n**(1/e)``."""
    if e == 2:
        return isqrt(n)
    low, high = 1, 1 << (n.bit_length()//e + 1)
    while low < high:
        mid = (low + high + 1)//2
        if mid**e <= n:
            low = mid
        else:
            high = mid - 1
    return low


def pollard_brent(n: int, seed: int = 1) -> int:
    """A nontrivial factor of the odd composite ``n`` (which is not a
    prime power) by Pollard's rho with Brent's cycle detection and batched
    gcds, for the maps ``x -> x**2 + c`` with ``c = seed, seed + 1, ...``
    until one succeeds."""
    c = seed
    while True:
        y, r, q = 2, 1, 1
        g = 1
        x = ys = y
        batch = 128
        while g == 1:
            x = y
            for _ in range(r):
                y = (y*y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(batch, r - k)):
                    y = (y*y + c) % n
                    q = q*abs(x - y) % n
                g = gcd(q, n)
                k += batch
            r *= 2
        if g == n:
            # the batch overshot: step back one at a time
            g = 1
            while g == 1:
                ys = (ys*ys + c) % n
                g = gcd(abs(x - ys), n)
        if g != n:
            return g
        c += 1


def _factor_into(n: int, factors: dict[int, int], multiplicity: int) -> None:
    """Add the prime factors of ``n > 1`` (with no prime factor below
    1000) to ``factors``, each ``multiplicity`` times."""
    if n == 1:
        return
    if isprime(n):
        factors[n] = factors.get(n, 0) + multiplicity
        return
    power = perfect_power(n)
    if power is not None:
        _factor_into(power[0], factors, multiplicity*power[1])
        return
    d = pollard_brent(n)
    k = 0
    while n % d == 0:
        n //= d
        k += 1
    _factor_into(d, factors, multiplicity*k)
    _factor_into(n, factors, multiplicity)


def factorint(n: int, limit: Optional[int] = None) -> dict[int, int]:
    """The prime factorisation of ``n`` as ``{prime: exponent}``, sorted
    by the primes (``-1`` first for a negative ``n``); with ``limit``,
    trial division by the primes up to ``limit`` only, the cofactor kept
    whole."""
    if n == 0:
        return {0: 1}
    factors: dict[int, int] = {}
    if n < 0:
        factors[-1] = 1
        n = -n
    bound = 997 if limit is None else limit
    for p in SMALL_PRIMES:
        if p > bound:
            break
        if n % p == 0:
            k = 0
            while n % p == 0:
                n //= p
                k += 1
            factors[p] = k
    if limit is not None and limit > 997:
        p = 1001
        while p <= limit and p*p <= n:
            if n % p == 0:
                k = 0
                while n % p == 0:
                    n //= p
                    k += 1
                factors[p] = k
            p += 2
        if n > 1 and p*p > n:
            factors[n] = factors.get(n, 0) + 1
            n = 1
    if n > 1:
        if limit is None:
            _factor_into(n, factors, 1)
        else:
            factors[n] = factors.get(n, 0) + 1
    return dict(sorted(factors.items()))



# PORTED FROM SYMPY: sympy/ntheory/factor_.py:is_carmichael
def is_carmichael(n: int) -> bool:
    """Whether ``n`` is a Carmichael number (Korselt's criterion: odd,
    composite, squarefree, ``p - 1 | n - 1`` for its prime factors).

    >>> from sympy_extras.ntheory import is_carmichael
    >>> is_carmichael(561), is_carmichael(563)
    (True, False)
    """
    if n < 561:
        return False
    return bool(n % 2) and not isprime(n) and all(e == 1 and (n - 1) % (p - 1) == 0 for p, e in factorint(n).items())


# PORTED FROM SYMPY: sympy/ntheory/factor_.py:find_carmichael_numbers_in_range
def find_carmichael_numbers_in_range(x: int, y: int) -> list[int]:
    """The Carmichael numbers in ``[x, y)``.

    >>> from sympy_extras.ntheory import find_carmichael_numbers_in_range
    >>> find_carmichael_numbers_in_range(1, 2000)
    [561, 1105, 1729]
    """
    if not 0 <= x <= y:
        raise ValueError('The provided range is not valid. x and y must be non-negative integers and x <= y')
    start = x + 1 if x % 2 == 0 else x
    return [i for i in range(start, y, 2) if is_carmichael(i)]


# PORTED FROM SYMPY: sympy/ntheory/factor_.py:find_first_n_carmichaels
def find_first_n_carmichaels(n: int) -> list[int]:
    """The first ``n`` Carmichael numbers."""
    i = 561
    found: list[int] = []
    while len(found) < n:
        if is_carmichael(i):
            found.append(i)
        i += 2
    return found

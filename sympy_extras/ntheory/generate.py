"""The next and the previous prime.

Examples
========

>>> from sympy_extras.ntheory.generate import nextprime, prevprime, primerange
>>> nextprime(100), nextprime(2), prevprime(100), prevprime(3)
(101, 3, 97, 2)
>>> list(primerange(10, 30))
[11, 13, 17, 19, 23, 29]
>>> 2**1024 - prevprime(2**1024)
105
"""
from __future__ import annotations

from typing import Iterator

from .primetest import isprime

__all__ = ['nextprime', 'prevprime', 'primerange']


def nextprime(n: int) -> int:
    """The least prime greater than ``n``."""
    if n < 2:
        return 2
    candidate = n + 1 if n % 2 == 0 else n + 2
    while not isprime(candidate):
        candidate += 2
    return candidate


def prevprime(n: int) -> int:
    """The largest prime less than ``n`` (``n > 2``)."""
    if n <= 2:
        raise ValueError('no prime below %d' % n)
    if n == 3:
        return 2
    candidate = n - 1 if n % 2 == 0 else n - 2
    while not isprime(candidate):
        candidate -= 2
    return candidate


def primerange(a: int, b: int) -> Iterator[int]:
    """The primes ``p`` with ``a <= p < b``, in increasing order."""
    p = nextprime(a - 1)
    while p < b:
        yield p
        p = nextprime(p)

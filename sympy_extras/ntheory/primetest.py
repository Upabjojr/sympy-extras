r"""Primality testing: the Baillie–PSW test.

``isprime(n)`` runs trial division by the small primes, a strong
Fermat (Miller–Rabin) test to base 2 and a strong Lucas test with
Selfridge's parameters (the first $D$ in $5, -7, 9, -11, \ldots$ with
Jacobi symbol $(D/n) = -1$, $P = 1$, $Q = (1 - D)/4$). No composite
number passing both is known; every composite below $2^{64}$ fails, so
the test is deterministic in that range (Feitsma–Galway's list of base-2
strong pseudoprimes, all of which fail the Lucas test).

Examples
========

>>> from sympy_extras.ntheory.primetest import isprime, is_strong_probable_prime, jacobi_symbol
>>> [n for n in range(30) if isprime(n)]
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
>>> isprime(2**61 - 1), isprime(2**61 + 1), isprime(3215031751)
(True, False, False)
>>> is_strong_probable_prime(3215031751, 2), jacobi_symbol(5, 21)
(True, 1)

References
==========

.. [1] R. Baillie, S. S. Wagstaff, *Lucas pseudoprimes*, Mathematics of
       Computation 35 (1980), 1391-1417.
.. [2] C. Pomerance, J. L. Selfridge, S. S. Wagstaff, *The
       pseudoprimes to 25 * 10^9*, Mathematics of Computation 35 (1980),
       1003-1026.
.. [3] R. Crandall, C. Pomerance, *Prime Numbers: A Computational
       Perspective*, 2nd ed., Springer, 2005, Sections 3.5 and 3.6.
"""
from __future__ import annotations

from math import isqrt

__all__ = ['isprime', 'is_strong_probable_prime', 'is_strong_lucas_probable_prime', 'jacobi_symbol',
           'SMALL_PRIMES']

#: the primes below 1000, for trial division
SMALL_PRIMES: list[int] = [p for p in range(2, 1000) if all(p % q for q in range(2, isqrt(p) + 1))]


def jacobi_symbol(a: int, n: int) -> int:
    """The Jacobi symbol ``(a/n)`` for odd ``n > 0`` (quadratic
    reciprocity and the rule for 2)."""
    if n <= 0 or n % 2 == 0:
        raise ValueError('the Jacobi symbol needs an odd positive modulus, got %d' % n)
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def is_strong_probable_prime(n: int, base: int) -> bool:
    """Whether the odd ``n > 2`` is a strong probable prime to ``base``:
    with ``n - 1 = 2**s d``, ``d`` odd, ``base**d = 1`` or
    ``base**(2**r d) = -1`` modulo ``n`` for some ``r < s``."""
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(base, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = x*x % n
        if x == n - 1:
            return True
    return False


def is_strong_lucas_probable_prime(n: int) -> bool:
    """The strong Lucas test with Selfridge's parameters, for odd ``n``
    which is not a perfect square: with ``n + 1 = 2**s d``, ``d`` odd,
    ``U_d = 0`` or ``V_(2**r d) = 0`` modulo ``n`` for some ``r < s``."""
    D = 5
    while True:
        j = jacobi_symbol(D, n)
        if j == -1:
            break
        if j == 0 and abs(D) != n:
            return False
        D = -D - 2 if D > 0 else -D + 2
    P, Q = 1, (1 - D)//4
    d = n + 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # U_k, V_k, Q**k by the binary expansion of d (doubling and adding one)
    U, V, Qk = 0, 2, 1
    inverse_two = (n + 1)//2
    for bit in bin(d)[2:]:
        # doubling: U_2k = U_k V_k, V_2k = V_k**2 - 2 Q**k
        U, V, Qk = U*V % n, (V*V - 2*Qk) % n, Qk*Qk % n
        if bit == '1':
            # k -> k + 1: U_(k+1) = (P U + V)/2, V_(k+1) = (D U + P V)/2
            U, V = (P*U + V)*inverse_two % n, (D*U + P*V)*inverse_two % n
            Qk = Qk*Q % n
    if U == 0 or V == 0:
        return True
    for _ in range(s - 1):
        V, Qk = (V*V - 2*Qk) % n, Qk*Qk % n
        if V == 0:
            return True
    return False


def isprime(n: int) -> bool:
    """Whether ``n`` is prime (Baillie–PSW; deterministic below
    ``2**64``)."""
    if n < 2:
        return False
    for p in SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    if n < 1000*1000:
        return True
    if not is_strong_probable_prime(n, 2):
        return False
    root = isqrt(n)
    if root*root == n:
        return False
    return is_strong_lucas_probable_prime(n)

"""Integer functions: gcd, lcm, modular inverse, integer roots and
logarithms, digit counts (``sympy.core.intfunc`` and the integer helpers
of ``sympy.core.numbers``).

References
==========

.. [1] D. E. Knuth, *The Art of Computer Programming*, vol. 2, section
       4.5.2 (the Euclidean algorithm and its extension).
.. [2] SymPy Development Team, *SymPy documentation: integer functions*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.intfunc
"""
from __future__ import annotations

from math import gcd, isqrt, lcm

__all__ = ['igcd', 'ilcm', 'igcdex', 'mod_inverse', 'integer_nthroot', 'integer_log', 'num_digits',
    'trailing', 'isqrt']


def igcd(*args: int) -> int:
    """The nonnegative greatest common divisor of integers.

    Examples
    ========

    >>> from sympy_extras.core import igcd
    >>> igcd(12, 18, 30), igcd(-4, 6), igcd(0, 0)
    (6, 2, 0)
    """
    result = 0
    for a in args:
        result = gcd(result, a)
    return result


def ilcm(*args: int) -> int:
    """The nonnegative least common multiple of integers (``0`` when one is zero).

    Examples
    ========

    >>> from sympy_extras.core import ilcm
    >>> ilcm(4, 6), ilcm(0, 3), ilcm(2, 3, 5)
    (12, 0, 30)
    """
    result = 1
    for a in args:
        result = lcm(result, a)
    return result


def igcdex(a: int, b: int) -> tuple[int, int, int]:
    """``(x, y, g)`` with ``a*x + b*y == g == gcd(a, b)`` (the extended
    Euclidean algorithm).

    Examples
    ========

    >>> from sympy_extras.core.intfunc import igcdex
    >>> igcdex(10, 6), igcdex(2, 3), igcdex(0, 5)
    ((-1, 2, 2), (-1, 1, 1), (0, 1, 5))
    """
    if a == 0 and b == 0:
        return (0, 1, 0)
    sign_a, sign_b = (-1 if a < 0 else 1), (-1 if b < 0 else 1)
    a, b = abs(a), abs(b)
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b:
        q, r = divmod(a, b)
        a, b = b, r
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return (sign_a * x0, sign_b * y0, a)


def mod_inverse(a: int, m: int) -> int:
    """The inverse of ``a`` modulo ``m``, in ``[1, m)`` for a positive
    ``m``; ``ValueError`` when ``a`` and ``m`` are not coprime.

    Examples
    ========

    >>> from sympy_extras.core import mod_inverse
    >>> mod_inverse(3, 11), mod_inverse(-3, 11)
    (4, 7)
    """
    x, _, g = igcdex(a, m)
    if g != 1:
        raise ValueError("inverse of %d (mod %d) does not exist" % (a, m))
    if m > 1:
        return x % m
    return -(x % -m) if m < -1 else x


def integer_nthroot(y: int, n: int) -> tuple[int, bool]:
    """The integer part of the ``n``-th root of ``y >= 0`` and whether the
    root is exact.

    Examples
    ========

    >>> from sympy_extras.core import integer_nthroot
    >>> integer_nthroot(16, 2), integer_nthroot(17, 2), integer_nthroot(27, 3), integer_nthroot(0, 2)
    ((4, True), (4, False), (3, True), (0, True))
    """
    if y < 0:
        raise ValueError("y must be nonnegative")
    if n < 1:
        raise ValueError("n must be positive")
    if y in (0, 1) or n == 1:
        return (y, True)
    if n == 2:
        r = isqrt(y)
        return (r, r * r == y)
    # Newton's iteration on the integers from an upper bound
    r = 1 << ((y.bit_length() + n - 1) // n)
    while True:
        s = ((n - 1) * r + y // r ** (n - 1)) // n
        if s >= r:
            break
        r = s
    while r ** n > y:
        r -= 1
    while (r + 1) ** n <= y:
        r += 1
    return (r, r ** n == y)


def integer_log(y: int, b: int) -> tuple[int, bool]:
    """The largest ``e`` with ``b**e <= y`` and whether ``b**e == y``, for
    ``y >= 1`` and ``b >= 2``.

    Examples
    ========

    >>> from sympy_extras.core import integer_log
    >>> integer_log(100, 10), integer_log(99, 10), integer_log(1, 2), integer_log(7, 2)
    ((2, True), (1, False), (0, True), (2, False))
    """
    if y < 1:
        raise ValueError("y must be positive")
    if b < 2:
        raise ValueError("b must be at least 2")
    e = 0
    power = 1
    while power * b <= y:
        power *= b
        e += 1
    return (e, power == y)


def num_digits(n: int, base: int = 10) -> int:
    """The number of digits of ``|n|`` in the given base (``1`` for zero).

    Examples
    ========

    >>> from sympy_extras.core import num_digits
    >>> num_digits(1234), num_digits(255, 2), num_digits(0)
    (4, 8, 1)
    """
    n = abs(n)
    if base == 2:
        return max(1, n.bit_length())
    digits = 0
    while n:
        n //= base
        digits += 1
    return max(1, digits)


def trailing(n: int) -> int:
    """The number of trailing zero bits of ``n`` (``0`` for zero).

    Examples
    ========

    >>> from sympy_extras.core import trailing
    >>> trailing(8), trailing(12), trailing(0), trailing(7)
    (3, 2, 0, 0)
    """
    if n == 0:
        return 0
    return (n & -n).bit_length() - 1

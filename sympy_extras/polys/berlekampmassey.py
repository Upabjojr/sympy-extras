r"""The Berlekamp–Massey algorithm: the shortest linear recurrence of a
sequence over any field.

For a sequence $s_0, s_1, \dots, s_{N-1}$ in a field, the algorithm finds
the monic polynomial $C(x) = x^L + c_1 x^{L-1} + \dots + c_L$ of least
degree $L$ with

.. math:: s_{k+L} + c_1 s_{k+L-1} + \dots + c_L s_k = 0, \qquad 0 \le k < N - L,

in $O(N^2)$ field operations, by correcting the current recurrence with a
multiple of the last one which failed whenever a discrepancy appears
(Massey's formulation of Berlekamp's decoding algorithm). When the
sequence does satisfy a recurrence of order $L$ and $N \ge 2L$, the
result is its minimal polynomial: the characteristic polynomial of the
shortest linear feedback shift register, the denominator of the rational
generating function, and, for the sequence $u^T A^k v$ of a matrix, a
divisor of its minimal polynomial (Wiedemann).

The domain is any field of the typed layer: the rationals, $GF(p)$,
$GF(p^n)$, number fields.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.polys.domains import GF, QQ
>>> from sympy_extras.polys.berlekampmassey import berlekamp_massey, extend_sequence
>>> berlekamp_massey([Fraction(v) for v in (0, 1, 1, 2, 3, 5, 8, 13)], QQ) == [1, -1, -1]      # x**2 - x - 1
True
>>> berlekamp_massey([1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1], GF(2))                     # x**3 + x + 1
[1, 0, 1, 1]
>>> extend_sequence([Fraction(v) for v in (0, 1, 1, 2, 3, 5, 8, 13)], QQ, 11)[-3:] == [21, 34, 55]
True

References
==========

.. [1] J. L. Massey, *Shift-register synthesis and BCH decoding*, IEEE
       Transactions on Information Theory 15 (1969), 122-127.
.. [2] J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
       Cambridge University Press, 2013, Section 12.3.
.. [3] D. H. Wiedemann, *Solving sparse linear equations over finite
       fields*, IEEE Transactions on Information Theory 32 (1986), 54-62.
"""
from __future__ import annotations

from typing import Sequence, TypeVar

from .domains.domain import Domain

__all__ = ['berlekamp_massey', 'extend_sequence', 'satisfies_recurrence']

T = TypeVar('T')


def berlekamp_massey(sequence: Sequence[T], K: Domain[T]) -> list[T]:
    """The monic polynomial of least degree ``L`` (dense, highest degree
    first) whose coefficients ``1, c_1, ..., c_L`` give a recurrence
    ``s[k + L] + c_1 s[k + L - 1] + ... + c_L s[k] = 0`` satisfied by the
    whole sequence, over the field ``K``.

    The zero sequence gives ``[1]`` (the recurrence ``s[k] = 0``).

    Examples
    ========

    >>> from sympy_extras.polys.domains import GF
    >>> from sympy_extras.polys.berlekampmassey import berlekamp_massey
    >>> berlekamp_massey([1, 2, 4, 3, 1, 2, 4, 3], GF(5))            # s[k + 1] = 2 s[k]
    [1, 3]
    >>> berlekamp_massey([0, 0, 0], GF(5)), len(berlekamp_massey([0, 0, 1], GF(5))) - 1     # no shorter than the spike
    ([1], 3)
    """
    # connection polynomials, lowest degree first: sum_i c[i] s[k - i] = 0
    current: list[T] = [K.one]
    previous: list[T] = [K.one]
    length = 0
    shift = 1
    last = K.one
    for k in range(len(sequence)):
        discrepancy = sequence[k]
        for i in range(1, length + 1):
            if i < len(current):
                discrepancy = K.add(discrepancy, K.mul(current[i], sequence[k - i]))
        if K.is_zero(discrepancy):
            shift += 1
            continue
        factor = K.quo(discrepancy, last)
        corrected = list(current) + [K.zero]*max(0, len(previous) + shift - len(current))
        for i, c in enumerate(previous):
            corrected[i + shift] = K.sub(corrected[i + shift], K.mul(factor, c))
        if 2*length <= k:
            previous = current
            last = discrepancy
            length = k + 1 - length
            shift = 1
        else:
            shift += 1
        current = corrected
    # x**L + c_1 x**(L - 1) + ... + c_L
    coefficients = list(current) + [K.zero]*(length + 1 - len(current))
    return coefficients[:length + 1]


def satisfies_recurrence(sequence: Sequence[T], polynomial: Sequence[T], K: Domain[T]) -> bool:
    """Whether the sequence satisfies the recurrence of the polynomial
    ``c_0 x**L + ... + c_L``: ``sum_i c_i s[k + L - i] = 0`` for every
    ``k``.

    Examples
    ========

    >>> from sympy_extras.polys.domains import GF
    >>> from sympy_extras.polys.berlekampmassey import satisfies_recurrence
    >>> satisfies_recurrence([1, 2, 4, 3, 1], [1, 3], GF(5)), satisfies_recurrence([1, 2, 4, 3, 2], [1, 3], GF(5))
    (True, False)
    """
    order = len(polynomial) - 1
    for k in range(len(sequence) - order):
        total = K.zero
        for i, c in enumerate(polynomial):
            total = K.add(total, K.mul(c, sequence[k + order - i]))
        if not K.is_zero(total):
            return False
    return True


def extend_sequence(sequence: Sequence[T], K: Domain[T], count: int) -> list[T]:
    """The sequence continued to ``count`` terms by its shortest
    recurrence.

    Examples
    ========

    >>> from sympy_extras.polys.domains import GF
    >>> from sympy_extras.polys.berlekampmassey import extend_sequence
    >>> extend_sequence([1, 1, 0, 1, 1, 0], GF(2), 9)
    [1, 1, 0, 1, 1, 0, 1, 1, 0]
    """
    polynomial = berlekamp_massey(sequence, K)
    order = len(polynomial) - 1
    result = list(sequence)
    while len(result) < count:
        total = K.zero
        for i in range(1, order + 1):
            total = K.sub(total, K.mul(polynomial[i], result[len(result) - i]))
        result.append(total)
    return result

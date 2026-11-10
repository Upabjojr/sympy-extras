r"""Recognition of algebraic numbers from approximations, by lattice
reduction.

Given a rational approximation $z \approx \alpha$ good to $D$ decimal
digits, an integer polynomial of degree at most $d$ with small
coefficients which nearly vanishes at $z$ is a short vector of the
lattice spanned by the rows

.. math:: (e_i,\ \lfloor C \operatorname{Re} z^i \rceil,\ \lfloor C
          \operatorname{Im} z^i \rceil), \qquad i = 0, \dots, d,

with $C = 10^D$: the vector $\sum c_i\, \mathrm{row}_i$ has the entries
$c_i$ followed by $C\, p(z)$ up to rounding, for $p = \sum c_i x^i$
(Kannan, Lenstra and Lovász). The LLL algorithm
(:mod:`sympy_extras.polys.lattice`) finds such a vector, and it is
accepted only when it is much shorter than what a number without
relation would give ($C^{k/(d+1)}$ with $k$ the number of constraints,
one for a real number and two otherwise): then the root of $p$ nearest
to $z$ is returned as an exact :class:`~sympy_extras.polys.numberfields.
qqbar.Algebraic`, provided it agrees with $z$ to the digits given.

The answer is a guess supported by the digits supplied, not a proof.

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.polys.numberfields.recognize import recognize_algebraic
>>> recognize_algebraic(Fraction(1414213562373095, 10**15), digits=15)
Algebraic([1, 0, -2], 1.414213562)
>>> recognize_algebraic(Fraction(3146264369941972, 10**15), digits=15, max_degree=4)      # sqrt(2) + sqrt(3)
Algebraic([1, 0, -10, 0, 1], 3.14626437)
>>> recognize_algebraic(Fraction(-1, 2), Fraction(866025403784439, 10**15), digits=15)
Algebraic([1, 1, 1], -0.5 + 0.8660254038*I)
>>> recognize_algebraic(Fraction(3141592653589793, 10**15), digits=15, max_degree=4) is None      # pi
True

References
==========

.. [1] R. Kannan, A. K. Lenstra, L. Lovász, *Polynomial factorization and
       nonrandomness of bits of algebraic and some transcendental
       numbers*, Mathematics of Computation 50 (1988), 235-250.
.. [2] H. Cohen, *A Course in Computational Algebraic Number Theory*,
       Springer, 1993, Section 2.7.2 (linear and algebraic dependence
       using LLL).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from sympy_extras.polys.lattice import lll
from sympy_extras.polys.numberfields.qqbar import Algebraic, algebraic_roots

__all__ = ['recognize_algebraic']


def _rounded(q: Fraction) -> int:
    return (2*q.numerator + q.denominator)//(2*q.denominator)


def recognize_algebraic(re: Fraction, im: Fraction = Fraction(0), digits: int = 15, max_degree: int = 6
                        ) -> Optional[Algebraic]:
    """An algebraic number of degree at most ``max_degree`` with small
    minimal polynomial which agrees with ``re + im*i`` to ``digits``
    decimal digits, or ``None`` when no convincing one exists.

    Parameters
    ==========

    re, im : the approximation (an exact rational is recognised as
        itself).
    digits : how many decimal digits after the point are correct.
    max_degree : the largest degree tried; the degrees are tried in
        increasing order, a degree ``d`` needing about ``3 (d + 1)``
        digits for coefficients of two or three digits.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.numberfields.recognize import recognize_algebraic
    >>> recognize_algebraic(Fraction(1618033988749895, 10**15))        # the golden ratio
    Algebraic([1, -1, -1], 1.618033989)
    >>> recognize_algebraic(Fraction(3, 7))
    Algebraic([7, -3], 3/7)
    """
    if digits < 1 or max_degree < 1:
        raise ValueError('digits and max_degree must be positive')
    re, im = Fraction(re), Fraction(im)
    scale = 10**digits
    tolerance = Fraction(10, scale)
    constraints = 1 if im == 0 else 2
    for degree in range(1, max_degree + 1):
        rows: list[list[int]] = []
        power_re, power_im = Fraction(1), Fraction(0)
        for i in range(degree + 1):
            row = [1 if j == i else 0 for j in range(degree + 1)]
            row.append(_rounded(scale*power_re))
            if constraints == 2:
                row.append(_rounded(scale*power_im))
            rows.append(row)
            power_re, power_im = power_re*re - power_im*im, power_re*im + power_im*re
        vector = lll(rows)[0]
        coefficients = vector[:degree + 1]
        # a number without relation gives vectors of length about
        # scale**(constraints/(degree + 1 + constraints)); a relation is believed when it
        # is ten times shorter
        norm_squared = sum(c*c for c in vector)
        if norm_squared**(degree + 1 + constraints)*10**(2*(degree + 1 + constraints)) > scale**(2*constraints):
            continue
        polynomial = coefficients[::-1]
        while polynomial and polynomial[0] == 0:
            polynomial = polynomial[1:]
        if len(polynomial) < 2:
            continue
        for root in algebraic_roots(polynomial):
            box = root.enclosure(tolerance/4)
            if (abs((box[0] + box[1])/2 - re) < tolerance and abs((box[2] + box[3])/2 - im) < tolerance
                    and root.is_real == (im == 0)):
                return root
    return None

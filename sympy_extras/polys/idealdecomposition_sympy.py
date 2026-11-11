"""The SymPy face of :mod:`sympy_extras.polys.idealdecomposition`: radicals,
prime components and primary decompositions of polynomial ideals of any
dimension, on the ideals of :mod:`sympy_extras.polys.ideals_sympy`.

The functions take and return the ideals of the face (SymPy expressions,
symbols and domains), as before the native port; the work is done by the
native module on the native ideal behind the face (the coefficients
converted term by term). Nothing of the computation stays on SymPy.

Examples
========

>>> from sympy.abc import x, y, z
>>> from sympy_extras.polys.ideals_sympy import Ideal
>>> from sympy_extras.polys.idealdecomposition_sympy import minimal_primes, primary_decomposition
>>> minimal_primes(Ideal([x*z - y**2, x**2 - y*z], x, y, z))
[Ideal([x - z, y - z], x, y, z), Ideal([x, y], x, y, z), Ideal([y**2 + y*z + z**2, x + y + z], x, y, z)]
>>> primary_decomposition(Ideal([x**2, x*y], x, y))
[(Ideal([x], x, y), Ideal([x], x, y)), (Ideal([x**2, y], x, y), Ideal([x, y], x, y))]
"""
from __future__ import annotations

from sympy.core.symbol import Symbol

from .ideals_sympy import Ideal

__all__ = ['radical', 'equidimensional_parts', 'minimal_primes', 'primary_decomposition',
           'associated_primes', 'is_primary', 'independent_set', 'PrimaryComponent']

#: a primary ideal and its associated prime
PrimaryComponent = tuple[Ideal, Ideal]


def radical(ideal: Ideal) -> Ideal:
    """:func:`sympy_extras.polys.idealdecomposition.radical` on the ideals
    of the face: the radical of an ideal of a polynomial ring over the
    rationals, through squarefree regular chains in every dimension.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.ideals_sympy import Ideal
    >>> from sympy_extras.polys.idealdecomposition_sympy import radical
    >>> radical(Ideal([x**2*y, x*y**2], x, y))
    Ideal([x*y], x, y)
    >>> radical(Ideal([(x**2 + y**2 - z**2)**2, (x - y)**3*(x**2 + y**2 - z**2)], x, y, z))
    Ideal([x**2 + y**2 - z**2], x, y, z)
    """
    return ideal._wrap(ideal._native.chain_radical())


def equidimensional_parts(ideal: Ideal) -> list[Ideal]:
    """:func:`sympy_extras.polys.idealdecomposition.equidimensional_parts`
    on the ideals of the face: the equidimensional parts of the radical of
    the ideal, by decreasing dimension.

    Examples
    ========

    A plane, a line and a point:

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.ideals_sympy import Ideal
    >>> from sympy_extras.polys.idealdecomposition_sympy import equidimensional_parts
    >>> equidimensional_parts(Ideal([x*z*(z - 1), x*y*(z - 1), x*(x - 1)*(z - 1)], x, y, z))
    [Ideal([x*z - x], x, y, z), Ideal([x - 1, y, z], x, y, z)]
    >>> equidimensional_parts(Ideal([x*z, x*y, x*(x - 1)*(z - 1)], x, y, z))
    [Ideal([x], x, y, z), Ideal([x - 1, y, z], x, y, z)]
    """
    return ideal.equidimensional_parts()


def minimal_primes(ideal: Ideal) -> list[Ideal]:
    """:func:`sympy_extras.polys.idealdecomposition.minimal_primes` on the
    ideals of the face: the minimal primes of the ideal, by decreasing
    dimension.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.ideals_sympy import Ideal
    >>> from sympy_extras.polys.idealdecomposition_sympy import minimal_primes
    >>> minimal_primes(Ideal([x*z - y**2, x**2 - y*z], x, y, z))
    [Ideal([x - z, y - z], x, y, z), Ideal([x, y], x, y, z), Ideal([y**2 + y*z + z**2, x + y + z], x, y, z)]
    >>> minimal_primes(Ideal([x**2 - 2*y**2], x, y))
    [Ideal([x**2 - 2*y**2], x, y)]
    """
    return ideal.minimal_primes()


def independent_set(ideal: Ideal) -> list[Symbol]:
    """:func:`sympy_extras.polys.idealdecomposition.independent_set` on the
    ideals of the face: a maximal independent set of variables modulo a
    proper ideal.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.ideals_sympy import Ideal
    >>> from sympy_extras.polys.idealdecomposition_sympy import independent_set
    >>> independent_set(Ideal([x*y, x*z], x, y, z))
    [y, z]
    >>> independent_set(Ideal([x**2 - y, z], x, y, z))
    [y]
    """
    return [ideal.symbols[i] for i in ideal._native.independent_set()]


def primary_decomposition(ideal: Ideal) -> list[PrimaryComponent]:
    """:func:`sympy_extras.polys.idealdecomposition.primary_decomposition`
    on the ideals of the face: an irredundant primary decomposition of an
    ideal of a polynomial ring over the rationals, as pairs ``(Q, P)`` of a
    primary ideal and its associated prime, by decreasing dimension of the
    primes (Gianni–Trager–Zacharias with the separators of
    Shimoyama–Yokoyama).

    Examples
    ========

    The ``y`` axis with an embedded point at the origin:

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.ideals_sympy import Ideal
    >>> from sympy_extras.polys.idealdecomposition_sympy import primary_decomposition
    >>> primary_decomposition(Ideal([x**2, x*y], x, y))
    [(Ideal([x], x, y), Ideal([x], x, y)), (Ideal([x**2, y], x, y), Ideal([x, y], x, y))]

    Three lines, one of them doubled:

    >>> for Q, P in primary_decomposition(Ideal([x*z, y**2*z, (x - 1)**2*y], x, y, z)):
    ...     print(Q, P)
    Ideal([x**2 - 2*x + 1, z], x, y, z) Ideal([x - 1, z], x, y, z)
    Ideal([x, y], x, y, z) Ideal([x, y], x, y, z)
    Ideal([y, z], x, y, z) Ideal([y, z], x, y, z)
    """
    return ideal.primary_decomposition()


def associated_primes(ideal: Ideal) -> list[Ideal]:
    """:func:`sympy_extras.polys.idealdecomposition.associated_primes` on
    the ideals of the face: the associated primes, minimal and embedded, by
    decreasing dimension.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.ideals_sympy import Ideal
    >>> from sympy_extras.polys.idealdecomposition_sympy import associated_primes
    >>> associated_primes(Ideal([x**2, x*y], x, y))
    [Ideal([x], x, y), Ideal([x, y], x, y)]
    >>> associated_primes(Ideal([x*y, x*z, y*z], x, y, z))
    [Ideal([x, y], x, y, z), Ideal([x, z], x, y, z), Ideal([y, z], x, y, z)]
    """
    return ideal.associated_primes()


def is_primary(ideal: Ideal) -> bool:
    """:func:`sympy_extras.polys.idealdecomposition.is_primary` on the
    ideals of the face: whether the ideal is primary.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.ideals_sympy import Ideal
    >>> from sympy_extras.polys.idealdecomposition_sympy import is_primary
    >>> is_primary(Ideal([x**2, y**3], x, y)), is_primary(Ideal([x**2, x*y], x, y))
    (True, False)
    >>> is_primary(Ideal([x**2, y], x, y, z)), is_primary(Ideal([x*y], x, y))
    (True, False)
    """
    return ideal.is_primary()

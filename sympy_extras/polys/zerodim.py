r"""Exact solutions of polynomial systems with finitely many solutions.

:func:`solve_zero_dimensional` returns *all* the complex solutions of a
system of polynomial equations with rational coefficients which has
finitely many, each coordinate an exact algebraic number
(:class:`~sympy_extras.polys.numberfields.qqbar.Algebraic`), so that the
solutions can be counted, compared, tested for reality and refined to
any precision.

The method is the classical one of the shape lemma:

1. a Gröbner basis for a graded order (F4) decides that the ideal $I$ is
   zero-dimensional;
2. the ideal is replaced by its radical, which has the same solutions:
   for each variable the monic generator of $I \cap \mathbb{Q}[x_i]$ is
   found by linear algebra on the normal forms of the powers of $x_i$,
   and its square-free part is added to the system (Seidenberg's lemma:
   over a perfect field the result is radical);
3. the lexicographic basis is obtained by the change of order
   (:mod:`sympy_extras.polys.fglm`). For a radical ideal in general
   position it has the *shape* $x_1 - g_1(x_n), \dots, x_{n-1} -
   g_{n-1}(x_n), h(x_n)$ with $h$ square-free; when it does not, the last
   variable is replaced by $x_n + \sum c_i x_i$ for small integers $c_i$
   until it does (the last coordinate must separate the solutions, and
   all but finitely many directions do), and the solutions are brought
   back to the original coordinates;
4. the roots $t$ of $h$ are isolated exactly, and the coordinates
   $g_i(t)$ are algebraic numbers computed by one resultant each.

Examples
========

>>> from sympy_extras.polys.domains import QQ
>>> from sympy_extras.polys.rings import PolyRing, lex
>>> from sympy_extras.polys.zerodim import solve_zero_dimensional
>>> R = PolyRing(QQ, ['x', 'y'], lex)
>>> x, y = R.gens
>>> solve_zero_dimensional([x*x + y*y - R.from_int(5), x*y - R.from_int(2)], R)
[[Algebraic([1, 1], -1), Algebraic([1, 2], -2)], [Algebraic([1, 2], -2), Algebraic([1, 1], -1)], [Algebraic([1, -2], 2), Algebraic([1, -1], 1)], [Algebraic([1, -1], 1), Algebraic([1, -2], 2)]]
>>> solutions = solve_zero_dimensional([x*x + y*y - R.from_int(1), y - x*x], R)       # a circle and a parabola
>>> len(solutions), sum(1 for s in solutions if s[0].is_real and s[1].is_real)
(4, 2)
>>> [s for s in solutions if s[0].is_real and s[0] > 0]
[[Algebraic([1, 0, 1, 0, -1], 0.7861513778), Algebraic([1, 1, -1], 0.6180339887)]]

References
==========

.. [1] P. Gianni, T. Mora, *Algebraic solution of systems of polynomial
       equations using Gröbner bases*, AAECC-5, LNCS 356, Springer, 1989,
       247-257 (the shape lemma).
.. [2] A. Seidenberg, *Constructions in algebra*, Transactions of the AMS
       197 (1974), 273-313 (Lemma 92).
.. [3] D. A. Cox, J. Little, D. O'Shea, *Using Algebraic Geometry*, 2nd
       ed., Springer, 2005, Chapter 2.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Optional, Sequence

from .domains import QQ
from .f4 import f4
from .fglm import fglm, is_zero_dimensional
from .numberfields.qqbar import Algebraic, algebraic_roots
from .rings import Monomial, PolyElement, PolyRing, grevlex, lex
from .sqfreetools import dup_sqf_part

__all__ = ['solve_zero_dimensional', 'univariate_in_ideal']

#: how many changes of coordinates are tried for the shape position
_DIRECTIONS = 60


def univariate_in_ideal(basis: Sequence[PolyElement[Fraction]], ring: PolyRing[Fraction], index: int) -> list[Fraction]:
    """The monic polynomial of least degree in the variable number
    ``index`` alone which lies in the zero-dimensional ideal of the
    Gröbner basis ``basis`` (dense, highest degree first): the first
    linear dependence between the normal forms of the powers of the
    variable.

    Examples
    ========

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.f4 import f4
    >>> from sympy_extras.polys.zerodim import univariate_in_ideal
    >>> R = PolyRing(QQ, ['x', 'y'], lex)
    >>> x, y = R.gens
    >>> G = f4([x*x + y*y - R.from_int(5), x*y - R.from_int(2)], R)
    >>> univariate_in_ideal(G, R, 0) == [1, 0, -5, 0, 4]
    True
    """
    n = len(ring.names)
    unit: Monomial = tuple(1 if k == index else 0 for k in range(n))
    generators = [g for g in basis if g.terms]
    # echelon rows: pivot monomial, vector, combination of the powers
    rows: list[tuple[Monomial, dict[Monomial, Fraction], list[Fraction]]] = []
    power = ring.one.rem(generators)
    degree = 0
    while True:
        vector = dict(power.terms)
        combination = [Fraction(0)]*(degree + 1)
        combination[degree] = Fraction(1)
        for pivot, row, row_combination in rows:
            c = vector.get(pivot)
            if c is None:
                continue
            for m, v in row.items():
                value = vector.get(m, Fraction(0)) - c*v
                if value:
                    vector[m] = value
                else:
                    vector.pop(m, None)
            for k, v in enumerate(row_combination):
                combination[k] -= c*v
        if not vector:
            # sum combination[k] x**k is in the ideal
            return [c/combination[degree] for c in reversed(combination)]
        pivot = next(iter(vector))
        inverse = 1/vector[pivot]
        rows.append((pivot, {m: v*inverse for m, v in vector.items()}, [c*inverse for c in combination]))
        power = power.mul_monom(unit).rem(generators)
        degree += 1


def _univariate_element(coefficients: list[Fraction], ring: PolyRing[Fraction], index: int) -> PolyElement[Fraction]:
    n = len(ring.names)
    degree = len(coefficients) - 1
    terms: dict[Monomial, Fraction] = {}
    for k, c in enumerate(coefficients):
        if c:
            terms[tuple(degree - k if j == index else 0 for j in range(n))] = c
    return PolyElement(terms, ring)


def _shear(f: PolyElement[Fraction], direction: Sequence[int]) -> PolyElement[Fraction]:
    """``f`` with the last variable replaced by ``x_n - sum c_i x_i``:
    the polynomial in the coordinates ``(x_1, ..., x_(n-1), x_n + sum
    c_i x_i)``."""
    ring = f.ring
    gens = ring.gens
    n = len(gens)
    replacement = gens[n - 1]
    for i, c in enumerate(direction):
        if c:
            replacement = replacement - gens[i].mul_ground(Fraction(c))
    powers = [ring.one]
    result = ring.zero
    for monomial, coefficient in f.terms.items():
        e = monomial[n - 1]
        while len(powers) <= e:
            powers.append(powers[-1]*replacement)
        head: Monomial = monomial[:n - 1] + (0,)
        result = result + powers[e].mul_term(head, coefficient)
    return result


def _shape(basis: Sequence[PolyElement[Fraction]], n: int) -> Optional[tuple[list[list[Fraction]], list[Fraction]]]:
    """For a lexicographic basis ``x_i - g_i(x_n), h(x_n)``: the dense
    polynomials ``g_1, ..., g_(n-1)`` and ``h``; ``None`` for another
    shape."""
    if len(basis) != n:
        return None

    def dense(f: PolyElement[Fraction], skip: Optional[Monomial]) -> Optional[list[Fraction]]:
        degree = max((m[n - 1] for m in f.terms), default=0)
        coefficients = [Fraction(0)]*(degree + 1)
        for m, c in f.terms.items():
            if m == skip:
                continue
            if any(m[:n - 1]):
                return None
            coefficients[degree - m[n - 1]] = c
        return coefficients

    images: list[list[Fraction]] = []
    for i in range(n - 1):
        f = basis[i]
        leading: Monomial = tuple(1 if k == i else 0 for k in range(n))
        if f.LM != leading or f.LC != 1:
            return None
        rest = dense(f, leading)
        if rest is None:
            return None
        images.append([-c for c in rest])
    last = dense(basis[n - 1], None)
    if last is None:
        return None
    return images, last


def _remainder(f: list[Fraction], h: list[Fraction]) -> list[Fraction]:
    """``f`` modulo the monic ``h`` (dense, highest degree first)."""
    r = list(f)
    while r and r[0] == 0:
        r = r[1:]
    d = len(h) - 1
    while len(r) - 1 >= d and r:
        c = r[0]
        for k in range(1, d + 1):
            r[k] -= c*h[k]
        r = r[1:]
        while r and r[0] == 0:
            r = r[1:]
    return r


def solve_zero_dimensional(polys: Sequence[PolyElement[Fraction]], ring: PolyRing[Fraction]) -> list[list[Algebraic]]:
    """All the complex solutions of the system ``polys = 0`` over the
    rationals, when there are finitely many: a list of points, each a
    list of algebraic numbers in the order of the variables of ``ring``;
    multiplicities are forgotten.

    Raises ``ValueError`` when the system has infinitely many solutions.

    Examples
    ========

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.zerodim import solve_zero_dimensional
    >>> R = PolyRing(QQ, ['x', 'y'], lex)
    >>> x, y = R.gens
    >>> solve_zero_dimensional([x*x, y*y], R)                         # a multiple point
    [[Algebraic([1, 0], 0), Algebraic([1, 0], 0)]]
    >>> solve_zero_dimensional([x*x - R.one, y*y - R.one], R)       # not in shape position as given
    [[Algebraic([1, -1], 1), Algebraic([1, 1], -1)], [Algebraic([1, -1], 1), Algebraic([1, -1], 1)], [Algebraic([1, 1], -1), Algebraic([1, 1], -1)], [Algebraic([1, 1], -1), Algebraic([1, -1], 1)]]
    >>> solve_zero_dimensional([x + y, R.one], R)
    []
    """
    n = len(ring.names)
    if n == 0:
        raise ValueError('a ring without variables')
    graded = ring.with_order(grevlex)
    system = [PolyElement(dict(f.terms), graded) for f in polys if f.terms]
    basis = f4(system, graded)
    if any(sum(g.LM) == 0 for g in basis if g.terms):
        return []
    if not is_zero_dimensional(basis, graded):
        raise ValueError('the system has infinitely many solutions')
    # the radical: the square-free parts of the univariate polynomials
    for i in range(n):
        univariate = univariate_in_ideal(basis, graded, i)
        system.append(_univariate_element(dup_sqf_part(univariate, QQ), graded, i))
    directions: list[tuple[int, ...]] = [tuple(0 for _ in range(n - 1))]
    for bound in (1, 2, 3):
        for direction in product(range(-bound, bound + 1), repeat=n - 1):
            if direction not in directions:
                directions.append(direction)
    for direction in directions[:_DIRECTIONS]:
        sheared = [_shear(f, direction) for f in system]
        shape = _shape(fglm(f4(sheared, graded), graded, lex), n)
        if shape is None:
            continue
        images, last = shape
        # the last coordinate is u = x_n + sum c_i x_i, a root t of h:
        # x_n = t - sum c_i g_i(t), modulo h
        final = [Fraction(1), Fraction(0)]
        for c, g in zip(direction, images):
            if c:
                size = max(len(final), len(g))
                final = [a - c*b for a, b in zip([Fraction(0)]*(size - len(final)) + final,
                                                  [Fraction(0)]*(size - len(g)) + g)]
        coordinates = images + [_remainder(final, last)]
        solutions: list[list[Algebraic]] = []
        for root in algebraic_roots([int(c*_common_denominator(last)) for c in last]):
            solutions.append([root.polynomial_value(g) for g in coordinates])
        return solutions
    raise ArithmeticError('no change of coordinates among %d gives the shape position' % _DIRECTIONS)


def _common_denominator(f: list[Fraction]) -> int:
    common = 1
    for c in f:
        d = c.denominator
        a, b = common, d
        while b:
            a, b = b, a % b
        common = common*d//a
    return common

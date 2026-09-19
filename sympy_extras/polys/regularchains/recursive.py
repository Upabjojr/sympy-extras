"""Polynomials of `\\mathbb{Z}[x_1, \\ldots, x_n]` seen as univariate ones.

The algorithms on regular chains see a polynomial which is not constant as
a univariate polynomial in its *main variable*, the greatest variable
occurring in it, with coefficients in the smaller variables: its leading
coefficient is its *initial*, the rest its *tail*. The variables are the
generators of a :class:`~sympy.polys.rings.PolyRing` over the integers and
are ordered as SymPy orders them: **the first generator is the greatest**,
so that a variable is given by its index, and a smaller index means a
greater variable. The constants have the level ``ring.ngens``.

Pseudo-divisions are *lazy*: instead of multiplying the dividend by the
full power `\\operatorname{init}(g)^{\\deg f - \\deg g + 1}`, each step
multiplies by the cofactor of the initial of the divisor which is needed,
so that

.. math:: m f = q g + r, \\qquad \\deg_x r < \\deg_x g,

where `m` is a product of divisors of `\\operatorname{init}(g)`. This is
all the algorithms need (`m` vanishes nowhere `\\operatorname{init}(g)`
does not, and is regular wherever `\\operatorname{init}(g)` is) and it
keeps the coefficients much smaller.

The subresultant chain is obtained from SymPy's subresultant polynomial
remainder sequence by the structure theorem of subresultants: the
remainders are the subresultants `S_{n_{i-1}-1}`, the subresultant
`S_{n_i}` is similar to `S_{n_{i-1}-1}` with the factor of [Ducos]_, and
the ones in between vanish.

References
==========

.. [Ducos] L. Ducos, Optimizations of the subresultant algorithm, Journal
   of Pure and Applied Algebra 145 (2000), 149-163.
.. [Mishra] B. Mishra, Algorithmic Algebra, Springer 1993, chapter 7.
"""
from __future__ import annotations

from sympy.polys.rings import PolyElement

__all__ = ['main_variable', 'degree', 'coefficient', 'initial', 'tail',
    'normalize', 'exact_quotient', 'primitive_part', 'pseudo_divide', 'pseudo_remainder',
    'subresultant_chain', 'distinct_factors']


def main_variable(p: PolyElement) -> int:
    """The index of the greatest variable of ``p``, the number of generators
    for a constant.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy_extras.polys.regularchains.recursive import main_variable
    >>> R, x, y, z = ring('x y z', ZZ)
    >>> main_variable(y*z + z**2), main_variable(R(3))
    (1, 3)
    """
    n: int = p.ring.ngens
    level = n
    for monom in p.itermonoms():
        for i in range(level):
            if monom[i]:
                level = i
                break
        if not level:
            break
    return level


def degree(p: PolyElement, i: int) -> int:
    """The degree of ``p`` in the variable of index ``i`` (``-1`` for zero)."""
    if not p:
        return -1
    return max(monom[i] for monom in p.itermonoms())


def coefficient(p: PolyElement, i: int, d: int) -> PolyElement:
    """The coefficient of the ``d``-th power of the variable ``i``."""
    result: PolyElement = p.coeff_wrt(i, d)
    return result


def initial(p: PolyElement) -> PolyElement:
    """The leading coefficient of ``p`` in its main variable.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy_extras.polys.regularchains.recursive import initial, tail
    >>> R, x, y = ring('x y', ZZ)
    >>> p = (y + 1)*x**2 + y*x + 3
    >>> initial(p), tail(p)
    (y + 1, x*y + 3)
    """
    i = main_variable(p)
    if i == p.ring.ngens:
        return p
    return coefficient(p, i, degree(p, i))


def _power(p: PolyElement, i: int, k: int) -> PolyElement:
    """``p`` times the ``k``-th power of the variable ``i``."""
    if not k:
        return p
    monom = [0]*p.ring.ngens
    monom[i] = k
    result: PolyElement = p.mul_monom(tuple(monom))
    return result


def exact_quotient(p: PolyElement, q: PolyElement) -> PolyElement:
    """``p/q`` for ``q`` dividing ``p``.

    The quotients of SymPy 1.14 (``div``, ``exquo``) come with a cached hash
    which is not the one of their terms (issue #25), so that equal
    polynomials are told apart by sets and dictionaries: the copy has none.
    """
    result: PolyElement = p.exquo(q).copy()
    return result


def tail(p: PolyElement) -> PolyElement:
    """``p`` without its term of highest degree in its main variable."""
    i = main_variable(p)
    if i == p.ring.ngens:
        return p.ring.zero
    d = degree(p, i)
    result: PolyElement = p - _power(coefficient(p, i, d), i, d)
    return result


def normalize(p: PolyElement) -> PolyElement:
    """The primitive part of ``p`` over the integers, with a positive leading
    coefficient."""
    if not p:
        return p
    q: PolyElement = p.primitive()[1]
    if q.LC < 0:
        q = -q
    return q


def primitive_part(p: PolyElement, i: int) -> PolyElement:
    """``p`` divided by the greatest common divisor of its coefficients as a
    polynomial in the variable ``i``, normalized.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy_extras.polys.regularchains.recursive import primitive_part
    >>> R, x, y = ring('x y', ZZ)
    >>> primitive_part(2*(y**2 - 1)*x**2 + (4*y + 4)*x, 0)
    x**2*y - x**2 + 2*x
    """
    if not p:
        return p
    d = degree(p, i)
    content: PolyElement = p.ring.zero
    for k in range(d, -1, -1):
        c = coefficient(p, i, k)
        if c:
            content = content.gcd(c)
            if content.is_ground:
                break
    if not content.is_ground:
        p = exact_quotient(p, content)
    return normalize(p)


def pseudo_divide(f: PolyElement, g: PolyElement, i: int) -> tuple[PolyElement, PolyElement, PolyElement]:
    """The lazy pseudo-division of ``f`` by ``g`` in the variable ``i``.

    Returns
    =======

    (m, q, r)
        with ``m*f == q*g + r``, the degree of ``r`` in the variable lower
        than the one of ``g``, and ``m`` a product of divisors of the leading
        coefficient of ``g``.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy_extras.polys.regularchains.recursive import pseudo_divide
    >>> R, x, y = ring('x y', ZZ)
    >>> f, g = x**3 + y, y*x**2 - 1
    >>> m, q, r = pseudo_divide(f, g, 0)
    >>> m, q, r
    (y, x, x + y**2)
    >>> m*f == q*g + r
    True
    """
    ring = f.ring
    dg = degree(g, i)
    if dg < 0:
        raise ZeroDivisionError("pseudo-division by zero")
    lg = coefficient(g, i, dg)
    m: PolyElement = ring.one
    q: PolyElement = ring.zero
    r = f
    while r:
        dr = degree(r, i)
        if dr < dg:
            break
        lr = coefficient(r, i, dr)
        _, a, b = lr.cofactors(lg)
        m = m*b
        q = q*b + _power(a, i, dr - dg)
        r = r*b - _power(a*g, i, dr - dg)
    return m, q, r


def pseudo_remainder(f: PolyElement, g: PolyElement, i: int) -> PolyElement:
    """The remainder of the lazy pseudo-division of ``f`` by ``g``."""
    dg = degree(g, i)
    if dg < 0:
        raise ZeroDivisionError("pseudo-division by zero")
    lg = coefficient(g, i, dg)
    r = f
    while r:
        dr = degree(r, i)
        if dr < dg:
            break
        lr = coefficient(r, i, dr)
        _, a, b = lr.cofactors(lg)
        r = r*b - _power(a*g, i, dr - dg)
    return r


def subresultant_chain(f: PolyElement, g: PolyElement, i: int) -> list[PolyElement]:
    """The subresultants `S_0, \\ldots, S_{n-1}` of ``f`` and ``g`` in the
    variable ``i``, up to signs, where `n = \\deg g \\le \\deg f`.

    `S_j` has degree at most `j`, and its coefficient of degree `j` is the
    `j`-th principal subresultant coefficient; `S_0` is the resultant.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy_extras.polys.regularchains.recursive import subresultant_chain
    >>> R, x, y = ring('x y', ZZ)
    >>> subresultant_chain(x**2 + y**2 - 1, x*y - 1, 0)
    [y**4 - y**2 + 1]
    >>> subresultant_chain((x - y)**2*(x + 1), (x - y)*(x - 2), 0)
    [0, -3*x*y + 6*x + 3*y**2 - 6*y]
    """
    ring = f.ring
    n = degree(g, i)
    if degree(f, i) < n or n < 1:
        raise ValueError("the degrees must satisfy deg f >= deg g >= 1")
    chain: list[PolyElement] = [ring.zero]*n
    remainders: list[PolyElement] = f.subresultants(g, i)
    s: PolyElement = coefficient(g, i, n)**(degree(f, i) - n)
    before = n
    for r in remainders[2:]:
        r = ring(r)
        if not r:
            break
        d = degree(r, i)
        chain[before - 1] = r
        delta = before - d
        if delta > 1:
            r = exact_quotient(coefficient(r, i, d)**(delta - 1)*r, s**(delta - 1))
            chain[d] = r
        s = coefficient(r, i, d)
        before = d
    return chain


def distinct_factors(p: PolyElement) -> list[PolyElement]:
    """The irreducible factors of ``p`` over the integers which are not
    constant, without their multiplicities."""
    if p.is_ground:
        return []
    pairs = p.factor_list()[1]
    factors: list[PolyElement] = [normalize(q) for q, _ in pairs if not q.is_ground]
    return sorted(factors, key=lambda q: (len(q), q.sort_key()))

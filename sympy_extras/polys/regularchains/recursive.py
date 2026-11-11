"""Polynomials of `\\mathbb{Z}[x_1, \\ldots, x_n]` seen as univariate ones.

The algorithms on regular chains see a polynomial which is not constant as
a univariate polynomial in its *main variable*, the greatest variable
occurring in it, with coefficients in the smaller variables: its leading
coefficient is its *initial*, the rest its *tail*. The variables are the
generators of a :class:`~sympy_extras.polys.rings.PolyRing` over the
integers (lex order) and are ordered as SymPy orders them: **the first
generator is the greatest**, so that a variable is given by its index, and
a smaller index means a greater variable. The constants have the level
``ring.ngens``.

Pseudo-divisions are *lazy*: instead of multiplying the dividend by the
full power `\\operatorname{init}(g)^{\\deg f - \\deg g + 1}`, each step
multiplies by the cofactor of the initial of the divisor which is needed,
so that

.. math:: m f = q g + r, \\qquad \\deg_x r < \\deg_x g,

where `m` is a product of divisors of `\\operatorname{init}(g)`. This is
all the algorithms need (`m` vanishes nowhere `\\operatorname{init}(g)`
does not, and is regular wherever `\\operatorname{init}(g)` is) and it
keeps the coefficients much smaller.

The subresultant chain is obtained from the subresultant polynomial
remainder sequence (SymPy's ``PolyElement.subresultants``, ported here
with its pseudo-remainder ``prem``) by the structure theorem of
subresultants: the remainders are the subresultants `S_{n_{i-1}-1}`, the
subresultant `S_{n_i}` is similar to `S_{n_{i-1}-1}` with the factor of
[Ducos]_, and the ones in between vanish.

The module is native (no SymPy): the gcds are those of
:class:`~sympy_extras.polys.domains.sparsepolynomialring.IntegerPolynomialRing`
(SymPy's conventions, the heuristic gcd), the factorisation is
:func:`~sympy_extras.polys.multifactor.zz_factor` (Wang's algorithm).

References
==========

.. [Ducos] L. Ducos, Optimizations of the subresultant algorithm, Journal
   of Pure and Applied Algebra 145 (2000), 149-163.
.. [Mishra] B. Mishra, Algorithmic Algebra, Springer 1993, chapter 7.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/rings.py (PolyElement.prem, PolyElement.subresultants,
# PolyElement.coeff_wrt, PolyElement.sort_key) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed functions on the native PolyElement over ZZ, the variable
# given by its index.
# ---------------------------------------------------------------------
from __future__ import annotations

from math import gcd as integer_gcd

from sympy_extras.polys.domains.integerring import ZZ
from sympy_extras.polys.domains.sparsepolynomialring import IntegerPolynomialRing
from sympy_extras.polys.multifactor import sqf_part as _sqf_part, zz_factor
from sympy_extras.polys.rings import Monomial, PolyElement, PolyRing

__all__ = ['ZPoly', 'SortKey', 'main_variable', 'degree', 'coefficient', 'initial', 'tail',
    'normalize', 'exact_quotient', 'primitive_part', 'pseudo_divide', 'pseudo_remainder', 'prem',
    'subresultants', 'subresultant_chain', 'distinct_factors', 'gcd', 'cofactors', 'sqf_part', 'max_norm',
    'sort_key', 'integer_ring']

#: a polynomial over the integers
ZPoly = PolyElement[int]
#: SymPy's sort key of a polynomial: its length, then its terms in decreasing order
SortKey = tuple[int, list[tuple[Monomial, int]]]

_RINGS: dict[tuple[str, ...], IntegerPolynomialRing] = {}


def integer_ring(ring: PolyRing[int]) -> IntegerPolynomialRing:
    """The gcd domain of the ring (cached by the names of the variables)."""
    key = tuple(ring.names)
    found = _RINGS.get(key)
    if found is None:
        found = IntegerPolynomialRing(ring.names)
        _RINGS[key] = found
    return found


def gcd(p: ZPoly, q: ZPoly) -> ZPoly:
    """The gcd of SymPy's ``PolyElement.gcd`` (positive leading coefficient)."""
    result = integer_ring(p.ring).gcd(p, q)
    return result if result.ring == p.ring else p.ring.from_dict(result.terms)


def cofactors(p: ZPoly, q: ZPoly) -> tuple[ZPoly, ZPoly, ZPoly]:
    """``(h, p/h, q/h)`` for the gcd ``h`` (SymPy's ``PolyElement.cofactors``)."""
    h = gcd(p, q)
    if not h:
        return h, p.ring.zero, p.ring.zero
    return h, exact_quotient(p, h), exact_quotient(q, h)


def max_norm(p: ZPoly) -> int:
    """The greatest absolute value of a coefficient."""
    return max((abs(c) for c in p.terms.values()), default=0)


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.sort_key
def sort_key(p: ZPoly) -> SortKey:
    return len(p.terms), sorted(p.terms.items(), reverse=True)


def main_variable(p: ZPoly) -> int:
    """The index of the greatest variable of ``p``, the number of generators
    for a constant.

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing
    >>> from sympy_extras.polys.regularchains.recursive import main_variable
    >>> R = PolyRing(ZZ, ['x', 'y', 'z'])
    >>> x, y, z = R.gens
    >>> main_variable(y*z + z**2), main_variable(R.from_int(3))
    (1, 3)
    """
    n: int = p.ring.ngens
    level = n
    for monom in p.terms:
        for i in range(level):
            if monom[i]:
                level = i
                break
        if not level:
            break
    return level


def degree(p: ZPoly, i: int) -> int:
    """The degree of ``p`` in the variable of index ``i`` (``-1`` for zero)."""
    if not p.terms:
        return -1
    return max(monom[i] for monom in p.terms)


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.coeff_wrt
def coefficient(p: ZPoly, i: int, d: int) -> ZPoly:
    """The coefficient of the ``d``-th power of the variable ``i``."""
    return PolyElement({m[:i] + (0,) + m[i + 1:]: c for m, c in p.terms.items() if m[i] == d}, p.ring)


def initial(p: ZPoly) -> ZPoly:
    """The leading coefficient of ``p`` in its main variable.

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing
    >>> from sympy_extras.polys.regularchains.recursive import initial, tail
    >>> R = PolyRing(ZZ, ['x', 'y'])
    >>> x, y = R.gens
    >>> p = (y + R.one)*x**2 + y*x + R.from_int(3)
    >>> print(initial(p), tail(p))
    y + 1 x*y + 3
    """
    i = main_variable(p)
    if i == p.ring.ngens:
        return p
    return coefficient(p, i, degree(p, i))


def _power(p: ZPoly, i: int, k: int) -> ZPoly:
    """``p`` times the ``k``-th power of the variable ``i``."""
    if not k:
        return p
    monom = [0]*p.ring.ngens
    monom[i] = k
    return p.mul_monom(tuple(monom))


def exact_quotient(p: ZPoly, q: ZPoly) -> ZPoly:
    """``p/q`` for ``q`` dividing ``p`` (``ValueError`` otherwise)."""
    if not q.terms:
        raise ZeroDivisionError("division by zero")
    if len(q.terms) == 1:
        ((m, c),) = q.terms.items()
        terms: dict[Monomial, int] = {}
        for n, d in p.terms.items():
            quotient, remainder = divmod(d, c)
            if remainder or any(a < b for a, b in zip(n, m)):
                raise ValueError('%s does not divide %s' % (q, p))
            terms[tuple(a - b for a, b in zip(n, m))] = quotient
        return PolyElement(terms, p.ring)
    return _exquo(p, q)


def _exquo(p: ZPoly, q: ZPoly) -> ZPoly:
    """The exact division by a polynomial of several terms, in the lex
    order: the leading term of the remainder is divided by the one of
    ``q`` at each step (exact when ``q`` divides ``p``)."""
    lm = max(q.terms)
    lc = q.terms[lm]
    rest = [(m, c) for m, c in q.terms.items() if m != lm]
    r = dict(p.terms)
    quotient: dict[Monomial, int] = {}
    while r:
        m = max(r)
        c = r[m]
        if any(a < b for a, b in zip(m, lm)):
            raise ValueError('%s does not divide %s' % (q, p))
        k, remainder = divmod(c, lc)
        if remainder:
            raise ValueError('%s does not divide %s' % (q, p))
        e = tuple(a - b for a, b in zip(m, lm))
        quotient[e] = k
        del r[m]
        for n, d in rest:
            t = tuple(a + b for a, b in zip(n, e))
            value = r.get(t, 0) - k*d
            if value:
                r[t] = value
            elif t in r:
                del r[t]
    return PolyElement(quotient, p.ring)


def tail(p: ZPoly) -> ZPoly:
    """``p`` without its term of highest degree in its main variable."""
    i = main_variable(p)
    if i == p.ring.ngens:
        return p.ring.zero
    d = degree(p, i)
    return PolyElement({m: c for m, c in p.terms.items() if m[i] != d}, p.ring)


def normalize(p: ZPoly) -> ZPoly:
    """The primitive part of ``p`` over the integers, with a positive leading
    coefficient."""
    if not p.terms:
        return p
    content = 0
    for c in p.terms.values():
        content = integer_gcd(content, c)
        if content == 1:
            break
    if p.terms[max(p.terms)] < 0:
        content = -content
    if content == 1:
        return p
    return PolyElement({m: c // content for m, c in p.terms.items()}, p.ring)


def primitive_part(p: ZPoly, i: int) -> ZPoly:
    """``p`` divided by the greatest common divisor of its coefficients as a
    polynomial in the variable ``i``, normalized.

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing
    >>> from sympy_extras.polys.regularchains.recursive import primitive_part
    >>> R = PolyRing(ZZ, ['x', 'y'])
    >>> x, y = R.gens
    >>> two = R.from_int(2)
    >>> print(primitive_part(two*(y**2 - R.one)*x**2 + two*two*(y + R.one)*x, 0))
    x**2*y - x**2 + 2*x
    """
    if not p.terms:
        return p
    d = degree(p, i)
    content: ZPoly = p.ring.zero
    for k in range(d, -1, -1):
        c = coefficient(p, i, k)
        if c.terms:
            content = gcd(content, c)
            if content.is_ground:
                break
    if not content.is_ground:
        p = exact_quotient(p, content)
    return normalize(p)


def pseudo_divide(f: ZPoly, g: ZPoly, i: int) -> tuple[ZPoly, ZPoly, ZPoly]:
    """The lazy pseudo-division of ``f`` by ``g`` in the variable ``i``.

    Returns
    =======

    (m, q, r)
        with ``m*f == q*g + r``, the degree of ``r`` in the variable lower
        than the one of ``g``, and ``m`` a product of divisors of the leading
        coefficient of ``g``.

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing
    >>> from sympy_extras.polys.regularchains.recursive import pseudo_divide
    >>> R = PolyRing(ZZ, ['x', 'y'])
    >>> x, y = R.gens
    >>> f, g = x**3 + y, y*x**2 - R.one
    >>> m, q, r = pseudo_divide(f, g, 0)
    >>> print(m, q, r)
    y x x + y**2
    >>> m*f == q*g + r
    True
    """
    ring = f.ring
    dg = degree(g, i)
    if dg < 0:
        raise ZeroDivisionError("pseudo-division by zero")
    lg = coefficient(g, i, dg)
    m: ZPoly = ring.one
    q: ZPoly = ring.zero
    r = f
    while r.terms:
        dr = degree(r, i)
        if dr < dg:
            break
        lr = coefficient(r, i, dr)
        _, a, b = cofactors(lr, lg)
        m = m*b
        q = q*b + _power(a, i, dr - dg)
        r = r*b - _power(a*g, i, dr - dg)
    return m, q, r


def pseudo_remainder(f: ZPoly, g: ZPoly, i: int) -> ZPoly:
    """The remainder of the lazy pseudo-division of ``f`` by ``g``."""
    dg = degree(g, i)
    if dg < 0:
        raise ZeroDivisionError("pseudo-division by zero")
    lg = coefficient(g, i, dg)
    r = f
    while r.terms:
        dr = degree(r, i)
        if dr < dg:
            break
        lr = coefficient(r, i, dr)
        _, a, b = cofactors(lr, lg)
        r = r*b - _power(a*g, i, dr - dg)
    return r


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.prem
def prem(f: ZPoly, g: ZPoly, i: int) -> ZPoly:
    """The pseudo-remainder of ``f`` by ``g`` in the variable ``i``: ``m f =
    q g + r`` with ``m`` the leading coefficient of ``g`` to the power
    `\\deg f - \\deg g + 1`."""
    df, dg = degree(f, i), degree(g, i)
    if dg < 0:
        raise ZeroDivisionError("polynomial division")
    r, dr = f, df
    if df < dg:
        return r
    N = df - dg + 1
    lc_g = coefficient(g, i, dg)
    while True:
        lc_r = coefficient(r, i, dr)
        j, N = dr - dg, N - 1
        r = r*lc_g - _power(g*lc_r, i, j)
        dr = degree(r, i)
        if dr < dg:
            break
    return r*lc_g**N


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.subresultants
def subresultants(f: ZPoly, g: ZPoly, i: int) -> list[ZPoly]:
    """The subresultant polynomial remainder sequence of ``f`` and ``g`` in
    the variable ``i``, starting with ``f`` and ``g`` (the one of greater
    degree first)."""
    ring = f.ring
    n, m = degree(f, i), degree(g, i)
    if n < m:
        f, g = g, f
        n, m = m, n
    if not f.terms:
        return [ring.zero, ring.zero]
    if not g.terms:
        return [f, ring.one]
    R = [f, g]
    d = n - m
    h = prem(f, g, i)
    if not d % 2:
        h = -h
    lc = coefficient(g, i, m)
    c = -(lc**d)
    while h.terms:
        k = degree(h, i)
        R.append(h)
        f, g, m, d = g, h, k, m - k
        b = -lc*c**d
        h = exact_quotient(prem(f, g, i), b)
        lc = coefficient(g, i, k)
        if d > 1:
            c = exact_quotient((-lc)**d, c**(d - 1))
        else:
            c = -lc
    return R


def subresultant_chain(f: ZPoly, g: ZPoly, i: int) -> list[ZPoly]:
    """The subresultants `S_0, \\ldots, S_{n-1}` of ``f`` and ``g`` in the
    variable ``i``, up to signs, where `n = \\deg g \\le \\deg f`.

    `S_j` has degree at most `j`, and its coefficient of degree `j` is the
    `j`-th principal subresultant coefficient; `S_0` is the resultant.

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing
    >>> from sympy_extras.polys.regularchains.recursive import subresultant_chain
    >>> R = PolyRing(ZZ, ['x', 'y'])
    >>> x, y = R.gens
    >>> [str(s) for s in subresultant_chain(x**2 + y**2 - R.one, x*y - R.one, 0)]
    ['y**4 - y**2 + 1']
    >>> [str(s) for s in subresultant_chain((x - y)**2*(x + R.one), (x - y)*(x - R.from_int(2)), 0)]
    ['0', '-3*x*y + 6*x + 3*y**2 - 6*y']
    """
    ring = f.ring
    n = degree(g, i)
    if degree(f, i) < n or n < 1:
        raise ValueError("the degrees must satisfy deg f >= deg g >= 1")
    chain: list[ZPoly] = [ring.zero]*n
    remainders = subresultants(f, g, i)
    s: ZPoly = coefficient(g, i, n)**(degree(f, i) - n)
    before = n
    for r in remainders[2:]:
        if not r.terms:
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


def sqf_part(p: ZPoly) -> ZPoly:
    """The square-free part, normalized."""
    return normalize(_sqf_part(p))


def _factor_list(p: ZPoly) -> list[ZPoly]:
    """The irreducible factors of a polynomial which is not constant
    (SymPy's ``dmp_factor_list`` over ``ZZ``: the monomial content, then
    Wang's algorithm in the variables which occur)."""
    ring = p.ring
    n = ring.ngens
    exponents = [min(m[k] for m in p.terms) for k in range(n)]
    shifted = {tuple(e - s for e, s in zip(m, exponents)): c for m, c in p.terms.items()}
    factors: list[ZPoly] = []
    for k in range(n):
        if exponents[k]:
            factors.append(ring.term(tuple(1 if j == k else 0 for j in range(n)), 1))
    used = [k for k in range(n) if any(m[k] for m in shifted)]
    if not used:
        return factors
    sub = PolyRing(ZZ, [ring.names[k] for k in used])
    g = sub.from_dict({tuple(m[k] for k in used): c for m, c in shifted.items()})
    for h, _ in zz_factor(g)[1]:
        terms: dict[Monomial, int] = {}
        for m, value in h.terms.items():
            full = [0]*n
            for position, e in zip(used, m):
                full[position] = e
            terms[tuple(full)] = value
        factors.append(PolyElement(terms, ring))
    return factors


def distinct_factors(p: ZPoly) -> list[ZPoly]:
    """The irreducible factors of ``p`` over the integers which are not
    constant, without their multiplicities."""
    if p.is_ground:
        return []
    factors = [normalize(q) for q in _factor_list(p) if not q.is_ground]
    return sorted(factors, key=sort_key)

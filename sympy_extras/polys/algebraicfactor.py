r"""Factorisation of multivariate polynomials over an algebraic number
field $\mathbb{Q}(\alpha)$: Trager's algorithm on sparse polynomials.

For $f \in \mathbb{Q}(\alpha)[x_1, \dots, x_n]$, made monic (its leading
coefficient in the lexicographic order divided out) and square-free, a
shift $g(X) = f(x_1 - s_1 \alpha, \dots, x_n - s_n \alpha)$ is chosen
whose norm $N(g) = \operatorname{Res}_y(m(y), g(X, y))$ ($m$ the minimal
polynomial of $\alpha$, the coefficients of $g$ written as polynomials in
$y = \alpha$) is square-free over $\mathbb{Q}$: the shifts $(0, \dots,
0)$, then one variable at a time ($1$ in a single position, the variables
of lowest degree first), then $(j, \dots, j)$ for $j = 1, 2, \dots$. The
norm is factored over $\mathbb{Q}$ (Wang's algorithm,
:mod:`~sympy_extras.polys.multifactor`); each of its irreducible factors
$h$ gives the irreducible factor $\gcd(h, g)$ of $g$ over
$\mathbb{Q}(\alpha)$, shifted back, and the multiplicities of the factors
of the original polynomial come from trial division (Trager [1]_). The
monomial content is split off first, as SymPy's ``dmp_factor_list`` does.

This is the algorithm of :mod:`~sympy_extras.polys.gaussianfactor` for
any number field (there specialised to $\mathbb{Q}(i)$, whose norm is
$A^2 + B^2$), with the norm computed as a resultant; like it, it is ported
from SymPy's ``dmp_ext_factor`` and ``dmp_sqf_norm``
(``sympy.polys.factortools``, ``sympy.polys.sqfreetools``).

Examples
========

>>> from fractions import Fraction
>>> from sympy_extras.core import Integer, sqrt
>>> from sympy_extras.polys.domains.algebraicfield import AlgebraicField
>>> from sympy_extras.polys.rings import PolyRing
>>> from sympy_extras.polys.algebraicfactor import ext_factor_list
>>> K = AlgebraicField([Fraction(1), Fraction(0), Fraction(-2)], sqrt(Integer(2)))
>>> R = PolyRing(K, ['x', 'y'])
>>> x, y = R.gens
>>> f = x*x - y*y.mul_ground(K.from_int(2))                       # x**2 - 2*y**2
>>> lc, factors = ext_factor_list(f)
>>> lc, sorted(str(g) for g, k in factors)
([Fraction(1, 1)], ['x + -sqrt(2)*y', 'x + sqrt(2)*y'])

References
==========

.. [1] B. M. Trager, *Algebraic factoring and rational function
       integration*, Proceedings of SYMSAC '76, ACM, 1976, 219-226
       (algorithms ``sqfr_norm`` and ``alg_factor``).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/factortools.py (dmp_ext_factor, dmp_trial_division,
# dmp_factor_list), sympy/polys/sqfreetools.py (dmp_sqf_norm, _dmp_sqf_norm_shifts,
# dmp_sqf_p, dmp_sqf_part), sympy/polys/densetools.py (dmp_shift) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: sparse polynomials over the typed AlgebraicField instead of dense
# nested lists of ANP, the norm by a resultant over QQ, strict typing.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Iterator

from typing import Sequence

from . import euclidtools
from .domains.algebraicfield import AlgebraicField
from .domains.rationalfield import QQ
from .domains.sparsepolynomialring import RationalPolynomialRing, SparsePolynomialRing
from .rings import Monomial, PolyElement, PolyRing

__all__ = ['AlgebraicPolynomialRing', 'ext_factor_list', 'ext_gcd']

#: a polynomial over a number field
ExtPoly = PolyElement[list[Fraction]]
#: factors with their multiplicities
ExtFactors = list[tuple[ExtPoly, int]]


class AlgebraicPolynomialRing(SparsePolynomialRing[list[Fraction]]):
    """The ring ``K[x_1, ..., x_n]`` over a number field ``K`` as a domain:
    the field has no order, so the canonical unit of a polynomial is the
    inverse of its leading coefficient (the gcd is monic), and the gcd in
    several variables recurses through rings of the same kind.

    >>> from fractions import Fraction
    >>> from sympy_extras.core import Integer, sqrt
    >>> K = AlgebraicField([Fraction(1), Fraction(0), Fraction(-2)], sqrt(Integer(2)))
    >>> R = AlgebraicPolynomialRing(K, ['x', 'y'])
    >>> x, y = R.ring.gens
    >>> s = R.ring.ground(K.alpha)
    >>> R.gcd((x - s*y)*(x + y), (x - s*y)*(x - y)) == x - s*y
    True
    """

    def __init__(self, ground: AlgebraicField, names: Sequence[str]) -> None:
        super().__init__(ground, names)
        self.field: AlgebraicField = ground

    def is_negative(self, a: ExtPoly) -> bool:
        return False

    def is_nonnegative(self, a: ExtPoly) -> bool:
        return True

    def canonical_unit(self, a: ExtPoly) -> ExtPoly:
        return self.ring.ground(self.field.invert(a.LC)) if a.terms else self.one

    def _gcd_general(self, a: ExtPoly, b: ExtPoly) -> ExtPoly:
        if len(self.names) == 1:
            return super()._gcd_general(a, b)
        inner = AlgebraicPolynomialRing(self.field, self.names[1:])
        g = euclidtools.dup_gcd(self.to_univariate(a, inner), self.to_univariate(b, inner), inner)
        return self.from_univariate(g)


def _field(f: ExtPoly) -> AlgebraicField:
    K = f.ring.domain
    if not isinstance(K, AlgebraicField):
        raise TypeError('a polynomial over an algebraic field is expected, got one over %s' % (K,))
    return K


def ext_gcd(f: ExtPoly, g: ExtPoly) -> ExtPoly:
    """The monic gcd of two polynomials over a number field (zero when
    both are zero)."""
    if not f.terms:
        return g.monic()
    if not g.terms:
        return f.monic()
    S = AlgebraicPolynomialRing(_field(f), f.ring.names)
    return S.gcd(f, g).monic()


# PORTED FROM SYMPY: sympy/polys/sqfreetools.py:dmp_sqf_part
def _sqf_part(f: ExtPoly) -> ExtPoly:
    """The square-free part of ``f``, monic."""
    g = f
    for i in range(f.ring.ngens):
        d = f.diff(i)
        if not d.terms:
            continue
        g = ext_gcd(g, d)
        if g.is_ground:
            return f.monic()
    return f.exquo(g).monic()


# PORTED FROM SYMPY: sympy/polys/densetools.py:dmp_shift
def _shift(f: ExtPoly, a: list[list[Fraction]]) -> ExtPoly:
    """``f(x_1 + a_1, ..., x_n + a_n)``."""
    K = _field(f)
    terms = dict(f.terms)
    for i, ai in enumerate(a):
        if not ai:
            continue
        shifted: dict[Monomial, list[Fraction]] = {}
        for m, c in terms.items():
            e = m[i]
            for k in range(e + 1):
                coefficient = K.mul(c, K.mul(K.from_int(comb(e, k)), K.pow(ai, e - k)))
                monomial = m[:i] + (k,) + m[i + 1:]
                shifted[monomial] = K.add(shifted.get(monomial, K.zero), coefficient)
        terms = {m: c for m, c in shifted.items() if c}
    return f.ring.from_dict(terms)


def _norm(f: ExtPoly) -> PolyElement[Fraction]:
    """``Res_y(m(y), f(X, y))`` over ``QQ``, the coefficients of ``f``
    written as polynomials in ``y = alpha``."""
    from .polytools import _resultant_first
    K = _field(f)
    names = ['_y'] + list(f.ring.names)
    ring = PolyRing(QQ, names)
    zeros = tuple(0 for _ in f.ring.names)
    n = len(K.minpoly) - 1
    minimal = ring.from_dict({(n - k,) + zeros: c for k, c in enumerate(K.minpoly) if c})
    terms: dict[Monomial, Fraction] = {}
    for m, c in f.terms.items():
        d = len(c) - 1
        for j, value in enumerate(c):
            if value:
                terms[(d - j,) + m] = value
    result = _resultant_first(minimal, ring.from_dict(terms), names, False)
    target = PolyRing(QQ, f.ring.names)
    return target.from_dict({m[1:]: c for m, c in result.terms.items()})


# PORTED FROM SYMPY: sympy/polys/sqfreetools.py:dmp_sqf_p
def _sqf_p(r: PolyElement[Fraction]) -> bool:
    """Whether ``r`` is square-free over ``QQ``."""
    if not r.terms:
        return True
    Q = RationalPolynomialRing(r.ring.names)
    for i in range(r.ring.ngens):
        d = r.diff(i)
        if d.terms and Q.gcd(r, d).degree_in(i) != 0:
            return False
    return True


# PORTED FROM SYMPY: sympy/polys/sqfreetools.py:_dmp_sqf_norm_shifts
def _shifts(f: ExtPoly) -> Iterator[list[int]]:
    """The candidate shifts: zero, one variable at a time (those of lowest
    degree first), then ``(j, ..., j)``."""
    n = f.ring.ngens
    yield [0] * n
    degrees = [f.degree_in(i) for i in range(n)]
    for _, i in sorted((d, i) for i, d in enumerate(degrees) if d > 0):
        s = [0] * n
        s[i] = 1
        yield s
    j = 0
    while True:
        j += 1
        yield [j] * n


# PORTED FROM SYMPY: sympy/polys/sqfreetools.py:dmp_sqf_norm
def _sqf_norm(f: ExtPoly) -> tuple[list[int], ExtPoly, PolyElement[Fraction]]:
    """``(s, g, r)``: ``g = f(X - s*alpha)`` with the square-free norm ``r``."""
    K = _field(f)
    for s in _shifts(f):
        g = _shift(f, [K.mul(K.from_int(-si), K.alpha) for si in s]) if any(s) else f
        r = _norm(g)
        if _sqf_p(r):
            return s, g, r
    raise RuntimeError('unreachable')  # pragma: no cover


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_trial_division
def _trial_division(f: ExtPoly, factors: list[ExtPoly]) -> ExtFactors:
    result: ExtFactors = []
    for factor in factors:
        k = 0
        while True:
            (q,), r = f.div([factor])
            if r.terms:
                break
            f, k = q, k + 1
        if k == 0:
            raise RuntimeError('trial division failed')
        result.append((factor, k))
    return result


def _sort_key(item: tuple[ExtPoly, int]) -> tuple[int, int, list[tuple[Monomial, list[Fraction]]]]:
    """The degree in the first variable, the multiplicity, the terms."""
    f, k = item
    return (f.degree_in(0), k, sorted(f.terms.items(), reverse=True))


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_ext_factor
def _ext_factor(f: ExtPoly) -> tuple[list[Fraction], list[ExtPoly]]:
    """The leading coefficient and the monic irreducible factors (without
    multiplicities) of a polynomial whose monomial content is one."""
    from .polytools import _factor_all
    K = _field(f)
    lc = f.LC
    f = f.monic()
    if f.is_ground:
        return lc, []
    f = _sqf_part(f)
    s, g, r = _sqf_norm(f)
    _, rational_factors = _factor_all(r)
    if len(rational_factors) == 1:
        return lc, [f]
    factors: list[ExtPoly] = []
    back = [K.mul(K.from_int(si), K.alpha) for si in s]
    for h, _ in rational_factors:
        hk = f.ring.from_dict({m: K.from_int(c) for m, c in h.terms.items()})
        common = ext_gcd(hk, g)
        g = g.exquo(common)
        factors.append(_shift(common, back) if any(s) else common)
    return lc, factors


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_factor_list
def ext_factor_list(f: ExtPoly) -> tuple[list[Fraction], ExtFactors]:
    """The coefficient and the monic irreducible factors of ``f`` over its
    number field, with their multiplicities: the variables dividing ``f``
    split off first, then Trager's algorithm; the factors are sorted by
    their degree in the first variable, their multiplicity and their
    terms."""
    ring = f.ring
    K = _field(f)
    n = ring.ngens
    if not f.terms:
        return K.zero, []
    J = [min(m[i] for m in f.terms) for i in range(n)]
    g = ring.from_dict({tuple(e - j for e, j in zip(m, J)): c for m, c in f.terms.items()})
    lc, irreducible = _ext_factor(g)
    factors = _trial_division(g.monic(), irreducible) if irreducible else []
    for i in range(n):
        if J[i]:
            factors.append((ring.term(tuple(1 if t == i else 0 for t in range(n)), K.one), J[i]))
    return lc, sorted(factors, key=_sort_key)

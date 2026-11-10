r"""Factorisation of multivariate polynomials over the Gaussian integers
``ZZ_I`` and the Gaussian rationals ``QQ_I``: Trager's algorithm over
$\mathbb{Q}(i)$.

A port of SymPy's ``dmp_zz_i_factor``, ``dmp_qq_i_factor`` and
``dmp_ext_factor`` (``sympy.polys.factortools``) with ``dmp_sqf_norm``
and ``dmp_sqf_part`` (``sympy.polys.sqfreetools``), specialised to the
field $\mathbb{Q}(i)$ and written on sparse polynomials
(:class:`~sympy_extras.polys.rings.PolyElement` over
:data:`~sympy_extras.polys.domains.gaussiandomains.QQ_I`).

For a polynomial $f \in \mathbb{Q}(i)[x_1, \dots, x_n]$, made monic (its
leading coefficient in the lexicographic order divided out) and
square-free, a shift $g(X) = f(x_1 - s_1 i, \dots, x_n - s_n i)$ is chosen
whose norm $N(g) = g \bar g = A^2 + B^2$ ($g = A + iB$ with $A, B$ over
$\mathbb{Q}$) is square-free: the shifts $(0, \dots, 0)$, then one
variable at a time ($1$ in a single position, the variables of lowest
degree first), then $(j, \dots, j)$ for $j = 1, 2, \dots$, in SymPy's
order. The norm is factored over $\mathbb{Q}$ (Wang's algorithm,
:mod:`~sympy_extras.polys.multifactor`); each of its irreducible factors
$h$ gives the irreducible factor $\gcd(h, g)$ of $g$ over
$\mathbb{Q}(i)$, shifted back, and the multiplicities of the factors of
the original polynomial come from trial division (Trager [1]_). Over
``ZZ_I`` the denominators and the Gaussian contents of the factors are
moved into the coefficient as SymPy does, and the factors are sorted by
SymPy's ``_sort_factors`` (the length of the dense representation, the
multiplicity, the dense coefficients, Gaussian numbers compared by their
imaginary parts first).

Examples
========

>>> from sympy_extras.polys.domains.gaussiandomains import QQ_I
>>> from sympy_extras.polys.rings import PolyRing
>>> from sympy_extras.polys.gaussianfactor import gaussian_ext_factor
>>> R = PolyRing(QQ_I, ['x', 'y'])
>>> f = R.from_dict({(2, 0): QQ_I.one, (0, 2): QQ_I.one})  # x**2 + y**2
>>> lc, factors = gaussian_ext_factor(f)
>>> lc, [(g.to_expr(), k) for g, k in factors]
(GaussianRational(1, 0), [(x - I*y, 1), (x + I*y, 1)])

References
==========

.. [1] B. M. Trager, *Algebraic factoring and rational function
       integration*, Proceedings of SYMSAC '76, ACM, 1976, 219-226
       (algorithms ``sqfr_norm`` and ``alg_factor``).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/factortools.py (dmp_zz_i_factor, dmp_qq_i_factor,
# dmp_ext_factor, dmp_trial_division, dmp_factor_list), sympy/polys/sqfreetools.py
# (dmp_sqf_norm, _dmp_sqf_norm_shifts, dmp_sqf_p, dmp_sqf_part),
# sympy/polys/polyutils.py (_sort_factors) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: sparse polynomials instead of dense nested lists, the field
# QQ<I> only (the norm computed as A**2 + B**2 instead of a resultant),
# the gcd over ZZ_I with the denominators cleared, strict typing.
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Iterator, Union

from .domains.gaussiandomains import QQ_I, ZZ_I, GaussianInteger, GaussianRational
from .domains.rationalfield import QQ
from .domains.sparsepolynomialring import RationalPolynomialRing, SparsePolynomialRing
from .rings import Monomial, PolyElement, PolyRing

__all__ = ['gaussian_ext_factor', 'gaussian_factor_list', 'gaussian_sort_key']

GaussPoly = PolyElement[GaussianRational]
GaussFactors = list[tuple[GaussPoly, int]]
GaussKey = Union[tuple[Fraction, Fraction], tuple['GaussKey', ...]]


def _cleared(f: GaussPoly, Z: SparsePolynomialRing[GaussianInteger]) -> PolyElement[GaussianInteger]:
    """``f`` times the lcm of the denominators of its coefficients (SymPy's
    ``dmp_clear_denoms`` over ``QQ_I``), over ``ZZ_I``."""
    common = ZZ_I.one
    for c in f.terms.values():
        common = ZZ_I.lcm(common, ZZ_I.denom_of(c))
    k = QQ_I.convert(common, ZZ_I)
    return Z.ring.from_dict({m: ZZ_I.convert(QQ_I.mul(c, k), QQ_I) for m, c in f.terms.items()})


def _gcd(f: GaussPoly, g: GaussPoly) -> GaussPoly:
    """The monic gcd over ``QQ<I>`` (SymPy's ``dmp_inner_gcd`` over an
    algebraic field returns it monic), computed over ``ZZ_I``."""
    ring = f.ring
    if not f.terms:
        return g.monic()
    if not g.terms:
        return f.monic()
    Z = SparsePolynomialRing(ZZ_I, ring.names)
    h = Z.gcd(_cleared(f, Z), _cleared(g, Z))
    return ring.from_dict({m: QQ_I.convert(c, ZZ_I) for m, c in h.terms.items()}).monic()


# PORTED FROM SYMPY: sympy/polys/sqfreetools.py:dmp_sqf_part
def _sqf_part(f: GaussPoly) -> GaussPoly:
    """The square-free part of ``f``, monic."""
    if not f.terms:
        return f
    g = f
    for i in range(f.ring.ngens):
        g = _gcd(g, f.diff(i))
        if g.is_ground:
            # the gcd of the remaining derivatives is 1 too
            return f.monic()
    return f.exquo(g).monic()


# PORTED FROM SYMPY: sympy/polys/densetools.py:dmp_shift
def _shift(f: GaussPoly, a: list[GaussianRational]) -> GaussPoly:
    """``f(x_1 + a_1, ..., x_n + a_n)``."""
    ring = f.ring
    terms = dict(f.terms)
    for i, ai in enumerate(a):
        if not ai:
            continue
        shifted: dict[Monomial, GaussianRational] = {}
        for m, c in terms.items():
            e = m[i]
            for k in range(e + 1):
                coefficient = QQ_I.mul(c, QQ_I.mul(QQ_I.from_int(comb(e, k)), QQ_I.pow(ai, e - k)))
                monomial = m[:i] + (k,) + m[i + 1:]
                shifted[monomial] = QQ_I.add(shifted.get(monomial, QQ_I.zero), coefficient)
        terms = {m: c for m, c in shifted.items() if c}
    return ring.from_dict(terms)


def _norm(f: GaussPoly) -> PolyElement[Fraction]:
    """The norm ``f * conjugate(f) = A**2 + B**2`` over ``QQ`` (SymPy's
    ``dmp_norm``: the resultant with ``y**2 + 1`` of ``A + y*B``)."""
    ring = PolyRing(QQ, f.ring.names)
    A = ring.from_dict({m: c.x for m, c in f.terms.items()})
    B = ring.from_dict({m: c.y for m, c in f.terms.items()})
    return A * A + B * B


# PORTED FROM SYMPY: sympy/polys/sqfreetools.py:dmp_sqf_p
def _sqf_p(r: PolyElement[Fraction]) -> bool:
    """Whether ``r`` is square-free over ``QQ``."""
    if not r.terms:
        return True
    Q = RationalPolynomialRing(r.ring.names)
    for i in range(r.ring.ngens):
        d = r.diff(i)
        if not d.terms:
            continue
        if Q.gcd(r, d).degree_in(i) != 0:
            return False
    return True


# PORTED FROM SYMPY: sympy/polys/sqfreetools.py:_dmp_sqf_norm_shifts
def _sqf_norm_shifts(f: GaussPoly) -> Iterator[tuple[list[int], GaussPoly]]:
    """The candidate shifts ``s`` and ``f(X - s*I)``, in SymPy's order."""
    n = f.ring.ngens
    s0 = [0] * n
    yield s0, f
    degrees = [f.degree_in(i) for i in range(n)]
    var_indices = [i for di, i in sorted(zip(degrees, range(n))) if di > 0]
    for i in var_indices:
        s1 = list(s0)
        s1[i] = 1
        yield s1, _shift(f, [GaussianRational(Fraction(0), Fraction(-si)) for si in s1])
    j = 0
    while True:
        j += 1
        yield [j] * n, _shift(f, [GaussianRational(Fraction(0), Fraction(-j))] * n)


# PORTED FROM SYMPY: sympy/polys/sqfreetools.py:dmp_sqf_norm
def _sqf_norm(f: GaussPoly) -> tuple[list[int], GaussPoly, PolyElement[Fraction]]:
    """``(s, g, r)``: ``g = f(X - s*I)`` with the square-free norm ``r``."""
    for s, g in _sqf_norm_shifts(f):
        r = _norm(g)
        if _sqf_p(r):
            return s, g, r
    raise RuntimeError('unreachable')  # pragma: no cover


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_trial_division
def _trial_division(f: GaussPoly, factors: list[GaussPoly]) -> GaussFactors:
    result: GaussFactors = []
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
    return sorted(result, key=gaussian_sort_key)


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_ext_factor
def gaussian_ext_factor(f: GaussPoly) -> tuple[GaussianRational, GaussFactors]:
    """The leading coefficient and the monic irreducible factors over
    ``QQ<I>`` with their multiplicities (Trager's algorithm).

    >>> from sympy_extras.polys.domains.gaussiandomains import QQ_I
    >>> from sympy_extras.polys.rings import PolyRing
    >>> R = PolyRing(QQ_I, ['x', 'y'])
    >>> x, y = R.gens
    >>> lc, factors = gaussian_ext_factor((x*x + y*y)*(x - y)*(x - y))
    >>> [(g.to_expr(), k) for g, k in factors]
    [(x - I*y, 1), (x + I*y, 1), (x - y, 2)]
    """
    from .polytools import _factor_all
    if not f.terms:
        return QQ_I.zero, []
    lc = f.LC
    f = f.monic()
    if f.is_ground:
        return lc, []
    F = f
    f = _sqf_part(f)
    s, g, r = _sqf_norm(f)
    _, rational_factors = _factor_all(r)
    factors: list[GaussPoly]
    if len(rational_factors) == 1:
        factors = [f]
    else:
        factors = []
        a = [GaussianRational(Fraction(0), Fraction(si)) for si in s]
        for h, _k in rational_factors:
            hk = f.ring.from_dict({m: GaussianRational(Fraction(c)) for m, c in h.terms.items()})
            hg = _gcd(hk, g)
            g = g.exquo(hg)
            factors.append(_shift(hg, a))
    return lc, _trial_division(F, factors)


def _dense_key(terms: dict[Monomial, GaussianRational], i: int, n: int) -> GaussKey:
    """The dense nested representation from the variable ``i`` on, the
    coefficients ``(y, x)`` as SymPy compares Gaussian numbers."""
    by_degree: dict[int, dict[Monomial, GaussianRational]] = {}
    for m, c in terms.items():
        by_degree.setdefault(m[i], {})[m] = c
    if not by_degree:
        return ()
    top = max(by_degree)
    if i == n - 1:
        values: list[GaussKey] = []
        for d in range(top, -1, -1):
            level = by_degree.get(d, {})
            c = next(iter(level.values())) if level else QQ_I.zero
            values.append((c.y, c.x))
        return tuple(values)
    return tuple(_dense_key(by_degree.get(d, {}), i + 1, n) for d in range(top, -1, -1))


GaussSortKey = tuple[int, int, GaussKey]


# PORTED FROM SYMPY: sympy/polys/polyutils.py:_sort_factors (multiple=True)
def gaussian_sort_key(item: tuple[GaussPoly, int]) -> GaussSortKey:
    """SymPy's key of a factor with its multiplicity: the length of the
    dense representation (the degree in the first variable plus one),
    the multiplicity, then the dense coefficients."""
    f, k = item
    length = f.degree_in(0) + 1 if f.terms else 0
    return (length, k, _dense_key(dict(f.terms), 0, f.ring.ngens))


def _content(f: PolyElement[GaussianInteger]) -> GaussianInteger:
    """SymPy's ``dmp_ground_content`` over ``ZZ_I``."""
    content = ZZ_I.zero
    for c in f.terms.values():
        content = ZZ_I.gcd(content, c)
        if content == ZZ_I.one:
            break
    return content


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_factor_list (the domains ZZ_I and QQ_I),
# dmp_zz_i_factor, dmp_qq_i_factor
def gaussian_factor_list(f: GaussPoly, integral: bool) -> tuple[GaussianRational, GaussFactors]:
    """The coefficient and the irreducible factors of ``f`` over ``ZZ_I``
    (``integral``, the coefficients Gaussian integers) or ``QQ_I``, as
    SymPy's ``dmp_factor_list`` returns them: the monomial of the
    variables split off, over ``ZZ_I`` the factors primitive Gaussian
    integer polynomials, over ``QQ_I`` monic; sorted.

    >>> from sympy_extras.polys.domains.gaussiandomains import QQ_I, GaussianRational
    >>> from sympy_extras.polys.rings import PolyRing
    >>> R = PolyRing(QQ_I, ['x', 'y'])
    >>> x, y = R.gens
    >>> two = R.ground(GaussianRational(Fraction(2)))
    >>> c, factors = gaussian_factor_list(two*x*(x*x + y*y), True)
    >>> QQ_I.to_expr(c), [(g.to_expr(), k) for g, k in factors]
    (2, [(x, 1), (x - I*y, 1), (x + I*y, 1)])
    """
    ring = f.ring
    n = ring.ngens
    if not f.terms:
        return QQ_I.zero, []
    # dmp_terms_gcd
    J = [min(m[i] for m in f.terms) for i in range(n)]
    f = ring.from_dict({tuple(e - j for e, j in zip(m, J)): c for m, c in f.terms.items()})
    cont = QQ_I.one
    Z = SparsePolynomialRing(ZZ_I, ring.names)
    if integral:
        # dmp_ground_primitive over ZZ_I
        content = _content(Z.ring.from_dict({m: ZZ_I.convert(c, QQ_I) for m, c in f.terms.items()}))
        cont = QQ_I.convert(content, ZZ_I)
        f = ring.from_dict({m: QQ_I.exquo(c, cont) for m, c in f.terms.items()})
    # dmp_qq_i_factor: over QQ<I> (the content over the field QQ_I only
    # changes the leading coefficient, which is that of f in the end)
    coeff, factors = gaussian_ext_factor(f)
    if integral:
        # dmp_zz_i_factor: the denominators and the contents over ZZ_I of
        # the factors moved into the coefficient
        new_factors: GaussFactors = []
        for fac, k in factors:
            denominator = ZZ_I.one
            for c in fac.terms.values():
                denominator = ZZ_I.lcm(denominator, ZZ_I.denom_of(c))
            d = QQ_I.convert(denominator, ZZ_I)
            numerator = Z.ring.from_dict({m: ZZ_I.convert(QQ_I.mul(c, d), QQ_I) for m, c in fac.terms.items()})
            fac_content = _content(numerator)
            primitive = ring.from_dict({m: QQ_I.convert(ZZ_I.exquo(c, fac_content), ZZ_I)
                                        for m, c in numerator.terms.items()})
            coeff = QQ_I.exquo(QQ_I.mul(coeff, QQ_I.pow(QQ_I.convert(fac_content, ZZ_I), k)), QQ_I.pow(d, k))
            new_factors.append((primitive, k))
        factors = new_factors
    for i in range(n - 1, -1, -1):
        if J[i]:
            factors.insert(0, (ring.term(tuple(1 if t == i else 0 for t in range(n)), QQ_I.one), J[i]))
    return QQ_I.mul(coeff, cont), sorted(factors, key=gaussian_sort_key)

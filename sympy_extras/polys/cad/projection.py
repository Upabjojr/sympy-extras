"""Projection operators for cylindrical algebraic decomposition.

The projection phase of CAD takes a set of polynomials in
`x_1, \\ldots, x_k` and produces a set of polynomials in
`x_1, \\ldots, x_{k-1}` such that, over every cell of a decomposition of
`\\mathbb{R}^{k-1}` which is sign-invariant for the projected set, the real
roots of the original polynomials in `x_k` are delineable: they are given by
finitely many continuous functions, which do not cross, and the polynomials
have constant signs between them.

Two operators are implemented:

* :func:`mccallum_projection` is the projection of McCallum [1]_ applied to a
  squarefree basis: coefficients, discriminants and pairwise resultants. It
  gives small projection sets but it is only guaranteed to work if no
  polynomial of the projection factor sets vanishes identically on a cell of
  positive dimension (the set is *well-oriented*). The lifting phase checks
  this and falls back to the second operator if the check fails.

* :func:`hong_projection` is the projection of Hong [2]_, an improvement of
  the original operator of Collins, based on the reducta of the polynomials
  and on their principal subresultant coefficients. It is always correct but
  produces many more polynomials.

When the formula implies that one of the polynomials vanishes (an
*equational constraint*), McCallum's projection is reduced to the
coefficients and discriminants of the factors of that polynomial and to
their resultants with the others [3]_: the other polynomials are then
sign-invariant on the sections of the constraint, which is where the
formula can hold. See :func:`mccallum_projection`.

The module is native: the polynomials are the typed
:class:`~sympy_extras.polys.polytools.Poly` over ``ZZ`` in symbols of the
core, and SymPy is not imported; the SymPy face, on SymPy's ``Poly``, is
:mod:`sympy_extras.polys.cad.projection_sympy`.

References
==========

.. [1] S. McCallum, An improved projection operation for cylindrical
       algebraic decomposition, in: Quantifier Elimination and Cylindrical
       Algebraic Decomposition, Springer, 1998, pp. 242-268.
.. [2] H. Hong, An improvement of the projection operator in cylindrical
       algebraic decomposition, ISSAC 1990, pp. 261-264.
.. [3] S. McCallum, On projection in CAD-based quantifier elimination with
       equational constraint, ISSAC 1999, pp. 145-149.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import lcm
from typing import Callable, Iterable, Optional, Sequence, Union

from sympy_extras.core.expr import Expr, ExprLike
from sympy_extras.polys.domains.sparsepolynomialring import IntegerPolynomialRing
from sympy_extras.polys.euclidtools import dup_inner_subresultants
from sympy_extras.polys.polyerrors import BasePolynomialError, PolynomialError
from sympy_extras.polys.polytools import Poly
from sympy_extras.polys.rings import Monomial, PolyElement

__all__ = ['squarefree_basis', 'mccallum_projection', 'hong_projection', 'projection_sets',
           'augmented_projection_sets', 'to_polys', 'PROJECTIONS', 'level_of', 'Projection']

#: the polynomial arguments
PolyLike = Union[Poly, ExprLike]

#: the coefficients of a polynomial: monomials to rational numbers
Terms = dict[Monomial, Fraction]

#: a projection operator
Projection = Callable[[Sequence[Poly], Expr], list[Poly]]


def _in_gens(f: Poly, gens: Sequence[Expr]) -> Poly:
    """``f`` written in the generators ``gens``, which must contain those
    of its generators which it depends on."""
    target = tuple(gens)
    if f.gens == target:
        return f
    positions: list[int] = []
    for i, g in enumerate(f.gens):
        if g in target:
            positions.append(target.index(g))
        elif f.degree(i) > 0:
            raise PolynomialError("%s depends on %s, which is not in %s" % (f, g, target))
        else:
            positions.append(-1)
    terms: dict[Monomial, Union[int, Fraction]] = {}
    for m, c in f.rep.terms.items():
        exponents = [0]*len(target)
        for i, e in enumerate(m[:len(positions)]):
            if e:
                exponents[positions[i]] = e
        terms[tuple(exponents)] = c
    return Poly.from_dict(terms, *target)


def to_polys(polys: Iterable[PolyLike], gens: Sequence[Expr]) -> list[Poly]:
    """Convert ``polys`` to polynomials over ZZ in ``gens``: rational
    coefficients are cleared of their denominators (``PolynomialError``
    for coefficients which are not rational)."""
    result = []
    generators = tuple(gens)
    for p in polys:
        try:
            if isinstance(p, Poly):
                q = p if p.gens == generators else Poly(p.as_expr(), *generators)
            else:
                q = Poly(p, *generators)
        except BasePolynomialError as error:
            raise PolynomialError("polynomial with rational coefficients expected: %s" % (error,))
        if q.domain not in ('ZZ', 'QQ'):
            raise PolynomialError("polynomial with rational coefficients expected, got %s" % (q,))
        if not q.is_integral:
            d = 1
            for c in q.rep.terms.values():
                d = lcm(d, c.denominator)
            q = Poly.from_dict({m: int(c*d) for m, c in q.rep.terms.items()}, *generators)
        result.append(q)
    return result


def level_of(f: Poly, gens: Sequence[Expr]) -> int:
    """Index of the last generator of ``gens`` that ``f`` depends on."""
    for k in range(len(gens), 0, -1):
        if gens[k - 1] in f.gens and f.degree(gens[k - 1]) > 0:
            return k
    return 0


def _leading_negative(f: Poly) -> bool:
    """Whether the leading coefficient of ``f`` (lexicographic order of
    its generators) is negative."""
    return f.rep.terms[max(f.rep.terms)] < 0


def _factors(f: Poly) -> list[Poly]:
    """Irreducible non-constant factors of ``f`` with positive leading
    coefficient."""
    factors = []
    for g, _ in f.factor_list()[1]:
        if g.is_ground:
            continue
        if _leading_negative(g):
            g = -g
        factors.append(g)
    return factors


def squarefree_basis(polys: Iterable[PolyLike], gens: Sequence[Expr]) -> list[list[Poly]]:
    """Finest squarefree basis of ``polys``, split by level.

    Returns a list ``[B_1, ..., B_n]`` where ``n = len(gens)`` and ``B_k``
    contains the irreducible factors (over ZZ, with positive leading
    coefficient) of the polynomials that depend on ``gens[k-1]`` but not on
    any later generator. Constants are dropped.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad.projection import squarefree_basis
    >>> x, y = symbols('x y')
    >>> squarefree_basis([(x**2 - 1)*(x + y)**2, 6], [x, y])
    [[Poly(x - 1, x, y, domain='ZZ'), Poly(x + 1, x, y, domain='ZZ')], [Poly(x + y, x, y, domain='ZZ')]]

    The polynomials in ``B_k`` are pairwise coprime and squarefree, so that
    the sign of each input polynomial is determined by the signs of the basis
    elements.
    """
    levels: list[list[Poly]] = [[] for _ in gens]
    for f in to_polys(polys, gens):
        for g in _factors(f):
            k = level_of(g, gens)
            if k and g not in levels[k - 1]:
                levels[k - 1].append(g)
    return levels


def _main_first(f: Poly, x: Expr) -> Poly:
    """Reorder the generators of ``f`` so that ``x`` comes first."""
    return _in_gens(f, (x,) + tuple(g for g in f.gens if g != x))


def _coefficients(f: Poly) -> list[Terms]:
    """Coefficients of ``f`` with respect to its first generator, from the
    leading one down (zeros included), as terms in the other generators."""
    n = f.degree()
    coefficients: list[Terms] = [{} for _ in range(n + 1)]
    for m, c in f.rep.terms.items():
        coefficients[n - m[0]][m[1:]] = c
    return coefficients


def _is_constant(terms: Terms) -> bool:
    return all(not any(m) for m in terms)


def _poly_of(terms: Terms, f: Poly) -> Poly:
    """The polynomial with the coefficients ``terms`` in the generators of
    ``f`` after the first one."""
    return Poly.from_dict(dict(terms), *f.gens[1:])


def _psc(f: Poly, g: Poly) -> list[Poly]:
    """Principal subresultant coefficients of ``f`` and ``g`` (with the
    same generators, at least two) with respect to their first generator;
    the constant ones are left out."""
    rest = f.gens[1:]
    names = ['_r%d' % i for i in range(len(rest))]
    R = IntegerPolynomialRing(names)

    def dense(h: Poly) -> list[PolyElement[int]]:
        n = h.degree()
        by_degree: list[dict[Monomial, int]] = [{} for _ in range(n + 1)]
        for m, c in h.rep.terms.items():
            by_degree[n - m[0]][m[1:]] = int(c)
        return [R.ring.from_dict(d) for d in by_degree]

    a, b = dense(f), dense(g)
    if not a or not b:
        return []
    S, T = dup_inner_subresultants(a, b, R)
    if not S:
        return []
    # the coefficient of degree j is the leading coefficient of the
    # subresultant of degree j, zero when it is defective; in increasing
    # order of j
    m = min(len(a), len(b)) - 1
    psc: list[PolyElement[int]] = [R.zero]*(m + 1)
    for r, s in zip(S[1:], T[1:]):
        psc[len(r) - 1] = s
    return [Poly.from_dict({m: c for m, c in s.terms.items()}, *rest) for s in psc if not s.is_ground]


def _reducta(f: Poly) -> list[Poly]:
    """The nonzero reducta of ``f`` with respect to its first generator:
    ``f``, ``f`` minus its leading term, and so on."""
    reducta = []
    terms = dict(f.rep.terms)
    while terms:
        reducta.append(Poly.from_dict(dict(terms), *f.gens))
        top = max(m[0] for m in terms)
        terms = {m: c for m, c in terms.items() if m[0] != top}
    return reducta


def _collect(items: Iterable[Poly]) -> list[Poly]:
    """Irreducible factors of the non-constant polynomials in ``items``."""
    result: list[Poly] = []
    for c in items:
        if c.is_ground:
            continue
        for g in _factors(c):
            if g not in result:
                result.append(g)
    return result


def mccallum_projection(polys: Sequence[Poly], x: Expr, equational: Sequence[Poly] = ()) -> list[Poly]:
    """McCallum's projection of ``polys`` with respect to ``x``.

    ``polys`` must be a squarefree basis (pairwise coprime squarefree
    polynomials) of positive degree in ``x``, see :func:`squarefree_basis`.
    The projection consists of the irreducible factors of the coefficients
    with respect to ``x``, of the discriminants and of the pairwise
    resultants.

    Only the coefficients from the leading one down to the first nonzero
    constant coefficient are used: they are all that is needed for the
    degree to be invariant on each cell, and if some coefficient is a nonzero
    constant the polynomial cannot vanish identically anywhere.

    ``equational`` is a subset `E` of ``polys`` such that the formula
    implies that one of them vanishes (the factors of an equational
    constraint): the projection is then reduced to the coefficients and
    discriminants of the elements of `E`, their pairwise resultants and
    their resultants with the other polynomials [3]_.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.polytools import Poly
    >>> from sympy_extras.polys.cad.projection import mccallum_projection
    >>> x, y = symbols('x y')
    >>> mccallum_projection([Poly(x**2 + y**2 - 1, x, y)], y)
    [Poly(x - 1, x, domain='ZZ'), Poly(x + 1, x, domain='ZZ')]
    >>> mccallum_projection([Poly(x*y - 1, x, y), Poly(y - x, x, y)], y)
    [Poly(x, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ'), Poly(x + 1, x, domain='ZZ')]
    >>> mccallum_projection([Poly(x*y - 1, x, y), Poly(y - x, x, y)], y, equational=[Poly(y - x, x, y)])
    [Poly(x - 1, x, domain='ZZ'), Poly(x + 1, x, domain='ZZ')]
    """
    main = [_main_first(f, x) for f in polys]
    constraints = [_main_first(f, x) for f in equational]
    if constraints:
        if any(e not in main for e in constraints):
            raise ValueError("the equational constraints must be among the polynomials")
    else:
        constraints = main
    others = [f for f in main if f not in constraints]
    items: list[Poly] = []

    def add(value: Union[Poly, Expr]) -> None:
        if isinstance(value, Poly):
            items.append(value)

    for f in constraints:
        if len(f.gens) > 1:
            for c in _coefficients(f):
                if not _is_constant(c):
                    items.append(_poly_of(c, f))
                elif c:
                    break
        if f.degree() >= 2:
            add(f.discriminant())
    for f, g in combinations(constraints, 2):
        add(f.resultant(g))
    for f in constraints:
        for g in others:
            add(f.resultant(g))
    return _collect(items)


def hong_projection(polys: Sequence[Poly], x: Expr) -> list[Poly]:
    """Hong's projection of ``polys`` with respect to ``x``.

    ``polys`` must be a squarefree basis of positive degree in ``x``. For
    every reductum `r` of every polynomial the projection contains the
    leading coefficient of `r` and the principal subresultant coefficients of
    `r` and its derivative; for every pair `f, g` it contains the principal
    subresultant coefficients of every reductum of `f` with `g`.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.polytools import Poly
    >>> from sympy_extras.polys.cad.projection import hong_projection
    >>> x, y = symbols('x y')
    >>> hong_projection([Poly(x**2 + y**2 - 1, x, y)], y)
    [Poly(x - 1, x, domain='ZZ'), Poly(x + 1, x, domain='ZZ')]
    >>> hong_projection([Poly(x*y - 1, x, y), Poly(y - x, x, y)], y)
    [Poly(x, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ'), Poly(x + 1, x, domain='ZZ')]
    """
    main = [_main_first(f, x) for f in polys]
    items: list[Poly] = []
    if all(len(f.gens) == 1 for f in main):
        return []
    for f in main:
        for r in _reducta(f):
            items.append(_poly_of(_coefficients(r)[0], r))
            if r.degree() >= 2:
                items.extend(_psc(r, r.diff(0)))
    for f, g in combinations(main, 2):
        for r in _reducta(f):
            if r.degree() == 0:
                items.append(_poly_of(_coefficients(r)[0], r))
            else:
                items.extend(_psc(r, g))
    return _collect(items)


#: the projection operators by name
PROJECTIONS: dict[str, Projection] = {
    'mccallum': mccallum_projection,
    'hong': hong_projection,
}


def projection_sets(polys: Iterable[PolyLike], gens: Sequence[Expr], method: str = 'mccallum',
                    equational: Optional[PolyLike] = None) -> list[list[Poly]]:
    """Projection factor sets of ``polys`` for the variables ``gens``.

    The last generator is projected first. Returns a list
    ``[P_1, ..., P_n]`` where ``P_k`` is a squarefree basis of polynomials in
    ``gens[:k]`` of positive degree in ``gens[k-1]``: ``P_n`` is the basis of
    the input and each ``P_k`` for ``k < n`` contains the projection of
    ``P_{k+1}`` together with the factors of the input (and of the
    projections of higher levels) that only depend on ``gens[:k]``.

    ``method`` is ``'mccallum'`` (default) or ``'hong'``. With McCallum's
    projection, ``equational`` may be one of ``polys`` which the formula
    to be decided implies to vanish: the projection of the last level is
    reduced to its factors (see :func:`mccallum_projection`), when they
    all depend on the last generator.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad.projection import projection_sets
    >>> x, y = symbols('x y')
    >>> projection_sets([x**2 + y**2 - 1, x - y], [x, y])
    [[Poly(x - 1, x, domain='ZZ'), Poly(x + 1, x, domain='ZZ'), Poly(2*x**2 - 1, x, domain='ZZ')], [Poly(x**2 + y**2 - 1, x, y, domain='ZZ'), Poly(x - y, x, y, domain='ZZ')]]
    >>> projection_sets([x**2 + y**2 - 1, x - y], [x, y], equational=x - y)
    [[Poly(2*x**2 - 1, x, domain='ZZ')], [Poly(x**2 + y**2 - 1, x, y, domain='ZZ'), Poly(x - y, x, y, domain='ZZ')]]
    """
    try:
        project = PROJECTIONS[method]
    except KeyError:
        raise ValueError("unknown projection method %r" % (method,))

    generators = list(gens)
    if not generators:
        raise ValueError("at least one generator is needed")

    levels = squarefree_basis(polys, generators)
    n = len(generators)
    constraints: list[Poly] = []
    if equational is not None and method == 'mccallum':
        [constraint] = to_polys([equational], generators)
        factors = _factors(constraint)
        if factors and all(level_of(g, generators) == n for g in factors):
            constraints = factors

    result: list[list[Poly]] = [[] for _ in range(n)]
    for k in range(n, 0, -1):
        current = [_in_gens(f, generators[:k]) for f in levels[k - 1]]
        result[k - 1] = current
        if k == 1:
            break
        if k == n and constraints:
            projected = mccallum_projection(current, generators[k - 1],
                                            [_in_gens(f, generators[:k]) for f in constraints])
        else:
            projected = project(current, generators[k - 1])
        for g in projected:
            g = _in_gens(g, generators)
            j = level_of(g, generators)
            if j and g not in levels[j - 1]:
                levels[j - 1].append(g)
    return result


def augmented_projection_sets(projection: Sequence[Sequence[Poly]], gens: Sequence[Expr],
                              extra: Iterable[PolyLike], method: str = 'mccallum') -> list[list[Poly]]:
    """The projection factor sets ``projection`` (``P_1, ..., P_n`` for
    the variables ``gens``, as :func:`projection_sets` gives them) with
    the irreducible factors of the polynomials ``extra`` added at their
    levels, and the levels below re-projected: from the top level down,
    the projection of the factors which are new at a level is added to
    the levels below (with McCallum's operator their coefficients and
    discriminants and their resultants with every factor of the level,
    with Hong's the projection of the whole level). Every set contains
    the one it comes from, so that a decomposition sign-invariant for the
    result refines one for ``projection``: this is the augmented
    projection of Collins and Hong for the solution formulas, with the
    derivatives of the factors for ``extra`` [1]_.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad.projection import projection_sets, augmented_projection_sets
    >>> x, y = symbols('x y')
    >>> projection = projection_sets([x - y**2], [x, y])
    >>> projection
    [[Poly(x, x, domain='ZZ')], [Poly(x - y**2, x, y, domain='ZZ')]]
    >>> augmented_projection_sets(projection, [x, y], [2*y])
    [[Poly(x, x, domain='ZZ')], [Poly(x - y**2, x, y, domain='ZZ'), Poly(y, x, y, domain='ZZ')]]
    >>> augmented_projection_sets(projection, [x, y], [y - 1])
    [[Poly(x, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ')], [Poly(x - y**2, x, y, domain='ZZ'), Poly(y - 1, x, y, domain='ZZ')]]

    References
    ==========

    .. [1] H. Hong, Simple solution formula construction in cylindrical
           algebraic decomposition based quantifier elimination, ISSAC
           1992, pp. 177-188.
    """
    try:
        project = PROJECTIONS[method]
    except KeyError:
        raise ValueError("unknown projection method %r" % (method,))
    generators = list(gens)
    n = len(generators)
    if len(projection) != n:
        raise ValueError("expected %d projection factor sets" % n)
    levels: list[list[Poly]] = [list(level) for level in projection]
    new: list[list[Poly]] = [[] for _ in generators]

    def add(g: Poly) -> None:
        g = _in_gens(g, generators)
        j = level_of(g, generators)
        if j:
            g = _in_gens(g, generators[:j])
            if g not in levels[j - 1]:
                levels[j - 1].append(g)
                new[j - 1].append(g)

    for f in to_polys(extra, generators):
        for g in _factors(f):
            add(g)
    for k in range(n, 1, -1):
        if not new[k - 1]:
            continue
        if method == 'mccallum':
            projected = mccallum_projection(levels[k - 1], generators[k - 1], new[k - 1])
        else:
            projected = project(levels[k - 1], generators[k - 1])
        for g in projected:
            add(g)
    return levels

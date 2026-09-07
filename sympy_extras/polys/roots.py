"""Exact roots of univariate polynomials in radicals.

SymPy writes the roots of a polynomial in radicals (:func:`sympy.roots`)
for degrees up to four, for binomials and cyclotomic polynomials, and for
polynomials which are compositions of such (functional decomposition,
:func:`sympy.decompose`); the exact root objects
(:class:`~sympy.polys.rootoftools.ComplexRootOf`) it returns otherwise are
isolated by validated numerics. This module converts root objects to
radicals when that is possible and the radical expressions are of a
reasonable size: for quadratic, binomial and cyclotomic polynomials (the
policy of SymPy's ``CRootOf(..., radicals=True)``), and for polynomials
which decompose into such factors.
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.evalf import N
from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, decompose
from sympy.polys.rootoftools import CRootOf, ComplexRootOf
from sympy.sets.sets import Set

from sympy_extras._typing import as_expr, as_set, free_symbols
from sympy_extras.settings import settings

__all__ = ['radical_form', 'in_radicals']


def _poly_of(root: ComplexRootOf) -> Poly:
    expr = root.expr
    [gen] = sorted(free_symbols(expr), key=str)
    return Poly(expr, gen)


def _small_components(poly: Poly) -> bool:
    """Whether every component of the functional decomposition has
    degree at most two, or is a binomial or cyclotomic."""
    for component in decompose(poly):
        if component.degree() <= 2 or component.length() <= 2 or component.is_cyclotomic:
            continue
        return False
    return True


def radical_form(root: ComplexRootOf) -> Optional[Expr]:
    """The root written in radicals, or ``None`` when it is left as a root
    object.

    Examples
    ========

    >>> from sympy import CRootOf
    >>> from sympy.abc import x
    >>> from sympy_extras.polys.roots import radical_form
    >>> radical_form(CRootOf(x**2 - 2, 1))
    sqrt(2)
    >>> radical_form(CRootOf(x**6 - 2*x**3 - 1, 1))
    (1 + sqrt(2))**(1/3)
    >>> radical_form(CRootOf(x**5 - x - 1, 0)) is None
    True
    """
    poly = _poly_of(root)
    index = int(as_expr(root.args[1]))
    direct = CRootOf(poly, index, radicals=True)
    if not isinstance(direct, ComplexRootOf):
        return as_expr(direct)
    if poly.degree() <= 4 or not _small_components(poly):
        return None
    candidates = [as_expr(r) for r, m in roots(poly).items() for _ in range(m)]
    if len(candidates) != poly.degree():
        return None
    return _identify(root, candidates)


def _identify(root: ComplexRootOf, candidates: list[Expr]) -> Optional[Expr]:
    """The candidate closest to the root, if clearly closer than the
    others (the candidates are exactly the roots, which are distinct)."""
    digits = settings.precision
    value = N(root, digits)
    distances: list[tuple[Expr, Expr]] = []
    for c in candidates:
        d = N(abs(as_expr(N(c, digits)) - value), digits)
        if not (isinstance(d, Expr) and d.is_number and d.is_finite):
            return None
        distances.append((as_expr(d), c))
    distances.sort(key=lambda pair: pair[0])
    closest, expression = distances[0]
    if len(distances) > 1 and distances[1][0] < Rational(1, 10)**(digits//3):
        return None
    if closest > Rational(1, 10)**(digits*2//3):
        return None
    return expression


def in_radicals(result: Set) -> Set:
    """The set with its root objects written in radicals when
    :func:`radical_form` allows.

    Examples
    ========

    >>> from sympy import CRootOf, FiniteSet
    >>> from sympy.abc import x
    >>> from sympy_extras.polys.roots import in_radicals
    >>> in_radicals(FiniteSet(CRootOf(x**2 - 2, 0), CRootOf(x**5 - x - 1, 0)))
    {-sqrt(2), CRootOf(x**5 - x - 1, 0)}
    """
    replacements: dict[Basic, Basic] = {}
    for r in result.atoms(ComplexRootOf):
        form = radical_form(r)
        if form is not None:
            replacements[r] = form
    return as_set(result.xreplace(replacements)) if replacements else result

"""Exact roots of univariate polynomials in radicals.

SymPy writes the roots of a polynomial in radicals (``roots``) for
degrees up to four, for binomials and cyclotomic polynomials, and for
polynomials which are compositions of such (functional decomposition);
the exact root objects (SymPy's ``CRootOf``) it returns otherwise are
isolated by validated numerics. This module converts root objects to
radicals when that is possible and the radical expressions are of a
reasonable size: for quadratic, binomial and cyclotomic polynomials (the
policy of SymPy's ``CRootOf(..., radicals=True)``), and for polynomials
which decompose into such factors.

The module works on objects of the core and imports no SymPy
(:mod:`sympy_extras.polys.roots_sympy` is its face on SymPy objects):
the roots in radicals are those of the native
:mod:`~sympy_extras.polys.polyroots`. The root objects are SymPy's
``CRootOf``, which is not ported: they reach the core as wrapped foreign
nodes (``ComplexRootOf(polynomial, index)``), recognised here by their
name, and their numerical values come from SymPy through the bridge of
the core (:mod:`sympy_extras.core.bridge`) when one is registered. One
step goes through SymPy at run time (the face holds it,
:func:`~sympy_extras.polys.roots_sympy.roots_of_core`): SymPy's ``roots``
of a polynomial whose components need its algebraic domain ``EX``, which
the native :func:`~sympy_extras.polys.polyroots.roots` does not port.
The trivial roots of a root object follow SymPy's
``ComplexRootOf._roots_trivial`` (``sympy/polys/rootoftools.py``), ported.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/rootoftools.py (ComplexRootOf._roots_trivial)
# (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed, on the native Poly and polyroots, radicals=True only.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.evalf import N
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Float, Integer, Rational
from sympy_extras.functions.elementary.complexes import abs_
from sympy_extras.sets.sets import Set
from sympy_extras.settings import settings

from .polyerrors import BasePolynomialError
from .polyroots import _decompose, _is_cyclotomic, roots, roots_binomial, roots_linear, roots_quadratic
from .polytools import Poly

__all__ = ['radical_form', 'in_radicals', 'polynomial_root']

#: the name of SymPy's root objects, wrapped in the core
_ROOT_CLASS = 'ComplexRootOf'


def polynomial_root(e: Basic) -> Optional[tuple[Poly, int]]:
    """The polynomial and the index of a root object (SymPy's ``CRootOf``
    wrapped in the core), ``None`` for any other node.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.roots import polynomial_root
    >>> polynomial_root(Symbol('x')) is None
    True

    The real roots with rational coefficients of the CAD
    (:class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot`) are root
    objects too:

    >>> from sympy_extras.polys.cad.indexedroot import IndexedRoot
    >>> x = Symbol('x')
    >>> polynomial_root(IndexedRoot(x**3 - 2*x, x, 2))
    (Poly(x**2 - 2, x, domain='ZZ'), 1)
    """
    from sympy_extras.polys.cad.indexedroot import IndexedRoot
    if isinstance(e, IndexedRoot):
        return e.polynomial_root()
    if not (e._is_foreign_node() and e._compare_name() == _ROOT_CLASS and len(e.args) == 2):
        return None
    expr, index = e.args
    if not isinstance(expr, Expr) or not isinstance(index, Integer):
        return None
    gens = sorted(expr.free_symbols, key=str)
    if len(gens) != 1:
        return None
    try:
        return Poly(expr, gens[0]), index.p
    except BasePolynomialError:
        return None


def _small_components(poly: Poly) -> bool:
    """Whether every component of the functional decomposition has
    degree at most two, or is a binomial or cyclotomic."""
    for component in _decompose(poly):
        if component.degree() <= 2 or component.length() <= 2 or _is_cyclotomic(component):
            continue
        return False
    return True


# PORTED FROM SYMPY: sympy/polys/rootoftools.py:ComplexRootOf._roots_trivial
def _trivial_roots(poly: Poly) -> Optional[list[Expr]]:
    """The roots of SymPy's ``CRootOf._roots_trivial`` with radicals: of a
    linear, quadratic or binomial polynomial, in the order of the root
    indices; ``None`` otherwise."""
    if poly.degree() == 1:
        return roots_linear(poly)
    if poly.degree() == 2:
        return roots_quadratic(poly)
    if poly.length() == 2 and poly.TC() != 0:
        return roots_binomial(poly)
    return None


def radical_form(root: Expr) -> Optional[Expr]:
    """The root object (see :func:`polynomial_root`) written in radicals,
    or ``None`` when it is left as a root object (or is not one).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.roots import radical_form
    >>> radical_form(Symbol('x')) is None
    True
    """
    found = polynomial_root(root)
    if found is None:
        return None
    poly, index = found
    # SymPy's CRootOf(poly, index, radicals=True)
    direct = _trivial_roots(poly)
    if direct is not None:
        return direct[index] if 0 <= index < len(direct) else None
    if poly.degree() <= 4 or not _small_components(poly):
        return None
    try:
        listed = roots(poly)
    except NotImplementedError:
        # the roots over SymPy's algebraic domain EX (after a
        # decomposition) are not ported: SymPy's roots, through the face
        from .roots_sympy import roots_of_core
        listed = roots_of_core(poly.as_expr(), poly.gen)
    if not isinstance(listed, dict):
        return None
    candidates = [r for r, m in listed.items() for _ in range(m)]
    if len(candidates) != poly.degree():
        return None
    return _identify(root, candidates)


def _identify(root: Expr, candidates: list[Expr]) -> Optional[Expr]:
    """The candidate closest to the root, if clearly closer than the
    others (the candidates are exactly the roots, which are distinct)."""
    digits = settings.precision
    value = N(root, digits)
    distances: list[tuple[Fraction, Expr]] = []
    for c in candidates:
        d = N(abs_(N(c, digits) - value), digits)
        if not isinstance(d, (Float, Rational)):
            return None
        distances.append((d.as_fraction(), c))
    distances.sort(key=lambda pair: pair[0])
    closest, expression = distances[0]
    if len(distances) > 1 and distances[1][0] < Fraction(1, 10)**(digits//3):
        return None
    if closest > Fraction(1, 10)**(digits*2//3):
        return None
    return expression


def in_radicals(result: Set) -> Set:
    """The set with its root objects written in radicals when
    :func:`radical_form` allows.

    Examples
    ========

    >>> from sympy_extras.core import Integer
    >>> from sympy_extras.sets import finite_set
    >>> from sympy_extras.polys.roots import in_radicals
    >>> in_radicals(finite_set(Integer(1), Integer(2)))
    {1, 2}
    """
    replacements: dict[Basic, Basic] = {}
    for node in _root_nodes(result):
        form = radical_form(node)
        if form is not None:
            replacements[node] = form
    if not replacements:
        return result
    rebuilt = result._xreplace_evaluated(replacements)
    if not isinstance(rebuilt, Set):
        raise TypeError("expected a set, got %s" % (rebuilt,))
    return rebuilt


def _root_nodes(e: Basic) -> list[Expr]:
    """The root objects of the tree, each once."""
    found: list[Expr] = []
    for node in preorder_traversal(e):
        if isinstance(node, Expr) and polynomial_root(node) is not None and node not in found:
            found.append(node)
    return found

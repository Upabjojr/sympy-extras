"""The SymPy face of the signs of expressions which are not polynomials
(:mod:`sympy_extras.assumptions.analysis`), and the steps of the analysis
which are not ported.

The functions :func:`certified_sign`, :func:`domain_of`, :func:`sign_on`,
:func:`signs_multivariate`, :func:`box_of` and :func:`decide_relational`
take and return SymPy objects, converted by :mod:`sympy_extras.interop`
around the native functions; the facts are those of
:mod:`sympy_extras.assumptions.facts_sympy`.

Three steps of the native analysis stay written on SymPy and are called
from it at run time (the functions ending in ``_of_core``, which take and
return objects of the core): SymPy's ``continuous_domain`` and
``solveset`` for the expressions which are not rational in the variable
(the finite sets of zeros, and whether the zeros are isolated points: a
union of ``ImageSet`` over the integers), and the CAD's ``solution_set``
(:mod:`sympy_extras.polys.cad`, on SymPy's polynomials) for a polynomial
assumption which is not a bound.

Examples
========

>>> from sympy import sin, exp, pi, Interval, S
>>> from sympy.abc import x
>>> from sympy_extras.assumptions.facts_sympy import Facts
>>> from sympy_extras.assumptions.analysis_sympy import sign_on, domain_of, decide_relational
>>> sign_on(sin(x), x, Interval.open(0, pi)), sorted(sign_on(exp(x) - 2, x, S.Reals))
(frozenset({1}), [-1, 0, 1])
>>> domain_of(x, Facts([x**2 < 4, x > 0]))
Interval.open(0, 2)
>>> decide_relational(sin(x) > 0, Facts([(x > 0) & (x < pi)]))
True
"""
from __future__ import annotations

from typing import Optional

from sympy.calculus.util import continuous_domain
from sympy.core.expr import Expr
from sympy.core.relational import Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.fancysets import ImageSet, Integers, Naturals, Naturals0
from sympy.sets.sets import EmptySet, FiniteSet, Intersection, Interval, Set, Union as SetUnion
from sympy.solvers.solveset import solveset

from sympy_extras._timeout import attempt
from sympy_extras._typing import Sign, Truth, as_boolean, as_set
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.relational import Relational as CoreRelational
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.polys.cad import solution_set
from sympy_extras.sets.sets import Interval as CoreInterval
from sympy_extras.sets.sets import Set as CoreSet
from sympy_extras.settings import settings

from . import analysis as native
from .facts_sympy import Facts
from .intervals_sympy import Box, Signs, core_expr, core_set, core_symbol, sympy_expr, sympy_set

__all__ = ['certified_sign', 'certified_compare', 'domain_of', 'sign_on', 'signs_multivariate', 'box_of',
    'decide_relational', 'continuous_on_of_core', 'finite_zeros_of_core', 'has_discrete_zeros_of_core',
    'solution_set_of_core', 'intersection_of_core']


# ---------------------------------------------------------------------------
# the steps of the native analysis which stay on SymPy

def continuous_on_of_core(f: CoreExpr, x: CoreSymbol, domain: CoreSet) -> bool:
    """Whether SymPy's ``continuous_domain`` of ``f`` contains ``domain``."""
    sympy_f, sympy_x, sympy_domain = sympy_expr(f), as_symbol_of(x), sympy_set(domain)
    continuity = attempt(lambda: as_set(continuous_domain(sympy_f, sympy_x, S.Reals)), settings.timeout)
    return continuity is not None and sympy_domain.is_subset(continuity) is True


def as_symbol_of(x: CoreSymbol) -> Symbol:
    """A symbol of the core as a SymPy symbol."""
    s = to_sympy(x)
    if not isinstance(s, Symbol):
        raise TypeError("%s is not a symbol" % (x,))
    return s


def _solveset(f: CoreExpr, x: CoreSymbol, domain: CoreSet) -> Optional[Set]:
    sympy_f, sympy_x, sympy_domain = sympy_expr(f), as_symbol_of(x), sympy_set(domain)
    return attempt(lambda: as_set(solveset(sympy_f, sympy_x, sympy_domain)), settings.timeout)


def finite_zeros_of_core(f: CoreExpr, x: CoreSymbol, domain: CoreSet) -> Optional[list[CoreExpr]]:
    """The zeros of ``f`` in ``domain`` when SymPy's ``solveset`` gives a
    finite set, ``None`` otherwise."""
    zeros = _solveset(f, x, domain)
    if isinstance(zeros, EmptySet):
        return []
    if isinstance(zeros, FiniteSet):
        return [core_expr(z) for z in zeros.args]
    return None


def _is_discrete(zeros: Set) -> bool:
    if isinstance(zeros, (FiniteSet, EmptySet)):
        return True
    if isinstance(zeros, ImageSet):
        return isinstance(zeros.base_set, (Integers, Naturals, Naturals0))
    if isinstance(zeros, SetUnion):
        return all(_is_discrete(as_set(a)) for a in zeros.args)
    if isinstance(zeros, Intersection):
        return any(_is_discrete(as_set(a)) for a in zeros.args)
    return False


def has_discrete_zeros_of_core(f: CoreExpr, x: CoreSymbol, piece: CoreSet) -> bool:
    """Whether SymPy's ``solveset`` describes the zeros of ``f`` on the
    piece as isolated points (finite sets, or images of the integers)."""
    zeros = _solveset(f, x, piece)
    return zeros is not None and _is_discrete(zeros)


def solution_set_of_core(atom: CoreBoolean, x: CoreSymbol) -> Optional[CoreSet]:
    """The solutions of a polynomial relation in ``x`` by the CAD
    (:func:`sympy_extras.polys.cad.solution_set`), ``None`` when it fails
    or gives a set which is not ported."""
    sympy_atom, sympy_x = as_boolean(to_sympy(atom)), as_symbol_of(x)
    found = attempt(lambda: solution_set(sympy_atom, sympy_x), settings.timeout)
    return None if found is None else core_set(found)


def intersection_of_core(a: CoreSet, b: CoreSet) -> CoreSet:
    """The intersection of two sets evaluated by SymPy (whose ends may be
    roots of polynomials, ``CRootOf``, which are not ported)."""
    found = core_set(as_set(Intersection(sympy_set(a), sympy_set(b))))
    if found is None:
        raise TypeError("the intersection of %s and %s is not a set of the core" % (a, b))
    return found


# ---------------------------------------------------------------------------
# the face

def certified_sign(value: Expr) -> Optional[Sign]:
    """The sign of a constant expression, ``None`` when undecided
    (:func:`sympy_extras.assumptions.analysis.certified_sign`)."""
    return native.certified_sign(core_expr(value))


def certified_compare(a: Expr, b: Expr) -> Optional[Sign]:
    """The sign of ``a - b`` for constants, infinities included."""
    return native.certified_compare(core_expr(a), core_expr(b))


def domain_of(x: Symbol, facts: Facts) -> Optional[Set]:
    """The set of real values of ``x`` allowed by the facts which only
    concern ``x`` (:func:`sympy_extras.assumptions.analysis.domain_of`)."""
    result = native.domain_of(core_symbol(x), facts.native)
    return None if result is None else sympy_set(result)


def _pieces(domain: Set) -> Optional[list[Interval]]:
    converted = core_set(domain)
    if converted is None:
        return None
    pieces = native._pieces(converted)
    if pieces is None:
        return None
    result: list[Interval] = []
    for p in pieces:
        s = sympy_set(p)
        if not isinstance(s, Interval):
            raise TypeError("expected an interval, got %s" % (s,))
        result.append(s)
    return result


def _between(p: Expr, q: Expr) -> Optional[Expr]:
    """A rational number strictly between the constants ``p < q``."""
    found = native._between(core_expr(p), core_expr(q))
    return None if found is None else sympy_expr(found)


def _sorted_constants(values: list[Expr]) -> Optional[list[Expr]]:
    """The constants sorted by certified comparisons (``None`` if two of
    them cannot be told apart)."""
    found = native._sorted_constants([core_expr(v) for v in values])
    return None if found is None else [sympy_expr(v) for v in found]


def _endpoint_value(f: Expr, x: Symbol, point: Expr, direction: str) -> Optional[Expr]:
    found = native._endpoint_value(core_expr(f), core_symbol(x), core_expr(point), direction)
    return None if found is None else sympy_expr(found)


def _has_discrete_zeros(f: Expr, x: Symbol, piece: Interval) -> bool:
    converted = core_set(piece)
    if not isinstance(converted, CoreInterval):
        raise TypeError("expected an interval, got %s" % (piece,))
    return native._has_discrete_zeros(core_expr(f), core_symbol(x), converted)


def sign_on(f: Expr, x: Symbol, domain: Set, depth: int = 1) -> Optional[Signs]:
    """The set of signs which ``f`` takes on ``domain`` (a union of
    intervals), or ``None`` when it cannot be established
    (:func:`sympy_extras.assumptions.analysis.sign_on`)."""
    converted = core_set(domain)
    if converted is None:
        return None
    return native.sign_on(core_expr(f), core_symbol(x), converted, depth)


def box_of(symbols: list[Symbol], facts: Facts) -> Optional[tuple[Box, dict[Symbol, tuple[bool, bool]]]]:
    """The box of the values of the symbols allowed by the facts
    (:func:`sympy_extras.assumptions.analysis.box_of`)."""
    found = native.box_of([core_symbol(s) for s in symbols], facts.native)
    if found is None:
        return None
    box, openness = found
    return ({as_symbol_of(v): (sympy_expr(lo), sympy_expr(hi)) for v, (lo, hi) in box.items()},
            {as_symbol_of(v): ends for v, ends in openness.items()})


def signs_multivariate(f: Expr, symbols: list[Symbol], facts: Facts) -> Optional[Signs]:
    """The signs of an expression of several real variables on the box of
    their assumptions (:func:`sympy_extras.assumptions.analysis.signs_multivariate`)."""
    return native.signs_multivariate(core_expr(f), [core_symbol(s) for s in symbols], facts.native)


def decide_relational(atom: Relational, facts: Facts) -> Truth:
    """Truth of a relation between expressions of real variables under
    the facts, by the sign analysis of the difference of its sides
    (:func:`sympy_extras.assumptions.analysis.decide_relational`).

    Examples
    ========

    >>> from sympy import sin, exp, pi, Abs
    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions.ask import ask
    >>> from sympy_extras.assumptions.refine import refine
    >>> ask(sin(x) > 0, (x > 0) & (x < pi))
    True
    >>> ask(exp(x) > 2, x > 1)
    True
    >>> refine(Abs(sin(x)), (x > 0) & (x < pi))
    sin(x)
    >>> ask(x*exp(x) > 1, x > 1)
    True
    >>> from sympy.abc import y
    >>> ask(exp(x) + y > 1, (x > 0) & (y > 0))
    True
    >>> ask(sin(x*y) > 0, (x > 0) & (x < 1) & (y > 0) & (y < 3))
    True
    """
    converted = from_sympy(atom)
    if not isinstance(converted, CoreRelational):
        return None
    return native.decide_relational(converted, facts.native)

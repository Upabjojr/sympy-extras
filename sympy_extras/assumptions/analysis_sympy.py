"""The SymPy face of the signs of expressions which are not polynomials
(:mod:`sympy_extras.assumptions.analysis`), and the steps of the analysis
which are not ported.

The functions :func:`certified_sign`, :func:`domain_of`, :func:`sign_on`,
:func:`signs_multivariate`, :func:`box_of` and :func:`decide_relational`
take and return SymPy objects, converted by :mod:`sympy_extras.interop`
around the native functions; the facts are those of
:mod:`sympy_extras.assumptions.facts_sympy`.

No step of the native analysis is written on SymPy: the continuity, the
zeros of the expressions which are not rational and the intersections of
intervals with roots of polynomials as ends are native.

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

from sympy.core.expr import Expr
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol
from sympy.sets.sets import Interval, Set

from sympy_extras._typing_sympy import Sign, Truth
from sympy_extras.core.relational import Relational as CoreRelational
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.sets.sets import Interval as CoreInterval

from . import analysis as native
from .facts_sympy import Facts
from .intervals_sympy import Box, Signs, core_expr, core_set, core_symbol, sympy_expr, sympy_set

__all__ = ['certified_sign', 'certified_compare', 'domain_of', 'sign_on', 'signs_multivariate', 'box_of',
    'decide_relational']


def as_symbol_of(x: CoreSymbol) -> Symbol:
    """A symbol of the core as a SymPy symbol."""
    s = to_sympy(x)
    if not isinstance(s, Symbol):
        raise TypeError("%s is not a symbol" % (x,))
    return s


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
    >>> from sympy_extras.assumptions.ask_sympy import ask
    >>> from sympy_extras.assumptions.refine_sympy import refine
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

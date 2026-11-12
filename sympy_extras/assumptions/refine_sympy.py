"""The SymPy face of refining and simplifying expressions under
assumptions (:mod:`sympy_extras.assumptions.refine`), and the steps which
are not ported.

:func:`refine` and :func:`simplify` take and return SymPy objects,
converted by :mod:`sympy_extras.interop` around the native functions;
:class:`_Refiner` is the face of the native refiner (its dummies are
SymPy dummies carrying the assumptions).

One step of the native refinement stays written on SymPy and is called
from it at run time (:func:`simplify_boolean_of_core`, which takes and
returns objects of the core): SymPy's ``simplify`` of a compound Boolean
formula (``simplify_univariate`` and the patterns of the relations of
``And`` and ``Or``, not ported; a single relation is simplified by the
native port). The proofs of the structure theorem are native
(:class:`sympy_extras.simplify.structure.ElementaryTower`).

Examples
========

>>> from sympy import atan, Rational, sqrt, asin, log, symbols
>>> from sympy_extras.assumptions.refine_sympy import simplify
>>> x, y = symbols('x y')
>>> simplify(4*atan(Rational(1, 5)) - atan(Rational(1, 239)))
pi/4
>>> simplify(asin(x) - atan(x/sqrt(1 - x**2)), (x > -1) & (x < 1))
0
>>> simplify(atan(x) + atan(1/x), x > 0), simplify(atan(x) + atan(1/x))
(pi/2, atan(1/x) + atan(x))
>>> simplify(log(x*y) - log(y), x > 0)
log(x)
"""
from __future__ import annotations

from typing import Final, Optional, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.sets.sets import Set

from sympy_extras._typing_sympy import Truth, as_symbol
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.interop import from_sympy, to_sympy

from . import refine as native
from .ask_sympy import Assumptions, _core_expr, core_assumptions, core_boolean, core_domain, \
    sympy_boolean
from .facts_sympy import Facts

__all__ = ['refine', 'simplify', 'simplify_boolean_of_core']


def _core(e: Union[Basic, int, bool]) -> CoreBasic:
    return from_sympy(sympify(e))


# ---------------------------------------------------------------------------
# the steps of the native refinement which stay on SymPy

def simplify_boolean_of_core(formula: CoreBoolean, ratio: float, rational: Optional[bool], inverse: bool,
                             doit: bool) -> CoreBoolean:
    """SymPy's ``simplify`` of a compound Boolean formula
    (``simplify_univariate`` and the patterns of the relations of ``And``
    and ``Or``, which are not ported)."""
    from sympy.simplify.simplify import simplify as _sympy_simplify
    return core_boolean(sympify(_sympy_simplify(sympy_boolean(formula), ratio=ratio, rational=rational,
                                                inverse=inverse, doit=doit)))


# ---------------------------------------------------------------------------
# the face

class _Refiner:
    """The face of :class:`sympy_extras.assumptions.refine._Refiner`:
    SymPy objects in and out, the facts of
    :mod:`sympy_extras.assumptions.facts_sympy`."""

    def __init__(self, facts: Facts) -> None:
        self.facts: Final[Facts] = facts
        self._native: Final[native._Refiner] = native._Refiner(facts.native)

    def holds(self, statement: Union[Boolean, bool]) -> Truth:
        """Three-valued truth of a statement under the facts."""
        return self._native.holds(core_boolean(statement))

    def dummy_for(self, f: Expr) -> Optional[Symbol]:
        """A dummy carrying the properties of ``f`` which follow from the
        facts, or ``None`` when there are none."""
        found = self._native.dummy_for(_core_expr(f))
        return None if found is None else as_symbol(to_sympy(found))

    def abstract(self, expr: Basic) -> tuple[Basic, dict[Basic, Basic], dict[Basic, Basic]]:
        """``expr`` with its symbols and relevant subexpressions replaced by
        dummies carrying the assumptions, and the two substitutions which
        undo it (subexpressions first, then symbols)."""
        abstracted, back_sub, back_symbols = self._native.abstract(_core(expr))
        return (to_sympy(abstracted), {to_sympy(a): to_sympy(b) for a, b in back_sub.items()},
                {to_sympy(a): to_sympy(b) for a, b in back_symbols.items()})

    def refine(self, expr: Basic) -> Basic:
        return to_sympy(self._native.refine(_core(expr)))


def _refine_boolean(formula: Boolean, facts: Facts) -> Boolean:
    """Replace the atoms of a Boolean formula which are decided by the
    facts (:func:`sympy_extras.assumptions.refine._refine_boolean`)."""
    return sympy_boolean(native._refine_boolean(core_boolean(formula), facts.native))


def refine(expr: Union[Expr, Boolean, bool], assumptions: Assumptions = None,
           domain: Optional[Set] = None) -> Basic:
    """Refine an expression using assumptions
    (:func:`sympy_extras.assumptions.refine.refine` on SymPy objects), the
    counterpart of Mathematica's ``Refine[expr, assum]``.

    Parameters
    ==========

    expr : Expr or Boolean
        The expression to refine.
    assumptions : Boolean or list of Booleans, optional
        Assumptions written as relations, memberships in sets or predicates,
        see :mod:`sympy_extras.assumptions.facts_sympy`. The global
        assumptions are used as well.
    domain : Set, optional
        A named SymPy set all the variables are assumed to belong to.

    Examples
    ========

    >>> from sympy import Abs, sqrt, sign, Max, floor, Piecewise, S, re, im, log, atan2
    >>> from sympy.abc import x, y, n
    >>> from sympy_extras.assumptions.refine_sympy import refine
    >>> from sympy_extras.assumptions.facts_sympy import element
    >>> refine(Abs(x - 1) + sqrt(x**2), x > 2)
    2*x - 1
    >>> refine(sqrt(x**2 - 2*x + 1), x > 1)
    x - 1
    >>> refine(log(x**2) + sqrt(x**2*y**2), (x > 0) & (y > 0))
    x*y + 2*log(x)
    >>> refine(sign(x*y), (x > 0) & (y < 0))
    -1
    >>> refine(Max(x, x**2), (x > 0) & (x < 1))
    x
    >>> refine(floor(n + 1), element(n, S.Integers))
    n + 1
    >>> refine(atan2(y, x - 1), x > 1)
    atan(y/(x - 1))
    >>> refine(Piecewise((1, x**2 + y**2 < 1), (2, x > 0), (3, True)), (x > 1) & (y > 0))
    2
    >>> refine(re(x) + im(y), element(x, S.Reals) & element(y, S.Reals))
    x
    >>> refine((x > 0) & (y > 0), x > 1)
    y > 0
    >>> refine(x*y > 0, (x > 0) & (y > 0))
    True
    """
    if expr is True or expr is False:
        return sympy_boolean(native.refine(expr))
    return to_sympy(native.refine(_core(expr), core_assumptions(assumptions), core_domain(domain)))


def simplify(expr: Union[Expr, Boolean, bool], assumptions: Assumptions = None,
             domain: Optional[Set] = None, ratio: float = 1.7, rational: Optional[bool] = False,
             inverse: bool = False, doit: bool = True) -> Basic:
    """Simplify an expression using assumptions
    (:func:`sympy_extras.assumptions.refine.simplify` on SymPy objects),
    the counterpart of Mathematica's ``Simplify[expr, assum]``.

    The expression is refined with :func:`refine`, then simplified (with
    ``ratio``, ``rational``, ``inverse`` and ``doit`` as in
    :func:`sympy.simplify`) while its symbols and the relevant
    subexpressions carry the assumptions, and refined again. A Boolean
    formula which is polynomial in real variables is minimised by
    quantifier elimination
    (:func:`~sympy_extras.assumptions.resolve_sympy.resolve`).

    Examples
    ========

    >>> from sympy import Abs, sqrt, cos, sin, log, exp, Eq, S, pi
    >>> from sympy.abc import x, y, n
    >>> from sympy_extras.assumptions.refine_sympy import simplify
    >>> from sympy_extras.assumptions.facts_sympy import element
    >>> simplify(sqrt(x**2) + Abs(x)*sin(x)**2 + Abs(x)*cos(x)**2, x < 0)
    -2*x
    >>> simplify((x**2 - 1)/(x - 1), x > 1)
    x + 1
    >>> simplify(log(x) + log(y), (x > 0) & (y > 0))
    log(x*y)
    >>> simplify(log(exp(x)), element(x, S.Reals))
    x
    >>> simplify(sin(n*pi) + cos(n*pi), element(n, S.Integers))
    (-1)**n
    >>> simplify(Eq(x**2, 1), x > 0)
    Eq(x, 1)
    >>> simplify((x > 1) | (x**2 > 1), domain=S.Reals)
    (x > 1) | (x < -1)
    """
    if expr is True or expr is False:
        return sympy_boolean(native.simplify(expr))
    return to_sympy(native.simplify(_core(expr), core_assumptions(assumptions), core_domain(domain),
                                    ratio=ratio, rational=rational, inverse=inverse, doit=doit))

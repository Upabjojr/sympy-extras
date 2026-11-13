"""The SymPy face of the truth values of statements under assumptions
(:mod:`sympy_extras.assumptions.ask`).

:func:`ask` takes SymPy objects (relations, memberships in SymPy's sets,
predicates of SymPy's ``Q``, the quantifiers of
:mod:`sympy_extras.assumptions.quantifiers_sympy`), converted by
:mod:`sympy_extras.interop` around the native function; the facts are
those of :mod:`sympy_extras.assumptions.facts_sympy`.

Examples
========

>>> from sympy import Eq, atan, pi, Rational, symbols, log
>>> from sympy_extras.assumptions.ask_sympy import ask
>>> x = symbols('x')
>>> ask(Eq(4*atan(Rational(1, 5)) - atan(Rational(1, 239)), pi/4))
True
>>> ask(Eq(log(x**2), 2*log(x)), x > 0), ask(Eq(log(x**2), 2*log(x)))
(True, None)
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.sets.sets import Set

from sympy_extras._typing_sympy import QuantifierPrefix, Truth, as_boolean, as_symbol
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.logic import BooleanLike as CoreBooleanLike
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.sets.sets import Set as CoreSet

from . import ask as native
from .context_sympy import global_assumptions
from .facts_sympy import Facts
from .quantifiers import Prefix

__all__ = ['ask', 'Assumptions', 'core_boolean', 'core_assumptions', 'core_domain', 'sympy_boolean']


Assumptions = Union[None, Boolean, bool, Iterable[Union[Boolean, bool]]]


# ---------------------------------------------------------------------------
# the conversions at the boundary

def core_boolean(b: Union[Basic, bool]) -> CoreBoolean:
    """A SymPy Boolean (or a Python bool) as a Boolean of the core."""
    converted = from_sympy(sympify(b))
    if not isinstance(converted, CoreBoolean):
        raise TypeError("expected a Boolean, got %s" % (b,))
    return converted


def sympy_boolean(b: CoreBasic) -> Boolean:
    """A Boolean of the core as a SymPy Boolean."""
    return as_boolean(to_sympy(b))


def _core_expr(e: Basic) -> CoreExpr:
    converted = from_sympy(e)
    if not isinstance(converted, CoreExpr):
        raise TypeError("expected an expression, got %s" % (e,))
    return converted


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy_expr(e: CoreBasic) -> Expr:
    converted = to_sympy(e)
    if not isinstance(converted, Expr):
        raise TypeError("expected an expression, got %s" % (converted,))
    return converted


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    return as_symbol(to_sympy(s))


def core_assumptions(assumptions: Assumptions) -> native.Assumptions:
    """Assumptions given with SymPy objects as assumptions of the core."""
    if assumptions is None:
        return None
    if isinstance(assumptions, (Basic, bool)):
        return core_boolean(assumptions)
    items: list[CoreBooleanLike] = [core_boolean(a) for a in assumptions]
    return items


def core_domain(domain: Optional[Set]) -> Optional[CoreSet]:
    """A named SymPy set as a set of the core (``TypeError`` for a set
    which is not ported)."""
    if domain is None:
        return None
    d = sympify(domain)
    if not isinstance(d, Set):
        raise TypeError("domain must be a SymPy set, got %s" % (domain,))
    converted = from_sympy(d)
    if not isinstance(converted, CoreSet):
        raise TypeError("domain must be a set ported to sympy_extras.sets, got %s" % (domain,))
    return converted


def core_prefix(prefix: QuantifierPrefix) -> Prefix:
    return [(kind, _core_symbol(v)) for kind, v in prefix]


def sympy_prefix(prefix: Prefix) -> QuantifierPrefix:
    return [(kind, _sympy_symbol(v)) for kind, v in prefix]


# ---------------------------------------------------------------------------
# the face

def _facts(assumptions: Assumptions, domain: Optional[Set], symbols: Iterable[Basic]) -> Facts:
    """The facts from the global assumptions and the explicit ones."""
    items: list[Union[Boolean, bool]] = list(global_assumptions)
    if isinstance(assumptions, (Boolean, bool)):
        items.append(assumptions)
    elif assumptions is not None:
        items.extend(assumptions)
    return Facts(items, domain, [s for s in symbols if isinstance(s, Symbol)])


def _evaluate(formula: Boolean, facts: Facts) -> Truth:
    """Three-valued evaluation of a normalized formula
    (:func:`sympy_extras.assumptions.ask._evaluate`)."""
    return native._evaluate(core_boolean(formula), facts.native)


def _evaluate_atom(atom: Boolean, facts: Facts) -> Truth:
    return native._evaluate_atom(core_boolean(atom), facts.native)


def _linear_ask(poly: Boolean, premise: Boolean, gens: list[Symbol]) -> Truth:
    """A formula linear in every variable decided by virtual substitution
    (:func:`sympy_extras.assumptions.ask._linear_ask`)."""
    return native._linear_ask(core_boolean(poly), core_boolean(premise), [_core_symbol(g) for g in gens])


def _clearly_not_zero(difference: Expr, facts: Facts) -> bool:
    """Whether ``difference`` has a value away from zero at one point of
    the region of the facts (:func:`sympy_extras.assumptions.ask._clearly_not_zero`)."""
    return native._clearly_not_zero(_core_expr(difference), facts.native)


def ask(query: Union[Boolean, bool], assumptions: Assumptions = None, domain: Optional[Set] = None) -> Truth:
    """Truth value of a statement under assumptions
    (:func:`sympy_extras.assumptions.ask.ask` on SymPy objects).

    Parameters
    ==========

    query : Boolean
        A relation (``x > 0``, ``Eq(x, y)``), a membership
        (``element(x, S.Integers)``), a predicate (``Q.positive(x)``), a
        Boolean combination of those or a quantified formula
        (``ForAll(x, ...)``, ``Exists(x, ...)``).
    assumptions : Boolean or list of Booleans, optional
        Assumptions in the same form. The assumptions in
        :data:`~sympy_extras.assumptions.context_sympy.global_assumptions`
        are always used as well.
    domain : Set, optional
        A named SymPy set (``S.Reals``, ``S.Integers``, ...) all the
        variables are assumed to belong to.

    Returns
    =======

    ``True`` if the statement follows from the assumptions, ``False`` if
    its negation does and ``None`` if neither can be established.

    The assumptions and the query are first translated to predicates and
    answered by the core's ``ask`` (which asks :func:`sympy.ask` what it
    cannot answer); what remains undecided and is a Boolean combination of
    polynomial relations between real variables is decided exactly by
    cylindrical algebraic decomposition (see :mod:`sympy_extras.polys.cad`).
    Note that an inequality between expressions states that both sides
    are real.

    Examples
    ========

    >>> from sympy import S, Eq, Q
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions.ask_sympy import ask
    >>> from sympy_extras.assumptions.facts_sympy import element
    >>> from sympy_extras.assumptions.quantifiers_sympy import ForAll
    >>> ask(x**3 > 0, x > 0)
    True
    >>> ask(x**2 - 2*x + 1 >= 0, x > 0)
    True
    >>> ask(x > 1, x > 2)
    True
    >>> ask(x > 3, x > 2) is None
    True
    >>> ask(x*y > 0, (x > 0) & (y < 0))
    False
    >>> ask(element(x + 1, S.Integers), element(x, S.Integers))
    True
    >>> ask(x**2 >= 0) is None
    True
    >>> ask(x**2 >= 0, domain=S.Reals)
    True
    >>> ask(ForAll(x, x**2 + 2*x*y + y**2 >= 0), domain=S.Reals)
    True
    >>> ask(Q.positive(x), Eq(x, 3))
    True
    """
    return native.ask(core_boolean(query), core_assumptions(assumptions), core_domain(domain))

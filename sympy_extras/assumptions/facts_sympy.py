"""The SymPy face of the assumptions written as mathematical statements
(:mod:`sympy_extras.assumptions.facts`).

:mod:`~sympy_extras.assumptions.facts` is written on the core. Code on
SymPy objects uses this module instead: :func:`element`, :func:`normalize`,
:func:`conjuncts`, :func:`to_predicates`, :func:`to_polynomial` and
:class:`Facts` take and return SymPy objects (the memberships in SymPy's
sets, the predicates of SymPy's ``Q``), converted by
:mod:`sympy_extras.interop` at the boundary around the native functions.
:func:`predicates_consistent` stays written on SymPy: it asks the SAT
solver of SymPy's assumptions together with SymPy's known facts about the
predicates (``get_all_relevant_facts``), which are not ported.

Examples
========

>>> from sympy import S
>>> from sympy.abc import x, y, z
>>> from sympy_extras.assumptions.facts_sympy import Facts, element
>>> f = Facts([x > 2, element(y, S.Integers), z**3 > x*y])
>>> f.predicates
Q.integer(y) & Q.positive(x) & Q.real(x) & Q.real(y) & Q.real(z) & Q.positive(x - 2) & Q.positive(-x*y + z**3)
>>> sorted(f.real, key=str), sorted(f.integer, key=str)
([x, y, z], [y])
>>> f.polynomial
(x > 2) & (z**3 > x*y)
"""
from __future__ import annotations

from typing import Final, Iterable, Optional, Union

from sympy.assumptions.cnf import CNF
from sympy.assumptions.satask import get_all_relevant_facts
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean, And, Or, Not, Implies, Equivalent, Xor, ITE
from sympy.logic.inference import satisfiable as _sympy_satisfiable
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Naturals, Naturals0, Integers, Rationals, Reals
from sympy.sets.sets import Set

from sympy_extras._typing_sympy import as_boolean, as_symbol
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.sets.sets import Set as CoreSet

from . import facts as native

__all__ = ['element', 'normalize', 'conjuncts', 'to_predicates', 'to_polynomial', 'predicates_consistent',
           'Facts']

# named sets whose members are real numbers
_REAL_SETS = (Naturals, Naturals0, Integers, Rationals, Reals)

_COMPOUND = (And, Or, Not, Implies, Equivalent, Xor, ITE)


# ---------------------------------------------------------------------------
# the conversions at the boundary

def _core(b: Union[Basic, bool]) -> CoreBoolean:
    converted = from_sympy(sympify(b))
    if not isinstance(converted, CoreBoolean):
        raise TypeError("expected a Boolean, got %s" % (b,))
    return converted


def _core_expr(e: Basic) -> CoreExpr:
    converted = from_sympy(e)
    if not isinstance(converted, CoreExpr):
        raise TypeError("expected an expression, got %s" % (e,))
    return converted


def _core_symbols(symbols: Iterable[Basic]) -> set[CoreSymbol]:
    result: set[CoreSymbol] = set()
    for s in symbols:
        converted = from_sympy(s)
        if not isinstance(converted, CoreSymbol):
            raise TypeError("%s is not a symbol" % (s,))
        result.add(converted)
    return result


def _sympy(b: CoreBasic) -> Boolean:
    return as_boolean(to_sympy(b))


def _sympy_optional(b: Optional[CoreBoolean]) -> Optional[Boolean]:
    return None if b is None else _sympy(b)


def _sympy_symbols(symbols: Iterable[CoreSymbol]) -> set[Symbol]:
    return {as_symbol(to_sympy(s)) for s in symbols}


# ---------------------------------------------------------------------------
# Building blocks

def element(x: Union[Expr, int], domain: Set) -> Boolean:
    """Membership of ``x`` in the SymPy set ``domain``, the counterpart of
    Mathematica's ``Element[x, dom]``.

    It is :class:`~sympy.sets.contains.Contains` and evaluates when the
    answer is known.

    Examples
    ========

    >>> from sympy import S, Interval
    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions.facts_sympy import element
    >>> element(x, S.Integers)
    Contains(x, Integers)
    >>> element(2, S.Integers)
    True
    >>> element(x, Interval(0, 1))
    Contains(x, Interval(0, 1))
    """
    member, s = sympify(x), sympify(domain)
    if not isinstance(s, Set):
        raise TypeError("expected a SymPy set, got %s" % (domain,))
    native_domain = from_sympy(s)
    if not isinstance(native_domain, CoreSet):
        # a set which is not ported: SymPy decides
        return as_boolean(Contains(member, s))
    return _sympy(native.element(_core_expr(member), native_domain))


def normalize(formula: Union[Boolean, bool]) -> Boolean:
    """Normal form of an assumption or query
    (:func:`sympy_extras.assumptions.facts.normalize` on SymPy objects).

    Memberships in intervals, finite sets and combinations of sets are
    rewritten as relations and Boolean combinations; memberships in the
    named sets (``S.Integers``, ``S.Reals``, ...) and predicates of
    :mod:`sympy.assumptions` are kept as atoms. ``True``/``False`` and
    Python Booleans become ``S.true``/``S.false``.

    Examples
    ========

    >>> from sympy import S, Interval, FiniteSet, Union, oo
    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions.facts_sympy import element, normalize
    >>> normalize(element(x, Interval(0, 1)))
    (x >= 0) & (x <= 1)
    >>> normalize(element(x, Union(FiniteSet(1, 2), Interval.open(3, 4))))
    Eq(x, 1) | Eq(x, 2) | ((x > 3) & (x < 4))
    >>> normalize(element(x, Interval(0, oo)))
    x >= 0
    >>> normalize(element(x, S.Integers) & (x > 0))
    Contains(x, Integers) & (x > 0)
    """
    return _sympy(native.normalize(_core(formula)))


def conjuncts(formula: Boolean) -> list[Boolean]:
    """The top level conjuncts of a normalized formula."""
    return [_sympy(c) for c in native.conjuncts(_core(formula))]


def _is_atom(formula: Boolean) -> bool:
    return not isinstance(formula, _COMPOUND)


def _predicate_of_atom(atom: Boolean) -> Optional[Boolean]:
    """Predicate of :mod:`sympy.assumptions` equivalent to an atom, or
    ``None`` if there is none."""
    return _sympy_optional(native._predicate_of_atom(_core(atom)))


def to_predicates(formula: Boolean) -> Optional[Boolean]:
    """Translate a normalized formula to an equivalent Boolean combination of
    predicates of :mod:`sympy.assumptions`, or ``None`` if some atom has no
    equivalent.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions.facts_sympy import element, to_predicates
    >>> to_predicates((x > 0) | element(y, S.Integers))
    Q.integer(y) | Q.positive(x)
    >>> to_predicates(x - y >= 2)
    Q.nonnegative(x - y - 2)
    """
    return _sympy_optional(native.to_predicates(_core(formula)))


def _consequences(atom: Boolean) -> list[Boolean]:
    """Predicates implied by a top level atom beyond its equivalent
    predicate: the reality of the variables of an inequality and simple
    sign conditions."""
    return [_sympy(p) for p in native._consequences(_core(atom))]


def _is_polynomial(expr: Expr, symbols: set[Symbol]) -> bool:
    """Whether ``expr`` is a polynomial with rational coefficients in
    ``symbols`` (which must contain its free symbols)."""
    return native._is_polynomial(_core_expr(expr), _core_symbols(symbols))


def to_polynomial(formula: Boolean, real: set[Symbol]) -> Optional[Boolean]:
    """Translate a normalized formula to an equivalent Boolean combination
    of polynomial relations with rational coefficients between the
    variables ``real``, which are known to be real, or ``None`` if some
    atom cannot be translated.

    Examples
    ========

    >>> from sympy import Q, S
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions.facts_sympy import element, to_polynomial
    >>> to_polynomial(Q.positive(x) & (x*y < 1) & element(y, S.Reals), {x, y})
    (x > 0) & (x*y < 1)
    >>> to_polynomial(x*y < 1, {x}) is None
    True
    """
    return _sympy_optional(native.to_polynomial(_core(formula), _core_symbols(real)))


# ---------------------------------------------------------------------------
# Consistency of predicates

def predicates_consistent(predicates: Boolean) -> bool:
    """Whether a Boolean combination of predicates of
    :mod:`sympy.assumptions` is consistent with the known facts about the
    predicates (``Q.positive(x) & Q.negative(x)`` is not).

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions.facts_sympy import predicates_consistent
    >>> predicates_consistent(Q.positive(x) & Q.integer(x))
    True
    >>> predicates_consistent(Q.positive(x) & Q.negative(x))
    False
    """
    cnf = CNF.from_prop(predicates)
    # the relevant known facts are gathered from the proposition, so the
    # predicates are passed as such too
    sat = get_all_relevant_facts(cnf, cnf, CNF())
    sat.add_from_cnf(cnf)
    return bool(_sympy_satisfiable(sat))


# ---------------------------------------------------------------------------
# Facts

class Facts:
    """The assumptions in force, translated for the backends: the SymPy
    face of :class:`sympy_extras.assumptions.facts.Facts` (SymPy objects in
    and out).

    Parameters
    ==========

    assumptions : Boolean or iterable of Booleans, optional
        Assumptions written as mathematical statements, see the module
        documentation of :mod:`sympy_extras.assumptions.facts`.
    domain : Set, optional
        A named SymPy set (``S.Reals``, ``S.Integers``, ...) all the
        variables in ``symbols`` are assumed to belong to, like the domain
        argument of Mathematica's ``Reduce`` or ``Resolve``.
    symbols : iterable of Symbol, optional
        The variables of the expression the assumptions are used for; with
        ``domain`` they are all assumed to be in the domain.

    Attributes
    ==========

    formula : Boolean
        The conjunction of all assumptions in normal form.
    predicates : Boolean
        Consequences of the assumptions as predicates of
        :mod:`sympy.assumptions`.
    real : set of Symbol
        The variables known to be real.
    integer : set of Symbol
        The variables known to be integers.
    polynomial : Boolean
        The part of the assumptions which is a Boolean combination of
        polynomial relations with rational coefficients between real
        variables, as used by the CAD.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.assumptions.facts_sympy import Facts, element
    >>> f = Facts([x > 2, element(y, S.Integers), z**3 > x*y])
    >>> f.predicates
    Q.integer(y) & Q.positive(x) & Q.real(x) & Q.real(y) & Q.real(z) & Q.positive(x - 2) & Q.positive(-x*y + z**3)
    >>> sorted(f.real, key=str), sorted(f.integer, key=str)
    ([x, y, z], [y])
    >>> f.polynomial
    (x > 2) & (z**3 > x*y)
    """

    def __init__(self, assumptions: Union[None, Boolean, bool, Iterable[Union[Boolean, bool]]] = None,
                 domain: Optional[Set] = None, symbols: Iterable[Symbol] = ()) -> None:
        items: list[CoreBoolean] = []
        if assumptions is not None:
            if isinstance(assumptions, (Basic, bool)):
                items.append(_core(assumptions))
            else:
                items.extend(_core(a) for a in assumptions)
        sympy_domain: Optional[Set] = None
        native_domain: Optional[CoreSet] = None
        if domain is not None:
            d = sympify(domain)
            if not isinstance(d, Set):
                raise TypeError("domain must be a SymPy set, got %s" % (domain,))
            converted = from_sympy(d)
            if not isinstance(converted, CoreSet):
                raise TypeError("domain must be a set ported to sympy_extras.sets, got %s" % (domain,))
            sympy_domain, native_domain = d, converted
        self.domain: Final[Optional[Set]] = sympy_domain
        self._native: Final[native.Facts] = native.Facts(items, native_domain, _core_symbols(symbols))
        self.formula: Final[Boolean] = _sympy(self._native.formula)
        self.conjuncts: Final[list[Boolean]] = [_sympy(c) for c in self._native.conjuncts]
        self.predicates: Final[Boolean] = _sympy(self._native.predicates)
        self.real: Final[set[Symbol]] = _sympy_symbols(self._native.real)
        self.integer: Final[set[Symbol]] = _sympy_symbols(self._native.integer)
        self.polynomial: Final[Boolean] = _sympy(self._native.polynomial)

    def __repr__(self) -> str:
        return "Facts(%s)" % (self.formula,)

    @property
    def native(self) -> native.Facts:
        """The facts of the core these are the face of."""
        return self._native

    def is_polynomial(self, formula: Boolean) -> bool:
        """Whether the normalized ``formula`` can be handled by the CAD
        given the variables known to be real."""
        return self._native.is_polynomial(_core(formula))

    def with_reals(self, symbols: Iterable[Symbol]) -> Facts:
        """The same facts with ``symbols`` also assumed real."""
        extra = [s for s in symbols if s not in self.real]
        if not extra:
            return self
        return Facts(self.conjuncts + [as_boolean(Contains(s, S.Reals)) for s in extra])

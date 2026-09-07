"""Assumptions written as ordinary mathematical statements.

In this module an assumption is a Boolean combination of

* relations between expressions, ``x > 0``, ``x**2 + y**2 <= 1``,
  ``Eq(x, y)``, ``Ne(x, 0)``;
* memberships in SymPy sets, ``element(x, S.Integers)`` (that is
  ``Contains(x, S.Integers)``), ``element(x, Interval(0, 1))``,
  ``element(n, S.Naturals)``;
* for compatibility, predicates of SymPy's assumptions system such as
  ``Q.positive(x)``.

The pieces are combined with ``&``, ``|``, ``~`` or ``And``, ``Or``, ``Not``,
``Implies``, ``Equivalent``, ``Xor``.

Following the convention of Mathematica, an inequality between two
expressions is only true when both sides are real: ``x > 0`` says that ``x``
is a positive *real* number. Equations and ``Ne`` do not carry this
implication.

The :class:`Facts` class translates such assumptions into the two forms
used by the backends: predicates of :mod:`sympy.assumptions` (for
:func:`sympy.ask` and :func:`sympy.refine`) and Boolean combinations of
polynomial relations between real variables (for the cylindrical algebraic
decomposition of :mod:`sympy_extras.polys.cad`).
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Union

from sympy.assumptions import Q, ask as _sympy_ask
from sympy.assumptions.assume import AppliedPredicate
from sympy.assumptions.cnf import CNF
from sympy.assumptions.satask import get_all_relevant_facts
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Relational, Eq, Ne, Lt, Le, Gt, Ge
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or,
    Not, Implies, Equivalent, Xor, ITE, true, false)
from sympy.logic.inference import satisfiable as _sympy_satisfiable
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly

from sympy_extras._typing import as_boolean, as_expr, as_set, free_symbols
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Naturals, Naturals0, Integers, Rationals, Reals, Complexes
from sympy.sets.sets import (Set, Interval, FiniteSet, Union as SetUnion,
    Intersection, Complement, EmptySet, UniversalSet)

__all__ = ['element', 'normalize', 'conjuncts', 'to_predicates',
    'to_polynomial', 'Facts']


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
    >>> from sympy_extras.assumptions import element
    >>> element(x, S.Integers)
    Contains(x, Integers)
    >>> element(2, S.Integers)
    True
    >>> element(x, Interval(0, 1))
    Contains(x, Interval(0, 1))
    """
    return Contains(sympify(x), sympify(domain))


_NAMED_SETS = (Naturals, Naturals0, Integers, Rationals, Reals, Complexes)

# predicate of sympy.assumptions equivalent to the membership in a named set
# (Naturals0 is a subclass of Naturals, so it must come first)
_SET_PREDICATES: list[tuple[type, Callable[[Expr], Boolean]]] = [
    (Naturals0, lambda x: And(Q.integer(x), Q.nonnegative(x))),
    (Naturals, lambda x: And(Q.integer(x), Q.positive(x))),
    (Integers, Q.integer),
    (Rationals, Q.rational),
    (Reals, Q.real),
    (Complexes, Q.complex),
]

# named sets whose members are real numbers
_REAL_SETS = (Naturals, Naturals0, Integers, Rationals, Reals)

# predicates equivalent to a sign condition on a real number
_SIGN_PREDICATES: dict[object, type] = {
    Q.positive: Gt, Q.negative: Lt, Q.nonnegative: Ge, Q.nonpositive: Le,
    Q.zero: Eq, Q.nonzero: Ne,
}

_COMPOUND = (And, Or, Not, Implies, Equivalent, Xor, ITE)


def _is_atom(formula: Boolean) -> bool:
    return not isinstance(formula, _COMPOUND)


def normalize(formula: Union[Boolean, bool]) -> Boolean:
    """Normal form of an assumption or query.

    Memberships in intervals, finite sets and combinations of sets are
    rewritten as relations and Boolean combinations; memberships in the
    named sets (``S.Integers``, ``S.Reals``, ...) and predicates of
    :mod:`sympy.assumptions` are kept as atoms. ``True``/``False`` and
    Python Booleans become ``S.true``/``S.false``.

    Examples
    ========

    >>> from sympy import S, Interval, FiniteSet, Union, oo
    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions import element
    >>> from sympy_extras.assumptions.facts import normalize
    >>> normalize(element(x, Interval(0, 1)))
    (x >= 0) & (x <= 1)
    >>> normalize(element(x, Union(FiniteSet(1, 2), Interval.open(3, 4))))
    Eq(x, 1) | Eq(x, 2) | ((x > 3) & (x < 4))
    >>> normalize(element(x, Interval(0, oo)))
    x >= 0
    >>> normalize(element(x, S.Integers) & (x > 0))
    Contains(x, Integers) & (x > 0)
    """
    formula_ = as_boolean(formula)
    if isinstance(formula_, (BooleanTrue, BooleanFalse)):
        return formula_
    if isinstance(formula_, Contains):
        return _normalize_contains(as_expr(formula_.args[0]), as_set(formula_.args[1]))
    if isinstance(formula_, ITE):
        c, a, b = [normalize(as_boolean(arg)) for arg in formula_.args]
        return Or(And(c, a), And(Not(c), b))
    if isinstance(formula_, _COMPOUND):
        return as_boolean(formula_.func(*[normalize(as_boolean(arg)) for arg in formula_.args]))
    if isinstance(formula_, Relational):
        canonical = as_boolean(formula_.canonical)
        if isinstance(canonical, Relational):
            # bounds at infinity, as produced by unbounded intervals
            if canonical.rhs is S.Infinity and isinstance(canonical, (Lt, Le)):
                return true
            if canonical.rhs is S.NegativeInfinity and isinstance(canonical, (Gt, Ge)):
                return true
        return canonical
    return formula_


def _normalize_contains(x: Expr, domain: Set) -> Boolean:
    if not isinstance(domain, Set):
        raise TypeError("expected a SymPy set, got %s" % (domain,))
    if isinstance(domain, _NAMED_SETS):
        return Contains(x, domain)
    if isinstance(domain, Interval):
        return normalize(as_boolean(domain.as_relational(x)))
    if isinstance(domain, FiniteSet):
        return Or(*[Eq(x, v) for v in domain.args])
    if isinstance(domain, SetUnion):
        return Or(*[_normalize_contains(x, as_set(s)) for s in domain.args])
    if isinstance(domain, Intersection):
        return And(*[_normalize_contains(x, as_set(s)) for s in domain.args])
    if isinstance(domain, Complement):
        a, b = domain.args
        return And(_normalize_contains(x, as_set(a)), Not(_normalize_contains(x, as_set(b))))
    if isinstance(domain, EmptySet):
        return false
    if isinstance(domain, UniversalSet):
        return true
    result = domain.contains(x)
    if result in (true, false):
        return result
    return Contains(x, domain)


def conjuncts(formula: Boolean) -> list[Boolean]:
    """The top level conjuncts of a normalized formula."""
    if isinstance(formula, And):
        return list(formula.args)
    if formula is true:
        return []
    return [formula]


# ---------------------------------------------------------------------------
# Translation to the predicates of sympy.assumptions

def _predicate_of_atom(atom: Boolean) -> Optional[Boolean]:
    """Predicate of :mod:`sympy.assumptions` equivalent to an atom, or
    ``None`` if there is none."""
    if isinstance(atom, AppliedPredicate):
        return atom
    if isinstance(atom, Contains):
        x, domain = atom.args
        for cls, pred in _SET_PREDICATES:
            if isinstance(domain, cls):
                return pred(as_expr(x))
        return None
    if isinstance(atom, Relational):
        d = atom.lhs - atom.rhs
        if isinstance(atom, Gt):
            return Q.positive(d)
        if isinstance(atom, Lt):
            return Q.negative(d)
        if isinstance(atom, Ge):
            return Q.nonnegative(d)
        if isinstance(atom, Le):
            return Q.nonpositive(d)
        if isinstance(atom, Eq):
            return Q.zero(d)
        if isinstance(atom, Ne):
            return Not(Q.zero(d))
    return None


def to_predicates(formula: Boolean) -> Optional[Boolean]:
    """Translate a normalized formula to an equivalent Boolean combination of
    predicates of :mod:`sympy.assumptions`, or ``None`` if some atom has no
    equivalent.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions import element
    >>> from sympy_extras.assumptions.facts import to_predicates
    >>> to_predicates((x > 0) | element(y, S.Integers))
    Q.integer(y) | Q.positive(x)
    >>> to_predicates(x - y >= 2)
    Q.nonnegative(x - y - 2)
    """
    if isinstance(formula, (BooleanTrue, BooleanFalse)):
        return formula
    if isinstance(formula, _COMPOUND):
        args = [to_predicates(as_boolean(arg)) for arg in formula.args]
        if any(arg is None for arg in args):
            return None
        return as_boolean(formula.func(*args))
    return _predicate_of_atom(formula)


def _consequences(atom: Boolean) -> list[Boolean]:
    """Predicates implied by a top level atom beyond its equivalent
    predicate: the reality of the variables of an inequality and simple
    sign conditions."""
    result: list[Boolean] = []
    if not isinstance(atom, Relational):
        return result
    lhs, rhs = atom.lhs, atom.rhs
    if isinstance(atom, (Gt, Lt, Ge, Le)):
        result.extend(Q.real(s) for s in free_symbols(atom))
    # sign conditions on the non-numeric side
    if lhs.is_number and not rhs.is_number:
        lhs, rhs = rhs, lhs
        atom = atom.reversed
    if rhs.is_number and not lhs.is_number and rhs.is_extended_real:
        c = rhs
        if isinstance(atom, Gt):
            if c >= 0:
                result.append(Q.positive(lhs))
        elif isinstance(atom, Ge):
            if c > 0:
                result.append(Q.positive(lhs))
            elif c == 0:
                result.append(Q.nonnegative(lhs))
            else:
                result.append(Q.real(lhs))
        elif isinstance(atom, Lt):
            if c <= 0:
                result.append(Q.negative(lhs))
        elif isinstance(atom, Le):
            if c < 0:
                result.append(Q.negative(lhs))
            elif c == 0:
                result.append(Q.nonpositive(lhs))
            else:
                result.append(Q.real(lhs))
        elif isinstance(atom, Eq):
            if c.is_integer:
                result.append(Q.integer(lhs))
            elif c.is_rational:
                result.append(Q.rational(lhs))
            else:
                result.append(Q.real(lhs))
            if c > 0:
                result.append(Q.positive(lhs))
            elif c < 0:
                result.append(Q.negative(lhs))
            else:
                result.append(Q.zero(lhs))
    return result


# ---------------------------------------------------------------------------
# Translation to polynomial relations for the CAD

def _is_polynomial(expr: Expr, symbols: set[Symbol]) -> bool:
    """Whether ``expr`` is a polynomial with rational coefficients in
    ``symbols`` (which must contain its free symbols)."""
    if not expr.free_symbols <= symbols:
        return False
    if expr.is_number:
        return bool(expr.is_rational)
    try:
        p = Poly(expr, *sorted(free_symbols(expr), key=lambda s: s.name))
    except PolynomialError:
        return False
    return p.domain.is_QQ or p.domain.is_ZZ


def _relational_of_atom(atom: Boolean, real: set[Symbol]) -> Optional[Boolean]:
    """Polynomial relation with rational coefficients between the real
    variables ``real`` equivalent to the atom, or ``None``."""
    if isinstance(atom, Relational):
        if _is_polynomial(atom.lhs - atom.rhs, real):
            return atom
        return None
    if isinstance(atom, Contains):
        x, domain = atom.args
        if isinstance(domain, Reals) and x.free_symbols <= real:
            return true
        return None
    if isinstance(atom, AppliedPredicate):
        if len(atom.arguments) != 1:
            return None
        [x] = atom.arguments
        if not _is_polynomial(x, real):
            return None
        if atom.function in (Q.real, Q.complex, Q.extended_real, Q.finite):
            return true
        rel = _SIGN_PREDICATES.get(atom.function)
        if rel is not None:
            return rel(x, 0)
        return None
    return None


def to_polynomial(formula: Boolean, real: set[Symbol]) -> Optional[Boolean]:
    """Translate a normalized formula to an equivalent Boolean combination
    of polynomial relations with rational coefficients between the
    variables ``real``, which are known to be real, or ``None`` if some
    atom cannot be translated.

    Examples
    ========

    >>> from sympy import Q, S
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions import element
    >>> from sympy_extras.assumptions.facts import to_polynomial
    >>> to_polynomial(Q.positive(x) & (x*y < 1) & element(y, S.Reals), {x, y})
    (x > 0) & (x*y < 1)
    >>> to_polynomial(x*y < 1, {x}) is None
    True
    """
    if isinstance(formula, (BooleanTrue, BooleanFalse)):
        return formula
    if isinstance(formula, _COMPOUND):
        args = [to_polynomial(as_boolean(arg), real) for arg in formula.args]
        if any(arg is None for arg in args):
            return None
        return as_boolean(formula.func(*args))
    return _relational_of_atom(formula, real)


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
    >>> from sympy_extras.assumptions.facts import predicates_consistent
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
    """The assumptions in force, translated for the backends.

    Parameters
    ==========

    assumptions : Boolean or iterable of Booleans, optional
        Assumptions written as mathematical statements, see the module
        documentation.
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
    >>> from sympy_extras.assumptions import element
    >>> from sympy_extras.assumptions.facts import Facts
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
        items: list[Union[Boolean, bool]] = []
        if assumptions is not None:
            if isinstance(assumptions, (Basic, bool)):
                items.append(assumptions)
            else:
                items.extend(assumptions)
        symbols = set(symbols)
        self.domain: Optional[Set] = None
        if domain is not None:
            domain = sympify(domain)
            if not isinstance(domain, Set):
                raise TypeError("domain must be a SymPy set, got %s" % (domain,))
            self.domain = domain
            for item in items:
                symbols |= free_symbols(sympify(item))
            items.extend(Contains(s, domain) for s in sorted(symbols, key=lambda s: s.name))
        self.formula: Boolean = normalize(And(*[as_boolean(a) for a in items]))
        if self.formula is false:
            raise ValueError("the assumptions are contradictory: %s" % (items,))
        if not isinstance(self.formula, Boolean):
            raise TypeError("the assumptions must be Booleans, got %s" % (items,))
        self.conjuncts: list[Boolean] = conjuncts(self.formula)

        # predicates implied by the assumptions
        predicates: list[Boolean] = []
        for c in self.conjuncts:
            p = to_predicates(c)
            if p is not None:
                predicates.append(p)
            if _is_atom(c):
                predicates.extend(_consequences(c))
            elif isinstance(c, Not) and _is_atom(c.args[0]):
                pass
        self.predicates: Boolean = And(*predicates)

        # variables known to be real or integer
        variables = self.formula.free_symbols | symbols
        self.real: set[Symbol] = set()
        self.integer: set[Symbol] = set()
        for s in variables:
            if not isinstance(s, Symbol):
                continue
            if s.is_extended_real or self._known(Q.real(s)):
                self.real.add(s)
            if s.is_integer or self._known(Q.integer(s)):
                self.integer.add(s)
        for c in self.conjuncts:
            if isinstance(c, (Gt, Lt, Ge, Le)):
                self.real.update(s for s in c.free_symbols if isinstance(s, Symbol))

        # the polynomial part
        polynomial: list[Boolean] = []
        for c in self.conjuncts:
            p = to_polynomial(c, self.real)
            if p is not None and p is not true:
                polynomial.append(p)
        self.polynomial: Boolean = And(*polynomial)

    def _known(self, predicate: Boolean) -> bool:
        try:
            return _sympy_ask(predicate, self.predicates) is True
        except ValueError:
            return False

    def __repr__(self) -> str:
        return "Facts(%s)" % (self.formula,)

    def is_polynomial(self, formula: Boolean) -> bool:
        """Whether the normalized ``formula`` can be handled by the CAD
        given the variables known to be real."""
        return to_polynomial(formula, self.real) is not None

    def with_reals(self, symbols: Iterable[Symbol]) -> Facts:
        """The same facts with ``symbols`` also assumed real."""
        extra = [s for s in symbols if s not in self.real]
        if not extra:
            return self
        return Facts(self.conjuncts + [Contains(s, S.Reals) for s in extra])

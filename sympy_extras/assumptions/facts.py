"""Assumptions written as ordinary mathematical statements, on the core.

In this module an assumption is a Boolean combination of

* relations between expressions, ``x > 0``, ``x**2 + y**2 <= 1``,
  ``Eq(x, y)``, ``Ne(x, 0)``;
* memberships in sets, ``element(x, Integers)`` (that is
  ``Contains(x, Integers)``), ``element(x, Interval(0, 1))``,
  ``element(n, Naturals)``, with the sets of :mod:`sympy_extras.sets`;
* for compatibility, predicates of the core's ``Q`` such as
  ``Q.positive(x)``.

The pieces are combined with ``&``, ``|``, ``~`` or the connectives of
:mod:`sympy_extras.core.logic`.

Following the convention of Mathematica, an inequality between two
expressions is only true when both sides are real: ``x > 0`` says that ``x``
is a positive *real* number. Equations and ``Ne`` do not carry this
implication.

The :class:`Facts` class translates such assumptions into the two forms
used by the backends: predicates of ``Q`` (for ``ask`` and ``refine``)
and Boolean combinations of polynomial relations between real variables
(for the cylindrical algebraic decomposition of
:mod:`sympy_extras.polys.cad`).

This module imports no SymPy: it takes and returns the expressions and
Booleans of :mod:`sympy_extras.core`. Code on SymPy objects uses
:mod:`sympy_extras.assumptions.facts_sympy`, which also keeps
``predicates_consistent`` on SymPy's predicates (a question to the SAT
solver of SymPy's assumptions with its known facts about the
predicates); its native counterpart, on the core's predicates, is
:func:`sympy_extras.assumptions.satask.predicates_consistent`.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.sets import Integers
>>> from sympy_extras.assumptions.facts import Facts, element
>>> x, y, z = symbols('x y z')
>>> f = Facts([x > 2, element(y, Integers), z**3 > x*y])
>>> f.polynomial, sorted(f.real, key=str), sorted(f.integer, key=str)
((x > 2) & (z**3 > x*y), [x, y, z], [y])
"""
from __future__ import annotations

from typing import Callable, Final, Iterable, Optional, Union

from sympy_extras.core.ask import Q, ask
from sympy_extras.core.assume import AppliedPredicate, Predicate
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.logic import (ITE, And, Boolean, BooleanAtom, BooleanLike, Equivalent, Implies, Not, Or,
    Xor, and_, false, not_, or_, to_boolean, true)
from sympy_extras.core.numbers import Infinity, NegativeInfinity
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational, eq, ge, gt, le, lt, ne
from sympy_extras.core.symbol import Symbol
from sympy_extras.sets.fancysets import ComplexesClass, IntegersClass, Naturals0Class, NaturalsClass, RationalsClass
from sympy_extras.sets.sets import (Complement, Contains, EmptySetClass, FiniteSet, Intersection, Interval,
    RealsClass, Reals, Set, Union as SetUnion, UniversalSetClass)

__all__ = ['element', 'normalize', 'conjuncts', 'to_predicates', 'to_polynomial', 'Facts']

#: what the functions taking assumptions accept
Assumption = BooleanLike


# ---------------------------------------------------------------------------
# Building blocks

def element(x: ExprLike, domain: Set) -> Boolean:
    """Membership of ``x`` in the set ``domain``, the counterpart of
    Mathematica's ``Element[x, dom]``.

    It is :class:`~sympy_extras.sets.sets.Contains`, evaluated to ``True``
    or ``False`` when the answer is known.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.sets import Integers, interval
    >>> from sympy_extras.assumptions.facts import element
    >>> x = Symbol('x')
    >>> element(x, Integers), element(2, Integers), element(x, interval(0, 1)), element(2, interval(0, 1))
    (Contains(x, Integers), True, Contains(x, Interval(0, 1)), False)
    """
    if not isinstance(domain, Set):
        raise TypeError("expected a set, got %s" % (domain,))
    member = to_expr(x)
    result = domain.contains(member)
    if isinstance(result, BooleanAtom):
        return result
    return Contains(member, domain)


#: the named sets of numbers, kept as atoms by normalize
_NAMED_SETS: Final = (NaturalsClass, Naturals0Class, IntegersClass, RationalsClass, RealsClass, ComplexesClass)

#: the named sets whose members are real numbers
_REAL_SETS: Final = (NaturalsClass, Naturals0Class, IntegersClass, RationalsClass, RealsClass)


def _naturals0(x: Expr) -> Boolean:
    return and_(Q.integer(x), Q.nonnegative(x))


def _naturals(x: Expr) -> Boolean:
    return and_(Q.integer(x), Q.positive(x))


def _integers(x: Expr) -> Boolean:
    return Q.integer(x)


def _rationals(x: Expr) -> Boolean:
    return Q.rational(x)


def _reals(x: Expr) -> Boolean:
    return Q.real(x)


def _complexes(x: Expr) -> Boolean:
    return Q.complex(x)


#: the predicate equivalent to the membership in a named set
_SET_PREDICATES: Final[list[tuple[type[Set], Callable[[Expr], Boolean]]]] = [
    (Naturals0Class, _naturals0),
    (NaturalsClass, _naturals),
    (IntegersClass, _integers),
    (RationalsClass, _rationals),
    (RealsClass, _reals),
    (ComplexesClass, _complexes),
]

#: the predicates equivalent to a sign condition on a real number
_SIGN_PREDICATES: Final[dict[Predicate, Callable[[Expr, Expr], Boolean]]] = {
    Q.positive: gt, Q.negative: lt, Q.nonnegative: ge, Q.nonpositive: le, Q.zero: eq, Q.nonzero: ne,
}

#: the predicates which hold for every polynomial with rational
#: coefficients in real variables
_REAL_PREDICATES: Final[frozenset[Predicate]] = frozenset({Q.real, Q.complex, Q.extended_real, Q.finite})

_COMPOUND: Final = (And, Or, Not, Implies, Equivalent, Xor, ITE)


def _is_atom(formula: Boolean) -> bool:
    return not isinstance(formula, _COMPOUND)


def _rebuilt(formula: Boolean, args: list[Boolean]) -> Boolean:
    """The connective ``formula`` with ``args``, evaluated."""
    return to_boolean(formula._evaluated(tuple(args)))


def _cleared_denominator(atom: Relational) -> Optional[Boolean]:
    """The relation with a symbolic denominator cleared, as a polynomial
    relation with ``den != 0`` alongside; ``None`` when there is none.

    ``x/a = 1`` is ``x - a = 0 and a != 0``; ``p/q > 0`` is ``p q > 0``
    (which implies ``q != 0``); ``p/q >= 0`` is ``p q >= 0 and q != 0``.
    The quantifier elimination, ``ask`` and ``refine`` all rejected a
    parameter in a denominator before (sympy-extras#32).
    """
    from sympy_extras.polys.rationaltools import together
    numerator, denominator = together(atom.lhs - atom.rhs).as_numer_denom()
    if not denominator.free_symbols:
        return None
    # only a polynomial denominator: the point is polynomial quantifier
    # elimination, and a transcendental one (exp(-x) - x**2 < 0 has exp(x)
    # below the line) is left for the solvers that read it as it is
    symbols = sorted(denominator.free_symbols | numerator.free_symbols, key=lambda s: s.name)
    if not (denominator.is_polynomial(*symbols) and numerator.is_polynomial(*symbols)):
        return None
    nonzero = ne(denominator, 0)
    if isinstance(atom, Eq):
        return and_(eq(numerator, 0), nonzero)
    if isinstance(atom, Ne):
        return and_(ne(numerator, 0), nonzero)
    product = numerator*denominator
    if isinstance(atom, Gt):
        return gt(product, 0)
    if isinstance(atom, Lt):
        return lt(product, 0)
    if isinstance(atom, Ge):
        return and_(ge(product, 0), nonzero)
    if isinstance(atom, Le):
        return and_(le(product, 0), nonzero)
    return None


def normalize(formula: Assumption) -> Boolean:
    """Normal form of an assumption or query.

    Memberships in intervals, finite sets and combinations of sets are
    rewritten as relations and Boolean combinations; memberships in the
    named sets (``Integers``, ``Reals``, ...) and predicates of ``Q`` are
    kept as atoms; relations are put in canonical form, with their
    symbolic denominators cleared. ``True``/``False`` become
    ``true``/``false``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.sets import Integers, interval, finite_set, Union
    >>> from sympy_extras.assumptions.facts import element, normalize
    >>> x = Symbol('x')
    >>> normalize(element(x, interval(0, 1)))
    (x >= 0) & (x <= 1)
    >>> normalize(element(x, Union(finite_set(1, 2), interval(3, 4, True, True))))
    Eq(x, 1) | Eq(x, 2) | ((x > 3) & (x < 4))
    >>> normalize(element(x, interval(0, oo))), normalize(element(x, Integers) & (x > 0))
    (x >= 0, Contains(x, Integers) & (x > 0))
    """
    f = to_boolean(formula)
    if isinstance(f, BooleanAtom):
        return f
    if isinstance(f, Contains):
        member, domain = f.args
        if not isinstance(member, Expr) or not isinstance(domain, Set):
            raise TypeError("Contains takes an expression and a set")
        return _normalize_contains(member, domain)
    if isinstance(f, ITE):
        c, a, b = [normalize(arg) for arg in f.args]
        return or_(and_(c, a), and_(not_(c), b))
    if isinstance(f, _COMPOUND):
        return _rebuilt(f, [normalize(arg) for arg in f.args])
    if isinstance(f, Relational):
        cleared = _cleared_denominator(f)
        if cleared is not None:
            return normalize(cleared)
        canonical = f.canonical
        # bounds at infinity, as produced by unbounded intervals
        if isinstance(canonical.rhs, Infinity) and isinstance(canonical, (Lt, Le)):
            return true
        if isinstance(canonical.rhs, NegativeInfinity) and isinstance(canonical, (Gt, Ge)):
            return true
        return canonical
    return f


def _normalize_contains(x: Expr, domain: Set) -> Boolean:
    if isinstance(domain, _NAMED_SETS):
        return Contains(x, domain)
    if isinstance(domain, Interval):
        return normalize(domain.as_relational(x))
    if isinstance(domain, FiniteSet):
        return or_(*[eq(x, v) for v in domain.elements])
    if isinstance(domain, SetUnion):
        return or_(*[_normalize_contains(x, s) for s in domain.sets])
    if isinstance(domain, Intersection):
        return and_(*[_normalize_contains(x, s) for s in domain.sets])
    if isinstance(domain, Complement):
        a, b = domain.args_sets
        return and_(_normalize_contains(x, a), not_(_normalize_contains(x, b)))
    if isinstance(domain, EmptySetClass):
        return false
    if isinstance(domain, UniversalSetClass):
        return true
    result = domain.contains(x)
    if isinstance(result, BooleanAtom):
        return result
    return Contains(x, domain)


def conjuncts(formula: Boolean) -> list[Boolean]:
    """The top level conjuncts of a normalized formula.

    Examples
    ========

    >>> from sympy_extras.core import symbols, true
    >>> from sympy_extras.assumptions.facts import conjuncts
    >>> x, y = symbols('x y')
    >>> conjuncts((x > 0) & (y > 0)), conjuncts((x > 0) | (y > 0)), conjuncts(true)
    ([x > 0, y > 0], [(x > 0) | (y > 0)], [])
    """
    if isinstance(formula, And):
        return list(formula.args)
    if formula == true:
        return []
    return [formula]


# ---------------------------------------------------------------------------
# Translation to the predicates of Q

def _predicate_of_atom(atom: Boolean) -> Optional[Boolean]:
    """Predicate of ``Q`` equivalent to an atom, or ``None`` if there is
    none."""
    if isinstance(atom, AppliedPredicate):
        return atom
    if isinstance(atom, Contains):
        x, domain = atom.args
        if not isinstance(x, Expr):
            return None
        for cls, pred in _SET_PREDICATES:
            if isinstance(domain, cls):
                return pred(x)
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
    predicates of ``Q``, or ``None`` if some atom has no equivalent.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.sets import Integers
    >>> from sympy_extras.assumptions.facts import element, to_predicates
    >>> x, y = symbols('x y')
    >>> to_predicates((x > 0) | element(y, Integers)), to_predicates(x - y >= 2)
    (Q.integer(y) | Q.positive(x), Q.nonnegative(x - y - 2))
    """
    if isinstance(formula, BooleanAtom):
        return formula
    if isinstance(formula, _COMPOUND):
        args: list[Boolean] = []
        for arg in formula.args:
            p = to_predicates(arg)
            if p is None:
                return None
            args.append(p)
        return _rebuilt(formula, args)
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
        result.extend(Q.real(s) for s in sorted(atom.free_symbols, key=lambda s: s.name))
    # sign conditions on the non-numeric side
    if lhs.is_number and not rhs.is_number:
        lhs, rhs = rhs, lhs
        atom = atom.reversed
    if rhs.is_number and not lhs.is_number and rhs.is_extended_real:
        c = rhs
        if isinstance(atom, Gt):
            if _holds(ge(c, 0)):
                result.append(Q.positive(lhs))
        elif isinstance(atom, Ge):
            if _holds(gt(c, 0)):
                result.append(Q.positive(lhs))
            elif _holds(eq(c, 0)):
                result.append(Q.nonnegative(lhs))
            else:
                result.append(Q.real(lhs))
        elif isinstance(atom, Lt):
            if _holds(le(c, 0)):
                result.append(Q.negative(lhs))
        elif isinstance(atom, Le):
            if _holds(lt(c, 0)):
                result.append(Q.negative(lhs))
            elif _holds(eq(c, 0)):
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
            if _holds(gt(c, 0)):
                result.append(Q.positive(lhs))
            elif _holds(lt(c, 0)):
                result.append(Q.negative(lhs))
            else:
                result.append(Q.zero(lhs))
    return result


def _holds(b: Boolean) -> bool:
    """The truth of a comparison of real numbers (which SymPy decides when
    a comparison of them is used as a Python ``bool``)."""
    if isinstance(b, BooleanAtom):
        return bool(b)
    raise TypeError("cannot determine truth value of Relational: %s" % (b,))


# ---------------------------------------------------------------------------
# Translation to polynomial relations for the CAD

def _is_polynomial(expr: Expr, symbols: set[Symbol]) -> bool:
    """Whether ``expr`` is a polynomial with rational coefficients in
    ``symbols`` (which must contain its free symbols)."""
    from sympy_extras.polys.polyerrors import BasePolynomialError
    from sympy_extras.polys.polytools import Poly
    free = expr.free_symbols
    if not free <= symbols:
        return False
    if not free:
        # a constant is polynomial when it is a rational number; anything
        # else without free symbols (an ``AccumBounds``, an unevaluated
        # limit) would become a generator of ``Poly`` and pass for one
        return bool(expr.is_rational)
    try:
        p = Poly(expr, *sorted(free, key=lambda s: s.name))
    except BasePolynomialError:
        # not a polynomial, or coefficients outside ZZ, QQ, ZZ[a], QQ[a]
        return False
    return p.domain in ('ZZ', 'QQ')


def _relational_of_atom(atom: Boolean, real: set[Symbol]) -> Optional[Boolean]:
    """Polynomial relation with rational coefficients between the real
    variables ``real`` equivalent to the atom, or ``None``."""
    if isinstance(atom, Relational):
        if _is_polynomial(atom.lhs - atom.rhs, real):
            return atom
        return None
    if isinstance(atom, Contains):
        x, domain = atom.args
        if isinstance(domain, RealsClass) and isinstance(x, Expr) and _is_polynomial(x, real):
            # a polynomial with rational coefficients in real variables;
            # (a + I*b)**2 has real variables and is not real
            return true
        return None
    if isinstance(atom, AppliedPredicate):
        if len(atom.arguments) != 1:
            return None
        [x] = atom.arguments
        if not isinstance(x, Expr) or not _is_polynomial(x, real):
            return None
        p = atom.function
        if p in _REAL_PREDICATES:
            return true
        relation = _SIGN_PREDICATES.get(p)
        if relation is not None:
            return relation(x, to_expr(0))
        return None
    return None


def to_polynomial(formula: Boolean, real: set[Symbol]) -> Optional[Boolean]:
    """Translate a normalized formula to an equivalent Boolean combination
    of polynomial relations with rational coefficients between the
    variables ``real``, which are known to be real, or ``None`` if some
    atom cannot be translated.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Q
    >>> from sympy_extras.sets import Reals
    >>> from sympy_extras.assumptions.facts import element, to_polynomial
    >>> x, y = symbols('x y')
    >>> to_polynomial(Q.positive(x) & (x*y < 1) & element(y, Reals), {x, y})
    (x > 0) & (x*y < 1)
    >>> to_polynomial(x*y < 1, {x}) is None
    True
    """
    if isinstance(formula, BooleanAtom):
        return formula
    if isinstance(formula, _COMPOUND):
        args: list[Boolean] = []
        for arg in formula.args:
            p = to_polynomial(arg, real)
            if p is None:
                return None
            args.append(p)
        return _rebuilt(formula, args)
    return _relational_of_atom(formula, real)


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
        A named set (``Reals``, ``Integers``, ...) all the variables in
        ``symbols`` are assumed to belong to, like the domain argument of
        Mathematica's ``Reduce`` or ``Resolve``.
    symbols : iterable of Symbol, optional
        The variables of the expression the assumptions are used for; with
        ``domain`` they are all assumed to be in the domain.

    Attributes
    ==========

    formula : Boolean
        The conjunction of all assumptions in normal form.
    conjuncts : list of Boolean
        Its conjuncts.
    predicates : Boolean
        Consequences of the assumptions as predicates of ``Q``.
    real : set of Symbol
        The variables known to be real.
    integer : set of Symbol
        The variables known to be integers.
    polynomial : Boolean
        The part of the assumptions which is a Boolean combination of
        polynomial relations with rational coefficients between real
        variables, as used by the CAD.

    The attributes are set once by the constructor and not changed.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.sets import Integers
    >>> from sympy_extras.assumptions.facts import Facts, element
    >>> x, y, z = symbols('x y z')
    >>> f = Facts([x > 2, element(y, Integers), z**3 > x*y])
    >>> f.predicates
    Q.integer(y) & Q.positive(x) & Q.real(x) & Q.real(y) & Q.real(z) & Q.positive(x - 2) & Q.positive(-x*y + z**3)
    >>> f.polynomial, f
    ((x > 2) & (z**3 > x*y), Facts(Contains(y, Integers) & (x > 2) & (z**3 > x*y)))
    """

    def __init__(self, assumptions: Union[None, Assumption, Iterable[Assumption]] = None,
                 domain: Optional[Set] = None, symbols: Iterable[Symbol] = ()) -> None:
        items: list[Boolean] = []
        if assumptions is not None:
            if isinstance(assumptions, (Basic, bool)):
                items.append(to_boolean(assumptions))
            else:
                items.extend(to_boolean(a) for a in assumptions)
        variables_given = set(symbols)
        self.domain: Final[Optional[Set]] = domain
        if domain is not None:
            if not isinstance(domain, Set):
                raise TypeError("domain must be a set, got %s" % (domain,))
            for item in items:
                variables_given |= item.free_symbols
            items.extend(Contains(s, domain) for s in sorted(variables_given, key=lambda s: s.name))
        formula = normalize(and_(*items))
        if formula == false:
            raise ValueError("the assumptions are contradictory: %s" % (items,))
        self.formula: Final[Boolean] = formula
        self.conjuncts: Final[list[Boolean]] = conjuncts(formula)

        # predicates implied by the assumptions
        predicates: list[Boolean] = []
        for c in self.conjuncts:
            p = to_predicates(c)
            if p is not None:
                predicates.append(p)
            if _is_atom(c):
                predicates.extend(_consequences(c))
        self.predicates: Final[Boolean] = and_(*predicates)

        # variables known to be real or integer
        real: set[Symbol] = set()
        integer: set[Symbol] = set()
        for s in formula.free_symbols | variables_given:
            if s.is_extended_real or self._known(Q.real(s)):
                real.add(s)
            if s.is_integer or self._known(Q.integer(s)):
                integer.add(s)
        for c in self.conjuncts:
            if isinstance(c, (Gt, Lt, Ge, Le)):
                real.update(c.free_symbols)
        self.real: Final[set[Symbol]] = real
        self.integer: Final[set[Symbol]] = integer

        # the polynomial part
        polynomial: list[Boolean] = []
        for c in self.conjuncts:
            p = to_polynomial(c, real)
            if p is not None and p != true:
                polynomial.append(p)
        self.polynomial: Final[Boolean] = and_(*polynomial)

    def _known(self, predicate: Boolean) -> bool:
        try:
            return ask(predicate, self.predicates) is True
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
        return Facts(self.conjuncts + [Contains(s, Reals) for s in extra])

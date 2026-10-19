"""``Q`` and ``ask``: questions about expressions under assumptions.

``ask(Q.positive(x + 1), Q.positive(x))`` answers ``True``, ``False`` or
``None``. The assumptions (with those of ``global_assumptions``) are
read as a conjunction of facts about subexpressions, ``Q.positive(x)``
giving the fact ``positive`` about ``x``; the facts about each subject
are closed under the implications of the old assumptions and checked
for consistency (``ValueError`` otherwise, as in SymPy). The proposition
is then answered by the old assumptions after replacing every subject by
a dummy symbol carrying its facts: ``ask(Q.positive(x + 1), Q.positive(x))``
is ``(_x + 1).is_positive`` with ``_x`` positive. A relation is asked
through ``Q.is_true`` and answered by the relational functions; a
proposition built with the connectives is answered by three-valued
logic; the handlers registered on a predicate are consulted first. What
stays unanswered is asked to SymPy's ``ask`` (its SAT solver finds, for
example, ``Q.nonzero(x)`` from ``Q.positive(x) | Q.negative(x)``), unless
the question involves a predicate of the user. As in SymPy, relations
among the assumptions are not used (``ask(Q.positive(x), x > 0)`` is
``None``) [1]_.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: ask*,
       https://docs.sympy.org/latest/modules/assumptions/ask.html
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

from .assume import AppliedPredicate, Predicate, global_assumptions
from .assumptions import PREDICATES, Truth, closure, facts_of, is_consistent, minimal_generators
from .basic import Basic
from .expr import Expr
from .logic import (ITE, And, Boolean, BooleanAtom, BooleanLike, Equivalent, Implies, Not, Or, Xor, and_, false,
    not_, to_boolean)
from .relational import Relational, eq, ge, gt, le, lt, ne
from .symbol import Dummy, Symbol

__all__ = ['AssumptionKeys', 'Q', 'ask']


class AssumptionKeys:
    """The predicates, as attributes of ``Q``.

    Examples
    ========

    >>> from sympy_extras.core import Q, Symbol
    >>> x = Symbol('x')
    >>> Q.positive(x), Q.extended_real(x), Q.gt(x, 1), Q.is_true(x > 1), Q.invertible(x)
    (Q.positive(x), Q.extended_real(x), Q.gt(x, 1), Q.is_true(x > 1), Q.invertible(x))
    """

    algebraic = Predicate._known('algebraic')
    antihermitian = Predicate._known('antihermitian')
    commutative = Predicate._known('commutative')
    complex = Predicate._known('complex')
    complex_elements = Predicate._known('complex_elements')
    composite = Predicate._known('composite')
    diagonal = Predicate._known('diagonal')
    eq = Predicate._known('eq', 2)
    even = Predicate._known('even')
    extended_negative = Predicate._known('extended_negative')
    extended_nonnegative = Predicate._known('extended_nonnegative')
    extended_nonpositive = Predicate._known('extended_nonpositive')
    extended_nonzero = Predicate._known('extended_nonzero')
    extended_positive = Predicate._known('extended_positive')
    extended_real = Predicate._known('extended_real')
    finite = Predicate._known('finite')
    fullrank = Predicate._known('fullrank')
    ge = Predicate._known('ge', 2)
    gt = Predicate._known('gt', 2)
    hermitian = Predicate._known('hermitian')
    imaginary = Predicate._known('imaginary')
    infinite = Predicate._known('infinite')
    integer = Predicate._known('integer')
    integer_elements = Predicate._known('integer_elements')
    invertible = Predicate._known('invertible')
    irrational = Predicate._known('irrational')
    is_true = Predicate._known('is_true')
    le = Predicate._known('le', 2)
    lower_triangular = Predicate._known('lower_triangular')
    lt = Predicate._known('lt', 2)
    ne = Predicate._known('ne', 2)
    negative = Predicate._known('negative')
    negative_infinite = Predicate._known('negative_infinite')
    noninteger = Predicate._known('noninteger')
    nonnegative = Predicate._known('nonnegative')
    nonpositive = Predicate._known('nonpositive')
    nonzero = Predicate._known('nonzero')
    normal = Predicate._known('normal')
    odd = Predicate._known('odd')
    orthogonal = Predicate._known('orthogonal')
    positive = Predicate._known('positive')
    positive_definite = Predicate._known('positive_definite')
    positive_infinite = Predicate._known('positive_infinite')
    prime = Predicate._known('prime')
    rational = Predicate._known('rational')
    real = Predicate._known('real')
    real_elements = Predicate._known('real_elements')
    singular = Predicate._known('singular')
    square = Predicate._known('square')
    symmetric = Predicate._known('symmetric')
    transcendental = Predicate._known('transcendental')
    triangular = Predicate._known('triangular')
    unit_triangular = Predicate._known('unit_triangular')
    unitary = Predicate._known('unitary')
    upper_triangular = Predicate._known('upper_triangular')
    zero = Predicate._known('zero')


#: the predicates
Q = AssumptionKeys()

#: the predicates of ``Q`` by name
PREDICATES_OF_Q: dict[str, Predicate] = {
    name: p for name, p in vars(AssumptionKeys).items() if isinstance(p, Predicate)}

#: the binary predicates answered by the relational functions
_RELATIONS: dict[str, Callable[[Expr, Expr], Basic]] = {'eq': eq, 'ne': ne, 'gt': gt, 'ge': ge, 'lt': lt, 'le': le}


def ask(proposition: BooleanLike, assumptions: BooleanLike = True,
        context: Optional[Iterable[BooleanLike]] = None) -> Truth:
    """``True``, ``False`` or ``None``: whether the proposition holds
    under the assumptions and the context (``global_assumptions``).

    Examples
    ========

    >>> from sympy_extras.core import Q, ask, Symbol, Integer
    >>> x, y = Symbol('x'), Symbol('y')
    >>> ask(Q.positive(x + 1), Q.positive(x)), ask(Q.even(x), Q.odd(x)), ask(Q.prime(x), Q.integer(x))
    (True, False, None)
    >>> ask(Q.integer(2)), ask(Q.positive(x*y), Q.positive(x) & Q.negative(y)), ask(x > 0, Q.positive(x))
    (True, False, True)
    >>> ask(Q.positive(x), Q.positive(x) | Q.negative(x)), ask(Q.nonzero(x), Q.positive(x) | Q.negative(x))
    (None, True)
    >>> ask(Q.positive(x), Q.positive(x) & Q.zero(x))
    Traceback (most recent call last):
    ...
    ValueError: inconsistent assumptions Q.positive(x) & Q.zero(x)
    """
    prop = _argument(proposition, 'proposition')
    given = _argument(assumptions, 'assumptions')
    everything = and_(given, *(global_assumptions if context is None else context))
    if everything == false:
        raise ValueError("Inconsistent assumptions")
    facts = _Facts(everything, prop)
    result = facts.evaluate(prop)
    if result is not None or _has_user_predicate(prop) or _has_user_predicate(everything):
        return result
    return _ask_sympy(prop, everything)


def _argument(value: object, role: str) -> Boolean:
    if isinstance(value, bool):
        return to_boolean(value)
    if isinstance(value, Boolean) and (role == 'assumptions' or not isinstance(value, Symbol)):
        return value
    raise TypeError("%s must be a valid logical expression" % role)


def _has_user_predicate(b: Basic) -> bool:
    return any(isinstance(p, Predicate) and not p._defined for p in b.atoms(Predicate))


def _ask_sympy(prop: Boolean, assumptions: Boolean) -> Truth:
    import sympy
    from .interop import to_sympy
    result = sympy.ask(to_sympy(prop), to_sympy(assumptions))
    if isinstance(result, bool):
        return result
    return None


class _Facts:
    """The assumptions read as facts about subjects, and the evaluation
    of propositions under them."""

    def __init__(self, assumptions: Boolean, proposition: Boolean) -> None:
        self.assumptions: Boolean = assumptions
        self.literals: set[Boolean] = set()
        self.facts: dict[Basic, dict[str, bool]] = {}
        for conjunct in And.make_args(assumptions):
            self._add(to_boolean(conjunct), True)
        self.dummies: dict[Basic, Basic] = {}
        for subject, given in self.facts.items():
            own: dict[str, bool] = dict(subject._assumptions) if isinstance(subject, Symbol) else {}
            # as in SymPy, the facts are checked for consistency when the
            # subject occurs in the proposition
            if proposition.has(subject) and (any(own.get(k) == (not v) for k, v in given.items())
                                             or not is_consistent({**own, **given})):
                raise ValueError("inconsistent assumptions %s" % self._about(subject))
            if isinstance(subject, Expr):
                closed = closure({**own, **given})
                name = subject.name if isinstance(subject, Symbol) else 'x'
                self.dummies[subject] = Dummy(name, **minimal_generators(closed))

    def _add(self, literal: Boolean, value: bool) -> None:
        self.literals.add(literal if value else not_(literal))
        if isinstance(literal, BooleanAtom):
            if bool(literal) != value:
                raise ValueError("Inconsistent assumptions")
            return
        if isinstance(literal, Not):
            self._add(literal.arg, not value)
            return
        if isinstance(literal, AppliedPredicate) and literal.function._defined:
            name = literal.function.name
            if name == 'is_true':
                inner = literal.arg
                if isinstance(inner, Boolean):
                    self._add(inner, value)
            elif name in PREDICATES and len(literal.arguments) == 1:
                self._fact(literal.arg, name, value)
            elif name in ('positive_infinite', 'negative_infinite') and value:
                self._fact(literal.arg, 'infinite', True)
                self._fact(literal.arg, 'extended_' + name.split('_')[0], True)

    def _fact(self, subject: Basic, name: str, value: bool) -> None:
        facts = self.facts.setdefault(subject, {})
        if facts.get(name) == (not value):
            raise ValueError("inconsistent assumptions %s" % self._about(subject))
        facts[name] = value

    def _about(self, subject: Basic) -> Boolean:
        """The conjunction of the literals about ``subject``."""
        return and_(*sorted((lit for lit in self.literals if lit.has(subject)), key=str))

    def substituted(self, e: Expr) -> Expr:
        """``e`` with the subjects replaced by the dummies carrying their facts."""
        return e.xreplace(self.dummies) if self.dummies else e

    def evaluate(self, prop: Boolean) -> Truth:
        if prop in self.literals:
            return True
        if not_(prop) in self.literals:
            return False
        if isinstance(prop, BooleanAtom):
            return bool(prop)
        if isinstance(prop, And):
            return _all([self.evaluate(a) for a in prop.args])
        if isinstance(prop, Or):
            return _any([self.evaluate(a) for a in prop.args])
        if isinstance(prop, Not):
            value = self.evaluate(prop.arg)
            return None if value is None else not value
        if isinstance(prop, Implies):
            a, b = prop.args
            return _any([_negated(self.evaluate(a)), self.evaluate(b)])
        if isinstance(prop, Xor):
            values = [self.evaluate(a) for a in prop.args]
            if any(v is None for v in values):
                return None
            return sum(1 for v in values if v) % 2 == 1
        if isinstance(prop, Equivalent):
            values = [self.evaluate(a) for a in prop.args]
            if any(v is None for v in values):
                return None
            return all(values) or not any(values)
        if isinstance(prop, ITE):
            c, a, b = prop.args
            condition = self.evaluate(c)
            if condition is None:
                return None
            return self.evaluate(a if condition else b)
        if isinstance(prop, AppliedPredicate):
            return self._predicate(prop)
        if isinstance(prop, Relational):
            return self._relation(prop)
        return None

    def _predicate(self, prop: AppliedPredicate) -> Truth:
        predicate, args = prop.function, prop.arguments
        if predicate._handlers and len(args) == 1:
            answer = predicate.eval(args[0], self.assumptions)
            if answer is not None:
                return answer
        if not predicate._defined:
            return None
        name = predicate.name
        if name == 'is_true':
            inner = args[0]
            return self.evaluate(inner) if isinstance(inner, Boolean) else None
        if name in _RELATIONS:
            a, b = args
            if isinstance(a, Expr) and isinstance(b, Expr):
                return _truth(_RELATIONS[name](self.substituted(a), self.substituted(b)))
            return None
        arg = args[0]
        if not isinstance(arg, Expr):
            return None
        facts = facts_of(self.substituted(arg))
        if name in PREDICATES:
            return facts.get(name)
        if name in ('positive_infinite', 'negative_infinite'):
            infinite, sign = facts.get('infinite'), facts.get('extended_' + name.split('_')[0])
            if infinite is True and sign is True:
                return True
            if infinite is False or sign is False:
                return False
        return None

    def _relation(self, prop: Relational) -> Truth:
        return _truth(prop._evaluated((self.substituted(prop.lhs), self.substituted(prop.rhs))))


def _truth(b: Basic) -> Truth:
    return bool(b) if isinstance(b, BooleanAtom) else None


def _negated(v: Truth) -> Truth:
    return None if v is None else not v


def _all(values: list[Truth]) -> Truth:
    if any(v is False for v in values):
        return False
    if all(v is True for v in values):
        return True
    return None


def _any(values: list[Truth]) -> Truth:
    if any(v is True for v in values):
        return True
    if all(v is False for v in values):
        return False
    return None

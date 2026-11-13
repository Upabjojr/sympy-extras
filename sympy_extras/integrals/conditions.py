"""Values which hold under a condition on the parameters.

The definite integrators of this package return a
:class:`ConditionalValue`: the value of an integral together with the
condition (an inequality on the parameters, ``a > 0``, or a conjunction of
them) under which the integral converges and the formula holds. The
conditions are decided against the assumptions by
:func:`sympy_extras.assumptions.ask.ask` and what remains undecided is
reported in a :class:`~sympy_extras.functions.elementary.piecewise.Piecewise`.
The module also holds the numerical comparisons of the integrators: the
sample points of the region of the assumptions (:func:`sample_values`),
:func:`numerically_equal` and :func:`numerical_verdict`, on the reliable
values of :mod:`sympy_extras._numeric`, with the structure theorem
(:mod:`sympy_extras.simplify.structure`) as a second opinion.

The module works on the objects of the core (:mod:`sympy_extras.core`,
the functions of :mod:`sympy_extras.functions`) with the native ``ask``,
``find_instance`` and structure theorem, and imports no SymPy:
:mod:`sympy_extras.integrals.conditions_sympy` is its face on SymPy
objects, used by the integrators which are still written on SymPy.

Examples
========

>>> from sympy_extras.core import Integer, symbols
>>> from sympy_extras.integrals.conditions import ConditionalValue
>>> a = symbols('a')
>>> v = ConditionalValue(1/a, a > 0)
>>> v
ConditionalValue(1/a, a > 0)
>>> v.scaled(Integer(2))
ConditionalValue(2/a, a > 0)
>>> v.add(ConditionalValue(a, a < 1))
ConditionalValue(a + 1/a, (a > 0) & (a < 1))
"""
from __future__ import annotations

import random
from fractions import Fraction
from typing import Callable, Final, Iterable, Optional, Sequence

import mpmath

from sympy_extras._numeric import reliable_value
from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import Facts, element
from sympy_extras.assumptions.sat import find_instance
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import evalf, numeric_value
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.logic import (And, Boolean, BooleanFalse, BooleanLike, BooleanTrue, Or, and_, false, or_,
                                     to_boolean, true)
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import (ComplexInfinity, Infinity, Integer, NaN, NegativeInfinity, Rational,
                                       number_from_fraction, pi)
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Ge, Gt, Le, Lt, Relational, ge, gt, le, lt, ne
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Abs, Arg, Im, Re, re
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
from sympy_extras.sets.fancysets import Integers
from sympy_extras.sets.sets import Reals

__all__ = ['ConditionalValue', 'decide', 'combine', 'real_form', 'numerically_equal', 'numerical_verdict',
           'positive', 'canonical', 'symbol_facts', 'with_symbol_facts', 'plain_symbols', 'sample_values',
           'Cases']

#: the ``(value, condition)`` pairs of a value which is a Piecewise of cases
Cases = list[tuple[Expr, Boolean]]


class ConditionalValue:
    """A value with the condition under which it holds.

    Attributes
    ==========

    value : Expr
    condition : Boolean
        ``true`` when the value holds unconditionally.
    """

    __slots__ = ('value', 'condition')

    def __init__(self, value: ExprLike, condition: BooleanLike = True) -> None:
        self.value: Final[Expr] = to_expr(value)
        self.condition: Final[Boolean] = to_boolean(condition)

    def __repr__(self) -> str:
        if isinstance(self.condition, BooleanTrue):
            return "ConditionalValue(%s)" % (self.value,)
        return "ConditionalValue(%s, %s)" % (self.value, self.condition)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, ConditionalValue) and self.value == other.value
                and self.condition == other.condition)

    def __hash__(self) -> int:
        return hash((self.value, self.condition))

    def scaled(self, factor: ExprLike) -> ConditionalValue:
        """The value multiplied by ``factor`` (which must not depend on the
        condition)."""
        return ConditionalValue(mul(to_expr(factor), self.value), self.condition)

    def add(self, other: ConditionalValue) -> ConditionalValue:
        """The sum, holding when both conditions do."""
        return ConditionalValue(add(self.value, other.value), and_(self.condition, other.condition))

    def with_condition(self, condition: BooleanLike) -> ConditionalValue:
        """The same value under ``condition`` as well."""
        return ConditionalValue(self.value, and_(self.condition, to_boolean(condition)))

    def as_piecewise(self, otherwise: ExprLike) -> Expr:
        """The value where the condition holds and ``otherwise`` elsewhere;
        the value alone when the condition is ``true``. A value which is
        itself a ``Piecewise`` of cases, the condition being the
        disjunction of theirs (see :meth:`cases`), keeps its branches,
        with ``otherwise`` as the last one.

        >>> from sympy_extras.core import Integer, symbols
        >>> from sympy_extras.core.logic import or_
        >>> from sympy_extras.functions import piecewise
        >>> from sympy_extras.integrals.conditions import ConditionalValue
        >>> a = symbols('a')
        >>> ConditionalValue(piecewise((a, a > 1), (-a, a < 0)), or_(a > 1, a < 0)).as_piecewise(Integer(0))
        Piecewise((a, a > 1), (-a, a < 0), (0, True))
        """
        if isinstance(self.condition, BooleanTrue):
            return self.value
        branches = self.cases()
        if branches is not None:
            return piecewise(*branches, (to_expr(otherwise), true))
        return piecewise((self.value, self.condition), (to_expr(otherwise), true))

    def cases(self) -> Optional[Cases]:
        """The ``(value, condition)`` pairs of a value which is a
        ``Piecewise`` of cases whose conditions the condition is the
        disjunction of; ``None`` for any other value."""
        if not isinstance(self.value, Piecewise):
            return None
        pairs = [(pair.expr, pair.cond) for pair in self.value.args]
        if any(isinstance(condition, BooleanTrue) for _, condition in pairs) \
                or self.condition != or_(*[c for _, c in pairs]):
            return None
        return pairs

    def mapped(self, function: Callable[[Expr, Boolean], Expr]) -> ConditionalValue:
        """The value with ``function`` applied to it and its condition,
        case by case for a value which is a ``Piecewise`` of cases."""
        branches = self.cases()
        if branches is None:
            return ConditionalValue(function(self.value, self.condition), self.condition)
        return ConditionalValue(Piecewise(*[(function(v, c), c) for v, c in branches]), self.condition)


def _conjuncts(condition: Boolean) -> tuple[Boolean, ...]:
    if isinstance(condition, And):
        return tuple(a for a in condition.args if isinstance(a, Boolean))
    return (condition,)


def _boolean_args(condition: Boolean) -> list[Boolean]:
    return [a for a in condition.args if isinstance(a, Boolean)]


def decide(condition: Boolean, assumptions: Assumptions) -> Optional[Boolean]:
    """``condition`` with the conjuncts settled by the assumptions removed:
    ``true`` when it follows from them, ``None`` when it is refuted (one
    conjunct is false under the assumptions), and otherwise the conjunction
    of what stays undecided.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.conditions import decide
    >>> a, b = symbols('a b')
    >>> decide((a > 0) & (b > 1), a > 1), decide(a < 0, a > 1)
    (b > 1, None)
    """
    if isinstance(condition, BooleanTrue):
        return true
    if isinstance(condition, BooleanFalse):
        return None
    remaining: list[Boolean] = []
    for part in _conjuncts(condition):
        part_ = _settled(part, assumptions)
        verdict = ask(part_, assumptions)
        if verdict is False:
            return None
        if verdict is None:
            remaining.append(part_)
    return and_(*remaining)


def _settled(part: Boolean, assumptions: Assumptions) -> Boolean:
    """A conjunct in its real form, canonical; a disjunction of such
    (``|arg z| <= 2 pi`` or ``|arg z| < 2 pi``) is true when one of its
    alternatives is."""
    if isinstance(part, Or):
        alternatives = [_settled(alternative, assumptions) for alternative in _boolean_args(part)]
        if any(isinstance(alternative, BooleanTrue) for alternative in alternatives):
            return true
        return or_(*alternatives)
    if isinstance(part, And):
        # an alternative which is a conjunction (the Mellin conditions on
        # two scales, (|arg a| <= pi/2 & |arg v| < pi/2) | ...): true when
        # every conjunct is
        conjuncts = [_settled(conjunct, assumptions) for conjunct in _boolean_args(part)]
        decided = [True if isinstance(conjunct, BooleanTrue) else ask(conjunct, assumptions)
                   for conjunct in conjuncts]
        if all(verdict is True for verdict in decided):
            return true
        return and_(*[conjunct for conjunct, verdict in zip(conjuncts, decided) if verdict is not True])
    return canonical(real_form(part, assumptions))


def _relation(relation: Relational, lhs: Expr, rhs: Expr) -> Boolean:
    """The relation of the class of ``relation`` between ``lhs`` and
    ``rhs``, evaluated (SymPy's ``relation.func(lhs, rhs)``)."""
    if isinstance(relation, Lt):
        return lt(lhs, rhs)
    if isinstance(relation, Le):
        return le(lhs, rhs)
    if isinstance(relation, Gt):
        return gt(lhs, rhs)
    if isinstance(relation, Ge):
        return ge(lhs, rhs)
    raise TypeError("expected an inequality, got %s" % (relation,))


def _is_inequality(condition: Basic) -> bool:
    return isinstance(condition, (Lt, Le, Gt, Ge))


def canonical(condition: Boolean) -> Boolean:
    """A relation with the constant term of its left side moved to the
    right: ``k + 1 > 0`` becomes ``k > -1``; other conditions are
    returned unchanged.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.conditions import canonical
    >>> k = symbols('k')
    >>> canonical(k + 1 < 1)
    k < 0
    """
    if isinstance(condition, Relational) and _is_inequality(condition) and isinstance(condition.lhs, Add):
        lhs = condition.lhs
        constant, rest = lhs.as_independent(*lhs.free_symbols, as_Add=True)
        if constant != 0:
            return _relation(condition, rest, add(condition.rhs, mul(Integer(-1), constant)))
    return condition


def _is_real(e: Expr, assumptions: Assumptions) -> bool:
    """Whether ``e`` is real, by its flags or under the assumptions."""
    return e.is_extended_real is True or ask(element(e, Reals), assumptions) is True


def _real_part(e: Expr, assumptions: Assumptions) -> Expr:
    """``re(e)`` with the parameters the assumptions make real taken as
    such: ``re(s/k + I)`` stays ``re(s/k)`` for plain symbols, and is
    ``s/k`` under ``s > 0`` and ``k > 0`` (the bug: the condition
    ``Abs(arg(s/k + I)) < pi/2`` of the periodic Laplace rule stayed
    undecided under those assumptions)."""
    real_part = re(e)
    if not real_part.has(Re, Im):
        return real_part
    replacement: dict[Basic, Basic] = {}
    back: dict[Basic, Basic] = {}
    for p in sorted(e.free_symbols, key=lambda s: s.name):
        if p.is_extended_real or ask(element(p, Reals), assumptions):
            d = Dummy(p.name, real=True)
            replacement[p] = d
            back[d] = p
    if not replacement:
        return real_part
    substituted = e._xreplace_evaluated(replacement)
    if not isinstance(substituted, Expr):
        return real_part
    restored = re(substituted)._xreplace_evaluated(back)
    return restored if isinstance(restored, Expr) else real_part


def real_form(condition: Boolean, assumptions: Assumptions) -> Boolean:
    """A condition on the argument of a parameter, ``Abs(arg(e)) < c``
    (also non-strict), rewritten for a real ``e`` under the assumptions:
    ``e > 0`` when ``c <= pi`` (the argument of a negative number is
    ``pi``) and true when ``c > pi``; other conditions are returned
    unchanged.

    >>> from sympy_extras.core import symbols, pi
    >>> from sympy_extras.functions import Abs, arg
    >>> from sympy_extras.sets import Reals
    >>> from sympy_extras.assumptions.facts import element
    >>> from sympy_extras.integrals.conditions import real_form
    >>> a = symbols('a')
    >>> real_form(Abs(arg(a)) < pi/2, element(a, Reals)), real_form(Abs(arg(a)) < 2*pi, None)
    (a > 0, True)
    """
    if not (isinstance(condition, (Lt, Le)) and isinstance(condition.lhs, Abs)
            and isinstance(condition.lhs.args[0], Arg)):
        return condition
    inner = condition.lhs.args[0].args[0]
    if not isinstance(inner, Expr):
        return condition
    e = inner
    bound = condition.rhs
    if bound.is_positive and positive(e, assumptions):
        return true
    if bound.is_positive and (add(bound, mul(Integer(-1), pi)).is_positive
                              or (bound == pi and isinstance(condition, Le))):
        # the argument of any number lies in (-pi, pi]
        return true
    if bound == pi and isinstance(e, Pow) and e.exp == 2 and not _is_real(e.base, assumptions):
        # |arg w**2| < pi unless w**2 <= 0, that is unless w is imaginary:
        # decided by the sign of the real part
        real_part = _real_part(e.base, assumptions)
        if not real_part.has(Re, Im) and (ask(real_part > 0, assumptions) is True
                                          or ask(real_part < 0, assumptions) is True):
            return true
    if bound == mul(pi, Rational(1, 2)) and not _is_real(e, assumptions):
        # |arg e| < pi/2 is Re e > 0: decided when the real part is an
        # expression in real parameters (a - I b with a > 0)
        real_part = _real_part(e, assumptions)
        if not real_part.has(Re, Im):
            verdict = ask(real_part > 0, assumptions)
            if verdict is True:
                return true
            if verdict is False:
                return false
    if _is_real(e, assumptions):
        excess = add(bound, mul(Integer(-1), pi))
        if bound.is_positive and excess.is_positive:
            return true
        if bound.is_positive and excess.is_nonpositive:
            return e > 0
    return condition


def positive(e: Expr, assumptions: Assumptions) -> bool:
    """Whether ``e`` is positive under the assumptions, factor by factor:
    a product of positive factors, of absolute values (of the nonzero
    scales of the kernels) and of even powers of real numbers.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import Abs
    >>> from sympy_extras.integrals.conditions import positive
    >>> a, b = symbols('a b')
    >>> positive(a*Abs(b), a > 0), positive(a, None)
    (True, False)
    """
    if isinstance(e, Mul):
        return all(positive(factor, assumptions) for factor in e.args)
    if isinstance(e, Pow):
        if positive(e.base, assumptions):
            return True
        return e.exp.is_even is True and _is_real(e.base, assumptions)
    if isinstance(e, Abs):
        return True
    if e.is_positive is True:
        return True
    if e.is_extended_real is False:
        return False
    return ask(e > 0, assumptions) is True


def _flagged(s: Symbol) -> bool:
    """Whether the symbol carries flags (SymPy's ``assumptions0`` is not
    empty: ``commutative`` is one, so every symbol does)."""
    return bool(s.assumptions0)


def plain_symbols(symbols: Iterable[Symbol]) -> dict[Symbol, Symbol]:
    """Each symbol carrying flags (``positive=True``, ...) mapped to a plain
    symbol of the same name: a relation on a flagged symbol evaluates to
    ``true`` at once, so the solvers reading statements
    (``find_instance``) must be given the flags as statements on plain
    symbols, see :func:`symbol_facts`."""
    return {s: Dummy(s.name) for s in symbols if _flagged(s)}


def symbol_facts(symbols: Iterable[Symbol], plain: dict[Symbol, Symbol]) -> list[Boolean]:
    """The assumptions carried by the flags of the symbols as statements on
    their plain counterparts.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.integrals.conditions import plain_symbols, symbol_facts
    >>> n, x = Symbol('n', integer=True, positive=True), Symbol('x')
    >>> symbol_facts([n, x], plain_symbols([n]))
    [_n > 0, Contains(_n, Integers)]
    """
    facts: list[Boolean] = []
    for s in symbols:
        p = plain.get(s, s)
        if s.is_positive:
            facts.append(p > 0)
        elif s.is_negative:
            facts.append(p < 0)
        elif s.is_nonnegative:
            facts.append(p >= 0)
        elif s.is_nonpositive:
            facts.append(p <= 0)
        elif s.is_nonzero:
            facts.append(ne(p, 0))
        if s.is_integer:
            facts.append(element(p, Integers))
        elif s.is_real:
            facts.append(element(p, Reals))
    return facts


def _items(assumptions: Assumptions) -> list[Boolean]:
    """The assumptions as a list of statements."""
    if assumptions is None:
        return []
    if isinstance(assumptions, (Basic, bool)):
        return [to_boolean(assumptions)]
    return [to_boolean(a) for a in assumptions]


def _on_plain(statement: Boolean, plain: dict[Symbol, Symbol]) -> Boolean:
    rule: dict[Basic, Basic] = {s: p for s, p in plain.items()}
    replaced = statement.xreplace(rule)
    return replaced if isinstance(replaced, Boolean) else statement


def with_symbol_facts(assumptions: Assumptions, symbols: Iterable[Symbol],
                      plain: dict[Symbol, Symbol]) -> list[Boolean]:
    """The assumptions (on the plain symbols) together with the facts of
    the symbol flags."""
    facts = symbol_facts(symbols, plain)
    facts.extend(_on_plain(a, plain) for a in _items(assumptions))
    return facts


def _random_value(rng: random.Random) -> Expr:
    return Rational(rng.randint(3, 19), 4)


def sample_values(symbols: Sequence[Symbol], assumptions: Assumptions,
                  rng: random.Random) -> Optional[dict[Symbol, Expr]]:
    """Values of the symbols satisfying the assumptions and their flags:
    an instance found by the SAT solver, moved off the boundary at random
    when the moved point still satisfies them, or random values when
    there is nothing to satisfy; ``None`` when there is no instance.

    >>> import random
    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.conditions import sample_values
    >>> a, b = symbols('a b')
    >>> point = sample_values([a, b], [a > 1, b < a], random.Random(1))
    >>> point is not None and point[a] > 1 and point[b] < point[a]
    True
    """
    if not symbols:
        return {}
    plain = plain_symbols(symbols)
    variables = [plain.get(s, s) for s in symbols]
    facts = Facts(with_symbol_facts(assumptions, symbols, plain), symbols=variables)
    if isinstance(facts.formula, BooleanTrue):
        return {s: _random_value(rng) for s in symbols}
    instances = find_instance(facts.formula, variables)
    if not instances:
        # irrational bounds (x > -sqrt(2)/2, a cell of a region) are beyond
        # the solver: the same region shrunk to rational bounds
        given = _items(assumptions)
        rationalised = [_rational_relation(a) for a in given]
        if rationalised == given:
            return None
        facts = Facts(with_symbol_facts(rationalised, symbols, plain), symbols=variables)
        instances = find_instance(facts.formula, variables)
        if not instances:
            return None
    witness = instances[0]
    values: dict[Symbol, Expr] = {}
    for s in symbols:
        p = plain.get(s, s)
        found = witness.get(p)
        values[s] = found if found is not None else _random_value(rng)
    # up, and then down: a witness on the upper boundary of the region
    # (y = 0 for y <= 0) cannot be moved up (the bug: the branch for
    # y <= 0 of a Piecewise value was checked at y = 0 only, where it
    # cannot be evaluated, and a wrong one was accepted)
    for direction in (1, -1):
        for _ in range(4):
            moved = {s: add(v, Rational(direction * rng.randint(1, 9), 8)) for s, v in values.items()}
            rule: dict[Basic, Basic] = {plain.get(s, s): v for s, v in moved.items()}
            instance = facts.formula._xreplace_evaluated(rule)
            if isinstance(instance, Boolean) and ask(instance) is True:
                return moved
    return values


def _rational_relation(relation: Boolean) -> Boolean:
    """An inequality whose constant term is an irrational number replaced
    by the same inequality with the constant approximated by a rational
    on the safe side (``x > -sqrt(2)/2`` becomes ``x > -0.7071...`` plus a
    margin), so that the solvers, which take rational coefficients, can
    find points inside the region; other statements unchanged.

    >>> from sympy_extras.core import symbols, sqrt
    >>> from sympy_extras.core.evalf import evalf
    >>> from sympy_extras.integrals.conditions import _rational_relation
    >>> x = symbols('x')
    >>> r = _rational_relation(x > -sqrt(2)/2)
    >>> evalf(r.rhs, 8), _rational_relation(x < 1)
    (-0.70710678, x < 1)
    """
    if not (isinstance(relation, Relational) and _is_inequality(relation)):
        return relation
    expression = add(relation.lhs, mul(Integer(-1), relation.rhs))
    variables = sorted(expression.free_symbols, key=lambda s: s.name)
    constant, rest = expression.as_independent(*variables, as_Add=True)
    if constant.is_rational is not False or not constant.is_real:
        return relation
    from sympy_extras.polys.polyerrors import PolynomialError
    from sympy_extras.polys.polytools import Poly
    try:
        if not all(c.is_rational for c in Poly(rest, *variables).coeffs()):
            return relation
    except (PolynomialError, ValueError, TypeError):
        return relation
    approximation = number_from_fraction(Fraction(str(evalf(constant, 20))))
    margin = Rational(1, 10**12)
    shifted = add(approximation, mul(Integer(-1), margin)) if isinstance(relation, (Gt, Ge)) \
        else add(approximation, margin)
    return _relation(relation, rest, mul(Integer(-1), shifted))


#: the budget of the second opinion of :func:`numerically_equal`, and the
#: size (in operations) beyond which it is not asked
_EXACT_SECONDS = 1.0
_EXACT_SIZE = 120


def _operations(e: Expr) -> int:
    count = e.count_ops()
    return count.p if isinstance(count, Integer) else _EXACT_SIZE + 1


def _provably_different(a: Expr, b: Expr, assumptions: Assumptions) -> bool:
    """Whether the structure theorem (:mod:`sympy_extras.simplify.structure`)
    proves that ``a - b`` is not identically zero: a nonzero canonical form
    in generators which are independent. ``False`` when it proves nothing
    (equal, not elementary, uncertified, too large, out of time).

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.conditions import _provably_different
    >>> x = symbols('x')
    >>> _provably_different(exp(-10**6*x**2), Integer(0), None), _provably_different(exp(2*x), exp(x)**2, None)
    (True, False)
    """
    from sympy_extras.simplify.structure import ElementaryTower, NotElementary
    difference = add(a, mul(Integer(-1), b))
    if difference == 0 or _operations(difference) > _EXACT_SIZE:
        return False

    def decide_() -> bool:
        tower = ElementaryTower(assumptions)
        try:
            value = tower.element(difference)
        except (NotElementary, ZeroDivisionError):
            return False
        return not tower.vanishes(value) and tower.certifies(value)

    return attempt(decide_, _EXACT_SECONDS, native=True) is True


def numerically_equal(a: Expr, b: Expr, assumptions: Assumptions = None, samples: int = 3) -> bool:
    """Whether ``a`` and ``b`` agree numerically at ``samples`` random
    values of their symbols satisfying the assumptions (see
    :func:`sample_values`); ``False`` when a value cannot be computed.
    Agreement at the samples is evidence, and gets a second opinion: the
    answer is ``False`` when the structure theorem proves the two
    different (``exp(-10**6*x**2)`` and ``0`` agree to every digit at
    points away from the origin).

    >>> from sympy_extras.core import symbols, sqrt, Integer
    >>> from sympy_extras.functions import exp, sin, cos
    >>> from sympy_extras.integrals.conditions import numerically_equal
    >>> x = symbols('x')
    >>> numerically_equal(sin(x)**2 + cos(x)**2, Integer(1)), numerically_equal(sqrt(x**2), x, x < 0)
    (True, False)
    >>> numerically_equal(exp(-10**6*x**2), Integer(0), x > 1)
    False
    """
    if _samples_verdict(a, b, assumptions, true, samples, patient=False) is not True:
        return False
    return not _provably_different(a, b, assumptions)


def numerical_verdict(a: Expr, b: Expr, assumptions: Assumptions = None, condition: Boolean = true,
                      samples: int = 3) -> Optional[bool]:
    """The verdict of a numerical comparison of ``a`` and ``b`` at
    ``samples`` random values of their symbols satisfying the assumptions
    and the condition: ``False`` when the two differ at a sample where
    both have a reliable value (see
    :func:`sympy_extras._numeric.reliable_value`), ``True`` when they
    agree at every sample, ``None`` when nothing was checked or a sample
    could not be (no values satisfying the condition were found, a value
    could not be computed or is decided by a rounding error).

    Unlike :func:`numerically_equal`, which answers ``False`` when it
    cannot compare, this tells "different" from "unknown": it is the test
    of a step which is kept unless it is found wrong, as the last
    simplification of a verified value (``marichev.checked_tidy``).

    Examples
    ========

    >>> from sympy_extras.core import symbols, sqrt, I, pi, Integer
    >>> from sympy_extras.functions import log, gamma
    >>> from sympy_extras.integrals.conditions import numerical_verdict
    >>> a = symbols('a')
    >>> numerical_verdict(sqrt(a**2), a, a > 0), numerical_verdict(sqrt(a**2), a, a < 0)
    (True, False)
    >>> numerical_verdict(log(-a), log(a) + I*pi, None, a > 0)
    True
    >>> numerical_verdict(gamma(Integer(0)), gamma(Integer(0)) + 1) is None
    True
    """
    return _samples_verdict(a, b, assumptions, condition, samples, patient=True)


def _samples_verdict(a: Expr, b: Expr, assumptions: Assumptions, condition: Boolean, samples: int,
                     patient: bool) -> Optional[bool]:
    """``False`` at the first sample where ``a`` and ``b`` differ,
    ``None`` when a sample could not be taken or compared (at the first
    one unless ``patient``, which goes on looking for a difference),
    ``True`` when they agree at every sample."""
    symbols = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    rng = random.Random(str((a, b)))
    verdict: Optional[bool] = True
    for _ in range(samples if symbols else 1):
        values = _sample_under(symbols, assumptions, condition, rng)
        outcome = None if values is None else _agree_at(a, b, values)
        if outcome is False:
            return False
        if outcome is None:
            if not patient or values is None:
                return None
            verdict = None
    return verdict


def _sample_under(symbols: Sequence[Symbol], assumptions: Assumptions, condition: Boolean,
                  rng: random.Random) -> Optional[dict[Symbol, Expr]]:
    """Values of the symbols satisfying the assumptions and the condition:
    an instance of both when the solver finds one, otherwise (the
    condition is beyond it: ``re(a) > 0``, ``Abs(arg(a)) < pi/2``) values
    satisfying the assumptions at which the condition evaluates to true."""
    if isinstance(condition, BooleanTrue):
        return sample_values(symbols, assumptions, rng)
    if isinstance(condition, BooleanFalse):
        return None
    together = sample_values(symbols, [condition] + _items(assumptions), rng)
    if together is not None:
        return together
    for _ in range(4):
        values = sample_values(symbols, assumptions, rng)
        if values is None:
            return None
        rule: dict[Basic, Basic] = {s: v for s, v in values.items()}
        if isinstance(condition._xreplace_evaluated(rule), BooleanTrue):
            return values
    return None


def _not_finite(e: Expr) -> bool:
    return e.has(NaN, ComplexInfinity, Infinity, NegativeInfinity)


def _agree_at(a: Expr, b: Expr, values: dict[Symbol, Expr]) -> Optional[bool]:
    """Whether ``a`` and ``b`` have the same value at the sample; ``None``
    when one of them has no reliable finite value there."""
    # values which a rounding error decides are no evidence (the bug:
    # log(-2*sqrt(2) - 2*sqrt(z)) with z a sum which is exactly zero
    # was found different from log(2*sqrt(2)) + I*pi, which it is, the
    # rounding of z putting the argument below the cut at thirty
    # digits)
    left, right = reliable_value(a, 30, values), reliable_value(b, 30, values)
    if left is None or right is None or _not_finite(left) or _not_finite(right):
        return None
    # mpmath's numbers, not Python's: exp(A*x**r) at a sampled point is
    # 1e300 and more, where abs() of a Python complex overflows
    prec = 30 * 4 + 20
    first, second = numeric_value(left, prec), numeric_value(right, prec)
    if first is None or second is None:
        return None
    with mpmath.workprec(prec):
        gap = abs(first - second)
        scale = 1 + abs(first) + abs(second)
        if not (mpmath.isfinite(gap) and mpmath.isfinite(scale)):
            return None
        return not bool(gap > scale / 10**12)


def combine(values: Iterable[ConditionalValue]) -> ConditionalValue:
    """The sum of the values, holding when every condition does.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.conditions import ConditionalValue, combine
    >>> a = symbols('a')
    >>> combine([ConditionalValue(a, a > 0), ConditionalValue(1, a < 2)])
    ConditionalValue(a + 1, (a > 0) & (a < 2))
    """
    total = ConditionalValue(Integer(0), true)
    for v in values:
        total = total.add(v)
    return total

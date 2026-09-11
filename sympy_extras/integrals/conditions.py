"""Values which hold under a condition on the parameters.

The definite integrators of this package return a
:class:`ConditionalValue`: the value of an integral together with the
condition (an inequality on the parameters, ``a > 0``, or a conjunction of
them) under which the integral converges and the formula holds. The
conditions are decided against the assumptions by
:func:`sympy_extras.assumptions.ask` and what remains undecided is
reported in a :class:`~sympy.functions.elementary.piecewise.Piecewise`.

Examples
========

>>> from sympy.abc import a
>>> from sympy_extras.integrals.conditions import ConditionalValue
>>> v = ConditionalValue(1/a, a > 0)
>>> v
ConditionalValue(1/a, a > 0)
>>> v.scaled(2)
ConditionalValue(2/a, a > 0)
>>> v.add(ConditionalValue(a, a < 1))
ConditionalValue(a + 1/a, (a > 0) & (a < 1))
"""
from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence

from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.numbers import Rational, pi
from sympy.core.symbol import Dummy, Symbol
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.relational import Ge, Gt, Le, Lt, Ne
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import Abs, arg
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And, Boolean, true, false

from sympy_extras._typing import ExprLike, as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import Facts, element
from sympy_extras.assumptions.sat import find_instance

__all__ = ['ConditionalValue', 'decide', 'combine', 'real_form', 'numerically_equal', 'positive', 'canonical',
           'symbol_facts', 'with_symbol_facts', 'plain_symbols', 'sample_values']


class ConditionalValue:
    """A value with the condition under which it holds.

    Attributes
    ==========

    value : Expr
    condition : Boolean
        ``S.true`` when the value holds unconditionally.
    """

    def __init__(self, value: ExprLike, condition: object = True) -> None:
        self.value: Expr = as_expr(value)
        self.condition: Boolean = as_boolean(condition)

    def __repr__(self) -> str:
        if self.condition is true:
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
        return ConditionalValue(as_expr(factor) * self.value, self.condition)

    def add(self, other: ConditionalValue) -> ConditionalValue:
        """The sum, holding when both conditions do."""
        return ConditionalValue(self.value + other.value, And(self.condition, other.condition))

    def with_condition(self, condition: object) -> ConditionalValue:
        """The same value under ``condition`` as well."""
        return ConditionalValue(self.value, And(self.condition, as_boolean(condition)))

    def as_piecewise(self, otherwise: ExprLike) -> Expr:
        """The value where the condition holds and ``otherwise`` elsewhere;
        the value alone when the condition is ``S.true``."""
        if self.condition is true:
            return self.value
        return as_expr(Piecewise((self.value, self.condition), (as_expr(otherwise), True)))


def decide(condition: Boolean, assumptions: Assumptions) -> Optional[Boolean]:
    """``condition`` with the conjuncts settled by the assumptions removed:
    ``S.true`` when it follows from them, ``None`` when it is refuted (one
    conjunct is false under the assumptions), and otherwise the conjunction
    of what stays undecided."""
    if condition is true:
        return true
    if condition is false:
        return None
    parts: Sequence[Boolean] = condition.args if isinstance(condition, And) else (condition,)
    remaining: list[Boolean] = []
    for part in parts:
        part_ = canonical(real_form(as_boolean(part), assumptions))
        verdict = ask(part_, assumptions)
        if verdict is False:
            return None
        if verdict is None:
            remaining.append(part_)
    return as_boolean(And(*remaining))


def canonical(condition: Boolean) -> Boolean:
    """A relation with the constant term of its left side moved to the
    right: ``k + 1 > 0`` becomes ``k > -1``; other conditions are
    returned unchanged.

    >>> from sympy.abc import k
    >>> from sympy_extras.integrals.conditions import canonical
    >>> canonical(k + 1 < 1)
    k < 0
    """
    if isinstance(condition, (Lt, Le, Gt, Ge)) and isinstance(condition.lhs, Add):
        constant, rest = condition.lhs.as_independent(*condition.lhs.free_symbols, as_Add=True)
        constant_ = as_expr(constant)
        if constant_ != 0:
            return as_boolean(condition.func(as_expr(rest), as_expr(condition.rhs) - constant_))
    return condition


def real_form(condition: Boolean, assumptions: Assumptions) -> Boolean:
    """A condition on the argument of a parameter, ``Abs(arg(e)) < c``
    (also non-strict), rewritten for a real ``e`` under the assumptions:
    ``e > 0`` when ``c <= pi`` (the argument of a negative number is
    ``pi``) and true when ``c > pi``; other conditions are returned
    unchanged."""
    if isinstance(condition, (Lt, Le)) and isinstance(condition.lhs, Abs) \
            and isinstance(condition.lhs.args[0], arg):
        e = as_expr(condition.lhs.args[0].args[0])
        bound = as_expr(condition.rhs)
        if bound.is_positive and positive(e, assumptions):
            return true
        if e.is_extended_real or ask(element(e, S.Reals), assumptions):
            if bound.is_positive and (bound - pi).is_positive:
                return true
            if bound.is_positive and (bound - pi).is_nonpositive:
                return as_boolean(e > 0)
    return condition


def positive(e: Expr, assumptions: Assumptions) -> bool:
    """Whether ``e`` is positive under the assumptions, factor by factor:
    a product of positive factors, of absolute values (of the nonzero
    scales of the kernels) and of even powers of real numbers."""
    if isinstance(e, Mul):
        return all(positive(as_expr(factor), assumptions) for factor in e.args)
    if isinstance(e, Pow):
        base, exponent = as_expr(e.base), as_expr(e.exp)
        if positive(base, assumptions):
            return True
        return exponent.is_even is True and (base.is_extended_real is True
                                              or ask(element(base, S.Reals), assumptions) is True)
    if isinstance(e, Abs):
        return True
    if e.is_positive is True:
        return True
    return ask(as_boolean(e > 0), assumptions) is True


def plain_symbols(symbols: Iterable[Symbol]) -> dict[Symbol, Symbol]:
    """Each symbol carrying SymPy flags (``positive=True``, ...) mapped to
    a plain symbol of the same name: a relation on a flagged symbol
    evaluates to ``True`` at once, so the solvers reading statements
    (``find_instance``) must be given the flags as statements on plain
    symbols, see :func:`symbol_facts`."""
    return {s: Dummy(s.name) for s in symbols if s.assumptions0}


def symbol_facts(symbols: Iterable[Symbol], plain: dict[Symbol, Symbol]) -> list[Boolean]:
    """The assumptions carried by the SymPy flags of the symbols as
    statements on their plain counterparts."""
    facts: list[Boolean] = []
    for s in symbols:
        p = plain.get(s, s)
        if s.is_positive:
            facts.append(as_boolean(p > 0))
        elif s.is_negative:
            facts.append(as_boolean(p < 0))
        elif s.is_nonnegative:
            facts.append(as_boolean(p >= 0))
        elif s.is_nonpositive:
            facts.append(as_boolean(p <= 0))
        elif s.is_nonzero:
            facts.append(as_boolean(Ne(p, 0)))
        if s.is_integer:
            facts.append(element(p, S.Integers))
        elif s.is_real:
            facts.append(element(p, S.Reals))
    return facts


def with_symbol_facts(assumptions: Assumptions, symbols: Iterable[Symbol],
                      plain: dict[Symbol, Symbol]) -> list[Boolean]:
    """The assumptions (on the plain symbols) together with the facts of
    the symbol flags."""
    facts = symbol_facts(symbols, plain)
    if isinstance(assumptions, (Basic, bool)):
        facts.append(as_boolean(as_boolean(assumptions).xreplace(plain)))
    elif assumptions is not None:
        facts.extend(as_boolean(as_boolean(a).xreplace(plain)) for a in assumptions)
    return facts


def sample_values(symbols: Sequence[Symbol], assumptions: Assumptions,
                  rng: random.Random) -> Optional[dict[Symbol, Expr]]:
    """Values of the symbols satisfying the assumptions and their flags:
    an instance found by the SAT solver, moved off the boundary at random
    when the moved point still satisfies them, or random values when
    there is nothing to satisfy; ``None`` when there is no instance."""
    if not symbols:
        return {}
    plain = plain_symbols(symbols)
    facts = Facts(with_symbol_facts(assumptions, symbols, plain), symbols=[plain.get(s, s) for s in symbols])
    if facts.formula is true:
        return {s: Rational(rng.randint(3, 19), 4) for s in symbols}
    instances = find_instance(facts.formula, [plain.get(s, s) for s in symbols])
    if not instances:
        return None
    witness = instances[0]
    values: dict[Symbol, Expr] = {}
    for s in symbols:
        p = plain.get(s, s)
        values[s] = as_expr(witness[p]) if p in witness else Rational(rng.randint(3, 19), 4)
    for _ in range(4):
        moved = {s: as_expr(v + Rational(rng.randint(1, 9), 8)) for s, v in values.items()}
        if ask(as_boolean(facts.formula.xreplace({plain.get(s, s): v for s, v in moved.items()}))) is True:
            return moved
    return values


def numerically_equal(a: Expr, b: Expr, assumptions: Assumptions = None, samples: int = 3) -> bool:
    """Whether ``a`` and ``b`` agree numerically at ``samples`` random
    values of their symbols satisfying the assumptions (see
    :func:`sample_values`); ``False`` when a value cannot be computed."""
    symbols = sorted_symbols(free_symbols(a) | free_symbols(b))
    rng = random.Random(str((a, b)))
    for _ in range(samples if symbols else 1):
        values = sample_values(symbols, assumptions, rng)
        if values is None:
            return False
        try:
            left = complex(as_expr(a.xreplace(values)).evalf(30))
            right = complex(as_expr(b.xreplace(values)).evalf(30))
        except (TypeError, ValueError):
            return False
        if abs(left - right) > 1e-12 * (1 + abs(left) + abs(right)):
            return False
    return True


def combine(values: Iterable[ConditionalValue]) -> ConditionalValue:
    """The sum of the values, holding when every condition does."""
    total = ConditionalValue(S.Zero, true)
    for v in values:
        total = total.add(v)
    return total

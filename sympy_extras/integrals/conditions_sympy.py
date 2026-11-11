"""The SymPy face of the values which hold under a condition
(:mod:`sympy_extras.integrals.conditions`), for the integrators which are
still written on SymPy.

The decisions and the numerical comparisons (:func:`decide`,
:func:`real_form`, :func:`canonical`, :func:`positive`,
:func:`sample_values`, :func:`numerically_equal`,
:func:`numerical_verdict`) take SymPy objects, converted by
:mod:`sympy_extras.interop` around the native functions.
:class:`ConditionalValue` is the same container on SymPy objects (its
arithmetic is SymPy's, with :meth:`ConditionalValue.to_core` and
:meth:`ConditionalValue.from_core` to the native class), and the
translation of SymPy's symbol flags (:func:`plain_symbols`,
:func:`symbol_facts`, :func:`with_symbol_facts`) stays on SymPy: the
plain symbols are SymPy ``Dummy`` objects the integrators substitute in
their SymPy expressions.

Examples
========

>>> from sympy.abc import a
>>> from sympy_extras.integrals.conditions_sympy import ConditionalValue, numerically_equal
>>> v = ConditionalValue(1/a, a > 0)
>>> v
ConditionalValue(1/a, a > 0)
>>> v.scaled(2)
ConditionalValue(2/a, a > 0)
>>> v.add(ConditionalValue(a, a < 1))
ConditionalValue(a + 1/a, (a > 0) & (a < 1))
>>> numerically_equal(abs(a), a, a > 0)
True
"""
from __future__ import annotations

import random
from typing import Callable, Final, Iterable, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And, Boolean, Or, true

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing import ExprLike, as_boolean, as_expr, as_symbol
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions, core_boolean, sympy_boolean
from sympy_extras.assumptions.facts_sympy import element
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import conditions as native

__all__ = ['ConditionalValue', 'decide', 'combine', 'real_form', 'numerically_equal', 'numerical_verdict',
           'positive', 'canonical', 'symbol_facts', 'with_symbol_facts', 'plain_symbols', 'sample_values']


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


class ConditionalValue:
    """A value with the condition under which it holds, on SymPy objects
    (:class:`sympy_extras.integrals.conditions.ConditionalValue` on the
    core).

    Attributes
    ==========

    value : Expr
    condition : Boolean
        ``S.true`` when the value holds unconditionally.
    """

    __slots__ = ('value', 'condition')

    def __init__(self, value: ExprLike, condition: object = True) -> None:
        self.value: Final[Expr] = as_expr(value)
        self.condition: Final[Boolean] = as_boolean(condition)

    def __repr__(self) -> str:
        if self.condition is true:
            return "ConditionalValue(%s)" % (self.value,)
        return "ConditionalValue(%s, %s)" % (self.value, self.condition)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, ConditionalValue) and self.value == other.value
                and self.condition == other.condition)

    def __hash__(self) -> int:
        return hash((self.value, self.condition))

    def to_core(self) -> native.ConditionalValue:
        """The same value on the objects of the core."""
        return native.ConditionalValue(core_expr(self.value), core_boolean(self.condition))

    @staticmethod
    def from_core(v: native.ConditionalValue) -> ConditionalValue:
        """A value of the core on SymPy objects."""
        return ConditionalValue(sympy_expr(v.value), sympy_boolean(v.condition))

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
        the value alone when the condition is ``S.true``. A value which is
        itself a ``Piecewise`` of cases, the condition being the
        disjunction of theirs (see :meth:`cases`), keeps its branches,
        with ``otherwise`` as the last one.

        >>> from sympy import symbols, Piecewise, Or
        >>> from sympy_extras.integrals.conditions_sympy import ConditionalValue
        >>> a = symbols('a')
        >>> ConditionalValue(Piecewise((a, a > 1), (-a, a < 0)), Or(a > 1, a < 0)).as_piecewise(0)
        Piecewise((a, a > 1), (-a, a < 0), (0, True))
        """
        if self.condition is true:
            return self.value
        branches = self.cases()
        if branches is not None:
            return as_expr(Piecewise(*[(v, c) for v, c in branches], (as_expr(otherwise), True)))
        return as_expr(Piecewise((self.value, self.condition), (as_expr(otherwise), True)))

    def cases(self) -> Optional[list[tuple[Expr, Boolean]]]:
        """The ``(value, condition)`` pairs of a value which is a
        ``Piecewise`` of cases whose conditions the condition is the
        disjunction of; ``None`` for any other value."""
        if not isinstance(self.value, Piecewise):
            return None
        pairs = [(as_expr(pair.args[0]), as_boolean(pair.args[1])) for pair in self.value.args]
        if any(condition is true for _, condition in pairs) or self.condition != Or(*[c for _, c in pairs]):
            return None
        return pairs

    def mapped(self, function: Callable[[Expr, Boolean], Expr]) -> ConditionalValue:
        """The value with ``function`` applied to it and its condition,
        case by case for a value which is a ``Piecewise`` of cases."""
        branches = self.cases()
        if branches is None:
            return ConditionalValue(function(self.value, self.condition), self.condition)
        return ConditionalValue(Piecewise(*[(function(v, c), c) for v, c in branches], evaluate=False), self.condition)


def combine(values: Iterable[ConditionalValue]) -> ConditionalValue:
    """The sum of the values, holding when every condition does."""
    total = ConditionalValue(S.Zero, true)
    for v in values:
        total = total.add(v)
    return total


def decide(condition: Boolean, assumptions: Assumptions) -> Optional[Boolean]:
    """:func:`sympy_extras.integrals.conditions.decide` on SymPy objects:
    ``condition`` with the conjuncts settled by the assumptions removed,
    ``S.true`` when it follows from them, ``None`` when it is refuted.

    >>> from sympy.abc import a, b
    >>> from sympy_extras.integrals.conditions_sympy import decide
    >>> decide((a > 0) & (b > 1), a > 1), decide(a < 0, a > 1)
    (b > 1, None)
    """
    decided = native.decide(core_boolean(condition), core_assumptions(assumptions))
    return None if decided is None else sympy_boolean(decided)


def canonical(condition: Boolean) -> Boolean:
    """:func:`sympy_extras.integrals.conditions.canonical` on SymPy objects:
    the constant term of the left side of a relation moved to the right.

    >>> from sympy.abc import k
    >>> from sympy_extras.integrals.conditions_sympy import canonical
    >>> canonical(k + 1 < 1)
    k < 0
    """
    return sympy_boolean(native.canonical(core_boolean(condition)))


def real_form(condition: Boolean, assumptions: Assumptions) -> Boolean:
    """:func:`sympy_extras.integrals.conditions.real_form` on SymPy
    objects: ``Abs(arg(e)) < c`` rewritten for a real ``e``.

    >>> from sympy import Abs, arg, pi, S
    >>> from sympy.abc import a
    >>> from sympy_extras.assumptions.facts_sympy import element
    >>> from sympy_extras.integrals.conditions_sympy import real_form
    >>> real_form(Abs(arg(a)) < pi/2, element(a, S.Reals))
    a > 0
    """
    return sympy_boolean(native.real_form(core_boolean(condition), core_assumptions(assumptions)))


def positive(e: Expr, assumptions: Assumptions) -> bool:
    """:func:`sympy_extras.integrals.conditions.positive` on SymPy objects.

    >>> from sympy import Abs
    >>> from sympy.abc import a, b
    >>> from sympy_extras.integrals.conditions_sympy import positive
    >>> positive(a*Abs(b), a > 0)
    True
    """
    return native.positive(core_expr(e), core_assumptions(assumptions))


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


def _items(assumptions: Assumptions) -> list[Boolean]:
    """The assumptions as a list of statements."""
    if assumptions is None:
        return []
    if isinstance(assumptions, (Boolean, bool)):
        return [as_boolean(assumptions)]
    return [as_boolean(a) for a in assumptions]


def with_symbol_facts(assumptions: Assumptions, symbols: Iterable[Symbol],
                      plain: dict[Symbol, Symbol]) -> list[Boolean]:
    """The assumptions (on the plain symbols) together with the facts of
    the symbol flags."""
    facts = symbol_facts(symbols, plain)
    facts.extend(as_boolean(a.xreplace(plain)) for a in _items(assumptions))
    return facts


def sample_values(symbols: Sequence[Symbol], assumptions: Assumptions,
                  rng: random.Random) -> Optional[dict[Symbol, Expr]]:
    """:func:`sympy_extras.integrals.conditions.sample_values` on SymPy
    objects: values of the symbols satisfying the assumptions and their
    flags, ``None`` when there is no instance.

    >>> import random
    >>> from sympy.abc import a, b
    >>> from sympy_extras.integrals.conditions_sympy import sample_values
    >>> point = sample_values([a, b], [a > 1, b < a], random.Random(1))
    >>> point is not None and point[a] > 1 and point[b] < point[a]
    True
    """
    found = native.sample_values([_core_symbol(s) for s in symbols], core_assumptions(assumptions), rng)
    if found is None:
        return None
    return {as_symbol(to_sympy(s)): sympy_expr(v) for s, v in found.items()}


def _rational_relation(relation: Boolean) -> Boolean:
    """:func:`sympy_extras.integrals.conditions._rational_relation` on SymPy
    objects: an irrational constant of an inequality replaced by a
    rational on the safe side.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.conditions_sympy import _rational_relation
    >>> x = symbols('x')
    >>> _rational_relation(x > -sqrt(2)/2).rhs.evalf(8), _rational_relation(x < 1)
    (-0.70710678, x < 1)
    """
    return sympy_boolean(native._rational_relation(core_boolean(relation)))


def _provably_different(a: Expr, b: Expr, assumptions: Assumptions) -> bool:
    """:func:`sympy_extras.integrals.conditions._provably_different` on
    SymPy objects: whether the structure theorem proves ``a - b`` not
    identically zero."""
    return native._provably_different(core_expr(a), core_expr(b), core_assumptions(assumptions))


def numerically_equal(a: Expr, b: Expr, assumptions: Assumptions = None, samples: int = 3) -> bool:
    """:func:`sympy_extras.integrals.conditions.numerically_equal` on SymPy
    objects: whether ``a`` and ``b`` agree numerically at ``samples``
    random values of their symbols satisfying the assumptions, and the
    structure theorem does not prove them different.

    >>> from sympy import exp, sin, cos, sqrt
    >>> from sympy.abc import x
    >>> from sympy_extras.integrals.conditions_sympy import numerically_equal
    >>> numerically_equal(sin(x)**2 + cos(x)**2, 1), numerically_equal(sqrt(x**2), x, x < 0)
    (True, False)
    >>> numerically_equal(exp(-10**6*x**2), 0, x > 1)
    False
    """
    return native.numerically_equal(core_expr(a), core_expr(b), core_assumptions(assumptions), samples)


def numerical_verdict(a: Expr, b: Expr, assumptions: Assumptions = None, condition: Boolean = true,
                      samples: int = 3) -> Optional[bool]:
    """:func:`sympy_extras.integrals.conditions.numerical_verdict` on SymPy
    objects: ``False`` when ``a`` and ``b`` differ at a sample where both
    have a reliable value, ``True`` when they agree at every sample,
    ``None`` when nothing was checked.

    >>> from sympy import symbols, log, sqrt, I, pi, gamma
    >>> from sympy_extras.integrals.conditions_sympy import numerical_verdict
    >>> a = symbols('a')
    >>> numerical_verdict(sqrt(a**2), a, a > 0), numerical_verdict(sqrt(a**2), a, a < 0)
    (True, False)
    >>> numerical_verdict(log(-a), log(a) + I*pi, None, a > 0)
    True
    >>> numerical_verdict(gamma(a - a), gamma(a - a) + 1) is None
    True
    """
    converted: CoreBoolean = core_boolean(condition)
    return native.numerical_verdict(core_expr(a), core_expr(b), core_assumptions(assumptions), converted, samples)

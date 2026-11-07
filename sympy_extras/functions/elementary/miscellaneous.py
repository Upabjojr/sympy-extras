"""``Min`` and ``Max`` on the typed core: the unevaluated nodes (any
number of arguments, kept as given) and the evaluating ``min_`` and
``max_`` (named with an underscore since ``min`` and ``max`` are
builtins).

Ported from SymPy's ``MinMaxBase``, ``Min`` and ``Max``: nested nodes
flattened, the identity (``-oo`` for ``max_``) dropped and the zero
(``oo``) absorbing, redundant arguments removed (``Min(a, Max(b, Min(a,
c)))`` is ``Min(a, Max(b, c))``), common arguments factored
(``Min(Max(x, y), Max(x, z)) = Max(x, Min(y, z))``), and arguments
whose order is decided (from the facts of their difference) reduced to
the extreme one. The derivative is SymPy's, with ``Heaviside``.

Examples
========

>>> from sympy_extras.core import Symbol, pi, oo
>>> from sympy_extras.functions.elementary.miscellaneous import Max, Min, max_, min_
>>> x, y = Symbol('x'), Symbol('y')
>>> p, n = Symbol('p', positive=True), Symbol('n', negative=True)
>>> max_(1, 3, 2), min_(pi, 3), max_(x, -oo), min_(x, oo, y), max_(n, p, x)
(3, 3, x, Min(x, y), Max(p, x))
>>> min_(x, max_(y, min_(x, 1))), Max(1, 2), max_(x, max_(y, x))
(Min(x, Max(1, y)), Max(1, 2), Max(x, y))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/miscellaneous.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions (SymPy's
# evaluate=False is _node), strict typing, the comparisons decided from
# the facts of the difference.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Sequence, Union

import mpmath

from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction, diff
from sympy_extras.core.mul import mul
from sympy_extras.core.add import add, sub
from sympy_extras.core.numbers import Integer, NaN, NegativeInfinity, oo
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Symbol

__all__ = ['Max', 'Min', 'max_', 'min_']

ZERO = Integer(0)

#: the fact names SymPy's MinMaxBase gives when all the arguments agree
_AGREEING = ('algebraic', 'antihermitian', 'commutative', 'complex', 'composite', 'even', 'finite', 'hermitian',
             'imaginary', 'infinite', 'integer', 'irrational', 'noninteger', 'nonpositive', 'nonzero', 'odd',
             'polar', 'prime', 'rational', 'real', 'extended_real', 'transcendental', 'zero')


def _ge(x: Expr, y: Expr) -> Optional[bool]:
    """``x >= y`` when decided (SymPy's ``is_ge`` on the difference)."""
    if x == y:
        return True
    difference = sub(x, y)
    if isinstance(difference, NaN):
        return None
    return difference.is_extended_nonnegative


class _ShortCircuit(Exception):
    """The zero of the lattice is an argument (SymPy's ``ShortCircuit``)."""


Connection = Union[bool, 'type[MinMaxBase]']


# PORTED FROM SYMPY: sympy/functions/elementary/miscellaneous.py:MinMaxBase
class MinMaxBase(DefinedFunction):
    """The common base of :class:`Max` and :class:`Min`."""

    variadic = True
    nargs = frozenset()
    call_nargs = frozenset()

    @classmethod
    def zero(cls) -> Expr:
        """The absorbing argument (``oo`` for ``Max``)."""
        raise NotImplementedError

    @classmethod
    def identity(cls) -> Expr:
        """The neutral argument (``-oo`` for ``Max``)."""
        raise NotImplementedError

    @classmethod
    def opposite(cls) -> type[MinMaxBase]:
        raise NotImplementedError

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        try:
            filtered = _unique(cls._new_args_filter(args))
        except _ShortCircuit:
            return cls.zero()
        collapsed = cls._collapse_arguments(filtered)
        zeros = cls._find_localzeros(collapsed)
        return cls._node(zeros)

    @classmethod
    def _node(cls, args: Iterable[Expr]) -> Expr:
        """SymPy's ``cls(*args, evaluate=False)``: the arguments without
        repetitions, in canonical order; the identity for none, the
        argument itself for one."""
        unique = sorted(_unique(args), key=default_sort_key)
        if not unique:
            return cls.identity()
        if len(unique) == 1:
            return unique[0]
        return cls(*unique)

    @classmethod
    def _new_args_filter(cls, args: Iterable[Expr]) -> Iterator[Expr]:
        for arg in args:
            if arg.is_extended_real is False or (arg.is_number and not arg.is_comparable):
                raise ValueError("The argument '%s' is not comparable." % (arg,))
            if arg == cls.zero():
                raise _ShortCircuit
            if arg == cls.identity():
                continue
            if isinstance(arg, cls):
                yield from (to_expr(a) for a in arg.args)
            else:
                yield arg

    @classmethod
    def _collapse_arguments(cls, given: Sequence[Expr]) -> list[Expr]:
        if not given:
            return []
        args = sorted(given, key=default_sort_key)
        other = cls.opposite()
        if args[0].is_number:
            mins: list[MinMaxBase] = []
            maxs: list[MinMaxBase] = []
            for a in args:
                for node in _walk(a):
                    if to_expr(node.args[0]).is_comparable:
                        (maxs if isinstance(node, Max) else mins).append(node)
            small: Expr = Min.identity()
            for v in mins:
                first = to_expr(v.args[0])
                if first.is_number and _ge(first, small) is False:
                    small = first
            big: Expr = Max.identity()
            for v in maxs:
                first = to_expr(v.args[0])
                if first.is_number and _ge(big, first) is False:
                    big = first
            for a in args:
                if not a.is_number:
                    break
                if cls is Min and _ge(a, small) is False:
                    small = a
                elif cls is Max and _ge(big, a) is False:
                    big = a
            threshold: Optional[Expr] = None
            if cls is Min:
                if small != Min.identity():
                    other, threshold = Max, small
            elif big != Max.identity():
                other, threshold = Min, big
            if threshold is not None:
                for i, a in enumerate(args):
                    if isinstance(a, other):
                        first = to_expr(a.args[0])
                        beyond = _ge(threshold, first) is False if other is Max else _ge(first, threshold) is False
                        if beyond:
                            args[i] = cls.identity()

        def do(ai: Expr, a: Expr) -> Expr:
            if not isinstance(ai, MinMaxBase):
                return ai
            sub_args = [to_expr(b) for b in ai.args]
            if a not in sub_args:
                return type(ai)._node([do(b, a) for b in sub_args])
            if isinstance(ai, cls):
                return type(ai)._node([do(b, a) for b in sub_args if b != a])
            return a

        for i in range(len(args)):
            a = args[i]
            args[i + 1:] = [do(ai, a) for ai in args[i + 1:]]
        if len(args) > 1:
            args = cls._factor_minmax(args, other)
        return args

    @classmethod
    def _factor_minmax(cls, args: list[Expr], other: type[MinMaxBase]) -> list[Expr]:
        other_args = [a for a in args if isinstance(a, other)]
        remaining = [a for a in args if not isinstance(a, other)]
        if not other_args:
            return args
        arg_sets = [set(to_expr(b) for b in a.args) for a in other_args]
        common = set.intersection(*arg_sets)
        if not common:
            return args
        new_other_args = sorted(common, key=default_sort_key)
        differences = [s - common for s in arg_sets]
        if all(differences):
            new_other_args.append(cls._node([other._node(d) for d in differences]))
        return remaining + [other._node(new_other_args)]

    @classmethod
    def _find_localzeros(cls, values: Sequence[Expr]) -> list[Expr]:
        zeros: list[Expr] = []
        for v in values:
            is_new = True
            for z in list(zeros):
                if v is z:
                    is_new = False
                    continue
                connection = cls._is_connected(v, z)
                if connection is not False:
                    is_new = False
                    if connection is True or connection is cls:
                        zeros.remove(z)
                        if v not in zeros:
                            zeros.append(v)
            if is_new and v not in zeros:
                zeros.append(v)
        return zeros

    @classmethod
    def _is_connected(cls, x: Expr, y: Expr) -> Connection:
        """``True`` for equal arguments, ``Max`` when ``x >= y`` is
        decided true (``Min`` when false), ... (SymPy's
        ``_is_connected``), ``False`` when no order is known."""
        from sympy_extras.core.exprtools import factor_terms
        for _ in range(2):
            if x == y:
                return True
            t: type[MinMaxBase] = Max
            f: type[MinMaxBase] = Min
            for op in '><':
                for _j in range(2):
                    if x.is_extended_real is False or y.is_extended_real is False:
                        return False
                    v = _ge(x, y) if op == '>' else _ge(y, x)
                    if v is not None:
                        return t if v else f
                    t, f = f, t
                    x, y = y, x
                x, y = y, x
            x, y = factor_terms(sub(x, y)), ZERO
        return False

    # PORTED FROM SYMPY: sympy/functions/elementary/miscellaneous.py:MinMaxBase._eval_derivative
    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        terms: list[Expr] = []
        for i, a in enumerate(self.args):
            da = diff(to_expr(a), s)
            if da.is_zero:
                continue
            terms.append(mul(self.fdiff(i + 1), da))
        return add(*terms)

    # PORTED FROM SYMPY: sympy/functions/elementary/miscellaneous.py:Max.fdiff, Min.fdiff
    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.functions.special.delta_functions import heaviside
        args = [to_expr(a) for a in self.args]
        i = argindex - 1
        if len(args) == 2:
            this, other = args[i], args[1 - i]
        else:
            this, other = args[i], type(self).apply(*(args[:i] + args[i + 1:]))
        return heaviside(sub(this, other) if isinstance(self, Max) else sub(other, this))

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {}
        sub_facts = [facts_of(to_expr(a)) for a in self.args]
        for name in _AGREEING + ('negative', 'nonnegative', 'positive'):
            values = [s.get(name) for s in sub_facts]
            if all(v is True for v in values):
                facts[name] = True
            elif all(v is False for v in values):
                facts[name] = False
        # the value is one of the arguments, so a fact all of them agree on
        # holds; one positive argument makes Max positive (SymPy's
        # fuzzy_or) only when all are finite: Max(0, x) is oo at x = oo,
        # and Min(0, oo) is not negative although oo is not nonnegative
        if all(s.get('finite') is True for s in sub_facts):
            for name in self._any_names():
                values = [s.get(name) for s in sub_facts]
                if any(v is True for v in values):
                    facts[name] = True
            for name in self._all_names():
                values = [s.get(name) for s in sub_facts]
                if any(v is False for v in values):
                    facts[name] = False
        return facts

    @classmethod
    def _any_names(cls) -> tuple[str, ...]:
        """The facts true when one argument has them (SymPy's fuzzy_or)."""
        raise NotImplementedError

    @classmethod
    def _all_names(cls) -> tuple[str, ...]:
        """The facts true when all arguments have them (fuzzy_and)."""
        raise NotImplementedError

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if any(mpmath.im(v) != 0 for v in values):
            return None
        reals = [mpmath.re(v) for v in values]
        return mpmath.mpc(max(reals) if isinstance(self, Max) else min(reals))


def _unique(args: Iterable[Expr]) -> list[Expr]:
    return list(dict.fromkeys(args))


def _walk(e: Basic) -> Iterator[MinMaxBase]:
    """The ``Min`` and ``Max`` nodes reached from ``e`` through ``Min`` and
    ``Max`` nodes only (SymPy's ``walk(e, Min, Max)``)."""
    if isinstance(e, MinMaxBase):
        yield e
        for a in e.args:
            yield from _walk(a)


# PORTED FROM SYMPY: sympy/functions/elementary/miscellaneous.py:Max
class Max(MinMaxBase):
    """The maximum, unevaluated (``max_`` evaluates)."""

    function_name = 'Max'

    @classmethod
    def zero(cls) -> Expr:
        return oo

    @classmethod
    def identity(cls) -> Expr:
        return NegativeInfinity()

    @classmethod
    def opposite(cls) -> type[MinMaxBase]:
        return Min

    @classmethod
    def _any_names(cls) -> tuple[str, ...]:
        return ('positive', 'nonnegative')

    @classmethod
    def _all_names(cls) -> tuple[str, ...]:
        return ('negative',)


# PORTED FROM SYMPY: sympy/functions/elementary/miscellaneous.py:Min
class Min(MinMaxBase):
    """The minimum, unevaluated (``min_`` evaluates)."""

    function_name = 'Min'

    @classmethod
    def zero(cls) -> Expr:
        return NegativeInfinity()

    @classmethod
    def identity(cls) -> Expr:
        return oo

    @classmethod
    def opposite(cls) -> type[MinMaxBase]:
        return Max

    @classmethod
    def _any_names(cls) -> tuple[str, ...]:
        return ('negative',)

    @classmethod
    def _all_names(cls) -> tuple[str, ...]:
        return ('positive', 'nonnegative')


def max_(*args: ExprLike) -> Expr:
    """The maximum of the arguments, evaluated."""
    return Max.apply(*args)


def min_(*args: ExprLike) -> Expr:
    """The minimum of the arguments, evaluated."""
    return Min.apply(*args)



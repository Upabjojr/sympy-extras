"""Sums and products with limits on the typed core: ``Sum(f, (i, a, b),
...)`` and ``Product(f, (i, a, b), ...)``, unevaluated, with the
constructing functions ``sum_`` and ``product_`` (SymPy's constructors:
nested sums of the same kind merged, the ``Piecewise`` factors free of
the variables kept and the others folded).

Ported from SymPy's ``sympy.concrete.expr_with_limits``,
``sympy.concrete.summations`` and ``sympy.concrete.products``: the bound
variables (``free_symbols``, substitution), the printing. The
evaluation (``doit``) goes through SymPy for the node.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.concrete.expr_with_limits import sum_, product_
>>> i, j, n, x = Symbol('i'), Symbol('j'), Symbol('n'), Symbol('x')
>>> s = sum_(x**i, (i, 0, n))
>>> s, s.free_symbols == {x, n}, s.subs(i, j), s.subs(n, 3)
(Sum(x**i, (i, 0, n)), True, Sum(x**i, (i, 0, n)), Sum(x**i, (i, 0, 3)))
>>> sum_(sum_(i*j, (i, 0, j)), (j, 0, n)), product_(i, (i, 1, n)).doit()
(Sum(i*j, (i, 0, j), (j, 0, n)), factorial(n))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/concrete/expr_with_limits.py (ExprWithLimits,
# _common_new), sympy/concrete/summations.py (Sum),
# sympy/concrete/products.py (Product) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the limits are (variable, lower, upper) triples (Sum and
# Product require them); the classes are unevaluated and sum_/product_
# construct as SymPy's __new__; doit goes through SymPy.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Optional, Sequence, Union

from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.printing import StrPrinter
from sympy_extras.core.sorting import ClassKey
from sympy_extras.core.symbol import Dummy, Symbol

__all__ = ['ExprWithLimits', 'Sum', 'Product', 'sum_', 'product_', 'Limits', 'LimitLike']

#: a limit: the variable with its lower and upper bound
Limits = tuple[Symbol, Expr, Expr]
LimitLike = Union[tuple[Symbol, ExprLike, ExprLike], Tuple]


def _limit(lim: LimitLike) -> Limits:
    items: Sequence[Union[Basic, ExprLike]] = lim.args if isinstance(lim, Tuple) else lim
    if len(items) != 3:
        raise ValueError('a limit is (variable, lower, upper), not %s' % (lim,))
    v, a, b = items
    if not isinstance(v, Symbol):
        raise TypeError('the variable of a limit is a symbol, not %s' % (v,))
    if isinstance(a, Basic) and not isinstance(a, Expr) or isinstance(b, Basic) and not isinstance(b, Expr):
        raise TypeError('the bounds of a limit are expressions')
    lo = a if isinstance(a, Expr) else to_expr(a)
    hi = b if isinstance(b, Expr) else to_expr(b)
    return (v, lo, hi)


# PORTED FROM SYMPY: sympy/concrete/expr_with_limits.py:ExprWithLimits
class ExprWithLimits(Expr):
    """An expression with limits: the function and the limits, the
    variables bound."""

    __slots__ = ('_function', '_limits')

    #: the name printed and of the SymPy class
    name: str = 'ExprWithLimits'

    def __init__(self, function: ExprLike, *limits: LimitLike) -> None:
        self._function: Final[Expr] = to_expr(function)
        self._limits: Final[tuple[Limits, ...]] = tuple(_limit(lim) for lim in limits)
        if not self._limits:
            raise ValueError('%s requires limits' % self.name)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._function,) + tuple(Tuple(*lim) for lim in self._limits)

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def limits(self) -> tuple[Limits, ...]:
        return self._limits

    @property
    def variables(self) -> list[Symbol]:
        return [lim[0] for lim in self._limits]

    @property
    def bound_symbols(self) -> tuple[Symbol, ...]:
        return tuple(lim[0] for lim in self._limits)

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, cls.name)

    def _compare_name(self) -> str:
        return self.name

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        function, limits = args[0], args[1:]
        if not isinstance(function, Expr) or not all(isinstance(lim, Tuple) for lim in limits):
            raise TypeError('%s takes an expression and limits' % self.name)
        return type(self)(function, *[lim for lim in limits if isinstance(lim, Tuple)])

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        return self._evaluated(args)

    # PORTED FROM SYMPY: sympy/printing/str.py:StrPrinter._print_Sum
    def _sympystr(self, printer: StrPrinter) -> str:
        limits = ', '.join('(%s)' % ', '.join(printer.doprint(e) for e in lim) for lim in self._limits)
        return '%s(%s, %s)' % (self.name, printer.doprint(self._function), limits)

    def _eval_facts(self) -> dict[str, bool]:
        commutative = self._function.is_commutative
        return {} if commutative is None else {'commutative': commutative}

    # PORTED FROM SYMPY: sympy/concrete/expr_with_limits.py:ExprWithLimits.free_symbols
    @property
    def free_symbols(self) -> set[Symbol]:
        symbols = set(self._function.free_symbols)
        for v, a, b in self._limits:
            symbols.discard(v)
            symbols |= a.free_symbols | b.free_symbols
        return symbols

    @property
    def is_number(self) -> bool:
        return not self.free_symbols

    # PORTED FROM SYMPY: sympy/concrete/expr_with_limits.py:ExprWithLimits.has_finite_limits
    @property
    def has_finite_limits(self) -> Optional[bool]:
        """Whether every bound is finite (``None`` when not known)."""
        ret_none = False
        for v, a, b in self._limits:
            if a.is_infinite is True or b.is_infinite is True:
                return False
            if (a.is_infinite is None or b.is_infinite is None) and v.is_infinite is None:
                ret_none = True
        return None if ret_none else True

    # PORTED FROM SYMPY: sympy/concrete/expr_with_limits.py:ExprWithLimits._eval_subs
    def _subs_partial(self, old: Basic, new: Basic) -> Optional[Basic]:
        func = self._function
        limits: list[tuple[Basic, ...]] = [tuple(lim) for lim in reversed(self._limits)]
        if not isinstance(old, Symbol) or old.free_symbols & self.free_symbols:
            sub_into_func = True
            for i, xab in enumerate(limits):
                limits[i] = (xab[0],) + tuple(e._subs(old, new) for e in xab[1:])
                if xab[0].free_symbols & old.free_symbols:
                    sub_into_func = False
                    break
            if sub_into_func:
                replaced = func._subs(old, new)
                assert isinstance(replaced, Expr)
                func = replaced
        else:
            for i, xab in enumerate(limits):
                limits[i] = (xab[0],) + tuple(e._subs(old, new) for e in xab[1:])
                if old == xab[0]:
                    break
        limits.reverse()
        new_limits: list[LimitLike] = [Tuple(*lim) for lim in limits]
        return type(self)(func, *new_limits)

    def doit(self) -> Expr:
        """The node evaluated (through SymPy)."""
        from sympy_extras.core.interop import expr_through_sympy
        return expr_through_sympy(self, lambda s: s.doit())


# PORTED FROM SYMPY: sympy/concrete/summations.py:Sum
class Sum(ExprWithLimits):
    """The sum of ``function`` over the limits, unevaluated (``sum_``
    constructs as SymPy's ``Sum``)."""

    __slots__ = ()

    name = 'Sum'

    def _eval_facts(self) -> dict[str, bool]:
        facts = ExprWithLimits._eval_facts(self)
        if self._function.is_zero is True:
            facts['zero'] = True
        return facts


# PORTED FROM SYMPY: sympy/concrete/products.py:Product
class Product(ExprWithLimits):
    """The product of ``function`` over the limits, unevaluated
    (``product_`` constructs as SymPy's ``Product``)."""

    __slots__ = ()

    name = 'Product'


# PORTED FROM SYMPY: sympy/concrete/expr_with_limits.py:_common_new
def _common_new(cls: type[ExprWithLimits], function: ExprLike, limits: Sequence[LimitLike]) -> Expr:
    """SymPy's constructor: ``nan`` stays, nested nodes of the same class
    merged, the ``Piecewise`` depending on the variables folded."""
    from sympy_extras.core.numbers import NaN
    from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise_fold
    f = to_expr(function)
    lims: list[LimitLike] = [_limit(lim) for lim in limits]
    if isinstance(f, NaN):
        return f
    while type(f) is cls:
        assert isinstance(f, ExprWithLimits)
        lims = list(f.limits) + lims
        f = f.function
    variables = {_limit(lim)[0] for lim in lims}
    reps: dict[Basic, Basic] = {}
    for p in f.atoms(Piecewise):
        if not p.has(*variables):
            reps[p] = Dummy()
    if f.has(Piecewise):
        g = f.xreplace(reps)
        assert isinstance(g, Expr)
        g = piecewise_fold(g)
        back = g.xreplace({v: k for k, v in reps.items()})
        assert isinstance(back, Expr)
        f = back
    return cls(f, *lims)


def sum_(function: ExprLike, *limits: LimitLike) -> Expr:
    """``Sum(function, *limits)`` as SymPy's constructor builds it."""
    return _common_new(Sum, function, limits)


def product_(function: ExprLike, *limits: LimitLike) -> Expr:
    """``Product(function, *limits)`` as SymPy's constructor builds it."""
    return _common_new(Product, function, limits)

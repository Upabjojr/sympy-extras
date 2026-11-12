"""The order term of a series: the class ``Order`` and the function ``order``.

``Order(expr, (x, 0), (y, 0))`` is the unevaluated ``O(expr)`` at the
given points (``0`` when not given, every free symbol of ``expr`` when no
variable is given); it prints as ``O(x)``, ``O(x, (x, oo))``,
``O(x*y, x, y)``. The function ``order`` evaluates it as SymPy's ``Order``
does (``order(x**2 + x)`` is ``O(x)``): the leading term of the
expression at the point, without its coefficient. ``contains`` decides
whether a term is absorbed by the order term (sums and products with
order terms use it), and the substitution of a variable moves the point
(``O(x).subs(x, x - 1)`` is ``O(x - 1, (x, 1))``). ``series`` returns
sums with order terms; ``removeO``, ``getO`` and ``getn`` of ``Expr``
take them apart.

The evaluation is ported from SymPy's ``sympy.series.order``; the limits
of ``contains`` and of the substitution are SymPy's (through the
conversion) until the limit algorithm is ported.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Order*,
       https://docs.sympy.org/latest/modules/series/series.html#order-terms
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/order.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the unevaluated class and the evaluating function
# order are separate; the variables without given ones are sorted.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Optional, TYPE_CHECKING, Union

from .basic import Basic
from .cache import cacheit
from .coreerrors import PoleError
from .containers import Tuple
from .expr import Expr, ExprLike, as_expr, factors_of, to_expr
from .kind import Kind, NumberKind
from .numbers import Integer
from .sorting import ClassKey
from .symbol import Symbol

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Order', 'order', 'O']

#: a variable of an order with its point
OrderSpec = Union[Symbol, tuple[Symbol, ExprLike], Tuple]


class Order(Expr):
    """The unevaluated order term ``O(expr, (x, point), ...)``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Order, oo
    >>> x, y = symbols('x y')
    >>> Order(x), Order(x, (x, oo)), Order(x*y), Order(x + x**2).doit()
    (O(x), O(x, (x, oo)), O(x*y, x, y), O(x))
    >>> Order(x**2).expr, Order(x**2).variables, Order(x**2).point, Order(x**2).getn()
    (x**2, (x,), (0,), 2)
    >>> x + Order(x**2) + x**2
    x + O(x**2)
    """

    __slots__ = ('_expr', '_limits')

    is_Order = True

    def __init__(self, expr: ExprLike, *variables: OrderSpec) -> None:
        self._expr: Final[Expr] = to_expr(expr)
        limits = _limits_of(variables)
        if not limits:
            from .sorting import default_sort_key
            limits = [(s, Integer(0)) for s in sorted(self._expr.free_symbols, key=default_sort_key)]
        self._limits: Final[tuple[tuple[Symbol, Expr], ...]] = tuple(limits)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._expr,) + tuple(Tuple(s, p) for s, p in self._limits)

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def variables(self) -> tuple[Symbol, ...]:
        return tuple(s for s, _ in self._limits)

    @property
    def point(self) -> tuple[Expr, ...]:
        return tuple(p for _, p in self._limits)

    @property
    def kind(self) -> Kind:
        return NumberKind

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'Order')

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        expr, rest = args[0], args[1:]
        return Order(as_expr(expr), *[_as_tuple(t) for t in rest])

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        expr, rest = args[0], args[1:]
        return order(as_expr(expr), *[_as_tuple(t) for t in rest])

    def removeO(self) -> Expr:
        return Integer(0)

    def getO(self) -> Optional[Expr]:
        return self

    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        return self

    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        return self

    # PORTED FROM SYMPY: sympy/series/order.py:Order.as_expr_variables
    def as_expr_variables(self, order_symbols: Optional[tuple[tuple[Symbol, Expr], ...]]
                          ) -> tuple[Expr, tuple[tuple[Symbol, Expr], ...]]:
        """The expression and the variables merged with ``order_symbols``
        (used by products)."""
        if order_symbols is None:
            return self._expr, self._limits
        if (not all(o[1] == order_symbols[0][1] for o in order_symbols) and
                not all(p == self.point[0] for p in self.point)):
            raise NotImplementedError('Order at points other than 0 or oo not supported, got %s as a point.'
                                      % (self.point,))
        if order_symbols and order_symbols[0][1] != self.point[0]:
            raise NotImplementedError("Multiplying Order at different points is not supported.")
        merged = dict(order_symbols)
        for s, p in self._limits:
            if s not in merged:
                merged[s] = p
        from .sorting import default_sort_key
        return self._expr, tuple(sorted(merged.items(), key=lambda item: default_sort_key(item[0])))

    def contains(self, expr: ExprLike) -> Optional[bool]:
        """``True`` when ``expr`` belongs to this order term, ``False``
        when this order term belongs to ``expr``, ``None`` when it cannot
        be decided.

        Examples
        ========

        >>> from sympy_extras.core import symbols, order
        >>> x, y = symbols('x y')
        >>> order(x).contains(x**2), order(x**2).contains(x), order(x).contains(order(y))
        (True, False, None)
        """
        return _contains(self, to_expr(expr))

    def _subs(self, old: Basic, new: Basic) -> Basic:
        if self == old:
            return new
        if isinstance(old, Symbol) and isinstance(new, Expr):
            replaced = _subs_variable(self, old, new)
            if replaced is not None:
                return replaced
        return super()._subs(old, new)

    def __neg__(self) -> Expr:
        return self

    def _eval_facts(self) -> dict[str, bool]:
        # SymPy's Order has no facts: not even real for a real expression
        # (AccumBounds(0, 1) + O(x**3) stays a sum); commutative is kept
        # for the products here
        from .assumptions import facts_of
        facts: dict[str, bool] = {}
        if facts_of(self._expr).get('commutative') is True:
            facts['commutative'] = True
        return facts

    def _sympystr(self, printer: StrPrinter) -> str:
        if all(p.is_Number and p == 0 for _, p in self._limits):
            if len(self._limits) <= 1:
                return 'O(%s)' % printer.doprint(self._expr)
            return 'O(%s, %s)' % (printer.doprint(self._expr), ', '.join(printer.doprint(s) for s, _ in self._limits))
        return 'O(%s, %s)' % (printer.doprint(self._expr),
                              ', '.join('(%s, %s)' % (printer.doprint(s), printer.doprint(p)) for s, p in self._limits))


def _as_tuple(b: Basic) -> Tuple:
    if not isinstance(b, Tuple):
        raise TypeError("a tuple is expected, got %s" % (b,))
    return b


def _limits_of(variables: tuple[OrderSpec, ...]) -> list[tuple[Symbol, Expr]]:
    """The variables and points of the arguments of ``Order``."""
    limits: list[tuple[Symbol, Expr]] = []
    for v in variables:
        if isinstance(v, Symbol):
            limits.append((v, Integer(0)))
        elif isinstance(v, Tuple):
            s, point = v[0], v[1]
            if not (isinstance(s, Symbol) and isinstance(point, Expr)):
                raise TypeError('Variables are not symbols, got %s' % (v,))
            limits.append((s, point))
        else:
            limits.append((v[0], to_expr(v[1])))
    return limits


def order(expr: ExprLike, *variables: OrderSpec) -> Expr:
    """The evaluated order term, as SymPy's ``Order`` constructor gives it.

    Examples
    ========

    >>> from sympy_extras.core import symbols, order, oo
    >>> x, y = symbols('x y')
    >>> order(x**2 + x), order(2*x), order(1), order(x**2 + x, (x, oo)), order(x*(x + y), x, y)
    (O(x), O(x), O(1), O(x**2, (x, oo)), O(x**2 + x*y, x, y))
    """
    return _order(to_expr(expr), tuple(_limits_of(variables)), bool(variables))


# PORTED FROM SYMPY: sympy/series/order.py:Order.__new__
@cacheit
def _order(expr: Expr, given: tuple[tuple[Symbol, Expr], ...], explicit: bool) -> Expr:
    from .add import Add, add
    from .function import Application, expand_log, expand_power_base
    from .mul import Mul, mul
    from .numbers import ImaginaryUnit, Infinity, NaN, NegativeInfinity, nan
    from .power import Pow, pow
    from .sorting import default_sort_key
    from .symbol import Dummy
    from sympy_extras.functions.elementary.exponential import exp, log
    variables: list[Symbol]
    point: list[Expr]
    if not explicit:
        if isinstance(expr, Order):
            variables = list(expr.variables)
            point = list(expr.point)
        else:
            variables = sorted(expr.free_symbols, key=default_sort_key)
            point = [Integer(0)] * len(variables)
    else:
        variables = [v for v, _ in given]
        point = [p for _, p in given]
    if len(set(variables)) != len(variables):
        raise ValueError('Variables are supposed to be unique symbols, got %s' % (variables,))
    if isinstance(expr, Order):
        expr_vp = dict(zip(expr.variables, expr.point))
        new_vp = dict(expr_vp)
        for v, p in zip(variables, point):
            if v in new_vp:
                if p != new_vp[v]:
                    raise NotImplementedError("Mixing Order at different points is not supported.")
            else:
                new_vp[v] = p
        if set(expr_vp) == set(new_vp):
            return expr
        variables = list(new_vp)
        point = [new_vp[v] for v in variables]
    if isinstance(expr, NaN):
        return nan
    if any(x in p.free_symbols for x in variables for p in point):
        raise ValueError('Got %s as a point.' % (point,))
    if variables:
        if any(p != point[0] for p in point):
            raise NotImplementedError("Multivariable orders at different points are not supported.")
        p0 = point[0]
        s: dict[Basic, Basic] = {}
        rs: dict[Basic, Basic] = {}
        dummies: list[Symbol] = []
        if isinstance(p0, Infinity) or _is_infinity_times_i(p0, Infinity):
            for k in variables:
                d = Dummy()
                s[k] = pow(d, Integer(-1))
                rs[d] = pow(k, Integer(-1))
                dummies.append(d)
        elif isinstance(p0, NegativeInfinity) or _is_infinity_times_i(p0, NegativeInfinity):
            for k in variables:
                d = Dummy()
                s[k] = mul(Integer(-1), pow(d, Integer(-1)))
                rs[d] = mul(Integer(-1), pow(k, Integer(-1)))
                dummies.append(d)
        elif not (isinstance(p0, Integer) and p0.p == 0):
            for k in variables:
                d = Dummy()
                s[k] = add(d, p0)
                rs[d] = add(k, mul(Integer(-1), p0))
                dummies.append(d)
        ps: list[Expr] = [Integer(0)] * len(point) if s else list(point)
        if s:
            expr = expr.subs(s)
        if isinstance(expr, Add):
            expr = expr.factor()
        args: tuple[Symbol, ...] = tuple(dummies) if s else tuple(variables)
        if len(variables) > 1:
            # SymPy expands (x*(x + y)).as_leading_term(x, y) is x*y, the
            # wrong order term
            expr = expr.expand()
        old_expr: Optional[Expr] = None
        while old_expr != expr:
            old_expr = expr
            if isinstance(expr, Add):
                lst = expr.extract_leading_order(args)
                expr = add(*[f.expr for (_, f) in lst])
            elif not (isinstance(expr, Integer) and expr.p == 0):
                try:
                    expr = expr.as_leading_term(*args)
                except PoleError:
                    if isinstance(expr, Application) or all(isinstance(a, Application) for a in expr.args):
                        # an expression of functions only (which raise on
                        # their leading term) cannot be simplified further
                        pass
                    else:
                        orders: list[Expr] = []
                        pts = tuple(zip(args, ps))
                        for a in expr.args:
                            if not isinstance(a, Expr):
                                continue
                            try:
                                lt = a.as_leading_term(*args)
                            except PoleError:
                                lt = a
                            if lt not in args:
                                orders.append(order(lt))
                            else:
                                orders.append(order(lt, *pts))
                        if isinstance(expr, Add):
                            new_expr = order(add(*orders), *pts)
                            if isinstance(new_expr, Add):
                                new_expr = order(add(*[_order_expr(a) for a in new_expr.args]), *pts)
                            expr = _order_expr(new_expr)
                        elif isinstance(expr, Mul):
                            expr = mul(*[_order_expr(a) for a in orders])
                        elif isinstance(expr, Pow):
                            expr = exp(mul(expr.exp, log(expr.base)))
                # a symbol with zero=True
                if expr.is_zero is True:
                    expr = Integer(0)
                else:
                    expr = expr.as_independent(*args, as_Add=False)[1]
                expr = expand_power_base(expr)
                expr = expand_log(expr)
                if len(args) == 1:
                    # the argument of f(x) in O(f(x)) is irrelevant: the
                    # exponents on top of the tree combine, x**p*(-x)**q is
                    # x**(p + q) for real p, q
                    expr = _combine_exponents(expr, args[0])
        if rs:
            expr = expr.subs(rs)
    if isinstance(expr, Order):
        expr = expr.expr
    if not expr.has(*variables) and expr.is_zero is not True:
        expr = Integer(1)
    vp = dict(zip(variables, point))
    variables.sort(key=default_sort_key)
    if not variables:
        return Order(expr)
    return Order(expr, *[Tuple(v, vp[v]) for v in variables])


def _is_infinity_times_i(p: Expr, infinity: type) -> bool:
    """Whether ``p`` is ``oo*I`` (``-oo*I`` with ``NegativeInfinity``)."""
    from .mul import Mul
    from .numbers import ImaginaryUnit
    return (isinstance(p, Mul) and len(p.args) == 2 and isinstance(p.args[0], infinity)
            and isinstance(p.args[1], ImaginaryUnit))


def _order_contains_or_none(o: Expr, e: Expr) -> Optional[bool]:
    """``o.contains(e)`` of an order term, ``None`` for ``nan``."""
    return o.contains(e) if isinstance(o, Order) else None


def _order_expr(e: Basic) -> Expr:
    """``e.expr`` of an order term (SymPy's ``a.expr``)."""
    if not isinstance(e, Order):
        raise TypeError("an order term is expected, got %s" % (e,))
    return e.expr


def _combine_exponents(expr: Expr, x: Symbol) -> Expr:
    """The factors ``x**q``, ``(-x)**q``, ``(x**r)**q`` and ``(-x**r)**q``
    of ``expr`` with real exponents made powers of ``x`` (SymPy's loop
    over the factors of the dependent part)."""
    from .mul import mul
    from .power import Pow, pow
    margs = list(factors_of(expr.as_independent(x, as_Add=False)[1]))
    minus_x = mul(Integer(-1), x)
    for i, t in enumerate(margs):
        if isinstance(t, Pow):
            b, q = t.base, t.exp
            if (b == x or b == minus_x) and q.is_real is True and not q.has(x):
                margs[i] = pow(x, q)
            elif isinstance(b, Pow) and not b.exp.has(x):
                b2, r = b.base, b.exp
                if (b2 == x or b2 == minus_x) and r.is_real is True:
                    margs[i] = pow(x, mul(r, q))
            elif b.is_Mul and isinstance(b.args[0], Integer) and b.args[0] == -1:
                nb = mul(Integer(-1), b)
                if isinstance(nb, Pow) and not nb.exp.has(x):
                    b2, r = nb.base, nb.exp
                    if (b2 == x or b2 == minus_x) and r.is_real is True:
                        margs[i] = pow(x, mul(r, q))
    return mul(*margs)


def _limit_is_nonzero(ratio: Expr, s: Symbol, point: Expr) -> Optional[bool]:
    """``Limit(ratio, s, point).doit(heuristics=False) != 0``, ``None``
    when the limit is not found."""
    from .limits import Limit
    value = Limit(ratio, s, point).doit()
    if isinstance(value, Limit):
        return None
    return not (isinstance(value, Integer) and value.p == 0)


def _power_rule(self: Order, other_expr: Expr, point: Expr) -> Optional[bool]:
    """The comparison of the exponents of ``x**a`` and ``x**b`` in the
    single variable of the order term."""
    from .power import Pow
    from .add import add
    from .mul import mul
    symbol = self.variables[0]
    other = other_expr.as_independent(symbol, as_Add=False)[1]
    mine = self.expr
    if isinstance(other, Pow) and other.base == symbol and isinstance(mine, Pow) and mine.base == symbol:
        rv: Optional[bool] = None
        difference = add(mine.exp, mul(Integer(-1), other.exp))
        if point.is_zero is True:
            rv = difference.is_nonpositive
        if point.is_infinite is True:
            rv = difference.is_nonnegative
        if rv is not None:
            return rv
    return None


# PORTED FROM SYMPY: sympy/series/order.py:Order.contains
@cacheit
def _contains(self: Order, expr: Expr) -> Optional[bool]:
    from .add import Add
    from .mul import mul
    from .numbers import NaN
    from .power import Pow, pow
    from sympy_extras.simplify.powsimp import powsimp
    if expr.is_zero is True:
        return True
    if isinstance(expr, NaN):
        return False
    point = self.point[0] if self.point else Integer(0)
    if isinstance(expr, Order):
        if any(p != point for p in expr.point) or any(p != point for p in self.point):
            return None
        if expr.expr == self.expr:
            # O(1) + O(1), O(1) + O(1, x), ...
            return all(t in self.args[1:] for t in expr.args[1:])
        if isinstance(expr.expr, Add):
            return all(self.contains(t) for t in expr.expr.args if isinstance(t, Expr))
        if isinstance(self.expr, Add) and point.is_zero is True:
            return any(_order_contains_or_none(_order(t, self._limits, True), expr)
                       for t in self.expr.args if isinstance(t, Expr))
        common_symbols: tuple[Symbol, ...]
        if self.variables and expr.variables:
            common_symbols = tuple(s for s in self.variables if s in expr.variables)
        elif self.variables:
            common_symbols = self.variables
        else:
            common_symbols = expr.variables
        if not common_symbols:
            return None
        if isinstance(self.expr, Pow) and len(self.variables) == 1 and self.variables == expr.variables:
            rv = _power_rule(self, expr.expr, point)
            if rv is not None:
                return rv
        r: Optional[bool] = None
        ratio = powsimp(mul(self.expr, pow(expr.expr, Integer(-1))), deep=True, combine='exp')
        for s in common_symbols:
            l = _limit_is_nonzero(ratio, s, point)
            if r is None:
                r = l
            elif r != l:
                return None
        return r
    if isinstance(self.expr, Pow) and len(self.variables) == 1:
        rv = _power_rule(self, expr, point)
        if rv is not None:
            return rv
    obj = _order(expr, self._limits, True)
    if not isinstance(obj, Order):
        return None
    return self.contains(obj)


# PORTED FROM SYMPY: sympy/series/order.py:Order._eval_subs
def _subs_variable(self: Order, old: Symbol, new: Expr) -> Optional[Expr]:
    from .function import Subs
    from .symbol import Dummy
    if old not in self.variables:
        return None
    newexpr = self.expr.subs(old, new)
    i = self.variables.index(old)
    newvars: list[Symbol] = list(self.variables)
    newpt: list[Expr] = list(self.point)
    if isinstance(new, Symbol):
        newvars[i] = new
    else:
        syms = new.free_symbols
        if len(syms) == 1 or old in syms:
            var = self.variables[i] if old in syms else next(iter(syms))
            o = new.getO()
            if isinstance(o, Order) and o.expr.limit(var, o.point[0]) == self.point[i]:
                return order(newexpr, (var, o.point[0]))
            point = new.subs(var, self.point[i])
            if point != self.point[i]:
                d = Dummy()
                solution = _solve_for(add_neg(old, new.subs(var, d)), d)
                if solution is None:
                    return None
                point = solution.limit(old, self.point[i])
            newvars[i] = var
            newpt[i] = point
        elif old not in syms:
            del newvars[i], newpt[i]
        else:
            return None
    return order(newexpr, *[Tuple(v, p) for v, p in zip(newvars, newpt)])


def add_neg(a: Expr, b: Expr) -> Expr:
    """``a - b``."""
    from .add import add
    from .mul import mul
    return add(a, mul(Integer(-1), b))


def _solve_for(equation: Expr, d: Symbol) -> Optional[Expr]:
    """A solution of ``equation == 0`` for ``d`` (the first one of
    SymPy's ``solveset``): solved directly when ``equation`` is linear in
    ``d``, from the roots in radicals when it is rational in ``d``, by the
    native ``solveset`` otherwise (by the foreign one when the native one
    gives no set of the forms SymPy's ``Order`` takes)."""
    from .add import add
    from .mul import mul
    from .power import pow
    from .function import diff
    slope = diff(equation, d)
    if not slope.has(d) and slope.is_zero is False:
        rest = equation.subs(d, Integer(0))
        return mul(Integer(-1), rest, pow(slope, Integer(-1)))
    native = _solve_rational(equation, d)
    if native is not None:
        return native
    solved, first = _solveset_first(equation, d)
    if solved:
        return first
    from .bridge import bridge
    return bridge().solve_first(equation, d)


# PORTED FROM SYMPY: sympy/series/order.py:Order.__new__ (the first solution of solveset)
def _solveset_first(equation: Expr, d: Symbol) -> tuple[bool, Optional[Expr]]:
    """``(True, s)`` with ``s`` the first solution of the native
    ``solveset(equation, d)`` over the complex numbers (``None`` when it
    has none), as SymPy's ``Order`` takes it: the element of a finite set
    (only when it is alone: the order of several is not SymPy's), the
    elements of a complement of finite sets, the value at ``n = 0`` of an
    image of the integers (SymPy's iteration starts there); ``(False,
    None)`` when the native solver does not give such a set."""
    from sympy_extras.sets.fancysets import ImageSet, IntegersClass
    from sympy_extras.sets.sets import Complement, EmptySetClass, FiniteSet
    from sympy_extras.sets import Complexes
    from sympy_extras.solvers.solveset import solveset
    from .relational import Eq
    try:
        solutions = solveset(Eq(equation, Integer(0)), d, Complexes)
    except (NotImplementedError, ValueError, TypeError):
        return False, None
    if isinstance(solutions, EmptySetClass):
        return True, None
    if isinstance(solutions, Complement):
        kept, removed = solutions.args
        if not (isinstance(kept, FiniteSet) and isinstance(removed, FiniteSet)):
            return False, None
        left = [a for a in kept.args if a not in removed.args]
        if len(left) > 1:
            return False, None
        first = left[0] if left else None
        return True, first if isinstance(first, Expr) or first is None else None
    if isinstance(solutions, FiniteSet):
        if len(solutions.args) > 1:
            return False, None
        only = solutions.args[0] if solutions.args else None
        return (True, only) if isinstance(only, Expr) or only is None else (False, None)
    if isinstance(solutions, ImageSet) and len(solutions.variables) == 1 \
            and len(solutions.base_sets) == 1 and isinstance(solutions.base_sets[0], IntegersClass):
        image = solutions.image
        if isinstance(image, Expr):
            return True, image.subs(solutions.variables[0], Integer(0))
    return False, None


def _solve_rational(equation: Expr, d: Symbol) -> Optional[Expr]:
    """The first solution of SymPy's ``solveset`` (the points excluded by
    its ``Complement`` left out) for an equation rational in ``d``
    (:mod:`sympy_extras.polys.rationalsolve`); ``None`` for the other
    equations or without a solution."""
    from sympy_extras.polys.rationalsolve import solveset_rational
    result = solveset_rational(equation, d)
    if result is None or not result[0]:
        return None
    return result[0][0]



O = order

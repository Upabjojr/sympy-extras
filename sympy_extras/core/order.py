"""The order term of a series: the class ``Order`` and the function ``order``.

``Order(expr, (x, 0), (y, 0))`` is the unevaluated ``O(expr)`` at the
given points (``0`` when not given, every free symbol of ``expr`` when no
variable is given); it prints as ``O(x)``, ``O(x, (x, oo))``,
``O(x*y, x, y)``. The function ``order`` evaluates it as SymPy's ``Order``
does (``order(x**2 + x)`` is ``O(x)``); sums and products with an order
term are evaluated by SymPy too, since the absorption of terms into an
order needs the series machinery. ``series`` returns such sums;
``removeO``, ``getO`` and ``getn`` of ``Expr`` take them apart.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Order*,
       https://docs.sympy.org/latest/modules/series/series.html#order-terms
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from .basic import Basic
from .containers import Tuple
from .expr import Expr, ExprLike, as_expr, to_expr
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
        self._expr: Expr = to_expr(expr)
        limits: list[tuple[Symbol, Expr]] = []
        for v in variables:
            if isinstance(v, Symbol):
                limits.append((v, Integer(0)))
            elif isinstance(v, Tuple):
                s, point = v[0], v[1]
                if not (isinstance(s, Symbol) and isinstance(point, Expr)):
                    raise TypeError("a variable and a point are expected, got %s" % (v,))
                limits.append((s, point))
            else:
                limits.append((v[0], to_expr(v[1])))
        if not limits:
            from .sorting import default_sort_key
            limits = [(s, Integer(0)) for s in sorted(self._expr.free_symbols, key=default_sort_key)]
        self._limits: tuple[tuple[Symbol, Expr], ...] = tuple(limits)
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

    def getn(self) -> Optional[Expr]:
        """The exponent of the order term in its single variable (``2``
        for ``O(x**2)``), ``None`` when it is not a power."""
        if len(self._limits) != 1:
            return None
        s = self._limits[0][0]
        c, e = self._expr.as_coeff_exponent(s)
        if c.has(s):
            return None
        return e

    def contains(self, expr: ExprLike) -> Optional[bool]:
        """Whether ``expr`` is bounded by this order term (SymPy decides)."""
        import sympy
        from .interop import to_sympy
        s = to_sympy(self)
        if not isinstance(s, sympy.Order):
            return None
        answer = s.contains(to_sympy(to_expr(expr)))
        return answer if isinstance(answer, bool) else None

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import facts_of
        facts: dict[str, bool] = {}
        for name in ('commutative', 'real'):
            if facts_of(self._expr).get(name) is True:
                facts[name] = True
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


def order(expr: ExprLike, *variables: OrderSpec) -> Expr:
    """The evaluated order term, as SymPy's ``Order`` constructor gives it.

    Examples
    ========

    >>> from sympy_extras.core import symbols, order
    >>> x, y = symbols('x y')
    >>> order(x**2 + x), order(2*x), order(1)
    (O(x), O(x), O(1))
    """
    from .interop import expr_from_sympy, to_sympy
    import sympy
    node = Order(expr, *variables)
    s = sympy.Order(to_sympy(node.expr), *[(to_sympy(v), to_sympy(p)) for v, p in zip(node.variables, node.point)])
    return expr_from_sympy(s)


O = order

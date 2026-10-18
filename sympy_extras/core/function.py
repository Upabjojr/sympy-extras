"""Functions, derivatives, lambdas and expansion.

* ``Function('f')`` makes an undefined function; ``f(x)`` is an
  :class:`AppliedUndef` node whose ``func`` is ``f``. ``Function`` is a
  factory function here, not a class: the base class of the applied
  functions is :class:`Application`, as in SymPy.
* ``SymPyFunction(sympy.sin)`` wraps a function of SymPy; ``sin(x)`` is an
  :class:`AppliedSymPyFunction` node, unevaluated, whose ``doit`` and
  ``evalf`` go through SymPy. This is how the rest of SymPy (``sin``,
  ``exp``, ``log``, ``gamma``, ...) is reached from this core, and how
  :func:`~sympy_extras.core.interop.from_sympy` represents it.
* :class:`Derivative` is the unevaluated derivative; :func:`diff`
  evaluates it with the rules of calculus (the derivative of an
  undefined function stays a ``Derivative``; a power with the variable
  in the exponent uses SymPy's ``log`` through ``SymPyFunction``).
* :class:`Lambda` is an anonymous function; calling it substitutes.
* :func:`expand` distributes products over sums and expands integer
  powers of sums, with the evaluating functions.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Function*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.function
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

from sympy.core.basic import Basic as SymPyBasic
from sympy.core.function import Function as SymPyFunctionClass

from .add import Add, add, to_expr
from .basic import Basic
from .containers import Tuple
from .expr import Expr, ExprLike, terms_of
from .mul import Mul, mul
from .numbers import Integer, Number, Rational
from .power import Pow, pow
from .sorting import ClassKey, SortKey
from .symbol import Symbol

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Application', 'UndefinedFunction', 'AppliedUndef', 'Function', 'SymPyFunction',
    'AppliedSymPyFunction', 'Lambda', 'Derivative', 'diff', 'expand', 'DiffSpec']

#: what ``diff`` accepts after the expression: symbols, counts, pairs
DiffSpec = Union[Symbol, int, tuple[Symbol, int]]


class Application(Expr):
    """The base class of the applied functions ``f(x, y)``."""

    __slots__ = ('_args',)

    is_Function = True

    def __init__(self, *args: ExprLike) -> None:
        self._args: tuple[Expr, ...] = tuple(to_expr(a) for a in args)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Expr, ...]:
        return self._args

    @property
    def name(self) -> str:
        """The name of the function."""
        return type(self).__name__

    @classmethod
    def class_key(cls) -> ClassKey:
        return (4, 1, cls.__name__)

    def _sympystr(self, printer: StrPrinter) -> str:
        return '%s(%s)' % (self.name, ', '.join(printer.doprint(a) for a in self._args))


class UndefinedFunction:
    """An undefined function ``f``; calling it builds ``f(x)``.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> f = Function('f')
    >>> x = Symbol('x')
    >>> f(x), f(x).func is f, f(x).args
    (f(x), True, (x,))
    """

    __slots__ = ('name', '_assumptions')

    def __init__(self, name: str, **assumptions: bool) -> None:
        self.name: str = name
        self._assumptions: tuple[tuple[str, bool], ...] = tuple(sorted(assumptions.items()))

    def __call__(self, *args: ExprLike) -> AppliedUndef:
        return AppliedUndef(self, *args)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UndefinedFunction):
            return NotImplemented
        return self.name == other.name and self._assumptions == other._assumptions

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __hash__(self) -> int:
        return hash(('UndefinedFunction', self.name, self._assumptions))

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


def Function(name: str, **assumptions: bool) -> UndefinedFunction:
    """An undefined function with the given name.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> f, g = Function('f'), Function('g')
    >>> x, y = symbols('x y')
    >>> f(x, y) + g(x)
    f(x, y) + g(x)
    >>> f(x).diff(x)
    Derivative(f(x), x)
    """
    return UndefinedFunction(name, **assumptions)


class AppliedUndef(Application):
    """An undefined function applied to arguments."""

    __slots__ = ('_function',)

    def __init__(self, function: UndefinedFunction, *args: ExprLike) -> None:
        self._function: UndefinedFunction = function
        Application.__init__(self, *args)

    @property
    def func(self) -> Callable[..., Basic]:
        return self._function

    @property
    def function(self) -> UndefinedFunction:
        """The function applied."""
        return self._function

    @property
    def name(self) -> str:
        return self._function.name

    def __hash__(self) -> int:
        return hash((self._function, self._args))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AppliedUndef):
            if isinstance(other, Basic):
                return False
            return NotImplemented
        return self._function == other._function and self._args == other._args

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @classmethod
    def class_key(cls) -> ClassKey:
        return (4, 1, 'AppliedUndef')

    def sort_key(self) -> SortKey:
        key = Expr.sort_key(self)
        return ((4, 1, self.name), key[1], key[2], key[3], key[4])


class SymPyFunction:
    """A function of SymPy usable in this core; calling it builds an
    unevaluated :class:`AppliedSymPyFunction` node.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import SymPyFunction, Symbol
    >>> sin = SymPyFunction(sympy.sin)
    >>> x = Symbol('x')
    >>> sin(x), sin(0), sin(0).doit(), sin(x).diff(x)
    (sin(x), sin(0), 0, cos(x))
    """

    __slots__ = ('sympy_class',)

    def __init__(self, sympy_class: type[SymPyFunctionClass]) -> None:
        self.sympy_class: type[SymPyFunctionClass] = sympy_class

    @property
    def name(self) -> str:
        return self.sympy_class.__name__

    def __call__(self, *args: ExprLike) -> AppliedSymPyFunction:
        return AppliedSymPyFunction(self.sympy_class, *args)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymPyFunction):
            return NotImplemented
        return self.sympy_class is other.sympy_class

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __hash__(self) -> int:
        return hash(('SymPyFunction', self.sympy_class.__name__))

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class AppliedSymPyFunction(Application):
    """A function of SymPy applied to arguments, unevaluated; ``doit``
    evaluates it with SymPy."""

    __slots__ = ('_sympy_class',)

    def __init__(self, sympy_class: type[SymPyFunctionClass], *args: ExprLike) -> None:
        self._sympy_class: type[SymPyFunctionClass] = sympy_class
        Application.__init__(self, *args)

    @property
    def func(self) -> Callable[..., Basic]:
        return SymPyFunction(self._sympy_class)

    @property
    def function(self) -> SymPyFunction:
        """The wrapped SymPy function."""
        return SymPyFunction(self._sympy_class)

    @property
    def sympy_class(self) -> type[SymPyFunctionClass]:
        return self._sympy_class

    @property
    def name(self) -> str:
        return self._sympy_class.__name__

    def __hash__(self) -> int:
        return hash((self._sympy_class.__name__, self._args))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AppliedSymPyFunction):
            if isinstance(other, Basic):
                return False
            return NotImplemented
        return self._sympy_class is other._sympy_class and self._args == other._args

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @classmethod
    def class_key(cls) -> ClassKey:
        return (4, 1, 'AppliedSymPyFunction')

    def sort_key(self) -> SortKey:
        key = Expr.sort_key(self)
        return ((4, 1, self.name), key[1], key[2], key[3], key[4])

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return apply_sympy(self._sympy_class, *[_as_expr(a) for a in args])


def _as_expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


def apply_sympy(sympy_class: type[SymPyFunctionClass], *args: ExprLike) -> Expr:
    """The function of SymPy applied and evaluated by SymPy.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import Symbol, E
    >>> from sympy_extras.core.function import apply_sympy
    >>> x = Symbol('x')
    >>> apply_sympy(sympy.log, E), apply_sympy(sympy.log, x), apply_sympy(sympy.sin, sympy.pi)
    (1, log(x), 0)
    """
    from .interop import from_sympy, to_sympy
    value = sympy_class(*[to_sympy(to_expr(a)) for a in args])
    if not isinstance(value, SymPyBasic):
        raise TypeError("SymPy returned %r" % (value,))
    return _as_expr(from_sympy(value))


class Lambda(Basic):
    """An anonymous function ``Lambda(x, expr)``; calling it substitutes.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Lambda
    >>> x, y = symbols('x y')
    >>> f = Lambda(x, x**2)
    >>> f(3), f.variables, f.expr, f.nargs
    (9, (x,), x**2, 1)
    >>> Lambda((x, y), x*y)(2, 3)
    6
    >>> Lambda(x, x + y).free_symbols == {y}
    True
    """

    __slots__ = ('_variables', '_expr')

    is_Lambda = True

    def __init__(self, variables: Union[Symbol, tuple[Symbol, ...], Tuple], expr: ExprLike) -> None:
        if isinstance(variables, Symbol):
            names: tuple[Symbol, ...] = (variables,)
        elif isinstance(variables, Tuple):
            names = tuple(_as_symbol(v) for v in variables)
        else:
            names = tuple(variables)
        self._variables: tuple[Symbol, ...] = names
        self._expr: Expr = to_expr(expr)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (Tuple(*self._variables), self._expr)

    @property
    def func(self) -> Callable[..., Basic]:
        return Lambda

    @property
    def variables(self) -> tuple[Symbol, ...]:
        return self._variables

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def nargs(self) -> int:
        return len(self._variables)

    @property
    def free_symbols(self) -> set[Symbol]:
        return self._expr.free_symbols - set(self._variables)

    def __call__(self, *values: ExprLike) -> Expr:
        if len(values) != len(self._variables):
            raise TypeError("%s takes %d arguments, %d given" % (self, len(self._variables), len(values)))
        rules: list[tuple[Basic, Basic]] = [(v, to_expr(a)) for v, a in zip(self._variables, values)]
        return self._expr.subs(rules)

    def _sympystr(self, printer: StrPrinter) -> str:
        if len(self._variables) == 1:
            return 'Lambda(%s, %s)' % (printer.doprint(self._variables[0]), printer.doprint(self._expr))
        return 'Lambda((%s), %s)' % (', '.join(printer.doprint(v) for v in self._variables),
                                     printer.doprint(self._expr))


def _as_symbol(b: Basic) -> Symbol:
    if not isinstance(b, Symbol):
        raise TypeError("a symbol is expected, got %s" % (b,))
    return b


#: a variable of differentiation with its multiplicity
VariableCount = tuple[Symbol, int]


class Derivative(Expr):
    """An unevaluated derivative.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Derivative, Function
    >>> x, y = symbols('x y')
    >>> d = Derivative(x**2, x)
    >>> d, d.args, d.doit()
    (Derivative(x**2, x), (x**2, (x, 1)), 2*x)
    >>> Derivative(x**3, (x, 2)).doit(), Derivative(x*y, x, y).doit()
    (6*x, 1)
    """

    __slots__ = ('_expr', '_counts')

    is_Derivative = True

    def __init__(self, expr: ExprLike, *variables: Union[Symbol, tuple[Symbol, int], Tuple]) -> None:
        self._expr: Expr = to_expr(expr)
        counts: list[VariableCount] = []
        for v in variables:
            if isinstance(v, Symbol):
                counts.append((v, 1))
            elif isinstance(v, Tuple):
                s, n = v[0], v[1]
                if not (isinstance(s, Symbol) and isinstance(n, Rational) and n.q == 1):
                    raise TypeError("a variable and a count are expected, got %s" % (v,))
                counts.append((s, n.p))
            else:
                counts.append((v[0], v[1]))
        self._counts: tuple[VariableCount, ...] = _merge_counts(counts)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._expr,) + tuple(Tuple(s, Integer(n)) for s, n in self._counts)

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def variable_count(self) -> tuple[VariableCount, ...]:
        return self._counts

    @property
    def variables(self) -> tuple[Symbol, ...]:
        result: list[Symbol] = []
        for s, n in self._counts:
            result.extend([s] * n)
        return tuple(result)

    @property
    def free_symbols(self) -> set[Symbol]:
        return self._expr.free_symbols | {s for s, _ in self._counts}

    @classmethod
    def class_key(cls) -> ClassKey:
        return (4, 1, 'Derivative')

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        expr, rest = args[0], args[1:]
        return Derivative(_as_expr(expr), *[_as_tuple(t) for t in rest])

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        expr, rest = args[0], args[1:]
        return diff(_as_expr(expr), *[_as_count(t) for t in rest])

    def _sympystr(self, printer: StrPrinter) -> str:
        parts = [printer.doprint(self._expr)]
        for s, n in self._counts:
            parts.append(printer.doprint(s) if n == 1 else '(%s, %d)' % (printer.doprint(s), n))
        return 'Derivative(%s)' % ', '.join(parts)


def _as_tuple(b: Basic) -> Tuple:
    if not isinstance(b, Tuple):
        raise TypeError("a tuple is expected, got %s" % (b,))
    return b


def _as_count(b: Basic) -> VariableCount:
    t = _as_tuple(b)
    s, n = t[0], t[1]
    if not (isinstance(s, Symbol) and isinstance(n, Rational) and n.q == 1):
        raise TypeError("a variable and a count are expected, got %s" % (b,))
    return (s, n.p)


def _merge_counts(counts: list[VariableCount]) -> tuple[VariableCount, ...]:
    merged: list[VariableCount] = []
    for s, n in counts:
        if merged and merged[-1][0] == s:
            merged[-1] = (s, merged[-1][1] + n)
        else:
            merged.append((s, n))
    return tuple(merged)


def diff(expr: ExprLike, *symbols: DiffSpec) -> Expr:
    """The derivative of ``expr`` with respect to the symbols.

    ``diff(f, x, 2)``, ``diff(f, x, x)`` and ``diff(f, (x, 2))`` are the
    second derivative. What cannot be differentiated structurally (an
    undefined function) is returned as a :class:`Derivative`.

    Examples
    ========

    >>> from sympy_extras.core import symbols, diff, Function, sqrt
    >>> x, y = symbols('x y')
    >>> diff(x**3, x), diff(x*y, x), diff(x**2, x, 2), diff(sqrt(x), x)
    (3*x**2, y, 2, 1/(2*sqrt(x)))
    >>> f = Function('f')
    >>> diff(x*f(x), x)
    x*Derivative(f(x), x) + f(x)
    >>> diff(2**x, x)
    2**x*log(2)
    """
    result = to_expr(expr)
    counts: list[VariableCount] = []
    for s in symbols:
        if isinstance(s, Symbol):
            counts.append((s, 1))
        elif isinstance(s, int):
            if not counts:
                raise TypeError("a count must follow a symbol")
            last, n = counts[-1]
            counts[-1] = (last, n - 1 + s)
        else:
            counts.append(s)
    for s, n in counts:
        for _ in range(n):
            result = _derivative(result, s)
    return result


def _derivative(e: Expr, s: Symbol) -> Expr:
    if e == s:
        return Integer(1)
    if s not in e.free_symbols:
        return Integer(0)
    if isinstance(e, Add):
        return add(*[_derivative(t, s) for t in e.args])
    if isinstance(e, Mul):
        terms: list[Expr] = []
        for i, f in enumerate(e.args):
            d = _derivative(f, s)
            if isinstance(d, Rational) and d.p == 0:
                continue
            terms.append(mul(*e.args[:i], d, *e.args[i + 1:]))
        return add(*terms)
    if isinstance(e, Pow):
        b, ex = e.base, e.exp
        if s not in ex.free_symbols:
            return mul(ex, pow(b, add(ex, Integer(-1))), _derivative(b, s))
        import sympy
        log_b = apply_sympy(sympy.log, b)
        return mul(e, add(mul(_derivative(ex, s), log_b), mul(ex, _derivative(b, s), pow(b, Integer(-1)))))
    if isinstance(e, AppliedSymPyFunction):
        from .interop import from_sympy, to_sympy
        node = e.sympy_class(*[to_sympy(a) for a in e.args])
        if not isinstance(node, e.sympy_class):
            if not isinstance(node, SymPyBasic):
                raise TypeError("SymPy returned %r" % (node,))
            return _derivative(_as_expr(from_sympy(node)), s)
        terms = []
        for i, a in enumerate(e.args):
            if s not in a.free_symbols:
                continue
            partial = node.fdiff(i + 1)
            if not isinstance(partial, SymPyBasic):
                raise TypeError("SymPy returned %r" % (partial,))
            terms.append(mul(_as_expr(from_sympy(partial)), _derivative(a, s)))
        return add(*terms)
    if isinstance(e, Derivative):
        return Derivative(e.expr, *e.variable_count, (s, 1))
    return Derivative(e, s)


def expand(expr: ExprLike, deep: bool = True) -> Expr:
    """Products distributed over sums and integer powers of sums expanded.

    Examples
    ========

    >>> from sympy_extras.core import symbols, expand
    >>> x, y = symbols('x y')
    >>> expand((x + 1)**2), expand((x + 1)*(y + 1)), expand((x + y)**3)
    (x**2 + 2*x + 1, x*y + x + y + 1, x**3 + 3*x**2*y + 3*x*y**2 + y**3)
    >>> expand(2**(x + 1)), expand(x/(x + 1))
    (2*2**x, x/(x + 1))
    """
    e = to_expr(expr)
    if deep and e.args and not isinstance(e, Derivative):
        e = _as_expr(e._evaluated(tuple(expand(a) if isinstance(a, Expr) else a for a in e.args)))
    if isinstance(e, Mul):
        result: Expr = Integer(1)
        for f in e.args:
            result = _expand_product(result, f)
        return result
    if isinstance(e, Pow):
        b, ex = e.base, e.exp
        if isinstance(b, Add) and isinstance(ex, Rational) and ex.q == 1 and ex.p != 0:
            n = abs(ex.p)
            power = _expand_power(b, n)
            return power if ex.p > 0 else pow(power, Integer(-1))
        if isinstance(b, Number) and isinstance(ex, Add):
            return mul(*[pow(b, t) for t in ex.args])
    return e


def _expand_product(a: Expr, b: Expr) -> Expr:
    return add(*[mul(s, t) for s in terms_of(a) for t in terms_of(b)])


def _expand_power(base: Expr, n: int) -> Expr:
    result: Expr = Integer(1)
    square = base
    while n:
        if n & 1:
            result = _expand_product(result, square)
        n >>= 1
        if n:
            square = _expand_product(square, square)
    return result


"""Functions, derivatives, lambdas, substitutions and expansion.

* ``Function('f')`` makes an undefined function; ``f(x)`` is an
  :class:`AppliedUndef` node whose ``func`` is ``f``. ``Function`` is a
  factory function here, not a class: the base class of the applied
  functions is :class:`Application`, as in SymPy, and the base class of
  the function objects is :class:`FunctionClass`.
* ``SymPyFunction(sympy.sin)`` wraps a function of SymPy; ``sin(x)`` is an
  :class:`AppliedSymPyFunction` node, unevaluated, whose ``doit`` and
  ``evalf`` go through SymPy. This is how the rest of SymPy (``sin``,
  ``exp``, ``log``, ``gamma``, ...) is reached from this core, and how
  :func:`~sympy_extras.core.interop.from_sympy` represents it.
* :class:`WildFunction` matches any applied function in a pattern.
* :class:`Derivative` is the unevaluated derivative; :func:`diff`
  evaluates it with the rules of calculus (the derivative of an
  undefined function stays a ``Derivative``, or becomes a :class:`Subs`
  of one when the argument is not a symbol; a power with the variable in
  the exponent uses SymPy's ``log`` through ``SymPyFunction``).
* :class:`Subs` is an unevaluated substitution; its ``doit`` substitutes,
  except around a derivative of an undefined function.
* :class:`Lambda` is an anonymous function; calling it substitutes.
* :func:`expand` distributes products over sums and expands integer
  powers of sums and powers with a sum in the exponent, with the
  evaluating functions; the hints of SymPy which need its special
  functions (``log``, ``func``, ``trig``, ``complex``) go through SymPy.
* :func:`count_ops` counts the operations, :func:`nfloat` turns the
  numbers into floats, :func:`arity` gives the number of arguments of a
  function object.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Function*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.function
"""
from __future__ import annotations

import inspect
from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union, overload

import sympy
from sympy.core.basic import Basic as SymPyBasic
from sympy.core.function import Function as SymPyFunctionClass

from .add import Add, add
from .basic import Basic
from .containers import Tuple
from .coreerrors import PoleError
from .expr import AtomicExpr, Expr, ExprLike, UnevaluatedExpr, as_expr, factors_of, terms_of, to_expr
from .kind import Kind, UndefinedKind
from .mul import Mul, mul
from .numbers import Float, Integer, Number, NumberSymbol, Rational
from .power import Pow, pow
from .sorting import ClassKey, SortKey
from .symbol import Dummy, Symbol

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['FunctionClass', 'Application', 'UndefinedFunction', 'AppliedUndef', 'Function',
    'SymPyFunction', 'AppliedSymPyFunction', 'WildFunction', 'apply_sympy', 'Lambda', 'Derivative',
    'Subs', 'diff', 'expand', 'expand_mul', 'expand_multinomial', 'expand_power_base',
    'expand_power_exp', 'expand_log', 'expand_func', 'expand_trig', 'expand_complex', 'count_ops',
    'nfloat', 'arity', 'PoleError', 'DiffSpec', 'CountOpsArg']

#: what ``diff`` accepts after the expression: symbols, counts, pairs
DiffSpec = Union[Symbol, int, tuple[Symbol, int]]
#: a set of numbers of arguments, ``None`` for any number
NargsSpec = Union[int, Iterable[int], None]


def _nargs_set(nargs: NargsSpec) -> Optional[frozenset[int]]:
    if nargs is None:
        return None
    if isinstance(nargs, int):
        return frozenset([nargs])
    return frozenset(nargs)


class FunctionClass:
    """The base class of the function objects (``Function('f')``,
    ``SymPyFunction(sympy.sin)``): what ``f(x).func`` is.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import Function, SymPyFunction, FunctionClass
    >>> f = Function('f')
    >>> isinstance(f, FunctionClass), isinstance(SymPyFunction(sympy.sin), FunctionClass), f.nargs
    (True, True, None)
    >>> SymPyFunction(sympy.sin).nargs, SymPyFunction(sympy.log).nargs == frozenset({1, 2})
    (frozenset({1}), True)
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def nargs(self) -> Optional[frozenset[int]]:
        """The numbers of arguments accepted, ``None`` for any number."""
        return None

    def __call__(self, *args: ExprLike) -> Application:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


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


class UndefinedFunction(FunctionClass):
    """An undefined function ``f``; calling it builds ``f(x)``.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> f = Function('f')
    >>> x = Symbol('x')
    >>> f(x), f(x).func is f, f(x).args
    (f(x), True, (x,))
    """

    __slots__ = ('_name', '_nargs', '_assumptions')

    def __init__(self, name: str, nargs: NargsSpec = None, **assumptions: bool) -> None:
        from .assumptions import PREDICATES
        for k in assumptions:
            if k not in PREDICATES:
                raise ValueError("unknown assumption %r" % (k,))
        self._name: str = name
        self._nargs: Optional[frozenset[int]] = _nargs_set(nargs)
        self._assumptions: tuple[tuple[str, bool], ...] = tuple(sorted(assumptions.items()))

    @property
    def name(self) -> str:
        return self._name

    @property
    def nargs(self) -> Optional[frozenset[int]]:
        return self._nargs

    @property
    def assumptions(self) -> dict[str, bool]:
        """The assumptions given to ``Function``."""
        return dict(self._assumptions)

    def __call__(self, *args: ExprLike) -> AppliedUndef:
        if self._nargs is not None and len(args) not in self._nargs:
            raise TypeError("%s takes %s arguments, %d given" % (self._name, sorted(self._nargs), len(args)))
        return AppliedUndef(self, *args)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UndefinedFunction):
            return NotImplemented
        return self._name == other._name and self._assumptions == other._assumptions and self._nargs == other._nargs

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __hash__(self) -> int:
        return hash(('UndefinedFunction', self._name, self._assumptions, self._nargs))


def Function(name: str, nargs: NargsSpec = None, **assumptions: bool) -> UndefinedFunction:
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
    >>> Function('h', real=True)(x).is_real
    True
    """
    return UndefinedFunction(name, nargs, **assumptions)


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

    @property
    def kind(self) -> Kind:
        return UndefinedKind

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
        return (4, 0, 'AppliedUndef')

    def _compare_name(self) -> str:
        return self._function.name

    def sort_key(self) -> SortKey:
        key = Expr.sort_key(self)
        return ((4, 0, self.name), key[1], key[2], key[3], key[4])

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import facts_of
        facts = self._function.assumptions
        commutative = _all_commutative([facts_of(a) for a in self._args])
        if commutative is not None:
            facts.setdefault('commutative', commutative)
        return facts


#: the ranks of the functions of SymPy in the canonical order (the
#: elementary functions first, as SymPy's ``class_key`` orders them)
_FUNCTION_RANKS: dict[str, int] = {
    'exp': 10, 'log': 11, 'sin': 20, 'cos': 21, 'tan': 22, 'cot': 23, 'sinh': 30, 'cosh': 31,
    'tanh': 32, 'coth': 33, 'conjugate': 40, 're': 41, 'im': 42, 'arg': 43}


def _all_commutative(facts: list[dict[str, bool]]) -> Optional[bool]:
    values = [f.get('commutative') for f in facts]
    if all(v is True for v in values):
        return True
    if any(v is False for v in values):
        return False
    return None


class SymPyFunction(FunctionClass):
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

    @property
    def nargs(self) -> Optional[frozenset[int]]:
        nargs = self.sympy_class.nargs
        if isinstance(nargs, sympy.FiniteSet):
            return frozenset(int(n) for n in nargs.args if isinstance(n, sympy.Integer))
        return None

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
        return (4, 10000, 'AppliedSymPyFunction')

    def _compare_name(self) -> str:
        return self._sympy_class.__name__

    def sort_key(self) -> SortKey:
        key = Expr.sort_key(self)
        return ((4, _FUNCTION_RANKS.get(self.name, 10000), self.name), key[1], key[2], key[3], key[4])

    @property
    def kind(self) -> Kind:
        from .interop import to_sympy
        from .kind import NumberKind
        return NumberKind if to_sympy(self).kind == NumberKind else UndefinedKind

    def as_base_exp(self) -> tuple[Expr, Expr]:
        # exp(a) is E**a, as in SymPy (where E**a builds exp(a))
        if self._sympy_class is sympy.exp:
            from .numbers import E
            return (E, self._args[0])
        return (self, Integer(1))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return apply_sympy(self._sympy_class, *[as_expr(a) for a in args])

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import sympy_facts
        return sympy_facts(self)


class WildFunction(AtomicExpr):
    """A pattern head matching any applied function (with one of the
    given numbers of arguments).

    Examples
    ========

    >>> from sympy_extras.core import symbols, Function, WildFunction
    >>> x, y = symbols('x y')
    >>> f = Function('f')
    >>> F = WildFunction('F')
    >>> (f(x, y) + 1).match(F + 1) == {F: f(x, y)}, f(x).match(WildFunction('G', nargs=2))
    (True, None)
    """

    __slots__ = ('name', '_nargs')

    is_Function = True

    def __init__(self, name: str, nargs: NargsSpec = None) -> None:
        self.name: str = name
        self._nargs: Optional[frozenset[int]] = _nargs_set(nargs)
        Basic.__init__(self)

    @property
    def nargs(self) -> Optional[frozenset[int]]:
        return self._nargs

    def accepts(self, count: int) -> bool:
        """Whether an application with ``count`` arguments may be matched."""
        return self._nargs is None or count in self._nargs

    def _hashable_content(self) -> tuple[Basic, ...]:
        return ()

    def __hash__(self) -> int:
        return hash(('WildFunction', self.name, self._nargs))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WildFunction):
            if isinstance(other, Basic):
                return False
            return NotImplemented
        return self.name == other.name and self._nargs == other._nargs

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @classmethod
    def class_key(cls) -> ClassKey:
        return (2, 0, 'WildFunction')

    def __str__(self) -> str:
        return self.name + '_'

    def _sympystr(self, printer: StrPrinter) -> str:
        return self.name + '_'


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
    from .interop import expr_from_sympy, to_sympy
    from .parameters import global_parameters
    if not global_parameters.evaluate:
        return AppliedSymPyFunction(sympy_class, *args)
    value = sympy_class(*[to_sympy(to_expr(a)) for a in args])
    if not isinstance(value, SymPyBasic):
        raise TypeError("SymPy returned %r" % (value,))
    return expr_from_sympy(value)


def arity(cls: Union[FunctionClass, Callable[..., object]]) -> Union[int, tuple[int, ...], None]:
    """The number of arguments of a function object (``None`` when any
    number is accepted, a tuple when several are).

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import Function, SymPyFunction, arity
    >>> arity(SymPyFunction(sympy.sin)), arity(SymPyFunction(sympy.log)), arity(Function('f')), arity(lambda a, b: 0)
    (1, (1, 2), None, 2)
    """
    if isinstance(cls, FunctionClass):
        nargs = cls.nargs
        if nargs is None:
            return None
        if len(nargs) == 1:
            return next(iter(nargs))
        return tuple(sorted(nargs))
    parameters = inspect.signature(cls).parameters.values()
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return None
    return sum(1 for p in parameters
               if p.default is inspect.Parameter.empty
               and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))


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

    def __init__(self, variables: Union[Symbol, Sequence[Symbol], Tuple], expr: ExprLike) -> None:
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

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        variables, expr = args
        if not isinstance(variables, Tuple):
            raise TypeError("a tuple of variables is expected, got %s" % (variables,))
        return Lambda(variables, as_expr(expr))

    @property
    def variables(self) -> tuple[Symbol, ...]:
        return self._variables

    @property
    def signature(self) -> tuple[Symbol, ...]:
        return self._variables

    @property
    def bound_symbols(self) -> tuple[Symbol, ...]:
        return self._variables

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def nargs(self) -> int:
        return len(self._variables)

    @property
    def is_identity(self) -> bool:
        """Whether the function maps its single variable to itself."""
        return len(self._variables) == 1 and self._expr == self._variables[0]

    @property
    def free_symbols(self) -> set[Symbol]:
        return self._expr.free_symbols - set(self._variables)

    def __call__(self, *values: ExprLike) -> Expr:
        if len(values) != len(self._variables):
            raise TypeError("%s takes %d arguments, %d given" % (self, len(self._variables), len(values)))
        rules: list[tuple[Basic, Basic]] = [(v, to_expr(a)) for v, a in zip(self._variables, values)]
        return self._expr.subs(rules)

    def rcall(self, *args: Basic) -> Basic:
        return self(*[as_expr(a) for a in args])

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
                counts.append(_as_count(v))
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

    @property
    def kind(self) -> Kind:
        return self._expr.kind

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'Derivative')

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        expr, rest = args[0], args[1:]
        return Derivative(as_expr(expr), *[_as_tuple(t) for t in rest])

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        expr, rest = args[0], args[1:]
        return diff(as_expr(expr), *[_as_count(_as_tuple(t)) for t in rest])

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import facts_of
        commutative = facts_of(self._expr).get('commutative')
        return {} if commutative is None else {'commutative': commutative}

    def _sympystr(self, printer: StrPrinter) -> str:
        parts = [printer.doprint(self._expr)]
        for s, n in self._counts:
            parts.append(printer.doprint(s) if n == 1 else '(%s, %d)' % (printer.doprint(s), n))
        return 'Derivative(%s)' % ', '.join(parts)


def _as_tuple(b: Basic) -> Tuple:
    if not isinstance(b, Tuple):
        raise TypeError("a tuple is expected, got %s" % (b,))
    return b


def _as_count(t: Tuple) -> VariableCount:
    s, n = t[0], t[1]
    if not (isinstance(s, Symbol) and isinstance(n, Rational) and n.q == 1):
        raise TypeError("a variable and a count are expected, got %s" % (t,))
    return (s, n.p)


def _merge_counts(counts: list[VariableCount]) -> tuple[VariableCount, ...]:
    merged: list[VariableCount] = []
    for s, n in counts:
        if merged and merged[-1][0] == s:
            merged[-1] = (s, merged[-1][1] + n)
        else:
            merged.append((s, n))
    return tuple(merged)


class Subs(Expr):
    """An unevaluated substitution ``Subs(expr, variables, point)``.

    ``doit`` substitutes, except for a variable of a derivative replaced
    by something which is not a symbol, or by a symbol which already
    occurs in the derivative (``Subs(Derivative(f(x), x), x, 2)`` stays,
    as in SymPy).

    Examples
    ========

    >>> from sympy_extras.core import symbols, Subs, Function
    >>> x, y = symbols('x y')
    >>> f = Function('f')
    >>> s = Subs(x**2 + y, (x, y), (1, 2))
    >>> s, s.args, s.doit(), Subs(f(x), x, 2).doit(), Subs(f(x).diff(x), x, 2).doit()
    (Subs(x**2 + y, (x, y), (1, 2)), (x**2 + y, (x, y), (1, 2)), 3, f(2), Subs(Derivative(f(x), x), x, 2))
    >>> f(2*x).diff(x)
    2*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, 2*x)
    """

    __slots__ = ('_expr', '_variables', '_point')

    def __init__(self, expr: ExprLike, variables: Union[Symbol, Sequence[Symbol], Tuple],
                 point: Union[ExprLike, Sequence[ExprLike], Tuple]) -> None:
        self._expr: Expr = to_expr(expr)
        if isinstance(variables, Symbol):
            names: tuple[Symbol, ...] = (variables,)
        elif isinstance(variables, Tuple):
            names = tuple(_as_symbol(v) for v in variables)
        else:
            names = tuple(variables)
        if isinstance(point, (Expr, int, float, Fraction)):
            values: tuple[Expr, ...] = (to_expr(point),)
        elif isinstance(point, Tuple):
            values = tuple(as_expr(p) for p in point)
        else:
            values = tuple(to_expr(p) for p in point)
        if len(names) != len(values):
            raise ValueError("Subs needs as many points as variables")
        self._variables: tuple[Symbol, ...] = names
        self._point: tuple[Expr, ...] = values
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._expr, Tuple(*self._variables), Tuple(*self._point))

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def variables(self) -> tuple[Symbol, ...]:
        return self._variables

    @property
    def point(self) -> tuple[Expr, ...]:
        return self._point

    @property
    def bound_symbols(self) -> tuple[Symbol, ...]:
        return self._variables

    @property
    def free_symbols(self) -> set[Symbol]:
        result = self._expr.free_symbols - set(self._variables)
        for p in self._point:
            result |= p.free_symbols
        return result

    @property
    def kind(self) -> Kind:
        return self._expr.kind

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'Subs')

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        expr, variables, point = args
        return Subs(as_expr(expr), _as_tuple(variables), _as_tuple(point))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        expr, variables, point = args
        return _substituted(as_expr(expr), tuple(_as_symbol(v) for v in _as_tuple(variables)),
                            tuple(as_expr(p) for p in _as_tuple(point)))

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import facts_of
        commutative = facts_of(self._expr).get('commutative')
        return {} if commutative is None else {'commutative': commutative}

    def _sympystr(self, printer: StrPrinter) -> str:
        if len(self._variables) == 1:
            return 'Subs(%s, %s, %s)' % (printer.doprint(self._expr), printer.doprint(self._variables[0]),
                                         printer.doprint(self._point[0]))
        return 'Subs(%s, (%s), (%s))' % (printer.doprint(self._expr),
                                         ', '.join(printer.doprint(v) for v in self._variables),
                                         ', '.join(printer.doprint(p) for p in self._point))


def _substituted(expr: Expr, variables: tuple[Symbol, ...], point: tuple[Expr, ...]) -> Expr:
    """``expr`` with the variables replaced by the points, keeping a
    ``Subs`` around the variables of a derivative of an undefined
    function which are not replaced by symbols."""
    derivatives = [d for d in expr.atoms(Derivative) if isinstance(d, Derivative)]
    blocked = [(v, p) for v, p in zip(variables, point)
               if any(v in d.variables for d in derivatives)
               and (not isinstance(p, Symbol) or p in expr.free_symbols - {v})]
    free = [(v, p) for v, p in zip(variables, point) if (v, p) not in blocked]
    rules: list[tuple[Basic, Basic]] = [(v, p) for v, p in free]
    inner = expr.subs(rules) if rules else expr
    if not blocked:
        return inner
    return Subs(inner, tuple(v for v, _ in blocked), tuple(p for _, p in blocked))


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
        log_b = apply_sympy(sympy.log, b)
        return mul(e, add(mul(_derivative(ex, s), log_b), mul(ex, _derivative(b, s), pow(b, Integer(-1)))))
    if isinstance(e, AppliedSymPyFunction):
        from .interop import expr_from_sympy, to_sympy
        node = e.sympy_class(*[to_sympy(a) for a in e.args])
        if not isinstance(node, e.sympy_class):
            if not isinstance(node, SymPyBasic):
                raise TypeError("SymPy returned %r" % (node,))
            return _derivative(expr_from_sympy(node), s)
        terms = []
        for i, a in enumerate(e.args):
            if s not in a.free_symbols:
                continue
            partial = node.fdiff(i + 1)
            if not isinstance(partial, SymPyBasic):
                raise TypeError("SymPy returned %r" % (partial,))
            terms.append(mul(expr_from_sympy(partial), _derivative(a, s)))
        return add(*terms)
    if isinstance(e, AppliedUndef):
        terms = []
        others_depend = sum(1 for a in e.args if s in a.free_symbols) > 1
        for i, a in enumerate(e.args):
            if s not in a.free_symbols:
                continue
            if isinstance(a, Symbol) and not others_depend:
                terms.append(Derivative(e, a))
                continue
            xi = Dummy('xi_%d' % (i + 1))
            inner = e.function(*e.args[:i], xi, *e.args[i + 1:])
            terms.append(mul(_substituted(Derivative(inner, xi), (xi,), (a,)), _derivative(a, s)))
        return add(*terms)
    if isinstance(e, Derivative):
        return Derivative(e.expr, *e.variable_count, (s, 1))
    if e.is_Order:
        from .interop import expr_through_sympy, to_sympy
        return expr_through_sympy(e, lambda o: o.diff(to_sympy(s)))
    if isinstance(e, Subs):
        terms = []
        if s in e.expr.free_symbols - set(e.variables):
            terms.append(_substituted(_derivative(e.expr, s), e.variables, e.point))
        for v, p in zip(e.variables, e.point):
            dp = _derivative(p, s)
            if isinstance(dp, Rational) and dp.p == 0:
                continue
            terms.append(mul(_substituted(_derivative(e.expr, v), e.variables, e.point), dp))
        return add(*terms)
    return Derivative(e, s)


def expand(expr: ExprLike, deep: bool = True, mul: bool = True, multinomial: bool = True, power_base: bool = True,
           power_exp: bool = True, log: bool = False, func: bool = False, trig: bool = False,
           complex: bool = False) -> Expr:
    """Products distributed over sums (``mul``), integer powers of sums
    expanded (``multinomial``), powers with a sum in the exponent split
    (``power_exp``); the hints ``log``, ``func``, ``trig`` and ``complex``
    are SymPy's and go through SymPy.

    Examples
    ========

    >>> from sympy_extras.core import symbols, expand
    >>> x, y = symbols('x y')
    >>> expand((x + 1)**2), expand((x + 1)*(y + 1)), expand((x + y)**3)
    (x**2 + 2*x + 1, x*y + x + y + 1, x**3 + 3*x**2*y + 3*x*y**2 + y**3)
    >>> expand(2**(x + 1)), expand(x/(x + 1)), expand(x*(y + 1), mul=False)
    (2*2**x, x/(x + 1), x*(y + 1))
    """
    e = to_expr(expr)
    if log or func or trig or complex:
        from .interop import expr_through_sympy
        return expr_through_sympy(e, lambda s: sympy.expand(s, deep=deep, mul=mul, multinomial=multinomial,
                                                            power_base=power_base, power_exp=power_exp, log=log,
                                                            func=func, trig=trig, complex=complex))
    return _expand(e, deep, mul, multinomial, power_exp)


def _expand(e: Expr, deep: bool, distribute: bool, multinomial: bool, power_exp: bool) -> Expr:
    if deep and e.args and not isinstance(e, (Derivative, UnevaluatedExpr)):
        e = as_expr(e._evaluated(tuple(_expand(a, deep, distribute, multinomial, power_exp)
                                       if isinstance(a, Expr) else a for a in e.args)))
    if isinstance(e, Mul) and distribute:
        result: Expr = Integer(1)
        for f in e.args:
            result = _expand_product(result, f)
        return result
    if isinstance(e, Pow):
        b, ex = e.base, e.exp
        if multinomial and isinstance(b, Add) and isinstance(ex, Rational) and ex.q == 1 and ex.p != 0:
            n = abs(ex.p)
            power = _expand_power(b, n)
            return power if ex.p > 0 else pow(power, Integer(-1))
        if power_exp and isinstance(ex, Add) and isinstance(b, Number):
            return mul(*[pow(b, t) for t in ex.args])
    if power_exp and isinstance(e, AppliedSymPyFunction) and e.sympy_class is sympy.exp and isinstance(e.args[0], Add):
        from .numbers import E
        return mul(*[pow(E, t) for t in e.args[0].args])
    return e


def _expand_product(a: Expr, b: Expr) -> Expr:
    return add(*[mul(s, t) for s in terms_of(a) for t in terms_of(b)])


def _expand_power(base: Expr, n: int) -> Expr:
    """``base**n`` expanded by the multinomial theorem: one term per
    composition of ``n`` into as many parts as ``base`` has terms."""
    from math import factorial
    terms = terms_of(base)
    expanded: list[Expr] = []
    for exponents in _compositions(n, len(terms)):
        coefficient = factorial(n)
        for k in exponents:
            coefficient //= factorial(k)
        factors: list[Expr] = [Integer(coefficient)]
        factors.extend(pow(t, Integer(k)) for t, k in zip(terms, exponents) if k)
        expanded.append(mul(*factors))
    return add(*expanded)


def _compositions(n: int, parts: int) -> Iterator[tuple[int, ...]]:
    """The tuples of ``parts`` nonnegative integers with the sum ``n``."""
    if parts == 1:
        yield (n,)
        return
    for first in range(n, -1, -1):
        for rest in _compositions(n - first, parts - 1):
            yield (first,) + rest


def expand_mul(expr: ExprLike, deep: bool = True) -> Expr:
    """Products distributed over sums only.

    Examples
    ========

    >>> from sympy_extras.core import symbols, expand_mul
    >>> x, y = symbols('x y')
    >>> expand_mul(x*(y + 1)), expand_mul((x + 1)**2)
    (x*y + x, (x + 1)**2)
    """
    return expand(expr, deep=deep, mul=True, multinomial=False, power_base=False, power_exp=False)


def expand_multinomial(expr: ExprLike, deep: bool = True) -> Expr:
    """Integer powers of sums expanded only.

    Examples
    ========

    >>> from sympy_extras.core import symbols, expand_multinomial
    >>> x, y = symbols('x y')
    >>> expand_multinomial((x + y)**2)
    x**2 + 2*x*y + y**2
    """
    return expand(expr, deep=deep, mul=False, multinomial=True, power_base=False, power_exp=False)


def expand_power_base(expr: ExprLike, deep: bool = True, force: bool = False) -> Expr:
    """Powers of products distributed (``force`` lets SymPy distribute
    without regard to the assumptions).

    Examples
    ========

    >>> from sympy_extras.core import symbols, expand_power_base
    >>> x, y, z = symbols('x y z')
    >>> expand_power_base((x*y)**2), expand_power_base((x*y)**z), expand_power_base((x*y)**z, force=True)
    (x**2*y**2, (x*y)**z, x**z*y**z)
    """
    if force:
        from .interop import expr_through_sympy
        return expr_through_sympy(to_expr(expr), lambda s: sympy.expand_power_base(s, deep=deep, force=True))
    return expand(expr, deep=deep, mul=False, multinomial=False, power_base=True, power_exp=False)


def expand_power_exp(expr: ExprLike, deep: bool = True) -> Expr:
    """Powers with a sum in the exponent split into products.

    Examples
    ========

    >>> from sympy_extras.core import symbols, expand_power_exp
    >>> x, y = symbols('x y')
    >>> expand_power_exp(2**(x + y))
    2**x*2**y
    """
    return expand(expr, deep=deep, mul=False, multinomial=False, power_base=False, power_exp=True)


def expand_log(expr: ExprLike, deep: bool = True, force: bool = False) -> Expr:
    """SymPy's ``expand_log``: logarithms of products and powers split.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import Symbol, SymPyFunction, expand_log
    >>> p, q = Symbol('p', positive=True), Symbol('q', positive=True)
    >>> expand_log(SymPyFunction(sympy.log)(p*q))
    log(p) + log(q)
    """
    from .interop import expr_through_sympy
    return expr_through_sympy(to_expr(expr), lambda s: sympy.expand_log(s, deep=deep, force=force))


def expand_func(expr: ExprLike, deep: bool = True) -> Expr:
    """SymPy's ``expand_func``: the special functions rewritten.

    Examples
    ========

    >>> from sympy_extras.core import expand_func, sympify
    >>> expand_func(sympify('gamma(x + 1)'))
    x*gamma(x)
    """
    from .interop import expr_through_sympy
    return expr_through_sympy(to_expr(expr), lambda s: sympy.expand_func(s, deep=deep))


def expand_trig(expr: ExprLike, deep: bool = True) -> Expr:
    """SymPy's ``expand_trig``.

    Examples
    ========

    >>> from sympy_extras.core import expand_trig, sympify
    >>> expand_trig(sympify('sin(2*x)'))
    2*sin(x)*cos(x)
    """
    from .interop import expr_through_sympy
    return expr_through_sympy(to_expr(expr), lambda s: sympy.expand_trig(s, deep=deep))


def expand_complex(expr: ExprLike, deep: bool = True) -> Expr:
    """SymPy's ``expand_complex``: the expression as real and imaginary parts.

    Examples
    ========

    >>> from sympy_extras.core import expand_complex, sympify
    >>> expand_complex(sympify('x + I*y'))
    I*(re(y) + im(x)) + re(x) - im(y)
    """
    from .interop import expr_through_sympy
    return expr_through_sympy(to_expr(expr), lambda s: sympy.expand_complex(s, deep=deep))


#: what ``count_ops`` accepts: a node, a sequence of them, a mapping
CountOpsArg = Union[Basic, Sequence['CountOpsArg'], Mapping[Basic, Basic]]


def count_ops(expr: CountOpsArg, visual: bool = False) -> Basic:
    """The number of operations of an expression, or with ``visual`` their
    kinds as a sum of symbols (``2*ADD + MUL + POW``), as SymPy counts.

    Examples
    ========

    >>> from sympy_extras.core import symbols, count_ops, sympify
    >>> x, y = symbols('x y')
    >>> count_ops(x**2 + 2*x + 1), count_ops(x**2 + 2*x + 1, visual=True)
    (4, 2*ADD + MUL + POW)
    >>> count_ops(x/y, visual=True), count_ops(-x, visual=True), count_ops(x - y, visual=True)
    (DIV, NEG, SUB)
    >>> count_ops(sympify('sin(x)*cos(x)'), visual=True)
    COS + MUL + SIN
    """
    counts: dict[str, int] = {}
    _count(expr, counts)
    if visual:
        return add(*[mul(n, Symbol(name)) for name, n in sorted(counts.items())])
    return Integer(sum(counts.values()))


def _tally(counts: dict[str, int], name: str, n: int = 1) -> None:
    if n:
        counts[name] = counts.get(name, 0) + n


def _count(e: CountOpsArg, counts: dict[str, int]) -> None:
    from .logic import BooleanFunction
    from .mod import Mod
    from .relational import Relational
    if isinstance(e, Mapping):
        for k, v in e.items():
            _count(k, counts)
            _count(v, counts)
        return
    if not isinstance(e, Basic):
        for item in e:
            _count(item, counts)
        return
    if isinstance(e, Rational):
        _tally(counts, 'DIV', int(e.q > 1))
        _tally(counts, 'NEG', int(e.p < 0))
        return
    if isinstance(e, Add):
        for i, t in enumerate(e.as_ordered_terms()):
            negative = t.could_extract_minus_sign()
            if i == 0:
                _tally(counts, 'NEG', int(negative))
            else:
                _tally(counts, 'SUB' if negative else 'ADD')
            _count(-t if negative else t, counts)
        return
    if isinstance(e, Mul):
        c, rest = e.as_coeff_Mul()
        numerator: list[Expr] = []
        denominator: list[Expr] = []
        if c.is_negative is True:
            _tally(counts, 'NEG')
            c = c.number_mul(Integer(-1))
        if isinstance(c, Rational):
            if c.p != 1:
                numerator.append(Integer(c.p))
            if c.q != 1:
                denominator.append(Integer(c.q))
        elif not (c.is_Number and c == 1):
            numerator.append(c)
        for f in factors_of(rest):
            b, ex = f.as_base_exp()
            if isinstance(ex, Rational) and ex.p < 0:
                denominator.append(pow(b, Rational(-ex.p, ex.q)))
            else:
                numerator.append(f)
        _tally(counts, 'MUL', max(len(numerator) - 1, 0))
        if denominator:
            _tally(counts, 'DIV')
            _tally(counts, 'MUL', len(denominator) - 1)
        for f in numerator + denominator:
            if not isinstance(f, Number):
                _count(f, counts)
        return
    if isinstance(e, Pow):
        if isinstance(e.exp, Rational) and e.exp.p == -1 and e.exp.q == 1:
            _tally(counts, 'DIV')
        else:
            _tally(counts, 'POW')
            _count(e.exp, counts)
        _count(e.base, counts)
        return
    if isinstance(e, (Relational, BooleanFunction)):
        _tally(counts, type(e).__name__.upper())
        for a in e.args:
            _count(a, counts)
        return
    if isinstance(e, Mod):
        _tally(counts, 'MOD')
        for a in e.args:
            _count(a, counts)
        return
    if isinstance(e, AppliedUndef):
        _tally(counts, 'FUNC_' + e.name.upper())
        for a in e.args:
            _count(a, counts)
        return
    if isinstance(e, AppliedSymPyFunction):
        _tally(counts, e.name.upper())
        for a in e.args:
            _count(a, counts)
        return
    if isinstance(e, Derivative):
        _tally(counts, 'DERIVATIVE')
        _count(e.expr, counts)
        return
    if isinstance(e, Subs):
        _count(e.expr, counts)
        return
    if isinstance(e, Lambda):
        _count(e.expr, counts)
        return
    if e.is_Order or e.is_Atom or isinstance(e, (Tuple, UnevaluatedExpr)) or type(e).__name__ == 'Dict':
        for child in e.args:
            _count(child, counts)
        return
    if e.args:
        _tally(counts, type(e).__name__.upper())
        for child in e.args:
            _count(child, counts)


@overload
def nfloat(expr: Basic, n: int = 15, exponent: bool = False) -> Basic: ...


@overload
def nfloat(expr: Sequence[Basic], n: int = 15, exponent: bool = False) -> list[Basic]: ...


def nfloat(expr: Union[Basic, Sequence[Basic]], n: int = 15, exponent: bool = False) -> Union[Basic, list[Basic]]:
    """The numbers of the expression (or of each expression of a sequence)
    turned into floats of ``n`` digits, the exponents of powers too when
    ``exponent`` is set.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Rational, nfloat, pi
    >>> x = Symbol('x')
    >>> nfloat(x**2 + Rational(1, 3)), nfloat(x**2, exponent=True), nfloat(pi, 5)
    (x**2 + 0.333333333333333, x**2.0, 3.1416)
    """
    if isinstance(expr, Basic):
        return _nfloat(expr, n, exponent)
    return [_nfloat(e, n, exponent) for e in expr]


def _nfloat(e: Basic, n: int, exponent: bool) -> Basic:
    if isinstance(e, Number):
        return Float(e, dps=n) if isinstance(e, Rational) else e
    if isinstance(e, NumberSymbol):
        return e.evalf(n)
    if isinstance(e, Pow) and not exponent:
        return pow(as_expr(_nfloat(e.base, n, exponent)), e.exp)
    if not e.args:
        return e
    return e._evaluated(tuple(_nfloat(a, n, exponent) for a in e.args))

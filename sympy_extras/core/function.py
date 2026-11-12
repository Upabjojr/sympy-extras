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
  :func:`~sympy_extras.interop.from_sympy` represents it.
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

import mpmath
from fractions import Fraction
from typing import Callable, Final, Iterable, Iterator, Mapping, Optional, Sequence, TYPE_CHECKING, Union, overload

from .add import Add, add
from .basic import Basic
from .containers import Tuple
from .coreerrors import PoleError
from .expr import AtomicExpr, Expr, ExprLike, UnevaluatedExpr, as_expr, factors_of, terms_of, to_expr
from .kind import Kind, UndefinedKind
from .mul import Mul, mul, neg
from .numbers import Exp1, Float, Integer, Number, NumberSymbol, Rational
from .power import Pow, pow
from .sorting import ClassKey
from .symbol import Dummy, Symbol

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['FunctionClass', 'Application', 'UndefinedFunction', 'AppliedUndef', 'Function',
    'WildFunction', 'Lambda', 'Derivative',
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
    >>> from sympy_extras.core import Function, FunctionClass
    >>> from sympy_extras.interop import SymPyFunction
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
        self._args: Final[tuple[Expr, ...]] = tuple(to_expr(a) for a in args)
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

    def _series_nargs(self) -> Optional[frozenset[int]]:
        """The numbers of arguments of the function for the series
        (``None``: any number, SymPy's ``Naturals0``)."""
        return None

    def _singularities(self) -> Union[None, bool, tuple[Expr, ...]]:
        """SymPy's ``_singularities`` of the function: the points where it
        is not holomorphic, ``True`` when it is nowhere holomorphic, ``None``
        when unknown."""
        return None

    # PORTED FROM SYMPY: sympy/core/function.py:Function.is_singular
    def is_singular(self, a: Expr) -> Optional[bool]:
        """Whether ``a`` is a singularity of the function."""
        from .fuzzy import fuzzy_or
        from .numbers import ComplexInfinity
        ss = self._singularities()
        if ss is None or isinstance(ss, bool):
            return ss
        return fuzzy_or(a.is_infinite if isinstance(s, ComplexInfinity) else (a - s).is_zero for s in ss)

    # PORTED FROM SYMPY: sympy/core/function.py:Function._eval_is_meromorphic
    def _eval_is_meromorphic(self, x: Symbol, a: Expr) -> Optional[bool]:
        from .fuzzy import fuzzy_not
        if not self._args:
            return True
        if any(t.has(x) for t in self._args[1:]):
            return False
        arg = self._args[0]
        if not arg._eval_is_meromorphic(x, a):
            return None
        return fuzzy_not(self.is_singular(arg.subs(x, a)))

    def _rebuilt(self, args: Sequence[Expr]) -> Expr:
        """The node of the same function with ``args``, evaluated."""
        return as_expr(self._evaluated(tuple(args)))

    # PORTED FROM SYMPY: sympy/core/function.py:Function._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        raise PoleError('Asymptotic expansion of %s around %s is not implemented.'
                        % (type(self).__name__, list(args0)))

    # PORTED FROM SYMPY: sympy/core/function.py:Function._eval_nseries
    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        from .numbers import NaN as _NaN, oo, zoo, nan
        from .order import Order, order
        args = self._args
        args0 = [t.limit(x, Integer(0)) for t in args]
        if any(t.is_finite is False for t in args0):
            a = [t.as_leading_term(x, logx=logx) for t in args]
            a0 = [t.limit(x, Integer(0)) for t in a]
            if any(t.has(oo, neg(oo), zoo, nan) for t in a0):
                return self._eval_aseries(n, args0, x, logx)
            # the argument goes to oo logarithmically only: a power series
            # around the logarithmic term, f(1 + x + log(x)) is
            # f(1 + logx) + x*f'(1 + logx) + O(x**2)
            terms = [t._eval_nseries(x, n, logx, cdir) for t in args]
            z = [r - r0 for r, r0 in zip(terms, a0)]
            p = [Dummy() for _ in z]
            q: list[Expr] = []
            v: Optional[Symbol] = None
            zv: Expr = Integer(0)
            for ai, zi, pi in zip(a0, z, p):
                if zi.has(x):
                    if v is not None:
                        raise NotImplementedError
                    q.append(ai + pi)
                    v = pi
                    zv = zi
                else:
                    q.append(ai)
            e1 = self._rebuilt(q)
            if v is None:
                return e1
            series = e1._eval_nseries(v, n, logx, cdir)
            o = series.getO()
            series = series.removeO()
            o_expr = o.expr if isinstance(o, Order) else Integer(0)
            return series.subs(v, zv).expand() + order(o_expr.subs(v, zv), x)
        nargs = self._series_nargs()
        if nargs is None or (nargs == frozenset({1}) and not _is_zero_number(args0[0])) or any(c > 1 for c in nargs):
            e: Expr = self
            expanded = e.expand()
            if e == expanded:
                # the general algorithm, for sin(x + 1) or sin(cos(x))
                if len(e.args) == 1 and isinstance(e, Application):
                    e = e._rebuilt([e._args[0].cancel()])
                term = e.subs(x, Integer(0))
                if term.is_finite is False or isinstance(term, _NaN):
                    raise PoleError("Cannot expand %s around 0" % (self,))
                result: Expr = term
                fact = 1
                _x = Dummy('xi')
                e = e.subs(x, _x)
                for i in range(1, n):
                    fact *= i
                    e = e.diff(_x)
                    value = e.subs(_x, Integer(0))
                    if isinstance(value, _NaN):
                        # a limit when it has to be
                        value = e.limit(_x, Integer(0))
                    if value.is_finite is False:
                        raise PoleError("Cannot expand %s around 0" % (self,))
                    result = result + (value * x**i / Integer(fact)).expand()
                return result + order(x**n, x)
            return expanded.nseries(x, n=n, logx=logx)
        arg = self._args[0]
        lst: list[Expr] = []
        g: Optional[Expr] = None
        # the number of terms predicted
        nterms = n + 2
        cf = order(arg.as_leading_term(x), x).getn()
        if cf is not None and cf != Integer(0):
            from .add import _ceiling_int
            nterms = _ceiling_int(Integer(n) / cf)
        for i in range(nterms):
            g = self.taylor_term(i, arg, g)
            g = g.nseries(x, n=n, logx=logx)
            lst.append(g)
        return add(*lst) + order(x**n, x)

    # PORTED FROM SYMPY: sympy/core/function.py:Function._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        from .order import order
        args = [a.as_leading_term(x, logx=logx) for a in self._args]
        from .order import Order
        o = order(Integer(1), x)
        if isinstance(o, Order) and any(x in a.free_symbols and o.contains(a) for a in args):
            # an argument vanishing as x -> 0 (x, x**2, not 3 or 1/x): the
            # class must give the first nonzero term (sin(x) is x)
            raise NotImplementedError('%s has no _eval_as_leading_term routine' % (type(self).__name__,))
        return self


def _is_zero_number(e: Expr) -> bool:
    return e.is_Number and e == 0


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
        self._name: Final[str] = name
        self._nargs: Final[Optional[frozenset[int]]] = _nargs_set(nargs)
        self._assumptions: Final[tuple[tuple[str, bool], ...]] = tuple(sorted(assumptions.items()))

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
        self._function: Final[UndefinedFunction] = function
        Application.__init__(self, *args)

    @property
    def func(self) -> Callable[..., Basic]:
        return self._function

    def _series_nargs(self) -> Optional[frozenset[int]]:
        return self._function.nargs

    # PORTED FROM SYMPY: sympy/core/function.py:AppliedUndef._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        return self

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

    def _sort_class_key(self) -> ClassKey:
        return (4, 0, self.name)

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
        self.name: Final[str] = name
        self._nargs: Final[Optional[frozenset[int]]] = _nargs_set(nargs)
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


# PORTED FROM SYMPY: sympy/core/function.py:Function._should_evalf
def _float_precision(e: Expr) -> int:
    """The precision of a ``Float``, or of the floats of a pure complex
    number ``a + b*I`` with numbers ``a``, ``b``; ``-1`` otherwise."""
    from .numbers import Float, ImaginaryUnit, Number
    if isinstance(e, Float):
        return e.precision
    if not isinstance(e, Add):
        return -1
    real: Optional[Expr] = None
    imaginary: Optional[Expr] = None
    for term in e.args:
        if isinstance(term, Number) and real is None:
            real = term
        elif isinstance(term, ImaginaryUnit) and imaginary is None:
            imaginary = Integer(1)
        elif isinstance(term, Mul) and len(term.args) == 2 and isinstance(term.args[0], Number) \
                and isinstance(term.args[1], ImaginaryUnit) and imaginary is None:
            imaginary = term.args[0]
        else:
            return -1
    if real is None or imaginary is None:
        return -1
    return max(p.precision if isinstance(p, Float) else -1 for p in (real, imaginary))


class DefinedFunction(Application):
    """A function of the typed core (``Exp``, ``Log``, ...): the class
    builds the unevaluated node, the classmethod :meth:`eval` holds the
    evaluation rules (``None`` when there is nothing to do) and
    :meth:`apply` (what the lower case functions ``exp``, ``log`` call)
    returns the evaluated form. A subclass defines ``function_name`` (the
    printed name, SymPy's), ``nargs``, :meth:`eval`, :meth:`fdiff` (the
    partial derivatives), :meth:`numeric` (the value from the values of
    the arguments, with mpmath) and ``_eval_facts``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions.elementary.exponential import Exp, exp
    >>> x = Symbol('x')
    >>> Exp(0), exp(0), Exp(x).diff(x), Exp(1).evalf(10)
    (exp(0), 1, exp(x), 2.718281828)
    """

    #: the printed name, SymPy's name of the function
    function_name: str = 'f'
    #: the numbers of arguments of the node
    nargs: frozenset[int] = frozenset({1})
    #: the numbers of arguments of the evaluating function (``log(x, b)``)
    call_nargs: frozenset[int] = frozenset({1})
    #: any number of arguments (``Max``, ``Min``): ``nargs`` is not checked
    variadic: bool = False
    #: SymPy's ``unbranched``: the function is single-valued, so that
    #: ``unpolarify`` may strip the polar parts of its arguments
    unbranched: bool = False

    def _series_nargs(self) -> Optional[frozenset[int]]:
        return None if self.variadic else self.nargs

    #: SymPy's ``_singularities`` of the class (see ``Application._singularities``)
    singularities: Union[None, bool, tuple[Expr, ...]] = None

    def _singularities(self) -> Union[None, bool, tuple[Expr, ...]]:
        return self.singularities

    def _eval_is_meromorphic(self, x: Symbol, a: Expr) -> Optional[bool]:
        if type(self).__name__ in _SYMPY_OWN_MEROMORPHIC:
            # the ported rules of the special functions (Bessel functions,
            # incomplete gamma functions)
            from sympy_extras.functions._meromorphic import rules
            rule = rules().get(type(self).__name__)
            if rule is not None:
                return rule(self, x, a)
        return Application._eval_is_meromorphic(self, x, a)

    #: the classes whose SymPy counterpart expands by the generic
    #: Function._eval_nseries (ported below); the others use SymPy's
    generic_nseries: bool = False
    #: the SymPy class takes any number of arguments (nargs Naturals0): the
    #: generic expansion goes by derivatives
    sympy_variadic: bool = False

    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        from sympy_extras.functions._series_rules import rules
        from .bridge import bridge
        rule = rules().get(type(self).__name__)
        if rule is not None:
            ported = rule(self, x, n, logx, cdir)
            if ported is not None:
                return ported
        if type(self).generic_nseries:
            found = _generic_function_nseries(self, x, n, logx)
            if found is not None:
                return found
        # the classes whose expansion is not ported: the foreign system's
        return bridge().nseries(self, x, n, logx, cdir)

    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        if type(self).__name__ not in _SYMPY_OWN_LEADING_TERM:
            return _generic_leading_term(self, x, logx)
        # the ported rules of the classes without a method of their own
        from sympy_extras.functions._leading_terms import rules
        rule = rules().get(type(self).__name__)
        if rule is not None:
            return rule(self, x, logx, cdir)
        return _generic_leading_term(self, x, logx)

    def __init__(self, *args: ExprLike) -> None:
        if not self.variadic and len(args) not in self.nargs:
            raise TypeError('%s takes %s arguments, %d given' % (self.function_name, sorted(self.nargs), len(args)))
        Application.__init__(self, *args)

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        """The evaluated form, or ``None`` when the node stays."""
        return None

    @classmethod
    def apply(cls, *args: ExprLike) -> Expr:
        """The function applied and evaluated (the node itself when
        ``evaluate(False)`` is in force or no rule applies)."""
        from .parameters import global_parameters
        expressions = tuple(to_expr(a) for a in args)
        if global_parameters.evaluate:
            value = cls.eval(*expressions)
            if value is not None:
                return value
            # PORTED FROM SYMPY: sympy/core/function.py:Function.__new__ (_should_evalf):
            # a function of floats is evaluated at their largest precision
            precisions = [_float_precision(a) for a in expressions]
            if expressions and min(precisions) > 0:
                from .numbers import prec_to_dps
                return cls(*expressions).evalf(prec_to_dps(max(precisions)))
        return cls(*expressions)

    # PORTED FROM SYMPY: sympy/core/function.py:Function.fdiff
    def fdiff(self, argindex: int) -> Expr:
        """The partial derivative with respect to the argument number
        ``argindex`` (from 1), for a function without its own rule:
        ``Derivative(f(x), x)`` when the argument is a symbol found in no
        other argument, else ``Subs(Derivative(f(xi), xi), xi, a)``."""
        index = argindex - 1
        a = self.args[index]
        if isinstance(a, Symbol) and not any(a in v.free_symbols for i, v in enumerate(self.args) if i != index):
            return Derivative(self, a)
        xi = Dummy('xi_%d' % argindex)
        rebuilt = type(self)(*(self.args[:index] + (xi,) + self.args[index + 1:]))
        return Subs(Derivative(rebuilt, xi), (xi,), (a,))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        """The value at the numerical values of the arguments (mpmath, at
        the working precision), ``None`` when undefined."""
        return None

    def _eval_derivative(self, s: Symbol) -> Optional[Expr]:
        """The derivative with respect to ``s`` when the function has its
        own rule (``re``, ``Abs``, ...: not holomorphic), ``None`` for the
        chain rule with :meth:`fdiff`."""
        return None

    def _eval_conjugate(self) -> Optional[Expr]:
        """The conjugate (SymPy's ``Expr._eval_conjugate``: the node itself
        when it is real, minus it when it is imaginary), ``None`` when
        unknown."""
        if self.is_extended_real:
            return self
        if self.is_imaginary:
            return mul(Integer(-1), self)
        return None

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        """The real and imaginary parts (SymPy's ``Expr.as_real_imag``:
        ``re(self)``, ``im(self)``; ``None`` when ``self`` is ``ignore``)."""
        if ignore is not None and ignore == self:
            return None
        from sympy_extras.functions.elementary.complexes import im, re
        return (re(self), im(self))

    @property
    def func(self) -> Callable[..., Basic]:
        return type(self)

    @property
    def name(self) -> str:
        return self.function_name

    def __hash__(self) -> int:
        return hash((self.function_name, self._args))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefinedFunction):
            if isinstance(other, Basic):
                return False
            return NotImplemented
        return type(self) is type(other) and self._args == other._args

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @classmethod
    def class_key(cls) -> ClassKey:
        # the place of SymPy's applied functions in the order of the arguments
        return (4, 10000, 'AppliedSymPyFunction')

    def _compare_name(self) -> str:
        return self.function_name

    def _sort_class_key(self) -> ClassKey:
        return (4, _FUNCTION_RANKS.get(self.function_name, 10000), self.function_name)

    @property
    def kind(self) -> Kind:
        # SymPy's default for a function: UndefinedKind (exp says NumberKind)
        return UndefinedKind

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return type(self).apply(*[as_expr(a) for a in args])

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}



def arity(cls: Union[FunctionClass, Callable[..., object]]) -> Union[int, tuple[int, ...], None]:
    """The number of arguments of a function object (``None`` when any
    number is accepted, a tuple when several are).

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import Function, arity
    >>> from sympy_extras.interop import SymPyFunction
    >>> arity(SymPyFunction(sympy.sin)), arity(SymPyFunction(sympy.log)), arity(Function('f')), arity(lambda a, b: 0)
    (1, (1, 2), None, 2)
    """
    if isinstance(cls, type) and issubclass(cls, DefinedFunction):
        if cls.variadic:
            return None
        accepted = sorted(cls.call_nargs)
        return accepted[0] if len(accepted) == 1 else tuple(accepted)
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
        self._variables: Final[tuple[Symbol, ...]] = names
        self._expr: Final[Expr] = to_expr(expr)
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
        self._expr: Final[Expr] = to_expr(expr)
        counts: list[VariableCount] = []
        for v in variables:
            if isinstance(v, Symbol):
                counts.append((v, 1))
            elif isinstance(v, Tuple):
                counts.append(_as_count(v))
            else:
                counts.append((v[0], v[1]))
        self._counts: Final[tuple[VariableCount, ...]] = _merge_counts(counts)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._expr,) + tuple(Tuple(s, Integer(n)) for s, n in self._counts)

    # PORTED FROM SYMPY: sympy/core/function.py:Derivative._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        """The derivative of the first term of the series of the expression
        with a nonzero derivative."""
        d: Expr = Integer(0)
        for leading_term in self.expr.lseries(x):
            d = diff(leading_term, *self.variables)
            if not (d.is_Number and d == 0):
                break
        return d

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

    # PORTED FROM SYMPY: sympy/core/function.py:Derivative._eval_subs
    def _subs_partial(self, old: Basic, new: Basic) -> Optional[Basic]:
        """The substitutions which become a ``Subs``: a variable replaced
        by what is not a symbol (``Derivative(f(x), x).subs(x, 0)``), a
        free symbol which would be captured, a variable renamed to a symbol
        already in the expression."""
        wrt = [v for v, _ in self._counts]
        if not isinstance(new, Expr):
            return None
        if isinstance(old, Symbol) and old in wrt and not _diff_wrt(new):
            return Subs(self, old, new)
        new_expr = self._expr._subs(old, new)
        if new_expr != self._expr and isinstance(old, Symbol):
            forbidden = self._expr.free_symbols & set(wrt)
            if (new.free_symbols - old.free_symbols) & forbidden:
                return Subs(self, old, new)
        renamed = [(v, v._subs(old, new)) for v in wrt]
        if any(nv != v for v, nv in renamed) and isinstance(old, Symbol):
            atoms = self._expr.atoms(Symbol)
            for v, nv in renamed:
                if nv != v and nv in atoms:
                    return Subs(self, old, new)
        if not all(isinstance(nv, Symbol) for _, nv in renamed):
            return None
        # SymPy's _derivative_dispatch: the derivative rebuilt unevaluated
        # (Derivative(2*f(x), x) stays)
        counts = [Tuple(nv, Integer(n)) for (_, nv), (_, n) in zip(renamed, self._counts)]
        return Derivative(as_expr(new_expr), *counts)

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
        self._expr: Final[Expr] = to_expr(expr)
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
        self._variables: Final[tuple[Symbol, ...]] = names
        self._point: Final[tuple[Expr, ...]] = values
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._expr, Tuple(*self._variables), Tuple(*self._point))

    # PORTED FROM SYMPY: sympy/core/function.py:Subs._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        if x in self.point:
            xvar = self.variables[self.point.index(x)]
            return self.expr.as_leading_term(xvar)
        if x in self.variables:
            # a bound variable: it does not exist after the substitution
            return self
        # the variable is independent of the substitution
        return self.expr.as_leading_term(x)

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

    # PORTED FROM SYMPY: sympy/core/function.py:Subs._eval_subs
    def _subs_partial(self, old: Basic, new: Basic) -> Optional[Basic]:
        """A substitution respecting the bound variables: a neutral
        renaming of a variable, otherwise the variable kept and the point
        substituted (``foo.doit().subs(reps) == foo.subs(reps).doit()``)."""
        if not isinstance(new, Expr):
            return None
        pt = list(self._point)
        if isinstance(old, Symbol) and old in self._variables:
            if isinstance(new, Symbol) and not any(a.has(new) for a in self.args):
                renamed = self.xreplace({old: new})
                return renamed
            i = self._variables.index(old)
            for j in range(i, len(self._variables)):
                moved = pt[j]._subs(old, new)
                assert isinstance(moved, Expr)
                pt[j] = moved
            return Subs(self._expr, self._variables, pt)
        v = [var._subs(old, new) for var in self._variables]
        if v != list(self._variables):
            if not isinstance(old, Symbol):
                return None
            return Subs(self._expr, self._variables + (old,), pt + [new])
        expr = self._expr._subs(old, new)
        assert isinstance(expr, Expr)
        points = [p._subs(old, new) for p in pt]
        return Subs(expr, self._variables, [p for p in points if isinstance(p, Expr)])

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
            # SymPy's form (e*db)*b**e/b: x**(-pi/2)/x stays apart from
            # x**(-pi/2 - 1), and 1/2 is distributed over the inner derivative
            return mul(mul(ex, _derivative(b, s)), e, pow(b, Integer(-1)))
        from sympy_extras.functions.elementary.exponential import log
        log_b = log(b)
        return mul(e, add(mul(_derivative(ex, s), log_b), mul(ex, _derivative(b, s), pow(b, Integer(-1)))))
    if isinstance(e, DefinedFunction):
        own = e._eval_derivative(s)
        if own is not None:
            return own
        # the chain rule with the partial derivatives of the function
        chain: list[Expr] = []
        for i, a in enumerate(e.args):
            da = _derivative(a, s)
            if isinstance(da, Rational) and da.p == 0:
                continue
            chain.append(mul(e.fdiff(i + 1), da))
        return add(*chain)
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
    from sympy_extras.functions.elementary.piecewise import Piecewise
    if isinstance(e, Piecewise):
        return e.map_expressions(lambda a: _derivative(a, s))
    own_node = e._eval_derivative_node(s)
    if own_node is not None:
        return own_node
    from .order import Order
    if isinstance(e, Order):
        # PORTED FROM SYMPY: sympy/series/order.py:Order._eval_derivative
        from .order import order
        return order(_derivative(e.expr, s), *[Tuple(v, p) for v, p in zip(e.variables, e.point)])
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
           power_exp: bool = True, log: bool = True, func: bool = False, trig: bool = False,
           complex: bool = False, force: bool = False) -> Expr:
    """Products distributed over sums (``mul``), integer powers of sums
    expanded (``multinomial``), powers with a sum in the exponent split
    (``power_exp``), products in the base of powers split
    (``power_base``, every factor with ``force``); the hints ``log``,
    ``func``, ``trig`` and ``complex`` are SymPy's, in SymPy's order.

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
    if complex:
        # SymPy applies its hints in sorted order: complex before func, log,
        # multinomial, mul, the power hints and trig
        e = as_expr(_expand_complex(e, deep, ExpandHints(mul, multinomial, power_base, power_exp, log, True, trig)))
        complex = False
    if func:
        # after complex, before log, multinomial, mul, the power hints, trig
        e = as_expr(_expand_func(e, deep, ExpandHints(mul, multinomial, power_base, power_exp, log, False, trig)))
        func = False
        if not (mul or multinomial or power_exp or log or trig or complex):
            return e
    if trig and not func:
        # SymPy's sorted hints: trig after log, multinomial, mul and the
        # power hints, then the loop of multinomial, mul and log
        hints = ExpandHints(mul, multinomial, power_base, power_exp, log, False, True)
        e = _expand(e, deep, mul, multinomial, power_exp, log) if (mul or multinomial or power_exp or log) else e
        e = as_expr(_expand_trig(e, deep, hints))
        trig = False
        if not (mul or multinomial or log):
            return e
    if log and (mul or multinomial):
        # SymPy's sorted hints run the log hint over the whole tree before
        # the mul hint: log(a*(-1 - I)) with a < 0 is split into
        # log(-a) + log(1 + I) before the product is distributed into
        # -a - a*I, where it no longer splits
        e = _expand(e, deep, False, False, False, True, False, force)
    result = _expand(e, deep, mul, multinomial, power_exp, log, power_base, force)
    # SymPy's Expr.expand repeats the multinomial and mul hints until
    # nothing changes (sqrt(c)**2 = c may create a new product with a sum)
    while mul or multinomial or log:
        was = result
        result = _expand(result, deep, mul, multinomial, False, log, False, force)
        if result == was:
            break
    return result


class ExpandHints:
    """The hints of an ``expand`` call, as SymPy passes them on to
    ``as_real_imag`` (``log.as_real_imag`` expands with them)."""

    __slots__ = ('mul', 'multinomial', 'power_base', 'power_exp', 'log', 'complex', 'trig')

    def __init__(self, mul: bool, multinomial: bool, power_base: bool, power_exp: bool, log: bool,
                 complex: bool, trig: bool = False) -> None:
        self.mul, self.multinomial, self.power_base = mul, multinomial, power_base
        self.power_exp, self.log, self.complex, self.trig = power_exp, log, complex, trig

    def expand(self, e: Expr, deep: bool) -> Expr:
        return expand(e, deep=deep, mul=self.mul, multinomial=self.multinomial, power_base=self.power_base,
                      power_exp=self.power_exp, log=self.log, complex=self.complex, trig=self.trig)


def expand_with_hints(e: Expr, hints: Optional[ExpandHints], complex: Optional[bool] = None) -> Expr:
    """SymPy's ``e.expand(deep, **hints)`` inside ``as_real_imag``: the
    hints of the ``expand`` call (``complex`` replaced when given), the
    default ones without a call."""
    if hints is None:
        return e.expand()
    if complex is not None:
        hints = ExpandHints(hints.mul, hints.multinomial, hints.power_base, hints.power_exp, hints.log, complex,
                            hints.trig)
    return hints.expand(e, True)


_TRIG_RULES: dict[type, Callable[[Expr, ExpandHints], Expr]] = {}
_FUNC_RULES: dict[type, Callable[[Expr, ExpandHints], Optional[Expr]]] = {}


# PORTED FROM SYMPY: sympy/core/expr.py:Expr._expand_hint with the _eval_expand_func methods
def _expand_func(e: Basic, deep: bool, hints: ExpandHints) -> Basic:
    """The special functions (their arguments first) expanded by their
    rules; a node without a ported rule is expanded by SymPy (that node
    only) when SymPy has one."""
    from .bridge import bridge
    if not _FUNC_RULES:
        from sympy_extras.functions._expand_func import rules
        _FUNC_RULES.update(rules())
    if deep and e.args and not e.is_Atom:
        args = tuple(_expand_func(a, deep, hints) for a in e.args)
        if any(new is not old and new != old for new, old in zip(args, e.args)):
            e = e._evaluated(args)
    if not isinstance(e, Expr):
        return e
    rule = _FUNC_RULES.get(type(e))
    new: Optional[Basic] = rule(e, hints) if rule is not None else None
    if new is None and (e._is_foreign_node() or type(e).__name__ in _SYMPY_EXPAND_FUNC):
        # a class with a SymPy rule not ported (or a wrapped foreign node)
        new = bridge().expand_func_node(e)
    if new is not None and new != e:
        return new
    return e


# PORTED FROM SYMPY: sympy/core/expr.py:Expr._expand_hint with the _eval_expand_trig methods
def _expand_trig(e: Basic, deep: bool, hints: ExpandHints) -> Basic:
    """The trigonometric and hyperbolic functions (their arguments first)
    expanded by their rules."""
    if not _TRIG_RULES:
        from sympy_extras.functions.elementary._expand_trig import rules
        _TRIG_RULES.update(rules())
    if deep and e.args and not e.is_Atom:
        args = tuple(_expand_trig(a, deep, hints) for a in e.args)
        if any(new is not old and new != old for new, old in zip(args, e.args)):
            e = e._evaluated(args)
    rule = _TRIG_RULES.get(type(e))
    if rule is not None and isinstance(e, Expr):
        new = rule(e, hints)
        if new != e:
            return new
    return e


# PORTED FROM SYMPY: sympy/core/expr.py:Expr._expand_hint with Expr._eval_expand_complex
def _expand_complex(e: Basic, deep: bool, hints: ExpandHints) -> Basic:
    """Every node (its arguments first) as ``re + I*im``."""
    from .numbers import I
    if deep and e.args and not e.is_Atom:
        args = tuple(_expand_complex(a, deep, hints) for a in e.args)
        if any(new is not old and new != old for new, old in zip(args, e.args)):
            e = e._evaluated(args)
    if isinstance(e, Expr):
        from sympy_extras.functions.elementary.complexes import as_real_imag, im, re
        parts = as_real_imag(e, True, None, hints)
        real, imaginary = parts if parts is not None else (re(e), im(e))
        new = add(real, mul(I, imaginary))
        if new != e:
            return new
    return e


def _expand(e: Expr, deep: bool, distribute: bool, multinomial: bool, power_exp: bool, log: bool,
            power_base: bool = False, force: bool = False) -> Expr:
    if deep and e.args and not isinstance(e, (Derivative, UnevaluatedExpr)):
        args = tuple(_expand(a, deep, distribute, multinomial, power_exp, log, power_base, force)
                     if isinstance(a, Expr) else a for a in e.args)
        # SymPy's _expand_hint rebuilds a node only when an argument
        # changed: 2*(y*log(2) + 2) stays under the log hint alone
        if any(new is not old and new != old for new, old in zip(args, e.args)):
            e = as_expr(e._evaluated(args))
    from sympy_extras.concrete.expr_with_limits import Sum, sum_
    if isinstance(e, Sum):
        # PORTED FROM SYMPY: sympy/concrete/expr_with_limits.py:AddWithLimits._eval_expand_basic
        summand = e.function
        if isinstance(summand, Add) and (force or (summand.is_commutative is True
                                                   and e.has_finite_limits is not False)):
            return add(*[sum_(t, *e.limits) for t in summand.args if isinstance(t, Expr)])
        return e
    if isinstance(e, Mul) and distribute:
        return _expand_mul_node(e, deep)
    if isinstance(e, Pow):
        b, ex = e.base, e.exp
        if multinomial and isinstance(b, Add) and isinstance(ex, Rational) and ex.q == 1 and ex.p != 0:
            n = abs(ex.p)
            power = _expand_power(b, n)
            return power if ex.p > 0 else pow(power, Integer(-1))
        if multinomial and isinstance(b, Add) and isinstance(ex, Rational) and ex.q > 1 and ex.p > ex.q:
            whole = ex.p // ex.q  # (y - x)**(3/2) is (y - x)*sqrt(y - x), distributed
            rest = Rational(ex.p - whole*ex.q, ex.q)
            return _expand_product(_expand_power(b, whole), pow(b, rest))
        if multinomial and isinstance(b, Add) and isinstance(ex, Rational) and ex.q > 1 and -ex.p > ex.q:
            # PORTED FROM SYMPY: sympy/core/power.py:Pow._eval_expand_multinomial
            # (a negative exponent: 1/(the expanded power with the opposite exponent))
            whole = -ex.p // ex.q
            rest = Rational(-ex.p - whole*ex.q, ex.q)
            return pow(_expand_product(_expand_power(b, whole), pow(b, rest)), Integer(-1))
        if power_base and isinstance(b, Mul):
            split = _expand_power_base_node(e, force)
            if split is not e:
                # the power_exp hint is a later pass of SymPy over the whole
                # tree: the new powers are split too
                return _expand(split, True, False, False, power_exp, False, False, force) if power_exp else split
        if power_exp and isinstance(ex, Add) and (force or b.is_zero is False or _all_nonneg_or_nonpos(ex)) \
                and ex.is_commutative is not False:
            # PORTED FROM SYMPY: sympy/core/power.py:Pow._eval_expand_power_exp
            return mul(*[pow(b, t) for t in ex.args if isinstance(t, Expr)])
    if power_exp and _is_exp_node(e) and isinstance(e.args[0], Add):
        from .numbers import E
        return mul(*[pow(E, t) for t in e.args[0].args])
    if log and _is_log_node(e):
        from sympy_extras.functions.elementary.exponential import Log
        if isinstance(e, Log):
            return _expand_log_node(e, force, False)
    return e


# PORTED FROM SYMPY: sympy/core/power.py:Pow._eval_expand_power_base
def _expand_power_base_node(e: Pow, force: bool) -> Expr:
    """``(a*b)**e`` split into ``a**e*b**e`` for the factors which allow it
    (nonnegative or polar ones, all with ``force`` or an integer ``e``),
    the others kept together (the signs and the powers of ``I`` sorted
    out first)."""
    from .numbers import I as _I, ImaginaryUnit
    from .fuzzy import fuzzy_and
    b, ex = e.base, e.exp
    if not isinstance(b, Mul):
        return e
    cargs = [f for f in b.args if isinstance(f, Expr) and f.is_commutative is not False]
    nc: list[Expr] = [f for f in b.args if isinstance(f, Expr) and f.is_commutative is False]
    if nc:
        nc = [_expand_power_base_node(f, force) if isinstance(f, Pow) else f for f in nc]
        if isinstance(ex, Integer):
            if ex.p > 0:
                rv: Expr = Mul(*(nc * ex.p))
            else:
                rv = Mul(*([pow(f, Integer(-1)) for f in nc[::-1]] * -ex.p))
            if cargs:
                rv = mul(rv, pow(mul(*cargs), ex))
            return rv
        if not cargs:
            return Pow(Mul(*nc), ex)
        nc = [Mul(*nc)]
    # the commutative bases sifted
    other = [f for f in cargs if f.is_extended_real is False]
    maybe_real = [f for f in cargs if f.is_extended_real is not False]
    nonneg: list[Expr] = []
    neg: list[Expr] = []
    imag: list[Expr] = []
    for f in maybe_real:
        if isinstance(f, ImaginaryUnit):
            imag.append(f)
            continue
        polar = f.is_polar
        if polar is True:
            nonneg.append(f)
        elif polar is None:
            key = f.is_extended_nonnegative
            if key is True:
                nonneg.append(f)
            elif key is False:
                neg.append(f)
            else:
                other.append(f)
        else:
            other.append(f)
    if imag:
        i = len(imag) % 4
        if i == 1:
            other.append(_I)
        elif i in (2, 3):
            if neg:
                nonn = -neg.pop()
                if not (isinstance(nonn, Integer) and nonn.p == 1):
                    nonneg.append(nonn)
            else:
                neg.append(Integer(-1))
            if i == 3:
                other.append(_I)
    # the bases which can be separated from the base
    if force or ex.is_integer is True:
        cargs = nonneg + neg + other
        other = list(nc)
    else:
        # the negatives made positive, the residual -1 in other
        if len(neg) > 1:
            o: Expr = Integer(1)
            if not other and isinstance(neg[0], Number):
                o = o * neg.pop(0)
            if len(neg) % 2:
                o = -o
            for n in neg:
                nonneg.append(-n)
            if not (isinstance(o, Integer) and o.p == 1):
                other.append(o)
        elif neg and other:
            if isinstance(neg[0], Number) and not (isinstance(neg[0], Integer) and neg[0].p == -1):
                other.append(Integer(-1))
                nonneg.append(-neg[0])
            else:
                other.extend(neg)
        else:
            other.extend(neg)
        other.extend(nc)
        cargs = nonneg
    evaluated: list[Expr] = []
    kept: list[Expr] = []
    if cargs:
        if isinstance(ex, Rational):
            npow = [f for f in cargs if isinstance(f, Pow) and isinstance(f.exp, Rational) and f.base.is_number]
            cargs = [f for f in cargs if not (isinstance(f, Pow) and isinstance(f.exp, Rational) and f.base.is_number)]
            evaluated = [pow(f.doit(), ex) for f in npow]
        kept = list(cargs)
    if other:
        kept.append(mul(*other))
    # SymPy's Mul(...) of the unevaluated powers evaluates them (Mul.flatten
    # rebuilds each power), a single one stays as it is
    if not evaluated and len(kept) == 1:
        return Pow(kept[0], ex)
    return mul(*evaluated, *[pow(f, ex) for f in kept])


# PORTED FROM SYMPY: sympy/core/expr.py:Expr._all_nonneg_or_nonppos
def _all_nonneg_or_nonpos(e: Add) -> bool:
    """Whether the terms are all nonnegative or all nonpositive."""
    nn = np = False
    for a in e.args:
        if not isinstance(a, Expr):
            return False
        if a.is_nonnegative is True:
            if np:
                return False
            nn = True
        elif a.is_nonpositive is True:
            if nn:
                return False
            np = True
        else:
            return False
    return True


def _is_exp_node(e: Expr) -> bool:
    from sympy_extras.functions.elementary.exponential import Exp
    return isinstance(e, Exp)


def _is_log_node(e: Expr) -> bool:
    from sympy_extras.functions.elementary.exponential import Log
    return isinstance(e, Log)


def _expanded_denominator(term: Expr) -> Expr:
    """A term whose denominator is a product with a sum written over the
    expanded denominator: ``p/(2*(x - y))`` becomes ``p/(2*x - 2*y)``, as
    SymPy's expansion does."""
    numerator, denominator = term.as_numer_denom()
    if not isinstance(denominator, Mul) or not any(isinstance(f, Add) for f in denominator.args):
        return term
    expanded = _expand(denominator, False, True, False, False, False)
    return mul(numerator, pow(expanded, Integer(-1)))


# PORTED FROM SYMPY: sympy/core/mul.py:Mul._eval_expand_mul
def _expand_mul_node(e: Mul, deep: bool) -> Expr:
    """SymPy's distribution of a product: the numerator and the
    denominator split first (a product denominator expanded, so that
    ``p/((c + 2)*t)`` keeps ``t*c + 2*t`` and no term cancels ``t``), then
    the sums of the numerator multiplied out."""
    expr: Expr = e
    # the denominator matters only when it is a product (two factors at least)
    if sum(1 for f in e.args if _denominator_factor(as_expr(f))) >= 2:
        from sympy_extras.simplify.radsimp import fraction
        n, d = fraction(e)
        if isinstance(d, Mul):
            n = _expand_mul_node(n, deep) if isinstance(n, Mul) else n
            d = _expand_mul_node(d, deep)
        # SymPy rebuilds n/d in any case: the denominator 2*(a + b) of
        # fraction is already distributed
        expr = mul(n, pow(d, Integer(-1)))
    if not isinstance(expr, Mul):
        return expr
    plain: list[Expr] = []
    sums: list[Expr] = []
    for f in expr.args:
        f = as_expr(f)
        if isinstance(f, Add):
            sums.append(f)
        elif f.is_commutative is not False:
            plain.append(f)
        else:
            sums.append(f)
    if not any(isinstance(t, Add) for t in sums):
        return expr
    plain_product = mul(*plain)
    result: list[Expr] = []
    for term in _expandsums(sums):
        t = mul(plain_product, term)
        if isinstance(t, Mul) and any(isinstance(a, Add) for a in t.args) and deep:
            t = _expand_mul_node(t, deep)
        result.append(t)
    return add(*result)


def _denominator_factor(f: Expr) -> bool:
    """Whether ``fraction`` puts (part of) ``f`` in the denominator."""
    from .numbers import Rational
    from .power import Pow
    if isinstance(f, Rational):
        return f.q != 1
    if isinstance(f, Pow) or _is_exp_node(f):
        exponent = f.as_base_exp()[1]
        return exponent.is_negative is True or isinstance(exponent, Mul)
    return False


# PORTED FROM SYMPY: sympy/core/mul.py:Mul._expandsums
def _expandsums(sums: list[Expr]) -> tuple[Expr, ...]:
    if len(sums) == 1:
        return terms_of(sums[0])
    half = len(sums) // 2
    left, right = _expandsums(sums[:half]), _expandsums(sums[half:])
    return terms_of(add(*[mul(a, b) for a in left for b in right]))


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
    return expand(expr, deep=deep, mul=True, multinomial=False, power_base=False, power_exp=False, log=False)


def expand_multinomial(expr: ExprLike, deep: bool = True) -> Expr:
    """Integer powers of sums expanded only.

    Examples
    ========

    >>> from sympy_extras.core import symbols, expand_multinomial
    >>> x, y = symbols('x y')
    >>> expand_multinomial((x + y)**2)
    x**2 + 2*x*y + y**2
    """
    return expand(expr, deep=deep, mul=False, multinomial=True, power_base=False, power_exp=False, log=False)


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
    return expand(expr, deep=deep, mul=False, multinomial=False, power_base=True, power_exp=False, log=False,
                  force=force)


def expand_power_exp(expr: ExprLike, deep: bool = True) -> Expr:
    """Powers with a sum in the exponent split into products.

    Examples
    ========

    >>> from sympy_extras.core import symbols, expand_power_exp
    >>> x, y = symbols('x y')
    >>> expand_power_exp(2**(x + y))
    2**x*2**y
    """
    return expand(expr, deep=deep, mul=False, multinomial=False, power_base=False, power_exp=True, log=False)


# PORTED FROM SYMPY: sympy/core/function.py:expand_log
def expand_log(expr: ExprLike, deep: bool = True, force: bool = False, factor: bool = False) -> Expr:
    """SymPy's ``expand_log``: logarithms of products and powers split
    where valid (positive factors and bases, real exponents; anything with
    ``force``), perfect powers taken out of integer arguments (all prime
    factors with ``factor``); a quotient of logarithms of integers like
    ``log(8)/log(2)`` is simplified first.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, expand_log, Integer
    >>> from sympy_extras.functions import log
    >>> p, q, x = Symbol('p', positive=True), Symbol('q', positive=True), Symbol('x')
    >>> expand_log(log(p*q)), expand_log(log(p**2/q)), expand_log(log(x*p)), expand_log(log(Integer(8)))
    (log(p) + log(q), 2*log(p) - log(q), log(p) + log(x), 3*log(2))
    >>> expand_log(log(x**2), force=True), expand_log(log(Integer(12)), factor=True), expand_log(log(Integer(8))/log(Integer(2)))
    (2*log(x), log(3) + 2*log(2), 3)
    """
    e = to_expr(expr)
    if not factor:
        e = as_expr(e.replace(_is_log_quotient, lambda x: _handle_log_quotient(as_expr(x), deep, force)))
    # SymPy's Expr.expand repeats the log hint until nothing changes
    while True:
        was = e
        e = _expand_logs(e, deep, force, factor)
        if e == was:
            return e


def _is_log_quotient(x: Basic) -> bool:
    from sympy_extras.functions.elementary.exponential import Log
    from .numbers import Rational
    if not isinstance(x, Mul):
        return False
    return all(any(isinstance(i, Log) and isinstance(i.args[0], Rational) for i in factors_of(j))
               for j in x.as_numer_denom())


def _count_logs(e: Basic) -> int:
    from sympy_extras.functions.elementary.exponential import Log
    from .basic import preorder_traversal
    return sum(1 for node in preorder_traversal(e) if isinstance(node, Log))


# PORTED FROM SYMPY: sympy/core/function.py:expand_log (_handleMul)
def _handle_log_quotient(x: Expr, deep: bool, force: bool) -> Expr:
    from sympy_extras.functions.elementary.exponential import Log, log
    from sympy_extras.simplify.radsimp import fraction
    from sympy_extras.simplify.powsimp import multiplicity
    from .numbers import Integer
    n, d = fraction(x)
    ns = [i for i in n.atoms(Log) if isinstance(i.args[0], Integer)]
    ds = [i for i in d.atoms(Log) if isinstance(i.args[0], Integer)]
    if len(ns) == 1 and len(ds) == 1:
        nl, dl = as_expr(ns[0]), as_expr(ds[0])
        na, da = nl.args[0], dl.args[0]
        if isinstance(na, Integer) and isinstance(da, Integer):
            m = multiplicity(da, na)
            if m:
                r = add(Integer(m), mul(log(Integer(na.p // da.p ** m)), pow(dl, Integer(-1))))
                x = as_expr(x.subs(nl, mul(dl, r)))
    x1 = expand_mul(expand_log(x, deep=deep, force=force, factor=True))
    if _count_logs(x1) <= _count_logs(x):
        return x1
    return x


def _expand_logs(e: Expr, deep: bool, force: bool, factor: bool) -> Expr:
    """The ``log`` hint of SymPy's ``expand``: the arguments first (with
    ``deep``), then the node."""
    from sympy_extras.functions.elementary.exponential import Log
    if deep and e.args and not isinstance(e, (Derivative, UnevaluatedExpr)):
        args = tuple(_expand_logs(a, deep, force, factor) if isinstance(a, Expr) else a for a in e.args)
        if any(new is not old and new != old for new, old in zip(args, e.args)):
            e = as_expr(e._evaluated(args))
    if isinstance(e, Log):
        return _expand_log_node(e, force, factor)
    return e


# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:log._eval_expand_log
def _expand_log_node(node: Expr, force: bool, factor: bool) -> Expr:
    from sympy_extras.functions.elementary.complexes import unpolarify
    from sympy_extras.functions.elementary.exponential import Log, log
    from sympy_extras.ntheory.factor_ import factorint, perfect_power
    from .numbers import Integer, Rational
    from .power import Pow
    arg = as_expr(node.args[0])
    if isinstance(arg, Integer):
        logarg: Optional[Expr] = None
        coeff: Expr = Integer(1)
        value = arg.p
        power = perfect_power(value) if value > 1 else None
        if power is not None:
            value, k = power
            coeff = Integer(k)
            logarg = log(Integer(value))
        if factor:
            primes = factorint(value) if value > 1 else {}
            if value not in primes:
                logarg = add(*[mul(Integer(k), log(Integer(p))) for p, k in primes.items()])
        if logarg is not None:
            return mul(coeff, logarg)
    elif isinstance(arg, Rational):
        return add(log(Integer(arg.p)), mul(Integer(-1), log(Integer(arg.q))))
    elif isinstance(arg, Mul):
        terms: list[Expr] = []
        nonpos: list[Expr] = []
        for x in arg.args:
            x = as_expr(x)
            if force or x.is_positive is True or x.is_polar is True:
                a = log(x)
                terms.append(_expand_log_node(a, force, factor) if isinstance(a, Log) else a)
            elif x.is_negative is True:
                terms.append(log(mul(Integer(-1), x)))
                nonpos.append(Integer(-1))
            else:
                nonpos.append(x)
        return add(add(*terms), log(mul(*nonpos)))
    elif isinstance(arg, Pow) or _is_exp_node(arg):
        b, ex = arg.as_base_exp()
        if force or (ex.is_extended_real is True and (b.is_positive is True or (
                add(ex, Integer(1)).is_positive is True and add(ex, Integer(-1)).is_nonpositive is True))) \
                or b.is_polar is True:
            a = log(b)
            if isinstance(a, Log):
                return mul(unpolarify(ex), _expand_log_node(a, force, factor))
            return mul(unpolarify(ex), a)
    return log(arg)


def expand_func(expr: ExprLike, deep: bool = True) -> Expr:
    """SymPy's ``expand_func``: the special functions rewritten.

    Examples
    ========

    >>> from sympy_extras.core import expand_func, sympify
    >>> expand_func(sympify('gamma(x + 1)'))
    x*gamma(x)
    """
    return expand(expr, deep=deep, mul=False, multinomial=False, power_base=False, power_exp=False, log=False,
                  func=True)


def expand_trig(expr: ExprLike, deep: bool = True) -> Expr:
    """SymPy's ``expand_trig``.

    Examples
    ========

    >>> from sympy_extras.core import expand_trig, sympify
    >>> expand_trig(sympify('sin(2*x)'))
    2*sin(x)*cos(x)
    """
    return expand(expr, deep=deep, mul=False, multinomial=False, power_base=False, power_exp=False, log=False,
                  trig=True)


def expand_complex(expr: ExprLike, deep: bool = True) -> Expr:
    """SymPy's ``expand_complex``: the expression as real and imaginary parts.

    Examples
    ========

    >>> from sympy_extras.core import expand_complex, sympify
    >>> expand_complex(sympify('x + I*y'))
    I*(re(y) + im(x)) + re(x) - im(y)
    """
    return expand(expr, deep=deep, mul=False, multinomial=False, power_base=False, power_exp=False, log=False,
                  complex=True)


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


# PORTED FROM SYMPY: sympy/core/function.py:_coeff_isneg
def _coeff_isneg(a: Expr) -> bool:
    """Whether the leading numerical coefficient of ``a`` is negative."""
    first = a.args[0] if isinstance(a, Mul) and a.args else a
    return isinstance(first, Number) and first.is_extended_negative is True


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
        # SymPy: every term negative gives NEG and (n - 1) SUB; otherwise a
        # SUB per negative term and an ADD per positive term but one
        # the counts do not depend on the order of the terms
        terms = [t for t in e.args if isinstance(t, Expr)]
        negatives = [_coeff_isneg(t) for t in terms]
        if all(negatives):
            _tally(counts, 'NEG')
            _tally(counts, 'SUB', len(terms) - 1)
        else:
            _tally(counts, 'SUB', sum(1 for n in negatives if n))
            _tally(counts, 'ADD', sum(1 for n in negatives if not n) - 1)
        for t, negative in zip(terms, negatives):
            _count(neg(t) if negative else t, counts)
        return
    if isinstance(e, Exp1):
        _tally(counts, 'EXP')  # SymPy: E is exp(1)
        return
    if isinstance(e, Mul):
        # PORTED FROM SYMPY: sympy/core/function.py:count_ops (the Mul branch)
        from sympy_extras.simplify.radsimp import fraction
        a: Expr = e
        if _coeff_isneg(e):
            _tally(counts, 'NEG')
            first = a.args[0]
            if isinstance(first, Integer) and first.p == -1:
                rest = [f for f in a.args[1:] if isinstance(f, Expr)]
                a = rest[0] if len(rest) == 1 else Mul(*rest)
            else:
                a = neg(a)
        n, d = fraction(a)
        if isinstance(n, Integer):
            _tally(counts, 'DIV')
            _tally(counts, 'NEG', int(n.p < 0))
            _count(d, counts)
            return
        if not (isinstance(d, Integer) and d.p == 1):
            if not isinstance(d, Integer):
                _count(d, counts)
            _tally(counts, 'DIV')
            _count(n, counts)
            return
        if isinstance(a, Mul):
            _tally(counts, 'MUL', len(a.args) - 1)
            for f in a.args:
                _count(f, counts)
            return
        _count(a, counts)
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
        for operand in e.args:
            _count(operand, counts)
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
    head = e._count_ops_label()
    if head is not None:
        _tally(counts, head)
        for child in e.args:
            _count(child, counts)
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


# PORTED FROM SYMPY: sympy/core/symbol.py:uniquely_named_symbol
def _uniquely_named_symbol(name: str, e: Basic) -> Symbol:
    """A symbol named ``name`` (with a number appended when a symbol of
    ``e`` has the name: ``xi0``, ``xi1``, ...)."""
    names = {str(s) for s in e.atoms(Symbol)}
    if name not in names:
        return Symbol(name)
    k = 0
    while '%s%d' % (name, k) in names:
        k += 1
    return Symbol('%s%d' % (name, k))


def _diff_wrt(e: Basic) -> bool:
    """SymPy's ``_diff_wrt``: a symbol or an undefined function application."""
    return isinstance(e, (Symbol, AppliedUndef))


# PORTED FROM SYMPY: sympy/core/function.py:Function._eval_nseries
def _generic_function_nseries(f: Expr, x: Symbol, n: int, logx: Optional[Expr]) -> Optional[Expr]:
    """SymPy's generic expansion of a function: by its derivatives when the
    argument does not vanish (or with several arguments, as for the nodes
    ``hyper`` and ``meijerg``), by its Taylor terms in the argument
    otherwise; ``None`` for an argument going to an infinity (SymPy's
    asymptotic branch is not ported)."""
    from sympy_extras.core.limits import limit
    from sympy_extras.functions.elementary.integers import ceiling
    from .coreerrors import PoleError
    from .numbers import NaN
    from .order import Order
    args = [a for a in f.args if isinstance(a, Expr)]
    args0 = [limit(a, x, Integer(0)) for a in args]
    if any(a.is_finite is False for a in args0):
        # SymPy's asymptotic branch: the class's own _eval_aseries when the
        # arguments go to an infinity faster than logarithmically (the
        # classes without a ported one, and the logarithmic case, are left
        # to the foreign system)
        from .numbers import oo, zoo, nan
        if isinstance(f, Application) and type(f)._eval_aseries is not Application._eval_aseries:
            a0 = [limit(a.as_leading_term(x, logx=logx), x, Integer(0)) for a in args]
            if any(t.has(oo, -oo, zoo, nan) for t in a0):
                return f._eval_aseries(n, args0, x, logx)
        return None
    one_argument = (isinstance(f, DefinedFunction) and not type(f).sympy_variadic
                    and len(f.nargs) == 1 and 1 in f.nargs)
    if not one_argument or not (args0[0] == 0):
        e: Expr = f
        e1 = e.expand()
        if e == e1:
            if len(e.args) == 1:
                from sympy_extras.polys.polytools import cancel
                rebuilt = e._evaluated((cancel(args[0]),))
                assert isinstance(rebuilt, Expr)
                e = rebuilt
            term = e.subs(x, Integer(0))
            assert isinstance(term, Expr)
            if term.is_finite is False or isinstance(term, NaN):
                raise PoleError('Cannot expand %s around 0' % (f,))
            series: Expr = term
            fact: Expr = Integer(1)
            xi = _uniquely_named_symbol('xi', f)
            moved = e.subs(x, xi)
            assert isinstance(moved, Expr)
            e = moved
            for i in range(1, n):
                fact = fact * Rational(i)
                e = e.diff(xi)
                value = e.subs(xi, Integer(0))
                assert isinstance(value, Expr)
                if isinstance(value, NaN):
                    value = limit(e, xi, Integer(0))
                if value.is_finite is False:
                    raise PoleError('Cannot expand %s around 0' % (f,))
                series = series + (value * x**i / fact).expand()
            return series + Order(x**n, x)
        return e1.nseries(x, n=n, logx=logx)
    arg = args[0]
    terms: list[Expr] = []
    nterms = n + 2
    cf = Order(arg.as_leading_term(x), x).getn()
    if cf is not None and cf != 0:
        count = ceiling(Integer(n) / cf)
        assert isinstance(count, Integer)
        nterms = count.p
    g: Optional[Expr] = None
    for i in range(nterms):
        g = f.taylor_term(i, arg, g)
        g = g.nseries(x, n=n, logx=logx)
        terms.append(g)
    return add(*terms) + Order(x**n, x)


#: the classes whose SymPy counterpart has an _eval_as_leading_term of its
#: own (the others use Function._eval_as_leading_term, ported below)
_SYMPY_OWN_LEADING_TERM: Final[frozenset[str]] = frozenset({
    'Acos', 'Acosh', 'Acot', 'Acoth', 'Acsc', 'Acsch', 'Arg', 'Asec', 'Asech', 'Asin', 'Asinh', 'Atan', 'Atanh',
    'BesselI', 'BesselJ', 'BesselK', 'BesselY', 'Binomial', 'Ceiling', 'ChiNode', 'CiNode', 'Cos', 'Cosh', 'Cot',
    'Coth', 'Csc', 'Csch', 'DiGamma', 'EiNode', 'Erf', 'Erfc', 'Erfi', 'Exp', 'Factorial', 'Floor', 'Frac',
    'FresnelC', 'FresnelS', 'Gamma', 'LambertW', 'Log', 'PolyGamma', 'Sec', 'Sech', 'ShiNode', 'SiNode', 'Sin',
    'SingularityFunctionNode', 'Sinh', 'Tan', 'Tanh', 'TriGamma', 'Zeta'})


# PORTED FROM SYMPY: sympy/core/function.py:Function._eval_as_leading_term
def _generic_leading_term(f: DefinedFunction, x: Symbol, logx: Optional[Expr]) -> Expr:
    """The function itself, unless an argument vanishes at ``x = 0`` (an
    argument in ``O(1)`` depending on ``x``): then SymPy raises, the class
    needing a rule of its own."""
    from .order import Order
    args = [a.as_leading_term(x, logx=logx) for a in f.args if isinstance(a, Expr)]
    o = Order(Integer(1), x)
    if any(x in a.free_symbols and o.contains(a) for a in args):
        raise NotImplementedError('%s has no _eval_as_leading_term routine' % f.function_name)
    return f


#: the classes whose SymPy counterpart has an ``_eval_is_meromorphic`` of
#: its own (ported in sympy_extras.functions._meromorphic; the others use
#: Application's)
_SYMPY_OWN_MEROMORPHIC: Final[frozenset[str]] = frozenset({
    'BesselI', 'BesselJ', 'BesselK', 'BesselY', 'Hankel1', 'Hankel2', 'LowerGamma', 'SphericalHankel1',
    'SphericalHankel2', 'SphericalJ', 'SphericalY', 'UpperGamma'})


#: the classes whose SymPy counterpart has an ``_eval_expand_func`` (the
#: others are not sent to the bridge; checked against SymPy by a test)
_SYMPY_EXPAND_FUNC: Final[frozenset[str]] = frozenset({
    'AiryAi', 'AiryAiPrime', 'AiryBi', 'AiryBiPrime', 'BesselI', 'BesselJ', 'BesselK', 'BesselY', 'Beta', 'Binomial',
    'DiGamma', 'Erf2', 'Erfc', 'Erfi', 'Expint', 'Gamma', 'Hankel1', 'Hankel2', 'Harmonic', 'LerchPhi', 'LogGamma',
    'PolyGamma', 'PolyLog', 'SphericalHankel1', 'SphericalHankel2', 'SphericalJ', 'SphericalY', 'TriGamma', 'YnmNode',
    'Zeta', 'Hyper', 'MeijerG', 'GoldenRatioType', 'TribonacciConstantType'})

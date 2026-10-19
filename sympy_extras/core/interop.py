"""Conversion between this core and SymPy's.

``to_sympy`` builds the SymPy expression of a tree (evaluated by SymPy's
constructors, or with ``evaluate=False`` to keep the structure);
``from_sympy`` builds the tree of a SymPy expression. Functions of SymPy
which have no class here (``sin``, ``exp``, ``log``, ...) are represented
by :class:`~sympy_extras.core.function.AppliedSymPyFunction`, and any
other object of SymPy (``Integral``, ``Piecewise``, ``And``, a set, ...)
by :class:`SymPyExprNode` or :class:`SymPyNode`, generic wrappers which
keep the SymPy class and the converted arguments, so that every
expression of SymPy has a tree here and the round trip is exact; symbols
keep the assumptions given to their constructor; floats keep their
binary value and precision. ``through_sympy(e, f)`` applies a function of
SymPy to a tree: the methods of ``Expr`` which are algorithms of other
packages of SymPy (``series``, ``simplify``, ``integrate``, ...) use it.
"""
from __future__ import annotations

from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

import mpmath
import sympy
from sympy.core.basic import Basic as SymPyBasic
from sympy.core.expr import Expr as SymPyExpr
from sympy.core.function import AppliedUndef as SymPyAppliedUndef
from sympy.core.function import Function as SymPyFunctionClass
from sympy.core.function import UndefinedFunction as SymPyUndefinedFunction
from sympy.assumptions.assume import AppliedPredicate as SymPyAppliedPredicate
from sympy.assumptions.assume import Predicate as SymPyPredicate
from sympy.core.relational import Relational as SymPyRelational

from .add import Add
from .assume import AppliedPredicate, Predicate
from .basic import Basic, SubsMapping
from .containers import Dict, Tuple
from .expr import Expr, UnevaluatedExpr, as_expr
from .function import (AppliedSymPyFunction, AppliedUndef, Derivative, Function, FunctionClass, Lambda,
    Subs, SymPyFunction, UndefinedFunction, WildFunction)
from .logic import ITE, And, Boolean, BooleanAtom, BooleanFunction, Equivalent, Implies, Not, Or, Xor, false, true
from .mod import Mod
from .mul import Mul
from .numbers import (AlgebraicNumber, Catalan, CatalanType, ComplexInfinity, EulerGamma,
    EulerGammaType, Exp1, Float, GoldenRatio, GoldenRatioType, ImaginaryUnit, Infinity, Integer,
    NaN, NegativeInfinity, Number, Pi, Rational, TribonacciConstant, TribonacciConstantType, E, I, nan,
    number_from_fraction, oo, pi, zoo)
from .order import Order
from .power import Pow
from .relational import Eq, Ge, Gt, Le, Lt, Ne, Relational
from .sorting import ClassKey, CompareItem
from .symbol import Dummy, Symbol, Wild

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['to_sympy', 'from_sympy', 'expr_from_sympy', 'sympy_expr', 'through_sympy', 'expr_through_sympy',
    'rewrite_targets', 'SymPyNode', 'SymPyExprNode']

#: the dummies already converted, so that a dummy bound in one place and
#: used in another stays one dummy on both sides
_dummies_from_sympy: dict[tuple[str, int], Dummy] = {}
_dummies_to_sympy: dict[tuple[str, int], sympy.Dummy] = {}
#: the objects of SymPy already converted (cleared when full, and by
#: ``clear_cache``)
_from_sympy_cache: dict[SymPyBasic, Basic] = {}
_FROM_SYMPY_CACHE_SIZE = 20000


class _ConversionCache:
    """The registration of the conversion cache with ``clear_cache``."""

    def clear(self) -> None:
        _from_sympy_cache.clear()
        _dummies_from_sympy.clear()
        _dummies_to_sympy.clear()

_RELATIONAL_TO_SYMPY: dict[type[Relational], type[SymPyRelational]] = {
    Eq: sympy.Eq, Ne: sympy.Ne, Lt: sympy.Lt, Le: sympy.Le, Gt: sympy.Gt, Ge: sympy.Ge}
_RELATIONAL_FROM_SYMPY: dict[type[SymPyRelational], type[Relational]] = {
    sympy.Eq: Eq, sympy.Ne: Ne, sympy.Lt: Lt, sympy.Le: Le, sympy.Gt: Gt, sympy.Ge: Ge}
_LOGIC_TO_SYMPY: dict[type[BooleanFunction], Callable[..., SymPyBasic]] = {
    And: sympy.And, Or: sympy.Or, Not: sympy.Not, Xor: sympy.Xor, Implies: sympy.Implies,
    Equivalent: sympy.Equivalent, ITE: sympy.ITE}
#: the predicates of SymPy's ``Q`` by name
_SYMPY_PREDICATES: dict[str, SymPyPredicate] = {
    'algebraic': sympy.Q.algebraic, 'antihermitian': sympy.Q.antihermitian, 'commutative': sympy.Q.commutative,
    'complex': sympy.Q.complex, 'complex_elements': sympy.Q.complex_elements, 'composite': sympy.Q.composite,
    'diagonal': sympy.Q.diagonal, 'eq': sympy.Q.eq, 'even': sympy.Q.even,
    'extended_negative': sympy.Q.extended_negative, 'extended_nonnegative': sympy.Q.extended_nonnegative,
    'extended_nonpositive': sympy.Q.extended_nonpositive, 'extended_nonzero': sympy.Q.extended_nonzero,
    'extended_positive': sympy.Q.extended_positive, 'extended_real': sympy.Q.extended_real,
    'finite': sympy.Q.finite, 'fullrank': sympy.Q.fullrank, 'ge': sympy.Q.ge, 'gt': sympy.Q.gt,
    'hermitian': sympy.Q.hermitian, 'imaginary': sympy.Q.imaginary, 'infinite': sympy.Q.infinite,
    'integer': sympy.Q.integer, 'integer_elements': sympy.Q.integer_elements, 'invertible': sympy.Q.invertible,
    'irrational': sympy.Q.irrational, 'is_true': sympy.Q.is_true, 'le': sympy.Q.le,
    'lower_triangular': sympy.Q.lower_triangular, 'lt': sympy.Q.lt, 'ne': sympy.Q.ne,
    'negative': sympy.Q.negative, 'negative_infinite': sympy.Q.negative_infinite,
    'noninteger': sympy.Q.noninteger, 'nonnegative': sympy.Q.nonnegative, 'nonpositive': sympy.Q.nonpositive,
    'nonzero': sympy.Q.nonzero, 'normal': sympy.Q.normal, 'odd': sympy.Q.odd, 'orthogonal': sympy.Q.orthogonal,
    'positive': sympy.Q.positive, 'positive_definite': sympy.Q.positive_definite,
    'positive_infinite': sympy.Q.positive_infinite, 'prime': sympy.Q.prime, 'rational': sympy.Q.rational,
    'real': sympy.Q.real, 'real_elements': sympy.Q.real_elements, 'singular': sympy.Q.singular,
    'square': sympy.Q.square, 'symmetric': sympy.Q.symmetric, 'transcendental': sympy.Q.transcendental,
    'triangular': sympy.Q.triangular, 'unit_triangular': sympy.Q.unit_triangular, 'unitary': sympy.Q.unitary,
    'upper_triangular': sympy.Q.upper_triangular, 'zero': sympy.Q.zero}
_LOGIC_FROM_SYMPY: dict[type[SymPyBasic], type[BooleanFunction]] = {
    sympy.And: And, sympy.Or: Or, sympy.Not: Not, sympy.Xor: Xor, sympy.Implies: Implies,
    sympy.Equivalent: Equivalent, sympy.ITE: ITE}


class _SymPyNodeMixin:
    """The behaviour shared by the generic wrappers of SymPy objects."""

    __slots__ = ()

    _sympy_class: type[SymPyBasic]
    _node_args: tuple[Basic, ...]

    @property
    def args(self) -> tuple[Basic, ...]:
        return self._node_args

    @property
    def sympy_class(self) -> type[SymPyBasic]:
        """The class of SymPy of the wrapped object."""
        return self._sympy_class

    @property
    def name(self) -> str:
        return self._sympy_class.__name__

    def _hashable_content(self) -> tuple[Basic, ...]:
        return (Symbol(self._sympy_class.__name__),) + self._node_args

    def _compare_name(self) -> str:
        return self._sympy_class.__name__

    def _compare_content(self) -> tuple[CompareItem, ...]:
        return tuple(a.compare_key() for a in self._node_args)

    def _sympy(self) -> SymPyBasic:
        return self._sympy_class(*[to_sympy(a) for a in self._node_args])

    def _sympystr(self, printer: StrPrinter) -> str:
        return sympy.sstr(self._sympy())


class SymPyNode(_SymPyNodeMixin, Basic):
    """A wrapper of an object of SymPy which is not an expression (a
    set, ...): the class of SymPy and the converted arguments.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import from_sympy, to_sympy, Symbol
    >>> x = sympy.Symbol('x')
    >>> t = from_sympy(sympy.Interval(0, x))
    >>> t, type(t).__name__, t.args, to_sympy(t) == sympy.Interval(0, x)
    (Interval(0, x), 'SymPyNode', (0, x, False, False), True)
    """

    __slots__ = ('_sympy_class', '_node_args')

    def __init__(self, sympy_class: type[SymPyBasic], *args: Basic) -> None:
        self._sympy_class: type[SymPyBasic] = sympy_class
        self._node_args: tuple[Basic, ...] = args
        Basic.__init__(self)

    @property
    def func(self) -> Callable[..., Basic]:
        return partial(SymPyNode, self._sympy_class)

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, 'SymPyNode')

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return from_sympy(self._sympy_class(*[to_sympy(a) for a in args]))

    def doit(self) -> Basic:
        return from_sympy(self._sympy().doit())


class SymPyExprNode(_SymPyNodeMixin, Expr):
    """A wrapper of an expression of SymPy with no class here (an
    ``Integral``, a ``Sum``, a ``Piecewise``, ...): ``doit`` evaluates
    it with SymPy, the assumptions are SymPy's.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import Symbol
    >>> x = Symbol('x')
    >>> i = (x**2).integrate()
    >>> i
    x**3/3
    >>> t = (1/x).integrate((x, 1, Symbol('t')))
    >>> t, type(t).__name__, t.doit()
    (log(t), 'AppliedSymPyFunction', log(t))
    """

    __slots__ = ('_sympy_class', '_node_args')

    def __init__(self, sympy_class: type[SymPyBasic], *args: Basic) -> None:
        self._sympy_class: type[SymPyBasic] = sympy_class
        self._node_args: tuple[Basic, ...] = args
        Basic.__init__(self)

    @property
    def func(self) -> Callable[..., Basic]:
        return partial(SymPyExprNode, self._sympy_class)

    @classmethod
    def class_key(cls) -> ClassKey:
        return (4, 2, 'SymPyExprNode')

    @property
    def free_symbols(self) -> set[Symbol]:
        return {_symbol(s) for s in self._sympy().free_symbols}

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return from_sympy(self._sympy_class(*[to_sympy(a) for a in args]))

    def doit(self) -> Expr:
        return expr_from_sympy(self._sympy().doit())

    def evalf(self, n: int = 15, subs: Optional[SubsMapping] = None) -> Expr:
        node: Expr = self
        if subs:
            node = as_expr(self.subs(subs))
        s = node._sympy() if isinstance(node, SymPyExprNode) else to_sympy(node)
        if not isinstance(s, SymPyExpr):
            raise TypeError("an expression is expected")
        return expr_from_sympy(s.evalf(n))

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import sympy_facts
        return sympy_facts(self)


def to_sympy(e: Basic, evaluate: bool = True) -> SymPyBasic:
    """The SymPy expression of ``e``.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import symbols, Add, to_sympy
    >>> x = symbols('x')
    >>> s = to_sympy(x**2 + 1)
    >>> s, type(s).__module__
    (x**2 + 1, 'sympy.core.add')
    >>> to_sympy(Add(x, x)), to_sympy(Add(x, x), evaluate=False)
    (2*x, x + x)
    """
    if evaluate:
        cached = e._sympy_cache
        if cached is None:
            cached = _to_sympy(e, True)
            e._sympy_cache = cached
        return cached
    return _to_sympy(e, False)


def _to_sympy(e: Basic, evaluate: bool) -> SymPyBasic:
    """The conversion itself; ``to_sympy`` caches the evaluated form on
    the node, so that the methods forwarded to SymPy convert a node once."""
    if isinstance(e, Wild):
        return sympy.Wild(e.name, exclude=[to_sympy(x) for x in e.exclude],
                          properties=[_property_on_sympy(p) for p in e.properties], **e.given_assumptions)
    if isinstance(e, Dummy):
        key = (e.name, e.dummy_index)
        if key not in _dummies_to_sympy:
            _dummies_to_sympy[key] = sympy.Dummy(e.name, **e.given_assumptions)
        return _dummies_to_sympy[key]
    if isinstance(e, Symbol):
        return sympy.Symbol(e.name, **e.given_assumptions)
    if isinstance(e, WildFunction):
        return sympy.WildFunction(e.name) if e.nargs is None else sympy.WildFunction(e.name, nargs=tuple(sorted(e.nargs)))
    if isinstance(e, Rational):
        return sympy.Rational(e.p, e.q)
    if isinstance(e, Float):
        value = e.as_fraction()
        with mpmath.workprec(e.precision):
            return sympy.Float(mpmath.mpf(value.numerator) / value.denominator, precision=e.precision)
    if isinstance(e, Pi):
        return sympy.pi
    if isinstance(e, Exp1):
        return sympy.E
    if isinstance(e, EulerGammaType):
        return sympy.EulerGamma
    if isinstance(e, CatalanType):
        return sympy.Catalan
    if isinstance(e, GoldenRatioType):
        return sympy.GoldenRatio
    if isinstance(e, TribonacciConstantType):
        return sympy.TribonacciConstant
    if isinstance(e, ImaginaryUnit):
        return sympy.I
    if isinstance(e, Infinity):
        return sympy.oo
    if isinstance(e, NegativeInfinity):
        return -sympy.oo
    if isinstance(e, ComplexInfinity):
        return sympy.zoo
    if isinstance(e, NaN):
        return sympy.nan
    if isinstance(e, BooleanAtom):
        return sympy.true if bool(e) else sympy.false
    if isinstance(e, AlgebraicNumber):
        alias = None if e.alias is None else e.alias.name
        return sympy.AlgebraicNumber(to_sympy(e.root), [to_sympy(c) for c in e.coeffs()], alias=alias)
    if isinstance(e, (SymPyNode, SymPyExprNode)):
        return e._sympy()
    args = [to_sympy(a, evaluate) for a in e.args]
    if isinstance(e, Add):
        return sympy.Add(*[_sympy_expr(a) for a in args], evaluate=evaluate)
    if isinstance(e, Mul):
        if evaluate and len(args) == 2 and isinstance(e.args[0], Number) and isinstance(e.args[1], Add):
            # the undistributed product 2*(x + 2*y) of factor_terms stays one in SymPy
            return sympy.Mul(*[_sympy_expr(a) for a in args], evaluate=False)
        return sympy.Mul(*[_sympy_expr(a) for a in args], evaluate=evaluate)
    if isinstance(e, Pow):
        return sympy.Pow(*[_sympy_expr(a) for a in args], evaluate=evaluate)
    if isinstance(e, Mod):
        return sympy.Mod(*[_sympy_expr(a) for a in args], evaluate=evaluate)
    if isinstance(e, AppliedUndef):
        return sympy.Function(e.name, **e.function.assumptions)(*args)
    if isinstance(e, AppliedSymPyFunction):
        return e.sympy_class(*args, evaluate=evaluate)
    if isinstance(e, Derivative):
        counts = [(to_sympy(s), n) for s, n in e.variable_count]
        return sympy.Derivative(args[0], *counts)
    if isinstance(e, Subs):
        return sympy.Subs(args[0], tuple(to_sympy(v) for v in e.variables), tuple(to_sympy(p) for p in e.point))
    if isinstance(e, Order):
        limits = [(to_sympy(v), to_sympy(p)) for v, p in zip(e.variables, e.point)]
        return sympy.Order(args[0], *limits)
    if isinstance(e, UnevaluatedExpr):
        return sympy.UnevaluatedExpr(args[0])
    if isinstance(e, Lambda):
        return sympy.Lambda(tuple(to_sympy(v) for v in e.variables), to_sympy(e.expr, evaluate))
    if isinstance(e, Tuple):
        return sympy.Tuple(*args)
    if isinstance(e, Dict):
        return sympy.Dict({to_sympy(k): to_sympy(v) for k, v in e.items()})
    if isinstance(e, Relational):
        return _RELATIONAL_TO_SYMPY[type(e)](*args, evaluate=evaluate)
    if isinstance(e, BooleanFunction):
        return _LOGIC_TO_SYMPY[type(e)](*args, evaluate=evaluate)
    if isinstance(e, Predicate):
        return _sympy_predicate(e)
    if isinstance(e, AppliedPredicate):
        return _sympy_predicate(e.function)(*[to_sympy(a, evaluate) for a in e.arguments])
    raise TypeError("cannot convert %s to SymPy" % (e,))


def _sympy_predicate(p: Predicate) -> SymPyPredicate:
    if p._defined and p.name in _SYMPY_PREDICATES:
        return _SYMPY_PREDICATES[p.name]
    return SymPyPredicate(p.name)


def _property_on_sympy(p: Callable[[Basic], bool]) -> Callable[[SymPyBasic], bool]:
    return lambda s: p(from_sympy(s))


def _property_on_core(p: Callable[[SymPyBasic], bool]) -> Callable[[Basic], bool]:
    return lambda b: bool(p(to_sympy(b)))


def sympy_expr(e: SymPyBasic) -> SymPyExpr:
    """``e``, which must be a SymPy ``Expr`` (narrowing at the boundary)."""
    if not isinstance(e, SymPyExpr):
        raise TypeError("a SymPy expression is expected, got %s" % (e,))
    return e


_sympy_expr = sympy_expr


def _symbol_assumptions(s: sympy.Symbol) -> dict[str, bool]:
    """The assumptions of a SymPy symbol among the predicates known here
    (SymPy's ``assumptions0`` is the closure of the given ones; two symbols
    with the same closure are the same symbol, in SymPy and here)."""
    from .assumptions import PREDICATES, minimal_generators
    facts = s.assumptions0
    known = {str(k): bool(v) for k, v in facts.items() if str(k) in PREDICATES and v is not None}
    return minimal_generators(known)


def from_sympy(e: Union[SymPyBasic, int]) -> Basic:
    """The tree of a SymPy expression.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import from_sympy, srepr
    >>> x = sympy.Symbol('x')
    >>> t = from_sympy(sympy.sin(x)**2 + 1)
    >>> t, srepr(t)
    (sin(x)**2 + 1, "Add(Integer(1), Pow(sin(Symbol('x')), Integer(2)))")
    >>> srepr(from_sympy(sympy.Symbol('p', positive=True)))
    Symbol('p', positive=True)
    >>> from_sympy(sympy.Symbol('p', positive=True)).is_positive
    True
    """
    if isinstance(e, int):
        return Integer(e)
    cached = _from_sympy_cache.get(e)
    if cached is not None:
        return cached
    converted = _from_sympy(e)
    if len(_from_sympy_cache) >= _FROM_SYMPY_CACHE_SIZE:
        _from_sympy_cache.clear()
    _from_sympy_cache[e] = converted
    if converted._sympy_cache is None and not isinstance(e, sympy.Float):
        converted._sympy_cache = e  # the round trip is exact
    return converted


def _from_sympy(e: SymPyBasic) -> Basic:
    """The conversion itself; ``from_sympy`` caches the results."""
    if isinstance(e, sympy.Wild):
        # SymPy keeps exclude and properties as the last two entries of the
        # hashable content (they are set dynamically on the instance)
        content = e._hashable_content()
        exclude = [from_sympy(x) for x in content[-2] if isinstance(x, SymPyBasic)]
        properties = [_property_on_core(p) for p in content[-1] if callable(p)]
        return Wild(e.name, exclude=exclude, properties=properties, **_symbol_assumptions(e))
    if isinstance(e, sympy.Dummy):
        index = e._hashable_content()[-1]  # SymPy keeps the index last
        key = (e.name, int(index) if isinstance(index, int) else hash(index))
        if key not in _dummies_from_sympy:
            dummy = Dummy(e.name, **_symbol_assumptions(e))
            _dummies_from_sympy[key] = dummy
            _dummies_to_sympy[(dummy.name, dummy.dummy_index)] = e
        return _dummies_from_sympy[key]
    if isinstance(e, sympy.Symbol):
        return Symbol(e.name, **_symbol_assumptions(e))
    if isinstance(e, sympy.WildFunction):
        nargs = e.nargs
        if isinstance(nargs, sympy.FiniteSet):
            return WildFunction(e.name, [int(n) for n in nargs.args if isinstance(n, sympy.Integer)])
        return WildFunction(e.name)
    if isinstance(e, sympy.Integer):
        return Integer(int(e.p))
    if isinstance(e, sympy.Rational):
        return Rational(int(e.p), int(e.q))
    if isinstance(e, sympy.Float):
        sign, man, exp, _ = e._mpf_
        value = number_from_fraction(_dyadic(int(sign), int(man), int(exp)))
        return Float(value, precision=int(e._prec))
    if isinstance(e, sympy.AlgebraicNumber):
        root, coeffs = e.args[0], e.args[1]
        alias = str(e.args[2]) if len(e.args) > 2 else None
        if not isinstance(coeffs, sympy.Tuple):
            raise TypeError("cannot convert %s" % (e,))
        return AlgebraicNumber(_expr(root), [_expr(c) for c in coeffs.args], alias=alias)
    if e is sympy.pi:
        return pi
    if e is sympy.E:
        return E
    if e is sympy.EulerGamma:
        return EulerGamma
    if e is sympy.Catalan:
        return Catalan
    if e is sympy.GoldenRatio:
        return GoldenRatio
    if e is sympy.TribonacciConstant:
        return TribonacciConstant
    if e is sympy.I:
        return I
    if e is sympy.oo:
        return oo
    if e is -sympy.oo:
        return NegativeInfinity()
    if e is sympy.zoo:
        return zoo
    if e is sympy.nan:
        return nan
    if e is sympy.true:
        return true
    if e is sympy.false:
        return false
    if isinstance(e, sympy.Add):
        return Add(*_canonical_order([_expr(a) for a in e.args], commutative=True))
    if isinstance(e, sympy.Mul):
        return Mul(*_canonical_order([_expr(a) for a in e.args], commutative=False))
    if isinstance(e, sympy.Pow):
        return Pow(_expr(e.base), _expr(e.exp))
    if isinstance(e, sympy.Mod):
        return Mod(_expr(e.args[0]), _expr(e.args[1]))
    if isinstance(e, SymPyAppliedUndef):
        f = e.func
        if not isinstance(f, SymPyUndefinedFunction):
            raise TypeError("cannot convert %s" % (e,))
        from .assumptions import PREDICATES
        assumptions = {str(k): bool(v) for k, v in f._kwargs.items() if str(k) in PREDICATES and v is not None}
        return Function(f.__name__, **assumptions)(*[_expr(a) for a in e.args])
    if isinstance(e, sympy.Derivative):
        counts = [(_symbol(s), int(n)) for s, n in e.variable_count]
        return Derivative(_expr(e.expr), *counts)
    if isinstance(e, sympy.Subs):
        return Subs(_expr(e.expr), tuple(_symbol(v) for v in e.variables), tuple(_expr(p) for p in e.point))
    if isinstance(e, sympy.Order):
        limits = [(_symbol(v), _expr(p)) for v, p in zip(e.variables, e.point)]
        return Order(_expr(e.expr), *limits)
    if isinstance(e, sympy.UnevaluatedExpr):
        return UnevaluatedExpr(_expr(e.args[0]))
    if isinstance(e, sympy.Lambda):
        return Lambda(tuple(_symbol(v) for v in e.variables), _expr(e.expr))
    if isinstance(e, sympy.Dict):
        return Dict(*[(from_sympy(k), from_sympy(v)) for k, v in e.items()])
    if isinstance(e, sympy.Tuple):
        return Tuple(*[from_sympy(a) for a in e.args])
    if isinstance(e, SymPyRelational):
        for sympy_class, cls in _RELATIONAL_FROM_SYMPY.items():
            if isinstance(e, sympy_class):
                return cls(_expr(e.lhs), _expr(e.rhs))
        raise TypeError("cannot convert the relation %s" % (e,))
    for sympy_logic, logic_class in _LOGIC_FROM_SYMPY.items():
        if isinstance(e, sympy_logic):
            return logic_class(*[_boolean(a) for a in e.args])
    if isinstance(e, SymPyAppliedPredicate):
        return AppliedPredicate(_predicate(e.function), *[from_sympy(a) for a in e.arguments])
    if isinstance(e, SymPyPredicate):
        return _predicate(e)
    if isinstance(e, SymPyFunctionClass) and all(isinstance(a, SymPyExpr) for a in e.args):
        cls = type(e)
        return AppliedSymPyFunction(cls, *[_expr(a) for a in e.args])
    if isinstance(e, SymPyExpr):
        return SymPyExprNode(type(e), *[from_sympy(a) for a in e.args])
    if isinstance(e, SymPyBasic):
        return SymPyNode(type(e), *[from_sympy(a) for a in e.args])
    raise TypeError("cannot convert %s (%s) from SymPy" % (e, type(e).__name__))


def expr_from_sympy(e: SymPyBasic) -> Expr:
    """``from_sympy(e)``, which must be an expression."""
    return as_expr(from_sympy(e))


def through_sympy(e: Basic, f: Callable[[SymPyBasic], SymPyBasic]) -> Basic:
    """``f`` of SymPy applied to the SymPy form of ``e``, converted back.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.core.interop import through_sympy
    >>> x = Symbol('x')
    >>> through_sympy(x**2 - 1, sympy.factor)
    (x - 1)*(x + 1)
    """
    return from_sympy(f(to_sympy(e)))


def expr_through_sympy(e: Expr, f: Callable[[SymPyExpr], SymPyBasic]) -> Expr:
    """:func:`through_sympy` for an expression, with an expression as result."""
    return as_expr(from_sympy(f(_sympy_expr(to_sympy(e)))))


def rewrite_targets(args: Sequence[Union[Basic, FunctionClass, type]]) -> list[Union[SymPyBasic, type]]:
    """The arguments of ``rewrite`` as SymPy objects: nodes converted,
    function objects replaced by their SymPy class."""
    targets: list[Union[SymPyBasic, type]] = []
    for a in args:
        if isinstance(a, Basic):
            targets.append(to_sympy(a))
        elif isinstance(a, SymPyFunction):
            targets.append(a.sympy_class)
        elif isinstance(a, UndefinedFunction):
            targets.append(sympy.Function(a.name))
        elif isinstance(a, type):
            targets.append(a)
        else:
            raise TypeError("cannot rewrite in terms of %r" % (a,))
    return targets


def _canonical_order(args: list[Expr], commutative: bool) -> list[Expr]:
    """The arguments of a sum or a product of SymPy in the order the
    evaluating functions of this core give (SymPy's own order): the
    numbers first, then the commutative arguments in the ``compare``
    order, then the noncommutative ones as given."""
    from .numbers import Number
    from .sorting import compare_key
    numbers = [a for a in args if isinstance(a, Number)]
    others = [a for a in args if not isinstance(a, Number)]
    if commutative:
        return numbers + sorted(others, key=compare_key)
    sortable = [a for a in others if a.is_commutative is not False]
    fixed = [a for a in others if a.is_commutative is False]
    return numbers + sorted(sortable, key=compare_key) + fixed


def _dyadic(sign: int, man: int, exp: int) -> Fraction:
    value = Fraction(man) * Fraction(2) ** exp
    return -value if sign else value


def _expr(e: SymPyBasic) -> Expr:
    converted = from_sympy(e)
    if not isinstance(converted, Expr):
        raise TypeError("an expression is expected, got %s" % (converted,))
    return converted


def _predicate(p: SymPyBasic) -> Predicate:
    """The predicate of this core for a predicate of SymPy: the one of
    ``Q`` with the same name when SymPy's is in its ``Q``."""
    from .ask import PREDICATES_OF_Q
    if not isinstance(p, SymPyPredicate):
        raise TypeError("a predicate is expected, got %s" % (p,))
    name = str(p.name)
    if name in PREDICATES_OF_Q and _SYMPY_PREDICATES.get(name) == p:
        return PREDICATES_OF_Q[name]
    return Predicate(name)


def _boolean(e: SymPyBasic) -> Boolean:
    converted = from_sympy(e)
    if not isinstance(converted, Boolean):
        raise TypeError("a boolean is expected, got %s" % (converted,))
    return converted


def _symbol(e: SymPyBasic) -> Symbol:
    converted = from_sympy(e)
    if not isinstance(converted, Symbol):
        raise TypeError("a symbol is expected, got %s" % (converted,))
    return converted


from .cache import register_clearable  # noqa: E402 (the registration needs the class above)

register_clearable(_ConversionCache())

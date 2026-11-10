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
from typing import Callable, Final, Optional, Sequence, TYPE_CHECKING, Union

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
from sympy.functions.elementary.piecewise import ExprCondPair as SymPyExprCondPair

from sympy_extras.core.add import Add
from sympy_extras.core.assume import AppliedPredicate, Predicate
from sympy_extras.core.basic import Basic, SubsMapping
from sympy_extras.core.bridge import Bridge, RewriteTarget, register_bridge
from sympy_extras.core.containers import Dict, Tuple
from sympy_extras.core.assumptions import Facts, Truth
from sympy_extras.core.expr import Expr, ExprLike, UnevaluatedExpr, as_expr, to_expr
from sympy_extras.core.function import (_FUNCTION_RANKS, AppliedUndef, Application, DefinedFunction, Derivative, Function,
    FunctionClass, Lambda, Subs, UndefinedFunction, WildFunction)
from sympy_extras.core.kind import Kind, UndefinedKind
from sympy_extras.core.logic import ITE, And, Boolean, BooleanAtom, BooleanFunction, Equivalent, Implies, Not, Or, Xor, false, true
from sympy_extras.core.mod import Mod
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import (AlgebraicNumber, Catalan, CatalanType, ComplexInfinity, EulerGamma,
    EulerGammaType, Exp1, Float, GoldenRatio, GoldenRatioType, ImaginaryUnit, Infinity, Integer,
    NaN, NegativeInfinity, Number, Pi, Rational, TribonacciConstant, TribonacciConstantType, E, I, nan,
    number_from_fraction, oo, pi, zoo)
from sympy_extras.core.coreerrors import PoleError
from sympy_extras.core.order import Order
from sympy_extras.core.limits import Limit
from sympy_extras.core.accumbounds import AccumBounds
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational
from sympy_extras.core.sorting import ClassKey, CompareItem
from sympy_extras.core.symbol import Dummy, Symbol, Wild

if TYPE_CHECKING:
    from sympy_extras.core.printing import StrPrinter

__all__ = ['to_sympy', 'from_sympy', 'expr_from_sympy', 'sympy_expr', 'through_sympy', 'expr_through_sympy',
    'rewrite_targets', 'SymPyNode', 'SymPyExprNode', 'SymPyFunction', 'AppliedSymPyFunction', 'apply_sympy',
    'sympy_facts', 'SymPyBridge']

#: the dummies already converted, so that a dummy bound in one place and
#: used in another stays one dummy on both sides
_dummies_from_sympy: dict[tuple[str, int], Dummy] = {}
_dummies_to_sympy: dict[tuple[str, int], sympy.Dummy] = {}
#: the objects of SymPy already converted (cleared when full, and by
#: ``clear_cache``)
_from_sympy_cache: dict[SymPyBasic, Basic] = {}
_FROM_SYMPY_CACHE_SIZE = 20000
#: the SymPy forms of the nodes already converted, by identity of the node
#: (the node is kept with its form, so that its id is not reused)
_to_sympy_cache: dict[int, tuple[Basic, SymPyBasic]] = {}


class _ConversionCache:
    """The registration of the conversion cache with ``clear_cache``."""

    def clear(self) -> None:
        _from_sympy_cache.clear()
        _to_sympy_cache.clear()
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

    def _is_foreign_node(self) -> bool:
        return True

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
    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.interop import from_sympy, to_sympy
    >>> x = sympy.Symbol('x')
    >>> t = from_sympy(sympy.Range(0, x))
    >>> t, type(t).__name__, t.args, to_sympy(t) == sympy.Range(0, x)
    (Range(0, x, 1), 'SymPyNode', (0, x, 1), True)
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
    (log(t), 'Log', log(t))
    """

    __slots__ = ('_sympy_class', '_node_args')

    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        """SymPy's leading term of the wrapped expression (when its class
        has a rule of its own), else the node."""
        found = leading_term_node_through_sympy(self, x, logx, cdir)
        return self if found is None else found

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
        return sympy_facts(self)

    def _eval_factor_terms(self, radical: bool, clear: bool, fraction: bool, sign: bool) -> Optional[Basic]:
        # SymPy's handling of its own objects (the content of an Integral is taken out)
        return through_sympy(self, lambda e: sympy.factor_terms(sympy_expr(e), radical=radical, clear=clear,
                                                                 fraction=fraction, sign=sign))


def to_sympy(e: Basic, evaluate: bool = True) -> SymPyBasic:
    """The SymPy expression of ``e``.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import symbols, Add
    >>> from sympy_extras.interop import to_sympy
    >>> x = symbols('x')
    >>> s = to_sympy(x**2 + 1)
    >>> s, type(s).__module__
    (x**2 + 1, 'sympy.core.add')
    >>> to_sympy(Add(x, x)), to_sympy(Add(x, x), evaluate=False)
    (2*x, x + x)
    """
    if evaluate:
        found = _to_sympy_cache.get(id(e))
        if found is not None and found[0] is e:
            return found[1]
        converted = _to_sympy(e, True)
        _remember(e, converted)
        return converted
    return _to_sympy(e, False)


def _remember(e: Basic, converted: SymPyBasic) -> None:
    if len(_to_sympy_cache) >= _FROM_SYMPY_CACHE_SIZE:
        _to_sympy_cache.clear()
    _to_sympy_cache[id(e)] = (e, converted)


def _to_sympy(e: Basic, evaluate: bool) -> SymPyBasic:
    """The conversion itself; ``to_sympy`` caches the evaluated form on
    the node, so that the methods forwarded to SymPy convert a node once."""
    if isinstance(e, Wild):
        return sympy.Wild(e.name, exclude=[to_sympy(x) for x in e.exclude],
                          properties=[_property_on_sympy(p) for p in e.properties], **e.given_assumptions)
    if isinstance(e, Dummy):
        key = (e.name, e.dummy_index)
        if key not in _dummies_to_sympy:
            converted = sympy.Dummy(e.name, **e.given_assumptions)
            _dummies_to_sympy[key] = converted
            # and back to the same dummy (not a new one)
            index = converted._hashable_content()[-1]
            _dummies_from_sympy[(e.name, int(index) if isinstance(index, int) else hash(index))] = e
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
    from sympy_extras.functions.elementary.piecewise import ExprCondPair, Piecewise
    if isinstance(e, Piecewise):
        return sympy.Piecewise(*[(to_sympy(p.expr, evaluate), to_sympy(p.cond, evaluate)) for p in e.args],
                               evaluate=evaluate)
    from sympy_extras.functions.special.hyper import Hyper, MeijerG
    if isinstance(e, Hyper):
        return sympy.hyper([to_sympy(a, evaluate) for a in e.ap], [to_sympy(b, evaluate) for b in e.bq],
                           to_sympy(e.argument, evaluate))
    if isinstance(e, MeijerG):
        return sympy.meijerg([to_sympy(a, evaluate) for a in e.an], [to_sympy(a, evaluate) for a in e.aother],
                             [to_sympy(b, evaluate) for b in e.bm], [to_sympy(b, evaluate) for b in e.bother],
                             to_sympy(e.argument, evaluate))
    if isinstance(e, ExprCondPair):
        return SymPyExprCondPair(to_sympy(e.expr, evaluate), to_sympy(e.cond, evaluate))
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
    if isinstance(e, DefinedFunction) and type(e).variadic:
        from sympy_extras.functions.elementary.miscellaneous import Max, Min
        if isinstance(e, Max):
            return sympy.Max(*args, evaluate=evaluate)
        if isinstance(e, Min):
            return sympy.Min(*args, evaluate=evaluate)
    if isinstance(e, DefinedFunction):
        sympy_class = _native_to_sympy().get(type(e))
        if sympy_class is None:
            raise TypeError('no SymPy counterpart of %s' % (e.function_name,))
        if sympy_class in (sympy.betainc, sympy.betainc_regularized):
            return sympy_class(*args)  # their __new__ takes no evaluate (they never evaluate)
        return sympy_class(*args, evaluate=evaluate)
    if isinstance(e, Derivative):
        counts = [(to_sympy(s), n) for s, n in e.variable_count]
        return sympy.Derivative(args[0], *counts)
    if isinstance(e, Subs):
        return sympy.Subs(args[0], tuple(to_sympy(v) for v in e.variables), tuple(to_sympy(p) for p in e.point))
    if isinstance(e, Order):
        limits = [(to_sympy(v), to_sympy(p)) for v, p in zip(e.variables, e.point)]
        return sympy.Order(args[0], *limits)
    if isinstance(e, Limit):
        return sympy.Limit(args[0], to_sympy(e.variable), to_sympy(e.point), e.direction)
    from sympy_extras.concrete.expr_with_limits import Product, Sum
    if isinstance(e, (Sum, Product)):
        sympy_limits = [sympy.Tuple(*[to_sympy(b) for b in lim]) for lim in e.limits]
        if isinstance(e, Sum):
            return sympy.Sum(args[0], *sympy_limits)
        return sympy.Product(args[0], *sympy_limits)
    if isinstance(e, AccumBounds):
        return sympy.AccumBounds(args[0], args[1])
    if isinstance(e, UnevaluatedExpr):
        return sympy.UnevaluatedExpr(args[0])
    if isinstance(e, Lambda):
        return sympy.Lambda(tuple(to_sympy(v) for v in e.variables), to_sympy(e.expr, evaluate))
    if isinstance(e, Tuple):
        return sympy.Tuple(*args)
    from sympy_extras.sets import sets as native_sets
    if isinstance(e, native_sets.Set) or isinstance(e, native_sets.Contains):
        return _set_to_sympy(e, args, evaluate)
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
    from sympy_extras.core.assumptions import PREDICATES, minimal_generators
    facts = s.assumptions0
    known = {str(k): bool(v) for k, v in facts.items() if str(k) in PREDICATES and v is not None}
    return minimal_generators(known)


def from_sympy(e: Union[SymPyBasic, int]) -> Basic:
    """The tree of a SymPy expression.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import srepr
    >>> from sympy_extras.interop import from_sympy
    >>> x = sympy.Symbol('x')
    >>> t = from_sympy(sympy.sin(x)**2 + 1)
    >>> t, srepr(t)
    (sin(x)**2 + 1, "Add(Pow(sin(Symbol('x')), Integer(2)), Integer(1))")
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
    if id(converted) not in _to_sympy_cache and not isinstance(e, sympy.Float):
        _remember(converted, e)  # the round trip is exact
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
        from sympy_extras.core.assumptions import PREDICATES
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
    if isinstance(e, sympy.Limit):
        return Limit(_expr(e.args[0]), _symbol(e.args[1]), _expr(e.args[2]), str(e.args[3]))
    if isinstance(e, (sympy.Sum, sympy.Product)) and all(len(lim) == 3 and isinstance(lim[0], sympy.Symbol)
                                                          for lim in e.limits):
        from sympy_extras.concrete.expr_with_limits import Product, Sum
        native_limits = [Tuple(_symbol(lim[0]), _expr(lim[1]), _expr(lim[2])) for lim in e.limits]
        native_class = Sum if isinstance(e, sympy.Sum) else Product
        return native_class(_expr(e.function), *native_limits)
    if isinstance(e, sympy.AccumBounds):
        return AccumBounds(_expr(e.args[0]), _expr(e.args[1]))
    if isinstance(e, sympy.UnevaluatedExpr):
        return UnevaluatedExpr(_expr(e.args[0]))
    if isinstance(e, sympy.Lambda):
        return Lambda(tuple(_symbol(v) for v in e.variables), _expr(e.expr))
    if isinstance(e, sympy.Dict):
        return Dict(*[(from_sympy(k), from_sympy(v)) for k, v in e.items()])
    if isinstance(e, sympy.Tuple):
        return Tuple(*[from_sympy(a) for a in e.args])
    if isinstance(e, (sympy.Set, sympy.Contains)):
        return _set_from_sympy(e)
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
    if isinstance(e, sympy.hyper):
        from sympy_extras.functions.special.hyper import Hyper
        return Hyper([_expr(a) for a in e.ap], [_expr(b) for b in e.bq], _expr(e.argument))
    if isinstance(e, sympy.meijerg):
        from sympy_extras.functions.special.hyper import MeijerG
        return MeijerG([_expr(a) for a in e.an], [_expr(a) for a in e.aother], [_expr(b) for b in e.bm],
                       [_expr(b) for b in e.bother], _expr(e.argument))
    if isinstance(e, sympy.Piecewise):
        from sympy_extras.functions.elementary.piecewise import ExprCondPair, Piecewise
        return Piecewise(*[ExprCondPair(_expr(pair.args[0]), _boolean(pair.args[1])) for pair in e.args])
    if isinstance(e, (sympy.Max, sympy.Min)) and all(isinstance(a, SymPyExpr) for a in e.args):
        from sympy_extras.functions.elementary.miscellaneous import Max, Min
        lattice = Max if isinstance(e, sympy.Max) else Min
        return lattice(*[_expr(a) for a in e.args])
    if isinstance(e, SymPyFunctionClass) and all(isinstance(a, SymPyExpr) for a in e.args):
        cls = type(e)
        native = _sympy_to_native().get(cls)
        if native is not None:
            return native(*[_expr(a) for a in e.args])
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
    >>> from sympy_extras.interop import through_sympy
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
            # a typed function class rewrites as its SymPy class (Cos as cos)
            if issubclass(a, DefinedFunction):
                if not _NATIVE_TO_SYMPY:
                    _fill_function_tables()
                if a in _NATIVE_TO_SYMPY:
                    targets.append(_NATIVE_TO_SYMPY[a])
                    continue
            targets.append(a)
        else:
            raise TypeError("cannot rewrite in terms of %r" % (a,))
    return targets


_SYMPY_HAS_RULE: dict[tuple[type, str], bool] = {}


def _sympy_has_rule(cls: type, name: str) -> bool:
    """Whether the SymPy class has a rewrite rule for the target name (its
    ``_eval_rewrite_as_<name>`` or an ``_eval_rewrite`` of its own)."""
    key = (cls, name)
    if key not in _SYMPY_HAS_RULE:
        method = '_eval_rewrite_as_' + name
        _SYMPY_HAS_RULE[key] = any(method in k.__dict__ for k in cls.__mro__) or any(
            '_eval_rewrite' in k.__dict__ for k in cls.__mro__ if k is not SymPyBasic)
    return _SYMPY_HAS_RULE[key]


_SYMPY_HAS_METHOD: dict[tuple[type, str], bool] = {}


def _sympy_has_method(cls: type, name: str) -> bool:
    """Whether a class of the SymPy class's MRO (not Basic) defines ``name``."""
    key = (cls, name)
    if key not in _SYMPY_HAS_METHOD:
        _SYMPY_HAS_METHOD[key] = any(name in k.__dict__ for k in cls.__mro__ if k is not SymPyBasic)
    return _SYMPY_HAS_METHOD[key]


def expand_func_node_through_sympy(node: Basic) -> Optional[Basic]:
    """The node expanded by SymPy's ``_eval_expand_func`` (not deep);
    ``None`` when its SymPy class has none."""
    original = to_sympy(node)
    if not _sympy_has_method(type(original), '_eval_expand_func'):
        return None
    return from_sympy(sympy.expand_func(original, deep=False))


def _sympy_defines_before(cls: type, name: str, stop: type) -> bool:
    """Whether a class of the SymPy class's MRO before ``stop`` defines
    ``name`` (an override of the method of ``stop``)."""
    for k in cls.__mro__:
        if k is stop:
            return False
        if name in k.__dict__:
            return True
    return False


def _sympy_series_args(x: Symbol, logx: Optional[Expr], cdir: Expr
                       ) -> tuple[SymPyBasic, Optional[SymPyBasic], SymPyBasic]:
    return to_sympy(x), None if logx is None else to_sympy(logx), to_sympy(cdir)


def nseries_node_through_sympy(node: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
    """SymPy's ``_eval_nseries`` of the node (the series of the classes
    whose expansion is not ported yet); SymPy's ``PoleError`` is raised
    as the one of the core."""
    original = sympy_expr(to_sympy(node))
    sx, slogx, scdir = _sympy_series_args(x, logx, cdir)
    try:
        return expr_from_sympy(original._eval_nseries(sx, n, slogx, scdir))
    except sympy.PoleError as error:
        raise PoleError(str(error)) from error


def leading_term_node_through_sympy(node: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    """SymPy's ``_eval_as_leading_term`` of the node when its SymPy class
    has its own (not the default of ``Expr`` or of ``Function``), ``None``
    otherwise."""
    original = sympy_expr(to_sympy(node))
    stop: type = SymPyFunctionClass if isinstance(original, SymPyFunctionClass) else SymPyExpr
    if not _sympy_defines_before(type(original), '_eval_as_leading_term', stop):
        return None
    sx, slogx, scdir = _sympy_series_args(x, logx, cdir)
    try:
        return expr_from_sympy(original._eval_as_leading_term(sx, logx=slogx, cdir=scdir))
    except sympy.PoleError as error:
        raise PoleError(str(error)) from error


def sympy_leading_term(node: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    """SymPy's ``_eval_as_leading_term`` of the node, whatever class
    defines it."""
    original = sympy_expr(to_sympy(node))
    sx, slogx, scdir = _sympy_series_args(x, logx, cdir)
    try:
        return expr_from_sympy(original._eval_as_leading_term(sx, logx=slogx, cdir=scdir))
    except sympy.PoleError as error:
        raise PoleError(str(error)) from error


def rewrite_named_node_through_sympy(node: Basic, args: tuple[Basic, ...], name: str,
                                     limitvar: Optional[Symbol]) -> Optional[Basic]:
    """The node, rebuilt with ``args``, rewritten by SymPy's
    ``_eval_rewrite_as_<name>`` (``tractable``, ``intractable``; not deep);
    ``None`` when its SymPy class has no such rule."""
    sympy_args = [to_sympy(a) for a in args]
    if isinstance(node, DefinedFunction):
        if not _NATIVE_TO_SYMPY:
            _fill_function_tables()
        sympy_class = _NATIVE_TO_SYMPY.get(type(node))
        if sympy_class is None:
            return None
        if not _sympy_has_method(sympy_class, '_eval_rewrite_as_' + name):
            return None
        rebuilt = sympy_class(*sympy_args, evaluate=False)
    else:
        original = to_sympy(node)
        if not _sympy_has_method(type(original), '_eval_rewrite_as_' + name):
            return None
        rebuilt = original.func(*sympy_args) if args else original
    hints = {} if limitvar is None else {'limitvar': to_sympy(limitvar)}
    return from_sympy(rebuilt.rewrite(name, deep=False, **hints))


def meromorphic_node_through_sympy(node: Basic, x: Symbol, a: Expr, own_only: bool) -> tuple[bool, Optional[bool]]:
    """SymPy's ``_eval_is_meromorphic`` of the node: ``(True, answer)``
    when its SymPy class has a rule of its own (with ``own_only``; any
    rule otherwise), ``(False, None)`` when not."""
    if isinstance(node, DefinedFunction):
        if not _NATIVE_TO_SYMPY:
            _fill_function_tables()
        sympy_class = _NATIVE_TO_SYMPY.get(type(node))
        if sympy_class is None:
            return False, None
        if own_only and not _sympy_defines_before(sympy_class, '_eval_is_meromorphic', SymPyFunctionClass):
            return False, None
        original = sympy_class(*[to_sympy(t) for t in node.args], evaluate=False)
    else:
        original = to_sympy(node)
        stop: type = SymPyFunctionClass if isinstance(original, SymPyFunctionClass) else SymPyExpr
        if own_only and not _sympy_defines_before(type(original), '_eval_is_meromorphic', stop):
            return False, None
    if not isinstance(original, SymPyExpr):
        return False, None
    answer = original._eval_is_meromorphic(to_sympy(x), to_sympy(a))
    return True, answer if isinstance(answer, bool) else None


def solveset_first_through_sympy(equation: Expr, d: Symbol) -> Optional[Expr]:
    """The first solution of SymPy's ``solveset(equation, d)`` (a
    complement taken as the difference of its sets, as ``Order`` does);
    ``None`` when there is none."""
    solutions = sympy.solveset(to_sympy(equation), to_sympy(d))
    if isinstance(solutions, sympy.Complement):
        kept, removed = solutions.args
        if not (isinstance(kept, sympy.FiniteSet) and isinstance(removed, sympy.FiniteSet)):
            return None
        solutions = sympy.FiniteSet(*[s for s in kept.args if s not in removed.args])
    if isinstance(solutions, sympy.FiniteSet):
        return expr_from_sympy(solutions.args[0]) if solutions.args else None
    # Order takes the first element of the iteration (an ImageSet over the
    # integers starts at n = 0)
    if isinstance(solutions, sympy.ImageSet):
        try:
            first = next(iter(solutions))
        except (TypeError, StopIteration):
            return None
        return expr_from_sympy(first) if isinstance(first, SymPyBasic) else None
    return None


def rewrite_node_through_sympy(node: Basic, args: tuple[Basic, ...], target: type,
                               piecewise: bool = True) -> Optional[Basic]:
    """The node, rebuilt with ``args``, rewritten by its SymPy rule for
    ``target`` (not deep); ``None`` when SymPy has no rule or the target
    has no SymPy class."""
    if not issubclass(target, DefinedFunction):
        return None
    if not _NATIVE_TO_SYMPY:
        _fill_function_tables()
    sympy_target = _NATIVE_TO_SYMPY.get(target)
    if sympy_target is None:
        return None
    sympy_args = [to_sympy(a) for a in args]
    if isinstance(node, DefinedFunction) and type(node) in _NATIVE_TO_SYMPY:
        # the node's own class, unevaluated (SymPy calls the rule of the
        # node itself with the rewritten arguments)
        sympy_class = _NATIVE_TO_SYMPY[type(node)]
        if not _sympy_has_rule(sympy_class, sympy_target.__name__):
            return None
        rebuilt = sympy_class(*sympy_args, evaluate=False)
    else:
        original = to_sympy(node)
        if not _sympy_has_rule(type(original), sympy_target.__name__):
            return None
        rebuilt = original.func(*sympy_args) if args else original
    return from_sympy(rebuilt.rewrite(sympy_target, deep=False, piecewise=piecewise))


def _canonical_order(args: list[Expr], commutative: bool) -> list[Expr]:
    """The arguments of a sum or a product of SymPy in the order the
    evaluating functions of this core give (SymPy's own order): the
    numbers first, then the commutative arguments in the ``compare``
    order, then the noncommutative ones as given."""
    from sympy_extras.core.numbers import Number
    from sympy_extras.core.sorting import compare_key
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
    from sympy_extras.core.ask import PREDICATES_OF_Q
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


from sympy_extras.core.cache import register_clearable  # noqa: E402 (the registration needs the class above)

register_clearable(_ConversionCache())


# the functions of the typed core and their SymPy counterparts (filled on
# first use: sympy_extras.functions imports the core)
_NATIVE_TO_SYMPY: dict[type[DefinedFunction], type[SymPyFunctionClass]] = {}
_SYMPY_TO_NATIVE: dict[type[SymPyFunctionClass], type[DefinedFunction]] = {}


def _fill_function_tables() -> None:
    if _NATIVE_TO_SYMPY:
        return
    from sympy_extras.functions.elementary.exponential import Exp, ExpPolar, LambertW, Log
    from sympy_extras.functions.elementary import complexes as c, hyperbolic as h, integers as n, trigonometric as t
    from sympy_extras.functions.combinatorial import factorials as f
    from sympy_extras.functions.elementary import miscellaneous as ms
    from sympy_extras.functions.special import delta_functions as d, error_functions as e, gamma_functions as g
    from sympy_extras.functions.special import beta_functions as b, bessel as bs, elliptic_integrals as el
    from sympy_extras.functions.special import polynomials as op, zeta_functions as z
    from sympy_extras.functions.special import spherical_harmonics as shm, tensor_functions as tens
    from sympy_extras.functions.special import mathieu_functions as mat, singularity_functions as sing
    from sympy_extras.functions.special import hyper as hy
    from sympy_extras.functions.combinatorial import numbers as cn
    pairs: list[tuple[type[DefinedFunction], type[SymPyFunctionClass]]] = [
        (Exp, sympy.exp), (Log, sympy.log), (LambertW, sympy.LambertW), (ExpPolar, sympy.exp_polar),
        (c.PolarLift, sympy.polar_lift), (c.PeriodicArgument, sympy.periodic_argument),
        (c.PrincipalBranch, sympy.principal_branch), (c.Adjoint, sympy.adjoint), (c.Transpose, sympy.transpose), (t.Sin, sympy.sin), (t.Cos, sympy.cos), (t.Tan, sympy.tan),
        (t.Cot, sympy.cot), (t.Sec, sympy.sec), (t.Csc, sympy.csc), (t.Asin, sympy.asin), (t.Acos, sympy.acos),
        (t.Atan, sympy.atan), (t.Acot, sympy.acot), (t.Asec, sympy.asec), (t.Acsc, sympy.acsc),
        (h.Sinh, sympy.sinh), (h.Cosh, sympy.cosh), (h.Tanh, sympy.tanh), (h.Coth, sympy.coth), (h.Sech, sympy.sech),
        (h.Csch, sympy.csch), (h.Asinh, sympy.asinh), (h.Acosh, sympy.acosh), (h.Atanh, sympy.atanh),
        (h.Acoth, sympy.acoth), (h.Asech, sympy.asech), (h.Acsch, sympy.acsch), (c.Re, sympy.re), (c.Im, sympy.im),
        (c.Abs, sympy.Abs), (c.Sign, sympy.sign), (c.Arg, sympy.arg), (c.Conjugate, sympy.conjugate),
        (c.Atan2, sympy.atan2), (n.Floor, sympy.floor), (n.Ceiling, sympy.ceiling), (n.Frac, sympy.frac),
        (f.Factorial, sympy.factorial), (f.Binomial, sympy.binomial), (f.RisingFactorial, sympy.RisingFactorial),
        (f.FallingFactorial, sympy.FallingFactorial), (f.Subfactorial, sympy.subfactorial), (f.Factorial2, sympy.factorial2),
        (t.Sinc, sympy.sinc), (ms.Rem, sympy.Rem), (g.Gamma, sympy.gamma), (g.LogGamma, sympy.loggamma),
        (g.PolyGamma, sympy.polygamma), (g.DiGamma, sympy.digamma), (g.TriGamma, sympy.trigamma),
        (d.DiracDelta, sympy.DiracDelta), (d.Heaviside, sympy.Heaviside), (z.Zeta, sympy.zeta),
        (z.DirichletEta, sympy.dirichlet_eta), (z.PolyLog, sympy.polylog), (e.Erf, sympy.erf), (e.Erfc, sympy.erfc),
        (e.Erfi, sympy.erfi), (e.Erfinv, sympy.erfinv), (e.Erfcinv, sympy.erfcinv), (b.Beta, sympy.beta),
        (cn.Catalan, sympy.catalan), (cn.Fibonacci, sympy.fibonacci), (cn.Lucas, sympy.lucas),
        (cn.Harmonic, sympy.harmonic), (cn.Bernoulli, sympy.bernoulli),
        (cn.Euler, sympy.euler), (cn.Bell, sympy.bell), (cn.Genocchi, sympy.genocchi), (cn.Partition, sympy.partition),
        (cn.Totient, sympy.totient), (cn.ReducedTotient, sympy.reduced_totient), (cn.DivisorSigma, sympy.divisor_sigma),
        (cn.UDivisorSigma, sympy.functions.udivisor_sigma), (cn.PrimeNu, sympy.primenu), (cn.PrimeOmega, sympy.primeomega),
        (cn.Mobius, sympy.mobius), (cn.PrimePi, sympy.primepi), (cn.LegendreSymbol, sympy.legendre_symbol),
        (cn.JacobiSymbol, sympy.jacobi_symbol), (cn.KroneckerSymbol, sympy.kronecker_symbol), (g.LowerGamma, sympy.lowergamma), (g.UpperGamma, sympy.uppergamma),
        (e.Expint, sympy.expint), (e.EiNode, sympy.Ei), (e.Erf2, sympy.erf2), (e.Erf2inv, sympy.erf2inv),
        (b.BetaInc, sympy.betainc), (b.BetaIncRegularized, sympy.betainc_regularized), (z.LerchPhi, sympy.lerchphi),
        (z.RiemannXi, sympy.riemann_xi), (z.Stieltjes, sympy.stieltjes), (g.MultiGamma, sympy.multigamma), (e.SiNode, sympy.Si), (e.CiNode, sympy.Ci), (e.ShiNode, sympy.Shi),
        (e.ChiNode, sympy.Chi), (e.LiNode, sympy.li), (e.OffsetLiNode, sympy.Li), (bs.BesselJ, sympy.besselj),
        (bs.BesselY, sympy.bessely), (bs.BesselI, sympy.besseli), (bs.BesselK, sympy.besselk),
        (bs.Hankel1, sympy.hankel1), (bs.Hankel2, sympy.hankel2), (bs.SphericalJ, sympy.jn), (bs.SphericalY, sympy.yn),
        (bs.AiryAi, sympy.airyai), (bs.AiryBi, sympy.airybi), (bs.AiryAiPrime, sympy.airyaiprime),
        (bs.AiryBiPrime, sympy.airybiprime), (bs.SphericalHankel1, sympy.hn1), (bs.SphericalHankel2, sympy.hn2),
        (bs.MarcumQ, sympy.marcumq), (hy.AppellF1, sympy.appellf1), (cn.Tribonacci, sympy.tribonacci), (cn.Motzkin, sympy.motzkin),
        (cn.Andre, sympy.andre), (op.Legendre, sympy.legendre), (op.ChebyshevT, sympy.chebyshevt),
        (op.ChebyshevU, sympy.chebyshevu), (op.Hermite, sympy.hermite), (op.Laguerre, sympy.laguerre),
        (op.AssocLaguerre, sympy.assoc_laguerre), (op.Jacobi, sympy.jacobi), (op.Gegenbauer, sympy.gegenbauer),
        (op.AssocLegendre, sympy.assoc_legendre), (op.HermiteProb, sympy.hermite_prob), (shm.YnmNode, sympy.Ynm),
        (shm.ZnmNode, sympy.Znm), (tens.KroneckerDeltaNode, sympy.KroneckerDelta), (tens.LeviCivitaNode, sympy.LeviCivita),
        (mat.MathieuS, sympy.mathieus), (mat.MathieuC, sympy.mathieuc), (mat.MathieuSPrime, sympy.mathieusprime),
        (mat.MathieuCPrime, sympy.mathieucprime), (sing.SingularityFunctionNode, sympy.SingularityFunction),
        (e.FresnelS, sympy.fresnels), (e.FresnelC, sympy.fresnelc), (el.EllipticK, sympy.elliptic_k),
        (el.EllipticF, sympy.elliptic_f), (el.EllipticE, sympy.elliptic_e), (el.EllipticPi, sympy.elliptic_pi)]
    import sympy.functions.special.hyper as sympy_hyper
    from sympy_extras.functions.special import hyperrep
    for name in hyperrep.__all__:
        if name.startswith('HyperRep_'):
            native_rep, sympy_rep = getattr(hyperrep, name), getattr(sympy_hyper, name)
            if isinstance(native_rep, type) and issubclass(native_rep, DefinedFunction) \
                    and isinstance(sympy_rep, type) and issubclass(sympy_rep, sympy.Function):
                pairs.append((native_rep, sympy_rep))
    from sympy_extras.concrete import qhyper, qhyper_sympy
    pairs.append((qhyper.QPochhammer, qhyper_sympy.QPochhammer))
    for native, counterpart in pairs:
        _NATIVE_TO_SYMPY[native] = counterpart
        _SYMPY_TO_NATIVE[counterpart] = native


def _native_to_sympy() -> dict[type[DefinedFunction], type[SymPyFunctionClass]]:
    _fill_function_tables()
    return _NATIVE_TO_SYMPY


def _sympy_to_native() -> dict[type[SymPyFunctionClass], type[DefinedFunction]]:
    _fill_function_tables()
    return _SYMPY_TO_NATIVE


def _set_to_sympy(e: Basic, args: list[SymPyBasic], evaluate: bool) -> SymPyBasic:
    """A set of :mod:`sympy_extras.sets` as SymPy's."""
    from sympy_extras.sets import sets as native_sets
    if isinstance(e, native_sets.RealsClass):
        return sympy.S.Reals
    if isinstance(e, native_sets.EmptySetClass):
        return sympy.S.EmptySet
    if isinstance(e, native_sets.UniversalSetClass):
        return sympy.S.UniversalSet
    if isinstance(e, native_sets.Interval):
        return sympy.Interval(*args)
    if isinstance(e, native_sets.FiniteSet):
        return sympy.FiniteSet(*args, evaluate=evaluate)
    if isinstance(e, native_sets.Union):
        return sympy.Union(*args, evaluate=evaluate)
    if isinstance(e, native_sets.Intersection):
        return sympy.Intersection(*args, evaluate=evaluate)
    if isinstance(e, native_sets.Complement):
        return sympy.Complement(args[0], args[1], evaluate=evaluate)
    if isinstance(e, native_sets.Contains):
        return sympy.Contains(args[0], args[1], evaluate=False)
    raise TypeError('no SymPy counterpart of %s' % (type(e).__name__,))


def _set_from_sympy(e: SymPyBasic) -> Basic:
    """A SymPy set as a set of :mod:`sympy_extras.sets` (unevaluated)."""
    from sympy_extras.sets import sets as native_sets

    if e is sympy.S.Reals:
        return native_sets.Reals
    if e is sympy.S.EmptySet:
        return native_sets.EmptySet
    if e is sympy.S.UniversalSet:
        return native_sets.UniversalSet
    if isinstance(e, sympy.Interval):
        return native_sets.Interval(_expr(e.start), _expr(e.end), bool(e.left_open), bool(e.right_open))
    if isinstance(e, sympy.FiniteSet):
        return native_sets.FiniteSet(*[_expr(a) for a in e.args])
    parts = [from_sympy(a) for a in e.args]
    sets = [a for a in parts if isinstance(a, native_sets.Set)]
    if isinstance(e, sympy.Union) and len(sets) == len(parts):
        return native_sets.Union(*sets)
    if isinstance(e, sympy.Intersection) and len(sets) == len(parts):
        return native_sets.Intersection(*sets)
    if isinstance(e, sympy.Complement) and len(sets) == 2:
        return native_sets.Complement(sets[0], sets[1])
    if isinstance(e, sympy.Contains) and isinstance(parts[0], Expr) and isinstance(parts[1], native_sets.Set):
        return native_sets.Contains(parts[0], parts[1])
    # a set not ported, or with a part not ported
    return SymPyNode(type(e), *parts)


# ---------------------------------------------------------------------
# the functions of SymPy wrapped in the core, and SymPy's facts (moved
# here from sympy_extras.core.function and sympy_extras.core.assumptions)

class SymPyFunction(FunctionClass):
    """A function of SymPy usable in this core; calling it builds an
    unevaluated :class:`AppliedSymPyFunction` node.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.interop import SymPyFunction
    >>> sin = SymPyFunction(sympy.sin)
    >>> x = Symbol('x')
    >>> sin(x), sin(0), sin(0).doit(), sin(x).diff(x)
    (sin(x), sin(0), 0, cos(x))
    """

    __slots__ = ('sympy_class',)

    def __init__(self, sympy_class: type[SymPyFunctionClass]) -> None:
        self.sympy_class: Final[type[SymPyFunctionClass]] = sympy_class

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
        self._sympy_class: Final[type[SymPyFunctionClass]] = sympy_class
        Application.__init__(self, *args)

    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
                return nseries_node_through_sympy(self, x, n, logx, cdir)

    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
                return sympy_leading_term(self, x, logx, cdir)

    def _eval_is_meromorphic(self, x: Symbol, a: Expr) -> Optional[bool]:
                return meromorphic_node_through_sympy(self, x, a, False)[1]

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

    def _sort_class_key(self) -> ClassKey:
        return (4, _FUNCTION_RANKS.get(self.name, 10000), self.name)

    @property
    def kind(self) -> Kind:
        from sympy.core.kind import NumberKind as SymPyNumberKind

        from sympy_extras.core.kind import NumberKind
        return NumberKind if to_sympy(self).kind is SymPyNumberKind else UndefinedKind

    def as_base_exp(self) -> tuple[Expr, Expr]:
        # exp(a) is E**a, as in SymPy (where E**a builds exp(a))
        if self._sympy_class is sympy.exp:
            from sympy_extras.core.numbers import E
            return (E, self._args[0])
        return (self, Integer(1))

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        if self._sympy_class is sympy.exp:
            arg = self._args[0]
            if arg.is_negative is True or arg.could_extract_minus_sign():  # exp(-x) is 1/exp(x), as SymPy decides
                return (Integer(1), apply_sympy(sympy.exp, mul(Integer(-1), arg)))
        return (self, Integer(1))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return apply_sympy(self._sympy_class, *[as_expr(a) for a in args])

    def _eval_facts(self) -> dict[str, bool]:
                return sympy_facts(self)

    def _numeric_node(self, prec: int) -> Optional[mpmath.mpc]:
        # SymPy's numeric value of its function
        return _sympy_numeric(self, int(mpmath.mp.dps) + 5)

    def _eval_derivative_node(self, s: Symbol) -> Optional[Expr]:
        # the derivative of a function of SymPy is SymPy's (Abs has its own rule)
        return expr_through_sympy(self, lambda node: sympy.diff(node, to_sympy(s)))

    def _count_ops_label(self) -> Optional[str]:
        return self.name.upper()

    def _srepr_name(self) -> Optional[str]:
        return self.name



def apply_sympy(sympy_class: type[SymPyFunctionClass], *args: ExprLike) -> Expr:
    """The function of SymPy applied and evaluated by SymPy.

    Examples
    ========

    >>> import sympy
    >>> from sympy_extras.core import Symbol, E
    >>> from sympy_extras.interop import apply_sympy
    >>> x = Symbol('x')
    >>> apply_sympy(sympy.log, E), apply_sympy(sympy.log, x), apply_sympy(sympy.sin, sympy.pi)
    (1, log(x), 0)
    """
    from sympy_extras.core.parameters import global_parameters
    if not global_parameters.evaluate:
        return AppliedSymPyFunction(sympy_class, *args)
    value = sympy_class(*[to_sympy(to_expr(a)) for a in args])
    if not isinstance(value, SymPyBasic):
        raise TypeError("SymPy returned %r" % (value,))
    return expr_from_sympy(value)



#: SymPy's ``is_*`` properties by predicate name (read without ``getattr``)
_SYMPY_QUERIES: dict[str, Callable[[SymPyBasic], Truth]] = {
    'commutative': lambda e: e.is_commutative, 'complex': lambda e: e.is_complex, 'real': lambda e: e.is_real,
    'extended_real': lambda e: e.is_extended_real, 'imaginary': lambda e: e.is_imaginary,
    'integer': lambda e: e.is_integer, 'noninteger': lambda e: e.is_noninteger, 'rational': lambda e: e.is_rational,
    'irrational': lambda e: e.is_irrational, 'algebraic': lambda e: e.is_algebraic,
    'transcendental': lambda e: e.is_transcendental, 'positive': lambda e: e.is_positive,
    'negative': lambda e: e.is_negative, 'nonnegative': lambda e: e.is_nonnegative,
    'nonpositive': lambda e: e.is_nonpositive, 'zero': lambda e: e.is_zero, 'nonzero': lambda e: e.is_nonzero,
    'extended_positive': lambda e: e.is_extended_positive, 'extended_negative': lambda e: e.is_extended_negative,
    'extended_nonnegative': lambda e: e.is_extended_nonnegative,
    'extended_nonpositive': lambda e: e.is_extended_nonpositive, 'extended_nonzero': lambda e: e.is_extended_nonzero,
    'even': lambda e: e.is_even, 'odd': lambda e: e.is_odd, 'prime': lambda e: e.is_prime,
    'composite': lambda e: e.is_composite, 'finite': lambda e: e.is_finite, 'infinite': lambda e: e.is_infinite,
    'hermitian': lambda e: e.is_hermitian, 'antihermitian': lambda e: e.is_antihermitian, 'polar': lambda e: e.is_polar,
}


def sympy_facts(e: Expr) -> Facts:
    """The facts SymPy knows about the SymPy form of ``e`` (its ``is_*``
    properties)."""
    facts: Facts = {}
    sympy_expr = to_sympy(e)
    for name, query in _SYMPY_QUERIES.items():
        try:
            answer = query(sympy_expr)
        except (TypeError, ValueError, AttributeError):  # SymPy refuses some questions
            continue
        if isinstance(answer, bool):
            facts[name] = answer
    return facts


def _with_fixed_random(f: Callable[[], object]) -> object:
    """``f()`` with SymPy's random generators seeded with a fixed value
    and restored after: SymPy's ``is_constant`` and ``equals`` evaluate at
    random points, so that their answer (and their running time: some
    points make the evaluation of number powers explode) depended on the
    calls made before, in any order the hash seed gave."""
    from sympy.core import random as sympy_random
    saved = (sympy_random.rng.getstate(), sympy_random._assumptions_rng.getstate())
    sympy_random.seed(0)
    try:
        return f()
    finally:
        sympy_random.rng.setstate(saved[0])
        sympy_random._assumptions_rng.setstate(saved[1])


def _sympy_numeric(e: Expr, dps: int) -> Optional[mpmath.mpc]:
    """SymPy's value of ``e`` at ``dps`` digits as an mpmath number."""
    s = to_sympy(e)
    if not isinstance(s, SymPyExpr):
        return None
    value = s.evalf(dps)
    if not isinstance(value, SymPyExpr):
        return None
    re, im = value.as_real_imag()
    if not (isinstance(re, sympy.Number) and isinstance(im, sympy.Number)):
        return None
    return mpmath.mpc(mpmath.mpf(str(re)), mpmath.mpf(str(im)))


# ---------------------------------------------------------------------
# the bridge of the core (sympy_extras.core.bridge): SymPy for the
# operations not yet ported, registered when this module is imported

class SymPyBridge(Bridge):
    """SymPy lent to the core."""

    def available(self) -> bool:
        return True

    def is_foreign(self, a: object) -> bool:
        return isinstance(a, SymPyBasic)

    def from_foreign(self, a: object) -> Basic:
        if not isinstance(a, SymPyBasic):
            raise TypeError("not an object of SymPy: %r" % (a,))
        return from_sympy(a)

    def parse(self, text: str, evaluate: bool) -> Basic:
        from sympy.parsing.sympy_parser import parse_expr
        parsed = parse_expr(text, evaluate=evaluate)
        if not isinstance(parsed, SymPyBasic):
            raise ValueError("cannot parse %r" % (text,))
        return from_sympy(parsed)

    def numeric(self, e: Expr, dps: int) -> Optional[mpmath.mpc]:
        return _sympy_numeric(e, dps)

    def ask(self, proposition: Boolean, assumptions: Boolean) -> Truth:
        result = sympy.ask(to_sympy(proposition), to_sympy(assumptions))
        return result if isinstance(result, bool) else None

    def equals_zero(self, e: Expr) -> Truth:
        s = to_sympy(e)
        if not isinstance(s, SymPyExpr):
            return None
        answer = _with_fixed_random(lambda: s.equals(0))
        return answer if isinstance(answer, bool) else None

    def is_constant(self, e: Expr, wrt: Sequence[Symbol]) -> Truth:
        s = to_sympy(e)
        if not isinstance(s, SymPyExpr):
            return None
        variables = [to_sympy(w) for w in wrt]
        answer = _with_fixed_random(lambda: s.is_constant(*variables))
        return answer if isinstance(answer, bool) else None

    def is_hypergeometric(self, e: Expr, k: Symbol) -> Truth:
        s = to_sympy(e)
        if not isinstance(s, SymPyExpr):
            return None
        answer = s.is_hypergeometric(to_sympy(k))
        return answer if isinstance(answer, bool) else None

    def minimal_polynomial(self, e: Expr, x: Symbol) -> Expr:
        return expr_from_sympy(sympy.minimal_polynomial(to_sympy(e), to_sympy(x)))

    def nseries(self, node: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        return nseries_node_through_sympy(node, x, n, logx, cdir)

    def meromorphic(self, node: Expr, x: Symbol, a: Expr, own_only: bool) -> tuple[bool, Truth]:
        return meromorphic_node_through_sympy(node, x, a, own_only)

    def rewrite(self, e: Basic, targets: Sequence[RewriteTarget], piecewise: bool) -> Basic:
        sympy_targets = rewrite_targets(targets)
        return through_sympy(e, lambda s: s.rewrite(*sympy_targets, piecewise=piecewise))

    def rewrite_node(self, e: Basic, args: tuple[Basic, ...], target: type, piecewise: bool) -> Optional[Basic]:
        return rewrite_node_through_sympy(e, args, target, piecewise)

    def rewrite_named_node(self, e: Basic, args: tuple[Basic, ...], name: str, limitvar: Optional[Symbol]
                           ) -> Optional[Basic]:
        return rewrite_named_node_through_sympy(e, args, name, limitvar)

    def expand_func_node(self, e: Basic) -> Optional[Basic]:
        return expand_func_node_through_sympy(e)

    def solve_first(self, equation: Expr, x: Symbol) -> Optional[Expr]:
        return solveset_first_through_sympy(equation, x)

    def simplify(self, e: Basic, ratio: float = 1.7, rational: Optional[bool] = False, inverse: bool = False,
                 doit: bool = True) -> Basic:
        return through_sympy(e, lambda s: sympy.simplify(s, ratio=ratio, rational=rational, inverse=inverse, doit=doit))

    def factor_foreign_integrals(self, e: Expr) -> Expr:
        def integrals(s: SymPyExpr) -> SymPyBasic:
            if s.has(sympy.Integral):
                return s.xreplace({i: sympy.factor_terms(i) for i in s.atoms(sympy.Integral)})
            return s
        return expr_through_sympy(e, integrals)

    def integrate(self, e: Expr, specs: Sequence[Union[Symbol, tuple[Symbol, Expr, Expr]]]) -> Expr:
        converted: list[Union[SymPyBasic, tuple[SymPyBasic, SymPyBasic, SymPyBasic]]] = []
        for a in specs:
            converted.append((to_sympy(a[0]), to_sympy(a[1]), to_sympy(a[2])) if isinstance(a, tuple) else to_sympy(a))
        return expr_through_sympy(e, lambda s: sympy.integrate(s, *converted))

    def fps(self, e: Expr, x: Optional[Symbol], x0: Expr, dir: int) -> Expr:
        sx = None if x is None else to_sympy(x)
        return expr_through_sympy(e, lambda s: sympy.fps(s, sx, to_sympy(x0), dir))

    def fourier_series(self, e: Expr, limits: Optional[tuple[Symbol, Expr, Expr]]) -> Expr:
        bounds = None if limits is None else (to_sympy(limits[0]), to_sympy(limits[1]), to_sympy(limits[2]))
        return expr_through_sympy(e, lambda s: sympy.fourier_series(s, bounds))

    def invert(self, e: Expr, g: Expr, gens: Sequence[Symbol]) -> Expr:
        return expr_through_sympy(e, lambda s: sympy.invert(s, to_sympy(g), *[to_sympy(v) for v in gens]))


register_bridge(SymPyBridge())

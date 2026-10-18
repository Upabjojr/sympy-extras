"""Conversion between this core and SymPy's.

``to_sympy`` builds the SymPy expression of a tree (evaluated by SymPy's
constructors, or with ``evaluate=False`` to keep the structure);
``from_sympy`` builds the tree of a SymPy expression. Functions of SymPy
which have no class here (``sin``, ``exp``, ``log``, ...) are represented
by :class:`~sympy_extras.core.function.AppliedSymPyFunction`, so that the
round trip is exact; symbols keep the assumptions given to their
constructor; floats keep their binary value and precision.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Union

import mpmath
import sympy
from sympy.core.basic import Basic as SymPyBasic
from sympy.core.expr import Expr as SymPyExpr
from sympy.core.function import AppliedUndef as SymPyAppliedUndef
from sympy.core.function import Function as SymPyFunctionClass
from sympy.core.function import UndefinedFunction as SymPyUndefinedFunction
from sympy.core.relational import Relational as SymPyRelational

from .add import Add
from .basic import Basic
from .containers import Tuple
from .expr import Expr
from .function import AppliedSymPyFunction, AppliedUndef, Derivative, Function, Lambda
from .logic import BooleanAtom, false, true
from .mul import Mul
from .numbers import (ComplexInfinity, Exp1, Float, ImaginaryUnit, Infinity, Integer, NaN,
    NegativeInfinity, Pi, Rational, E, I, nan, number_from_fraction, oo, pi, zoo)
from .power import Pow
from .relational import Eq, Ge, Gt, Le, Lt, Ne, Relational
from .symbol import Dummy, Symbol

__all__ = ['to_sympy', 'from_sympy']

_RELATIONAL_TO_SYMPY: dict[type[Relational], type[SymPyRelational]] = {
    Eq: sympy.Eq, Ne: sympy.Ne, Lt: sympy.Lt, Le: sympy.Le, Gt: sympy.Gt, Ge: sympy.Ge}
_RELATIONAL_FROM_SYMPY: dict[type[SymPyRelational], type[Relational]] = {
    sympy.Eq: Eq, sympy.Ne: Ne, sympy.Lt: Lt, sympy.Le: Le, sympy.Gt: Gt, sympy.Ge: Ge}


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
    if isinstance(e, Dummy):
        return sympy.Dummy(e.name, **e.given_assumptions)
    if isinstance(e, Symbol):
        return sympy.Symbol(e.name, **e.given_assumptions)
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
    args = [to_sympy(a, evaluate) for a in e.args]
    if isinstance(e, Add):
        return sympy.Add(*[_sympy_expr(a) for a in args], evaluate=evaluate)
    if isinstance(e, Mul):
        return sympy.Mul(*[_sympy_expr(a) for a in args], evaluate=evaluate)
    if isinstance(e, Pow):
        return sympy.Pow(*[_sympy_expr(a) for a in args], evaluate=evaluate)
    if isinstance(e, AppliedUndef):
        return sympy.Function(e.name)(*args)
    if isinstance(e, AppliedSymPyFunction):
        return e.sympy_class(*args, evaluate=evaluate)
    if isinstance(e, Derivative):
        counts = [(to_sympy(s), n) for s, n in e.variable_count]
        return sympy.Derivative(args[0], *counts)
    if isinstance(e, Lambda):
        return sympy.Lambda(tuple(to_sympy(v) for v in e.variables), to_sympy(e.expr, evaluate))
    if isinstance(e, Tuple):
        return sympy.Tuple(*args)
    if isinstance(e, Relational):
        return _RELATIONAL_TO_SYMPY[type(e)](*args, evaluate=evaluate)
    raise TypeError("cannot convert %s to SymPy" % (e,))


def _sympy_expr(e: SymPyBasic) -> SymPyExpr:
    if not isinstance(e, SymPyExpr):
        raise TypeError("a SymPy expression is expected, got %s" % (e,))
    return e


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
    if isinstance(e, sympy.Dummy):
        return Dummy(e.name, **_symbol_assumptions(e))
    if isinstance(e, sympy.Symbol):
        return Symbol(e.name, **_symbol_assumptions(e))
    if isinstance(e, sympy.Integer):
        return Integer(int(e.p))
    if isinstance(e, sympy.Rational):
        return Rational(int(e.p), int(e.q))
    if isinstance(e, sympy.Float):
        sign, man, exp, _ = e._mpf_
        value = number_from_fraction(_dyadic(int(sign), int(man), int(exp)))
        return Float(value, precision=int(e._prec))
    if e is sympy.pi:
        return pi
    if e is sympy.E:
        return E
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
    if isinstance(e, SymPyAppliedUndef):
        f = e.func
        if not isinstance(f, SymPyUndefinedFunction):
            raise TypeError("cannot convert %s" % (e,))
        return Function(f.__name__)(*[_expr(a) for a in e.args])
    if isinstance(e, sympy.Derivative):
        counts = [(_symbol(s), int(n)) for s, n in e.variable_count]
        return Derivative(_expr(e.expr), *counts)
    if isinstance(e, sympy.Lambda):
        return Lambda(tuple(_symbol(v) for v in e.variables), _expr(e.expr))
    if isinstance(e, sympy.Tuple):
        return Tuple(*[from_sympy(a) for a in e.args])
    if isinstance(e, SymPyRelational):
        for sympy_class, cls in _RELATIONAL_FROM_SYMPY.items():
            if isinstance(e, sympy_class):
                return cls(_expr(e.lhs), _expr(e.rhs))
        raise TypeError("cannot convert the relation %s" % (e,))
    if isinstance(e, SymPyFunctionClass):
        cls = type(e)
        return AppliedSymPyFunction(cls, *[_expr(a) for a in e.args])
    raise TypeError("cannot convert %s (%s) from SymPy" % (e, type(e).__name__))


def _canonical_order(args: list[Expr], commutative: bool) -> list[Expr]:
    """The arguments of a sum or a product of SymPy in the order the
    evaluating functions of this core give: the numbers first, then the
    commutative arguments sorted, then the noncommutative ones as given."""
    from .numbers import Number
    from .sorting import default_sort_key
    numbers = [a for a in args if isinstance(a, Number)]
    others = [a for a in args if not isinstance(a, Number)]
    if commutative:
        return numbers + sorted(others, key=default_sort_key)
    sortable = [a for a in others if a.is_commutative is not False]
    fixed = [a for a in others if a.is_commutative is False]
    return numbers + sorted(sortable, key=default_sort_key) + fixed


def _dyadic(sign: int, man: int, exp: int) -> Fraction:
    value = Fraction(man) * Fraction(2) ** exp
    return -value if sign else value


def _expr(e: SymPyBasic) -> Expr:
    converted = from_sympy(e)
    if not isinstance(converted, Expr):
        raise TypeError("an expression is expected, got %s" % (converted,))
    return converted


def _symbol(e: SymPyBasic) -> Symbol:
    converted = from_sympy(e)
    if not isinstance(converted, Symbol):
        raise TypeError("a symbol is expected, got %s" % (converted,))
    return converted


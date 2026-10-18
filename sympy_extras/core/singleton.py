"""The registry ``S`` of the singletons.

``S.Zero``, ``S.One``, ``S.Half``, ``S.Pi``, ``S.true``, ... are the usual
names; ``S(2)`` sympifies as in SymPy. The numbers are ordinary instances
(``Integer(0) == S.Zero`` and the two hash alike), nothing relies on
identity.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Union, overload

from .basic import Basic
from .expr import Expr, as_expr
from .logic import BooleanFalse, BooleanTrue, false, true
from .numbers import (ComplexInfinity, Exp1, ImaginaryUnit, Infinity, Integer, NaN,
    NegativeInfinity, Pi, Rational, E, I, nan, oo, pi, zoo)
from .sympify import SympifyLike, sympify

__all__ = ['S', 'SingletonRegistry']


class SingletonRegistry:
    """The named constants, and ``S(x)`` for ``sympify(x)``.

    Examples
    ========

    >>> from sympy_extras.core import S
    >>> S.One + S.Half
    3/2
    >>> S(2)/3
    2/3
    """

    def __init__(self) -> None:
        self.Zero: Integer = Integer(0)
        self.One: Integer = Integer(1)
        self.NegativeOne: Integer = Integer(-1)
        self.Half: Rational = Rational(1, 2)
        self.Pi: Pi = pi
        self.Exp1: Exp1 = E
        self.ImaginaryUnit: ImaginaryUnit = I
        self.Infinity: Infinity = oo
        self.NegativeInfinity: NegativeInfinity = NegativeInfinity()
        self.ComplexInfinity: ComplexInfinity = zoo
        self.NaN: NaN = nan
        self.true: BooleanTrue = true
        self.false: BooleanFalse = false

    @overload
    def __call__(self, x: Union[Expr, int, float, Fraction]) -> Expr: ...

    @overload
    def __call__(self, x: SympifyLike) -> Basic: ...

    def __call__(self, x: SympifyLike) -> Basic:
        result = sympify(x)
        if isinstance(x, (Expr, int, float, Fraction)) and not isinstance(x, bool):
            return as_expr(result)
        return result


S = SingletonRegistry()

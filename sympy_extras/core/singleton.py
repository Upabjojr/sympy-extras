"""The registry ``S`` of the singletons.

``S.Zero``, ``S.One``, ``S.Half``, ``S.Pi``, ``S.true``, ... are the usual
names; ``S(2)`` sympifies as in SymPy. The numbers are ordinary instances
(``Integer(0) == S.Zero`` and the two hash alike), nothing relies on
identity.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Final, Union, overload

from .basic import Basic
from .expr import Expr, as_expr
from .logic import BooleanFalse, BooleanTrue, false, true
from .numbers import (Catalan, CatalanType, ComplexInfinity, EulerGamma, EulerGammaType, Exp1,
    GoldenRatio, GoldenRatioType, ImaginaryUnit, Infinity, Integer, NaN, NegativeInfinity, Pi,
    Rational, TribonacciConstant, TribonacciConstantType, E, I, nan, oo, pi, zoo)
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
        self.Zero: Final[Integer] = Integer(0)
        self.One: Final[Integer] = Integer(1)
        self.NegativeOne: Final[Integer] = Integer(-1)
        self.Half: Final[Rational] = Rational(1, 2)
        self.Pi: Final[Pi] = pi
        self.Exp1: Final[Exp1] = E
        self.EulerGamma: Final[EulerGammaType] = EulerGamma
        self.Catalan: Final[CatalanType] = Catalan
        self.GoldenRatio: Final[GoldenRatioType] = GoldenRatio
        self.TribonacciConstant: Final[TribonacciConstantType] = TribonacciConstant
        self.ImaginaryUnit: Final[ImaginaryUnit] = I
        self.Infinity: Final[Infinity] = oo
        self.NegativeInfinity: Final[NegativeInfinity] = NegativeInfinity()
        self.ComplexInfinity: Final[ComplexInfinity] = zoo
        self.NaN: Final[NaN] = nan
        self.true: Final[BooleanTrue] = true
        self.false: Final[BooleanFalse] = false

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

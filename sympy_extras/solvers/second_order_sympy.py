"""The SymPy face of the integrating factors and the linearisations of
second order equations (:mod:`sympy_extras.solvers.second_order`).

The functions take and return SymPy objects (the symmetries of
:mod:`sympy_extras.solvers.lie_sympy`), converted by
:mod:`sympy_extras.interop` around the native functions;
:class:`FirstIntegral` and :class:`Linearization` hold SymPy
expressions. The dummies of :func:`second_order_rhs` stay one dummy on
both sides of the conversion.

Examples
========

>>> from sympy import Function
>>> from sympy.abc import x
>>> from sympy_extras.solvers.second_order_sympy import dsolve_second_order
>>> y = Function('y')(x)
>>> dsolve_second_order(y.diff(x, 2) + 3*y*y.diff(x) + y**3, y)
[Eq(y(x), 2*(C2 + x)/(2*C1 + 2*C2*x + x**2))]
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol

from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import second_order as native
from .lie_sympy import Symmetry, _core_expr, _core_function, _core_symbol, _sympy_expr, _sympy_symbol

__all__ = ['second_order_rhs', 'integrating_factor_xy', 'integrating_factor_p', 'is_linearizable',
           'linearize', 'rectify_symmetries', 'commuting_pair', 'transformed_rhs', 'dsolve_second_order',
           'FirstIntegral', 'Linearization']


# ---------------------------------------------------------------------------
# conversions

def _core_basic(e: Basic) -> CoreBasic:
    return from_sympy(e)


def _sympy_basic(e: CoreBasic) -> Basic:
    converted = to_sympy(e)
    if not isinstance(converted, Basic):
        raise TypeError("%s cannot be converted" % (e,))
    return converted


def _symbols(x: Symbol, y: Symbol, p: Symbol) -> tuple[CoreSymbol, CoreSymbol, CoreSymbol]:
    return _core_symbol(x), _core_symbol(y), _core_symbol(p)


def _optional_list(found: Optional[list[CoreBasic]]) -> Optional[list[Basic]]:
    return None if found is None else [_sympy_basic(e) for e in found]


# ---------------------------------------------------------------------------
# the results

class FirstIntegral:
    """A first integral ``R(x, y, y') = C1`` of a second order equation
    and the integrating factor which produced it
    (:class:`sympy_extras.solvers.second_order.FirstIntegral` with SymPy
    attributes).

    Attributes
    ==========

    factor : Expr
        The integrating factor ``mu`` (in ``x``, ``y`` and ``p`` for
        ``y'``).
    integral : Expr
        ``R(x, y, p)``.
    """

    def __init__(self, factor: Expr, integral: Expr, x: Symbol, y: Symbol, p: Symbol) -> None:
        self.factor: Final[Expr] = factor
        self.integral: Final[Expr] = integral
        self.x: Final[Symbol] = x
        self.y: Final[Symbol] = y
        self.p: Final[Symbol] = p

    @classmethod
    def _of_native(cls, R: native.FirstIntegral) -> FirstIntegral:
        return cls(_sympy_expr(R.factor), _sympy_expr(R.integral), _sympy_symbol(R.x), _sympy_symbol(R.y),
                   _sympy_symbol(R.p))

    def _native(self) -> native.FirstIntegral:
        return native.FirstIntegral(_core_expr(self.factor), _core_expr(self.integral), *_symbols(self.x, self.y,
                                                                                                    self.p))

    def __repr__(self) -> str:
        return "FirstIntegral(%s, %s)" % (self.factor, self.integral)

    def equation(self, f: AppliedUndef) -> Eq:
        """``R(x, f, f') = C1`` as an equation in the function ``f``."""
        converted = to_sympy(self._native().equation(_core_function(f)))
        if not isinstance(converted, Eq):
            raise TypeError("expected an equation, got %s" % (converted,))
        return converted


class Linearization:
    """A point transformation ``t = phi(x, y)``, ``u = psi(x, y)`` taking a
    second order equation to ``u'' = rhs(u')``
    (:class:`sympy_extras.solvers.second_order.Linearization` with SymPy
    attributes).

    Attributes
    ==========

    t, u : Expr
        The new independent and dependent variables as expressions in
        ``x`` and ``y``.
    rhs : Expr
        The right-hand side of the transformed equation, in ``v`` for
        ``u'``.
    """

    def __init__(self, t: Expr, u: Expr, x: Symbol, y: Symbol, v: Symbol, rhs: Expr) -> None:
        self.t: Final[Expr] = t
        self.u: Final[Expr] = u
        self.x: Final[Symbol] = x
        self.y: Final[Symbol] = y
        self.v: Final[Symbol] = v
        self.rhs: Final[Expr] = rhs

    @classmethod
    def _of_native(cls, L: native.Linearization) -> Linearization:
        return cls(_sympy_expr(L.t), _sympy_expr(L.u), _sympy_symbol(L.x), _sympy_symbol(L.y), _sympy_symbol(L.v),
                   _sympy_expr(L.rhs))

    def __repr__(self) -> str:
        if self.rhs == 0:
            return "Linearization(t=%s, u=%s)" % (self.t, self.u)
        return "Linearization(t=%s, u=%s, rhs=%s)" % (self.t, self.u, self.rhs)


def _first_integral(R: Optional[native.FirstIntegral]) -> Optional[FirstIntegral]:
    return None if R is None else FirstIntegral._of_native(R)


def _linearization(L: Optional[native.Linearization]) -> Optional[Linearization]:
    return None if L is None else Linearization._of_native(L)


# ---------------------------------------------------------------------------
# the face

def second_order_rhs(equation: Basic, f: AppliedUndef) -> tuple[Expr, Symbol, Symbol, Symbol]:
    """``(Phi, x, y, p)`` with the equation written as ``y'' = Phi(x, y, p)``
    (:func:`sympy_extras.solvers.second_order.second_order_rhs` on SymPy
    objects; ``y`` and ``p`` are dummies).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.second_order_sympy import second_order_rhs
    >>> y = Function('y')(x)
    >>> second_order_rhs(x*y.diff(x, 2) + y.diff(x) - y**2, y)[0]
    (-_p + _y**2)/x
    """
    Phi, x, y, p = native.second_order_rhs(_core_basic(equation), _core_function(f))
    return _sympy_expr(Phi), _sympy_symbol(x), _sympy_symbol(y), _sympy_symbol(p)


def integrating_factor_xy(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[FirstIntegral]:
    """An integrating factor ``mu(x, y)`` of ``y'' = Phi`` and the first
    integral ``mu y' + S(x, y)``, or ``None``
    (:func:`sympy_extras.solvers.second_order.integrating_factor_xy` on
    SymPy objects).

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.second_order_sympy import integrating_factor_xy
    >>> x, y, p = symbols('x y p')
    >>> integrating_factor_xy(-p**2/y, x, y, p)          # y y'' + y'^2 = 0
    FirstIntegral(y, p*y)
    >>> integrating_factor_xy(-p/x, x, y, p)             # x y'' + y' = 0
    FirstIntegral(x, p*x)
    """
    return _first_integral(native.integrating_factor_xy(_core_expr(Phi), *_symbols(x, y, p)))


def integrating_factor_p(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[FirstIntegral]:
    """An integrating factor ``mu(y')`` of ``y'' = Phi`` and the first
    integral ``M(y') + S(x, y)``, or ``None``
    (:func:`sympy_extras.solvers.second_order.integrating_factor_p` on
    SymPy objects).

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.second_order_sympy import integrating_factor_p
    >>> x, y, p = symbols('x y p')
    >>> integrating_factor_p(p**3*(x + y*p), x, y, p)     # y'' = y'^3 (x + y y')
    FirstIntegral(p**(-3), -x**2/2 - y**2/2 - 1/(2*p**2))
    """
    return _first_integral(native.integrating_factor_p(_core_expr(Phi), *_symbols(x, y, p)))


def is_linearizable(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> bool:
    """Lie's test: whether ``y'' = Phi`` is equivalent to ``u'' = 0`` under
    a point transformation
    (:func:`sympy_extras.solvers.second_order.is_linearizable` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.second_order_sympy import is_linearizable
    >>> x, y, p = symbols('x y p')
    >>> is_linearizable(-3*y*p - y**3, x, y, p)
    True
    >>> is_linearizable(y**2, x, y, p)
    False
    """
    return native.is_linearizable(_core_expr(Phi), *_symbols(x, y, p))


def linearize(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[Linearization]:
    """A point transformation taking ``y'' = Phi`` to ``u'' = 0``, or
    ``None`` (:func:`sympy_extras.solvers.second_order.linearize` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.second_order_sympy import linearize
    >>> x, y, p = symbols('x y p')
    >>> linearize(p**2, x, y, p)             # y'' = y'^2, u = exp(-y)
    Linearization(t=x, u=-exp(-y))
    """
    return _linearization(native.linearize(_core_expr(Phi), *_symbols(x, y, p)))


def transformed_rhs(Phi: Expr, t: Expr, u: Expr, x: Symbol, y: Symbol, p: Symbol, v: Symbol) -> Optional[Expr]:
    """``y'' = Phi`` written in the variables ``t(x, y)``, ``u(x, y)`` as
    ``u'' = rhs(v)``, or ``None``
    (:func:`sympy_extras.solvers.second_order.transformed_rhs` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.second_order_sympy import transformed_rhs
    >>> x, y, p, v = symbols('x y p v')
    >>> transformed_rhs(-3*y*p - y**3, x - 1/y, 1/(2*y**2), x, y, p, v)
    1
    """
    found = native.transformed_rhs(_core_expr(Phi), _core_expr(t), _core_expr(u), *_symbols(x, y, p),
                                   _core_symbol(v))
    return None if found is None else _sympy_expr(found)


def commuting_pair(symmetries: Sequence[Symmetry], x: Symbol, y: Symbol) -> Optional[tuple[Symmetry, Symmetry]]:
    """Two commuting symmetries which are pointwise independent as vector
    fields, or ``None``
    (:func:`sympy_extras.solvers.second_order.commuting_pair` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie_sympy import JetSpace, Symmetry
    >>> from sympy_extras.solvers.second_order_sympy import commuting_pair
    >>> x = symbols('x')
    >>> jets = JetSpace([Function('y')(x)])
    >>> u = jets.u[0]
    >>> commuting_pair([Symmetry(jets, [1], [0]), Symmetry(jets, [x], [u]), Symmetry(jets, [0], [1])], x, u)
    (Symmetry(xi=(1,), eta=(0,)), Symmetry(xi=(0,), eta=(1,)))
    """
    if not symmetries:
        return None
    jets = symmetries[0].jets
    found = native.commuting_pair([s.native for s in symmetries], _core_symbol(x), _core_symbol(y))
    if found is None:
        return None
    return Symmetry._of_native(jets, found[0]), Symmetry._of_native(jets, found[1])


def rectify_symmetries(Phi: Expr, x: Symbol, y: Symbol, p: Symbol, f: AppliedUndef,
                       degree: int = 3) -> Optional[Linearization]:
    """Coordinates in which two commuting point symmetries of
    ``y'' = Phi`` are the translations, or ``None``
    (:func:`sympy_extras.solvers.second_order.rectify_symmetries` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.second_order_sympy import rectify_symmetries
    >>> x, y, p = symbols('x y p')
    >>> f = Function('y')(x)
    >>> rectify_symmetries(-3*y*p - y**3, x, y, p, f)      # y'' + 3 y y' + y^3 = 0
    Linearization(t=x - 1/y, u=1/(2*y**2), rhs=1)
    """
    return _linearization(native.rectify_symmetries(_core_expr(Phi), *_symbols(x, y, p), _core_function(f),
                                                    degree))


def dsolve_second_order(equation: Basic, f: AppliedUndef, degree: int = 3,
                        special_values: bool = True) -> Optional[list[Basic]]:
    """Solutions of a second order equation through an integrating
    factor, its linearisation or two commuting symmetries; ``None`` when
    none applies (:func:`sympy_extras.solvers.second_order.dsolve_second_order`
    on SymPy objects).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.second_order_sympy import dsolve_second_order
    >>> y = Function('y')(x)
    >>> dsolve_second_order(y.diff(x, 2) - y.diff(x)**2, y)
    [Eq(y(x), log(-1/(C1*x + C2)))]
    >>> dsolve_second_order(y.diff(x, 2) + 3*y*y.diff(x) + y**3, y)
    [Eq(y(x), 2*(C2 + x)/(2*C1 + 2*C2*x + x**2))]
    """
    return _optional_list(native.dsolve_second_order(_core_basic(equation), _core_function(f), degree,
                                                     special_values))

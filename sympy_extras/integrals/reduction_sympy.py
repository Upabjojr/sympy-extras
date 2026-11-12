"""The SymPy face of reduction-based creative telescoping
(:mod:`sympy_extras.integrals.reduction`), for the integrators which are
still written on SymPy (:mod:`.definite`) and the old interface.

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the telescopers of
:mod:`.telescoping_sympy`, the values of :mod:`.conditions_sympy`).
Nothing of the module stays on SymPy: only the initial values and the
numerical check of :func:`reduction_integral` reach it, through
:mod:`.telescoping_sympy`.

Examples
========

>>> from sympy import symbols, exp, oo
>>> from sympy_extras.integrals.reduction_sympy import hermite_reduce, reduction_telescoper, reduction_integral
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> g, r = hermite_reduce(1/x**2, -2*x, x)
>>> g, r
(-1/x, -2)
>>> reduction_telescoper(exp(-x**2 - t**2/x**2), x, t)
DifferentialTelescoper([-4, 0, 1], 2/x)
>>> reduction_integral(exp(-x**2 - t**2/x**2), x, 0, oo, t)
ConditionalValue(sqrt(pi)*exp(-2*t)/2)
"""
from __future__ import annotations

from typing import Final, Optional

from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.polys.polytools import Poly

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions

from . import reduction as native
from .conditions_sympy import ConditionalValue
from .telescoping_sympy import DifferentialTelescoper, _core_symbol, _optional_eq

__all__ = ['HermiteReducer', 'hermite_reduce', 'reduction_telescoper', 'reduction_ode', 'reduction_integral']


class HermiteReducer:
    """:class:`sympy_extras.integrals.reduction.HermiteReducer` on SymPy
    objects: the Hermite reduction with respect to ``D(g) = g' + q*g``.

    Attributes
    ==========

    shell : Expr
        The polynomial ``S`` with ``q = q_hat + S'/S``.
    A, B : Poly
        Numerator and denominator of ``q_hat`` (polynomials in ``x`` over
        the field of the parameters).
    q : Expr
        ``q_hat``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.reduction_sympy import HermiteReducer
    >>> x, t = symbols('x t')
    >>> reducer = HermiteReducer(2/x + 1/(x - t)**2, x, t)
    >>> reducer.shell, reducer.A.as_expr(), reducer.B.as_expr()
    (x**2, 1, t**2 - 2*t*x + x**2)
    >>> g, r = reducer.reduce(1/x)
    >>> g, r
    (t**2/2 - t*x + x**2/2, t - 1/2)
    """

    def __init__(self, q: ExprLike, x: Symbol, t: Symbol) -> None:
        self.x: Final[Symbol] = x
        self.t: Final[Symbol] = t
        self.native: Final[native.HermiteReducer] = native.HermiteReducer(core_expr(sympify(q)), _core_symbol(x),
                                                                          _core_symbol(t))

    @property
    def shell(self) -> Expr:
        return sympy_expr(self.native.shell)

    @property
    def A(self) -> Poly:
        return Poly(sympy_expr(self.native.A), self.x, field=True)

    @property
    def B(self) -> Poly:
        return Poly(sympy_expr(self.native.B), self.x, field=True)

    @property
    def q(self) -> Expr:
        return sympy_expr(self.native.q)

    def _image(self, j: int) -> Poly:
        """``E(x**j) = B (x**j)' + (A + B') x**j``, the image of the
        monomials in the polynomial reduction."""
        return Poly(sympy_expr(self.native.context.poly_expr(self.native._image(j))), self.x, field=True)

    def derivation(self, g: ExprLike) -> Expr:
        """``D(g) = g' + q_hat*g`` (with the normalised ``q``)."""
        return sympy_expr(self.native.derivation(core_expr(sympify(g))))

    def reduce(self, p: ExprLike) -> tuple[Expr, Expr]:
        """``(g, r)`` with ``p*S = D(g) + r`` and ``r`` a reduced form."""
        g, r = self.native.reduce(core_expr(sympify(p)))
        return sympy_expr(g), sympy_expr(r)


def hermite_reduce(p: ExprLike, q: ExprLike, x: Symbol, t: Optional[Symbol] = None) -> tuple[Expr, Expr]:
    """:func:`sympy_extras.integrals.reduction.hermite_reduce` on SymPy
    objects: ``(g, r)`` with ``p = g' + q*g + r`` and ``r`` a reduced form.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.reduction_sympy import hermite_reduce
    >>> x, t = symbols('x t')
    >>> hermite_reduce(x, -2*x, x)          # x exp(-x**2) = (-exp(-x**2)/2)'
    (-1/2, 0)
    >>> hermite_reduce(1, -2*x, x)          # exp(-x**2) is not integrable
    (0, 1)
    >>> hermite_reduce(1/(x**2 + t), 0, x, t)
    (0, 1/(t + x**2))
    """
    g, r = native.hermite_reduce(core_expr(sympify(p)), core_expr(sympify(q)), _core_symbol(x),
                                 None if t is None else _core_symbol(t))
    return sympy_expr(g), sympy_expr(r)


def reduction_telescoper(F: ExprLike, x: Symbol, t: Symbol, max_order: int = 6) -> Optional[DifferentialTelescoper]:
    """:func:`sympy_extras.integrals.reduction.reduction_telescoper` on SymPy
    objects: the minimal telescoper of a hyperexponential ``F(x, t)`` (or a
    sum of such terms) by Hermite reduction, or ``None``.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.reduction_sympy import reduction_telescoper
    >>> x, t = symbols('x t')
    >>> found = reduction_telescoper(1/(x**2 + t**2), x, t)
    >>> found.coefficients, found.certificate, found.check()
    ([1, t], -x, True)
    >>> reduction_telescoper(exp(-t*x**2) + exp(-t**2*x**2), x, t).coefficients
    [1, 5*t, 2*t**2]
    """
    found = native.reduction_telescoper(core_expr(sympify(F)), _core_symbol(x), _core_symbol(t), max_order)
    return None if found is None else DifferentialTelescoper.from_core(found)


def reduction_ode(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                  assumptions: Assumptions = None, name: str = 'I') -> Optional[Eq]:
    """:func:`sympy_extras.integrals.reduction.reduction_ode` on SymPy
    objects.

    >>> from sympy import symbols, exp, oo
    >>> from sympy_extras.integrals.reduction_sympy import reduction_ode
    >>> x, t = symbols('x t')
    >>> reduction_ode(exp(-x**2)*exp(2*x*t), x, -oo, oo, t)
    Eq(-2*t*I(t) + Derivative(I(t), t), 0)
    """
    return _optional_eq(native.reduction_ode(core_expr(sympify(F)), _core_symbol(x), core_expr(sympify(a)),
                                             core_expr(sympify(b)), _core_symbol(t), core_assumptions(assumptions),
                                             name))


def reduction_integral(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.reduction.reduction_integral` on SymPy
    objects: ``Integral(F, (x, a, b))`` as a function of ``t`` by the
    holonomic method with the reduction-based telescoper.

    >>> from sympy import symbols, exp, cos, oo
    >>> from sympy_extras.integrals.reduction_sympy import reduction_integral
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> reduction_integral(exp(-x**2)*cos(2*t*x), x, 0, oo, t)
    ConditionalValue(sqrt(pi)*exp(-t**2)/2)
    """
    found = native.reduction_integral(core_expr(sympify(F)), _core_symbol(x), core_expr(sympify(a)),
                                      core_expr(sympify(b)), _core_symbol(t), core_assumptions(assumptions))
    return None if found is None else ConditionalValue.from_core(found)

"""The SymPy face of the first order solvers
(:mod:`sympy_extras.solvers.first_order`), and the steps which are not
ported.

:func:`riccati_ode`, :func:`chini_ode`, :func:`abel_ode`,
:func:`lagrange_ode` and :func:`dsolve_first_order` take and return
SymPy objects, converted by :mod:`sympy_extras.interop` around the
native functions. The private helpers ``_derivative_polynomial`` and
``_explicit_rhs`` stay written on SymPy for
:mod:`~sympy_extras.solvers.darboux`, which is written on SymPy too.

The linear equation of a Lagrange equation and the inverse Riccati
equations of the AIR class (:mod:`~sympy_extras.solvers.abel`) are solved
by the native :func:`sympy_extras.solvers.dsolve.dsolve`, and the
antiderivatives are native
(:func:`sympy_extras.integrals.indefinite.closed_antiderivative`).

Examples
========

>>> from sympy import Function
>>> from sympy.abc import x
>>> from sympy_extras.solvers.first_order_sympy import riccati_ode
>>> y = Function('y')(x)
>>> riccati_ode(y.diff(x) + y**2 - 2/x**2, y)
Eq(y(x), (2*C1*x**3 - 1)/(C1*x**4 + x))
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.function import AppliedUndef

from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.interop import from_sympy, to_sympy

from . import first_order as native

__all__ = ['riccati_ode', 'chini_ode', 'abel_ode', 'lagrange_ode', 'dsolve_first_order']


# ---------------------------------------------------------------------------
# conversions

def _core_basic(e: Basic) -> CoreBasic:
    converted = from_sympy(e)
    if not isinstance(converted, CoreBasic):
        raise TypeError("%s cannot be converted" % (e,))
    return converted


def _core_function(f: AppliedUndef) -> CoreAppliedUndef:
    converted = from_sympy(f)
    if not isinstance(converted, CoreAppliedUndef):
        raise TypeError("%s is not an applied undefined function" % (f,))
    return converted


def _sympy_function(f: CoreAppliedUndef) -> AppliedUndef:
    converted = to_sympy(f)
    if not isinstance(converted, AppliedUndef):
        raise TypeError("%s is not an applied undefined function" % (f,))
    return converted


def _sympy_basic(e: CoreBasic) -> Basic:
    converted = to_sympy(e)
    if not isinstance(converted, Basic):
        raise TypeError("%s cannot be converted" % (e,))
    return converted


def _optional(found: Optional[CoreBasic]) -> Optional[Basic]:
    return None if found is None else _sympy_basic(found)


# ---------------------------------------------------------------------------
# the steps of the native modules which stay on SymPy


# ---------------------------------------------------------------------------
# the face

def chini_ode(equation: Basic, f: AppliedUndef) -> Optional[Basic]:
    """An implicit solution of Chini's equation ``y' = f(x) y**n + g(x) y +
    h(x)`` (``n != 1`` a number) when its invariant is constant, else
    ``None`` (:func:`sympy_extras.solvers.first_order.chini_ode` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.first_order_sympy import chini_ode
    >>> y = Function('y')(x)
    >>> chini_ode(y.diff(x) - x**2*y**3 + x**2, y)      # doctest: +ELLIPSIS
    Eq(...)
    """
    return _optional(native.chini_ode(_core_basic(equation), _core_function(f)))


def abel_ode(equation: Basic, f: AppliedUndef) -> Optional[Basic]:
    """An implicit solution of Abel's equations ``y' = f3 y**3 + f2 y**2 +
    f1 y + f0`` (first kind) and ``(y + g) y' = f2 y**2 + f1 y + f0``
    (second kind) in the constant-invariant cases, else ``None``
    (:func:`sympy_extras.solvers.first_order.abel_ode` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.first_order_sympy import abel_ode
    >>> y = Function('y')(x)
    >>> abel_ode(y.diff(x) - y**3 - 3*y**2 - 3*y, y)     # doctest: +ELLIPSIS
    Eq(...)
    """
    return _optional(native.abel_ode(_core_basic(equation), _core_function(f)))


def lagrange_ode(equation: Basic, f: AppliedUndef) -> Optional[list[Basic]]:
    """The parametric solution ``[Eq(x, X(p)), Eq(y, Y(p))]`` of a
    d'Alembert–Lagrange equation ``y = x F(y') + G(y')`` (``p = y'`` the
    parameter), or ``None`` (:func:`sympy_extras.solvers.first_order.lagrange_ode`
    on SymPy objects).

    Examples
    ========

    >>> from sympy import Function, Eq
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.first_order_sympy import lagrange_ode
    >>> y = Function('y')(x)
    >>> lagrange_ode(Eq(y, 2*x*y.diff(x) + y.diff(x)**2), y)
    [Eq(x, C1/p**2 - 2*p/3), Eq(y(x), (6*C1 - p**3)/(3*p))]
    """
    found = native.lagrange_ode(_core_basic(equation), _core_function(f))
    return None if found is None else [_sympy_basic(e) for e in found]


def riccati_ode(equation: Basic, f: AppliedUndef, special_values: bool = True) -> Optional[Basic]:
    """The general solution of a Riccati equation ``y' = a(x) y**2 + b(x) y
    + c(x)`` with rational coefficients through its linearisation
    (:func:`sympy_extras.solvers.first_order.riccati_ode` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.first_order_sympy import riccati_ode
    >>> y = Function('y')(x)
    >>> riccati_ode(y.diff(x) + y**2 - 2/x**2, y)
    Eq(y(x), (2*C1*x**3 - 1)/(C1*x**4 + x))
    """
    return _optional(native.riccati_ode(_core_basic(equation), _core_function(f), special_values))


def dsolve_first_order(equation: Basic, f: AppliedUndef, special_values: bool = True) -> Optional[Basic]:
    """The Riccati general solution, the Chini/Abel implicit solution,
    the Prelle–Singer first integral or the Lagrange parametric solution
    (as a ``Tuple``) of a first order equation, else ``None``
    (:func:`sympy_extras.solvers.first_order.dsolve_first_order` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function, Eq
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.first_order_sympy import dsolve_first_order
    >>> y = Function('y')(x)
    >>> dsolve_first_order(Eq(y, 2*x*y.diff(x) + y.diff(x)**2), y)
    (Eq(x, C1/p**2 - 2*p/3), Eq(y(x), (6*C1 - p**3)/(3*p)))
    """
    return _optional(native.dsolve_first_order(_core_basic(equation), _core_function(f), special_values))

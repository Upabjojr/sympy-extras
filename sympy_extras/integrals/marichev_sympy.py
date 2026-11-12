"""The SymPy face of the Marichev–Adamchik method
(:mod:`sympy_extras.integrals.marichev`), for the integrators which are
still written on SymPy (:mod:`.definite`,
:mod:`.algebraic`, :mod:`.series`, ...).

The functions take and return SymPy objects, converted by
:mod:`sympy_extras.interop` around the native code: the values as
:class:`~.conditions_sympy.ConditionalValue`, the decomposed integrands
as :class:`~.mellin_sympy.Product`. Nothing of the module stays on SymPy
(the native ``hyperexpand``, ``simplify``, ``refine`` and limits lend the
nodes they have not ported to SymPy through the bridge of the core).

Examples
========

>>> from sympy import symbols, exp, sin, log
>>> from sympy_extras.integrals.marichev_sympy import mellin_integrate
>>> x, k = symbols('x k')
>>> a, b = symbols('a b', positive=True)
>>> mellin_integrate(exp(-a*x)*sin(b*x), x)
ConditionalValue(b/(a**2 + b**2))
>>> mellin_integrate(x**k/(x + 3), x)
ConditionalValue(-3**k*pi/sin(pi*k), (k > -1) & (k < 0))
>>> mellin_integrate(exp(-x**2)*log(x), x)
ConditionalValue(sqrt(pi)*(-log(4) - EulerGamma)/4)
>>> mellin_integrate(x**(b - 1)*(1 - x)**(a - 1), x, cutoff='lower')
ConditionalValue(gamma(a)*gamma(b)/gamma(a + b))
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean, true

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions, core_boolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy

from . import marichev as native
from .conditions_sympy import ConditionalValue
from .mellin_sympy import GammaQuotient, Product

__all__ = ['mellin_integrate', 'integrate_product', 'evaluate_quotient', 'tidy', 'checked_tidy', 'real_logarithms',
           'polar_elliptic', 'polar_lowergamma', 'right_half_plane_powers', 'bessel_k_forms', 'exponential_form',
           'reduce_positive_powers']


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def evaluate_quotient(quotient: GammaQuotient, point: Expr) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.marichev.evaluate_quotient` on SymPy
    objects: the value of the quotient at ``point``, by a limit when a pole
    of the numerator meets a zero of the denominator; ``None`` when the
    limit is not found."""
    found = native.evaluate_quotient(quotient.to_core(), core_expr(point))
    return None if found is None else sympy_expr(found)


def tidy(value: Expr, assumptions: Assumptions = None, condition: Boolean = true) -> Expr:
    """:func:`sympy_extras.integrals.marichev.tidy` on SymPy objects: the
    value with polar numbers removed and simplified, under the time limit
    of the settings; ``refine`` uses the assumptions and the condition
    under which the value holds."""
    return sympy_expr(native.tidy(core_expr(value), core_assumptions(assumptions), core_boolean(condition)))


def checked_tidy(value: Expr, assumptions: Assumptions = None, condition: Boolean = true) -> Expr:
    """:func:`sympy_extras.integrals.marichev.checked_tidy` on SymPy
    objects: :func:`tidy` with a safety net, ``value`` itself when the
    tidied form differs from it numerically.

    Examples
    ========

    >>> from sympy import symbols, exp_polar, log, sqrt
    >>> from sympy_extras.integrals.marichev_sympy import checked_tidy
    >>> a = symbols('a')
    >>> checked_tidy(sqrt(a**2) + log(a*exp_polar(0)), a > 0)
    a + log(a)
    """
    return sympy_expr(native.checked_tidy(core_expr(value), core_assumptions(assumptions), core_boolean(condition)))


def polar_elliptic(value: Expr) -> Expr:
    """:func:`sympy_extras.integrals.marichev.polar_elliptic` on SymPy
    objects: the parameter ``m*exp_polar(I*pi)`` of a Legendre elliptic
    function written ``-m``.

    >>> from sympy import elliptic_k, exp_polar, I, pi, symbols
    >>> from sympy_extras.integrals.marichev_sympy import polar_elliptic
    >>> a = symbols('a', positive=True)
    >>> polar_elliptic(elliptic_k(4*exp_polar(I*pi)/a**2))
    elliptic_k(-4/a**2)
    """
    return sympy_expr(native.polar_elliptic(core_expr(value)))


def polar_lowergamma(value: Expr) -> Expr:
    """:func:`sympy_extras.integrals.marichev.polar_lowergamma` on SymPy
    objects: ``lowergamma(a, w*exp_polar(I*pi))`` as the real form
    ``(-1)**a * w**a * hyper((a,), (a + 1,), w) / a``.

    >>> from sympy import lowergamma, exp_polar, I, pi, symbols, Rational
    >>> from sympy_extras.integrals.marichev_sympy import polar_lowergamma
    >>> w = symbols('w', positive=True)
    >>> polar_lowergamma(lowergamma(Rational(1, 3), w*exp_polar(I*pi)))
    3*(-1)**(1/3)*w**(1/3)*hyper((1/3,), (4/3,), w)
    """
    return sympy_expr(native.polar_lowergamma(core_expr(value)))


def right_half_plane_powers(value: Expr, assumptions: Assumptions = None) -> Expr:
    """:func:`sympy_extras.integrals.marichev.right_half_plane_powers` on
    SymPy objects: ``(w**2)**p`` with ``w`` of positive real part under the
    assumptions written ``w**(2*p)``.

    >>> from sympy import I, sqrt, symbols
    >>> from sympy_extras.integrals.marichev_sympy import right_half_plane_powers
    >>> a, b = symbols('a b', positive=True)
    >>> right_half_plane_powers(sqrt((a + I*b)**2)/(a + I*b))
    1
    """
    return sympy_expr(native.right_half_plane_powers(core_expr(value), core_assumptions(assumptions)))


def bessel_k_forms(value: Expr) -> Expr:
    """:func:`sympy_extras.integrals.marichev.bessel_k_forms` on SymPy
    objects: ``besseli(-n, z)`` written with ``besselk(n, z)`` where
    ``besseli(n, z)`` occurs too.

    >>> from sympy import symbols, besseli, sin, pi
    >>> from sympy_extras.integrals.marichev_sympy import bessel_k_forms
    >>> n, z = symbols('n z')
    >>> bessel_k_forms(pi*(besseli(-n, z) - besseli(n, z))/sin(pi*n))
    2*besselk(n, z)
    """
    return sympy_expr(native.bessel_k_forms(core_expr(value)))


def real_logarithms(value: Expr, assumptions: Assumptions = None) -> Expr:
    """:func:`sympy_extras.integrals.marichev.real_logarithms` on SymPy
    objects: ``log(u)`` with ``u`` negative under the assumptions written
    ``log(-u) + I*pi``.

    >>> from sympy import I, log, symbols
    >>> from sympy_extras.integrals.marichev_sympy import real_logarithms
    >>> c = symbols('c')
    >>> real_logarithms(I*c**2*(log(c**2) - log(-c**2)), c < 0)
    pi*c**2
    """
    return sympy_expr(native.real_logarithms(core_expr(value), core_assumptions(assumptions)))


def integrate_product(product: Product, assumptions: Assumptions = None,
                      regularize: bool = False) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.marichev.integrate_product` on SymPy
    objects: ``Integral(product, (x, 0, oo))`` for a decomposed integrand;
    ``None`` when the method does not apply."""
    found = native.integrate_product(product.to_core(), core_assumptions(assumptions), regularize)
    return None if found is None else ConditionalValue.from_core(found)


def exponential_form(term: Expr, x: Symbol) -> Expr:
    """:func:`sympy_extras.integrals.marichev.exponential_form` on SymPy
    objects: the trigonometric and hyperbolic factors of ``term`` written
    as exponentials, the linear exponentials of each term combined.

    >>> from sympy import symbols, cosh, exp
    >>> from sympy_extras.integrals.marichev_sympy import exponential_form
    >>> x, a = symbols('x a')
    >>> exponential_form(exp(-x)*cosh(a*x), x)
    exp(x*(a - 1))/2 + exp(-x*(a + 1))/2
    """
    return sympy_expr(native.exponential_form(core_expr(term), _core_symbol(x)))


def reduce_positive_powers(f: Expr, x: Symbol) -> Expr:
    """:func:`sympy_extras.integrals.marichev.reduce_positive_powers` on
    SymPy objects: ``(1 + b*x**g)**e`` with a positive non-integer ``e``
    written as a sum of terms with the kernel ``(1 + b*x**g)**(e - n)``.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.marichev_sympy import reduce_positive_powers
    >>> x = symbols('x')
    >>> reduce_positive_powers(sqrt(1 + x), x)
    x/sqrt(x + 1) + 1/sqrt(x + 1)
    """
    return sympy_expr(native.reduce_positive_powers(core_expr(f), _core_symbol(x)))


def mellin_integrate(f: ExprLike, x: Symbol, assumptions: Assumptions = None,
                     cutoff: Optional[str] = None, regularize: bool = False) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.marichev.mellin_integrate` on SymPy
    objects: ``Integral(f, (x, 0, oo))`` by the Marichev–Adamchik method,
    or over ``(0, 1)`` with ``cutoff='lower'`` and over ``(1, oo)`` with
    ``cutoff='upper'``; ``None`` when a term of the integrand is not of
    the form of the method or the conditions are refuted by the
    assumptions.

    Examples
    ========

    >>> from sympy import symbols, exp, cos, besselk, log
    >>> from sympy_extras.integrals.marichev_sympy import mellin_integrate
    >>> x, s = symbols('x s')
    >>> a, nu = symbols('a nu', positive=True)
    >>> mellin_integrate(exp(-s*x)*cos(a*x), x, s > 0)
    ConditionalValue(s/(a**2 + s**2))
    >>> mellin_integrate(x**(2*nu)*besselk(nu, x), x)
    ConditionalValue(2**(2*nu)*gamma((nu + 1)/2)*gamma((3*nu + 1)/2)/2)
    >>> mellin_integrate(log(x)**2, x, cutoff='lower')
    ConditionalValue(2)
    """
    found = native.mellin_integrate(core_expr(sympify(f)), _core_symbol(x), core_assumptions(assumptions), cutoff,
                                    regularize)
    return None if found is None else ConditionalValue.from_core(found)

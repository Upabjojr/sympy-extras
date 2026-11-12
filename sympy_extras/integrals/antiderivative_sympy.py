"""The SymPy face of the definite integrals from an antiderivative
(:mod:`sympy_extras.integrals.antiderivative`), for the integrators which
are still written on SymPy (:mod:`.definite`, :mod:`.series`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`). The native module has no step left on SymPy:
the polynomials in elementary functions and the parts of the
integrations by parts, which SymPy's ``integrate`` did, are integrated by
the native :func:`~sympy_extras.integrals.antiderivative.elementary_antiderivative`.

Examples
========

>>> from sympy import symbols, cos, pi
>>> from sympy_extras.integrals.antiderivative_sympy import antiderivative_integral
>>> x = symbols('x')
>>> antiderivative_integral(1/(2 + cos(x)), x, 0, 2*pi)
ConditionalValue(2*sqrt(3)*pi/3)
>>> antiderivative_integral(1/x**2, x, -1, 1)
ConditionalValue(oo)
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr

from . import antiderivative as native
from . import conditions as native_conditions
from .conditions_sympy import ConditionalValue

__all__ = ['antiderivative', 'elementary_polynomial', 'quick_shape', 'by_parts_shape', 'singularities',
           'discontinuities', 'antiderivative_integral', 'one_sided_limit', 'select_branch', 'real_logarithms',
           'principal_value_integral', 'finite_part_integral']


def _conditional(found: Optional[native_conditions.ConditionalValue]) -> Optional[ConditionalValue]:
    return None if found is None else ConditionalValue.from_core(found)


def _optional(found: Optional[CoreExpr]) -> Optional[Expr]:
    return None if found is None else sympy_expr(found)


def _points(found: Optional[list[CoreExpr]]) -> Optional[list[Expr]]:
    return None if found is None else [sympy_expr(p) for p in found]


def _core(e: ExprLike) -> CoreExpr:
    return core_expr(sympify(e))


def antiderivative(f: Expr, x: Symbol, assumptions: Assumptions = None, late: bool = False) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.antiderivative.antiderivative` on SymPy
    objects: an antiderivative of ``f`` by the radical table, by the rules
    of ``manualintegrate`` for a polynomial in elementary functions, by parts for a
    logarithm or an error function times one, and by the verified methods
    of :mod:`.indefinite` otherwise (the heuristic ones with ``late``);
    ``None`` when none is found.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.antiderivative_sympy import antiderivative
    >>> x = symbols('x')
    >>> a = symbols('a', positive=True)
    >>> antiderivative(exp(-a*x), x)
    -exp(-a*x)/a
    """
    return _optional(native.antiderivative(_core(f), _core_symbol(x), core_assumptions(assumptions), late))


def elementary_polynomial(f: Expr, x: Symbol) -> bool:
    """:func:`sympy_extras.integrals.antiderivative.elementary_polynomial`
    on SymPy objects: whether ``f`` is a polynomial in ``x`` and in sines,
    cosines, exponentials and hyperbolic functions of linear arguments.

    >>> from sympy import symbols, sin, cos
    >>> from sympy_extras.integrals.antiderivative_sympy import elementary_polynomial
    >>> t = symbols('t')
    >>> elementary_polynomial(t**2*cos(3*t - 1), t), elementary_polynomial(sin(t**2), t)
    (True, False)
    """
    return native.elementary_polynomial(_core(f), _core_symbol(x))


def quick_shape(f: Expr, x: Symbol) -> bool:
    """:func:`sympy_extras.integrals.antiderivative.quick_shape` on SymPy
    objects.

    >>> from sympy import symbols, exp, log
    >>> from sympy_extras.integrals.antiderivative_sympy import quick_shape
    >>> x = symbols('x')
    >>> quick_shape(x*exp(-x)*log(x), x)
    True
    """
    return native.quick_shape(_core(f), _core_symbol(x))


def by_parts_shape(f: Expr, x: Symbol) -> bool:
    """:func:`sympy_extras.integrals.antiderivative.by_parts_shape` on SymPy
    objects: whether ``f`` is a polynomial in ``x`` and exponentials times
    ``log(x)``.

    >>> from sympy import symbols, exp, log
    >>> from sympy_extras.integrals.antiderivative_sympy import by_parts_shape
    >>> x = symbols('x')
    >>> by_parts_shape(x*exp(-x)*log(x), x), by_parts_shape(x*log(x), x)
    (True, False)
    """
    return native.by_parts_shape(_core(f), _core_symbol(x))


def singularities(f: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[list[Expr]]:
    """:func:`sympy_extras.integrals.antiderivative.singularities` on SymPy
    objects: the real isolated singularities of ``f`` under the
    assumptions; ``None`` when they are not a finite set.

    >>> from sympy import symbols, log
    >>> from sympy_extras.integrals.antiderivative_sympy import singularities
    >>> x = symbols('x')
    >>> singularities(1/(x - 1) + log(x), x, [x > -1, x < 2])
    [1, 0]
    """
    return _points(native.singularities(_core(f), _core_symbol(x), core_assumptions(assumptions)))


def discontinuities(F: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                    assumptions: Assumptions = None) -> Optional[list[Expr]]:
    """:func:`sympy_extras.integrals.antiderivative.discontinuities` on
    SymPy objects: the points of ``(a, b)`` where the antiderivative ``F``
    may be discontinuous, sorted; ``None`` when they cannot be located.

    >>> from sympy import symbols, atan, tan, pi
    >>> from sympy_extras.integrals.antiderivative_sympy import discontinuities
    >>> x = symbols('x')
    >>> discontinuities(2*atan(tan(x/2)), x, 0, 2*pi)
    [pi]
    """
    return _points(native.discontinuities(_core(F), _core_symbol(x), _core(a), _core(b),
                                          core_assumptions(assumptions)))


def select_branch(F: Expr, x: Symbol, a: ExprLike, b: ExprLike, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.antiderivative.select_branch` on SymPy
    objects: the branch of a ``Piecewise`` antiderivative which holds on
    ``(a, b)``; ``None`` when it is undecided or changes inside the range.

    >>> from sympy import symbols, Piecewise
    >>> from sympy_extras.integrals.antiderivative_sympy import select_branch
    >>> t = symbols('t')
    >>> select_branch(Piecewise((t**2, t > 1), (t**3, True)), t, 0, 1)
    t**3
    """
    return _optional(native.select_branch(_core(F), _core_symbol(x), _core(a), _core(b),
                                          core_assumptions(assumptions)))


def one_sided_limit(F: Expr, x: Symbol, point: ExprLike, direction: str,
                    assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.antiderivative.one_sided_limit` on
    SymPy objects: ``lim F(x)`` as ``x`` tends to ``point`` from
    ``direction``; ``oo`` or ``-oo`` for a signed infinity, ``None`` when
    the limit does not exist or is not found.

    >>> from sympy import symbols, atan, tan, pi
    >>> from sympy_extras.integrals.antiderivative_sympy import one_sided_limit
    >>> x = symbols('x')
    >>> one_sided_limit(2*atan(tan(x/2)), x, pi, '-'), one_sided_limit(2*atan(tan(x/2)), x, pi, '+')
    (pi, -pi)
    """
    return _optional(native.one_sided_limit(_core(F), _core_symbol(x), _core(point), direction,
                                            core_assumptions(assumptions)))


def real_logarithms(F: Expr, x: Symbol) -> Expr:
    """:func:`sympy_extras.integrals.antiderivative.real_logarithms` on
    SymPy objects: ``log(u)`` written ``log(Abs(u))`` for the real
    arguments ``u`` depending on ``x``.

    >>> from sympy import symbols, log
    >>> from sympy_extras.integrals.antiderivative_sympy import real_logarithms
    >>> x = symbols('x')
    >>> real_logarithms(log(x - 1), x)
    log(Abs(x - 1))
    """
    return sympy_expr(native.real_logarithms(_core(F), _core_symbol(x)))


def antiderivative_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                            assumptions: Assumptions = None, late: bool = False,
                            known: Optional[Expr] = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.antiderivative.antiderivative_integral`
    on SymPy objects: ``Integral(f, (x, a, b))`` from an antiderivative
    (``known``, or by :func:`antiderivative`) cut at its discontinuities
    and at the singularities of ``f`` and evaluated by one-sided limits;
    ``oo`` or ``-oo`` for a divergence to a signed infinity, ``None`` when
    a step fails or the infinities cancel.

    >>> from sympy import symbols, pi, cos, oo
    >>> from sympy_extras.integrals.antiderivative_sympy import antiderivative_integral
    >>> x = symbols('x')
    >>> antiderivative_integral(1/(5 - 4*cos(x)), x, 0, 2*pi)
    ConditionalValue(2*pi/3)
    >>> antiderivative_integral(1/(1 + x**2), x, -oo, oo)
    ConditionalValue(pi)
    >>> antiderivative_integral(1/x, x, 0, 1)
    ConditionalValue(oo)
    """
    return _conditional(native.antiderivative_integral(_core(f), _core_symbol(x), _core(a), _core(b),
                                                       core_assumptions(assumptions), late,
                                                       None if known is None else _core(known)))


def principal_value_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                             assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.antiderivative.principal_value_integral`
    on SymPy objects: Cauchy's principal value of ``Integral(f, (x, a, b))``.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.antiderivative_sympy import principal_value_integral
    >>> x = symbols('x')
    >>> principal_value_integral(1/x, x, -1, 2)
    ConditionalValue(log(2))
    >>> principal_value_integral(1/x**2, x, -1, 1) is None
    True
    """
    return _conditional(native.principal_value_integral(_core(f), _core_symbol(x), _core(a), _core(b),
                                                        core_assumptions(assumptions)))


def finite_part_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                         assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.antiderivative.finite_part_integral` on
    SymPy objects: Hadamard's finite part of ``Integral(f, (x, a, b))``.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.antiderivative_sympy import finite_part_integral
    >>> x = symbols('x')
    >>> finite_part_integral(1/x**2, x, -1, 1)
    ConditionalValue(-2)
    >>> finite_part_integral(1/x, x, -1, 2)
    ConditionalValue(log(2))
    """
    return _conditional(native.finite_part_integral(_core(f), _core_symbol(x), _core(a), _core(b),
                                                    core_assumptions(assumptions)))


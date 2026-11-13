"""Antiderivatives of expressions with Dirac delta functions.

Ported from SymPy's ``sympy.integrals.deltafunctions``. The integral of
``f(x)*DiracDelta(x - a)`` is ``f(a)*Heaviside(x - a)``; a delta of a
polynomial argument is first written by its simple roots,
``DiracDelta(g(x)) = sum(DiracDelta(x - r)/|g'(r)|)``, and the
derivatives of the delta are integrated by parts, ``f*DiracDelta(x - a,
n)`` giving ``(-1)**n f^{(n)}(a) Heaviside(x - a)`` or, when that
vanishes, the delta of lower order [Gelfand]_ (chapter 1).

The module works on the objects of the core and imports no SymPy; the
terms without a delta are integrated by the verified methods of
:mod:`.indefinite`.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import DiracDelta, cos, exp
>>> from sympy_extras.integrals.deltafunctions import deltaintegrate
>>> x, y = symbols('x y')
>>> deltaintegrate(exp(x)*DiracDelta(x - 1), x)
E*Heaviside(x - 1)
>>> deltaintegrate(DiracDelta(x**2 - 1), x)
Heaviside(x - 1)/2 + Heaviside(x + 1)/2
>>> deltaintegrate(y*DiracDelta(x, 1), x)
y*DiracDelta(x)

References
==========

.. [Gelfand] I. M. Gel'fand, G. E. Shilov, *Generalized functions*,
   volume 1, Academic Press, 1964.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/deltafunctions.py,
# sympy/functions/special/delta_functions.py (DiracDelta.is_simple,
# DiracDelta._eval_expand_diracdelta) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing on the core; the expansion and the simplicity
# test are functions; the terms without a delta go to the verified
# methods of the native indefinite.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import Pow
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.complexes import abs_
from sympy_extras.functions.special.delta_functions import DiracDelta, dirac_delta, heaviside
from sympy_extras.polys.polyerrors import BasePolynomialError

__all__ = ['deltaintegrate', 'change_mul', 'is_simple', 'expand_diracdelta']


def _order(d: DiracDelta) -> Expr:
    return d.args[1] if len(d.args) > 1 else Integer(0)


# PORTED FROM SYMPY: sympy/functions/special/delta_functions.py:DiracDelta.is_simple
def is_simple(d: DiracDelta, x: Symbol) -> bool:
    """Whether the argument of ``d`` is linear in ``x``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import DiracDelta
    >>> from sympy_extras.integrals.deltafunctions import is_simple
    >>> x, y = symbols('x y')
    >>> is_simple(DiracDelta(x*y), x), is_simple(DiracDelta(x**2 + 1), x)
    (True, False)
    """
    p = d.args[0].as_poly(x)
    return p is not None and p.degree() == 1


# PORTED FROM SYMPY: sympy/functions/special/delta_functions.py:DiracDelta._eval_expand_diracdelta
def expand_diracdelta(d: DiracDelta, x: Symbol) -> Expr:
    """``d`` written by the simple real roots of its argument,
    ``sum(DiracDelta(x - r)/|g'(r)|)``; ``d`` itself when its argument has
    no ``x``, a complex or multiple root, or ``d`` is a derivative.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import DiracDelta
    >>> from sympy_extras.integrals.deltafunctions import expand_diracdelta
    >>> x, y = symbols('x y')
    >>> expand_diracdelta(DiracDelta(3*x**2 - 3*x - 6), x)
    DiracDelta(x - 2)/9 + DiracDelta(x + 1)/9
    """
    from sympy_extras.polys.polyroots import roots
    g = d.args[0]
    if not g.has(x) or _order(d) != 0:
        return d
    try:
        found = roots(g, x)
    except (BasePolynomialError, ValueError, TypeError):
        return d
    if not isinstance(found, dict) or not found:
        return d
    derivative = abs_(g.diff(x))
    result: Expr = Integer(0)
    for r, m in found.items():
        if not isinstance(r, Expr) or r.is_real is False or m != 1:
            return d
        result = result + dirac_delta(x - r) / derivative.subs(x, r)
    return result


# PORTED FROM SYMPY: sympy/integrals/deltafunctions.py:change_mul
def change_mul(node: Expr, x: Symbol) -> tuple[Optional[DiracDelta], Optional[Expr]]:
    """The factors of the product ``node`` with a simple delta in front:
    ``(delta, rest)``; ``(None, expanded)`` when there is no simple delta
    but the deltas could be expanded, ``(None, None)`` when nothing
    changes.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import DiracDelta, cos
    >>> from sympy_extras.integrals.deltafunctions import change_mul
    >>> x, y = symbols('x y')
    >>> change_mul(x*y*DiracDelta(x)*cos(x), x)
    (DiracDelta(x), x*y*cos(x))
    >>> change_mul(x*y*DiracDelta(x**2 - 1)*cos(x), x)
    (None, x*y*cos(x)*DiracDelta(x - 1)/2 + x*y*cos(x)*DiracDelta(x + 1)/2)
    >>> change_mul(x*y*DiracDelta(cos(x))*cos(x), x)
    (None, None)
    """
    new_args: list[Expr] = []
    dirac: Optional[DiracDelta] = None
    sorted_args = sorted(Mul.make_args(node), key=default_sort_key)
    for arg in sorted_args:
        if isinstance(arg, Pow) and isinstance(arg.base, DiracDelta):
            new_args.append(arg.base**(arg.exp - 1))
            factor: Expr = arg.base
        else:
            factor = arg
        if dirac is None and isinstance(factor, DiracDelta) and is_simple(factor, x):
            dirac = factor
        else:
            new_args.append(factor)
    if dirac is None:
        expanded_args: list[Expr] = []
        for arg in sorted_args:
            if isinstance(arg, DiracDelta):
                expanded_args.append(expand_diracdelta(arg, x))
            elif isinstance(arg, Pow) and isinstance(arg.base, DiracDelta):
                expanded_args.append(expand_diracdelta(arg.base, x)**arg.exp)
            else:
                expanded_args.append(arg)
        if expanded_args != sorted_args:
            return None, expand(mul(*expanded_args))
        return None, None
    return dirac, mul(*new_args)


def _integrate(g: Expr, x: Symbol) -> Optional[Expr]:
    """An antiderivative of ``g``: its terms with a delta by
    :func:`deltaintegrate`, the others by the verified methods."""
    from .indefinite import verified_antiderivative
    parts: list[Expr] = []
    for term in Add.make_args(g):
        coeff, rest = term.as_independent(x, as_Add=False)
        if not rest.has(x):
            parts.append(term * x)
            continue
        if rest.has(DiracDelta):
            found = deltaintegrate(rest, x)
        else:
            verified = verified_antiderivative(rest, x)
            found = None if verified is None else verified[0]
        if found is None:
            return None
        parts.append(coeff * found)
    return add(*parts)


def _solve_linear(e: Expr, x: Symbol) -> Optional[Expr]:
    p = e.as_poly(x)
    if p is None or p.degree() != 1:
        return None
    coefficients = p.all_coeffs()
    return -coefficients[1] / coefficients[0]


# PORTED FROM SYMPY: sympy/integrals/deltafunctions.py:deltaintegrate
def deltaintegrate(f: Expr, x: Symbol) -> Optional[Expr]:
    """An antiderivative of ``f``, which has delta functions: a simple
    delta (of a linear argument) integrates to a Heaviside function times
    the rest at its point, the rest's derivatives for the derivatives of
    the delta (by parts); a delta of a polynomial argument is written by
    its simple roots first. ``None`` when ``f`` has no delta or a delta
    which cannot be simplified (``DiracDelta(cos(x))``).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import DiracDelta, sin, cos
    >>> from sympy_extras.integrals.deltafunctions import deltaintegrate
    >>> x, y, z = symbols('x y z')
    >>> deltaintegrate(x*sin(x)*cos(x)*DiracDelta(x - 1), x)
    sin(1)*cos(1)*Heaviside(x - 1)
    >>> deltaintegrate(y**2*DiracDelta(x - z)*DiracDelta(y - z), y)
    z**2*DiracDelta(x - z)*Heaviside(y - z)
    """
    if not f.has(DiracDelta):
        return None
    if isinstance(f, DiracDelta):
        h = expand_diracdelta(f, x)
        if h == f:
            if is_simple(f, x):
                if _order(f) == 0:
                    return heaviside(f.args[0])
                p = f.args[0].as_poly(x)
                if p is None:
                    return None
                return dirac_delta(f.args[0], _order(f) - 1) / p.LC()
            return None
        return _integrate(h, x)
    if isinstance(f, (Mul, Pow)):
        g = expand(f)
        if f != g:
            return _integrate(g, x)
        deltaterm, rest_mult = change_mul(f, x)
        if deltaterm is None:
            if rest_mult is not None:
                return _integrate(rest_mult, x)
            return None
        rest = rest_mult if rest_mult is not None else Integer(1)
        expanded = expand_diracdelta(deltaterm, x)
        if isinstance(expanded, Mul):
            # factors taken out by the expansion
            inner, rest_2 = change_mul(expanded, x)
            if inner is None or rest_2 is None:
                return None
            deltaterm = inner
            rest = rest * rest_2
        elif isinstance(expanded, DiracDelta):
            deltaterm = expanded
        else:
            return None
        point = _solve_linear(deltaterm.args[0], x)
        if point is None:
            return None
        order = _order(deltaterm)
        if not isinstance(order, Integer):
            return None
        # the largest hyperreal term left by the integrations by parts:
        # the integral of y*DiracDelta(x, 1) is y*DiracDelta(x), not 0
        n, m = order.p, 0
        while n >= 0:
            r = Integer(-1)**n * rest.diff((x, n)).subs(x, point) if n else rest.subs(x, point)
            if r.is_zero:
                n -= 1
                m += 1
            else:
                if m == 0:
                    return r * heaviside(x - point)
                return r * dirac_delta(x - point, m - 1)
        # every term vanished: the derivative of the delta integrates to 0
        return Integer(0)
    return None

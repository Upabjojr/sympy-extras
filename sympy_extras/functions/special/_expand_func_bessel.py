"""The ``expand(func=True)`` rules of the Airy functions and of the
spherical harmonics (SymPy's ``_eval_expand_func`` of ``airyai``,
``airybi``, ``airyaiprime``, ``airybiprime`` and ``Ynm``), as a typed
table for :func:`sympy_extras.core.function.expand`. Ported from SymPy's
``sympy.functions.special.bessel`` and
``sympy.functions.special.spherical_harmonics``.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/bessel.py (airyai, airybi, airyaiprime, airybiprime
# ._eval_expand_func), sympy/functions/special/spherical_harmonics.py (Ynm._eval_expand_func) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table keyed by class in place of methods.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

from sympy_extras.core.expr import Expr
from sympy_extras.core.function import ExpandHints
from sympy_extras.core.numbers import I, Integer, Rational, pi
from sympy_extras.core.power import pow, sqrt
from sympy_extras.core.symbol import Symbol, Wild

__all__ = ['rules', 'ynm_expand_func']

Rule = Callable[[Expr, ExpandHints], Optional[Expr]]

HALF = Rational(1, 2)
ONE = Integer(1)


def _airy_match(node: Expr) -> Optional[tuple[Expr, Expr, Expr, Expr, Symbol]]:
    """SymPy's ``arg.match(c*(d*z**n)**m)`` of the argument (``z`` its only
    symbol), ``(c, d, n, m, z)`` when it matches with ``3*m`` an integer."""
    arg = node.args[0]
    assert isinstance(arg, Expr)
    symbols = arg.free_symbols
    if len(symbols) != 1:
        return None
    [z] = symbols
    if not isinstance(z, Symbol):
        return None
    c, d, m, n = (Wild(name, exclude=[z]) for name in ('c', 'd', 'm', 'n'))
    matched = arg.match(c * pow(d * pow(z, n), m))
    if matched is None:
        return None
    values: list[Expr] = []
    for w in (c, d, n, m):
        v = matched.get(w)
        if not isinstance(v, Expr):
            return None
        values.append(v)
    if (3 * values[3]).is_integer is not True:
        return None
    return values[0], values[1], values[2], values[3], z


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyai._eval_expand_func
def _airyai(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from .bessel import airyai, airybi
    found = _airy_match(node)
    if found is None:
        return None
    c, d, n, m, z = found
    pf = pow(d * pow(z, n), m) / (pow(d, m) * pow(z, m * n))
    newarg = c * pow(d, m) * pow(z, m * n)
    return HALF * ((pf + ONE) * airyai(newarg) - (pf - ONE) / sqrt(Integer(3)) * airybi(newarg))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airybi._eval_expand_func
def _airybi(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from .bessel import airyai, airybi
    found = _airy_match(node)
    if found is None:
        return None
    c, d, n, m, z = found
    pf = pow(d * pow(z, n), m) / (pow(d, m) * pow(z, m * n))
    newarg = c * pow(d, m) * pow(z, m * n)
    return HALF * (sqrt(Integer(3)) * (ONE - pf) * airyai(newarg) + (ONE + pf) * airybi(newarg))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyaiprime._eval_expand_func
def _airyaiprime(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from .bessel import airyaiprime, airybiprime
    found = _airy_match(node)
    if found is None:
        return None
    c, d, n, m, z = found
    pf = (pow(d, m) * pow(z, n * m)) / pow(d * pow(z, n), m)
    newarg = c * pow(d, m) * pow(z, n * m)
    return HALF * ((pf + ONE) * airyaiprime(newarg) + (pf - ONE) / sqrt(Integer(3)) * airybiprime(newarg))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airybiprime._eval_expand_func
def _airybiprime(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from .bessel import airyaiprime, airybiprime
    found = _airy_match(node)
    if found is None:
        return None
    c, d, n, m, z = found
    pf = (pow(d, m) * pow(z, n * m)) / pow(d * pow(z, n), m)
    newarg = c * pow(d, m) * pow(z, n * m)
    return HALF * (sqrt(Integer(3)) * (pf - ONE) * airyaiprime(newarg) + (pf + ONE) * airybiprime(newarg))


# PORTED FROM SYMPY: sympy/functions/special/spherical_harmonics.py:Ynm._eval_expand_func
def ynm_expand_func(node: Expr) -> Expr:
    """``Ynm(n, m, theta, phi)`` in terms of the associated Legendre
    function of ``cos(theta)``, ``sin(theta)`` for ``sqrt(1 - cos(theta)**2)``."""
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.elementary.exponential import exp
    from sympy_extras.functions.elementary.trigonometric import cos, sin
    from .polynomials import assoc_legendre
    n, m, theta, phi = (a for a in node.args if isinstance(a, Expr))
    rv = (sqrt((2 * n + 1) / (4 * pi) * factorial(n - m) / factorial(n + m))
          * exp(I * m * phi) * assoc_legendre(n, m, cos(theta)))
    replaced = rv.subs(sqrt(-cos(theta)**2 + 1), sin(theta))
    assert isinstance(replaced, Expr)
    return replaced


def _ynm(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    return ynm_expand_func(node)


def rules() -> dict[type, Rule]:
    """The rules keyed by the class of the node."""
    from . import bessel as b
    from .spherical_harmonics import YnmNode
    return {b.AiryAi: _airyai, b.AiryBi: _airybi, b.AiryAiPrime: _airyaiprime, b.AiryBiPrime: _airybiprime,
            YnmNode: _ynm}

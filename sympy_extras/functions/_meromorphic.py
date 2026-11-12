"""The rules of ``is_meromorphic`` of the special functions with one of
their own in SymPy (the incomplete gamma functions, the Bessel, Hankel
and spherical Bessel functions), as a table by class name for
:meth:`~sympy_extras.core.function.DefinedFunction._eval_is_meromorphic`.

``lowergamma(s, z)`` is ``z**s*gamma(s)*gammastar(s, z)`` with
``gammastar`` entire, so its singularities are the branch point ``z = 0``
and the poles of ``gamma(s)``; ``uppergamma`` has the same rule. A Bessel
function of an order which does not depend on ``x`` is meromorphic at a
finite point, except at ``z = 0`` for a noninteger order (and for the
functions of the second kind of order 0) [1]_.

The module is a typed port of the ``_eval_is_meromorphic`` methods of
SymPy's :mod:`sympy.functions.special.gamma_functions` and
:mod:`sympy.functions.special.bessel`.

References
==========

.. [1] F. W. J. Olver et al., *NIST Digital Library of Mathematical
       Functions*, sections 8.2 and 10.2, https://dlmf.nist.gov
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py (lowergamma, uppergamma
#   _eval_is_meromorphic), sympy/functions/special/bessel.py (BesselBase._eval_is_meromorphic)
#   (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table of functions by class name in place of methods.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

from sympy_extras.core.expr import Expr
from sympy_extras.core.fuzzy import fuzzy_and, fuzzy_not, fuzzy_or
from sympy_extras.core.symbol import Symbol

__all__ = ['rules']

Rule = Callable[[Expr, Symbol, Expr], Optional[bool]]


def _arg(f: Expr, i: int) -> Expr:
    a = f.args[i]
    assert isinstance(a, Expr)
    return a


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:lowergamma._eval_is_meromorphic
def _incomplete_gamma(f: Expr, x: Symbol, a: Expr) -> Optional[bool]:
    s, z = _arg(f, 0), _arg(f, 1)
    args_merom = fuzzy_and([z._eval_is_meromorphic(x, a), s._eval_is_meromorphic(x, a)])
    if not args_merom:
        return args_merom
    z0 = z.subs(x, a)
    if s.is_integer:
        return fuzzy_and([s.is_positive, z0.is_finite])
    s0 = s.subs(x, a)
    return fuzzy_and([s0.is_finite, z0.is_finite, fuzzy_not(z0.is_zero)])


def _bessel(entire_at_integer_order: bool) -> Rule:
    # PORTED FROM SYMPY: sympy/functions/special/bessel.py:BesselBase._eval_is_meromorphic
    def rule(f: Expr, x: Symbol, a: Expr) -> Optional[bool]:
        nu, z = _arg(f, 0), _arg(f, 1)
        if nu.has(x):
            return False
        if not z._eval_is_meromorphic(x, a):
            return None
        z0 = z.subs(x, a)
        if nu.is_integer:
            if entire_at_integer_order or not nu.is_zero:
                return fuzzy_not(z0.is_infinite)
        return fuzzy_not(fuzzy_or([z0.is_zero, z0.is_infinite]))
    return rule


def rules() -> dict[str, Rule]:
    """The rules by class name (besselj, besseli, the spherical functions
    jn, yn, hn1, hn2 have no singularity at ``z = 0`` for an integer
    order; bessely, besselk, hankel1, hankel2 have one at order 0)."""
    entire, second_kind = _bessel(True), _bessel(False)
    return {'LowerGamma': _incomplete_gamma, 'UpperGamma': _incomplete_gamma,
            'BesselJ': entire, 'BesselI': entire, 'SphericalJ': entire, 'SphericalY': entire,
            'SphericalHankel1': entire, 'SphericalHankel2': entire,
            'BesselY': second_kind, 'BesselK': second_kind, 'Hankel1': second_kind, 'Hankel2': second_kind}

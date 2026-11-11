"""The SymPy face of quantifier elimination
(:mod:`sympy_extras.assumptions.resolve`).

:func:`resolve` takes and returns SymPy objects (the quantifiers of
:mod:`sympy_extras.assumptions.quantifiers_sympy`), converted by
:mod:`sympy_extras.interop` around the native function. Nothing of the
elimination stays on SymPy: the comprehensive Gröbner systems over the
complex numbers are native too (:mod:`sympy_extras.polys.comprehensive`,
whose face is :mod:`sympy_extras.polys.comprehensive_sympy`).
"""
from __future__ import annotations

from typing import Optional, Union

from sympy.core.singleton import S
from sympy.logic.boolalg import Boolean
from sympy.sets.sets import Set

from . import resolve as native
from .ask_sympy import Assumptions, core_assumptions, core_boolean, core_domain, sympy_boolean

__all__ = ['resolve']


# ---------------------------------------------------------------------------
# the face

def resolve(formula: Union[Boolean, bool], domain: Optional[Set] = S.Reals,
            assumptions: Assumptions = None, method: Optional[str] = None) -> Boolean:
    """Eliminate the quantifiers of a formula over the real numbers
    (:func:`sympy_extras.assumptions.resolve.resolve` on SymPy objects),
    the counterpart of Mathematica's ``Resolve[expr, Reals]`` (and of
    ``Reduce`` for quantifier-free formulas).

    Parameters
    ==========

    formula : Boolean
        A Boolean combination of polynomial relations with rational
        coefficients, memberships in ``S.Reals`` and sign predicates, with
        quantifiers :class:`~sympy_extras.assumptions.quantifiers_sympy.ForAll`
        and :class:`~sympy_extras.assumptions.quantifiers_sympy.Exists`
        anywhere (the formula is put in prenex form first).
    domain : Set
        ``S.Reals`` (the default, also for ``None``), ``S.Integers``
        (Presburger arithmetic, Cooper's algorithm) or ``S.Complexes``
        (equations and inequations, comprehensive Gröbner systems), see
        :func:`sympy_extras.assumptions.resolve.resolve`.
    assumptions : Boolean or list of Booleans, optional
        Polynomial assumptions on the free variables. The result is
        equivalent to the input under the assumptions and does not repeat
        them.
    method : ``'mccallum'``, ``'hong'`` or None
        The projection operator of the cylindrical algebraic decomposition.

    Returns
    =======

    ``S.true`` or ``S.false`` for a closed formula, otherwise a
    quantifier-free formula in the free variables equivalent to the input
    over the reals (with one free variable a union of intervals with exact
    endpoints).

    Examples
    ========

    >>> from sympy import S, Eq
    >>> from sympy.abc import a, b, c, x, y
    >>> from sympy_extras.assumptions.resolve_sympy import resolve
    >>> from sympy_extras.assumptions.quantifiers_sympy import ForAll, Exists
    >>> resolve(ForAll(x, x**2 + b*x + c > 0))
    b**2 < 4*c
    >>> resolve(Exists(x, Eq(a*x**2 + b*x + c, 0) & (a > 0)))
    (a > 0) & (4*a*c - b**2 <= 0)
    >>> resolve(ForAll(x, Exists(y, y > x)))
    True
    >>> resolve(Exists(y, ForAll(x, y > x)))
    False
    >>> resolve(Exists(y, Eq(x**2 + y**2, 1)))
    (x >= -1) & (x <= 1)
    >>> resolve(~ForAll(x, x**2 > 0))
    True
    >>> from sympy import CRootOf       # SymPy prints a CRootOf in the variable it cached first
    >>> resolve(x**2 > 2) == (x > CRootOf(x**2 - 2, 1)) | (x < CRootOf(x**2 - 2, 0))
    True
    >>> resolve(Exists(x, Eq(2*x, y)), domain=S.Integers)
    Eq(Mod(y, 2), 0)
    >>> resolve(Exists(x, Eq(x**2 + 1, 0)), domain=S.Complexes)
    True
    >>> resolve(Exists(x, Eq(a*x, 1)), domain=S.Complexes)
    Ne(a, 0)
    >>> resolve(ForAll(x, Exists(y, Eq(3*y, x) | Eq(3*y, x + 1) | Eq(3*y, x + 2))), domain=S.Integers)
    True
    """
    try:
        native_domain = core_domain(S.Reals if domain is None else domain)
    except TypeError:
        raise NotImplementedError(
            "quantifier elimination is only implemented over the real numbers, "
            "the complex numbers and, for linear formulas, the integers")
    return sympy_boolean(native.resolve(core_boolean(formula), native_domain, core_assumptions(assumptions), method))

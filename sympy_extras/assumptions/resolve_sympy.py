"""The SymPy face of quantifier elimination
(:mod:`sympy_extras.assumptions.resolve`), and the eliminations which are
not ported.

:func:`resolve` takes and returns SymPy objects (the quantifiers of
:mod:`sympy_extras.assumptions.quantifiers_sympy`), converted by
:mod:`sympy_extras.interop` around the native function.

The eliminations stay written on SymPy's polynomials and are called from
the native ``resolve`` at run time (the functions ending in ``_of_core``,
which take and return objects of the core): the cylindrical algebraic
decomposition (:func:`sympy_extras.polys.cad.quantifier_elimination`),
virtual substitution (:func:`~sympy_extras.polys.virtual_substitution.virtual_substitution_elimination`,
:func:`~sympy_extras.polys.virtual_substitution.with_sides`), Cooper's
algorithm (:func:`~sympy_extras.solvers.integers.is_presburger`,
:func:`~sympy_extras.solvers.integers.presburger_quantifier_elimination`)
and comprehensive Gröbner systems
(:func:`~sympy_extras.polys.comprehensive.complex_quantifier_elimination`,
:func:`~sympy_extras.polys.comprehensive.reduce_complex`).
"""
from __future__ import annotations

from typing import Optional, Union

from sympy.core.singleton import S
from sympy.logic.boolalg import Boolean
from sympy.sets.sets import Set

from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol

from . import resolve as native
from .ask_sympy import (Assumptions, _sympy_symbol, core_assumptions, core_boolean, core_domain, core_prefix,
    sympy_boolean, sympy_prefix)
from .quantifiers import Prefix

__all__ = ['resolve', 'quantifier_elimination_of_core', 'virtual_substitution_elimination_of_core',
    'with_sides_of_core', 'is_presburger_of_core', 'presburger_quantifier_elimination_of_core',
    'complex_quantifier_elimination_of_core', 'reduce_complex_of_core']


# ---------------------------------------------------------------------------
# the eliminations of the native resolve which stay on SymPy

def quantifier_elimination_of_core(formula: CoreBoolean, prefix: Prefix, free: list[CoreSymbol],
                                   method: Optional[str]) -> CoreBoolean:
    """:func:`sympy_extras.polys.cad.quantifier_elimination`."""
    from sympy_extras.polys.cad.qe import quantifier_elimination
    return core_boolean(quantifier_elimination(sympy_boolean(formula), sympy_prefix(prefix),
                                               free=[_sympy_symbol(s) for s in free], method=method))


def virtual_substitution_elimination_of_core(formula: CoreBoolean, prefix: Prefix) -> tuple[CoreBoolean, Prefix]:
    """:func:`sympy_extras.polys.virtual_substitution.virtual_substitution_elimination`."""
    from sympy_extras.polys.virtual_substitution import virtual_substitution_elimination
    result, remaining = virtual_substitution_elimination(sympy_boolean(formula), sympy_prefix(prefix))
    return core_boolean(result), core_prefix(remaining)


def with_sides_of_core(formula: CoreBoolean) -> CoreBoolean:
    """:func:`sympy_extras.polys.virtual_substitution.with_sides`."""
    from sympy_extras.polys.virtual_substitution import with_sides
    return core_boolean(with_sides(sympy_boolean(formula)))


def is_presburger_of_core(formula: CoreBoolean, integers: set[CoreSymbol]) -> bool:
    """:func:`sympy_extras.solvers.integers.is_presburger`."""
    from sympy_extras.solvers.integers import is_presburger
    return is_presburger(sympy_boolean(formula), {_sympy_symbol(s) for s in integers})


def presburger_quantifier_elimination_of_core(formula: CoreBoolean, prefix: Prefix) -> CoreBoolean:
    """:func:`sympy_extras.solvers.integers.presburger_quantifier_elimination`."""
    from sympy_extras.solvers.integers import presburger_quantifier_elimination
    return core_boolean(presburger_quantifier_elimination(sympy_boolean(formula), sympy_prefix(prefix)))


def complex_quantifier_elimination_of_core(formula: CoreBoolean, prefix: Prefix) -> CoreBoolean:
    """:func:`sympy_extras.polys.comprehensive.complex_quantifier_elimination`."""
    from sympy_extras.polys.comprehensive import complex_quantifier_elimination
    return core_boolean(complex_quantifier_elimination(sympy_boolean(formula), sympy_prefix(prefix)))


def reduce_complex_of_core(formula: CoreBoolean) -> CoreBoolean:
    """:func:`sympy_extras.polys.comprehensive.reduce_complex`."""
    from sympy_extras.polys.comprehensive import reduce_complex
    return core_boolean(reduce_complex(sympy_boolean(formula)))


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
    >>> resolve(x**2 > 2)
    (x > CRootOf(x**2 - 2, 1)) | (x < CRootOf(x**2 - 2, 0))
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

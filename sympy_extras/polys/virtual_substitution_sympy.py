"""The SymPy face of quantifier elimination by virtual substitution
(:mod:`sympy_extras.polys.virtual_substitution`).

The functions take and return SymPy objects (formulas, symbols and
quantifier prefixes of SymPy symbols), converted by
:mod:`sympy_extras.interop` around the native functions.

Examples
========

>>> from sympy import Eq
>>> from sympy.abc import a, b, c, x, y
>>> from sympy_extras.polys.virtual_substitution_sympy import eliminate_linear, eliminate_quadratic
>>> eliminate_linear((x > y) & (x < 1), x), eliminate_linear(Eq(a*x, 1), x)
(y < 1, Ne(a, 0))
>>> eliminate_quadratic(Eq(x**2 + b*x + c, 0), x)
b**2 >= 4*c
"""
from __future__ import annotations

from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean

from sympy_extras._typing_sympy import QuantifierPrefix
from sympy_extras.assumptions.ask_sympy import core_boolean, core_prefix, sympy_boolean, sympy_prefix
from sympy_extras.assumptions.intervals_sympy import core_symbol
from sympy_extras.core.logic import Boolean as CoreBoolean

from . import virtual_substitution as native

__all__ = ['is_linear_in', 'is_quadratic_in', 'eliminate_linear', 'eliminate_quadratic',
           'linear_quantifier_elimination', 'virtual_substitution_elimination', 'with_sides']


# ---------------------------------------------------------------------------
# the face

def _core(formula: Boolean) -> CoreBoolean:
    """The formula as a Boolean of the core (``TypeError`` for anything
    else)."""
    return core_boolean(formula)


def is_linear_in(formula: Boolean, x: Symbol) -> bool:
    """Whether every relation of the formula is linear in ``x``
    (:func:`sympy_extras.polys.virtual_substitution.is_linear_in`).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.virtual_substitution_sympy import is_linear_in
    >>> is_linear_in((2*x + y > 0) & (x*y < 1), x)
    True
    >>> is_linear_in(x**2 + y > 0, x)
    False
    """
    return native.is_linear_in(_core(formula), core_symbol(x))


def eliminate_linear(formula: Boolean, x: Symbol) -> Boolean:
    """`\\exists x\\, \\varphi` for a formula linear in ``x``, as a
    quantifier-free formula in the other variables
    (:func:`sympy_extras.polys.virtual_substitution.eliminate_linear` on
    SymPy objects).

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, a
    >>> from sympy_extras.polys.virtual_substitution_sympy import eliminate_linear
    >>> eliminate_linear((x > y) & (x < 1), x)
    y < 1
    >>> eliminate_linear(Eq(a*x, 1), x)
    Ne(a, 0)
    >>> eliminate_linear((x > 0) & (x < 0), x)
    False
    """
    return sympy_boolean(native.eliminate_linear(_core(formula), core_symbol(x)))


def with_sides(formula: Boolean) -> Boolean:
    """Every relation ``p - q rel 0`` of the formula written as ``p rel q``
    (:func:`sympy_extras.polys.virtual_substitution.with_sides`).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.virtual_substitution_sympy import with_sides
    >>> with_sides((x - y > 0) & (x + 1 < 0))
    (x > y) & (x < -1)
    """
    return sympy_boolean(native.with_sides(_core(formula)))


def linear_quantifier_elimination(formula: Boolean, prefix: QuantifierPrefix
                                  ) -> tuple[Boolean, QuantifierPrefix]:
    """Eliminate, from the innermost outwards, the quantified variables in
    which the formula is linear
    (:func:`sympy_extras.polys.virtual_substitution.linear_quantifier_elimination`
    on SymPy objects).

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.virtual_substitution_sympy import linear_quantifier_elimination
    >>> linear_quantifier_elimination((x > y) & (y > z), [('exists', x)])
    (y > z, [])
    >>> linear_quantifier_elimination((x > y) & (y > z), [('forall', z), ('exists', x)])
    (False, [])
    >>> linear_quantifier_elimination((x**2 > y) & (y > z), [('exists', x), ('exists', z)])
    (x**2 > y, [('exists', x)])
    """
    result, remaining = native.linear_quantifier_elimination(_core(formula), core_prefix(prefix))
    return sympy_boolean(result), sympy_prefix(remaining)


def is_quadratic_in(formula: Boolean, x: Symbol) -> bool:
    """Whether every relation of the formula has degree at most two in
    ``x`` (:func:`sympy_extras.polys.virtual_substitution.is_quadratic_in`).

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.virtual_substitution_sympy import is_quadratic_in
    >>> is_quadratic_in((x**2 + y*x > 0) & (x < 1), x)
    True
    >>> is_quadratic_in(x**3 > y, x)
    False
    """
    return native.is_quadratic_in(_core(formula), core_symbol(x))


def eliminate_quadratic(formula: Boolean, x: Symbol) -> Boolean:
    """`\\exists x\\, \\varphi` for a formula of degree at most two in ``x``
    (:func:`sympy_extras.polys.virtual_substitution.eliminate_quadratic`
    on SymPy objects).

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, a, b, c
    >>> from sympy_extras.polys.virtual_substitution_sympy import eliminate_quadratic
    >>> eliminate_quadratic(Eq(x**2 + b*x + c, 0), x)
    b**2 >= 4*c
    >>> eliminate_quadratic((x**2 < a) & (x > 1), x)
    a > 1
    """
    return sympy_boolean(native.eliminate_quadratic(_core(formula), core_symbol(x)))


def virtual_substitution_elimination(formula: Boolean, prefix: QuantifierPrefix, max_atoms: int = 12,
                                     max_free: int = 2) -> tuple[Boolean, QuantifierPrefix]:
    """Eliminate, innermost first, the quantified variables in which the
    formula is linear or quadratic
    (:func:`sympy_extras.polys.virtual_substitution.virtual_substitution_elimination`
    on SymPy objects).

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, b, c
    >>> from sympy_extras.polys.virtual_substitution_sympy import virtual_substitution_elimination
    >>> virtual_substitution_elimination(Eq(x**2 + b*x + c, 0), [('exists', x)])
    (b**2 >= 4*c, [])
    >>> virtual_substitution_elimination(x**2 + b*x + c > 0, [('forall', x)])
    (b**2 < 4*c, [])
    """
    result, remaining = native.virtual_substitution_elimination(_core(formula), core_prefix(prefix),
                                                                max_atoms=max_atoms, max_free=max_free)
    return sympy_boolean(result), sympy_prefix(remaining)

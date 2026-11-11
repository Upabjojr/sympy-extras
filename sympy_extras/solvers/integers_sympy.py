"""The SymPy face of the linear problems over the integers
(:mod:`sympy_extras.solvers.integers`), and the step which stays on
SymPy.

:func:`linear_diophantine_system`, :func:`cooper`,
:func:`presburger_quantifier_elimination` and :func:`is_presburger` take
and return SymPy objects, converted by :mod:`sympy_extras.interop`
around the native functions; :func:`hermite_normal_form_with_transform`,
:func:`hilbert_basis` and :func:`minimal_nonnegative_solutions` work on
integer matrices and are the native functions themselves.

The step of the native module which stays written on SymPy is called
from it at run time (:func:`simplify_logic_of_core`, which takes and
returns a formula of the core, the function of
:mod:`sympy_extras.polys.virtual_substitution_sympy`): SymPy's
``simplify_logic`` of the formulas returned by Cooper's algorithm.

Examples
========

>>> from sympy import Eq, Mod
>>> from sympy.abc import x, y, z
>>> from sympy_extras.solvers.integers_sympy import linear_diophantine_system, cooper
>>> linear_diophantine_system([Eq(3*x + 5*y, 7)], [x, y])
([14 - 5*t0, 3*t0 - 7], [t0])
>>> cooper((3*x > y) & (3*x < y + 3), x)
Eq(Mod(y + 1, 3), 0) | Eq(Mod(y + 2, 3), 0)
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean

from sympy_extras._typing_sympy import QuantifierPrefix, as_expr
from sympy_extras.assumptions.ask_sympy import _sympy_symbol, core_boolean, core_prefix, sympy_boolean
from sympy_extras.assumptions.intervals_sympy import core_expr, core_symbol, sympy_expr
from sympy_extras.polys.virtual_substitution_sympy import simplify_logic_of_core

from . import integers as native
from .integers import (IntMatrix, IntVector, hermite_normal_form_with_transform, hilbert_basis,
    minimal_nonnegative_solutions)

__all__ = ['hermite_normal_form_with_transform', 'linear_diophantine_system',
    'hilbert_basis', 'minimal_nonnegative_solutions', 'cooper',
    'presburger_quantifier_elimination', 'is_presburger', 'simplify_logic_of_core', 'IntMatrix', 'IntVector']


# ---------------------------------------------------------------------------
# the face

def _linear_system(equations: Sequence[Expr], symbols: Sequence[Symbol]) -> Optional[tuple[IntMatrix, IntVector]]:
    """``A``, ``b`` with the equations ``A x = b`` (integer coefficients),
    :func:`sympy_extras.solvers.integers._linear_system` on SymPy
    expressions."""
    return native._linear_system([core_expr(e) for e in equations], [core_symbol(s) for s in symbols])


def linear_diophantine_system(equations: Sequence[Union[Expr, Eq]], symbols: Sequence[Symbol]
                              ) -> Optional[tuple[list[Expr], list[Symbol]]]:
    """The general integer solution of a system of linear equations with
    integer coefficients (:func:`sympy_extras.solvers.integers.linear_diophantine_system`
    on SymPy objects): the values of the symbols as affine expressions in
    free integer parameters, and the parameters; ``None`` when the system
    has no integer solution.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.solvers.integers_sympy import linear_diophantine_system
    >>> values, parameters = linear_diophantine_system([Eq(3*x + 5*y, 7)], [x, y])
    >>> values
    [14 - 5*t0, 3*t0 - 7]
    >>> linear_diophantine_system([Eq(2*x + 4*y, 3)], [x, y]) is None
    True
    >>> values, parameters = linear_diophantine_system([x + y + z - 6, x - y - 2], [x, y, z])
    >>> values
    [t0, t0 - 2, 8 - 2*t0]
    """
    exprs = [core_expr(as_expr(e.lhs - e.rhs) if isinstance(e, Eq) else as_expr(e)) for e in equations]
    found = native.linear_diophantine_system(exprs, [core_symbol(s) for s in symbols])
    if found is None:
        return None
    values, parameters = found
    return [sympy_expr(v) for v in values], [_sympy_symbol(p) for p in parameters]


def is_presburger(formula: Boolean, integers: set[Symbol]) -> bool:
    """Whether the formula is a Boolean combination of linear relations
    with integer coefficients and divisibility conditions
    ``Eq(Mod(e, k), 0)`` in the given integer variables
    (:func:`sympy_extras.solvers.integers.is_presburger`)."""
    return native.is_presburger(core_boolean(formula), {core_symbol(s) for s in integers})


def cooper(formula: Boolean, x: Symbol) -> Boolean:
    """`\\exists x \\in \\mathbb{Z}\\, \\varphi` for a formula of Presburger
    arithmetic (:func:`sympy_extras.solvers.integers.cooper` on SymPy
    objects), as a quantifier-free formula in the other integer variables
    (with divisibility conditions ``Eq(Mod(e, k), 0)``).

    Examples
    ========

    >>> from sympy import Eq, Mod
    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.integers_sympy import cooper
    >>> cooper(Eq(2*x, y), x)
    Eq(Mod(y, 2), 0)
    >>> cooper((x > y) & (x < y + 2), x)
    True
    >>> cooper((x > y) & (x < y + 1), x)
    False
    >>> cooper((3*x > y) & (3*x < y + 3), x)
    Eq(Mod(y + 1, 3), 0) | Eq(Mod(y + 2, 3), 0)
    """
    if not isinstance(formula, Boolean):
        raise TypeError("a Boolean formula is expected, got %s" % (formula,))
    return sympy_boolean(native.cooper(core_boolean(formula), core_symbol(x)))


def _drop_covered(formula: Boolean) -> Boolean:
    """:func:`sympy_extras.solvers.integers._drop_covered` on a SymPy
    formula."""
    return sympy_boolean(native._drop_covered(core_boolean(formula)))


def presburger_quantifier_elimination(formula: Boolean, prefix: QuantifierPrefix) -> Boolean:
    """Eliminate all the quantifiers of a formula of Presburger
    arithmetic (:func:`sympy_extras.solvers.integers.presburger_quantifier_elimination`
    on SymPy objects; innermost first, a universal quantifier through
    `\\neg \\exists \\neg`).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.integers_sympy import presburger_quantifier_elimination
    >>> presburger_quantifier_elimination((2*x > y) & (2*x < y + 2), [('exists', x)])
    Eq(Mod(y + 1, 2), 0)
    >>> presburger_quantifier_elimination(x < y, [('exists', y), ('forall', x)])
    False
    """
    return sympy_boolean(native.presburger_quantifier_elimination(core_boolean(formula), core_prefix(prefix)))

"""The SymPy face of the polynomial bounds of elementary functions
(:mod:`sympy_extras.assumptions.bounds`).

:func:`polynomial_abstraction` takes a SymPy formula and returns an
:class:`Abstraction` of SymPy objects, converted by
:mod:`sympy_extras.interop` around the native function.

Examples
========

>>> from sympy import exp, sqrt
>>> from sympy.abc import x
>>> from sympy_extras.assumptions.bounds_sympy import polynomial_abstraction
>>> a = polynomial_abstraction(exp(x) > 1 + x, {x})
>>> a.formula, len(a.constraints)
(_t > x + 1, 6)
>>> polynomial_abstraction(sqrt(x) > 1, {x}).constraints
[x >= 0, _t >= 0, Eq(_t**2, x)]
"""
from __future__ import annotations

from typing import Final, Optional

from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean

from sympy_extras._typing_sympy import as_boolean, as_symbol
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import bounds as native

__all__ = ['polynomial_abstraction', 'Abstraction']


class Abstraction:
    """A formula with its function applications replaced by variables
    (SymPy objects; see :class:`sympy_extras.assumptions.bounds.Abstraction`).

    Attributes
    ==========

    formula : Boolean
        The polynomial formula.
    constraints : list of Boolean
        The polynomial constraints satisfied by the true values of the
        new variables.
    variables : list of Symbol
        The new variables.
    requirements : list of Boolean
        The conditions under which the replaced applications are real.
    """

    __slots__ = ('formula', 'constraints', 'variables', 'requirements')

    def __init__(self, formula: Boolean, constraints: list[Boolean], variables: list[Symbol],
                 requirements: Optional[list[Boolean]] = None) -> None:
        self.formula: Final[Boolean] = formula
        self.constraints: Final[list[Boolean]] = list(constraints)
        self.variables: Final[list[Symbol]] = list(variables)
        self.requirements: Final[list[Boolean]] = list(requirements or [])

    def __repr__(self) -> str:
        return "Abstraction(%s, %s)" % (self.formula, self.constraints)


def _sympy(b: CoreBoolean) -> Boolean:
    return as_boolean(to_sympy(b))


def polynomial_abstraction(formula: Boolean, real: set[Symbol], max_variables: int = 3) -> Optional[Abstraction]:
    """Replace the elementary functions of a formula by variables with
    polynomial constraints
    (:func:`sympy_extras.assumptions.bounds.polynomial_abstraction` on
    SymPy objects)."""
    core_formula = from_sympy(formula)
    if not isinstance(core_formula, CoreBoolean):
        raise TypeError("expected a Boolean, got %s" % (formula,))
    core_real: set[CoreSymbol] = set()
    for s in real:
        converted = from_sympy(s)
        if not isinstance(converted, CoreSymbol):
            raise TypeError("%s is not a symbol" % (s,))
        core_real.add(converted)
    result = native.polynomial_abstraction(core_formula, core_real, max_variables)
    if result is None:
        return None
    return Abstraction(_sympy(result.formula), [_sympy(c) for c in result.constraints],
                       [as_symbol(to_sympy(v)) for v in result.variables],
                       [_sympy(r) for r in result.requirements])

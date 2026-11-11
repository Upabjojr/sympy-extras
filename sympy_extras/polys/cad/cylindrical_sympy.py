"""The SymPy face of the cylindrical descriptions of semialgebraic sets
(:mod:`sympy_extras.polys.cad.cylindrical`).

The functions take SymPy formulas and symbols and give SymPy formulas and
sets, converted around the native functions by
:mod:`sympy_extras.interop`: the root functions are
:class:`~sympy_extras.polys.cad.indexedroot_sympy.IndexedRoot` (a SymPy
``Function``), the real algebraic numbers which are not written in
radicals ``CRootOf``. Nothing of the descriptions stays on SymPy.

Examples
========

>>> from sympy import Eq
>>> from sympy.abc import a, x, y, z
>>> from sympy_extras.polys.cad.cylindrical_sympy import cylindrical_formula, cylindrical_cases
>>> cylindrical_formula(Eq(z**2, x) & (z > y), [x, y], [('exists', z)])
(x >= 0) & (y < sqrt(x))
>>> cylindrical_cases(Eq(x**2, a), [a], [x])
[(a >= 0, {-sqrt(a), sqrt(a)})]
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean
from sympy.sets.sets import Set

from sympy_extras._typing import QuantifierSpec, as_set

from sympy_extras.sets.sets import Set as CoreSet

from . import cylindrical as native
from .indexedroot_sympy import IndexedRoot
from .qe_sympy import _errors, core_formula, core_quantifiers, core_symbols, sympy_formula

__all__ = ['IndexedRoot', 'cylindrical_formula', 'cylindrical_set', 'cylindrical_cases']


def _sympy_set(s: CoreSet) -> Set:
    from sympy_extras.interop import to_sympy
    return as_set(to_sympy(s))


def cylindrical_formula(formula: Union[Basic, bool], gens: Sequence[Symbol],
                        quantifiers: QuantifierSpec = (), method: Optional[str] = None,
                        partial: bool = True) -> Boolean:
    """:func:`~sympy_extras.polys.cad.cylindrical.cylindrical_formula` on
    SymPy formulas.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.cylindrical_sympy import cylindrical_formula
    >>> cylindrical_formula(x**2 + y**2 < 1, [x, y])
    (x > -1) & (x < 1) & (y < sqrt(1 - x**2)) & (y > -sqrt(1 - x**2))
    >>> cylindrical_formula((x**2 + y**2 <= 1) & (y >= x), [x, y])
    ((y >= x) & (x <= sqrt(2)/2) & (x > -sqrt(2)/2) & (y <= sqrt(1 - x**2))) | ((x >= -1) & (x <= -sqrt(2)/2) & (y <= sqrt(1 - x**2)) & (y >= -sqrt(1 - x**2)))
    >>> cylindrical_formula((y**3 - 3*y + x > 0) & (x > 1), [x, y])
    (Eq(x, 2) & ((y > 1) | ((y > -2) & (y < 1)))) | ((x > 2) & (y > IndexedRoot(x + y**3 - 3*y, y, 0))) | ((x > 1) & (x < 2) & ((y > IndexedRoot(x + y**3 - 3*y, y, 2)) | ((y > IndexedRoot(x + y**3 - 3*y, y, 0)) & (y < IndexedRoot(x + y**3 - 3*y, y, 1)))))
    """
    with _errors():
        result = native.cylindrical_formula(core_formula(formula), core_symbols(list(gens)),
                                            core_quantifiers(quantifiers), method, partial)
    return sympy_formula(result)


def cylindrical_set(formula: Union[Basic, bool], gens: Sequence[Symbol], method: Optional[str] = None,
                    partial: bool = True) -> Set:
    """:func:`~sympy_extras.polys.cad.cylindrical.cylindrical_set` on
    SymPy formulas.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.cylindrical_sympy import cylindrical_set
    >>> cylindrical_set((x**2 + y**2 <= 1) & (x + y >= 1), [x, y])
    ConditionSet((x, y), (x >= 0) & (x <= 1) & (y >= 1 - x) & (y <= sqrt(1 - x**2)), ProductSet(Reals, Reals))
    >>> cylindrical_set((x**2 + y**2)*((x - 3)**2 + y**2 - 1) <= 0, [x, y])
    Union(ConditionSet((x, y), (x >= 2) & (x <= 4) & (y <= sqrt(-x**2 + 6*x - 8)) & (y >= -sqrt(-x**2 + 6*x - 8)), ProductSet(Reals, Reals)), {(0, 0)})
    >>> cylindrical_set(x**2 + y**2 < 0, [x, y])
    EmptySet
    """
    with _errors():
        result = native.cylindrical_set(core_formula(formula), core_symbols(list(gens)), method, partial)
    return _sympy_set(result)


def cylindrical_cases(formula: Union[Basic, bool], parameters: Sequence[Symbol], unknowns: Sequence[Symbol],
                      method: Optional[str] = None, partial: bool = True) -> list[tuple[Boolean, Set]]:
    """:func:`~sympy_extras.polys.cad.cylindrical.cylindrical_cases` on
    SymPy formulas.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import a, x
    >>> from sympy_extras.polys.cad.cylindrical_sympy import cylindrical_cases
    >>> cylindrical_cases(x**2 <= a, [a], [x])
    [(a >= 0, Interval(-sqrt(a), sqrt(a)))]
    >>> cylindrical_cases(Eq(a*x, 1), [a], [x])
    [((a > 0) | (a < 0), {1/a})]
    """
    with _errors():
        found = native.cylindrical_cases(core_formula(formula), core_symbols(list(parameters)),
                                         core_symbols(list(unknowns)), method, partial)
    return [(sympy_formula(condition), _sympy_set(solutions)) for condition, solutions in found]

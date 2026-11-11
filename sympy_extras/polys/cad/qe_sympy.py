"""The SymPy face of quantifier elimination by cylindrical algebraic
decomposition (:mod:`sympy_extras.polys.cad.qe`).

The functions take SymPy formulas and symbols, convert them to the core
(:mod:`sympy_extras.interop`), run the native functions and convert the
answers back: formulas and sets of SymPy, whose real algebraic numbers
are SymPy's ``CRootOf`` (the native
:class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot` with rational
coefficients), and cells of :mod:`sympy_extras.polys.cad.lifting_sympy`.
The errors of the native polynomials come out as SymPy's
``PolynomialError``. Nothing of the elimination stays on SymPy.

Examples
========

>>> from sympy import Eq
>>> from sympy.abc import a, b, c, x, y
>>> from sympy_extras.polys.cad.qe_sympy import quantifier_elimination, solution_set
>>> quantifier_elimination(Eq(a*x**2 + b*x + c, 0), [('exists', x)])
Eq(c, 0) | (4*a*c - b**2 < 0) | (Ne(b, 0) & (4*a*c - b**2 <= 0))
>>> [float(r) for r in solution_set(x**2 > 2, x).boundary]      # CRootOf(x**2 - 2, 0) and 1
[-1.4142135623730951, 1.4142135623730951]
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly
from sympy.sets.sets import Set

from sympy_extras._typing_sympy import QuantifierSpec, as_boolean, as_expr, as_set, as_symbol
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import Poly as CorePoly

from . import lifting as native_lifting
from . import qe as native
from .lifting_sympy import CADCell, NotWellOriented, _Faces
from .projection_sympy import core_poly, sympy_poly

__all__ = ['quantifier_elimination', 'decide', 'sample_points', 'solution_set', 'truth_tables', 'CellTruth',
           'core_formula', 'core_symbols', 'core_quantifiers', 'sympy_formula']

#: the truth value of a formula on a cell (of the face)
CellTruth = tuple[CADCell, bool]


@contextmanager
def _errors() -> Iterator[None]:
    """The errors of the native decompositions as SymPy's."""
    try:
        yield
    except native_lifting.NotWellOriented as error:
        raise NotWellOriented(str(error)) from error
    except BasePolynomialError as error:
        raise PolynomialError(str(error)) from error


def core_formula(formula: Union[Basic, bool]) -> CoreBoolean:
    """A SymPy formula (or a Python bool) as a Boolean of the core."""
    converted = from_sympy(as_boolean(sympify(formula)))
    if not isinstance(converted, CoreBoolean):
        # a symbol is a Boolean of SymPy, not a formula
        raise ValueError("unsupported formula %s" % (formula,))
    return converted


def sympy_formula(formula: CoreBasic) -> Boolean:
    """A Boolean of the core as a SymPy formula."""
    return as_boolean(to_sympy(formula))


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(as_symbol(s))
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def core_symbols(symbols: Sequence[Basic]) -> list[CoreSymbol]:
    """SymPy symbols as symbols of the core."""
    return [_core_symbol(s) for s in symbols]


def core_quantifiers(quantifiers: QuantifierSpec) -> native.QuantifierSpec:
    """A quantifier prefix of SymPy symbols with symbols of the core."""
    result: list[tuple[str, Union[CoreSymbol, Sequence[CoreSymbol]]]] = []
    for kind, variables in quantifiers:
        if isinstance(variables, Basic):
            result.append((kind, _core_symbol(variables)))
        else:
            result.append((kind, core_symbols(list(variables))))
    return result


def _sympy_expr(e: CoreExpr) -> Expr:
    return as_expr(to_sympy(e))


def _compile(formula: Union[Boolean, bool], gens: Sequence[Symbol], polys: list[Poly],
             index: dict[Poly, int]) -> native._Compiled:
    """:func:`~sympy_extras.polys.cad.qe._compile` on SymPy objects (for
    the region integrals): the polynomials of the atoms are SymPy's, in
    ``gens``, collected into ``polys`` and ``index`` as the native
    function collects them."""
    generators = core_symbols(list(gens))
    native_polys: list[CorePoly] = [core_poly(p, generators) for p in polys]
    native_index: dict[CorePoly, int] = {p: i for i, p in enumerate(native_polys)}
    with _errors():
        compiled = native._compile(core_formula(formula), generators, native_polys, native_index)
    for p in native_polys[len(polys):]:
        converted = sympy_poly(p, tuple(gens))
        index[converted] = len(polys)
        polys.append(converted)
    return compiled


def quantifier_elimination(formula: Union[Boolean, bool], quantifiers: QuantifierSpec = (),
                           free: Optional[Sequence[Symbol]] = None, method: Optional[str] = None,
                           partial: bool = True) -> Boolean:
    """:func:`~sympy_extras.polys.cad.qe.quantifier_elimination` on SymPy
    formulas: ``S.true``, ``S.false`` or a quantifier-free formula in the
    free variables, equivalent to the input over the reals.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import a, b, c, x, y, z
    >>> from sympy_extras.polys.cad.qe_sympy import quantifier_elimination as qe
    >>> qe(x**2 + b*x + c > 0, [('forall', x)])
    b**2 - 4*c < 0
    >>> qe(Eq(x**2 + y**2, 1), [('exists', y)])
    (x >= -1) & (x <= 1)
    >>> qe(x**2 + y**2 < 1, [('forall', x), ('exists', y)])
    False
    >>> qe(Eq(y, x**2), [('forall', x), ('exists', y)])
    True
    >>> qe(Eq(z**2, x) & (z > y), [('exists', z)], free=[x, y])
    ((x >= 0) & (y < 0)) | (x - y**2 > 0)
    """
    with _errors():
        result = native.quantifier_elimination(core_formula(formula), core_quantifiers(quantifiers),
                                               None if free is None else core_symbols(list(free)), method, partial)
    return sympy_formula(result)


def decide(formula: Union[Boolean, bool], quantifiers: QuantifierSpec, method: Optional[str] = None,
           partial: bool = True) -> bool:
    """:func:`~sympy_extras.polys.cad.qe.decide` on SymPy formulas.

    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.qe_sympy import decide
    >>> decide(Eq(x, 2*y), [('forall', x), ('exists', y)])
    True
    >>> decide(x**2 + y**2 < 0, [('exists', [x, y])])
    False
    """
    with _errors():
        return native.decide(core_formula(formula), core_quantifiers(quantifiers), method, partial)


def sample_points(formula: Union[Boolean, bool], gens: Sequence[Symbol],
                  method: Optional[str] = None, partial: bool = True) -> list[dict[Symbol, Expr]]:
    """:func:`~sympy_extras.polys.cad.qe.sample_points` on SymPy
    formulas: the coordinates are rational numbers and ``CRootOf``.

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.qe_sympy import sample_points
    >>> points = sample_points((x**2 + y**2 < 1) & (x > y), [x, y])
    >>> points[0], points[2]
    ({x: 0, y: -1/2}, {x: 3/4, y: 0})
    >>> type(points[1][x]).__name__, points[1][x].evalf(10), points[1][y]   # a root of 2*x**2 - 1
    ('ComplexRootOf', 0.7071067812, 0)
    >>> sample_points(x**2 + y**2 < 0, [x, y])
    []
    """
    variables = [as_symbol(g) for g in gens]
    generators = core_symbols(variables)
    with _errors():
        points = native.sample_points(core_formula(formula), generators, method, partial)
    return [{v: _sympy_expr(point[g]) for v, g in zip(variables, generators)} for point in points]


def solution_set(formula: Union[Boolean, bool], x: Symbol, quantifiers: QuantifierSpec = (),
                 method: Optional[str] = None, partial: bool = True) -> Set:
    """:func:`~sympy_extras.polys.cad.qe.solution_set` on SymPy formulas:
    a union of intervals and points whose ends are rational numbers and
    ``CRootOf``.

    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.qe_sympy import solution_set
    >>> from sympy import CRootOf, Interval      # SymPy prints a CRootOf in the variable it cached first
    >>> found = solution_set(Eq(3*x**2 + 2*x*y + y**2 - x + y - 7, 0), x, [('exists', y)])
    >>> found == Interval(CRootOf(8*x**2 - 8*x - 29, 0), CRootOf(8*x**2 - 8*x - 29, 1))
    True
    """
    with _errors():
        found = native.solution_set(core_formula(formula), _core_symbol(x), core_quantifiers(quantifiers), method,
                                    partial)
    return as_set(to_sympy(found))


def truth_tables(formulas: Sequence[Union[Boolean, bool]], gens: Sequence[Symbol],
                 method: Optional[str] = None, partial: bool = True) -> tuple[list[CADCell], list[list[bool]]]:
    """:func:`~sympy_extras.polys.cad.qe.truth_tables` on SymPy formulas,
    with the cells of the face (their sample points are SymPy numbers).

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.qe_sympy import truth_tables
    >>> cells, (a, b) = truth_tables([x > 0, x**3 > 0], [x])
    >>> [c.point for c in cells]
    [(-1,), (0,), (1,)]
    >>> a == b
    True
    >>> cells, (a, b) = truth_tables([x > 0, x*y > 0], [x, y])
    >>> [(c.point, s, t) for c, s, t in zip(cells, a, b)]
    [((-1, -1), False, True), ((-1, 0), False, False), ((-1, 1), False, False), ((0,), False, False), ((1, -1), True, False), ((1, 0), True, False), ((1, 1), True, True)]
    >>> [c.point for c in truth_tables([x > 0, x*y > 0], [x, y], partial=False)[0]]
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    """
    variables = [as_symbol(g) for g in gens]
    with _errors():
        cells, tables = native.truth_tables([core_formula(f) for f in formulas], core_symbols(variables), method,
                                            partial)
    faces = _Faces(variables)
    return [faces.cell(cell) for cell in cells], tables

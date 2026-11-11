"""The SymPy face of the table of definite integrals
(:mod:`sympy_extras.integrals.tables`), for the integrators which are
still written on SymPy.

:class:`TableEntry` is the entry of the native module on SymPy objects
(:meth:`TableEntry.from_core`), :data:`TABLE` the native table so
converted, and :data:`X`, :data:`A`, ... the variable and the Wild
parameters of the patterns as SymPy objects. :func:`lookup` and
:func:`table_integral` convert their arguments by
:mod:`sympy_extras.interop` and run the native code; nothing of the
module stays on SymPy.

Examples
========

>>> from sympy import symbols, sin, oo
>>> from sympy_extras.integrals.tables_sympy import table_integral
>>> x = symbols('x')
>>> a = symbols('a', positive=True)
>>> table_integral(sin(a*x)**3/x**3, x, 0, oo)
ConditionalValue(3*pi*a**2/8)
"""
from __future__ import annotations

from typing import Final, Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.logic.boolalg import Boolean

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions, core_boolean, sympy_boolean
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol, Wild as CoreWild
from sympy_extras.interop import from_sympy, to_sympy

from . import tables as native
from .conditions_sympy import ConditionalValue

__all__ = ['TableEntry', 'TABLE', 'lookup', 'table_integral', 'X', 'A', 'B', 'N', 'M', 'C', 'K', 'Sample']


def _sympy_wild(w: CoreWild) -> Wild:
    converted = to_sympy(w)
    if not isinstance(converted, Wild):
        raise TypeError("%s is not a Wild" % (w,))
    return converted


def _core_wild(w: Basic) -> CoreWild:
    converted = from_sympy(w)
    if not isinstance(converted, CoreWild):
        raise TypeError("%s is not a Wild" % (w,))
    return converted


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy_dummy(d: CoreBasic) -> Dummy:
    converted = to_sympy(d)
    if not isinstance(converted, Dummy):
        raise TypeError("%s is not a Dummy" % (d,))
    return converted


#: the integration variable of the patterns
X: Final[Dummy] = _sympy_dummy(native.X)
#: the parameters of the patterns (they never match the variable)
A: Final[Wild] = _sympy_wild(native.A)
B: Final[Wild] = _sympy_wild(native.B)
N: Final[Wild] = _sympy_wild(native.N)
M: Final[Wild] = _sympy_wild(native.M)
#: a coefficient which is not imaginary
C: Final[Wild] = _sympy_wild(native.C)
#: the parameter of a range ``(K, oo)``
K: Final[Wild] = _sympy_wild(native.K)

#: a sample of parameter values (for the numerical checks of the tests)
Sample = dict[Wild, Expr]


def _sympy_values(values: dict[CoreWild, CoreExpr]) -> dict[Wild, Expr]:
    return {_sympy_wild(key): sympy_expr(value) for key, value in values.items()}


def _core_values(values: Sample) -> native.Sample:
    return {_core_wild(key): core_expr(value) for key, value in values.items()}


class TableEntry:
    """:class:`sympy_extras.integrals.tables.TableEntry` on SymPy objects:
    one entry of the table (see the native class for the attributes)."""

    __slots__ = ('pattern', 'lower', 'upper', 'value', 'condition', 'reference', 'samples', 'oscillatory')

    def __init__(self, pattern: ExprLike, lower: ExprLike, upper: ExprLike, value: ExprLike,
                 condition: object, reference: str, samples: tuple[Sample, ...] = ({},),
                 oscillatory: bool = False) -> None:
        self.pattern: Final[Expr] = as_expr(pattern)
        self.lower: Final[Expr] = as_expr(lower)
        self.upper: Final[Expr] = as_expr(upper)
        self.value: Final[Expr] = as_expr(value)
        self.condition: Final[Boolean] = as_boolean(condition)
        self.reference: Final[str] = reference
        self.samples: Final[tuple[Sample, ...]] = samples
        self.oscillatory: Final[bool] = oscillatory

    @staticmethod
    def from_core(entry: native.TableEntry) -> TableEntry:
        """An entry of the native table on SymPy objects."""
        return TableEntry(sympy_expr(entry.pattern), sympy_expr(entry.lower), sympy_expr(entry.upper),
                          sympy_expr(entry.value), sympy_boolean(entry.condition), entry.reference,
                          tuple(_sympy_values(sample) for sample in entry.samples), entry.oscillatory)

    def to_core(self) -> native.TableEntry:
        """The same entry on the objects of the core."""
        return native.TableEntry(core_expr(self.pattern), core_expr(self.lower), core_expr(self.upper),
                                 core_expr(self.value), core_boolean(self.condition), self.reference,
                                 tuple(_core_values(sample) for sample in self.samples), self.oscillatory)

    def __repr__(self) -> str:
        return "TableEntry(%s, (%s, %s), %s)" % (self.pattern, self.lower, self.upper, self.reference)

    def integrand(self, x: Symbol, sample: Sample) -> Expr:
        """The integrand in ``x`` at the sample values."""
        return as_expr(self.pattern.xreplace(dict(sample)).xreplace({X: x}))

    def range(self, sample: Sample) -> tuple[Expr, Expr]:
        """The range at the sample values."""
        return as_expr(self.lower.xreplace(dict(sample))), as_expr(self.upper.xreplace(dict(sample)))


#: the table, the entries of :data:`sympy_extras.integrals.tables.TABLE` on SymPy objects
TABLE: Final[list[TableEntry]] = [TableEntry.from_core(entry) for entry in native.TABLE]

#: the face of each native entry
_FACES: Final[dict[int, TableEntry]] = {id(entry): face for entry, face in zip(native.TABLE, TABLE)}


def lookup(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
           assumptions: Assumptions = None) -> Optional[tuple[TableEntry, dict[Wild, Expr], Boolean]]:
    """:func:`sympy_extras.integrals.tables.lookup` on SymPy objects: the
    first entry of the table matching ``Integral(f, (x, a, b))`` whose
    condition is not refuted by the assumptions, with the values of its
    parameters and the part of the condition left undecided; ``None``
    when no entry matches.

    Examples
    ========

    >>> from sympy import symbols, sin, oo
    >>> from sympy_extras.integrals.tables_sympy import lookup
    >>> x, a = symbols('x a')
    >>> entry, match, condition = lookup(sin(a*x)**2/x**2, x, 0, oo)
    >>> entry.reference, condition
    ('GR 3.821.9', a > 0)
    """
    found = native.lookup(core_expr(as_expr(f)), _core_symbol(x), core_expr(as_expr(a)), core_expr(as_expr(b)),
                          core_assumptions(assumptions))
    if found is None:
        return None
    entry, values, condition = found
    face = _FACES.get(id(entry))
    return (face if face is not None else TableEntry.from_core(entry), _sympy_values(values),
            sympy_boolean(condition))


def table_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                   assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.tables.table_integral` on SymPy
    objects: ``Integral(f, (x, a, b))`` from the table, with the condition
    on the parameters the assumptions do not settle; ``None`` when no
    entry matches.

    Examples
    ========

    >>> from sympy import symbols, cos, pi, log
    >>> from sympy_extras.integrals.tables_sympy import table_integral
    >>> x, a = symbols('x a')
    >>> table_integral(log(1 - 2*a*cos(x) + a**2), x, 0, pi, a > 1)
    ConditionalValue(2*pi*log(a))
    """
    found = native.table_integral(core_expr(as_expr(f)), _core_symbol(x), core_expr(as_expr(a)),
                                  core_expr(as_expr(b)), core_assumptions(assumptions))
    return None if found is None else ConditionalValue.from_core(found)

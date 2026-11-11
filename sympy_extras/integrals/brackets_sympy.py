"""The SymPy face of Ramanujan's master theorem and the method of brackets
(:mod:`sympy_extras.integrals.brackets`), for the integrators which are
still written on SymPy (:mod:`.definite`, :mod:`.series`).

:class:`SeriesCoefficient` is the container of the native module on SymPy
objects, with ``to_core`` and ``from_core``; the functions convert their
arguments by :mod:`sympy_extras.interop`, run the native code and return
the transforms of :mod:`.mellin_sympy`. The steps of the native module
which stay on SymPy are here too (the functions ending in ``_of_core``):
SymPy's formal power series (:func:`fps_of_core`) and the strip of SymPy's
``mellin_transform`` (:func:`mellin_strip_of_core`).

Examples
========

>>> from sympy import symbols, exp, sin, atan
>>> from sympy_extras.integrals.brackets_sympy import ramanujan_master_theorem, mellin_transform_series
>>> x, s = symbols('x s')
>>> ramanujan_master_theorem(exp(-x), x, s)
MellinTransform(gamma(s), (0, oo))
>>> ramanujan_master_theorem(atan(x), x, s)
MellinTransform(-pi/(2*s*cos(pi*s/2)), (-1, 0))
>>> mellin_transform_series(exp(-x)*sin(x), x, s)
MellinTransform(2**(s + 1)*gamma(s/4 + 1/2)*gamma(s/2 + 1/2)/(8*gamma(1 - s/4)), (-1, oo))
"""
from __future__ import annotations

from typing import Final, Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Dummy, Symbol

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Dummy as CoreDummy, Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.settings import settings

from . import brackets as native
from . import mellin as native_mellin
from .mellin_sympy import MellinTransform

__all__ = ['SeriesCoefficient', 'taylor_coefficient', 'ramanujan_master_theorem', 'method_of_brackets',
           'mellin_transform_series', 'fps_of_core', 'mellin_strip_of_core']


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _core_dummy(d: Basic) -> CoreDummy:
    converted = from_sympy(d)
    if not isinstance(converted, CoreDummy):
        raise TypeError("%s is not a dummy" % (d,))
    return converted


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    converted = to_sympy(s)
    if not isinstance(converted, Symbol):
        raise TypeError("%s is not a symbol" % (converted,))
    return converted


def _sympy_dummy(d: CoreDummy) -> Dummy:
    converted = to_sympy(d)
    if not isinstance(converted, Dummy):
        raise TypeError("%s is not a dummy" % (converted,))
    return converted


class SeriesCoefficient:
    """:class:`sympy_extras.integrals.brackets.SeriesCoefficient` on SymPy
    objects: the power series of a function about `0` in the form
    `f(x) = x^r \\sum_{j \\ge 0} c(j)\\, (x^m)^j`.

    Attributes
    ==========

    coefficient : Expr
        `c(j)`, a closed form in the index.
    index : Dummy
        The index `j` (a nonnegative integer).
    period : int
        `m`, the step of the exponents (2 for an even or odd function).
    offset : Expr
        `r`, the exponent of the first term.
    """

    __slots__ = ('coefficient', 'index', 'period', 'offset')

    def __init__(self, coefficient: Expr, index: Dummy, period: int, offset: Expr) -> None:
        self.coefficient: Final[Expr] = coefficient
        self.index: Final[Dummy] = index
        self.period: Final[int] = period
        self.offset: Final[Expr] = offset

    def to_core(self) -> native.SeriesCoefficient:
        """The same series on the objects of the core."""
        return native.SeriesCoefficient(core_expr(self.coefficient), _core_dummy(self.index), self.period,
                                        core_expr(self.offset))

    @staticmethod
    def from_core(c: native.SeriesCoefficient) -> SeriesCoefficient:
        """A series of the core on SymPy objects."""
        return SeriesCoefficient(sympy_expr(c.coefficient), _sympy_dummy(c.index), c.period, sympy_expr(c.offset))

    def __repr__(self) -> str:
        return "SeriesCoefficient(%s, %s, period=%d, offset=%s)" % (self.coefficient, self.index,
                                                                    self.period, self.offset)

    def phi(self) -> Optional[Expr]:
        """`\\varphi(j) = (-1)^j j!\\, c(j)` of the master theorem
        (:meth:`sympy_extras.integrals.brackets.SeriesCoefficient.phi`);
        ``None`` when a factor `(-1)^j` remains."""
        value = self.to_core().phi()
        return None if value is None else sympy_expr(value)


def _transform(t: Optional[native_mellin.MellinTransform]) -> Optional[MellinTransform]:
    return None if t is None else MellinTransform.from_core(t)


def taylor_coefficient(f: Expr, x: Symbol) -> Optional[SeriesCoefficient]:
    """:func:`sympy_extras.integrals.brackets.taylor_coefficient` on SymPy
    objects: the power series of ``f`` about 0 as a
    :class:`SeriesCoefficient`, ``None`` when there is none or its
    coefficients have no single closed form.

    Examples
    ========

    >>> from sympy import symbols, exp, sin
    >>> from sympy_extras.integrals.brackets_sympy import taylor_coefficient
    >>> x = symbols('x')
    >>> c = taylor_coefficient(exp(-x), x)
    >>> c.coefficient.subs(c.index, 3), c.period, c.offset
    (-1/6, 1, 0)
    >>> c = taylor_coefficient(sin(x), x)
    >>> c.period, c.offset, c.coefficient.subs(c.index, 1)
    (2, 1, -1/6)
    """
    found = native.taylor_coefficient(core_expr(f), _core_symbol(x))
    return None if found is None else SeriesCoefficient.from_core(found)


def ramanujan_master_theorem(f: Expr, x: Symbol, s: Symbol,
                             assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """:func:`sympy_extras.integrals.brackets.ramanujan_master_theorem` on
    SymPy objects: the Mellin transform of ``f`` by Ramanujan's master
    theorem, with its strip; ``None`` when ``f`` has no usable power series
    or no decay at infinity.

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.brackets_sympy import ramanujan_master_theorem
    >>> x, s = symbols('x s')
    >>> ramanujan_master_theorem(1/(1 + x), x, s)
    MellinTransform(pi/sin(pi*s), (0, 1))
    >>> ramanujan_master_theorem(exp(-x**2), x, s)
    MellinTransform(gamma(s/2)/2, (0, oo))
    >>> ramanujan_master_theorem(exp(x), x, s) is None
    True
    """
    return _transform(native.ramanujan_master_theorem(core_expr(f), _core_symbol(x), _core_symbol(s),
                                                      core_assumptions(assumptions)))


def method_of_brackets(f: Expr, x: Symbol, s: Symbol, assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """:func:`sympy_extras.integrals.brackets.method_of_brackets` on SymPy
    objects: the Mellin transform of a product of two series-expandable
    factors by the method of brackets.

    Examples
    ========

    >>> from sympy import symbols, exp, sin
    >>> from sympy_extras.integrals.brackets_sympy import method_of_brackets
    >>> x, s = symbols('x s')
    >>> method_of_brackets(exp(-x)*sin(x), x, s)
    MellinTransform(2**(s + 1)*gamma(s/4 + 1/2)*gamma(s/2 + 1/2)/(8*gamma(1 - s/4)), (-1, oo))
    """
    return _transform(native.method_of_brackets(core_expr(f), _core_symbol(x), _core_symbol(s),
                                                core_assumptions(assumptions)))


def mellin_transform_series(f: Expr, x: Symbol, s: Symbol, assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """:func:`sympy_extras.integrals.brackets.mellin_transform_series` on
    SymPy objects: the Mellin transform of ``f`` from its power series
    (the master theorem for one factor, the method of brackets for two).

    Examples
    ========

    >>> from sympy import symbols, exp, cos, sqrt
    >>> from sympy_extras.integrals.brackets_sympy import mellin_transform_series
    >>> x, s = symbols('x s')
    >>> mellin_transform_series(x**2*exp(-3*x), x, s)
    MellinTransform(3**(-s - 2)*gamma(s + 2), (-2, oo))
    >>> mellin_transform_series(cos(sqrt(x)), x, s)
    MellinTransform(2**(2*s)*sqrt(pi)*gamma(s)/gamma(1/2 - s), (0, 1/2))
    """
    return _transform(native.mellin_transform_series(core_expr(f), _core_symbol(x), _core_symbol(s),
                                                     core_assumptions(assumptions)))


# ---------------------------------------------------------------------------
# the steps of the native module which stay on SymPy

def fps_of_core(f: CoreExpr, x: CoreSymbol) -> Optional[native.FormalSeriesTerms]:
    """SymPy's formal power series of ``f`` about 0 (``sympy.fps``) under
    the time limit of the settings, read as the formula of the
    coefficients, its index, its start and the independent terms; ``None``
    when there is none or ``fps`` fails.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.brackets_sympy import fps_of_core
    >>> x = symbols('x')
    >>> terms = fps_of_core(exp(-x), x)
    >>> terms.formula.subs(terms.index, 2), terms.start, terms.independent
    (1/2, 1, 1)
    """
    from sympy.series.formal import FormalPowerSeries, fps
    g, v = sympy_expr(f), _sympy_symbol(x)
    try:
        series = attempt(lambda: fps(g, v), settings.timeout)
    except (KeyError, IndexError, ValueError, TypeError):
        # fps fails inside its hypergeometric algorithm on some inputs
        # (a KeyError popping the symbol of a constant)
        return None
    if not isinstance(series, FormalPowerSeries):
        return None
    k = series.ak.variables[0]
    if not isinstance(k, Symbol):
        return None
    try:
        return native.FormalSeriesTerms(core_expr(as_expr(series.ak.formula)), _core_symbol(k),
                                        core_expr(as_expr(series.ak.start)), core_expr(as_expr(series.ind)))
    except TypeError:
        return None


def mellin_strip_of_core(f: CoreExpr, x: CoreSymbol, s: CoreSymbol) -> Optional[tuple[CoreExpr, CoreExpr]]:
    """The strip of convergence of SymPy's ``mellin_transform(f, x, s)``
    under the time limit of the settings, ``None`` when it finds no
    transform.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.brackets_sympy import mellin_strip_of_core
    >>> x, s = symbols('x s')
    >>> mellin_strip_of_core(exp(-x), x, s)
    (0, oo)
    """
    from sympy.integrals.transforms import mellin_transform
    g, v, w = sympy_expr(f), _sympy_symbol(x), _sympy_symbol(s)
    known = attempt(lambda: mellin_transform(g, v, w), settings.timeout)
    if isinstance(known, tuple) and len(known) == 3:
        strip = known[1]
        return (core_expr(as_expr(strip[0])), core_expr(as_expr(strip[1])))
    return None

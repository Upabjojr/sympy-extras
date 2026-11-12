"""The SymPy face of the definite integrals by series expansion and
termwise integration (:mod:`sympy_extras.integrals.series`), for the
integrators which are still written on SymPy (:mod:`.definite`).

:class:`Expansion` is the container of the native module on SymPy
objects, with ``to_core`` and ``from_core``; the functions convert their
arguments by :mod:`sympy_extras.interop`, run the native code and convert
the results back (the values of :mod:`.conditions_sympy`). The moments
and the mean values of the native module reach :func:`~sympy_extras.integrals.definite.definite_integral`
through :func:`~sympy_extras.integrals.laplace_sympy.definite_integral_of_core`.

Examples
========

>>> from sympy import symbols, log, exp, atan, oo, sin, pi
>>> from sympy_extras.integrals.series_sympy import series_integral
>>> x = symbols('x')
>>> series_integral(log(1 - x)/x, x, 0, 1)
ConditionalValue(-pi**2/6)
>>> series_integral(x/(exp(x) - 1), x, 0, oo)
ConditionalValue(pi**2/6)
>>> series_integral(atan(x)/x, x, 0, 1)
ConditionalValue(Catalan)
>>> series_integral(log(sin(x)), x, 0, pi)
ConditionalValue(-pi*log(2))
>>> series_integral(log(sin(x))**2, x, 0, pi).value.expand()
pi*log(2)**2 + pi**3/12
"""
from __future__ import annotations

from typing import Final, Iterator, Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions import ask as native_ask
from sympy_extras.assumptions.ask_sympy import (Assumptions, _core_symbol, core_assumptions,
                                                core_boolean, sympy_boolean)
from sympy_extras.concrete.eulersums import polygamma_series
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean, to_boolean
from sympy_extras.core.symbol import Dummy as CoreDummy
from sympy_extras.interop import from_sympy, to_sympy

from . import conditions as native_conditions
from . import series as native
from .conditions_sympy import ConditionalValue

__all__ = ['Expansion', 'expansions', 'fourier_expansion', 'moments', 'sum_series', 'fourier_integral',
           'series_integral']


def _core_dummy(d: Basic) -> CoreDummy:
    converted = from_sympy(d)
    if not isinstance(converted, CoreDummy):
        raise TypeError("%s is not a dummy" % (d,))
    return converted


def _sympy_dummy(d: CoreDummy) -> Dummy:
    converted = to_sympy(d)
    if not isinstance(converted, Dummy):
        raise TypeError("%s is not a dummy" % (converted,))
    return converted


class Expansion:
    """:class:`sympy_extras.integrals.series.Expansion` on SymPy objects: a
    factor of the integrand expanded as a series, ``constant +
    Sum(coefficient(j) * basis(j), (j, start, oo))``.

    Attributes
    ==========

    factor : Expr
        The factor expanded.
    coefficient : Expr
        The coefficient as a function of the index.
    index : Dummy
        The index (a nonnegative integer).
    basis : Expr
        The function of ``x`` and the index the coefficient multiplies.
    start : int
        The first index of the series.
    constant : Expr
        The term of the series outside the sum (the mean of a Fourier
        series); zero by default.
    condition : Boolean
        The condition on the parameters under which the expansion holds;
        true by default.
    """

    __slots__ = ('factor', 'coefficient', 'index', 'basis', 'start', 'constant', 'condition')

    def __init__(self, factor: Expr, coefficient: Expr, index: Dummy, basis: Expr, start: int,
                 constant: ExprLike = 0, condition: Boolean | bool = True) -> None:
        self.factor: Final[Expr] = factor
        self.coefficient: Final[Expr] = coefficient
        self.index: Final[Dummy] = index
        self.basis: Final[Expr] = basis
        self.start: Final[int] = start
        self.constant: Final[Expr] = sympy_expr(core_expr(sympify(constant)))
        self.condition: Final[Boolean] = sympy_boolean(core_boolean(condition))

    def to_core(self) -> native.Expansion:
        """The same expansion on the objects of the core."""
        return native.Expansion(core_expr(self.factor), core_expr(self.coefficient), _core_dummy(self.index),
                                core_expr(self.basis), self.start, core_expr(self.constant),
                                core_boolean(self.condition))

    @staticmethod
    def from_core(e: native.Expansion) -> Expansion:
        """An expansion of the core on SymPy objects."""
        return Expansion(sympy_expr(e.factor), sympy_expr(e.coefficient), _sympy_dummy(e.index),
                         sympy_expr(e.basis), e.start, sympy_expr(e.constant), sympy_boolean(e.condition))

    def __repr__(self) -> str:
        if self.constant != 0:
            return "Expansion(%s, %s, %s, start=%d, constant=%s)" % (
                self.factor, self.coefficient, self.basis, self.start, self.constant)
        return "Expansion(%s, %s, %s, start=%d)" % (self.factor, self.coefficient, self.basis, self.start)


def _expansion(e: Optional[native.Expansion]) -> Optional[Expansion]:
    return None if e is None else Expansion.from_core(e)


def _conditional(found: Optional[native_conditions.ConditionalValue]) -> Optional[ConditionalValue]:
    return None if found is None else ConditionalValue.from_core(found)


def _optional_core(e: Optional[Expr]) -> Optional[CoreExpr]:
    return None if e is None else core_expr(e)


def expansions(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike) -> Iterator[tuple[Expansion, Expr]]:
    """:func:`sympy_extras.integrals.series.expansions` on SymPy objects:
    the ways of writing ``f`` as an expanded factor times a rest, each as
    ``(expansion, rest)``, produced lazily.

    Examples
    ========

    >>> from sympy import symbols, log
    >>> from sympy_extras.integrals.series_sympy import expansions
    >>> x = symbols('x')
    >>> [(e.factor, rest) for e, rest in expansions(log(1 - x)/x, x, 0, 1)]
    [(log(1 - x), 1/x), (log(1 - x)/x, 1)]
    """
    for e, rest in native.expansions(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                     core_expr(sympify(b))):
        yield (Expansion.from_core(e), sympy_expr(rest))


def fourier_expansion(g: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                      assumptions: Assumptions = None, compute: bool = False) -> Optional[Expansion]:
    """:func:`sympy_extras.integrals.series.fourier_expansion` on SymPy
    objects: the Fourier series of ``g`` valid on ``(a, b)`` from the table
    of the classical series (and the parametric entries, and with
    ``compute=True`` the computed coefficients) as an :class:`Expansion`;
    ``None`` when there is none.

    Examples
    ========

    >>> from sympy import symbols, log, sin, pi, Integer
    >>> from sympy_extras.integrals.series_sympy import fourier_expansion
    >>> x = symbols('x')
    >>> e = fourier_expansion(log(sin(x)), x, 0, pi)
    >>> e.coefficient.subs(e.index, 2), e.basis.subs(e.index, 2), e.constant
    (-1/2, cos(4*x), -log(2))
    >>> fourier_expansion(log(sin(x)), x, 0, 2*pi) is None
    True
    >>> e = fourier_expansion(x**2, x, 0, pi, compute=True)
    >>> e.coefficient.subs(e.index, 3), e.constant
    (-4/9, pi**2/3)
    """
    return _expansion(native.fourier_expansion(core_expr(sympify(g)), _core_symbol(x), core_expr(sympify(a)),
                                               core_expr(sympify(b)), core_assumptions(assumptions), compute))


def _computed_expansion(g: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[Expansion]:
    """The Fourier series of ``g`` on ``(a, b)`` with computed coefficients
    (the private step of :func:`fourier_expansion` with ``compute=True``,
    for the tests)."""
    return _expansion(native._computed_expansion(core_expr(g), _core_symbol(x), core_expr(a), core_expr(b),
                                                 core_assumptions(assumptions)))


def moments(expansion: Expansion, rest: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
            assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.series.moments` on SymPy objects:
    ``Integral(basis(j) * rest, (x, a, b))`` as a closed form in the index
    ``j``, or ``None``.

    >>> from sympy import symbols, log, S
    >>> from sympy_extras.integrals.series_sympy import expansions, moments
    >>> x = symbols('x')
    >>> expansion, rest = next(expansions(log(1 - x)/x, x, 0, 1))
    >>> moments(expansion, rest, x, 0, 1).subs(expansion.index, 3)
    1/3
    """
    found = native.moments(expansion.to_core(), core_expr(sympify(rest)), _core_symbol(x), core_expr(sympify(a)),
                           core_expr(sympify(b)), core_assumptions(assumptions))
    return None if found is None else sympy_expr(found)


def sum_series(term: ExprLike, j: Dummy, start: int, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.series.sum_series` on SymPy objects:
    ``Sum(term, (j, start, oo))`` in closed form, ``None`` when no
    summation algorithm closes it.

    Examples
    ========

    >>> from sympy import Dummy
    >>> from sympy_extras.integrals.series_sympy import sum_series
    >>> j = Dummy('j', integer=True, nonnegative=True)
    >>> sum_series((-1)**j/(2*j + 1)**2, j, 0)
    Catalan
    """
    found = native.sum_series(core_expr(sympify(term)), _core_dummy(j), start, core_assumptions(assumptions))
    return None if found is None else sympy_expr(found)


def fourier_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, assumptions: Assumptions = None,
                     expand: Optional[Expr] = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.series.fourier_integral` on SymPy
    objects: ``Integral(f, (x, a, b))`` through the Fourier series of the
    factors of ``f``; ``None`` when no factor has a series on ``(a, b)``
    or no step closes.

    Examples
    ========

    >>> from sympy import symbols, log, sin, cos, pi
    >>> from sympy_extras.integrals.series_sympy import fourier_integral
    >>> x = symbols('x')
    >>> n = symbols('n', integer=True, positive=True)
    >>> a = symbols('a', positive=True)
    >>> fourier_integral(log(sin(x)), x, 0, pi)
    ConditionalValue(-pi*log(2))
    >>> fourier_integral(log(sin(x))*log(cos(x)), x, 0, pi/2).value.expand()
    -pi**3/48 + pi*log(2)**2/2
    >>> fourier_integral(log(2*sin(x/2))*cos(n*x), x, 0, 2*pi)
    ConditionalValue(-pi/n)
    >>> fourier_integral(x**2*log(sin(x)), x, 0, pi)
    ConditionalValue(-pi**3*log(2)/3 - pi*zeta(3)/2)
    >>> fourier_integral(cos(n*x)/(1 - 2*a*cos(x) + a**2), x, 0, pi, a < 1)
    ConditionalValue(-pi*a**n/(a**2 - 1))
    """
    return _conditional(native.fourier_integral(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                                core_expr(sympify(b)), core_assumptions(assumptions),
                                                _optional_core(expand)))


def series_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, assumptions: Assumptions = None,
                    expand: Optional[Expr] = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.series.series_integral` on SymPy
    objects: ``Integral(f, (x, a, b))`` by series expansion of a factor
    and termwise integration; ``None`` when no factor expands, no moment
    or sum closes, or the interchange of sum and integral is neither
    justified nor confirmed numerically.

    Examples
    ========

    >>> from sympy import symbols, log, exp, oo, sin, pi
    >>> from sympy_extras.integrals.series_sympy import series_integral
    >>> x = symbols('x')
    >>> p = symbols('p', positive=True)
    >>> series_integral(log(1 + x)/x, x, 0, 1)
    ConditionalValue(pi**2/12)
    >>> series_integral(x**p*log(x)/(1 - x), x, 0, 1)
    ConditionalValue(-polygamma(1, p + 1))
    >>> series_integral(exp(-x)*log(1 + x), x, 0, oo) is None
    True
    >>> series_integral(x*log(sin(x)), x, 0, pi)
    ConditionalValue(-pi**2*log(2)/2)
    """
    return _conditional(native.series_integral(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                               core_expr(sympify(b)), core_assumptions(assumptions),
                                               _optional_core(expand)))


def _polygamma_series_sympy(term: Expr, n: Symbol, lower: int) -> Optional[Expr]:
    """:func:`~sympy_extras.concrete.eulersums.polygamma_series` (on the
    native core) for SymPy expressions (used by the fuzzer of the
    benchmarks)."""
    found = polygamma_series(core_expr(term), _core_symbol(n), lower)
    return None if found is None else sympy_expr(found)


# ---------------------------------------------------------------------------
# the step of the native module which stays on SymPy

def _sympy_assumptions(assumptions: native_ask.Assumptions) -> Assumptions:
    if assumptions is None:
        return None
    if isinstance(assumptions, bool):
        return assumptions
    if isinstance(assumptions, CoreBoolean):
        return sympy_boolean(assumptions)
    items: list[Boolean] = [sympy_boolean(to_boolean(a)) for a in assumptions]
    return items

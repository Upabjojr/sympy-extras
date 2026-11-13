"""Fourier series on the typed core (SymPy's ``sympy.series.fourier``):
:func:`fourier_series` computes the trigonometric series of a function on
an interval ``(a, b)``,

.. math::
    \\frac{a_0}{2} + \\sum_{n=1}^\\infty \\left(a_n \\cos\\frac{2 n \\pi x}{L}
    + b_n \\sin\\frac{2 n \\pi x}{L}\\right), \\quad L = b - a,

as a :class:`FourierSeries` (the coefficients as formulas in the index)
or, for a trigonometric polynomial of the period, as a
:class:`FiniteFourierSeries` (its coefficients read off).

The coefficients ``a_n = 2/L int_a^b f(x) cos(2 n pi x/L) dx`` and ``b_n``
are definite integrals with the index ``n`` a positive integer
(:func:`~sympy_extras.integrals.definite.definite_integral`), ``a_0``
the integral of ``f`` itself; for a centred interval an even ``f`` has
no sines and an odd ``f`` no cosines. A trigonometric polynomial is
recognised (:func:`finite_check`) after ``fu``'s ``TR1``, ``TR2`` and
``sincos_to_sum`` (:mod:`sympy_extras.simplify.fu`) have written it as a
sum of sines and cosines of multiples of ``pi x/L``.

This module is a typed port of SymPy's ``sympy/series/fourier.py``.
Differences: the coefficient formulas hold for the positive integer index
(SymPy integrates with an index without assumptions, a ``Piecewise`` of
the value at ``n = 0``); :class:`FiniteFourierSeries` is not a subclass
of :class:`FourierSeries` (its coefficients are dictionaries, not
sequences) and the class does not convert an expression
(:func:`finite_fourier_series` does); the operations of a finite series
keep all its terms (SymPy's ``shiftx``, ``scale`` and ``scalex`` kept the
first three only), and the sum of two finite series is the finite series
of the sum.

Examples
========

>>> from sympy_extras.core import Symbol, pi
>>> from sympy_extras.functions import sin
>>> from sympy_extras.series.fourier import fourier_series
>>> x = Symbol('x')
>>> s = fourier_series(x**2, (x, -pi, pi))
>>> s.truncate()
-4*cos(x) + cos(2*x) + pi**2/3
>>> s.shift(1).truncate(), s.scalex(2).truncate()
(-4*cos(x) + cos(2*x) + 1 + pi**2/3, -4*cos(2*x) + cos(4*x) + pi**2/3)
>>> fourier_series(sin(x)**3).truncate()
3*sin(x)/4 - sin(3*x)/4

References
==========

.. [MW] https://mathworld.wolfram.com/FourierSeries.html
.. [Sigma] https://en.wikipedia.org/wiki/Sigma_approximation
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/series/fourier.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; the coefficients integrated for a positive integer index;
# FiniteFourierSeries split from FourierSeries and from its conversion (finite_fourier_series).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Iterator, Optional, Sequence, Union, overload

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Dict, Tuple
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.numbers import Infinity, Integer, NegativeInfinity, Number, oo, pi
from sympy_extras.core.symbol import Dummy, Symbol, Wild
from sympy_extras.functions.elementary.complexes import abs_
from sympy_extras.functions.elementary.trigonometric import Cos, Sin, cos, sin, sinc
from sympy_extras.series.sequences import SeqFormula
from sympy_extras.series.series_class import SeriesBase

__all__ = ['fourier_series', 'FourierSeries', 'FiniteFourierSeries', 'finite_fourier_series', 'finite_check',
           'fourier_cos_seq', 'fourier_sin_seq']

#: the limits of a Fourier series: ``(x, start, stop)`` or ``(start, stop)``
FourierLimits = Sequence[ExprLike]

_ZERO: Final[Integer] = Integer(0)


def _is_zero(e: Expr) -> bool:
    return isinstance(e, Number) and e == 0


def _as_seq(s: Basic) -> SeqFormula:
    if not isinstance(s, SeqFormula):
        raise TypeError("a sequence of a formula is expected, got %s" % (s,))
    return s


def _index() -> Dummy:
    return Dummy('n', integer=True, positive=True)


def _integral(f: Expr, limits: Tuple, n: Optional[Dummy]) -> Expr:
    """``Integral(f, limits)``, with the index ``n`` a positive integer."""
    from sympy_extras.integrals.definite import definite_integral
    x, a, b = limits.args
    if not isinstance(x, Symbol):
        raise TypeError("the variable of the limits is a symbol")
    if n is None:
        return definite_integral(f, (x, as_expr(a), as_expr(b)))
    from sympy_extras.assumptions.facts import element
    from sympy_extras.sets import Naturals
    return definite_integral(f, (x, as_expr(a), as_expr(b)), [element(n, Naturals)])


def _limits_of(limits: Tuple) -> tuple[Symbol, Expr, Expr]:
    x, a, b = limits.args
    if not isinstance(x, Symbol):
        raise TypeError("the variable of the limits is a symbol")
    return x, as_expr(a), as_expr(b)


# PORTED FROM SYMPY: sympy/series/fourier.py:fourier_cos_seq
def fourier_cos_seq(func: Expr, limits: Tuple, n: Dummy) -> tuple[Expr, SeqFormula]:
    """The constant term and the sequence of the cosine terms of the
    Fourier series of ``func`` (the index ``n`` a positive integer)."""
    x, a, b = _limits_of(limits)
    L = b - a
    cos_term = cos(2 * n * pi * x / L)
    a0 = _integral(func, limits, None) / L
    return a0, SeqFormula(2 * cos_term * _integral(func * cos_term, limits, n) / L, (n, 1, oo))


# PORTED FROM SYMPY: sympy/series/fourier.py:fourier_sin_seq
def fourier_sin_seq(func: Expr, limits: Tuple, n: Dummy) -> SeqFormula:
    """The sequence of the sine terms of the Fourier series of ``func``."""
    x, a, b = _limits_of(limits)
    L = b - a
    sin_term = sin(2 * n * pi * x / L)
    return SeqFormula(2 * sin_term * _integral(func * sin_term, limits, n) / L, (n, 1, oo))


# PORTED FROM SYMPY: sympy/series/fourier.py:_process_limits
def _process_limits(func: Expr, limits: Optional[Union[FourierLimits, Tuple]]) -> Tuple:
    """The limits ``(x, start, stop)``: ``x`` from ``func`` when not given,
    ``(-pi, pi)`` by default; both ends bounded."""
    def find_x() -> Symbol:
        free = func.free_symbols
        if len(free) == 1:
            return next(iter(free))
        if not free:
            return Dummy('k')
        raise ValueError(" specify dummy variables for %s. If the function contains more than one free symbol, "
                         "a dummy variable should be supplied explicitly e.g. FourierSeries(m*n**2, (n, -pi, pi))"
                         % func)

    given: Optional[Sequence[ExprLike]]
    if isinstance(limits, Tuple):
        given = [as_expr(a) for a in limits.args]
    else:
        given = limits
    x: Optional[ExprLike] = None
    start: Optional[ExprLike] = None
    stop: Optional[ExprLike] = None
    if given is None:
        x, start, stop = find_x(), -pi, pi
    elif len(given) == 3:
        x, start, stop = given
    elif len(given) == 2:
        x = find_x()
        start, stop = given
    if not isinstance(x, Symbol) or start is None or stop is None:
        raise ValueError('Invalid limits given: %s' % (limits,))
    s, e = to_expr(start), to_expr(stop)
    if isinstance(s, (Infinity, NegativeInfinity)) or isinstance(e, (Infinity, NegativeInfinity)):
        raise ValueError("Both the start and end value should be bounded")
    return Tuple(x, s, e)


def _harmonic_wilds(x: Symbol) -> tuple[Wild, Wild]:
    a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k != 0])
    b = Wild('b', properties=[lambda k: x not in k.free_symbols])
    return a, b


# PORTED FROM SYMPY: sympy/series/fourier.py:finite_check
def finite_check(f: Expr, x: Symbol, L: Expr) -> tuple[bool, Expr]:
    """Whether ``f`` is a trigonometric polynomial of the period ``2 L``,
    with its form as a sum of sines and cosines of multiples of
    ``pi x/L`` (``f`` itself when it is not).

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.functions import sin, cos
    >>> from sympy_extras.series.fourier import finite_check
    >>> x = Symbol('x')
    >>> finite_check(sin(x)*cos(x), x, pi)
    (True, sin(2*x)/2)
    >>> finite_check(x*sin(x), x, pi)
    (False, x*sin(x))
    """
    from sympy_extras.simplify.fu import TR1, TR2, sincos_to_sum
    a, b = _harmonic_wilds(x)
    pattern = a * (pi / L) * x + b

    def check_sincos(t: Expr) -> bool:
        if isinstance(t, (Sin, Cos)):
            return as_expr(t.args[0]).match(pattern) is not None
        return False

    expr = sincos_to_sum(TR2(TR1(f)))
    for s in expr.as_coeff_add()[1]:
        for t in s.as_coeff_mul()[1]:
            if not (x not in t.free_symbols or check_sincos(t)):
                return False, f
    return True, expr


class _FourierBase(SeriesBase):
    """What the infinite and the finite Fourier series share."""

    __slots__ = ('_function', '_limits')

    def __init__(self, function: Expr, limits: Tuple) -> None:
        self._function: Final[Expr] = function
        _limits_of(limits)
        self._limits: Final[Tuple] = limits

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def limits(self) -> Tuple:
        """``(x, start, stop)``."""
        return self._limits

    @property
    def x(self) -> Symbol:
        return _limits_of(self._limits)[0]

    @property
    def period(self) -> tuple[Expr, Expr]:
        _, a, b = _limits_of(self._limits)
        return (a, b)

    @property
    def L(self) -> Expr:
        """Half the period."""
        a, b = self.period
        return abs_(b - a) / 2

    @property
    def a0(self) -> Expr:
        """The constant term."""
        raise NotImplementedError

    @property
    def start(self) -> Expr:
        return _ZERO

    def _check_free(self, s: ExprLike) -> Expr:
        e = to_expr(s)
        if self.x in e.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (e, self.x))
        return e

    def _subs(self, old: Basic, new: Basic) -> Basic:
        if self == old:
            return new
        if old.has(self.x):
            return self
        return SeriesBase._subs(self, old, new)

    @overload
    def truncate(self, n: None) -> Iterator[Expr]: ...

    @overload
    def truncate(self, n: Union[int, Expr] = 3) -> Expr: ...

    # PORTED FROM SYMPY: sympy/series/fourier.py:FourierSeries.truncate
    def truncate(self, n: Union[int, Expr, None] = 3) -> Union[Expr, Iterator[Expr]]:
        """The sum of the first ``n`` nonzero terms (an iterator over the
        terms for ``n=None``, all the terms of a finite series for ``oo``).

        >>> from sympy_extras.core import Symbol, pi
        >>> from sympy_extras.series.fourier import fourier_series
        >>> x = Symbol('x')
        >>> fourier_series(x, (x, -pi, pi)).truncate(4)
        2*sin(x) - sin(2*x) + 2*sin(3*x)/3 - sin(4*x)/2
        """
        if n is None:
            return iter(self)
        limit: Optional[int] = None if isinstance(n, Infinity) else int(n) if isinstance(n, (int, Integer)) else None
        if limit is None and not isinstance(n, Infinity):
            raise ValueError("the number of terms is an integer or oo")
        terms: list[Expr] = []
        for t in self:
            if limit is not None and len(terms) == limit:
                break
            if not _is_zero(t):
                terms.append(t)
        return add(*terms)

    # PORTED FROM SYMPY: sympy/series/fourier.py:FourierSeries.sigma_approximation
    def sigma_approximation(self, n: int = 3) -> Expr:
        """The sigma-approximation of order ``n`` (the terms of degree below
        ``n`` multiplied by Lanczos' factors ``sinc(pi k/n)``), which
        removes the Gibbs phenomenon [Sigma]_.

        >>> from sympy_extras.core import Symbol, pi
        >>> from sympy_extras.series.fourier import fourier_series
        >>> x = Symbol('x')
        >>> fourier_series(x, (x, -pi, pi)).sigma_approximation(4)
        2*sin(x)*sinc(pi/4) - 2*sin(2*x)/pi + 2*sin(3*x)*sinc(3*pi/4)/3
        """
        return add(*[sinc(pi * i / n) * t for i, t in enumerate(self[:n]) if not _is_zero(t)])

    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        for t in self:
            if not _is_zero(t):
                return t
        return _ZERO

    def shift(self, s: ExprLike) -> Expr:
        raise NotImplementedError

    def scale(self, s: ExprLike) -> Expr:
        raise NotImplementedError

    def __neg__(self) -> Expr:
        return self.scale(-1)

    def __sub__(self, other: ExprLike) -> Expr:
        return self.__add__(-to_expr(other))


# PORTED FROM SYMPY: sympy/series/fourier.py:FourierSeries
class FourierSeries(_FourierBase):
    """The Fourier sine and cosine series of a function: ``(function,
    (x, start, stop), (a0, an, bn))`` with ``an`` and ``bn`` the sequences
    of the cosine and sine terms. No computation is done:
    :func:`fourier_series` computes the series."""

    __slots__ = ('_a0', '_an', '_bn')

    def __init__(self, function: ExprLike, limits: Union[Tuple, Sequence[ExprLike]],
                 coefficients: Union[Tuple, tuple[ExprLike, SeqFormula, SeqFormula]]) -> None:
        lims = limits if isinstance(limits, Tuple) else Tuple(*limits)
        parts: tuple[Basic, ...] = (coefficients.args if isinstance(coefficients, Tuple)
                                    else (to_expr(coefficients[0]), coefficients[1], coefficients[2]))
        if len(parts) != 3:
            raise TypeError("the coefficients of a Fourier series are (a0, an, bn)")
        _FourierBase.__init__(self, to_expr(function), lims)
        self._a0: Final[Expr] = as_expr(parts[0])
        self._an: Final[SeqFormula] = _as_seq(parts[1])
        self._bn: Final[SeqFormula] = _as_seq(parts[2])
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._function, self._limits, Tuple(self._a0, self._an, self._bn))

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, limits, coefficients = args
        if not (isinstance(limits, Tuple) and isinstance(coefficients, Tuple)):
            raise TypeError("invalid arguments of a FourierSeries: %s" % (args,))
        return FourierSeries(as_expr(f), limits, coefficients)

    @property
    def a0(self) -> Expr:
        return self._a0

    @property
    def an(self) -> SeqFormula:
        """The sequence of the cosine terms."""
        return self._an

    @property
    def bn(self) -> SeqFormula:
        """The sequence of the sine terms."""
        return self._bn

    @property
    def stop(self) -> Expr:
        return oo

    @property
    def length(self) -> Expr:
        return oo

    def _make(self, function: Expr, a0: Expr, an: Basic, bn: Basic) -> FourierSeries:
        return FourierSeries(function, self._limits, (a0, _as_seq(an), _as_seq(bn)))

    # PORTED FROM SYMPY: sympy/series/fourier.py:FourierSeries.shift
    def shift(self, s: ExprLike) -> Expr:
        """The series of ``f(x) + s``, ``s`` free of ``x``."""
        e = self._check_free(s)
        return self._make(self._function + e, self._a0 + e, self._an, self._bn)

    # PORTED FROM SYMPY: sympy/series/fourier.py:FourierSeries.shiftx
    def shiftx(self, s: ExprLike) -> Expr:
        """The series of ``f(x + s)``, ``s`` free of ``x``."""
        e, x = self._check_free(s), self.x
        return self._make(self._function.subs(x, x + e), self._a0, self._an.subs(x, x + e), self._bn.subs(x, x + e))

    # PORTED FROM SYMPY: sympy/series/fourier.py:FourierSeries.scale
    def scale(self, s: ExprLike) -> Expr:
        """The series of ``s f(x)``, ``s`` free of ``x``."""
        e = self._check_free(s)
        return self._make(self._function * e, self._a0 * e, self._an.coeff_mul(e), self._bn.coeff_mul(e))

    # PORTED FROM SYMPY: sympy/series/fourier.py:FourierSeries.scalex
    def scalex(self, s: ExprLike) -> Expr:
        """The series of ``f(s x)``, ``s`` free of ``x``."""
        e, x = self._check_free(s), self.x
        return self._make(self._function.subs(x, x * e), self._a0, self._an.subs(x, x * e), self._bn.subs(x, x * e))

    def _eval_term(self, pt: Expr) -> Expr:
        if pt == 0:
            return self._a0
        return self._an.coeff(pt) + self._bn.coeff(pt)

    # PORTED FROM SYMPY: sympy/series/fourier.py:FourierSeries.__add__
    def __add__(self, other: ExprLike) -> Expr:
        o = to_expr(other)
        if isinstance(o, FourierSeries):
            if self.period != o.period:
                raise ValueError("Both the series should have same periods")
            x, y = self.x, o.x
            function = self._function + o.function.subs(y, x)
            if x not in function.free_symbols:
                return function
            return self._make(function, self._a0 + o.a0, self._an + o.an, self._bn + o.bn)
        return add(self, o)


def _as_dict(d: Basic) -> dict[Expr, Expr]:
    if not isinstance(d, Dict):
        raise TypeError("the coefficients of a finite Fourier series are dictionaries")
    return {as_expr(k): as_expr(v) for k, v in d.items()}


# PORTED FROM SYMPY: sympy/series/fourier.py:FiniteFourierSeries
class FiniteFourierSeries(_FourierBase):
    """A trigonometric polynomial as a Fourier series: ``(function, (x,
    start, stop), (a0, an, bn))`` with the dictionaries ``an``, ``bn`` of
    the coefficients of ``cos(k pi x/L)`` and ``sin(k pi x/L)``
    (:func:`finite_fourier_series` reads them off an expression)."""

    __slots__ = ('_a0', '_an', '_bn')

    def __init__(self, function: ExprLike, limits: Union[Tuple, Sequence[ExprLike]],
                 coefficients: Union[Tuple, tuple[ExprLike, dict[Expr, Expr], dict[Expr, Expr]]]) -> None:
        lims = limits if isinstance(limits, Tuple) else Tuple(*limits)
        if isinstance(coefficients, Tuple):
            if len(coefficients.args) != 3:
                raise TypeError("the coefficients of a Fourier series are (a0, an, bn)")
            a0, an, bn = as_expr(coefficients.args[0]), _as_dict(coefficients.args[1]), _as_dict(coefficients.args[2])
        else:
            a0, an, bn = to_expr(coefficients[0]), dict(coefficients[1]), dict(coefficients[2])
        _FourierBase.__init__(self, to_expr(function), lims)
        self._a0: Final[Expr] = a0
        self._an: Final[dict[Expr, Expr]] = an
        self._bn: Final[dict[Expr, Expr]] = bn
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._function, self._limits, Tuple(self._a0, Dict(*self._an.items()), Dict(*self._bn.items())))

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, limits, coefficients = args
        if not (isinstance(limits, Tuple) and isinstance(coefficients, Tuple)):
            raise TypeError("invalid arguments of a FiniteFourierSeries: %s" % (args,))
        return FiniteFourierSeries(as_expr(f), limits, coefficients)

    @property
    def a0(self) -> Expr:
        return self._a0

    @property
    def an(self) -> dict[Expr, Expr]:
        """The coefficients of the cosines, by their multiple."""
        return dict(self._an)

    @property
    def bn(self) -> dict[Expr, Expr]:
        """The coefficients of the sines, by their multiple."""
        return dict(self._bn)

    @property
    def stop(self) -> Expr:
        keys = [int(k) for k in list(self._an) + list(self._bn) if isinstance(k, Integer)]
        length = (0 if _is_zero(self._a0) else 1) + (max(keys) if keys else 0) + 1
        return Integer(length)

    @property
    def length(self) -> Expr:
        return self.stop - self.start

    def _converted(self, function: Expr, expr: Expr) -> FiniteFourierSeries:
        return finite_fourier_series(function, self._limits, expr)

    def shift(self, s: ExprLike) -> Expr:
        """The series of ``f(x) + s``, ``s`` free of ``x``."""
        e = self._check_free(s)
        return FiniteFourierSeries(self._function + e, self._limits, (self._a0 + e, self._an, self._bn))

    # PORTED FROM SYMPY: sympy/series/fourier.py:FiniteFourierSeries.shiftx
    def shiftx(self, s: ExprLike) -> Expr:
        """The series of ``f(x + s)``, ``s`` free of ``x``."""
        e, x = self._check_free(s), self.x
        return self._converted(self._function.subs(x, x + e), self.truncate(oo).subs(x, x + e))

    # PORTED FROM SYMPY: sympy/series/fourier.py:FiniteFourierSeries.scale
    def scale(self, s: ExprLike) -> Expr:
        """The series of ``s f(x)``, ``s`` free of ``x``."""
        e = self._check_free(s)
        return self._converted(self._function * e, self.truncate(oo) * e)

    # PORTED FROM SYMPY: sympy/series/fourier.py:FiniteFourierSeries.scalex
    def scalex(self, s: ExprLike) -> Expr:
        """The series of ``f(s x)``, ``s`` free of ``x``."""
        e, x = self._check_free(s), self.x
        return self._converted(self._function.subs(x, x * e), self.truncate(oo).subs(x, x * e))

    def _eval_term(self, pt: Expr) -> Expr:
        if pt == 0:
            return self._a0
        x, L = self.x, self.L
        return (self._an.get(pt, _ZERO) * cos(pt * (pi / L) * x)
                + self._bn.get(pt, _ZERO) * sin(pt * (pi / L) * x))

    # PORTED FROM SYMPY: sympy/series/fourier.py:FiniteFourierSeries.__add__
    def __add__(self, other: ExprLike) -> Expr:
        o = to_expr(other)
        if isinstance(o, FourierSeries):
            return o.__add__(fourier_series(self._function, self._limits, finite=False))
        if isinstance(o, FiniteFourierSeries):
            if self.period != o.period:
                raise ValueError("Both the series should have same periods")
            x, y = self.x, o.x
            function = self._function + o.function.subs(y, x)
            if x not in function.free_symbols:
                return function
            return fourier_series(function, self._limits)
        return add(self, o)


# PORTED FROM SYMPY: sympy/series/fourier.py:FiniteFourierSeries.__new__
def finite_fourier_series(f: ExprLike, limits: Union[Tuple, Sequence[ExprLike]], expr: ExprLike) -> FiniteFourierSeries:
    """The finite Fourier series of ``f`` whose trigonometric polynomial
    (in sines and cosines of multiples of ``pi x/L``) is ``expr``: the
    coefficients read off after the angles are expanded (``TR10``).

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.functions import sin, cos
    >>> from sympy_extras.series.fourier import finite_fourier_series
    >>> x = Symbol('x')
    >>> s = finite_fourier_series(sin(x + 1), (x, -pi, pi), sin(x + 1))
    >>> s.a0, s.an, s.bn
    (0, {1: sin(1)}, {1: cos(1)})
    """
    from sympy_extras.core.function import expand
    from sympy_extras.simplify.fu import TR10
    lims = limits if isinstance(limits, Tuple) else Tuple(*limits)
    x, start, stop = _limits_of(lims)
    e = to_expr(expr)
    c, rest = e.as_coeff_add()
    rexpr = c + add(*[TR10(t) for t in rest])
    expanded = expand(rexpr, trig=False, power_base=False, power_exp=False, log=False)
    number, terms = expanded.as_coeff_add()
    a0: Expr = number
    L = abs_(stop - start) / 2
    a, b = _harmonic_wilds(x)
    an: dict[Expr, Expr] = {}
    bn: dict[Expr, Expr] = {}
    cos_pattern = b * cos(a * (pi / L) * x)
    sin_pattern = b * sin(a * (pi / L) * x)
    for p in terms:
        t = p.match(cos_pattern)
        q = p.match(sin_pattern)
        if t:
            k = as_expr(t[a])
            an[k] = as_expr(t[b]) + an.get(k, _ZERO)
        elif q:
            k = as_expr(q[a])
            bn[k] = as_expr(q[b]) + bn.get(k, _ZERO)
        else:
            a0 = a0 + p
    return FiniteFourierSeries(to_expr(f), lims, (a0, an, bn))


# PORTED FROM SYMPY: sympy/series/fourier.py:fourier_series
def fourier_series(f: ExprLike, limits: Optional[Union[FourierLimits, Tuple]] = None, finite: bool = True) -> Expr:
    """The Fourier trigonometric series of ``f`` on ``(start, stop)``
    (``(-pi, pi)`` by default): a :class:`FiniteFourierSeries` for a
    trigonometric polynomial of the period (with ``finite``), a
    :class:`FourierSeries` otherwise; ``f`` itself when it does not depend
    on ``x``. The series converges to the periodic extension of ``f`` from
    the interval.

    Parameters
    ==========

    limits : (x, start, stop) or (start, stop), optional
        The variable and the interval; the variable is the only free
        symbol of ``f`` when not given.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.functions import piecewise
    >>> from sympy_extras.series.fourier import fourier_series
    >>> x = Symbol('x')
    >>> fourier_series(x, (x, -pi, pi)).truncate()
    2*sin(x) - sin(2*x) + 2*sin(3*x)/3
    >>> fourier_series(x, (x, 0, 1)).truncate(3)
    -sin(2*pi*x)/pi - sin(4*pi*x)/(2*pi) + 1/2
    >>> s = fourier_series(piecewise((0, x < 0), (pi, True)), (x, -pi, pi))
    >>> s.term(3), s.truncate()
    (2*sin(3*x)/3, 2*sin(x) + 2*sin(3*x)/3 + pi/2)
    """
    fe = to_expr(f)
    lims = _process_limits(fe, limits)
    x, start, stop = _limits_of(lims)
    if x not in fe.free_symbols:
        return fe
    if finite:
        L = abs_(stop - start) / 2
        is_finite, res_f = finite_check(fe, x, L)
        if is_finite:
            return finite_fourier_series(fe, lims, res_f)
    n = _index()
    center = (start + stop) / 2
    if center.is_zero:
        neg_f = fe.subs(x, -x)
        if fe == neg_f:
            a0, an = fourier_cos_seq(fe, lims, n)
            return FourierSeries(fe, lims, (a0, an, SeqFormula(_ZERO, (1, oo))))
        if fe == -neg_f:
            return FourierSeries(fe, lims, (_ZERO, SeqFormula(_ZERO, (1, oo)), fourier_sin_seq(fe, lims, n)))
    a0, an = fourier_cos_seq(fe, lims, n)
    bn = fourier_sin_seq(fe, lims, n)
    return FourierSeries(fe, lims, (a0, an, bn))

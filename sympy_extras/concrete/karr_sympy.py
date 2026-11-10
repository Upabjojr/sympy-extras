r"""The SymPy face of Karr's algorithm (:mod:`sympy_extras.concrete.karr`).

:mod:`~sympy_extras.concrete.karr` is written on the core and takes and
returns core expressions. The summation modules still written on SymPy
(:mod:`~sympy_extras.concrete.qhyper`) use this module instead:

- :func:`karr_sum`, :func:`karr_term` and :func:`summation` take and
  return SymPy expressions, converted at the boundary
  (:mod:`sympy_extras.interop`) around the native functions;
- :func:`with_sum_special_values` gives a sum the cases of the isolated
  values of its parameters (:mod:`sympy_extras._special_values`, on
  SymPy), for these modules and, through
  :func:`sum_special_values_of_core`, for the native ones
  (:mod:`~sympy_extras.concrete.karr`,
  :mod:`~sympy_extras.concrete.creative`,
  :mod:`~sympy_extras.concrete.zeilberger`, ...);
- :func:`reliable_value_of_core` evaluates a core expression at sample
  values of its parameters with
  :func:`~sympy_extras._numeric.reliable_value`, for the checks of
  :mod:`~sympy_extras.concrete.creative`.

Examples
========

>>> from sympy import harmonic, factorial, binomial
>>> from sympy.abc import k, n
>>> from sympy_extras.concrete.karr_sympy import karr_sum
>>> karr_sum(harmonic(k), (k, 1, n))
n*harmonic(n) - n + harmonic(n)
>>> karr_sum(binomial(2*k, k)/4**k, (k, 0, n))
(2*n + 1)*binomial(2*n, n)/4**n
>>> karr_sum(2**k/k, (k, 1, n)) is None
True
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.traversal import preorder_traversal
from sympy.core.expr import Expr
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.polys.polytools import Poly
from sympy.core.numbers import Integer, Rational, nan, zoo
from sympy.polys.rationaltools import together
from sympy.polys.polytools import cancel, factor_list

from sympy_extras._numeric import reliable_value
from sympy_extras._special_values import Point, with_special_values
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings
from sympy_extras.assumptions.ask import Assumptions

from sympy_extras._typing import as_expr, as_symbol, free_symbols, sorted_symbols

from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import expr_from_sympy, to_sympy

from . import karr as core_karr
from .karr import SumCompute, SumPoint

__all__ = ['karr_term', 'karr_sum', 'summation', 'with_sum_special_values', 'sum_special_values_of_core',
           'reliable_value_of_core']


# ----------------------------------------------------------------------
# the boundary with the core

def _core(e: Union[Basic, int]) -> CoreExpr:
    return expr_from_sympy(sympify(e))


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = _core(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy(e: CoreExpr) -> Expr:
    return as_expr(to_sympy(e))


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    return as_symbol(to_sympy(s))


def _core_spec(k: Union[Symbol, Sequence[Union[Symbol, Expr, int]]]) -> Union[CoreSymbol, tuple[CoreSymbol, CoreExpr, CoreExpr]]:
    if isinstance(k, Symbol):
        return _core_symbol(k)
    if len(k) != 3:
        raise ValueError("a summation limit is (k, a, b), not %s" % (k,))
    index, a, b = k
    if not isinstance(index, Symbol):
        raise TypeError("%s is not a symbol" % (index,))
    return (_core_symbol(index), _core(a), _core(b))


def _result(value: Optional[CoreExpr]) -> Optional[Expr]:
    return None if value is None else _sympy(value)


def karr_term(f: Union[Expr, int], k: Symbol, extensions: Sequence[Expr] = (), auto: bool = True,
              special_values: bool = True) -> Optional[Expr]:
    """:func:`sympy_extras.concrete.karr.karr_term` on SymPy expressions.

    >>> from sympy import harmonic
    >>> from sympy.abc import k
    >>> from sympy_extras.concrete.karr_sympy import karr_term
    >>> karr_term(harmonic(k), k)
    k*(harmonic(k) - 1)
    """
    return _result(core_karr.karr_term(_core(f), _core_symbol(as_symbol(k)), [_core(e) for e in extensions], auto,
                                       special_values))


def karr_sum(f: Union[Expr, int], k: Union[Symbol, Sequence[Union[Symbol, Expr, int]]],
             extensions: Sequence[Expr] = (), auto: bool = True, special_values: bool = True) -> Optional[Expr]:
    """:func:`sympy_extras.concrete.karr.karr_sum` on SymPy expressions.

    >>> from sympy.abc import k, n, y
    >>> from sympy_extras.concrete.karr_sympy import karr_sum
    >>> karr_sum(y**k, (k, 0, n))
    Piecewise((n + 1, Eq(y, 1)), ((y*y**n - 1)/(y - 1), True))
    """
    return _result(core_karr.karr_sum(_core(f), _core_spec(k), [_core(e) for e in extensions], auto, special_values))


def summation(f: Union[Expr, int], *symbols: Sequence[Union[Symbol, Expr, int]],
              extensions: Sequence[Expr] = (), auto: bool = True, special_values: bool = True) -> Expr:
    """:func:`sympy_extras.concrete.karr.summation` on SymPy expressions.

    >>> from sympy import harmonic
    >>> from sympy.abc import k, n
    >>> from sympy_extras.concrete.karr_sympy import summation
    >>> summation(harmonic(k)/(n + 1 - k), (k, 1, n))
    harmonic(n + 1)**2 - harmonic(n + 1, 2)
    """
    limits: list[tuple[CoreSymbol, CoreExpr, CoreExpr]] = []
    for spec in symbols:
        core_spec = _core_spec(spec)
        if isinstance(core_spec, CoreSymbol):
            raise ValueError("summation needs the limits (k, a, b), not %s" % (spec,))
        limits.append(core_spec)
    return _sympy(core_karr.summation(_core(f), *limits, extensions=[_core(e) for e in extensions], auto=auto,
                                      special_values=special_values))


def sum_special_values_of_core(value: CoreExpr, f: CoreExpr, indices: Sequence[CoreSymbol],
                               limits: Sequence[CoreExpr], compute: SumCompute,
                               keep_unevaluated: bool = True,
                               defined: Optional[Callable[[SumPoint], bool]] = None) -> CoreExpr:
    """:func:`with_sum_special_values` for core expressions, the sum at a
    point computed by ``compute`` on core expressions (and ``defined``
    deciding on core points)."""
    def at_point(point: Point, at: Assumptions) -> Optional[Expr]:
        core_point: SumPoint = {_core_symbol(s): _core(v) for s, v in point.items()}
        return _result(compute(core_point))

    sympy_defined: Optional[Callable[[Point], bool]] = None
    if defined is not None:
        core_defined = defined

        def sympy_defined_(point: Point) -> bool:
            return core_defined({_core_symbol(s): _core(v) for s, v in point.items()})
        sympy_defined = sympy_defined_
    return _core(with_sum_special_values(_sympy(value), _sympy(f), [_sympy_symbol(i) for i in indices],
                                         [_sympy(e) for e in limits], at_point, keep_unevaluated,
                                         sympy_defined))


def reliable_value_of_core(e: CoreExpr, digits: int, values: SumPoint) -> Optional[complex]:
    """:func:`~sympy_extras._numeric.reliable_value` of a core expression
    at the ``values`` of its symbols, as a complex number (``None`` when
    it is not reliable), for :mod:`~sympy_extras.concrete.creative`."""
    point: dict[Symbol, Expr] = {_sympy_symbol(s): _sympy(v) for s, v in values.items()}
    value = reliable_value(_sympy(e), digits, point)
    return None if value is None else complex(value)


# ----------------------------------------------------------------------
# the isolated values of the parameters

def with_sum_special_values(value: Expr, f: Expr, indices: Sequence[Symbol], limits: Sequence[Expr],
                            compute: Callable[[Point, Assumptions], Optional[Expr]],
                            keep_unevaluated: bool = True,
                            defined: Optional[Callable[[Point], bool]] = None) -> Expr:
    """The value of a sum of ``f`` with a case for each isolated value of
    the parameters of the summand (not the ``indices``, nor the symbols
    of the ``limits``, whose small values the algorithms settle
    themselves) at which it is undefined, the sum computed again there
    by ``compute`` (:mod:`sympy_extras._special_values`): the sum of
    ``y**k`` is ``n + 1`` at ``y = 1``.

    ``indices`` starts with the index of summation and ``limits`` are its
    bounds ``(lower, upper)``, or are empty for an indefinite sum (an
    antidifference in the index). A point where the sum itself is not
    defined gets no case: one where the summand is ``nan`` or ``zoo``, or
    has a pole at an integer of the range of summation (``1/((k + a)*(k +
    b))`` from ``k = 0`` at ``a = 0``), or which ``defined`` refutes (a
    point outside the region of convergence of a series). A summand with
    powers of ``0`` at the point, which the algorithms do not take
    (``binomial(n, k)*y**k/(k + 1)`` at ``y = 0``), is summed by the
    values of ``0**(c*k + d)``: ``1`` where the exponent is zero, at the
    lower limit, and ``0`` beyond it. With the
    numerical checks of the settings each case is compared with the sum
    computed term by term, for a few values of the upper limit (or, for
    an antidifference, with the summand at a few values of the index),
    the other parameters at sample values, through
    :func:`~sympy_extras._numeric.reliable_value`; a case which disagrees
    is left out. ``keep_unevaluated`` is that of
    :func:`~sympy_extras._special_values.with_special_values`.

    >>> from sympy.abc import k, n, y
    >>> from sympy_extras.concrete.karr_sympy import with_sum_special_values
    >>> with_sum_special_values((y*y**n - 1)/(y - 1), y**k, [k], [0, n], lambda point, at: n + 1)
    Piecewise((n + 1, Eq(y, 1)), ((y*y**n - 1)/(y - 1), True))
    """
    excluded: set[Symbol] = set(indices)
    for limit in limits:
        excluded |= free_symbols(sympify(limit))
    parameters = sorted(free_symbols(f) - excluded, key=lambda s: s.name)
    if not parameters:
        return value
    index = indices[0]
    bounds = (as_expr(sympify(limits[0])), as_expr(sympify(limits[1]))) if len(limits) == 2 else None
    indefinite = not limits and len(indices) == 1

    def defined_at(point: Point) -> bool:
        term = as_expr(f.xreplace(point))
        if term.has(nan, zoo):
            return False
        if bounds is not None and _pole_in_range(term, index, bounds[0], bounds[1]):
            return False
        return defined is None or defined(point)

    def checked(point: Point, at: Assumptions) -> Optional[Expr]:
        term = as_expr(f.xreplace(point))
        if bounds is not None and _zero_powers(term, index):
            special = _sum_of_zero_powers(term, index, bounds[0], bounds[1])
        else:
            special = compute(point, at)
        if special is None or not settings.numerical_checks:
            return special
        if _disagrees(special, as_expr(f.xreplace(point)), index, bounds, indefinite):
            return None
        return special

    return with_special_values(value, parameters, None, checked, defined_at, keep_unevaluated=keep_unevaluated)


#: the seconds given to the search for the poles of a summand at a point
_POLE_SECONDS = 2.0
#: the seconds given to the evaluation of a case at one sample point
_CHECK_SECONDS = 2.0
#: the sample values of the parameters left free by a point, in the checks
_SAMPLES = (Rational(17, 7), Rational(23, 9), Rational(31, 11), Rational(41, 13))
#: the number of values of the upper limit (or of the index) checked
_CHECKS = 3
#: the relative difference beyond which a case disagrees with the direct sum
_TOLERANCE = 1e-9


def _zero_powers(term: Expr, k: Symbol) -> list[Pow]:
    """The powers of ``0`` with an exponent in ``k`` in ``term``."""
    return [node for node in preorder_traversal(term)
            if isinstance(node, Pow) and as_expr(node.base).is_zero and as_expr(node.exp).has(k)]


def _sum_of_zero_powers(term: Expr, k: Symbol, lower: Expr, upper: Expr) -> Optional[Expr]:
    """The sum of ``term`` from ``lower`` to ``upper`` when its powers
    ``0**(c*k + d)`` (``c`` a positive integer, ``d`` a rational number)
    are ``0`` over the range except at the lower limit, where the exponent
    may vanish: the sum of ``term`` with them written ``0``, and the term
    at the lower limit, where they are ``1``, for its own; ``None`` when a
    power is not of this form or is infinite somewhere in the range."""
    if not isinstance(lower, Integer):
        return None
    zeroed: dict[Basic, Basic] = {}
    at_lower = False
    for power in _zero_powers(term, k):
        exponent = as_expr(power.exp)
        if not exponent.is_polynomial(k):
            return None
        polynomial = Poly(exponent, k)
        c, d = as_expr(polynomial.nth(1)), as_expr(polynomial.nth(0))
        if polynomial.degree() != 1 or not isinstance(c, Integer) or c <= 0 or not isinstance(d, Rational):
            return None
        vanishing = as_expr(-d / c)
        if vanishing > lower:
            # 0 to a negative power inside the range
            return None
        at_lower = at_lower or vanishing == lower
        zeroed[power] = S.Zero
    rest = as_expr(term.xreplace(zeroed))
    value = summation(rest, (k, lower, upper)) if rest.has(k) else as_expr(rest * (upper - lower + 1))
    if at_lower:
        value = as_expr(value + term.subs(k, lower) - rest.subs(k, lower))
    return value


def _pole_in_range(term: Expr, k: Symbol, lower: Expr, upper: Expr) -> bool:
    """Whether ``term`` has a pole at an integer ``k`` of the range from
    ``lower`` (an integer) to ``upper`` (an integer, else unbounded): a
    factor ``c*k + d`` of its denominator with an integer root there. For
    another lower limit, or a denominator which is no polynomial in ``k``,
    ``False``: nothing is known."""
    if not isinstance(lower, Integer):
        return False
    denominator = as_expr(together(term).as_numer_denom()[1])
    if not denominator.has(k) or not denominator.is_polynomial(k):
        return False
    factored = attempt(lambda: factor_list(denominator, k), _POLE_SECONDS)
    if factored is None:
        return False
    for factor_, _ in factored[1]:
        polynomial = Poly(factor_, k)
        if polynomial.degree() != 1:
            continue
        root = as_expr(cancel(-polynomial.nth(0) / polynomial.LC()))
        if isinstance(root, Integer) and root >= lower and (not isinstance(upper, Integer) or root <= upper):
            return True
    return False


def _upper_values(lower: Integer, upper: Expr) -> Optional[list[tuple[Point, int]]]:
    """The values of the symbol of ``upper`` (of the form ``c*m + d``,
    ``c`` a positive and ``d`` any integer, or an integer) at which a sum
    from ``lower`` is checked, with the upper limit there; ``None`` for
    another upper limit."""
    if isinstance(upper, Integer):
        return [({}, int(upper))]
    symbols = sorted_symbols(free_symbols(upper))
    if len(symbols) != 1 or not upper.is_polynomial(symbols[0]):
        return None
    m = symbols[0]
    polynomial = Poly(upper, m)
    c, d = polynomial.nth(1), polynomial.nth(0)
    if polynomial.degree() != 1 or not isinstance(c, Integer) or not isinstance(d, Integer) or c <= 0:
        return None
    # the first value of m with an upper limit of at least lower - 1 (the
    # empty sum) and nonnegative (the limits of definite_sum are for m >= 0)
    first = max(0, -((d - int(lower) + 1) // int(c)))
    values: list[tuple[Point, int]] = [({m: Integer(v)}, int(c) * v + int(d))
                                       for v in range(first, first + _CHECKS)]
    return values


def _disagrees(value: Expr, term: Expr, k: Symbol, bounds: Optional[tuple[Expr, Expr]], indefinite: bool) -> bool:
    """Whether ``value``, the sum of ``term`` from ``bounds[0]`` to
    ``bounds[1]`` (or its antidifference in ``k``, when ``indefinite``),
    is refuted by the sum computed term by term at a few values of the
    upper limit (by the difference of the antidifference at a few values
    of ``k``), the other symbols at sample values. A value which cannot
    be evaluated reliably there (:func:`~sympy_extras._numeric.reliable_value`
    gives ``None``) is not refuted."""
    checks: list[tuple[Expr, Expr]] = []
    if bounds is not None:
        lower, upper = bounds
        if not isinstance(lower, Integer):
            return False
        uppers = _upper_values(lower, upper)
        if uppers is None:
            return False
        for at, top in uppers:
            direct = as_expr(Add(*[term.xreplace(at).subs(k, i) for i in range(int(lower), top + 1)]))
            checks.append((as_expr(value.xreplace(at)), direct))
    elif indefinite:
        for i in range(2, 2 + _CHECKS):
            checks.append((as_expr(value.subs(k, i + 1) - value.subs(k, i)), as_expr(term.subs(k, i))))
    else:
        return False
    for case, direct in checks:
        others = sorted_symbols((free_symbols(case) | free_symbols(direct)) - {k})
        samples: Point = {s: _SAMPLES[i % len(_SAMPLES)] + i // len(_SAMPLES)
                          for i, s in enumerate(others)}
        evaluated = attempt(lambda: as_expr(case.xreplace(samples).doit()), _CHECK_SECONDS)
        if evaluated is None:
            continue
        difference = reliable_value(as_expr(evaluated - direct.xreplace(samples)), 15)
        size = reliable_value(as_expr(direct.xreplace(samples)), 15)
        if difference is None or size is None:
            continue
        if float(abs(difference)) > _TOLERANCE * (1 + float(abs(size))):
            return True
    return False

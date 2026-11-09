r"""The SymPy face of Karr's algorithm (:mod:`sympy_extras.concrete.karr`).

:mod:`~sympy_extras.concrete.karr` is written on the core and takes and
returns core expressions. The summation modules still written on SymPy
(:mod:`~sympy_extras.concrete.creative`,
:mod:`~sympy_extras.concrete.zeilberger`,
:mod:`~sympy_extras.concrete.rational`,
:mod:`~sympy_extras.concrete.dirichlet`,
:mod:`~sympy_extras.concrete.qhyper`) use this module instead:

- :func:`karr_sum`, :func:`karr_term` and :func:`summation` take and
  return SymPy expressions, converted at the boundary
  (:mod:`sympy_extras.interop`) around the native functions;
- :func:`with_sum_special_values` gives a sum the cases of the isolated
  values of its parameters (:mod:`sympy_extras._special_values`, on
  SymPy), for these modules and, through
  :func:`sum_special_values_of_core`, for the native ones;
- the builder of a $\Pi\Sigma$-field on SymPy expressions
  (:class:`_Builder`, on the
  :class:`~sympy_extras.concrete.pisigma_sympy.SymPyPiSigmaField`), which
  creative telescoping extends as it goes; it goes when
  :mod:`~sympy_extras.concrete.creative` is moved onto the core.

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

from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.traversal import preorder_traversal
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import harmonic
from sympy.polys.polytools import Poly
from sympy.core.numbers import Integer, Rational, nan, zoo
from sympy.polys.rationaltools import together
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import cancel, factor_list
from sympy.simplify.simplify import hypersimp

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
from .pisigma import PiSigmaElement
from .pisigma_sympy import SymPyPiSigmaField

__all__ = ['karr_term', 'karr_sum', 'summation', 'with_sum_special_values', 'sum_special_values_of_core',
           'creative_sum_of_core']


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
                               keep_unevaluated: bool = True) -> CoreExpr:
    """:func:`with_sum_special_values` for core expressions, the sum at a
    point computed by ``compute`` on core expressions."""
    def at_point(point: Point, at: Assumptions) -> Optional[Expr]:
        core_point: SumPoint = {_core_symbol(s): _core(v) for s, v in point.items()}
        return _result(compute(core_point))

    return _core(with_sum_special_values(_sympy(value), _sympy(f), [_sympy_symbol(i) for i in indices],
                                         [_sympy(e) for e in limits], at_point, keep_unevaluated))


def creative_sum_of_core(f: CoreExpr, limit: tuple[CoreSymbol, CoreExpr, CoreExpr],
                         extensions: Sequence[CoreExpr]) -> Optional[CoreExpr]:
    """The sum by creative telescoping
    (:func:`~sympy_extras.concrete.creative.definite_sum`, on SymPy) of
    core expressions, without the cases of the special values, within
    the time limit of the settings; ``None`` if there is none."""
    from .creative import definite_sum
    k, a, b = limit
    f_, k_, a_, b_ = _sympy(f), _sympy_symbol(k), _sympy(a), _sympy(b)
    extensions_ = [_sympy(e) for e in extensions]
    value = attempt(lambda: definite_sum(f_, (k_, a_, b_), extensions=extensions_, special_values=False),
                    settings.timeout)
    return None if value is None else _core(value)


# ----------------------------------------------------------------------
# the builder of a field on SymPy expressions, for creative telescoping

def _atoms(expr: Expr, k: Symbol) -> set[Expr]:
    """The subexpressions of ``expr`` depending on ``k`` which are not
    rational operations: functions of ``k``, powers with ``k`` in the
    exponent or with a non-integer exponent, sums and products."""
    expr = sympify(expr)
    if not expr.has(k) or expr == k:
        return set()
    if isinstance(expr, (Add, Mul)):
        return set().union(*[_atoms(a, k) for a in expr.args])
    if isinstance(expr, Pow):
        if expr.exp.is_Integer and not expr.exp.has(k):
            return _atoms(expr.base, k)
        return {expr}
    return {expr}


def _linear_shift(arg: Expr, k: Symbol) -> Optional[int]:
    """``c`` if ``arg == k + c`` with ``c`` an integer, else ``None``."""
    c = arg - k
    if c.is_Integer:
        return int(c)
    return None


def _linear(arg: Expr, k: Symbol) -> Optional[tuple[int, int]]:
    """``(a, b)`` if ``arg == a*k + b`` with ``a`` a positive integer and
    ``b`` an integer, else ``None``."""
    arg = sympify(arg).expand()
    a = arg.coeff(k)
    b = (arg - a*k).expand()
    if a.is_Integer and a > 0 and b.is_Integer:
        return int(a), int(b)
    return None


def _evaluate_at(expr: Expr, k: Symbol, k0: int) -> Optional[Expr]:
    value = expr.subs(k, k0)
    try:
        value = value.doit()
    except Exception:
        return None
    if value.has(S.NaN, S.ComplexInfinity, S.Infinity, S.NegativeInfinity):
        return None
    return value


class _Builder:
    """Builds a $\\Pi\\Sigma$-field containing given expressions."""

    def __init__(self, k: Symbol, params: Sequence[Symbol]) -> None:
        self.k = k
        self.field = SymPyPiSigmaField(k, params)
        # atom -> expression in k and the generator symbols
        self.known: dict[Expr, Expr] = {}

    def convert(self, expr: Union[Expr, int]) -> PiSigmaElement:
        """The element of the field equal to ``expr``, adjoining
        generators as needed."""
        expr = sympify(expr)
        atoms = sorted(_atoms(expr, self.k), key=lambda a: (a.count_ops(), str(a)))
        for atom in atoms:
            if atom not in self.known:
                self.known[atom] = self._represent(atom)
        rewritten = expr.xreplace(self.known)
        try:
            return self.field.from_expr(rewritten)
        except (ValueError, PolynomialError, TypeError) as e:
            raise ValueError("%s is not a rational expression in the summation "
                             "index, the parameters and the sequences of the "
                             "field: %s" % (expr, e))

    def _represent(self, atom: Expr) -> Expr:
        k = self.k
        # sums and products with a shifted upper limit are expressed through
        # the unshifted ones
        canonical, shift = self._canonical(atom)
        if shift:
            if canonical not in self.known:
                self.known[canonical] = self._represent(canonical)
            element = self.field.from_expr(self.known[canonical])
            return self.field.to_expr(self.field.sigma(element, shift), substitute=False)
        if isinstance(atom, harmonic):
            order = atom.args[1] if len(atom.args) > 1 else S.One
            linear = _linear(atom.args[0], k)
            if linear is not None and not order.has(k):
                a, b = linear
                # H_{a(k+1)+b} - H_{ak+b} = sum of the a reciprocals in between
                beta = sum(1/(a*k + b + i)**order for i in range(1, a + 1))
                return self._sigma(atom, beta)
        if isinstance(atom, Sum) and len(atom.limits) == 1:
            j, lo, hi = atom.limits[0]
            if hi == k and not lo.has(k) and not atom.function.has(k):
                return self._sigma(atom, atom.function.subs(j, k + 1))
        if isinstance(atom, Product) and len(atom.limits) == 1:
            j, lo, hi = atom.limits[0]
            if hi == k and not lo.has(k) and not atom.function.has(k):
                return self._pi(atom, atom.function.subs(j, k + 1))
        ratio = hypersimp(atom, k)
        if ratio is not None and ratio != 0:
            return self._pi(atom, ratio)
        raise ValueError("%s is not a hypergeometric term or an indefinite sum "
                         "in %s" % (atom, k))

    def _canonical(self, atom: Expr) -> tuple[Expr, int]:
        """``(atom with k in place of k + c, c)`` for harmonic numbers, sums
        and products with argument or upper limit ``k + c``."""
        k = self.k
        if isinstance(atom, harmonic):
            linear = _linear(atom.args[0], k)
            if linear is not None:
                a, b = linear
                b0 = b % a
                if b != b0:
                    return atom.func(a*k + b0, *atom.args[1:]), (b - b0)//a
        if isinstance(atom, (Sum, Product)) and len(atom.limits) == 1:
            j, lo, hi = atom.limits[0]
            shift = _linear_shift(hi, k)
            if shift and not lo.has(k) and not atom.function.has(k):
                return atom.func(atom.function, (j, lo, k)), shift
        return atom, 0

    def _initial_constant(self, atom: Expr, w_expr: Expr, ratio: bool = False) -> Optional[Expr]:
        """The constant ``atom - w`` (or ``atom/w``) as an element of the
        constants, by evaluation at a point, or ``None``."""
        for k0 in range(0, 12):
            a0 = _evaluate_at(atom, self.k, k0)
            w0 = _evaluate_at(w_expr, self.k, k0)
            if a0 is None or w0 is None:
                continue
            if ratio:
                if w0 == 0:
                    continue
                c = cancel(a0/w0)
            else:
                c = cancel(a0 - w0)
            if not self.field.is_constant_expr(c):
                return None
            # the identity must hold for all k: check at another point
            k1 = k0 + 1
            a1 = _evaluate_at(atom, self.k, k1)
            w1 = _evaluate_at(w_expr, self.k, k1)
            if a1 is not None and w1 is not None:
                check = cancel(a1/w1 - c) if ratio else cancel(a1 - w1 - c)
                if check != 0:
                    return None
            return c
        return None

    def _sigma(self, atom: Expr, beta: Expr) -> Expr:
        b = self.convert(beta)
        w = self.field.telescope(b)
        if w is not None:
            c = self._initial_constant(atom, self.field.to_expr(w))
            if c is not None:
                return self.field.to_expr(w, substitute=False) + c
        return self.field.add_sigma(self.field.to_expr(b, substitute=False), atom)

    def _pi(self, atom: Expr, alpha: Expr) -> Expr:
        a = self.convert(alpha)
        sols = self.field.solve(a, [])
        if sols:
            for _, w in sols:
                if w != self.field.zero:
                    c = self._initial_constant(atom, self.field.to_expr(w), ratio=True)
                    if c is not None:
                        return c*self.field.to_expr(w, substitute=False)
        return self.field.add_pi(self.field.to_expr(a, substitute=False), atom)


def _auto_extensions(f: Expr, k: Symbol) -> list[Expr]:
    """Harmonic numbers to adjoin when the telescoping fails: for a summand
    with harmonic numbers or nested sums of total degree ``d`` and highest
    order ``m``, the harmonic numbers of orders up to ``m + d``."""
    f = sympify(f)
    atoms = [a for a in _atoms(f, k) if isinstance(a, (harmonic, Sum))]
    if not atoms:
        # a rational summand with no closed form in the rational functions
        # sums to harmonic numbers: partial fractions give orders up to
        # the multiplicity of the poles (sum 1/k is harmonic(n),
        # sympy-extras#34)
        numerator, denominator = f.as_numer_denom()
        if not denominator.has(k) or not f.is_rational_function(k):
            return []
        multiplicity = max((int(m) for _, m in Poly(denominator, k).factor_list()[1]), default=0)
        return [harmonic(k, m) for m in range(1, multiplicity + 1)]
    order = 1
    for a in atoms:
        if isinstance(a, harmonic) and len(a.args) > 1 and a.args[1].is_Integer:
            order = max(order, int(a.args[1]))
    degree = 0
    for a in atoms:
        degree = max(degree, _degree_in(f, a))
    return [harmonic(k, m) for m in range(1, order + max(degree, 1) + 1)]


def _degree_in(expr: Expr, atom: Expr) -> int:
    """The degree of ``expr`` in ``atom`` (an upper bound)."""
    expr = sympify(expr)
    if expr == atom:
        return 1
    if not expr.has(atom):
        return 0
    if isinstance(expr, Add):
        return max(_degree_in(a, atom) for a in expr.args)
    if isinstance(expr, Mul):
        return sum(_degree_in(a, atom) for a in expr.args)
    if isinstance(expr, Pow) and expr.exp.is_Integer:
        return abs(int(expr.exp))*_degree_in(expr.base, atom)
    return 1


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

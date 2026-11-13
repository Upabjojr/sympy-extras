"""Validated numerical integration: a value with a rigorous error bound.

When no closed form is found, a number with a proved error bound is the
next best answer. Every panel `[c, d]` of width `h` of a subdivision of
the range is integrated by a quadrature rule with a rigorous error bound:
the Simpson rule with

.. math::

    \\left| \\int_c^d f\\, dx - \\frac{h}{6}\\bigl(f(c) + 4 f(m) + f(d)\\bigr) \\right|
    \\le \\frac{h^5}{2880} \\max_{[c, d]} |f^{(4)}|,

or the `n`-point Gauss–Legendre rule with [Hildebrand]_

.. math::

    |E_n| \\le \\frac{h^{2n+1} (n!)^4}{(2n+1)\\, ((2n)!)^3} \\max_{[c, d]} |f^{(2n)}|,

whichever bound is smaller on the panel (`n = 5`, so the tenth
derivative is enclosed; the nodes are enclosed in rational intervals
verified by a sign change of the Legendre polynomial, and the weights
computed from them in interval arithmetic). Interval arithmetic
(:mod:`sympy_extras.assumptions.intervals`, mpmath's outward rounding)
gives proved bounds of the derivatives on the panel and proved enclosures
of the values of `f` at the nodes, so the sum over the panels is an
enclosure of the integral [Moore]_. The panels are bisected adaptively
where the bound is largest [Petras]_ until the total is below the
requested tolerance. A panel on which no derivative has a finite bound is
bounded by its width times the range of `f`; a panel at an endpoint on
which even the range is infinite (`\\log x` at `0`) is bounded by the
integrable model `\\int_0^h |c(v)|\\, v^\\alpha |\\log v|^j dv`, `\\alpha
> -1`, after writing the integrand as a sum of such terms with bounded
coefficients `c`. Removable singularities (`\\sin x / x` at `0`) are
enclosed through the Taylor remainder of the numerator. An algebraic
endpoint singularity `(x - a)^{p/q}` is first removed by `x = a + (b -
a) u^q`. An infinite range is mapped onto `[0, 1]` by `x = a + t/(1 - t)`
(algebraic decay) or by `x = a - \\log(1 - t)` (exponential decay, after
`x = s^{1/k}` when the decay is `e^{-c x^k}`); an oscillatory tail
`\\int_A^\\infty g(x) \\sin(\\omega x + c)\\, dx` with `g^{(k)}` monotone to
`0` is integrated by parts `k` times and the remainder bounded by `2
|g^{(k)}(A)| / \\omega^{k+1}` (Bonnet's mean value theorem), `A` chosen so
that the remainder is below the tolerance. The method follows the
principles of verified quadrature [Petras]_ and of Arb's ball-arithmetic
integration [Johansson]_ in a simple form.

The module works on the objects of the core (:mod:`sympy_extras.core`, the
functions of :mod:`sympy_extras.functions`) and imports no SymPy: the
enclosures come from the native interval evaluator, the symbolic
preparations (derivatives, ``together``, ``cancel``, ``simplify``,
``powdenest``, limits at infinity) are the core's, the polynomials in the
variable are lists of coefficients and Sturm's theorem is that of
:mod:`sympy_extras.polys.rootisolation`.
:mod:`sympy_extras.integrals.validated_sympy` is its face on SymPy objects.

Examples
========

>>> from sympy_extras.core import symbols, sqrt, pi, oo
>>> from sympy_extras.functions import exp, erf
>>> from sympy_extras.integrals.validated import validated_integral
>>> x = symbols('x')
>>> value, error = validated_integral(exp(-x**2), x, 0, 1)
>>> round(float(value), 12), float(error) < 1e-15
(0.746824132812, True)
>>> value, error = validated_integral(exp(-x**2), x, 0, oo)
>>> float(error) < 1e-15
True

References
==========

.. [Petras] K. Petras, *Principles of verified numerical integration*,
   Journal of Computational and Applied Mathematics 199 (2007) 317–328.
.. [Johansson] F. Johansson, *Numerical integration in arbitrary-precision
   ball arithmetic*, Mathematical Software – ICMS 2018, LNCS 10931.
.. [Moore] R. E. Moore, *Interval Analysis*, Prentice-Hall, 1966.
.. [Hildebrand] F. B. Hildebrand, *Introduction to Numerical Analysis*,
   2nd ed., Dover, 1987, §8.5 (the error of Gaussian quadrature).
"""
from __future__ import annotations

import heapq
import math
import time
from fractions import Fraction
from typing import Callable, Final, Optional, TypeGuard

import mpmath
from mpmath.ctx_iv import ivmpf

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.intervals import evaluate
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import count_ops, expand, expand_power_exp
from sympy_extras.core.limits import limit
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Float, Integer, Rational, number_from_fraction, oo
from sympy_extras.core.numbers import dps_to_prec
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Abs
from sympy_extras.functions.elementary.exponential import Exp, Log, log
from sympy_extras.functions.elementary.hyperbolic import Cosh, Sinh, Tanh
from sympy_extras.functions.elementary.integers import ceiling
from sympy_extras.functions.elementary.trigonometric import Atan, Cos, Sin, Tan, cos, sin
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import cancel, factor
from sympy_extras.polys.rationaltools import together
from sympy_extras.polys.rootisolation import dup_count_real_roots
from sympy_extras.settings import settings
from sympy_extras.simplify.powsimp import powdenest
from sympy_extras.simplify.simplify import simplify

__all__ = ['validated_integral', 'enclosure']

_iv = mpmath.iv

#: the largest number of panels before the refinement stops
_MAX_PANELS = 6000
#: the nodes of the Gauss–Legendre rule (its bound needs the derivative
#: of order twice the number of nodes)
_GAUSS_NODES = 5
#: the largest size of a derivative still enclosed on every panel
_MAX_OPS = 5000
#: how many times an oscillatory tail is integrated by parts at most
_MAX_PARTS = 10

_FUNCTIONS = (Exp, Log, Sin, Cos, Tan, Atan, Sinh, Cosh, Tanh, Abs)

#: when the whole computation must stop refining (set by
#: :func:`validated_integral` from ``settings.timeout``)
_deadline: Optional[float] = None

#: the coefficients of a polynomial in the variable, lowest degree first
#: (no trailing zero; the zero polynomial is ``[0]``)
Coefficients = list[Expr]


def _budget() -> Optional[float]:
    """The time allowed to one symbolic preparation (a derivative, a
    ``together``, an ``expand``): an eighth of the time limit."""
    return None if settings.timeout is None else settings.timeout / 8


def _out_of_time() -> bool:
    return _deadline is not None and time.monotonic() > _deadline


def _finite(value: Optional[ivmpf]) -> TypeGuard[ivmpf]:
    return value is not None and bool(mpmath.isfinite(value.a)) and bool(mpmath.isfinite(value.b))


def _rational(value: Expr) -> Optional[Rational]:
    if isinstance(value, Rational):
        return value
    if isinstance(value, Float):
        return number_from_fraction(value.as_fraction())
    return None


def _exact(value: mpmath.mpf) -> Rational:
    """An mpmath float as the exact rational it represents."""
    sign, mantissa, exponent, _ = value._mpf_
    if exponent >= 0:
        result = Fraction(int(mantissa) * 2**int(exponent))
    else:
        result = Fraction(int(mantissa), 2**(-int(exponent)))
    return number_from_fraction(-result if sign else result)


def _interval(value: Rational) -> ivmpf:
    return _iv.mpf(int(value.p)) / _iv.mpf(int(value.q))


def _sum(lo: Rational, hi: Rational) -> Rational:
    return number_from_fraction(lo.as_fraction() + hi.as_fraction())


def _difference(hi: Rational, lo: Rational) -> Rational:
    return number_from_fraction(hi.as_fraction() - lo.as_fraction())


def _scaled(value: Rational, factor_: Rational) -> Rational:
    return number_from_fraction(value.as_fraction() * factor_.as_fraction())


def _mid(lo: Rational, hi: Rational) -> Rational:
    return number_from_fraction((lo.as_fraction() + hi.as_fraction()) / 2)


def _integer(e: Basic) -> Optional[int]:
    """``e`` as a Python integer when it is an integer number (an
    ``Integer``, or a ``Rational`` with denominator one)."""
    if isinstance(e, Rational) and e.q == 1:
        return e.p
    return None


def _ops(e: Expr) -> int:
    counted = _integer(count_ops(e))
    return 0 if counted is None else counted


def _is_true(value: Optional[bool]) -> bool:
    return value is True


# ---------------------------------------------------------------------------
# polynomials in the variable
# ---------------------------------------------------------------------------

def _polynomial(expr: Expr, x: Symbol) -> Optional[Coefficients]:
    """The coefficients of ``expr`` as a polynomial in ``x`` (after
    expansion), any expressions free of ``x``; ``None`` when ``x`` occurs
    otherwise than in powers with nonnegative integer exponents (what
    ``Poly(expr, x)`` raises ``PolynomialError`` on)."""
    expanded = expand(expr)
    collected: dict[int, list[Expr]] = {}
    for term in Add.make_args(expanded):
        degree = 0
        rest: list[Expr] = []
        for factor_ in Mul.make_args(term):
            if factor_ == x:
                degree += 1
            elif isinstance(factor_, Pow) and factor_.base == x:
                n = _integer(factor_.exp)
                if n is None or n < 0:
                    return None
                degree += n
            elif x in factor_.free_symbols:
                return None
            else:
                rest.append(factor_)
        collected.setdefault(degree, []).append(mul(*rest))
    top = max(collected) if collected else 0
    coefficients: Coefficients = [add(*collected.get(k, [])) for k in range(top + 1)]
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    return coefficients


def _degree(p: Coefficients) -> int:
    return len(p) - 1 if len(p) > 1 or p[0] != 0 else -1


def _divided(p: Coefficients, c: Expr) -> tuple[Coefficients, Expr]:
    """The quotient of ``p`` by ``x - c`` and the remainder ``p(c)``
    (Ruffini–Horner)."""
    quotient: list[Expr] = []
    carry: Expr = S.Zero
    for coefficient in reversed(p):
        carry = add(mul(carry, c), coefficient)
        quotient.append(carry)
    remainder = quotient.pop()
    quotient.reverse()
    return (quotient if quotient else [S.Zero]), remainder


def _expr_of(p: Coefficients, x: Symbol) -> Expr:
    return add(*[mul(c, pow(x, Integer(k))) for k, c in enumerate(p)])


def _rational_coefficients(p: Coefficients) -> Optional[list[Fraction]]:
    """The coefficients as fractions, highest degree first, when they
    are all rational numbers."""
    fractions: list[Fraction] = []
    for c in reversed(p):
        if not isinstance(c, Rational):
            return None
        fractions.append(c.as_fraction())
    return fractions


# ---------------------------------------------------------------------------
# enclosures
# ---------------------------------------------------------------------------

def _enclose(expr: Expr, x: Symbol, lo: Expr, hi: Expr, removable: bool = True) -> Optional[ivmpf]:
    """An enclosure of ``expr`` on ``[lo, hi]``: the interval evaluator,
    extended by the limits of ``log`` and of negative powers at ``0``
    (``log([0, c]) = [-oo, log c]``), by the powers of a nonnegative base
    and a nonnegative exponent (``x**x`` at ``0``), and, for the whole
    expression only (the recursion would try every subexpression), by the
    Taylor remainder at removable singularities (``sin(x)/x`` at ``0``).
    The result may be infinite (the panel is then unbounded)."""
    value = evaluate(expr, {x: (lo, hi)})
    if value is None:
        value = _extended(expr, x, lo, hi)
    if _finite(value) or not removable:
        return value
    fixed = _removable(expr, x, lo, hi)
    return fixed if fixed is not None else value


def _extended(expr: Expr, x: Symbol, lo: Expr, hi: Expr) -> Optional[ivmpf]:
    if isinstance(expr, (Add, Mul)):
        total: Optional[ivmpf] = None
        for part in expr.args:
            value = _enclose(as_expr(part), x, lo, hi, removable=False)
            if value is None:
                return None
            if total is None:
                total = value
            elif isinstance(expr, Add):
                total = total + value
            else:
                total = total * value
        return total
    if isinstance(expr, Pow):
        return _extended_power(expr.base, expr.exp, x, lo, hi)
    if isinstance(expr, _FUNCTIONS):
        argument = _enclose(as_expr(expr.args[0]), x, lo, hi, removable=False)
        if argument is None:
            return None
        return _apply(expr, argument)
    return None


def _apply(f: Expr, v: ivmpf) -> Optional[ivmpf]:
    try:
        if isinstance(f, Exp):
            return _iv.exp(v)
        if isinstance(f, Log):
            if v.a > 0:
                return _iv.log(v)
            if v.a == 0 and v.b > 0:
                return _iv.mpf([-mpmath.inf, _iv.log(_iv.mpf(v.b)).b])
            return None
        if isinstance(f, Sin):
            return _iv.sin(v)
        if isinstance(f, Cos):
            return _iv.cos(v)
        if isinstance(f, Tan):
            return _iv.tan(v)
        if isinstance(f, Atan):
            return _iv.atan2(v, _iv.mpf(1))
        if isinstance(f, Sinh):
            return (_iv.exp(v) - _iv.exp(-v)) / 2
        if isinstance(f, Cosh):
            return (_iv.exp(v) + _iv.exp(-v)) / 2
        if isinstance(f, Tanh):
            w = _iv.exp(2 * v)
            return (w - 1) / (w + 1)
        if isinstance(f, Abs):
            return abs(v)
    except (ValueError, ZeroDivisionError, TypeError, mpmath.libmp.libmpf.ComplexResult):
        return None
    return None


def _extended_power(base_: Expr, exponent_: Expr, x: Symbol, lo: Expr, hi: Expr) -> Optional[ivmpf]:
    base = _enclose(base_, x, lo, hi, removable=False)
    if base is None:
        return None
    try:
        n = _integer(exponent_)
        if n is not None:
            if n >= 0:
                return base**n
            if base.a > 0 or base.b < 0:
                return _iv.mpf(1) / base**(-n)
            if base.a == 0 and base.b > 0:
                return _iv.mpf([(_iv.mpf(1) / _iv.mpf(base.b)**(-n)).a, mpmath.inf])
            if base.b == 0 and base.a < 0:
                q = _iv.mpf(1) / _iv.mpf(base.a)**(-n)
                return _iv.mpf([q.a, mpmath.inf]) if n % 2 == 0 else _iv.mpf([-mpmath.inf, q.b])
            return None
        exponent = _enclose(exponent_, x, lo, hi, removable=False)
        if exponent is None:
            return None
        if base.a > 0:
            return _iv.exp(exponent * _iv.log(base))
        if _is_true(exponent_.is_number) and _is_true(exponent_.is_extended_real):
            if base.a == 0 and base.b > 0:
                at_top = _iv.exp(exponent * _iv.log(_iv.mpf(base.b)))
                if exponent.a > 0:
                    return _iv.mpf([0, at_top.b])
                if exponent.b < 0:
                    return _iv.mpf([at_top.a, mpmath.inf])
            if base.a == 0 and base.b == 0 and exponent.a > 0:
                return _iv.mpf(0)
            return None
        # a nonnegative base and a nonnegative exponent: the power
        # increases with the base, and 1 covers 0**0
        if base.a < 0 or exponent.a < 0:
            return None
        top = _iv.mpf(1) if exponent.a == 0 else _iv.mpf(0)
        if base.b > 0:
            for e in (exponent.a, exponent.b):
                candidate = _iv.exp(_iv.mpf(e) * _iv.log(_iv.mpf(base.b)))
                top = _iv.mpf([0, max(top.b, candidate.b)])
        return _iv.mpf([0, top.b])
    except (ValueError, ZeroDivisionError, TypeError, mpmath.libmp.libmpf.ComplexResult):
        return None


_removable_forms: dict[tuple[Expr, Symbol, Expr], Optional[Expr]] = {}


def _removable_form(expr: Expr, x: Symbol, c: Expr) -> Optional[Expr]:
    """``g**(n)/n!`` when ``expr = g/(x - c)**n`` with ``g`` and its first
    ``n - 1`` derivatives vanishing at ``c``: by Taylor's theorem with the
    Lagrange remainder, ``expr(x) = g**(n)(xi)/n!`` for some ``xi``
    between ``c`` and ``x``."""
    key = (expr, x, c)
    if key in _removable_forms:
        return _removable_forms[key]
    if len(_removable_forms) > 2000:
        _removable_forms.clear()
    _removable_forms[key] = form = attempt(lambda: _find_removable_form(expr, x, c), _budget(), native=True)
    return form


def _at(e: Expr, x: Symbol, c: Expr) -> Expr:
    return as_expr(e.subs(x, c))


def _find_removable_form(expr: Expr, x: Symbol, c: Expr) -> Optional[Expr]:
    numerator, denominator = expr.as_numer_denom()
    if _at(denominator, x, c) != 0:
        return None
    p = _polynomial(denominator, x)
    if p is None:
        return None
    n = 0
    while _degree(p) > 0:
        quotient, remainder = _divided(p, c)
        if remainder != 0:
            break
        p = quotient
        n += 1
    if n == 0:
        return None
    derivative = _cancelled(numerator / _expr_of(p, x))
    # the derivatives one after the other (each from the one before)
    for i in range(n):
        if _at(derivative, x, c) != 0:
            return None
        derivative = derivative.diff(x)
    return derivative / math.factorial(n)


def _cancelled(e: Expr) -> Expr:
    """``cancel(e)``, or ``e`` itself when the coefficients are in a
    domain the typed polynomials do not have (``pi``, ``E``, ``sqrt(2)``:
    SymPy's ``EX``)."""
    try:
        return cancel(e)
    except BasePolynomialError:
        return e


def _removable(expr: Expr, x: Symbol, lo: Expr, hi: Expr) -> Optional[ivmpf]:
    if not isinstance(expr, (Mul, Pow, Add)) or _out_of_time():
        return None
    for c in (lo, hi):
        form = _removable_form(expr, x, c)
        if form is not None:
            value = _enclose(form, x, lo, hi, removable=False)
            if _finite(value):
                return value
    return None


# ---------------------------------------------------------------------------
# the Gauss–Legendre rule
# ---------------------------------------------------------------------------

class _Rule:
    """The ``n``-point Gauss–Legendre rule on ``[0, 1]``: the nodes as
    exact rational intervals, the weights as intervals, and the constant
    ``(n!)**4/((2n + 1) ((2n)!)**3)`` of the error bound."""

    def __init__(self, n: int, nodes: list[tuple[Rational, Rational]], weights: list[ivmpf]) -> None:
        self.n: Final[int] = n
        self.nodes: Final[list[tuple[Rational, Rational]]] = nodes
        self.weights: Final[list[ivmpf]] = weights
        self.order: Final[int] = 2 * n
        self.constant: Final[ivmpf] = _interval(Rational(math.factorial(n)**4,
                                                         (2 * n + 1) * math.factorial(2 * n)**3))


_rules: dict[tuple[int, int], Optional[_Rule]] = {}


def _legendre(n: int, t: ivmpf) -> tuple[ivmpf, ivmpf]:
    """``P_n(t)`` and ``P_(n-1)(t)`` by the three-term recurrence."""
    previous, current = _iv.mpf(1), t
    for k in range(1, n):
        previous, current = current, ((2 * k + 1) * t * current - k * previous) / (k + 1)
    return current, previous


def _gauss_rule(n: int) -> Optional[_Rule]:
    key = (n, int(_iv.dps))
    if key not in _rules:
        _rules[key] = _make_rule(n)
    return _rules[key]


def _make_rule(n: int) -> Optional[_Rule]:
    """The nodes found by Newton's method at a higher precision, each
    enclosed in an interval of radius ``10**-(dps + 12)`` verified by a
    sign change of ``P_n`` (in interval arithmetic at a precision raised
    by twenty digits, which the enclosures keep at the working
    precision); ``None`` when a verification fails."""
    nodes: list[tuple[Rational, Rational]] = []
    weights: list[ivmpf] = []
    dps = _iv.dps
    half = Rational(1, 2)
    try:
        _iv.dps = dps + 20
        with mpmath.mp.workdps(dps + 20):
            delta = mpmath.mpf(10)**(-(dps + 12))
            for i in range(1, n + 1):
                guess = mpmath.cos(mpmath.pi * (4 * i - 1) / (4 * n + 2))
                root = mpmath.findroot(lambda t: mpmath.legendre(n, t), guess)
                lo, hi = _exact(root - delta), _exact(root + delta)
                left, right = _legendre(n, _interval(lo))[0], _legendre(n, _interval(hi))[0]
                if not (left.b < 0 < right.a or right.b < 0 < left.a):
                    return None
                t = _iv.mpf([_interval(lo).a, _interval(hi).b])
                pn, pn1 = _legendre(n, t)
                derivative = n * (t * pn - pn1) / (t**2 - 1)
                weights.append(2 / ((1 - t**2) * derivative**2))
                nodes.append((_scaled(_sum(Integer(1), lo), half), _scaled(_sum(Integer(1), hi), half)))
    finally:
        _iv.dps = dps
    ordered = sorted(nodes, key=lambda node: (node[0].as_fraction(), node[1].as_fraction()))
    if any(ordered[i][1].as_fraction() >= ordered[i + 1][0].as_fraction() for i in range(len(ordered) - 1)):
        return None
    return _Rule(n, nodes, weights)


# ---------------------------------------------------------------------------
# the endpoint model
# ---------------------------------------------------------------------------

#: a term ``c(v) v**alpha log(v)**j`` of the model: the enclosure of
#: ``alpha``, ``j`` and the coefficient ``c``
ModelTerm = tuple[ivmpf, int, Expr]


class _Model:
    """The integrand near an endpoint as a sum of terms ``c(v) v**alpha
    log(v)**j`` with ``alpha > -1`` and ``c`` bounded, ``v`` the distance
    from the endpoint, which bounds the integral over ``[0, h]``."""

    def __init__(self, v: Symbol, terms: list[ModelTerm]) -> None:
        self.v: Final[Symbol] = v
        self.terms: Final[list[ModelTerm]] = terms

    def bound(self, h: Rational) -> Optional[mpmath.mpf]:
        if h.as_fraction() >= 1:
            return None
        width = _interval(h)
        total = _iv.mpf(0)
        for alpha, j, coefficient in self.terms:
            c = _enclose(coefficient, self.v, S.Zero, h)
            if not _finite(c):
                return None
            total = total + abs(c) * _model_integral(alpha, j, width)
        return mpmath.mpf(total.b)


def _model_integral(alpha: ivmpf, j: int, h: ivmpf) -> ivmpf:
    """``Integral(v**alpha*(-log(v))**j, (v, 0, h))`` for ``h < 1`` by the
    recurrence ``J_j = h**(alpha + 1) (-log h)**j/(alpha + 1) + j J_(j-1)/(alpha + 1)``."""
    a1 = alpha + 1
    power = _iv.exp(a1 * _iv.log(h))
    minus_log = -_iv.log(h)
    value = power / a1
    for k in range(1, j + 1):
        value = power * minus_log**k / a1 + k * value / a1
    return value


def _model(f: Expr, x: Symbol, end: Rational, side: str) -> Optional[_Model]:
    v = Dummy('v', positive=True)
    g = _at(f, x, end + v if side == 'left' else end - v)
    expanded = attempt(lambda: expand(g), _budget(), native=True)
    if expanded is not None:
        g = expanded
    log_v = log(v)
    terms: list[ModelTerm] = []
    for term in Add.make_args(g):
        alpha: Expr = S.Zero
        j = 0
        rest: list[Expr] = []
        for factor_ in Mul.make_args(term):
            base: Expr = factor_
            exponent: Expr = S.One
            if isinstance(factor_, Pow):
                base, exponent = factor_.base, factor_.exp
            n = _integer(exponent)
            if base == v and _is_true(exponent.is_number) and _is_true(exponent.is_extended_real):
                alpha = alpha + exponent
            elif base == log_v and n is not None and n > 0:
                j += n
            else:
                rest.append(factor_)
        if not _is_true((alpha + 1).is_positive):
            return None
        alpha_ = _enclose(alpha, v, S.Zero, S.Zero)
        if alpha_ is None:
            return None
        terms.append((alpha_, j, mul(*rest)))
    return _Model(v, terms)


# ---------------------------------------------------------------------------
# centred forms
# ---------------------------------------------------------------------------

class _Form:
    """An expression as ``N(x)/D(x)`` with ``N`` and ``D`` polynomials in
    ``x`` whose coefficients may contain the transcendental
    subexpressions (``exp(x**2)``), evaluated on a panel by the Taylor
    shift of the coefficients to the midpoint and a sum in the offset:
    interval arithmetic loses much less to the dependency problem than on
    the expression tree (the tenth derivative of ``1/(2x**2 - 2x + 1)``
    is a quotient of polynomials of degrees 10 and 22 whose naive
    enclosure is wider by many orders of magnitude)."""

    def __init__(self, x: Symbol, numerator: Coefficients, denominator: Coefficients) -> None:
        self.x: Final[Symbol] = x
        #: the coefficients, lowest degree first
        self.numerator: Final[Coefficients] = numerator
        self.denominator: Final[Coefficients] = denominator

    def evaluate(self, lo: Rational, hi: Rational) -> Optional[ivmpf]:
        mid = _interval(_mid(lo, hi))
        radius = _interval(_difference(hi, lo)) / 2
        offset = _iv.mpf([-radius.b, radius.b])
        values: list[ivmpf] = []
        for coefficients in (self.numerator, self.denominator):
            enclosed: list[ivmpf] = []
            for c in coefficients:
                value = _enclose(c, self.x, lo, hi)
                if not _finite(value):
                    return None
                enclosed.append(value)
            values.append(_sum_of_powers(_shifted(enclosed, mid), offset))
        n, d = values
        if d.a <= 0 <= d.b:
            return None
        return n / d


def _shifted(coefficients: list[ivmpf], mid: ivmpf) -> list[ivmpf]:
    """The coefficients of ``p(mid + d)`` in ``d`` (Ruffini–Horner)."""
    a = list(coefficients)
    for i in range(len(a) - 1):
        for j in range(len(a) - 2, i - 1, -1):
            a[j] = a[j] + mid * a[j + 1]
    return a


def _sum_of_powers(coefficients: list[ivmpf], offset: ivmpf) -> ivmpf:
    total = _iv.mpf(0)
    for k, c in enumerate(coefficients):
        total = total + c * offset**k
    return total


def _hidden(expr: Expr, x: Symbol, back: dict[Basic, Basic]) -> Expr:
    """``expr`` with its transcendental subexpressions in ``x`` replaced
    by dummies (recorded in ``back``), a polynomial in ``x``."""
    if x not in expr.free_symbols:
        return expr
    if expr == x:
        return expr
    if isinstance(expr, Add):
        return add(*[_hidden(as_expr(a), x, back) for a in expr.args])
    if isinstance(expr, Mul):
        return mul(*[_hidden(as_expr(a), x, back) for a in expr.args])
    if isinstance(expr, Pow):
        n = _integer(expr.exp)
        if n is not None and n > 0:
            return pow(_hidden(expr.base, x, back), expr.exp)
    dummy = Dummy('g')
    back[dummy] = expr
    return dummy


def _form(expr: Expr, x: Symbol) -> Optional[_Form]:
    """The centred form of ``expr`` within the budget of a preparation,
    ``None`` when it is not a quotient of polynomials in ``x`` with
    transcendental coefficients or the budget is spent."""
    return attempt(lambda: _make_form(expr, x), _budget(), native=True)


def _make_form(expr: Expr, x: Symbol) -> Optional[_Form]:
    combined = together(expr)
    parts: list[Coefficients] = []
    for part in combined.as_numer_denom():
        back: dict[Basic, Basic] = {}
        p = _polynomial(_hidden(part, x, back), x)
        if p is None or _degree(p) > 80:
            return None
        # the rebuilt coefficients evaluated, as SymPy's xreplace does
        parts.append([as_expr(c._xreplace_evaluated(back)) for c in p])
    return _Form(x, parts[0], parts[1])


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

class _Problem:
    """An integrand on ``[a, b]`` with the derivatives of the bounds, the
    rule, and the endpoint models (made when first needed)."""

    def __init__(self, f: Expr, x: Symbol, a: Rational, b: Rational) -> None:
        self.f: Final[Expr] = f
        self.x: Final[Symbol] = x
        self.a: Final[Rational] = a
        self.b: Final[Rational] = b
        self.fourth: Final[Optional[Expr]] = attempt(lambda: f.diff(x, 4), _budget(), native=True)
        self.rule: Final[Optional[_Rule]] = _gauss_rule(_GAUSS_NODES)
        high: Optional[Expr] = None
        if self.rule is not None and not _out_of_time():
            derivative = attempt(lambda: f.diff(x, 2 * _GAUSS_NODES), _budget(), native=True)
            if derivative is not None and _ops(derivative) <= _MAX_OPS:
                high = derivative
        self.high: Final[Optional[Expr]] = high
        self.forms: Final[dict[Expr, Optional[_Form]]] = {}
        for expr in (f, self.fourth, self.high):
            if expr is not None and not _out_of_time():
                self.forms[expr] = _form(expr, x)
        self.models: Final[dict[str, Optional[_Model]]] = {}

    def bound(self, expr: Expr, lo: Rational, hi: Rational) -> Optional[ivmpf]:
        """An enclosure of ``expr`` on the panel, through its centred
        form when it has one and the form is finite there."""
        form = self.forms.get(expr)
        if form is not None:
            value = form.evaluate(lo, hi)
            if _finite(value):
                return value
        return _enclose(expr, self.x, lo, hi)

    def model(self, side: str) -> Optional[_Model]:
        if _out_of_time():
            return None
        if side not in self.models:
            self.models[side] = _model(self.f, self.x, self.a if side == 'left' else self.b, side)
        return self.models[side]


class _Panel:
    """A panel of the subdivision with its enclosure and error bound."""

    def __init__(self, lo: Rational, hi: Rational, value: ivmpf, error: mpmath.mpf) -> None:
        self.lo: Final[Rational] = lo
        self.hi: Final[Rational] = hi
        self.value: Final[ivmpf] = value
        self.error: Final[mpmath.mpf] = error
        #: what splitting the panel can gain: the bound and the radius of
        #: the enclosure of the value
        self.priority: Final[mpmath.mpf] = (error + mpmath.mpf(value.delta) / 2
                                            if mpmath.isfinite(error) else error)

    def __lt__(self, other: _Panel) -> bool:
        return bool(self.priority > other.priority)    # the largest first


def _simpson_value(problem: _Problem, lo: Rational, hi: Rational) -> Optional[ivmpf]:
    values = [_enclose(problem.f, problem.x, p, p) for p in (lo, _mid(lo, hi), hi)]
    finite = [v for v in values if _finite(v)]
    if len(finite) != 3:
        return None
    f0, f1, f2 = finite
    return _interval(_difference(hi, lo)) / 6 * (f0 + 4 * f1 + f2)


def _gauss_value(problem: _Problem, lo: Rational, hi: Rational) -> Optional[ivmpf]:
    if problem.rule is None:
        return None
    width = _difference(hi, lo)
    total = _iv.mpf(0)
    for (node_lo, node_hi), weight in zip(problem.rule.nodes, problem.rule.weights):
        value = _enclose(problem.f, problem.x, _sum(lo, _scaled(width, node_lo)),
                         _sum(lo, _scaled(width, node_hi)))
        if not _finite(value):
            return None
        total = total + weight * value
    return _interval(width) / 2 * total


#: a quadrature rule on a panel
PanelRule = Callable[[_Problem, Rational, Rational], Optional[ivmpf]]


def _panel(problem: _Problem, lo: Rational, hi: Rational) -> _Panel:
    """The panel with the rule of the smaller bound; the range bound when
    no derivative has a finite enclosure; the endpoint model at an
    endpoint where even the range has none; an infinite error otherwise
    (the panel is then split first: interval arithmetic loses its
    dependency problem on narrow panels)."""
    width = _interval(_difference(hi, lo))
    candidates: list[tuple[mpmath.mpf, PanelRule]] = []
    if problem.high is not None and problem.rule is not None:
        bound = problem.bound(problem.high, lo, hi)
        if _finite(bound):
            error = mpmath.mpf((width**(problem.rule.order + 1) * problem.rule.constant * abs(bound)).b)
            candidates.append((error, _gauss_value))
    bound = None if problem.fourth is None else problem.bound(problem.fourth, lo, hi)
    if _finite(bound):
        candidates.append((mpmath.mpf((width**5 / 2880 * abs(bound)).b), _simpson_value))
    for error, rule in sorted(candidates, key=lambda c: c[0]):
        value = rule(problem, lo, hi)
        if value is not None:
            return _Panel(lo, hi, value, error)
    values = problem.bound(problem.f, lo, hi)
    if _finite(values):
        return _Panel(lo, hi, width * values, mpmath.mpf(0))
    for side, end, touching in (('left', problem.a, lo), ('right', problem.b, hi)):
        if touching == end:
            model = problem.model(side)
            if model is not None:
                bounded = model.bound(_difference(hi, lo))
                if bounded is not None:
                    return _Panel(lo, hi, _iv.mpf(0), bounded)
    return _Panel(lo, hi, _iv.mpf(0), mpmath.inf)


class _Subdivision:
    """The heap of panels with a running estimate of the radius of the
    enclosure of the integral (the sum of the bounds and of the radii of
    the panels, in floating point: the exact sums are recomputed when the
    estimate says the tolerance is met), and the number of panels without
    a finite bound."""

    def __init__(self) -> None:
        self.heap: Final[list[_Panel]] = []
        self.estimate: mpmath.mpf = mpmath.mpf(0)
        self.unbounded: int = 0

    def push(self, panel: _Panel) -> None:
        heapq.heappush(self.heap, panel)
        if mpmath.isfinite(panel.error):
            self.estimate += panel.priority
        else:
            self.unbounded += 1

    def pop(self) -> _Panel:
        panel = heapq.heappop(self.heap)
        if mpmath.isfinite(panel.error):
            self.estimate -= panel.priority
        else:
            self.unbounded -= 1
        return panel

    def exact(self) -> tuple[ivmpf, mpmath.mpf]:
        """The sum of the enclosures and the sum of the bounds, in interval
        arithmetic; the estimate is reset from them."""
        total = _iv.mpf(0)
        spread = mpmath.mpf(0)
        for panel in self.heap:
            total = total + panel.value
            spread += panel.error
        self.estimate = mpmath.mpf(total.delta) / 2 + spread
        return total, spread


def enclosure(f: Expr, x: Symbol, a: Rational, b: Rational, tolerance: mpmath.mpf,
              max_panels: int = _MAX_PANELS) -> Optional[ivmpf]:
    """An interval containing ``Integral(f, (x, a, b))`` for rational
    ``a < b``, refined until its width is below ``tolerance`` or the panel
    budget or the time limit (``settings.timeout``, counted from the call
    of :func:`validated_integral`) is spent; ``None``
    when a panel narrower than the tolerance still cannot be bounded (a
    non-integrable singularity, or a function the interval evaluator does
    not know).

    Examples
    ========

    >>> import mpmath
    >>> from sympy_extras.core import Rational, symbols
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.validated import enclosure
    >>> x = symbols('x')
    >>> r = enclosure(log(1 + x), x, Rational(0), Rational(1), mpmath.mpf('1e-12'))
    >>> r.a < 2*mpmath.log(2) - 1 < r.b, r.delta < 1e-12
    (True, True)
    """
    problem = _Problem(f, x, a, b)
    panels = _Subdivision()
    panels.push(_panel(problem, a, b))
    iterations = 0
    span = float(b.as_fraction() - a.as_fraction())
    while len(panels.heap) < max_panels:
        # the estimate drifts by the rounding of the large early bounds
        # it has subtracted, so it is reset now and then
        if not panels.unbounded and (panels.estimate <= tolerance or iterations % 64 == 0):
            total, spread = panels.exact()
            if mpmath.mpf(total.delta) / 2 + spread <= tolerance:
                break
        iterations += 1
        if _out_of_time():
            break
        worst = panels.pop()
        if (not mpmath.isfinite(worst.error)
                and float(worst.hi.as_fraction() - worst.lo.as_fraction()) < float(tolerance) * span):
            return None
        mid_ = _mid(worst.lo, worst.hi)
        panels.push(_panel(problem, worst.lo, mid_))
        panels.push(_panel(problem, mid_, worst.hi))
    if panels.unbounded:
        return None
    total, spread = panels.exact()
    return _iv.mpf([total.a - spread, total.b + spread])


# ---------------------------------------------------------------------------
# substitutions
# ---------------------------------------------------------------------------

def _denest(g: Expr) -> Expr:
    """Powers with rational exponents of a factored base, denested: the
    substitution leaves ``sqrt(1 - (1 - w**2)**2)``, which is
    ``w*sqrt(2 - w**2)``."""

    def factored(node: Basic) -> Basic:
        if not isinstance(node, Pow):
            return node
        return pow(factor(expand(node.base)), node.exp)

    def is_radical(node: Basic) -> bool:
        return isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q > 1

    replaced = as_expr(g.replace(is_radical, factored))
    return powdenest(replaced, force=True)


def _remove_power_singularity(f: Expr, x: Symbol, a: Rational, b: Rational,
                              side: str) -> Optional[tuple[Expr, Symbol]]:
    """``f`` with ``x = a + (b - a) u**k`` (``dx = (b - a) k u**(k-1) du``;
    at the right end ``x = b - (b - a) u**k``) on ``u`` in ``[0, 1]``
    when it has powers with rational exponents ``p/q > -1`` of bases
    vanishing at that end, ``k`` the least common multiple of the
    denominators ``q``."""
    end = a if side == 'left' else b
    denominators: set[int] = set()
    for node in f.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        exponent = node.exp
        if (isinstance(exponent, Rational) and 1 < exponent.q <= 12 and exponent.as_fraction() > -1
                and _at(node.base, x, end) == 0):
            denominators.add(int(exponent.q))
    if not denominators:
        return None
    k = math.lcm(*denominators)
    u = Dummy('u', positive=True)
    scale = _difference(b, a)
    image = a + scale * u**k if side == 'left' else b - scale * u**k
    g = _at(f, x, image) * scale * k * u**(k - 1)
    denested = attempt(lambda: _denest(g), _budget(), native=True)
    return (denested if denested is not None else g), u


def _desingularised(f: Expr, x: Symbol, lo: Rational, hi: Rational) -> tuple[Expr, Symbol, Rational, Rational]:
    g, u, a, b = f, x, lo, hi
    for side in ('left', 'right'):
        transformed = _remove_power_singularity(g, u, a, b, side)
        if transformed is not None:
            g, u = transformed
            a, b = Integer(0), Integer(1)
    return g, u, a, b


def _rounded(value: mpmath.mpf) -> Rational:
    """``value`` rounded to 40 digits, as the exact rational it is."""
    with mpmath.mp.workprec(dps_to_prec(40)):
        return _exact(+mpmath.mpf(value))


#: rational bounds inside the range and the strips at irrational endpoints
Bounds = tuple[Rational, Rational, list[tuple[Rational, Rational]]]


def _bounds(a: Expr, b: Expr) -> Optional[Bounds]:
    """Rational bounds inside ``[a, b]`` and the strips left over at
    irrational endpoints (``pi``), each a rational interval containing
    the endpoint."""
    strips: list[tuple[Rational, Rational]] = []
    lo, hi = _rational(a), _rational(b)
    if lo is None:
        value = _enclose(a, Dummy('x'), Integer(0), Integer(0))
        if value is None:
            return None
        lo = _rounded(value.b)
        strips.append((_rounded(value.a), lo))
    if hi is None:
        value = _enclose(b, Dummy('x'), Integer(0), Integer(0))
        if value is None:
            return None
        hi = _rounded(value.a)
        strips.append((hi, _rounded(value.b)))
    return lo, hi, strips


def _enclosed(f: Expr, x: Symbol, a: Expr, b: Expr, tolerance: mpmath.mpf) -> Optional[ivmpf]:
    """An enclosure of ``Integral(f, (x, a, b))`` for any bounds."""
    if a == b:
        return _iv.mpf(0)
    if _out_of_time():
        return None
    if a in (oo, -oo) or b in (oo, -oo):
        return _infinite(f, x, a, b, tolerance)
    bounds = _bounds(a, b)
    if bounds is None:
        return None
    lo, hi, strips = bounds
    if lo.as_fraction() > hi.as_fraction():
        reversed_ = _enclosed(f, x, b, a, tolerance)
        return None if reversed_ is None else -reversed_
    g, u, lo_, hi_ = _desingularised(f, x, lo, hi)
    result = enclosure(g, u, lo_, hi_, tolerance)
    if result is None and g is not f:
        result = enclosure(f, x, lo, hi, tolerance)
    if result is None:
        return None
    for strip_lo, strip_hi in strips:
        values = _enclose(f, x, strip_lo, strip_hi)
        if not _finite(values):
            return None
        result = result + _interval(_difference(strip_hi, strip_lo)) * values
    return result


# ---------------------------------------------------------------------------
# infinite ranges
# ---------------------------------------------------------------------------

def _tidy(g: Expr) -> Expr:
    """The transformed integrand with the exponentials of the
    substitution expanded and cancelled (``exp(-x)`` under ``x = -log(1 -
    t)`` is ``1 - t``)."""
    tidied = attempt(lambda: cancel(expand_power_exp(g)), _budget(), native=True)
    return g if tidied is None else tidied


def _rational_map(f: Expr, x: Symbol, a: Expr, tolerance: mpmath.mpf) -> Optional[ivmpf]:
    """``x = a + t/(1 - t)`` on ``[0, 1]``: algebraic decay."""
    t = Dummy('t', positive=True)
    g = _at(f, x, a + t / (1 - t)) / (1 - t)**2
    simpler = attempt(lambda: simplify(cancel(g)), _budget(), native=True)
    return _enclosed(simpler if simpler is not None else g, t, S.Zero, S.One, tolerance)


def _exponential_map(f: Expr, x: Symbol, a: Expr, tolerance: mpmath.mpf) -> Optional[ivmpf]:
    """``x = a - log(1 - t)`` on ``[0, 1]``: exponential decay."""
    t = Dummy('t', positive=True)
    g = _tidy(_at(f, x, a - log(1 - t)) / (1 - t))
    return _enclosed(g, t, S.Zero, S.One, tolerance)


def _ceiling(a: Expr) -> Optional[int]:
    """The ceiling of the number ``a`` as a Python integer, ``None`` when
    it is not found."""
    exact = _rational(a)
    if exact is not None:
        return math.ceil(exact.as_fraction())
    return _integer(ceiling(a))


def _stretched(f: Expr, x: Symbol, a: Expr, tolerance: mpmath.mpf) -> Optional[ivmpf]:
    """A decay ``exp(-c x**k)``, ``k >= 2``: the range cut at ``C =
    max(1, ceiling(a))``, and ``x = s**(1/k)`` on ``[C, oo)`` makes the
    decay exponential in ``s``."""
    degrees: set[int] = set()
    for node in f.atoms(Exp):
        if not isinstance(node, Exp):
            continue
        argument = as_expr(node.args[0])
        if x not in argument.free_symbols:
            continue
        p = _polynomial(argument, x)
        if p is None or _degree(p) < 2 or not _is_true(p[-1].is_negative):
            return None
        degrees.add(_degree(p))
    if len(degrees) != 1:
        return None
    k = degrees.pop()
    start = _ceiling(a)
    if start is None:
        return None
    cut = Integer(max(start, 1))
    finite = _enclosed(f, x, a, cut, tolerance / 2)
    if finite is None:
        return None
    s = Dummy('s', positive=True)
    g = _at(f, x, s**Rational(1, k)) * s**(Rational(1, k) - 1) / k
    tail = _exponential_map(g, s, cut**k, tolerance / 2)
    return None if tail is None else finite + tail


def _trigonometric_antiderivative(e: Expr, x: Symbol) -> Optional[Expr]:
    """An antiderivative of ``c sin(w x + d)`` or ``c cos(w x + d)``,
    ``c`` and ``w`` free of ``x``: ``-c cos(w x + d)/w`` and ``c sin(w x +
    d)/w``; ``None`` for any other expression."""
    rest: list[Expr] = []
    found: Optional[Expr] = None
    for factor_ in Mul.make_args(e):
        if isinstance(factor_, (Sin, Cos)) and found is None:
            found = factor_
        elif x in factor_.free_symbols:
            return None
        else:
            rest.append(factor_)
    if found is None:
        return None
    argument = as_expr(found.args[0])
    p = _polynomial(argument, x)
    if p is None or _degree(p) != 1:
        return None
    c = mul(*rest) / p[1]
    if isinstance(found, Sin):
        return -c * cos(argument)
    return c * sin(argument)


def _oscillatory(f: Expr, x: Symbol, a: Expr, tolerance: mpmath.mpf) -> Optional[ivmpf]:
    """``f = g(x) sin(w x + c)`` (or ``cos``): the tail ``Integral(f, (x,
    A, oo))`` integrated by parts ``k`` times,

        sum((-1)**(i + 1) g**(i)(A) Phi_(i+1)(A), i = 0..k-1) + (-1)**k Integral(g**(k) Phi_k),

    ``Phi_i`` the ``i``-th antiderivative of the trigonometric factor,
    ``|Phi_i| <= 1/w**i``. When ``g**(k)`` is monotone with a constant
    sign on ``[A, oo)`` and ``g, ..., g**(k)`` vanish at infinity, the
    last integral is bounded by ``2 |g**(k)(A)|/w**(k + 1)`` (Bonnet's
    mean value theorem); ``A`` and ``k`` are the first pair that brings
    it below half the tolerance."""
    factors = list(Mul.make_args(f))
    trigonometric = [i for i, factor_ in enumerate(factors) if isinstance(factor_, (Sin, Cos))]
    if len(trigonometric) != 1:
        return None
    phi = factors.pop(trigonometric[0])
    g = mul(*factors)
    if g.has(Sin, Cos, Tan):
        return None
    argument = as_expr(phi.args[0])
    p = _polynomial(argument, x)
    if p is None or _degree(p) != 1:
        return None
    w = p[1]
    if not _is_true(w.is_number):
        return None
    if _is_true(w.is_negative):
        # sin(-y) = -sin(y), cos(-y) = cos(y), the nodes kept unevaluated
        if isinstance(phi, Sin):
            phi, g = Sin(-argument), -g
        else:
            phi = Cos(-argument)
        w = -w
    w_ = _enclose(w, x, S.Zero, S.Zero)
    if not _finite(w_) or w_.a <= 0:
        return None
    derivatives = [g]
    antiderivatives = [phi]
    start = _ceiling(a)
    if start is None:
        return None
    for m in range(24):
        A = Integer((start + 1) * 2**m)
        for k in range(_MAX_PARTS + 1):
            if _out_of_time():
                return None
            while len(derivatives) <= k + 1:
                derivatives.append(derivatives[-1].diff(x))
            while len(antiderivatives) <= k:
                antiderivative = _trigonometric_antiderivative(antiderivatives[-1], x)
                if antiderivative is None:
                    return None
                antiderivatives.append(antiderivative)
            at_a = _enclose(derivatives[k], x, A, A)
            if not _finite(at_a):
                break
            remainder = mpmath.mpf((2 * abs(at_a) / w_**(k + 1)).b)
            if remainder > tolerance / 2:
                continue
            if not _monotone_to_zero(derivatives[:k + 2], x, A):
                continue
            tail = _iv.mpf(0)
            for i in range(k):
                term = _enclose(Integer((-1)**(i + 1)) * derivatives[i] * antiderivatives[i + 1], x, A, A)
                if not _finite(term):
                    return None
                tail = tail + term
            finite = _enclosed(f, x, a, A, tolerance / 2)
            if finite is None:
                return None
            return finite + _iv.mpf([tail.a - remainder, tail.b + remainder])
    return None


def _sign_on_ray(expr: Expr, x: Symbol, A: Integer) -> Optional[int]:
    """The sign of ``expr`` on ``[A, oo)``: the polynomial factors in
    ``x`` of its numerator and denominator have no root there by Sturm's
    theorem and so the sign of their value at ``A``, the other factors a
    sign read from their enclosure on the ray; ``1`` or ``-1`` for a
    constant sign, ``0`` for the zero expression, ``None`` when
    undecided."""
    if expr == 0:
        return 0
    return attempt(lambda: _find_sign_on_ray(expr, x, A), _budget(), native=True)


def _roots_on_ray(p: list[Fraction], A: Fraction) -> int:
    """The number of distinct roots of ``p`` (highest degree first) in
    ``[A, oo)``."""
    at_a = Fraction(0)
    for c in p:
        at_a = at_a * A + c
    return dup_count_real_roots(p, A) + (1 if at_a == 0 and len(p) > 1 else 0)


def _find_sign_on_ray(expr: Expr, x: Symbol, A: Integer) -> Optional[int]:
    combined = together(expr)
    sign = 1
    for part in combined.as_numer_denom():
        for factor_ in Mul.make_args(part):
            base, exponent = factor_, 1
            if isinstance(factor_, Pow):
                n = _integer(factor_.exp)
                if n is not None:
                    base, exponent = factor_.base, n
            factor_sign: Optional[int] = None
            if x not in base.free_symbols:
                enclosed = _enclose(base, x, S.Zero, S.Zero)
                factor_sign = None if enclosed is None else _interval_sign(enclosed)
            else:
                p = _polynomial(base, x)
                rational = None if p is None else _rational_coefficients(p)
                if rational is not None:
                    if _roots_on_ray(rational, A.as_fraction()) != 0:
                        return None
                    at_a = _enclose(base, x, A, A)
                    factor_sign = None if at_a is None else _interval_sign(at_a)
                else:
                    enclosed = _enclose(base, x, A, oo)
                    factor_sign = None if enclosed is None else _interval_sign(enclosed)
            if factor_sign is None or factor_sign == 0:
                return None
            if exponent % 2:
                sign *= factor_sign
    return sign


def _interval_sign(value: ivmpf) -> Optional[int]:
    if value.a > 0:
        return 1
    if value.b < 0:
        return -1
    return None


def _monotone_to_zero(derivatives: list[Expr], x: Symbol, A: Integer) -> bool:
    """Whether the last derivative but one has a constant sign and the
    last the opposite sign (or vanishes) on ``[A, oo)``, and all of them
    vanish at infinity."""
    last, next_ = _sign_on_ray(derivatives[-2], x, A), _sign_on_ray(derivatives[-1], x, A)
    if last is None or next_ is None or last * next_ > 0:
        return False
    for derivative in derivatives[:-1]:
        at_infinity = attempt(lambda: limit(derivative, x, oo), _budget(), native=True)
        if at_infinity is None or at_infinity != 0:
            return False
    return True


def _infinite(f: Expr, x: Symbol, a: Expr, b: Expr, tolerance: mpmath.mpf) -> Optional[ivmpf]:
    """An infinite range: the whole line cut at 0 and ``(-oo, b]``
    reflected; then the oscillatory tail, the stretched exponential map
    and the two maps onto ``[0, 1]`` are tried in turn."""
    if a == -oo and b == oo:
        left = _infinite(f, x, -oo, S.Zero, tolerance / 2)
        right = _infinite(f, x, S.Zero, oo, tolerance / 2)
        if left is None or right is None:
            return None
        return left + right
    if a == -oo:
        return _infinite(_at(f, x, -x), x, -b, oo, tolerance)
    if a == oo:
        reversed_ = _infinite(f, x, b, oo, tolerance)
        return None if reversed_ is None else -reversed_
    if b != oo:
        return None
    result = _oscillatory(f, x, a, tolerance)
    if result is None and f.has(Exp):
        result = _stretched(f, x, a, tolerance)
    maps: list[Callable[[Expr, Symbol, Expr, mpmath.mpf], Optional[ivmpf]]] = (
        [_exponential_map, _rational_map] if f.has(Exp) else [_rational_map, _exponential_map])
    for mapping in maps:
        if result is not None or _out_of_time():
            break
        result = mapping(f, x, a, tolerance)
    return result


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------

def validated_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, digits: int = 15,
                       assumptions: Assumptions = None) -> Optional[tuple[Float, Float]]:
    """``(value, error)`` with ``|Integral(f, (x, a, b)) - value| <= error``
    proved by interval arithmetic, the error aimed at ``10**(-digits)``
    (it may be larger when the panel budget or the time limit is spent
    first); ``None`` when the integrand has parameters, uses functions
    the interval evaluator does not know, or cannot be bounded on some
    panel (a non-integrable singularity, an infinite range on which no
    transformation makes the integrand bounded). Irrational bounds are
    enclosed and the strips between the enclosures bounded by the range
    of ``f``. The value and the error are floats of ``digits + 3`` and
    of 5 digits.

    Examples
    ========

    >>> from fractions import Fraction
    >>> import mpmath
    >>> from sympy_extras.core import symbols, sqrt, oo
    >>> from sympy_extras.functions import log, sin
    >>> from sympy_extras.integrals.validated import validated_integral
    >>> x = symbols('x')
    >>> value, error = validated_integral(sqrt(x), x, 0, 1)
    >>> abs(value.as_fraction() - Fraction(2, 3)) < error.as_fraction(), float(error) < 1e-15
    (True, True)
    >>> value, error = validated_integral(1/(1 + x**2), x, 0, oo, 20)
    >>> with mpmath.workdps(30):
    ...     abs(mpmath.mpf(str(value)) - mpmath.pi/2) < mpmath.mpf(str(error)), float(error) < 1e-20
    (True, True)
    >>> value, error = validated_integral(log(x), x, 0, 1)
    >>> abs(value.as_fraction() + 1) < error.as_fraction()
    True
    >>> value, error = validated_integral(sin(x)/x, x, 0, oo, 8)
    >>> abs(float(value) - float(mpmath.pi/2)) < float(error), float(error) < 1e-8
    (True, True)
    >>> validated_integral(1/x, x, 0, 1) is None
    True
    """
    f_, a_, b_ = to_expr(f), to_expr(a), to_expr(b)
    if f_.free_symbols - {x} or a_.free_symbols or b_.free_symbols:
        return None
    if a_ == b_:
        return (Float(0, digits), Float(0, digits))
    # the enclosure is refined to a quarter of the target, so that the
    # reported bound, with its margins, stays below 10**(-digits)
    tolerance = mpmath.mpf(10)**(-digits) / 4
    global _deadline
    previous, previous_deadline = _iv.dps, _deadline
    try:
        _iv.dps = digits + 10
        if settings.timeout is not None and _deadline is None:
            _deadline = time.monotonic() + settings.timeout
        result = _enclosed(f_, x, a_, b_, tolerance)
        if result is None:
            return None
        # converted while the working precision is still in force
        return _pair(result, digits)
    finally:
        _iv.dps, _deadline = previous, previous_deadline


def _pair(result: ivmpf, digits: int) -> tuple[Float, Float]:
    """The midpoint and the radius of the enclosure as floats, converted
    at the working precision (the default 53 bits would round the
    midpoint outside the enclosure)."""
    with mpmath.mp.workdps(digits + 10):
        value = Float(mpmath.nstr(mpmath.mpf(result.mid.a), digits + 3), digits + 3)
        radius = _iv.mpf(result.delta) / 2
        # the rounding of the value to digits + 3 places, and a margin for
        # the five places of the error itself, are added to the radius
        rounding = mpmath.mpf(abs(result.mid.b)) * mpmath.mpf(10)**(-(digits + 2))
        total = (mpmath.mpf(radius.b) + rounding) * (1 + mpmath.mpf('2e-3'))
        error = Float(mpmath.nstr(total, 5, min_fixed=-100), 5)
    return (value, error)

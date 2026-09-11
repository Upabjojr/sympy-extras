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

Examples
========

>>> from sympy import symbols, exp, sqrt, pi, erf, oo
>>> from sympy_extras.integrals.validated import validated_integral
>>> x = symbols('x')
>>> value, error = validated_integral(exp(-x**2), x, 0, 1)
>>> round(value, 12), error < 1e-15
(0.746824132812, True)
>>> abs(value - sqrt(pi)*erf(1)/2) < error
True
>>> value, error = validated_integral(exp(-x**2), x, 0, oo)
>>> abs(value - sqrt(pi)/2) < error, error < 1e-15
(True, True)

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
from typing import Callable, Optional, TypeGuard

import mpmath
from mpmath.ctx_iv import ivmpf

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import count_ops, expand, expand_power_exp
from sympy.core.mul import Mul
from sympy.core.numbers import Float, Integer, Rational, oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.trigonometric import atan, cos, sin, tan
from sympy.integrals.integrals import integrate
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel, factor
from sympy.polys.rationaltools import together
from sympy.series.limits import limit
from sympy.simplify.powsimp import powdenest
from sympy.simplify.simplify import simplify

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_expr, free_symbols
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.intervals import Box, evaluate
from sympy_extras.settings import settings

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

_FUNCTIONS = (exp, log, sin, cos, tan, atan, sinh, cosh, tanh, Abs)

#: when the whole computation must stop refining (set by
#: :func:`validated_integral` from ``settings.timeout``)
_deadline: Optional[float] = None


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
        return Rational(value)
    return None


def _exact(value: mpmath.mpf) -> Rational:
    """An mpmath float as the exact rational it represents."""
    sign, mantissa, exponent, _ = value._mpf_
    if exponent >= 0:
        result = Rational(int(mantissa) * 2**int(exponent))
    else:
        result = Rational(int(mantissa), 2**(-int(exponent)))
    return -result if sign else result


def _interval(value: Rational) -> ivmpf:
    return _iv.mpf(int(value.p)) / _iv.mpf(int(value.q))


def _sum(lo: Rational, hi: Rational) -> Rational:
    result = _rational(as_expr(lo + hi))
    assert result is not None
    return result


def _difference(hi: Rational, lo: Rational) -> Rational:
    result = _rational(as_expr(hi - lo))
    assert result is not None
    return result


def _scaled(value: Rational, factor_: Rational) -> Rational:
    result = _rational(as_expr(value * factor_))
    assert result is not None
    return result


def _mid(lo: Rational, hi: Rational) -> Rational:
    return _scaled(_sum(lo, hi), Rational(1, 2))


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
    box: Box = {x: (lo, hi)}
    value = evaluate(expr, box)
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
        return _extended_power(as_expr(expr.base), as_expr(expr.exp), x, lo, hi)
    if isinstance(expr, _FUNCTIONS):
        argument = _enclose(as_expr(expr.args[0]), x, lo, hi, removable=False)
        if argument is None:
            return None
        return _apply(expr, argument)
    return None


def _apply(f: Expr, v: ivmpf) -> Optional[ivmpf]:
    try:
        if isinstance(f, exp):
            return _iv.exp(v)
        if isinstance(f, log):
            if v.a > 0:
                return _iv.log(v)
            if v.a == 0 and v.b > 0:
                return _iv.mpf([-mpmath.inf, _iv.log(_iv.mpf(v.b)).b])
            return None
        if isinstance(f, sin):
            return _iv.sin(v)
        if isinstance(f, cos):
            return _iv.cos(v)
        if isinstance(f, tan):
            return _iv.tan(v)
        if isinstance(f, atan):
            return _iv.atan2(v, _iv.mpf(1))
        if isinstance(f, sinh):
            return (_iv.exp(v) - _iv.exp(-v)) / 2
        if isinstance(f, cosh):
            return (_iv.exp(v) + _iv.exp(-v)) / 2
        if isinstance(f, tanh):
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
        if isinstance(exponent_, Integer):
            n = int(exponent_)
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
        if exponent_.is_number and exponent_.is_extended_real:
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


def _polynomial(expr: Expr, x: Symbol) -> Optional[Poly]:
    try:
        return Poly(expr, x)
    except PolynomialError:
        return None


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
    _removable_forms[key] = form = attempt(lambda: _find_removable_form(expr, x, c), _budget())
    return form


def _find_removable_form(expr: Expr, x: Symbol, c: Expr) -> Optional[Expr]:
    numerator, denominator = expr.as_numer_denom()
    if as_expr(denominator).subs(x, c) != 0:
        return None
    p = _polynomial(as_expr(denominator), x)
    if p is None:
        return None
    n = 0
    linear = Poly(x - c, x)
    while p.degree() > 0 and p.eval(c) == 0:
        p = p.quo(linear)
        n += 1
    if n == 0:
        return None
    g = as_expr(cancel(as_expr(numerator) / as_expr(p.as_expr())))
    for i in range(n):
        if as_expr(g.diff(x, i)).subs(x, c) != 0:
            return None
    return as_expr(g.diff(x, n) / math.factorial(n))


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
        self.n = n
        self.nodes = nodes
        self.weights = weights
        self.order = 2 * n
        self.constant = _interval(Rational(math.factorial(n)**4, (2 * n + 1) * math.factorial(2 * n)**3))


_rules: dict[tuple[int, int], Optional[_Rule]] = {}


def _legendre(n: int, t: ivmpf) -> tuple[ivmpf, ivmpf]:
    """``P_n(t)`` and ``P_(n-1)(t)`` by the three-term recurrence."""
    previous, current = _iv.mpf(1), t
    for k in range(1, n):
        previous, current = current, ((2 * k + 1) * t * current - k * previous) / (k + 1)
    return current, previous


def _gauss_rule(n: int) -> Optional[_Rule]:
    key = (n, _iv.dps)
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
                nodes.append((_scaled(_sum(Rational(1), lo), Rational(1, 2)),
                              _scaled(_sum(Rational(1), hi), Rational(1, 2))))
    finally:
        _iv.dps = dps
    ordered = sorted(nodes)
    if any(ordered[i][1] >= ordered[i + 1][0] for i in range(len(ordered) - 1)):
        return None
    return _Rule(n, nodes, weights)


# ---------------------------------------------------------------------------
# the endpoint model
# ---------------------------------------------------------------------------

class _Model:
    """The integrand near an endpoint as a sum of terms ``c(v) v**alpha
    log(v)**j`` with ``alpha > -1`` and ``c`` bounded, ``v`` the distance
    from the endpoint, which bounds the integral over ``[0, h]``."""

    def __init__(self, v: Symbol, terms: list[tuple[ivmpf, int, Expr]]) -> None:
        self.v = v
        self.terms = terms

    def bound(self, h: Rational) -> Optional[mpmath.mpf]:
        if h >= 1:
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
    g = as_expr(f.subs(x, end + v if side == 'left' else end - v))
    expanded = attempt(lambda: as_expr(expand(g)), _budget())
    if expanded is not None:
        g = expanded
    terms: list[tuple[ivmpf, int, Expr]] = []
    for term in Add.make_args(g):
        alpha: Expr = S.Zero
        j = 0
        rest: list[Expr] = []
        for factor_ in Mul.make_args(term):
            base: Expr = as_expr(factor_)
            exponent: Expr = S.One
            if isinstance(factor_, Pow):
                base, exponent = as_expr(factor_.base), as_expr(factor_.exp)
            if base == v and exponent.is_number and exponent.is_extended_real:
                alpha = as_expr(alpha + exponent)
            elif base == log(v) and isinstance(exponent, Integer) and exponent > 0:
                j += int(exponent)
            else:
                rest.append(as_expr(factor_))
        if not (alpha + 1).is_positive:
            return None
        alpha_ = _enclose(alpha, v, S.Zero, S.Zero)
        if alpha_ is None:
            return None
        terms.append((alpha_, j, as_expr(Mul(*rest))))
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

    def __init__(self, x: Symbol, numerator: list[Expr], denominator: list[Expr]) -> None:
        self.x = x
        #: the coefficients, lowest degree first
        self.numerator = numerator
        self.denominator = denominator

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


def _hidden(expr: Expr, x: Symbol, back: dict[Expr, Expr]) -> Expr:
    """``expr`` with its transcendental subexpressions in ``x`` replaced
    by dummies (recorded in ``back``), a polynomial in ``x``."""
    if x not in free_symbols(expr):
        return expr
    if expr == x:
        return expr
    if isinstance(expr, (Add, Mul)):
        return as_expr(expr.func(*[_hidden(as_expr(a), x, back) for a in expr.args]))
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer) and expr.exp > 0:
        return as_expr(Pow(_hidden(as_expr(expr.base), x, back), expr.exp))
    dummy = Dummy('g')
    back[dummy] = expr
    return dummy


def _form(expr: Expr, x: Symbol) -> Optional[_Form]:
    """The centred form of ``expr`` within the budget of a preparation,
    ``None`` when it is not a quotient of polynomials in ``x`` with
    transcendental coefficients or the budget is spent."""
    return attempt(lambda: _make_form(expr, x), _budget())


def _make_form(expr: Expr, x: Symbol) -> Optional[_Form]:
    combined = as_expr(together(expr))
    parts: list[list[Expr]] = []
    for part in combined.as_numer_denom():
        back: dict[Expr, Expr] = {}
        p = _polynomial(_hidden(as_expr(part), x, back), x)
        if p is None or p.degree() > 80:
            return None
        parts.append([as_expr(c).xreplace(back) for c in reversed(p.all_coeffs())])
    return _Form(x, parts[0], parts[1])


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

class _Problem:
    """An integrand on ``[a, b]`` with the derivatives of the bounds, the
    rule, and the endpoint models (made when first needed)."""

    def __init__(self, f: Expr, x: Symbol, a: Rational, b: Rational) -> None:
        self.f = f
        self.x = x
        self.a = a
        self.b = b
        self.fourth = attempt(lambda: as_expr(f.diff(x, 4)), _budget())
        self.rule = _gauss_rule(_GAUSS_NODES)
        self.high: Optional[Expr] = None
        if self.rule is not None and not _out_of_time():
            high = attempt(lambda: as_expr(f.diff(x, 2 * _GAUSS_NODES)), _budget())
            if high is not None and count_ops(high) <= _MAX_OPS:
                self.high = high
        self.forms: dict[Expr, Optional[_Form]] = {}
        for expr in (f, self.fourth, self.high):
            if expr is not None and not _out_of_time():
                self.forms[expr] = _form(expr, x)
        self.models: dict[str, Optional[_Model]] = {}

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
        self.lo = lo
        self.hi = hi
        self.value = value
        self.error = error
        #: what splitting the panel can gain: the bound and the width of
        #: the enclosure of the value
        #: the enclosure of the value (the panel's radius)
        self.priority: mpmath.mpf = error + mpmath.mpf(value.delta) / 2 if mpmath.isfinite(error) else error

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


def _panel(problem: _Problem, lo: Rational, hi: Rational) -> _Panel:
    """The panel with the rule of the smaller bound; the range bound when
    no derivative has a finite enclosure; the endpoint model at an
    endpoint where even the range has none; an infinite error otherwise
    (the panel is then split first: interval arithmetic loses its
    dependency problem on narrow panels)."""
    width = _interval(_difference(hi, lo))
    candidates: list[tuple[mpmath.mpf, Callable[[_Problem, Rational, Rational], Optional[ivmpf]]]] = []
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
        self.heap: list[_Panel] = []
        self.estimate = mpmath.mpf(0)
        self.unbounded = 0

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
    not know)."""
    problem = _Problem(f, x, a, b)
    panels = _Subdivision()
    panels.push(_panel(problem, a, b))
    iterations = 0
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
        if not mpmath.isfinite(worst.error) and float(worst.hi - worst.lo) < float(tolerance) * float(b - a):
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

    def factored(node: Expr) -> Expr:
        if not isinstance(node, Pow):
            return node
        return as_expr(Pow(factor(expand(as_expr(node.base))), node.exp))

    def is_radical(node: Expr) -> bool:
        return isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q > 1

    replaced = as_expr(g.replace(is_radical, factored))
    return as_expr(powdenest(replaced, force=True))


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
        exponent = as_expr(node.exp)
        if (isinstance(exponent, Rational) and 1 < exponent.q <= 12 and exponent > -1
                and as_expr(node.base).subs(x, end) == 0):
            denominators.add(int(exponent.q))
    if not denominators:
        return None
    k = math.lcm(*denominators)
    u = Dummy('u', positive=True)
    scale = _difference(b, a)
    image = a + scale * u**k if side == 'left' else b - scale * u**k
    g = as_expr(f.subs(x, image) * scale * k * u**(k - 1))
    denested = attempt(lambda: _denest(g), _budget())
    return (denested if denested is not None else g), u


def _desingularised(f: Expr, x: Symbol, lo: Rational, hi: Rational) -> tuple[Expr, Symbol, Rational, Rational]:
    g, u, a, b = f, x, lo, hi
    for side in ('left', 'right'):
        transformed = _remove_power_singularity(g, u, a, b, side)
        if transformed is not None:
            g, u = transformed
            a, b = Rational(0), Rational(1)
    return g, u, a, b


def _bounds(a: Expr, b: Expr) -> Optional[tuple[Rational, Rational, list[tuple[Rational, Rational]]]]:
    """Rational bounds inside ``[a, b]`` and the strips left over at
    irrational endpoints (``pi``), each a rational interval containing
    the endpoint."""
    strips: list[tuple[Rational, Rational]] = []
    lo, hi = _rational(a), _rational(b)
    if lo is None:
        value = _enclose(a, Dummy('x'), Rational(0), Rational(0))
        if value is None:
            return None
        lo = Rational(Float(mpmath.mpf(value.b), 40))
        strips.append((Rational(Float(mpmath.mpf(value.a), 40)), lo))
    if hi is None:
        value = _enclose(b, Dummy('x'), Rational(0), Rational(0))
        if value is None:
            return None
        hi = Rational(Float(mpmath.mpf(value.a), 40))
        strips.append((hi, Rational(Float(mpmath.mpf(value.b), 40))))
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
    if lo > hi:
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
    tidied = attempt(lambda: as_expr(cancel(expand_power_exp(g))), _budget())
    return g if tidied is None else tidied


def _rational_map(f: Expr, x: Symbol, a: Expr, tolerance: mpmath.mpf) -> Optional[ivmpf]:
    """``x = a + t/(1 - t)`` on ``[0, 1]``: algebraic decay."""
    t = Dummy('t', positive=True)
    g = as_expr(f.subs(x, a + t / (1 - t)) / (1 - t)**2)
    simpler = attempt(lambda: as_expr(simplify(cancel(g))), _budget())
    return _enclosed(simpler if simpler is not None else g, t, S.Zero, S.One, tolerance)


def _exponential_map(f: Expr, x: Symbol, a: Expr, tolerance: mpmath.mpf) -> Optional[ivmpf]:
    """``x = a - log(1 - t)`` on ``[0, 1]``: exponential decay."""
    t = Dummy('t', positive=True)
    g = _tidy(as_expr(f.subs(x, a - log(1 - t)) / (1 - t)))
    return _enclosed(g, t, S.Zero, S.One, tolerance)


def _stretched(f: Expr, x: Symbol, a: Expr, tolerance: mpmath.mpf) -> Optional[ivmpf]:
    """A decay ``exp(-c x**k)``, ``k >= 2``: the range cut at ``C =
    max(1, ceiling(a))``, and ``x = s**(1/k)`` on ``[C, oo)`` makes the
    decay exponential in ``s``."""
    degrees: set[int] = set()
    for node in f.atoms(exp):
        argument = as_expr(node.args[0])
        if x not in free_symbols(argument):
            continue
        p = _polynomial(argument, x)
        if p is None or p.degree() < 2 or not as_expr(p.LC()).is_negative:
            return None
        degrees.add(int(p.degree()))
    if len(degrees) != 1:
        return None
    k = degrees.pop()
    cut = Integer(max(int(ceiling(a)), 1))
    finite = _enclosed(f, x, a, cut, tolerance / 2)
    if finite is None:
        return None
    s = Dummy('s', positive=True)
    g = as_expr(f.subs(x, s**Rational(1, k)) * s**(Rational(1, k) - 1) / k)
    tail = _exponential_map(g, s, cut**k, tolerance / 2)
    return None if tail is None else finite + tail


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
    trigonometric = [i for i, factor_ in enumerate(factors) if isinstance(factor_, (sin, cos))]
    if len(trigonometric) != 1:
        return None
    phi = as_expr(factors.pop(trigonometric[0]))
    g = as_expr(Mul(*factors))
    if g.has(sin, cos, tan):
        return None
    p = _polynomial(as_expr(phi.args[0]), x)
    if p is None or p.degree() != 1:
        return None
    w = as_expr(p.LC())
    if not w.is_number:
        return None
    if w.is_negative:
        # sin(-y) = -sin(y), cos(-y) = cos(y)
        phi, w = as_expr(phi.func(-as_expr(phi.args[0]))), as_expr(-w)
        if isinstance(phi, sin):
            g = as_expr(-g)
    w_ = _enclose(w, x, S.Zero, S.Zero)
    if not _finite(w_) or w_.a <= 0:
        return None
    derivatives = [g]
    antiderivatives = [phi]
    start = Integer(int(ceiling(a)) + 1)
    for m in range(24):
        A = as_expr(start * Integer(2)**m)
        for k in range(_MAX_PARTS + 1):
            if _out_of_time():
                return None
            while len(derivatives) <= k + 1:
                derivatives.append(as_expr(derivatives[-1].diff(x)))
            while len(antiderivatives) <= k:
                antiderivatives.append(as_expr(integrate(antiderivatives[-1], x)))
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
                term = _enclose(as_expr((-1)**(i + 1) * derivatives[i] * antiderivatives[i + 1]), x, A, A)
                if not _finite(term):
                    return None
                tail = tail + term
            finite = _enclosed(f, x, a, A, tolerance / 2)
            if finite is None:
                return None
            return finite + _iv.mpf([tail.a - remainder, tail.b + remainder])
    return None


def _sign_on_ray(expr: Expr, x: Symbol, A: Expr) -> Optional[int]:
    """The sign of ``expr`` on ``[A, oo)``: the polynomial factors in
    ``x`` of its numerator and denominator have no root there by Sturm's
    theorem (``count_roots``) and so the sign of their value at ``A``,
    the other factors a sign read from their enclosure on the ray; ``1``
    or ``-1`` for a constant sign, ``0`` for the zero expression, ``None``
    when undecided."""
    if expr == 0:
        return 0
    return attempt(lambda: _find_sign_on_ray(expr, x, A), _budget())


def _find_sign_on_ray(expr: Expr, x: Symbol, A: Expr) -> Optional[int]:
    combined = as_expr(together(expr))
    sign = 1
    for part in combined.as_numer_denom():
        for factor_ in Mul.make_args(as_expr(part)):
            base, exponent = as_expr(factor_), 1
            if isinstance(factor_, Pow) and isinstance(factor_.exp, Integer):
                base, exponent = as_expr(factor_.base), int(factor_.exp)
            if x not in free_symbols(base):
                enclosed = _enclose(base, x, S.Zero, S.Zero)
                factor_sign = None if enclosed is None else _interval_sign(enclosed)
            else:
                p = _polynomial(base, x)
                if p is not None and p.domain.is_QQ or p is not None and p.domain.is_ZZ:
                    if p.count_roots(A) != 0:
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


def _monotone_to_zero(derivatives: list[Expr], x: Symbol, A: Expr) -> bool:
    """Whether the last derivative but one has a constant sign and the
    last the opposite sign (or vanishes) on ``[A, oo)``, and all of them
    vanish at infinity."""
    last, next_ = _sign_on_ray(derivatives[-2], x, A), _sign_on_ray(derivatives[-1], x, A)
    if last is None or next_ is None or last * next_ > 0:
        return False
    for derivative in derivatives[:-1]:
        at_infinity = attempt(lambda: as_expr(limit(derivative, x, oo)), _budget())
        if at_infinity != 0:
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
        return _infinite(as_expr(f.subs(x, -x)), x, as_expr(-b), oo, tolerance)
    if a == oo:
        reversed_ = _infinite(f, x, b, oo, tolerance)
        return None if reversed_ is None else -reversed_
    if b != oo:
        return None
    result = _oscillatory(f, x, a, tolerance)
    if result is None and f.has(exp):
        result = _stretched(f, x, a, tolerance)
    maps = [_exponential_map, _rational_map] if f.has(exp) else [_rational_map, _exponential_map]
    for mapping in maps:
        if result is not None or _out_of_time():
            break
        result = mapping(f, x, a, tolerance)
    return result


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------

def validated_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, digits: int = 15,
                       assumptions: Assumptions = None) -> Optional[tuple[Expr, Expr]]:
    """``(value, error)`` with ``|Integral(f, (x, a, b)) - value| <= error``
    proved by interval arithmetic, the error aimed at ``10**(-digits)``
    (it may be larger when the panel budget or the time limit is spent
    first); ``None`` when the integrand has parameters, uses functions
    the interval evaluator does not know, or cannot be bounded on some
    panel (a non-integrable singularity, an infinite range on which no
    transformation makes the integrand bounded). Irrational bounds are
    enclosed and the strips between the enclosures bounded by the range
    of ``f``.

    Examples
    ========

    >>> from sympy import symbols, sqrt, pi, oo, Rational, log, sin
    >>> from sympy_extras.integrals.validated import validated_integral
    >>> x = symbols('x')
    >>> value, error = validated_integral(sqrt(x), x, 0, 1)
    >>> abs(value - Rational(2, 3)) < error, error < 1e-15
    (True, True)
    >>> value, error = validated_integral(1/(1 + x**2), x, 0, oo, 20)
    >>> abs(value - pi/2) < error, error < 1e-20
    (True, True)
    >>> value, error = validated_integral(log(x), x, 0, 1)
    >>> abs(value + 1) < error
    True
    >>> value, error = validated_integral(sin(x)/x, x, 0, oo, 8)
    >>> abs(value - pi/2) < error, error < 1e-8
    (True, True)
    >>> validated_integral(1/x, x, 0, 1) is None
    True
    """
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    if free_symbols(f_) - {x} or free_symbols(a_) or free_symbols(b_):
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


def _pair(result: ivmpf, digits: int) -> tuple[Expr, Expr]:
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

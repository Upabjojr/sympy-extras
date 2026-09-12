r"""Antiderivatives of trigonometric and hyperbolic integrands.

The classical families, integrated by their reduction formulas and
substitutions, every result checked by differentiation before it is
returned (``None`` otherwise, never an unevaluated ``Integral``):

- `\sin^n(ax)\cos^m(ax)` for integers `n, m` of any sign
  (Gradshteyn–Ryzhik 2.510): an odd exponent by the substitution
  `v = \cos ax` or `v = \sin ax`, which leaves a Laurent polynomial or a
  rational function of `v`; even exponents by the reduction formulas
  obtained from `d(\sin^{n+1}\cos^{m+1})`,

  .. math::

      \int \sin^n\cos^m = \frac{\sin^{n+1}\cos^{m+1}}{n + 1}
        + \frac{n + m + 2}{n + 1}\int \sin^{n+2}\cos^m,
      \qquad
      \int \sin^n\cos^m = -\frac{\sin^{n+1}\cos^{m+1}}{m + 1}
        + \frac{n + m + 2}{m + 1}\int \sin^n\cos^{m+2},

  which raise a negative exponent to `0`, and their inverses which lower
  a positive one; `\tan^n`, `\sec^n`, `\cot^n`, `\csc^n` by their own
  recursions (GR 2.526, 2.527), `\int \sec = \log(\sec + \tan)`,
  `\int \csc = \log\tan(x/2)`;
- the hyperbolic counterparts `\sinh^n(ax)\cosh^m(ax)` (GR 2.41–2.43)
  by the same scheme with `\cosh^2 = 1 + \sinh^2`, `\int \operatorname{sech}
  = \arctan\sinh`, `\int \operatorname{csch} = \log\tanh(x/2)`;
- `x^k e^{bx}\sin ax`, `x^k e^{bx}\cos ax` (GR 2.66) by the closed
  recursion in `k` of the integration by parts, with
  `I_0 = e^{bx}(b\sin ax - a\cos ax)/(a^2 + b^2)` and
  `J_0 = e^{bx}(b\cos ax + a\sin ax)/(a^2 + b^2)`, and `x^k e^{cx}` by
  `e^{cx}\sum_j (-1)^j k!/(k - j)!\, x^{k-j}/c^{j+1}`; products of powers
  of sines and cosines with polynomials and exponentials are first
  written as sums of single sines and cosines (product to sum), the
  hyperbolic ones as sums of exponentials;
- `\sin ax\sin bx`, `\sin ax\cos bx`, `\cos ax\cos bx` by product to
  sum; when a frequency `a - b` of the sum may vanish (symbolic `a`,
  `b`), the antiderivative is a ``Piecewise`` with the case `a = b`
  integrated separately;
- rational functions of `\sin ax`, `\cos ax` by the Weierstrass
  substitution `t = \tan(ax/2)`, the rational function of `t`
  integrated by :func:`~sympy_extras.integrals.risch.rationaltools.ratint`
  and `t` written back; the result is an antiderivative on each interval
  where `\tan(ax/2)` is finite, with a jump at the odd multiples of
  `\pi/a` (the docstring of :func:`trigonometric_antiderivative` says
  how to integrate across them); rational functions of `\sinh ax`,
  `\cosh ax` by `u = e^{ax}`;
- `p(x)\,g(cx)` for a polynomial `p` and `g` an inverse trigonometric or
  hyperbolic function, by parts, the algebraic remainder through
  :func:`~sympy_extras.integrals.radicals.quadratic_radical_antiderivative`
  and the rational one through ``ratint``; `\arcsin^2`, `\arccos^2`,
  `\operatorname{arsinh}^2`, `\operatorname{arcosh}^2` in closed form
  (`\arctan^2` is not elementary: its integral has a dilogarithm).

The powers of sines and cosines are a port of SymPy's
``sympy.integrals.trigonometry`` (BSD licence, see ``risch/LICENSE-SymPy``),
with the even case rewritten on the raising and lowering formulas above
instead of the expansion of one power in the other, so that negative
exponents no longer recurse through ``integrate``.

Examples
========

>>> from sympy import symbols, sin, cos, exp, tan, asin
>>> from sympy_extras.integrals.trigonometric import trigonometric_antiderivative
>>> x = symbols('x')
>>> trigonometric_antiderivative(sin(x)**2, x)
(x - sin(x)*cos(x))/2
>>> trigonometric_antiderivative(tan(x)**3, x)
log(cos(x)) + tan(x)**2/2
>>> trigonometric_antiderivative(x*exp(x)*sin(x), x)
(2*x*(sin(x) - cos(x)) + 2*cos(x))*exp(x)/4
>>> trigonometric_antiderivative(asin(x), x)
x*asin(x) + sqrt(1 - x**2)

References
==========

.. [GR] I. S. Gradshteyn and I. M. Ryzhik, *Table of Integrals, Series,
   and Products*, 7th ed., Academic Press, 2007, 2.41–2.43 (hyperbolic
   functions), 2.51–2.53 (trigonometric functions), 2.66 (products with
   exponentials).
"""
from __future__ import annotations

from math import factorial
from typing import Callable, Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, count_ops
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational
from sympy.core.power import Pow
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import (sinh, cosh, tanh, coth, sech, csch,
                                                    asinh, acosh, atanh, acoth)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (sin, cos, tan, cot, sec, csc,
                                                       asin, acos, atan, acot)
from sympy.logic.boolalg import Boolean
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel
from sympy.simplify.fu import TR8
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.settings import settings
from .conditions import numerically_equal
from .radicals import quadratic_radical_antiderivative
from .risch.rationaltools import ratint

__all__ = ['trigonometric_antiderivative']

_TRIGONOMETRIC = (sin, cos, tan, cot, sec, csc)
_HYPERBOLIC = (sinh, cosh, tanh, coth, sech, csch)
_INVERSE = (asin, acos, atan, acot, asinh, acosh, atanh, acoth)


def _budget() -> Optional[float]:
    return None if settings.timeout is None else settings.timeout / 4


# ---------------------------------------------------------------------------
# The structure of a term

class _Term:
    """A product ``coefficient * polynomial * exp(rate*x) * (trigonometric
    and hyperbolic powers of one linear argument each) * inverse``."""

    def __init__(self, x: Symbol) -> None:
        self.x = x
        self.coefficient: Expr = S.One
        self.polynomial: Expr = S.One
        self.rate: Expr = S.Zero
        #: (kind 'sin' | 'cos' | 'sinh' | 'cosh', frequency) -> exponent
        self.powers: dict[tuple[str, Expr], int] = {}
        #: an inverse function with its frequency and exponent
        self.inverse: Optional[tuple[type[Function], Expr, int]] = None

    def frequencies(self, kinds: tuple[str, ...]) -> set[Expr]:
        return {a for (kind, a) in self.powers if kind in kinds}

    def trigonometric(self) -> bool:
        return any(kind in ('sin', 'cos') for kind, _ in self.powers)

    def hyperbolic(self) -> bool:
        return any(kind in ('sinh', 'cosh') for kind, _ in self.powers)


def _frequency(argument: Expr, x: Symbol) -> Optional[Expr]:
    """``a`` when ``argument == a*x`` with ``a`` free of ``x``."""
    if not argument.has(x):
        return None
    a = as_expr(cancel(argument / x))
    return None if a.has(x) else a


def _parse(term: Expr, x: Symbol) -> Optional[_Term]:
    """The structure of ``term``, ``None`` when a factor is not of the
    forms above."""
    result = _Term(x)
    for factor in Mul.make_args(term):
        f = as_expr(factor)
        if not f.has(x):
            result.coefficient = result.coefficient * f
            continue
        base, exponent = (as_expr(f.base), as_expr(f.exp)) if isinstance(f, Pow) else (f, S.One)
        if isinstance(base, exp) and exponent.is_integer:
            rate = _frequency(as_expr(base.args[0]), x)
            if rate is None:
                return None
            result.rate = result.rate + rate * exponent
            continue
        if isinstance(base, (_TRIGONOMETRIC + _HYPERBOLIC)) and isinstance(exponent, Integer):
            a = _frequency(as_expr(base.args[0]), x)
            if a is None:
                return None
            k = int(exponent)
            for kind, power in _sincos_powers(base, k):
                result.powers[(kind, a)] = result.powers.get((kind, a), 0) + power
            continue
        if isinstance(base, _INVERSE) and isinstance(exponent, Integer) and exponent > 0:
            c = _frequency(as_expr(base.args[0]), x)
            if c is None or result.inverse is not None:
                return None
            result.inverse = (type(base), c, int(exponent))
            continue
        if f.is_polynomial(x):
            result.polynomial = result.polynomial * f
            continue
        return None
    result.powers = {key: k for key, k in result.powers.items() if k != 0}
    return result


def _sincos_powers(function: Function, k: int) -> list[tuple[str, int]]:
    """``tan**k`` as ``sin**k cos**-k`` and so on."""
    table: dict[type[Function], list[tuple[str, int]]] = {
        sin: [('sin', 1)], cos: [('cos', 1)], tan: [('sin', 1), ('cos', -1)], cot: [('cos', 1), ('sin', -1)],
        sec: [('cos', -1)], csc: [('sin', -1)],
        sinh: [('sinh', 1)], cosh: [('cosh', 1)], tanh: [('sinh', 1), ('cosh', -1)],
        coth: [('cosh', 1), ('sinh', -1)], sech: [('cosh', -1)], csch: [('sinh', -1)]}
    return [(kind, power * k) for kind, power in table[type(function)]]


# ---------------------------------------------------------------------------
# Powers of sine and cosine of one argument

def _laurent_antiderivative(e: Expr, v: Symbol) -> Optional[Expr]:
    """The antiderivative in ``v`` of a Laurent polynomial or a rational
    function of ``v``."""
    expanded = as_expr(e.expand())
    total: Expr = S.Zero
    for term in Add.make_args(expanded):
        coefficient, rest = as_expr(term).as_independent(v, as_Add=False)
        coefficient_, rest_ = as_expr(coefficient), as_expr(rest)
        if rest_ == 1:
            total = total + coefficient_ * v
        elif rest_ == v:
            total = total + coefficient_ * v**2 / 2
        elif isinstance(rest_, Pow) and rest_.base == v and isinstance(rest_.exp, Integer):
            k = int(rest_.exp)
            total = total + coefficient_ * (log(v) if k == -1 else v**(k + 1) / (k + 1))
        else:
            rational = attempt(lambda: as_expr(ratint(expanded, v)), _budget())
            return rational
    return total


def _sin_cos(n: int, m: int, u: Symbol) -> Optional[Expr]:
    """``Integral(sin(u)**n * cos(u)**m, u)`` for integers ``n``, ``m``."""
    if n == 0 and m == 0:
        return u
    if n % 2 or m % 2:
        # an odd exponent: the substitution, the smaller exponent in
        # absolute value when both are odd and one is negative
        v = Dummy('v')
        use_cos = n % 2 == 1 and not (m % 2 == 1 and (m > 0 > n or (n < 0 and m < 0 and n < m)))
        if use_cos:
            # sin**n cos**m du = -(1 - v**2)**((n-1)/2) v**m dv, v = cos u
            integrand = as_expr(-(1 - v**2)**((n - 1) // 2) * v**m)
            found = _laurent_antiderivative(integrand, v)
            return None if found is None else as_expr(found.subs(v, cos(u)))
        integrand = as_expr(v**n * (1 - v**2)**((m - 1) // 2))
        found = _laurent_antiderivative(integrand, v)
        return None if found is None else as_expr(found.subs(v, sin(u)))
    # both even
    if n < 0:
        rest = _sin_cos(n + 2, m, u)
        if rest is None:
            return None
        return as_expr(sin(u)**(n + 1) * cos(u)**(m + 1) / (n + 1) + Rational(n + m + 2, n + 1) * rest)
    if m < 0:
        rest = _sin_cos(n, m + 2, u)
        if rest is None:
            return None
        return as_expr(-sin(u)**(n + 1) * cos(u)**(m + 1) / (m + 1) + Rational(n + m + 2, m + 1) * rest)
    if n > 0:
        rest = _sin_cos(n - 2, m, u)
        if rest is None:
            return None
        return as_expr(-sin(u)**(n - 1) * cos(u)**(m + 1) / (n + m) + Rational(n - 1, n + m) * rest)
    rest = _sin_cos(n, m - 2, u)
    if rest is None:
        return None
    return as_expr(sin(u)**(n + 1) * cos(u)**(m - 1) / (n + m) + Rational(m - 1, n + m) * rest)


def _tangent_family(n: int, m: int, u: Symbol) -> Optional[Expr]:
    """The nicer recursions for ``tan**k``, ``cot**k``, ``sec**k``,
    ``csc**k`` (``k > 0``), ``None`` for other ``(n, m)``."""
    if n > 0 and m == -n:
        return _tan_power(n, u)
    if m > 0 and n == -m:
        return _cot_power(m, u)
    if n == 0 and m < 0:
        return _sec_power(-m, u)
    if m == 0 and n < 0:
        return _csc_power(-n, u)
    return None


def _tan_power(k: int, u: Symbol) -> Expr:
    if k == 0:
        return u
    if k == 1:
        return as_expr(-log(cos(u)))
    return as_expr(tan(u)**(k - 1) / (k - 1) - _tan_power(k - 2, u))


def _cot_power(k: int, u: Symbol) -> Expr:
    if k == 0:
        return u
    if k == 1:
        return as_expr(log(sin(u)))
    return as_expr(-cot(u)**(k - 1) / (k - 1) - _cot_power(k - 2, u))


def _sec_power(k: int, u: Symbol) -> Expr:
    if k == 1:
        return as_expr(log(sec(u) + tan(u)))
    if k == 2:
        return as_expr(tan(u))
    return as_expr(sec(u)**(k - 2) * tan(u) / (k - 1) + Rational(k - 2, k - 1) * _sec_power(k - 2, u))


def _csc_power(k: int, u: Symbol) -> Expr:
    if k == 1:
        return as_expr(log(tan(u / 2)))
    if k == 2:
        return as_expr(-cot(u))
    return as_expr(-csc(u)**(k - 2) * cot(u) / (k - 1) + Rational(k - 2, k - 1) * _csc_power(k - 2, u))


# ---------------------------------------------------------------------------
# Powers of hyperbolic sine and cosine of one argument

def _sinh_cosh(n: int, m: int, u: Symbol) -> Optional[Expr]:
    """``Integral(sinh(u)**n * cosh(u)**m, u)`` for integers ``n``, ``m``."""
    if n == 0 and m == 0:
        return u
    if n % 2 or m % 2:
        v = Dummy('v')
        use_cosh = n % 2 == 1 and not (m % 2 == 1 and (m > 0 > n or (n < 0 and m < 0 and n < m)))
        if use_cosh:
            # sinh**n cosh**m du = (v**2 - 1)**((n-1)/2) v**m dv, v = cosh u
            found = _laurent_antiderivative(as_expr((v**2 - 1)**((n - 1) // 2) * v**m), v)
            return None if found is None else as_expr(found.subs(v, cosh(u)))
        found = _laurent_antiderivative(as_expr(v**n * (v**2 + 1)**((m - 1) // 2)), v)
        return None if found is None else as_expr(found.subs(v, sinh(u)))
    if n < 0:
        rest = _sinh_cosh(n + 2, m, u)
        if rest is None:
            return None
        return as_expr(sinh(u)**(n + 1) * cosh(u)**(m + 1) / (n + 1) - Rational(n + m + 2, n + 1) * rest)
    if m < 0:
        rest = _sinh_cosh(n, m + 2, u)
        if rest is None:
            return None
        return as_expr(-sinh(u)**(n + 1) * cosh(u)**(m + 1) / (m + 1) + Rational(n + m + 2, m + 1) * rest)
    if n > 0:
        rest = _sinh_cosh(n - 2, m, u)
        if rest is None:
            return None
        return as_expr(sinh(u)**(n - 1) * cosh(u)**(m + 1) / (n + m) - Rational(n - 1, n + m) * rest)
    rest = _sinh_cosh(n, m - 2, u)
    if rest is None:
        return None
    return as_expr(sinh(u)**(n + 1) * cosh(u)**(m - 1) / (n + m) + Rational(m - 1, n + m) * rest)


def _hyperbolic_family(n: int, m: int, u: Symbol) -> Optional[Expr]:
    """``tanh**k``, ``coth**k``, ``sech**k``, ``csch**k``, ``None`` otherwise."""
    if n > 0 and m == -n:
        return _tanh_power(n, u)
    if m > 0 and n == -m:
        return _coth_power(m, u)
    if n == 0 and m < 0:
        return _sech_power(-m, u)
    if m == 0 and n < 0:
        return _csch_power(-n, u)
    return None


def _tanh_power(k: int, u: Symbol) -> Expr:
    if k == 0:
        return u
    if k == 1:
        return as_expr(log(cosh(u)))
    return as_expr(-tanh(u)**(k - 1) / (k - 1) + _tanh_power(k - 2, u))


def _coth_power(k: int, u: Symbol) -> Expr:
    if k == 0:
        return u
    if k == 1:
        return as_expr(log(sinh(u)))
    return as_expr(-coth(u)**(k - 1) / (k - 1) + _coth_power(k - 2, u))


def _sech_power(k: int, u: Symbol) -> Expr:
    if k == 1:
        return as_expr(atan(sinh(u)))
    if k == 2:
        return as_expr(tanh(u))
    return as_expr(sech(u)**(k - 2) * tanh(u) / (k - 1) + Rational(k - 2, k - 1) * _sech_power(k - 2, u))


def _csch_power(k: int, u: Symbol) -> Expr:
    if k == 1:
        return as_expr(log(tanh(u / 2)))
    if k == 2:
        return as_expr(-coth(u))
    return as_expr(-csch(u)**(k - 2) * coth(u) / (k - 1) - Rational(k - 2, k - 1) * _csch_power(k - 2, u))


def _powers_of_one_argument(term: _Term) -> Optional[Expr]:
    """``sin(ax)**n cos(ax)**m`` or the hyperbolic pair, without other
    factors."""
    if term.polynomial != 1 or term.rate != 0 or term.inverse is not None or not term.powers:
        return None
    kinds = tuple(sorted({kind for kind, _ in term.powers}))
    frequencies = term.frequencies(kinds)
    if len(frequencies) != 1:
        return None
    a = frequencies.pop()
    hyperbolic = term.hyperbolic()
    if hyperbolic and term.trigonometric():
        return None
    n = term.powers.get(('sinh' if hyperbolic else 'sin', a), 0)
    m = term.powers.get(('cosh' if hyperbolic else 'cos', a), 0)
    u = Dummy('u')
    if hyperbolic:
        found = _hyperbolic_family(n, m, u)
        if found is None:
            found = _sinh_cosh(n, m, u)
    else:
        found = _tangent_family(n, m, u)
        if found is None:
            found = _sin_cos(n, m, u)
    if found is None:
        return None
    return as_expr(term.coefficient * found.subs(u, a * term.x) / a)


# ---------------------------------------------------------------------------
# Polynomials and exponentials times sines and cosines

def _x_power_exp(k: int, c: Expr, x: Symbol) -> Expr:
    """``Integral(x**k * exp(c*x), x)``."""
    if c == 0:
        return as_expr(x**(k + 1) / (k + 1))
    total: Expr = S.Zero
    for j in range(k + 1):
        total = total + S.NegativeOne**j * Rational(factorial(k), factorial(k - j)) * x**(k - j) / c**(j + 1)
    return as_expr(exp(c * x) * total)


def _x_power_exp_trig(k: int, b: Expr, a: Expr, kind: str, x: Symbol) -> Expr:
    """``Integral(x**k * exp(b*x) * sin(a*x), x)`` (``kind == 'sin'``) or with
    ``cos``, by the closed recursion of the integration by parts."""
    den = as_expr(a**2 + b**2)
    i0 = as_expr(exp(b * x) * (b * sin(a * x) - a * cos(a * x)) / den)
    j0 = as_expr(exp(b * x) * (b * cos(a * x) + a * sin(a * x)) / den)
    i_prev, j_prev = i0, j0
    for j in range(1, k + 1):
        i_next = as_expr(x**j * i0 - j * (b * i_prev - a * j_prev) / den)
        j_next = as_expr(x**j * j0 - j * (b * j_prev + a * i_prev) / den)
        i_prev, j_prev = i_next, j_next
    return i_prev if kind == 'sin' else j_prev


def _single_harmonic(term: _Term) -> Optional[Expr]:
    """``polynomial * exp(b x) * sin(a x)`` or ``cos(a x)`` (one power),
    or ``polynomial * exp(b x)`` alone."""
    if term.inverse is not None or term.hyperbolic():
        return None
    x = term.x
    if not term.powers:
        harmonic: Optional[tuple[str, Expr]] = None
    else:
        if len(term.powers) != 1:
            return None
        ((kind, a), power), = term.powers.items()
        if power != 1:
            return None
        harmonic = (kind, a)
    try:
        poly = Poly(term.polynomial, x)
    except PolynomialError:
        return None
    total: Expr = S.Zero
    for (k,), c in poly.terms():
        if harmonic is None:
            total = total + c * _x_power_exp(int(k), term.rate, x)
        else:
            kind, a = harmonic
            if a == 0:
                total = total + (c * _x_power_exp(int(k), term.rate, x) if kind == 'cos' else S.Zero)
            else:
                total = total + c * _x_power_exp_trig(int(k), term.rate, a, kind, x)
    return as_expr(term.coefficient * total)


def _product_to_sum(term: _Term, integrate: Callable[[Expr], Optional[Expr]]) -> Optional[Expr]:
    """Powers and products of sines and cosines (several frequencies,
    polynomial and exponential factors) written as a sum of single
    harmonics, the hyperbolic ones as exponentials, and integrated term
    by term; a frequency which may vanish gives a ``Piecewise``."""
    if term.inverse is not None or not term.powers:
        return None
    x = term.x
    product: Expr = S.One
    for (kind, a), k in term.powers.items():
        function: Function = {'sin': sin, 'cos': cos, 'sinh': sinh, 'cosh': cosh}[kind](a * x)
        product = product * function**k
    if term.hyperbolic():
        expanded = as_expr(product.rewrite(exp).expand())
    else:
        expanded = as_expr(TR8(product).expand())
    if expanded == product:
        return None
    integrand = as_expr(term.coefficient * term.polynomial * exp(term.rate * x) * expanded).expand()
    # frequencies which may vanish: a - b with a, b symbols
    doubtful: list[Expr] = []
    for node in integrand.atoms(sin, cos):
        c = _frequency(as_expr(node.args[0]), x)
        if c is not None and c.free_symbols and c.is_zero is None and c not in doubtful:
            doubtful.append(c)
    generic = integrate(integrand)
    if generic is None:
        return None
    if len(doubtful) != 1:
        return generic
    c = doubtful[0]
    symbols = sorted_symbols(free_symbols(c))
    solutions = attempt(lambda: solve(c, symbols[-1]), _budget())
    if not isinstance(solutions, list) or len(solutions) != 1:
        return generic
    special_integrand = as_expr(integrand.subs(symbols[-1], solutions[0]))
    special = integrate(special_integrand)
    if special is None or not _verified(special, special_integrand, x):
        return generic
    return as_expr(Piecewise((generic, Ne(c, 0)), (special, True)))


# ---------------------------------------------------------------------------
# Rational functions of sine and cosine, of hyperbolic sine and cosine

def _in_sines_and_cosines(f: Expr) -> Expr:
    """``tan``, ``cot``, ``sec``, ``csc`` and their hyperbolic counterparts
    written in sines and cosines of the same argument (SymPy's
    ``rewrite`` doubles the argument of a tangent)."""
    quotients: dict[type[Function], Callable[[Expr], Expr]] = {
        tan: lambda u: sin(u) / cos(u), cot: lambda u: cos(u) / sin(u), sec: lambda u: 1 / cos(u),
        csc: lambda u: 1 / sin(u), tanh: lambda u: sinh(u) / cosh(u), coth: lambda u: cosh(u) / sinh(u),
        sech: lambda u: 1 / cosh(u), csch: lambda u: 1 / sinh(u)}
    replacement: dict[Expr, Expr] = {}
    for node in f.atoms(*quotients):
        replacement[as_expr(node)] = as_expr(quotients[type(node)](as_expr(node.args[0])))
    return as_expr(f.xreplace(replacement)) if replacement else f


def _rational_in_sincos(f: Expr, x: Symbol) -> Optional[Expr]:
    """The Weierstrass substitution ``t = tan(a x/2)`` for a rational
    function of ``sin(a x)`` and ``cos(a x)``, and ``u = exp(a x)`` for
    one of ``sinh(a x)`` and ``cosh(a x)``."""
    g = _in_sines_and_cosines(f)
    frequencies: set[Expr] = set()
    kinds: set[str] = set()
    for node in g.atoms(sin, cos, sinh, cosh):
        a = _frequency(as_expr(node.args[0]), x)
        if a is None:
            return None
        frequencies.add(a)
        kinds.add('hyperbolic' if isinstance(node, (sinh, cosh)) else 'trigonometric')
    if len(frequencies) != 1 or len(kinds) != 1:
        return None
    a = frequencies.pop()
    s, c, t = Dummy('s'), Dummy('c'), Dummy('t')
    if 'trigonometric' in kinds:
        h = as_expr(g.xreplace({sin(a * x): s, cos(a * x): c}))
        if h.has(x) or not h.is_rational_function(s, c):
            return None
        rational = as_expr(cancel(h.xreplace({s: 2 * t / (1 + t**2), c: (1 - t**2) / (1 + t**2)}) * 2 / (a * (1 + t**2))))
        back: Expr = tan(a * x / 2)
    else:
        h = as_expr(g.xreplace({sinh(a * x): s, cosh(a * x): c}))
        if h.has(x) or not h.is_rational_function(s, c):
            return None
        rational = as_expr(cancel(h.xreplace({s: (t - 1 / t) / 2, c: (t + 1 / t) / 2}) / (a * t)))
        back = exp(a * x)
    found = attempt(lambda: as_expr(ratint(rational, t)), _budget())
    if found is None:
        return None
    return as_expr(found.subs(t, back))


# ---------------------------------------------------------------------------
# Inverse functions

def _inverse_times_power(term: _Term, integrate: Callable[[Expr], Optional[Expr]]) -> Optional[Expr]:
    """``p(x) g(c x)`` by parts, and ``g(c x)**2`` in closed form."""
    if term.inverse is None or term.powers or term.rate != 0:
        return None
    function, c, power = term.inverse
    x = term.x
    argument = as_expr(c * x)
    derivatives: dict[type[Function], Expr] = {
        asin: 1 / sqrt(1 - argument**2), acos: -1 / sqrt(1 - argument**2), atan: 1 / (1 + argument**2),
        acot: -1 / (1 + argument**2), asinh: 1 / sqrt(1 + argument**2), acosh: 1 / sqrt(argument**2 - 1),
        atanh: 1 / (1 - argument**2), acoth: 1 / (1 - argument**2)}
    g: Expr = function(argument)
    if power == 1:
        try:
            p = Poly(term.polynomial, x)
        except PolynomialError:
            return None
        primitive = as_expr(p.integrate().as_expr())
        remainder = as_expr(cancel(primitive * derivatives[function] * c))
        rest = _algebraic_or_rational(remainder, x)
        if rest is None:
            return None
        return as_expr(term.coefficient * (primitive * g - rest))
    if power == 2 and term.polynomial == 1:
        squares: dict[type[Function], Expr] = {
            asin: x * g**2 + 2 * sqrt(1 - argument**2) * g / c - 2 * x,
            acos: x * g**2 - 2 * sqrt(1 - argument**2) * g / c - 2 * x,
            asinh: x * g**2 - 2 * sqrt(1 + argument**2) * g / c + 2 * x,
            acosh: x * g**2 - 2 * sqrt(argument**2 - 1) * g / c + 2 * x}
        if function not in squares:
            return None
        return as_expr(term.coefficient * squares[function])
    return None


def _algebraic_or_rational(f: Expr, x: Symbol) -> Optional[Expr]:
    """An antiderivative of the remainder of an integration by parts:
    ``p(x) Q**(-1/2)`` through the radical table, a rational function
    through ``ratint``."""
    if f.is_rational_function(x):
        return attempt(lambda: as_expr(ratint(f, x)), _budget())
    return quadratic_radical_antiderivative(f, x)


# ---------------------------------------------------------------------------
# The check and the entry point

def _facts(f: Expr, x: Symbol) -> list[Boolean]:
    """Where the integrand is real: the radicands positive, the arguments
    of the inverse functions in their domains."""
    facts: list[Boolean] = []
    for node in f.atoms(Pow):
        if isinstance(node.exp, Rational) and node.exp.q == 2 and node.base.has(x):
            facts.append(as_boolean(as_expr(node.base) > 0))
    for node in f.atoms(asin, acos, atanh):
        facts.append(as_boolean(as_expr(node.args[0]) > -1))
        facts.append(as_boolean(as_expr(node.args[0]) < 1))
    for node in f.atoms(acosh):
        facts.append(as_boolean(as_expr(node.args[0]) > 1))
    for node in f.atoms(acoth):
        facts.append(as_boolean(as_expr(node.args[0]) > 1))
    return facts


def _verified(F: Expr, f: Expr, x: Symbol) -> bool:
    """Whether ``F`` is an antiderivative of ``f``: the difference of the
    derivative and ``f`` cancels, simplifies to zero, or vanishes at
    random points where ``f`` is real."""
    if isinstance(F, Piecewise):
        # the generic branch, the special one being checked where it is built
        return _verified(as_expr(F.args[0].args[0]), f, x)
    difference = as_expr(F.diff(x) - f)
    zero = attempt(lambda: as_expr(cancel(difference)), _budget())
    if zero == 0:
        return True
    zero = attempt(lambda: as_expr(simplify(difference)), _budget())
    if zero == 0:
        return True
    return numerically_equal(as_expr(F.diff(x)), f, _facts(f, x))


def _simpler(F: Expr) -> Expr:
    """The result with its terms collected (``trigsimp`` is not applied:
    it rewrites ``sin(x) - cos(x)`` as a phase-shifted cosine)."""
    candidate = attempt(lambda: as_expr(factor_terms(F)), _budget())
    if candidate is not None and count_ops(candidate) < count_ops(F):
        return candidate
    return F


def _antiderivative(f: Expr, x: Symbol, depth: int) -> Optional[Expr]:
    if depth > 3:
        return None
    if not f.has(x):
        return as_expr(f * x)
    if isinstance(f, Add):
        total: Expr = S.Zero
        for term in f.args:
            found = _antiderivative(as_expr(term), x, depth)
            if found is None:
                return None
            total = total + found
        return as_expr(total)
    coefficient, rest = f.as_independent(x, as_Add=False)
    coefficient_, rest_ = as_expr(coefficient), as_expr(rest)
    if coefficient_ != 1:
        found = _antiderivative(rest_, x, depth)
        return None if found is None else as_expr(coefficient_ * found)

    def deeper(g: Expr) -> Optional[Expr]:
        return _antiderivative(g, x, depth + 1)

    structure = _parse(f, x)
    if structure is not None:
        for route in (_powers_of_one_argument, _single_harmonic):
            found = route(structure)
            if found is not None:
                return found
        found = _inverse_times_power(structure, deeper)
        if found is not None:
            return found
        found = _product_to_sum(structure, deeper)
        if found is not None:
            return found
    return _rational_in_sincos(f, x)


def trigonometric_antiderivative(f: Expr, x: Symbol) -> Optional[Expr]:
    """An antiderivative of ``f`` in ``x`` by the reduction formulas and
    substitutions of the trigonometric and hyperbolic families listed in
    the module documentation, checked by differentiation; ``None`` when
    ``f`` is outside them or the check fails.

    The antiderivative of a rational function of ``sin`` and ``cos``
    through ``tan(a x/2)`` is one on each interval where ``tan(a x/2)``
    is finite: a definite integral across an odd multiple of ``pi/a``
    is the sum of the one-sided limits at it, which
    :func:`~sympy_extras.integrals.antiderivative.antiderivative_integral`
    computes.

    Examples
    ========

    >>> from sympy import symbols, sin, cos, sinh, sec
    >>> from sympy_extras.integrals.trigonometric import trigonometric_antiderivative
    >>> x = symbols('x')
    >>> trigonometric_antiderivative(sin(x)**3*cos(x)**2, x)
    cos(x)**5/5 - cos(x)**3/3
    >>> trigonometric_antiderivative(sec(x), x)
    log(tan(x) + sec(x))
    >>> trigonometric_antiderivative(sinh(x)**2, x)
    (-x + sinh(x)*cosh(x))/2
    >>> trigonometric_antiderivative(1/(2 + cos(x)), x)
    2*sqrt(3)*atan(sqrt(3)*tan(x/2)/3)/3
    >>> trigonometric_antiderivative(x**2, x) is None
    True
    """
    f_ = as_expr(f)
    if not f_.has(*_TRIGONOMETRIC, *_HYPERBOLIC, *_INVERSE):
        return None
    found = _antiderivative(f_, x, 0)
    if found is None:
        return None
    found = _simpler(found)
    return found if _verified(found, f_, x) else None

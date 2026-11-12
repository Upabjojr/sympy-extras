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
``sympy.integrals.trigonometry``, with the even case rewritten on the
raising and lowering formulas above instead of the expansion of one power
in the other, so that negative exponents no longer recurse through
``integrate``.

The module works on the objects of the core (:mod:`sympy_extras.core`,
the functions of :mod:`sympy_extras.functions`) with the native
``cancel``, ``roots``, ``TR8``, ``simplify`` and ``factor_terms`` and the
numerical checks of :mod:`sympy_extras.integrals.conditions`, and imports
no SymPy. One step stays on SymPy and is called at run time: the rational
functions (of `v = \cos ax`, of `t = \tan(ax/2)`, of `u = e^{ax}`, the
remainders of the integrations by parts) are integrated by ``ratint``,
written on SymPy, through
:func:`sympy_extras.integrals.radicals_sympy.ratint_of_core`.
:mod:`sympy_extras.integrals.trigonometric_sympy` is the face of the
module on SymPy objects.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import sin, cos, exp, tan, asin
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
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/trigonometry.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing, the objects of the core; the even powers by the
# raising and lowering formulas; every result verified by differentiation.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Callable, Optional

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add
from sympy_extras.core.assumptions import check_assumptions
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.function import count_ops, expand
from sympy_extras.core.logic import Boolean, true
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import I, Integer, Rational, number_from_fraction
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import ne
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, exp, log
from sympy_extras.functions.elementary.hyperbolic import (Acosh, Acoth, Asinh, Atanh, Cosh, Coth, Csch, Sech, Sinh,
                                                           Tanh, acosh, acoth, asinh, atanh, cosh, coth, csch, sech,
                                                           sinh, tanh)
from sympy_extras.functions.elementary.piecewise import ExprCondPair, Piecewise, piecewise
from sympy_extras.functions.elementary.trigonometric import (Acos, Acot, Asin, Atan, Cos, Cot, Csc, Sec, Sin, Tan,
                                                              acos, acot, asin, atan, cos, cot, csc, sec, sin, tan)
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel
from sympy_extras.settings import settings
from sympy_extras.simplify.fu import TR8
from sympy_extras.simplify.simplify import simplify
from .conditions import numerically_equal
from .radicals import _coefficients, quadratic_radical_antiderivative

__all__ = ['trigonometric_antiderivative']

#: an elementary function of one argument, evaluating (``sin``, ``asinh``, ...)
Elementary = Callable[[Expr], Expr]

_TRIGONOMETRIC: tuple[type[Expr], ...] = (Sin, Cos, Tan, Cot, Sec, Csc)
_HYPERBOLIC: tuple[type[Expr], ...] = (Sinh, Cosh, Tanh, Coth, Sech, Csch)
_INVERSE: tuple[type[Expr], ...] = (Asin, Acos, Atan, Acot, Asinh, Acosh, Atanh, Acoth)

#: the evaluating function of each node class of the inverse functions
_INVERSE_FUNCTIONS: dict[type[Expr], Elementary] = {
    Asin: asin, Acos: acos, Atan: atan, Acot: acot, Asinh: asinh, Acosh: acosh, Atanh: atanh, Acoth: acoth}

#: the kinds of the powers of a term
_KINDS: dict[str, Elementary] = {'sin': sin, 'cos': cos, 'sinh': sinh, 'cosh': cosh}


def _budget() -> Optional[float]:
    return None if settings.timeout is None else settings.timeout / 4


def _ratio(p: int, q: int) -> Expr:
    """``p/q`` as a number."""
    return number_from_fraction(Fraction(p, q))


def _ratint(f: Expr, v: Symbol) -> Optional[Expr]:
    """The rational integrator, written on SymPy, under a quarter of the
    time limit."""
    from .radicals_sympy import ratint_of_core
    return ratint_of_core(f, v, _budget())


def _argument(e: Basic) -> Expr:
    """The argument of a function of one argument."""
    argument = e.args[0]
    if not isinstance(argument, Expr):
        raise TypeError("%s has no argument" % (e,))
    return argument


# ---------------------------------------------------------------------------
# The structure of a term

class _Term:
    """A product ``coefficient * polynomial * exp(rate*x) * (trigonometric
    and hyperbolic powers of one linear argument each) * inverse``."""

    def __init__(self, x: Symbol) -> None:
        self.x = x
        self.coefficient: Expr = Integer(1)
        self.polynomial: Expr = Integer(1)
        self.rate: Expr = Integer(0)
        #: (kind 'sin' | 'cos' | 'sinh' | 'cosh', frequency) -> exponent
        self.powers: dict[tuple[str, Expr], int] = {}
        #: an inverse function (its node class) with its frequency and exponent
        self.inverse: Optional[tuple[type[Expr], Expr, int]] = None

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
    a = cancel(argument / x)
    return None if a.has(x) else a


def _parse(term: Expr, x: Symbol) -> Optional[_Term]:
    """The structure of ``term``, ``None`` when a factor is not of the
    forms above."""
    result = _Term(x)
    for f in Mul.make_args(term):
        if not f.has(x):
            result.coefficient = result.coefficient * f
            continue
        base, exponent = (f.base, f.exp) if isinstance(f, Pow) else (f, Integer(1))
        if isinstance(base, Exp) and exponent.is_integer:
            rate = _frequency(_argument(base), x)
            if rate is None:
                return None
            result.rate = result.rate + rate * exponent
            continue
        if isinstance(base, _TRIGONOMETRIC + _HYPERBOLIC) and isinstance(exponent, Integer):
            a = _frequency(_argument(base), x)
            if a is None:
                return None
            for kind, power in _sincos_powers(base, exponent.p):
                result.powers[(kind, a)] = result.powers.get((kind, a), 0) + power
            continue
        if isinstance(base, _INVERSE) and isinstance(exponent, Integer) and exponent.p > 0:
            c = _frequency(_argument(base), x)
            if c is None or result.inverse is not None:
                return None
            result.inverse = (type(base), c, exponent.p)
            continue
        if f.is_polynomial(x):
            result.polynomial = result.polynomial * f
            continue
        return None
    result.powers = {key: k for key, k in result.powers.items() if k != 0}
    return result


def _sincos_powers(function: Expr, k: int) -> list[tuple[str, int]]:
    """``tan**k`` as ``sin**k cos**-k`` and so on."""
    table: dict[type[Expr], list[tuple[str, int]]] = {
        Sin: [('sin', 1)], Cos: [('cos', 1)], Tan: [('sin', 1), ('cos', -1)], Cot: [('cos', 1), ('sin', -1)],
        Sec: [('cos', -1)], Csc: [('sin', -1)],
        Sinh: [('sinh', 1)], Cosh: [('cosh', 1)], Tanh: [('sinh', 1), ('cosh', -1)],
        Coth: [('cosh', 1), ('sinh', -1)], Sech: [('cosh', -1)], Csch: [('sinh', -1)]}
    return [(kind, power * k) for kind, power in table[type(function)]]


# ---------------------------------------------------------------------------
# Powers of sine and cosine of one argument

def _laurent_antiderivative(e: Expr, v: Symbol) -> Optional[Expr]:
    """The antiderivative in ``v`` of a Laurent polynomial or a rational
    function of ``v``."""
    expanded = expand(e)
    total: Expr = Integer(0)
    for term in Add.make_args(expanded):
        coefficient, rest = term.as_independent(v, as_Add=False)
        if rest == 1:
            total = total + coefficient * v
        elif rest == v:
            total = total + coefficient * v**2 / 2
        elif isinstance(rest, Pow) and rest.base == v and isinstance(rest.exp, Integer):
            k = rest.exp.p
            total = total + coefficient * (log(v) if k == -1 else v**(k + 1) / (k + 1))
        else:
            return _ratint(expanded, v)
    return total


# PORTED FROM SYMPY: sympy/integrals/trigonometry.py:trigintegrate
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
            integrand = -(1 - v**2)**((n - 1) // 2) * v**m
            found = _laurent_antiderivative(integrand, v)
            return None if found is None else found.subs(v, cos(u))
        integrand = v**n * (1 - v**2)**((m - 1) // 2)
        found = _laurent_antiderivative(integrand, v)
        return None if found is None else found.subs(v, sin(u))
    # both even
    if n < 0:
        rest = _sin_cos(n + 2, m, u)
        if rest is None:
            return None
        return sin(u)**(n + 1) * cos(u)**(m + 1) / (n + 1) + _ratio(n + m + 2, n + 1) * rest
    if m < 0:
        rest = _sin_cos(n, m + 2, u)
        if rest is None:
            return None
        return -sin(u)**(n + 1) * cos(u)**(m + 1) / (m + 1) + _ratio(n + m + 2, m + 1) * rest
    if n > 0:
        rest = _sin_cos(n - 2, m, u)
        if rest is None:
            return None
        return -sin(u)**(n - 1) * cos(u)**(m + 1) / (n + m) + _ratio(n - 1, n + m) * rest
    rest = _sin_cos(n, m - 2, u)
    if rest is None:
        return None
    return sin(u)**(n + 1) * cos(u)**(m - 1) / (n + m) + _ratio(m - 1, n + m) * rest


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
        return -log(cos(u))
    return tan(u)**(k - 1) / (k - 1) - _tan_power(k - 2, u)


def _cot_power(k: int, u: Symbol) -> Expr:
    if k == 0:
        return u
    if k == 1:
        return log(sin(u))
    return -cot(u)**(k - 1) / (k - 1) - _cot_power(k - 2, u)


def _sec_power(k: int, u: Symbol) -> Expr:
    if k == 1:
        return log(sec(u) + tan(u))
    if k == 2:
        return tan(u)
    return sec(u)**(k - 2) * tan(u) / (k - 1) + _ratio(k - 2, k - 1) * _sec_power(k - 2, u)


def _csc_power(k: int, u: Symbol) -> Expr:
    if k == 1:
        return log(tan(u / 2))
    if k == 2:
        return -cot(u)
    return -csc(u)**(k - 2) * cot(u) / (k - 1) + _ratio(k - 2, k - 1) * _csc_power(k - 2, u)


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
            found = _laurent_antiderivative((v**2 - 1)**((n - 1) // 2) * v**m, v)
            return None if found is None else found.subs(v, cosh(u))
        found = _laurent_antiderivative(v**n * (v**2 + 1)**((m - 1) // 2), v)
        return None if found is None else found.subs(v, sinh(u))
    if n < 0:
        rest = _sinh_cosh(n + 2, m, u)
        if rest is None:
            return None
        return sinh(u)**(n + 1) * cosh(u)**(m + 1) / (n + 1) - _ratio(n + m + 2, n + 1) * rest
    if m < 0:
        rest = _sinh_cosh(n, m + 2, u)
        if rest is None:
            return None
        return -sinh(u)**(n + 1) * cosh(u)**(m + 1) / (m + 1) + _ratio(n + m + 2, m + 1) * rest
    if n > 0:
        rest = _sinh_cosh(n - 2, m, u)
        if rest is None:
            return None
        return sinh(u)**(n - 1) * cosh(u)**(m + 1) / (n + m) - _ratio(n - 1, n + m) * rest
    rest = _sinh_cosh(n, m - 2, u)
    if rest is None:
        return None
    return sinh(u)**(n + 1) * cosh(u)**(m - 1) / (n + m) + _ratio(m - 1, n + m) * rest


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
        return log(cosh(u))
    return -tanh(u)**(k - 1) / (k - 1) + _tanh_power(k - 2, u)


def _coth_power(k: int, u: Symbol) -> Expr:
    if k == 0:
        return u
    if k == 1:
        return log(sinh(u))
    return -coth(u)**(k - 1) / (k - 1) + _coth_power(k - 2, u)


def _sech_power(k: int, u: Symbol) -> Expr:
    if k == 1:
        return atan(sinh(u))
    if k == 2:
        return tanh(u)
    return sech(u)**(k - 2) * tanh(u) / (k - 1) + _ratio(k - 2, k - 1) * _sech_power(k - 2, u)


def _csch_power(k: int, u: Symbol) -> Expr:
    if k == 1:
        return log(tanh(u / 2))
    if k == 2:
        return -coth(u)
    return -csch(u)**(k - 2) * coth(u) / (k - 1) - _ratio(k - 2, k - 1) * _csch_power(k - 2, u)


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
    return term.coefficient * found.subs(u, a * term.x) / a


# ---------------------------------------------------------------------------
# Polynomials and exponentials times sines and cosines

def _x_power_exp(k: int, c: Expr, x: Symbol) -> Expr:
    """``Integral(x**k * exp(c*x), x)``."""
    if c == 0:
        return x**(k + 1) / (k + 1)
    total: Expr = Integer(0)
    for j in range(k + 1):
        total = total + Integer((-1)**j * (factorial(k) // factorial(k - j))) * x**(k - j) / c**(j + 1)
    return exp(c * x) * total


def _x_power_exp_trig(k: int, b: Expr, a: Expr, kind: str, x: Symbol) -> Expr:
    """``Integral(x**k * exp(b*x) * sin(a*x), x)`` (``kind == 'sin'``) or with
    ``cos``, by the closed recursion of the integration by parts."""
    den = a**2 + b**2
    i0 = exp(b * x) * (b * sin(a * x) - a * cos(a * x)) / den
    j0 = exp(b * x) * (b * cos(a * x) + a * sin(a * x)) / den
    i_prev, j_prev = i0, j0
    for j in range(1, k + 1):
        i_next = x**j * i0 - j * (b * i_prev - a * j_prev) / den
        j_next = x**j * j0 - j * (b * j_prev + a * i_prev) / den
        i_prev, j_prev = i_next, j_next
    return i_prev if kind == 'sin' else j_prev


def _single_harmonic(term: _Term) -> Optional[Expr]:
    """``polynomial * exp(b x) * sin(a x)`` or ``cos(a x)`` (one power),
    or ``polynomial * exp(b x)`` alone."""
    if term.inverse is not None or term.hyperbolic():
        return None
    x = term.x
    harmonic: Optional[tuple[str, Expr]] = None
    if term.powers:
        if len(term.powers) != 1:
            return None
        ((kind, a), power), = term.powers.items()
        if power != 1:
            return None
        harmonic = (kind, a)
    coefficients = _coefficients(term.polynomial, x)
    if coefficients is None:
        return None
    total: Expr = Integer(0)
    for k in sorted(coefficients, reverse=True):
        c = coefficients[k]
        if harmonic is None:
            total = total + c * _x_power_exp(k, term.rate, x)
        else:
            kind, a = harmonic
            if a == 0:
                total = total + (c * _x_power_exp(k, term.rate, x) if kind == 'cos' else Integer(0))
            else:
                total = total + c * _x_power_exp_trig(k, term.rate, a, kind, x)
    return term.coefficient * total


def _in_exponentials(e: Expr) -> Expr:
    """The sines, cosines and their hyperbolic counterparts written in
    exponentials, in the forms of SymPy's ``rewrite(exp)``."""
    replacement: dict[Basic, Basic] = {}
    for node in e.atoms(Sin, Cos, Sinh, Cosh):
        u = _argument(node)
        if isinstance(node, Sin):
            replacement[node] = -I * (exp(I * u) - exp(-I * u)) / 2
        elif isinstance(node, Cos):
            replacement[node] = exp(I * u) / 2 + exp(-I * u) / 2
        elif isinstance(node, Sinh):
            replacement[node] = exp(u) / 2 - exp(-u) / 2
        else:
            replacement[node] = exp(u) / 2 + exp(-u) / 2
    return e.xreplace(replacement) if replacement else e


def _solutions(c: Expr, s: Symbol) -> Optional[list[Expr]]:
    """The solutions of ``c = 0`` in ``s`` (SymPy's ``solve`` for a
    polynomial: the distinct roots, those which contradict the assumptions
    of ``s`` left out); ``None`` when they are not all found."""
    try:
        poly = Poly(c, s)
    except BasePolynomialError:
        return None
    if poly.degree() < 1:
        return None
    found = attempt(lambda: roots(poly), _budget(), native=True)
    if not isinstance(found, dict) or sum(found.values()) != poly.degree():
        return None
    return [r for r in found if check_assumptions(r, s) is not False]


def _product_to_sum(term: _Term, integrate: Callable[[Expr], Optional[Expr]]) -> Optional[Expr]:
    """Powers and products of sines and cosines (several frequencies,
    polynomial and exponential factors) written as a sum of single
    harmonics, the hyperbolic ones as exponentials, and integrated term
    by term; a frequency which may vanish gives a ``Piecewise``."""
    if term.inverse is not None or not term.powers:
        return None
    x = term.x
    product: Expr = Integer(1)
    for (kind, a), k in term.powers.items():
        product = product * _KINDS[kind](a * x)**k
    if term.hyperbolic():
        expanded = expand(_in_exponentials(product))
    else:
        expanded = expand(TR8(product))
    if expanded == product:
        return None
    integrand = expand(term.coefficient * term.polynomial * exp(term.rate * x) * expanded)
    # frequencies which may vanish: a - b with a, b symbols
    doubtful: list[Expr] = []
    for node in integrand.atoms(Sin, Cos):
        c = _frequency(_argument(node), x)
        if c is not None and c.free_symbols and c.is_zero is None and c not in doubtful:
            doubtful.append(c)
    generic = integrate(integrand)
    if generic is None:
        return None
    if len(doubtful) != 1:
        return generic
    c = doubtful[0]
    symbols = sorted([s for s in c.free_symbols if isinstance(s, Symbol)], key=lambda s: s.name)
    solutions = _solutions(c, symbols[-1])
    if solutions is None or len(solutions) != 1:
        return generic
    special_integrand = integrand.subs(symbols[-1], solutions[0])
    special = integrate(special_integrand)
    if special is None or not _verified(special, special_integrand, x):
        return generic
    return piecewise((generic, ne(c, 0)), (special, true))


# ---------------------------------------------------------------------------
# Rational functions of sine and cosine, of hyperbolic sine and cosine

def _in_sines_and_cosines(f: Expr) -> Expr:
    """``tan``, ``cot``, ``sec``, ``csc`` and their hyperbolic counterparts
    written in sines and cosines of the same argument (SymPy's
    ``rewrite`` doubles the argument of a tangent)."""
    quotients: dict[type[Expr], Callable[[Expr], Expr]] = {
        Tan: lambda u: sin(u) / cos(u), Cot: lambda u: cos(u) / sin(u), Sec: lambda u: 1 / cos(u),
        Csc: lambda u: 1 / sin(u), Tanh: lambda u: sinh(u) / cosh(u), Coth: lambda u: cosh(u) / sinh(u),
        Sech: lambda u: 1 / cosh(u), Csch: lambda u: 1 / sinh(u)}
    replacement: dict[Basic, Basic] = {}
    for cls, rule in quotients.items():
        for node in f.atoms(cls):
            replacement[node] = rule(_argument(node))
    return f.xreplace(replacement) if replacement else f


def _rational_in_sincos(f: Expr, x: Symbol) -> Optional[Expr]:
    """The Weierstrass substitution ``t = tan(a x/2)`` for a rational
    function of ``sin(a x)`` and ``cos(a x)``, and ``u = exp(a x)`` for
    one of ``sinh(a x)`` and ``cosh(a x)``."""
    g = _in_sines_and_cosines(f)
    frequencies: set[Expr] = set()
    kinds: set[str] = set()
    for node in g.atoms(Sin, Cos, Sinh, Cosh):
        a = _frequency(_argument(node), x)
        if a is None:
            return None
        frequencies.add(a)
        kinds.add('hyperbolic' if isinstance(node, (Sinh, Cosh)) else 'trigonometric')
    if len(frequencies) != 1 or len(kinds) != 1:
        return None
    a = frequencies.pop()
    s, c, t = Dummy('s'), Dummy('c'), Dummy('t')
    if 'trigonometric' in kinds:
        to_symbols: dict[Basic, Basic] = {sin(a * x): s, cos(a * x): c}
        to_t: dict[Basic, Basic] = {s: 2 * t / (1 + t**2), c: (1 - t**2) / (1 + t**2)}
        h = g.xreplace(to_symbols)
        if h.has(x) or not h.is_rational_function(s, c):
            return None
        rational = cancel(h.xreplace(to_t) * 2 / (a * (1 + t**2)))
        back: Expr = tan(a * x / 2)
    else:
        to_symbols = {sinh(a * x): s, cosh(a * x): c}
        to_t = {s: (t - 1 / t) / 2, c: (t + 1 / t) / 2}
        h = g.xreplace(to_symbols)
        if h.has(x) or not h.is_rational_function(s, c):
            return None
        rational = cancel(h.xreplace(to_t) / (a * t))
        back = exp(a * x)
    found = _ratint(rational, t)
    if found is None:
        return None
    return found.subs(t, back)


# ---------------------------------------------------------------------------
# Inverse functions

def _inverse_times_power(term: _Term, integrate: Callable[[Expr], Optional[Expr]]) -> Optional[Expr]:
    """``p(x) g(c x)`` by parts, and ``g(c x)**2`` in closed form."""
    if term.inverse is None or term.powers or term.rate != 0:
        return None
    function, c, power = term.inverse
    x = term.x
    argument = c * x
    derivatives: dict[type[Expr], Expr] = {
        Asin: 1 / sqrt(1 - argument**2), Acos: -1 / sqrt(1 - argument**2), Atan: 1 / (1 + argument**2),
        Acot: -1 / (1 + argument**2), Asinh: 1 / sqrt(1 + argument**2), Acosh: 1 / sqrt(argument**2 - 1),
        Atanh: 1 / (1 - argument**2), Acoth: 1 / (1 - argument**2)}
    g = _INVERSE_FUNCTIONS[function](argument)
    if power == 1:
        coefficients = _coefficients(term.polynomial, x)
        if coefficients is None:
            return None
        primitive: Expr = Integer(0)
        for n in sorted(coefficients, reverse=True):
            primitive = primitive + coefficients[n] * x**(n + 1) / (n + 1)
        remainder = cancel(primitive * derivatives[function] * c)
        rest = _algebraic_or_rational(remainder, x)
        if rest is None:
            return None
        return term.coefficient * (primitive * g - rest)
    if power == 2 and term.polynomial == 1:
        squares: dict[type[Expr], Expr] = {
            Asin: x * g**2 + 2 * sqrt(1 - argument**2) * g / c - 2 * x,
            Acos: x * g**2 - 2 * sqrt(1 - argument**2) * g / c - 2 * x,
            Asinh: x * g**2 - 2 * sqrt(1 + argument**2) * g / c + 2 * x,
            Acosh: x * g**2 - 2 * sqrt(argument**2 - 1) * g / c + 2 * x}
        if function not in squares:
            return None
        return term.coefficient * squares[function]
    return None


def _algebraic_or_rational(f: Expr, x: Symbol) -> Optional[Expr]:
    """An antiderivative of the remainder of an integration by parts:
    ``p(x) Q**(-1/2)`` through the radical table, a rational function
    through ``ratint``."""
    if f.is_rational_function(x):
        return _ratint(f, x)
    return quadratic_radical_antiderivative(f, x)


# ---------------------------------------------------------------------------
# The check and the entry point

def _facts(f: Expr, x: Symbol) -> list[Boolean]:
    """Where the integrand is real: the radicands positive, the arguments
    of the inverse functions in their domains."""
    facts: list[Boolean] = []
    for node in f.atoms(Pow):
        if isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q == 2 and node.base.has(x):
            facts.append(node.base > 0)
    for node in f.atoms(Asin, Acos, Atanh):
        facts.append(_argument(node) > -1)
        facts.append(_argument(node) < 1)
    for node in f.atoms(Acosh):
        facts.append(_argument(node) > 1)
    for node in f.atoms(Acoth):
        facts.append(_argument(node) > 1)
    return facts


def _verified(F: Expr, f: Expr, x: Symbol) -> bool:
    """Whether ``F`` is an antiderivative of ``f``: the difference of the
    derivative and ``f`` cancels, simplifies to zero, or vanishes at
    random points where ``f`` is real."""
    if isinstance(F, Piecewise):
        # the generic branch, the special one being checked where it is built
        pair = F.args[0]
        if not isinstance(pair, ExprCondPair):
            return False
        return _verified(pair.expr, f, x)
    difference = F.diff(x) - f
    zero = attempt(lambda: cancel(difference), _budget(), native=True)
    if zero == 0:
        return True
    zero = attempt(lambda: simplify(difference), _budget(), native=True)
    if zero == 0:
        return True
    return numerically_equal(F.diff(x), f, _facts(f, x))


def _operations(e: Expr) -> int:
    """The number of operations of ``e`` (``count_ops``)."""
    counted = count_ops(e)
    return counted.p if isinstance(counted, Integer) else 0


def _simpler(F: Expr) -> Expr:
    """The result with its terms collected (``trigsimp`` is not applied:
    it rewrites ``sin(x) - cos(x)`` as a phase-shifted cosine)."""
    candidate = attempt(lambda: factor_terms(F), _budget(), native=True)
    if candidate is not None and _operations(candidate) < _operations(F):
        return candidate
    return F


def _antiderivative(f: Expr, x: Symbol, depth: int) -> Optional[Expr]:
    if depth > 3:
        return None
    if not f.has(x):
        return f * x
    if isinstance(f, Add):
        total: Expr = Integer(0)
        for term in f.args:
            if not isinstance(term, Expr):
                return None
            found = _antiderivative(term, x, depth)
            if found is None:
                return None
            total = total + found
        return total
    coefficient, rest = f.as_independent(x, as_Add=False)
    if coefficient != 1:
        found = _antiderivative(rest, x, depth)
        return None if found is None else coefficient * found

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

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sin, cos, sinh, sec
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
    if not f.has(*_TRIGONOMETRIC, *_HYPERBOLIC, *_INVERSE):
        return None
    found = _antiderivative(f, x, 0)
    if found is None:
        return None
    found = _simpler(found)
    return found if _verified(found, f, x) else None

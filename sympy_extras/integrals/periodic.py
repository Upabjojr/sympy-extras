"""Integrals of analytic periodic integrands over a period by the mean
value theorem: the constant coefficient of a Laurent series.

For `f(x) = F(e^{ix})` with `F` analytic in an annulus containing the
unit circle,

.. math::

    \\int_c^{c + 2\\pi} F(e^{ix})\\, dx = \\frac{1}{i}\\oint_{|z| = 1} F(z)\\, \\frac{dz}{z}
    = 2\\pi\\, a_0, \\qquad F(z) = \\sum_{n \\in \\mathbb{Z}} a_n z^n,

by Cauchy's formula for the Laurent coefficients [Ahlfors]_, section
4.5; a range of `k` periods gives `2\\pi k\\, a_0`. This is the integral
over a period of a function of `\\cos x` and `\\sin x` which is not
rational (a rational one is done by the residues inside the unit circle,
:func:`sympy_extras.integrals.residues.period_integral`): the origin is
an essential singularity of `F` (`e^{\\cos x} = e^{z/2} e^{1/(2z)}`) and
the coefficient is read from the series instead of a residue.

The constant coefficient of a product `G(z) H(1/z)` of two entire
functions with Taylor coefficients `g_j` and `h_j` is the convergent sum
`\\sum_j g_j h_j`, which is hypergeometric for the elementary functions
and is summed in closed form (`\\sum_j (a/2)^{2j}/j!^2 = I_0(a)`, whence
`\\int_0^{2\\pi} e^{a \\cos x}\\, dx = 2\\pi I_0(a)`, [GR]_ 3.915.1 and
3.937); a power `z^m` shifts the index, which gives
`\\int_0^{2\\pi} e^{a\\cos x} \\cos(n x)\\, dx = 2\\pi I_n(a)` ([GR]_ 3.915.2).
The Taylor coefficients come from SymPy's formal power series through
:func:`sympy_extras.integrals.brackets.taylor_coefficient`.

Examples
========

>>> from sympy import symbols, exp, cos, sin, pi
>>> from sympy_extras.integrals.periodic import mean_value_integral, laurent_coefficient
>>> x, z = symbols('x z')
>>> mean_value_integral(exp(cos(x))*cos(sin(x)), x, 0, 2*pi)
ConditionalValue(2*pi)
>>> mean_value_integral(exp(cos(x)), x, 0, 2*pi)
ConditionalValue(2*pi*besseli(0, 1))
>>> laurent_coefficient(exp(z)*exp(2/z), z, 1)
sqrt(2)*besseli(1, 2*sqrt(2))/2

References
==========

.. [Ahlfors] L. V. Ahlfors, *Complex Analysis*, 3rd ed., McGraw-Hill,
   1979, chapter 4, section 5 (Laurent series and Cauchy's formula for
   the coefficients).
.. [GR] I. S. Gradshteyn, I. M. Ryzhik, *Table of Integrals, Series, and
   Products*, 7th ed., Academic Press, 2007, 3.915.1-2 and 3.937.
"""
from __future__ import annotations

from typing import Optional

from sympy.concrete.summations import summation
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I, Integer, oo, pi
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.trigonometric import sin, cos, tan, cot, sec, csc, TrigonometricFunction
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.simplify.simplify import simplify

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.settings import settings
from .brackets import SeriesCoefficient, taylor_coefficient
from .conditions import ConditionalValue
from .mellin import monomial

__all__ = ['to_laurent', 'laurent_coefficient', 'mean_value_integral']


def _multiple(argument: Expr, x: Symbol) -> Optional[Integer]:
    """``k`` when ``argument == k*x`` with an integer ``k``."""
    coefficient, rest = argument.as_independent(x, as_Add=False)
    if as_expr(rest) != x:
        return None
    k = as_expr(coefficient)
    return k if isinstance(k, Integer) else None


def to_laurent(f: Expr, x: Symbol, z: Symbol) -> Optional[Expr]:
    """``f`` with the trigonometric functions of integer multiples of
    ``x`` (and ``exp(I*k*x)``) written in ``z = exp(I*x)``: ``cos(k x) =
    (z**k + z**(-k))/2``, ``sin(k x) = (z**k - z**(-k))/(2 I)``; ``None``
    when ``x`` occurs otherwise.

    >>> from sympy import symbols, cos, sin
    >>> from sympy_extras.integrals.periodic import to_laurent
    >>> x, z = symbols('x z')
    >>> to_laurent(cos(2*x) + sin(x), x, z)
    z**2/2 - I*(z - 1/z)/2 + 1/(2*z**2)
    """
    replacement: dict[Expr, Expr] = {}
    for node in f.atoms(Function):
        if not node.has(x) or len(node.args) != 1:
            continue
        argument = as_expr(node.args[0])
        if isinstance(node, exp):
            k = _multiple(as_expr(argument / I), x)
            if k is not None:
                replacement[as_expr(node)] = z**k
            # an outer function of trigonometric functions (exp(cos(x)))
            # is left as it is: its argument is replaced
            continue
        k = _multiple(argument, x)
        if k is None:
            continue
        cosine = as_expr((z**k + z**(-k)) / 2)
        sine = as_expr((z**k - z**(-k)) / (2 * I))
        if isinstance(node, cos):
            replacement[as_expr(node)] = cosine
        elif isinstance(node, sin):
            replacement[as_expr(node)] = sine
        elif isinstance(node, tan):
            replacement[as_expr(node)] = sine / cosine
        elif isinstance(node, cot):
            replacement[as_expr(node)] = cosine / sine
        elif isinstance(node, sec):
            replacement[as_expr(node)] = 1 / cosine
        elif isinstance(node, csc):
            replacement[as_expr(node)] = 1 / sine
        else:
            return None
    result = as_expr(f.xreplace(replacement))
    if result.has(x):
        return None
    # the outer trigonometric and hyperbolic functions of a Laurent
    # polynomial (cos(sin(x))) as exponentials, which expand splits into
    # exponentials of monomials
    if result.has(TrigonometricFunction, HyperbolicFunction):
        result = as_expr(result.rewrite(exp))
    return result


class _Factor:
    """A factor ``g(c z**p)`` of a term, with the exponents of its
    series in ``z``: ``sign*(offset + period*j)`` for ``j >= 0`` and the
    coefficient ``c(j)``."""

    def __init__(self, series: SeriesCoefficient, sign: int, scale: Expr) -> None:
        self.series = series
        self.sign = sign
        self.scale = scale

    def exponent(self, j: Expr) -> Expr:
        return as_expr(self.sign * (self.series.offset + self.series.period * j))

    def coefficient(self, j: Expr) -> Expr:
        """The coefficient of ``z**exponent(j)``, the scale of the
        argument included."""
        return as_expr(self.series.coefficient.subs(self.series.index, j)
                       * self.scale**(self.series.offset + self.series.period * j))


def _factor(e: Expr, z: Symbol) -> Optional[_Factor]:
    """A function of ``c z**p`` as a series in ``z`` (``p > 0``) or in
    ``1/z`` (``p < 0``)."""
    if not isinstance(e, Function) or len(e.args) != 1:
        return None
    found = monomial(as_expr(e.args[0]), z)
    if found is None:
        return None
    scale, power = found
    if not isinstance(power, Integer) or power == 0:
        return None
    y = Dummy('y')
    head = as_expr(e.func(y))
    series = taylor_coefficient(head, y)
    if series is None:
        return None
    # the argument c z**p: the series in y = c z**p has exponents
    # p*(offset + period*j) in z, with the coefficient c(j) c**(...)
    scaled = SeriesCoefficient(series.coefficient, series.index, series.period * abs(int(power)),
                               as_expr(series.offset * abs(int(power))))
    return _Factor(scaled, 1 if power > 0 else -1, as_expr(scale**(1 / Integer(abs(int(power))))))


def _term_coefficient(term: Expr, z: Symbol, m: Expr) -> Optional[Expr]:
    """The coefficient of ``z**m`` in one term ``constant * z**n * g(c z**p) * h(d z**q)``."""
    constant: Expr = S.One
    shift: Expr = S.Zero
    factors: list[_Factor] = []
    for part in Mul.make_args(_merged_exponentials(term, z)):
        e = as_expr(part)
        if not e.has(z):
            constant = constant * e
        elif e == z:
            shift = shift + 1
        elif isinstance(e, Pow) and e.base == z and isinstance(e.exp, Integer):
            shift = shift + e.exp
        elif isinstance(e, Pow) and isinstance(e.exp, Integer) and e.exp > 0 and isinstance(e.base, Function):
            for _ in range(int(e.exp)):
                found = _factor(as_expr(e.base), z)
                if found is None:
                    return None
                factors.append(found)
        else:
            found = _factor(e, z)
            if found is None:
                return None
            factors.append(found)
    target = as_expr(m - shift)
    if not factors:
        return constant if target == 0 else S.Zero
    if len(factors) == 1:
        return constant * _single(factors[0], target)
    if len(factors) == 2:
        value = _double(factors[0], factors[1], target)
        return None if value is None else constant * value
    return None


def _merged_exponentials(term: Expr, z: Symbol) -> Expr:
    """The exponential factors of a term combined into one exponential
    per power of ``z``: ``exp(1/(2 z)) exp(I/(2 z))`` is
    ``exp((1 + I)/(2 z))`` (``expand`` splits an exponential of a Laurent
    polynomial into one exponential per term, but does not merge the
    terms of equal power coming from different exponentials)."""
    others: list[Expr] = []
    by_power: dict[Expr, Expr] = {}
    for part in Mul.make_args(term):
        e = as_expr(part)
        if isinstance(e, exp) and e.has(z):
            argument = as_expr(e.args[0].expand())
            for piece in Add.make_args(argument):
                found = monomial(as_expr(piece), z)
                if found is None:
                    others.append(exp(as_expr(piece)))
                else:
                    by_power[found[1]] = by_power.get(found[1], S.Zero) + found[0]
        else:
            others.append(e)
    merged: Expr = S.One
    for power, scale in by_power.items():
        merged = merged * exp(as_expr(scale) * z**power)
    return as_expr(Mul(*others) * merged)


def _single(factor: _Factor, target: Expr) -> Expr:
    """The coefficient of ``z**target`` in one series."""
    j = as_expr((target * factor.sign - factor.series.offset) / factor.series.period)
    if not isinstance(j, Integer) or j < 0:
        return S.Zero
    return factor.coefficient(j)


def _double(first: _Factor, second: _Factor, target: Expr) -> Optional[Expr]:
    """The coefficient of ``z**target`` in the product of two series: the
    sum over the pairs of indices whose exponents add up to ``target``,
    in closed form."""
    t = Dummy('t', integer=True, nonnegative=True)
    m1, r1, s1 = first.series.period, first.series.offset, first.sign
    m2, r2, s2 = second.series.period, second.series.offset, second.sign
    total: Expr = S.Zero
    # j2 = m1*t + residue, so that j1 = (target - s2*(r2 + m2*j2) - s1*r1)/(s1*m1) is an integer
    for residue in range(m1):
        remainder = as_expr(target - s2 * (r2 + m2 * residue) - s1 * r1)
        if not isinstance(remainder, Integer) or remainder % m1 != 0:
            continue
        j2 = as_expr(m1 * t + residue)
        j1 = as_expr((target - s2 * (r2 + m2 * j2) - s1 * r1) / (s1 * m1))
        slope = as_expr(-s2 * m2 * m1 / (s1 * m1))
        # j1 >= 0 bounds t from below (slope > 0) or above (slope < 0)
        lower: Expr = S.Zero
        upper: Expr = oo
        bound = as_expr(j1.subs(t, 0))          # j1 at t = 0
        if slope > 0:
            if bound < 0:
                lower = as_expr(ceiling(-bound / slope))
        else:
            if bound < 0:
                continue
            upper = as_expr(floor(bound / (-slope)))
        summand = as_expr(first.coefficient(j1) * second.coefficient(j2))
        value = attempt(lambda: as_expr(summation(summand, (t, lower, upper))), settings.timeout)
        if value is None or value.has(Sum):
            return None
        total = total + value
    return as_expr(simplify(total))


def laurent_coefficient(F: ExprLike, z: Symbol, m: ExprLike = 0) -> Optional[Expr]:
    """The coefficient of ``z**m`` in the Laurent expansion of ``F`` about
    0, for ``F`` a sum of terms ``constant * z**n * g(c z**p) * h(d z**q)``
    with at most two functions of ``z`` (each with a formal power series,
    ``p`` and ``q`` nonzero integers) -- exponentials of Laurent
    polynomials are split into such factors by ``expand``; ``None`` when
    ``F`` is not of this form or a sum does not close.

    Examples
    ========

    >>> from sympy import symbols, exp, cos
    >>> from sympy_extras.integrals.periodic import laurent_coefficient
    >>> z = symbols('z')
    >>> laurent_coefficient(exp(z/2)*exp(1/(2*z)), z)
    besseli(0, 1)
    >>> laurent_coefficient(exp(z), z, 3)
    1/6
    >>> laurent_coefficient((z + 1/z)**4, z)
    6
    """
    expression = as_expr(as_expr(F).expand())
    m_ = as_expr(m)
    numerator, denominator = expression.as_numer_denom()
    for part in Mul.make_args(as_expr(denominator)):
        e = as_expr(part)
        if not e.has(z) or e == z or (isinstance(e, Pow) and e.base == z) or isinstance(e, exp):
            continue
        # a denominator vanishing away from the origin: the residues, not
        # the series
        return None
    total: Expr = S.Zero
    for term in Add.make_args(expression):
        value = _term_coefficient(as_expr(term), z, m_)
        if value is None:
            return None
        total = total + value
    return as_expr(simplify(total))


def mean_value_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                        assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` over ``k`` full periods (``b - a = 2 pi
    k``) of an integrand analytic in ``z = exp(I x)`` away from the
    origin: ``2 pi k`` times the constant Laurent coefficient; ``None``
    when the range is not a number of periods, ``x`` occurs otherwise
    than in trigonometric functions of its integer multiples, the
    integrand is rational in ``z`` (left to the residues) or the
    coefficient is not found.

    Examples
    ========

    >>> from sympy import symbols, exp, cos, sin, pi
    >>> from sympy_extras.integrals.periodic import mean_value_integral
    >>> x = symbols('x')
    >>> a = symbols('a', positive=True)
    >>> mean_value_integral(exp(a*cos(x))*cos(x), x, 0, 2*pi)
    ConditionalValue(2*pi*besseli(1, a))
    >>> mean_value_integral(exp(cos(x))*sin(sin(x)), x, -pi, pi)
    ConditionalValue(0)
    >>> mean_value_integral(cos(x)**4, x, 0, 4*pi)
    ConditionalValue(3*pi/2)
    """
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    periods = as_expr((b_ - a_) / (2 * pi))
    if not isinstance(periods, Integer) or periods <= 0:
        return None
    z = Dummy('z', positive=True)
    F = to_laurent(f_, x, z)
    if F is None:
        return None
    coefficient = laurent_coefficient(F, z, S.Zero)
    if coefficient is None:
        return None
    return ConditionalValue(as_expr(2 * pi * periods * coefficient))

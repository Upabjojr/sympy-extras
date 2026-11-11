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
The Taylor coefficients of `exp`, `sin`, `cos`, `sinh` and `cosh` are
written down; those of the other functions come from the formal power
series through :func:`sympy_extras.integrals.brackets.taylor_coefficient`.

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native expansion, summation (hypergeometric sums in closed
form) and ``simplify``. Only the formal power series of a function other
than the five above reaches SymPy's ``fps``, through
:mod:`.brackets`. The functions on SymPy objects are in
:mod:`.periodic_sympy`.

Examples
========

>>> from sympy_extras.core import symbols, pi
>>> from sympy_extras.functions import exp, cos, sin
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

from typing import Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.concrete.expr_with_limits import Sum
from sympy_extras.concrete.summations import summation
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import Application
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, Rational, oo, pi
from sympy_extras.core.power import Pow
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import factorial, rf
from sympy_extras.functions.elementary.exponential import Exp, exp
from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Csch, Sech, Sinh, Tanh
from sympy_extras.functions.elementary.integers import ceiling, floor
from sympy_extras.functions.elementary.trigonometric import Cos, Cot, Csc, Sec, Sin, Tan
from sympy_extras.settings import settings
from sympy_extras.simplify.simplify import simplify

from .brackets import SeriesCoefficient, taylor_coefficient
from .conditions import ConditionalValue
from .mellin import monomial

__all__ = ['to_laurent', 'laurent_coefficient', 'mean_value_integral']

#: the trigonometric and hyperbolic functions (SymPy's
#: ``TrigonometricFunction`` and ``HyperbolicFunction``), which an outer
#: function of a Laurent polynomial is rewritten from
_TRIGONOMETRIC_AND_HYPERBOLIC: Final = (Sin, Cos, Tan, Cot, Sec, Csc, Sinh, Cosh, Tanh, Coth, Sech, Csch)


def _multiple(argument: Expr, x: Symbol) -> Optional[Integer]:
    """``k`` when ``argument == k*x`` with an integer ``k``."""
    coefficient, rest = argument.as_independent(x, as_Add=False)
    if rest != x:
        return None
    return coefficient if isinstance(coefficient, Integer) else None


def to_laurent(f: Expr, x: Symbol, z: Symbol) -> Optional[Expr]:
    """``f`` with the trigonometric functions of integer multiples of
    ``x`` (and ``exp(I*k*x)``) written in ``z = exp(I*x)``: ``cos(k x) =
    (z**k + z**(-k))/2``, ``sin(k x) = (z**k - z**(-k))/(2 I)``; ``None``
    when ``x`` occurs otherwise.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import cos, sin
    >>> from sympy_extras.integrals.periodic import to_laurent
    >>> x, z = symbols('x z')
    >>> to_laurent(cos(2*x) + sin(x), x, z)
    z**2/2 - I*(z - 1/z)/2 + 1/(2*z**2)
    """
    f = as_expr(f)
    replacement: dict[Basic, Basic] = {}
    for node in f.atoms(Application):
        if not isinstance(node, Application) or not node.has(x) or len(node.args) != 1:
            continue
        argument = node.args[0]
        if isinstance(node, Exp):
            k = _multiple(argument / I, x)
            if k is not None:
                replacement[node] = z**k
            # an outer function of trigonometric functions (exp(cos(x)))
            # is left as it is: its argument is replaced
            continue
        k = _multiple(argument, x)
        if k is None:
            continue
        cosine = (z**k + z**(-k)) / 2
        sine = (z**k - z**(-k)) / (2 * I)
        if isinstance(node, Cos):
            replacement[node] = cosine
        elif isinstance(node, Sin):
            replacement[node] = sine
        elif isinstance(node, Tan):
            replacement[node] = sine / cosine
        elif isinstance(node, Cot):
            replacement[node] = cosine / sine
        elif isinstance(node, Sec):
            replacement[node] = 1 / cosine
        elif isinstance(node, Csc):
            replacement[node] = 1 / sine
        else:
            return None
    result = as_expr(f._xreplace_evaluated(replacement))
    if result.has(x):
        return None
    # the outer trigonometric and hyperbolic functions of a Laurent
    # polynomial (cos(sin(x))) as exponentials, which expand splits into
    # exponentials of monomials
    if result.has(*_TRIGONOMETRIC_AND_HYPERBOLIC):
        result = result.rewrite(Exp)
    return result


class _Factor:
    """A factor ``g(c z**p)`` of a term, with the exponents of its
    series in ``z``: ``sign*(offset + period*j)`` for ``j >= 0`` and the
    coefficient ``c(j)``."""

    __slots__ = ('series', 'sign', 'scale')

    def __init__(self, series: SeriesCoefficient, sign: int, scale: Expr) -> None:
        self.series: Final[SeriesCoefficient] = series
        self.sign: Final[int] = sign
        self.scale: Final[Expr] = scale

    def exponent(self, j: Expr) -> Expr:
        return self.sign * (self.series.offset + self.series.period * j)

    def coefficient(self, j: Expr) -> Expr:
        """The coefficient of ``z**exponent(j)``, the scale of the
        argument included."""
        return (self.series.coefficient.subs(self.series.index, j)
                * self.scale**(self.series.offset + self.series.period * j))


def _known_series(head: Expr, y: Symbol) -> Optional[SeriesCoefficient]:
    """The Taylor series of ``exp(y)``, ``sin(y)``, ``cos(y)``,
    ``sinh(y)`` and ``cosh(y)`` written down, in the forms of the formal
    power series (`(2j+1)! = 4^j j! (3/2)_j`, `(2j)! = 4^j j! (1/2)_j`);
    ``None`` for the other functions."""
    if not isinstance(head, (Exp, Sin, Cos, Sinh, Cosh)) or head.args[0] != y:
        return None
    j = Dummy('j', integer=True, nonnegative=True)
    if isinstance(head, Exp):
        return SeriesCoefficient(1 / factorial(j), j, 1, S.Zero)
    odd = isinstance(head, (Sin, Sinh))
    sign = Rational(-1, 4) if isinstance(head, (Sin, Cos)) else Rational(1, 4)
    coefficient = sign**j / (rf(Rational(3, 2) if odd else Rational(1, 2), j) * factorial(j))
    return SeriesCoefficient(coefficient, j, 2, S.One if odd else S.Zero)


def _factor(e: Expr, z: Symbol) -> Optional[_Factor]:
    """A function of ``c z**p`` as a series in ``z`` (``p > 0``) or in
    ``1/z`` (``p < 0``)."""
    if not isinstance(e, Application) or len(e.args) != 1:
        return None
    found = monomial(e.args[0], z)
    if found is None:
        return None
    scale, power = found
    if not isinstance(power, Integer) or power == 0:
        return None
    y = Dummy('y')
    head = as_expr(e._xreplace_evaluated({e.args[0]: y}))
    series = _known_series(head, y)
    if series is None:
        series = taylor_coefficient(head, y)
    if series is None:
        return None
    # the argument c z**p: the series in y = c z**p has exponents
    # p*(offset + period*j) in z, with the coefficient c(j) c**(...)
    p = abs(int(power))
    scaled = SeriesCoefficient(series.coefficient, series.index, series.period * p, series.offset * p)
    return _Factor(scaled, 1 if int(power) > 0 else -1, scale**Rational(1, p))


def _term_coefficient(term: Expr, z: Symbol, m: Expr) -> Optional[Expr]:
    """The coefficient of ``z**m`` in one term ``constant * z**n * g(c z**p) * h(d z**q)``."""
    constant: Expr = S.One
    shift: Expr = S.Zero
    factors: list[_Factor] = []
    for e in Mul.make_args(_merged_exponentials(term, z)):
        if not e.has(z):
            constant = constant * e
        elif e == z:
            shift = shift + 1
        elif isinstance(e, Pow) and e.base == z and isinstance(e.exp, Integer):
            shift = shift + e.exp
        elif (isinstance(e, Pow) and isinstance(e.exp, Integer) and int(e.exp) > 0
              and isinstance(e.base, Application)):
            for _ in range(int(e.exp)):
                found = _factor(e.base, z)
                if found is None:
                    return None
                factors.append(found)
        else:
            found = _factor(e, z)
            if found is None:
                return None
            factors.append(found)
    target = m - shift
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
    for e in Mul.make_args(term):
        if isinstance(e, Exp) and e.has(z):
            argument = e.args[0].expand()
            for piece in Add.make_args(argument):
                found = monomial(piece, z)
                if found is None:
                    others.append(exp(piece))
                else:
                    by_power[found[1]] = by_power.get(found[1], S.Zero) + found[0]
        else:
            others.append(e)
    merged: Expr = S.One
    for power, scale in by_power.items():
        merged = merged * exp(scale * z**power)
    return mul(*others) * merged


def _single(factor: _Factor, target: Expr) -> Expr:
    """The coefficient of ``z**target`` in one series."""
    j = (target * factor.sign - factor.series.offset) / factor.series.period
    if not isinstance(j, Integer) or int(j) < 0:
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
    # j1 decreases (slope < 0) or increases (slope > 0) with t
    slope = Rational(-s2 * m2 * m1, s1 * m1)
    # j2 = m1*t + residue, so that j1 = (target - s2*(r2 + m2*j2) - s1*r1)/(s1*m1) is an integer
    for residue in range(m1):
        remainder = target - s2 * (r2 + m2 * residue) - s1 * r1
        if not isinstance(remainder, Integer) or int(remainder) % m1 != 0:
            continue
        j2 = m1 * t + residue
        j1 = (target - s2 * (r2 + m2 * j2) - s1 * r1) / (s1 * m1)
        # j1 >= 0 bounds t from below (slope > 0) or above (slope < 0)
        lower: Expr = S.Zero
        upper: Expr = oo
        bound = j1.subs(t, S.Zero)          # j1 at t = 0, a rational number
        if not isinstance(bound, Rational):
            return None
        if slope.is_positive is True:
            if bound.is_negative is True:
                lower = ceiling(-bound / slope)
        else:
            if bound.is_negative is True:
                continue
            upper = floor(bound / (-slope))
        summand = first.coefficient(j1) * second.coefficient(j2)
        value = attempt(lambda: summation(summand, (t, lower, upper)), settings.timeout, native=True)
        if value is None or value.has(Sum):
            return None
        total = total + value
    return simplify(total)


def laurent_coefficient(F: ExprLike, z: Symbol, m: ExprLike = 0) -> Optional[Expr]:
    """The coefficient of ``z**m`` in the Laurent expansion of ``F`` about
    0, for ``F`` a sum of terms ``constant * z**n * g(c z**p) * h(d z**q)``
    with at most two functions of ``z`` (each with a formal power series,
    ``p`` and ``q`` nonzero integers) -- exponentials of Laurent
    polynomials are split into such factors by ``expand``; ``None`` when
    ``F`` is not of this form or a sum does not close.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.periodic import laurent_coefficient
    >>> z = symbols('z')
    >>> laurent_coefficient(exp(z/2)*exp(1/(2*z)), z)
    besseli(0, 1)
    >>> laurent_coefficient(exp(z), z, 3)
    1/6
    >>> laurent_coefficient((z + 1/z)**4, z)
    6
    """
    expression = to_expr(F).expand()
    m_ = to_expr(m)
    numerator, denominator = expression.as_numer_denom()
    for e in Mul.make_args(denominator):
        if not e.has(z) or e == z or (isinstance(e, Pow) and e.base == z) or isinstance(e, Exp):
            continue
        # a denominator vanishing away from the origin: the residues, not
        # the series
        return None
    total: Expr = S.Zero
    for term in Add.make_args(expression):
        value = _term_coefficient(term, z, m_)
        if value is None:
            return None
        total = total + value
    return simplify(total)


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

    >>> from sympy_extras.core import symbols, pi
    >>> from sympy_extras.functions import exp, cos, sin
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
    f_, a_, b_ = to_expr(f), to_expr(a), to_expr(b)
    periods = (b_ - a_) / (2 * pi)
    if not isinstance(periods, Integer) or int(periods) <= 0:
        return None
    z = Dummy('z', positive=True)
    F = to_laurent(f_, x, z)
    if F is None:
        return None
    coefficient = laurent_coefficient(F, z, S.Zero)
    if coefficient is None:
        return None
    return ConditionalValue(2 * pi * periods * coefficient)

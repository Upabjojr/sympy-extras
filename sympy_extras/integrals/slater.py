"""Slater's theorem: a Meijer G-function as a sum of hypergeometric
series, and the Mellin–Barnes integral of a gamma quotient as a
G-function.

The G-function is the Mellin–Barnes integral

.. math::

    G^{m,n}_{p,q}\\left(z \\,\\middle|\\, \\begin{matrix} a_1, \\ldots, a_p \\\\
    b_1, \\ldots, b_q \\end{matrix}\\right) = \\frac{1}{2\\pi i} \\int_L
    \\frac{\\prod_{j=1}^m \\Gamma(b_j - s) \\prod_{j=1}^n \\Gamma(1 - a_j + s)}
         {\\prod_{j=m+1}^q \\Gamma(1 - b_j + s) \\prod_{j=n+1}^p \\Gamma(a_j - s)}
    z^s\\, ds,

so a quotient of gamma functions of `s` (a :class:`~.mellin.GammaQuotient`)
is the integrand of a G-function once every `\\Gamma(a + b s)` has
`b = \\pm 1`: Gauss's multiplication formula

.. math::

    \\Gamma(k s + a) = (2\\pi)^{(1-k)/2} k^{k s + a - 1/2}
    \\prod_{j=0}^{k-1} \\Gamma\\left(s + \\frac{a + j}{k}\\right)

reduces an integer `b = k` to `k` factors with `b = 1`, and a rational
`b` is made an integer by scaling `s` (:func:`mellin_barnes`).

Slater's theorem [Slater]_ (DLMF 16.17.2) evaluates the integral by the
residues at the poles of the `\\Gamma(b_j - s)`: when no two of
`b_1, \\ldots, b_m` differ by an integer and `p < q`, or `p = q` and
`|z| < 1`,

.. math::

    G^{m,n}_{p,q}(z) = \\sum_{h=1}^m
    \\frac{\\prod_{j \\ne h}^m \\Gamma(b_j - b_h) \\prod_{j=1}^n \\Gamma(1 + b_h - a_j)}
         {\\prod_{j=m+1}^q \\Gamma(1 + b_h - b_j) \\prod_{j=n+1}^p \\Gamma(a_j - b_h)}
    z^{b_h}\\, {}_pF_{q-1}\\left(\\begin{matrix} 1 + b_h - a_1, \\ldots, 1 + b_h - a_p \\\\
    1 + b_h - b_1, \\ldots, *, \\ldots, 1 + b_h - b_q \\end{matrix}; (-1)^{p-m-n} z\\right),

and the cases `p > q` or `|z| > 1` follow from
`G^{m,n}_{p,q}(z \\mid a; b) = G^{n,m}_{q,p}(1/z \\mid 1 - b; 1 - a)`. The
hypergeometric series are then reduced to elementary and special
functions by :func:`sympy_extras.simplify.hyperexpand.hyperexpand`. When
two of the `b_j` differ by an integer (a logarithmic case) the parameters
are moved apart and the limit is taken, or the expansion is left to
``hyperexpand`` of the G-function.

The module works on the objects of the core (:mod:`sympy_extras.core`, the
functions of :mod:`sympy_extras.functions`) with the native ``ask``,
``hyperexpand``, ``simplify`` and limits, and imports no SymPy:
:mod:`sympy_extras.integrals.slater_sympy` is its face on SymPy objects.

Examples
========

>>> from sympy_extras.core import Rational, Symbol
>>> from sympy_extras.functions import meijerg
>>> from sympy_extras.integrals.slater import slater_expansion
>>> z = Symbol('z', positive=True)
>>> slater_expansion(meijerg([], [], [0], [], z))
exp(-z)
>>> slater_expansion(meijerg([1], [], [Rational(1, 2)], [0], z))
sqrt(pi)*erf(sqrt(z))

References
==========

.. [Slater] L. J. Slater, *Generalized hypergeometric functions*,
   Cambridge University Press, 1966, section 5.2.
.. [DLMF] NIST Digital Library of Mathematical Functions, chapter 16.17,
   https://dlmf.nist.gov/16.17.
"""
from __future__ import annotations

import random
from math import lcm
from typing import Final, Optional, Sequence

from sympy_extras._numeric import reliable_value
from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import evalf
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import Application
from sympy_extras.core.limits import Limit, limit
from sympy_extras.core.logic import Boolean, BooleanTrue, and_, or_, true
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Infinity, Integer, NegativeInfinity, Rational, pi
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import le, lt
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Abs, abs_, arg as arg_, im, re
from sympy_extras.functions.elementary.exponential import ExpPolar, exp
from sympy_extras.functions.elementary.piecewise import piecewise
from sympy_extras.functions.special.gamma_functions import gamma, lowergamma
from sympy_extras.functions.special.hyper import Hyper, MeijerG as MeijerGFunction, hyper, meijerg
from sympy_extras.settings import settings
from sympy_extras.simplify.hyperexpand import hyperexpand
from sympy_extras.simplify.simplify import simplify
from .conditions import ConditionalValue, numerically_equal, sample_values
from .mellin import GammaFactor, GammaQuotient

__all__ = ['MeijerG', 'mellin_barnes', 'slater_expansion', 'expand_meijerg', 'line_conditions']


class MeijerG:
    """A G-function `G^{m,n}_{p,q}(z \\mid a_n, a_p; b_m, b_q)` with a
    multiplicative prefactor.

    >>> from sympy_extras.core import Integer, Symbol
    >>> from sympy_extras.integrals.slater import MeijerG
    >>> z = Symbol('z')
    >>> g = MeijerG(Integer(2), [Integer(0)], [], [Integer(1)], [], z)
    >>> g, (g.m, g.n, g.p, g.q, g.delta)
    (MeijerG(2, meijerg(((0,), ()), ((1,), ()), z)), (1, 1, 1, 1, 1))
    >>> g.reflected()
    MeijerG(2, meijerg(((0,), ()), ((1,), ()), 1/z))
    """

    __slots__ = ('prefactor', 'an', 'ap', 'bm', 'bq', 'z')

    def __init__(self, prefactor: ExprLike, an: Sequence[ExprLike], ap: Sequence[ExprLike],
                 bm: Sequence[ExprLike], bq: Sequence[ExprLike], z: ExprLike) -> None:
        self.prefactor: Final[Expr] = to_expr(prefactor)
        self.an: Final[tuple[Expr, ...]] = tuple(to_expr(a) for a in an)
        self.ap: Final[tuple[Expr, ...]] = tuple(to_expr(a) for a in ap)
        self.bm: Final[tuple[Expr, ...]] = tuple(to_expr(b) for b in bm)
        self.bq: Final[tuple[Expr, ...]] = tuple(to_expr(b) for b in bq)
        self.z: Final[Expr] = to_expr(z)

    def __repr__(self) -> str:
        return "MeijerG(%s, %s)" % (self.prefactor, self.as_expr())

    @property
    def m(self) -> int:
        return len(self.bm)

    @property
    def n(self) -> int:
        return len(self.an)

    @property
    def p(self) -> int:
        return len(self.an) + len(self.ap)

    @property
    def q(self) -> int:
        return len(self.bm) + len(self.bq)

    @property
    def delta(self) -> Rational:
        """`m + n - (p + q)/2`."""
        return Rational(2 * (self.m + self.n) - self.p - self.q, 2)

    def as_expr(self) -> Expr:
        """The :func:`~sympy_extras.functions.special.hyper.meijerg`
        expression (without the prefactor)."""
        return meijerg(list(self.an), list(self.ap), list(self.bm), list(self.bq), self.z)

    def reflected(self) -> MeijerG:
        """The same function written with argument ``1/z``."""
        return MeijerG(self.prefactor, [1 - b for b in self.bm], [1 - b for b in self.bq],
                       [1 - a for a in self.an], [1 - a for a in self.ap], 1 / self.z)


def _xreplace(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced, evaluated (SymPy's ``xreplace``)."""
    replaced = e._xreplace_evaluated(rule)
    if not isinstance(replaced, Expr):
        raise TypeError("an expression is expected, got %s" % (replaced,))
    return replaced


def _has_infinity(e: Expr) -> bool:
    """``e.has(oo, zoo, nan)``."""
    return e.has(S.Infinity, S.ComplexInfinity, S.NaN)


def _multiplication(shift: Expr, k: int) -> tuple[Expr, Expr, list[GammaFactor]]:
    """Gauss's multiplication formula for ``Gamma(shift + k*u)``: the
    constant, the base of the power ``|k|**(k*u)`` and the factors
    ``Gamma((shift + j)/|k| + sign(k)*u)``."""
    size = abs(k)
    constant = (2 * pi)**Rational(1 - size, 2) * Integer(size)**(shift - S.Half)
    sign = Integer(1) if k > 0 else Integer(-1)
    factors: list[GammaFactor] = [((shift + j) / size, sign) for j in range(size)]
    return constant, Integer(size), factors


def mellin_barnes(quotient: GammaQuotient) -> Optional[MeijerG]:
    """The G-function equal to ``(1/2 pi i) Integral(quotient(t), t)``
    along a vertical line inside the strip; ``None`` when the quotient has
    an ``extra`` factor.

    Examples
    ========

    >>> from sympy_extras.core import S
    >>> from sympy_extras.integrals.mellin import GammaQuotient
    >>> from sympy_extras.integrals.slater import mellin_barnes
    >>> q = GammaQuotient(1, [(3, -1)], [(0, S.One), (1, -S.One)], [], 0, 1)
    >>> mellin_barnes(q)
    MeijerG(1, meijerg(((1,), ()), ((1,), ()), 1/3))
    >>> q = GammaQuotient(1, [], [(S.Half, S.Half)], [(1, -S.Half)], -1, 1)
    >>> mellin_barnes(q)
    MeijerG(2, meijerg(((1/2,), (1,)), ((), ()), 1))
    """
    if not quotient.is_gamma_quotient():
        return None
    scales = [b.q for _, b in quotient.numerator + quotient.denominator if b != 0]
    lam = 1
    for d in scales:
        lam = lcm(lam, d)
    constant = quotient.constant * lam
    z: Expr = S.One
    for base, coefficient in quotient.bases:
        z = z * base**(coefficient * lam)
    an: list[Expr] = []
    ap: list[Expr] = []
    bm: list[Expr] = []
    bq: list[Expr] = []
    for factors, numerator in ((quotient.numerator, True), (quotient.denominator, False)):
        for shift, scale in factors:
            # scale * lam, an integer since lam is a multiple of the denominator
            k = scale.p * (lam // scale.q)
            if k == 0:
                constant = constant * gamma(shift) if numerator else constant / gamma(shift)
                continue
            unit: list[GammaFactor] = [(shift, Integer(k))]
            if abs(k) > 1:
                c, base, unit = _multiplication(shift, k)
                if numerator:
                    constant = constant * c
                    z = z * base**k
                else:
                    constant = constant / c
                    z = z * base**(-k)
            for a, b in unit:
                if numerator and b.p > 0:
                    an.append(1 - a)
                elif numerator:
                    bm.append(a)
                elif b.p > 0:
                    bq.append(1 - a)
                else:
                    ap.append(a)
    return MeijerG(constant, an, ap, bm, bq, z)


def _integer_difference(a: Expr, b: Expr) -> Optional[bool]:
    d = a - b
    if d.is_integer is True:
        return True
    if d.is_integer is False:
        return False
    return None


def _slater_series(g: MeijerG) -> Optional[Expr]:
    """The sum of DLMF 16.17.2 for ``p < q`` or ``|z| < 1``. In a
    logarithmic case (two of ``b_1, ..., b_m`` differing by an integer)
    the coincident parameters are moved apart by multiples of a small
    ``epsilon`` and the limit ``epsilon -> 0`` of the sum is taken;
    ``None`` when the differences are not known to be integers or not,
    or the limit fails."""
    bs = list(g.bm) + list(g.bq)
    as_ = list(g.an) + list(g.ap)
    coincident: list[int] = []
    for i in range(g.m):
        for j in range(i + 1, g.m):
            difference = _integer_difference(g.bm[i], g.bm[j])
            if difference is None:
                return None
            if difference and j not in coincident:
                coincident.append(j)
    if coincident:
        return _logarithmic_case(g, coincident)
    total: Expr = S.Zero
    sign = Integer(-1)**(g.p - g.m - g.n)
    for h in range(g.m):
        bh = g.bm[h]
        coefficient: Expr = S.One
        for j in range(g.m):
            if j != h:
                coefficient = coefficient * gamma(g.bm[j] - bh)
        for a in g.an:
            coefficient = coefficient * gamma(1 + bh - a)
        for b in g.bq:
            coefficient = coefficient / gamma(1 + bh - b)
        for a in g.ap:
            coefficient = coefficient / gamma(a - bh)
        upper = [1 + bh - a for a in as_]
        lower = [1 + bh - b for j, b in enumerate(bs) if j != h]
        total = total + coefficient * g.z**bh * hyper(upper, lower, sign * g.z)
    return total


def _logarithmic_case(g: MeijerG, coincident: list[int]) -> Optional[Expr]:
    # an irrational perturbation, so that the moved parameters do not
    # differ by integers any more
    epsilon = Dummy('epsilon', positive=True, rational=False)
    bm = [b + (coincident.index(i) + 1) * epsilon if i in coincident else b for i, b in enumerate(g.bm)]
    perturbed = MeijerG(g.prefactor, g.an, g.ap, bm, g.bq, g.z)
    series = _slater_series(perturbed)
    value: Optional[Expr] = None
    if series is not None:
        expanded = _hyperexpand(series)
        if not expanded.has(Hyper):
            # an eighth of the time limit: the limit (gruntz on the perturbed
            # hypergeometric sum) took a minute for the Parseval integrand
            # of expint(1, a*t)*exp(-(s - a)*t), where the expansion of the
            # G-function below is immediate; the limit first, since its
            # forms are the real ones (acosh(s/k)/sqrt(s**2 - k**2) for the
            # transform of besselk(0, k*t), where the expansion gives
            # (pi/2 - asin(s/k))/sqrt(k**2 - s**2), real by a cancellation)
            budget = None if settings.timeout is None else settings.timeout / 8
            value = attempt(lambda: limit(expanded, epsilon, 0), budget, native=True)
    if value is not None and not (_has_infinity(value) or value.has(Limit)):
        return value
    try:
        direct = _hyperexpand(g.as_expr())
    except ValueError:
        return None
    if direct.has(MeijerGFunction, Hyper) or _has_infinity(direct):
        return None
    return direct


def _hyperexpand(e: Expr) -> Expr:
    try:
        expanded = attempt(lambda: hyperexpand(e), settings.timeout, native=True)
    except AttributeError:
        # SymPy 1.14 (through the bridge): meijerg._eval_evalf fails on some
        # arguments (AttributeError: 'NoneType' object has no attribute 'has')
        return e
    result = e if expanded is None else expanded
    if result.has(Hyper):
        result = _incomplete_gamma_forms(result)
    return result


def _incomplete_gamma_forms(e: Expr) -> Expr:
    """``1F1(1; b; z)``, which ``hyperexpand`` leaves alone, written
    ``(b - 1) z**(1 - b) exp(z) lowergamma(b - 1, z)`` (DLMF 8.5.1) for
    ``b`` not an integer below 2 and ``z`` not negative.

    >>> from sympy_extras.core import Rational, Symbol
    >>> from sympy_extras.functions import hyper
    >>> from sympy_extras.integrals.slater import _incomplete_gamma_forms
    >>> z = Symbol('z')
    >>> _incomplete_gamma_forms(hyper([1], [Rational(5, 3)], z))
    2*exp(z)*lowergamma(2/3, z)/(3*z**(2/3))
    """
    replacement: dict[Basic, Basic] = {}
    for h in e.atoms(Hyper):
        if not isinstance(h, Hyper):
            continue
        ap = list(h.ap)
        bq = list(h.bq)
        if ap == [S.One] and len(bq) == 1:
            b = bq[0]
            z = h.argument
            if ((b - 1).is_integer and (b - 1).is_nonpositive) or z.is_negative:
                # a negative argument gives lowergamma of a polar number,
                # which no branch resolves: the series stays as it is
                continue
            replacement[h] = (b - 1) * z**(1 - b) * exp(z) * lowergamma(b - 1, z)
    if not replacement:
        return e
    return _xreplace(e, replacement)


def slater_expansion(g: Expr, assumptions: Assumptions = None) -> Expr:
    """A :func:`~sympy_extras.functions.special.hyper.meijerg` expression
    as a sum of hypergeometric functions by Slater's theorem, each then
    expanded by ``hyperexpand``; with ``p = q``, the assumptions decide
    whether ``|z| < 1`` and otherwise a ``Piecewise`` on it is returned.
    The expression is returned unchanged when the theorem does not apply
    (a logarithmic case ``hyperexpand`` cannot expand either, or ``|z| = 1``).

    Examples
    ========

    >>> from sympy_extras.core import Rational, symbols
    >>> from sympy_extras.functions import meijerg
    >>> from sympy_extras.integrals.slater import slater_expansion
    >>> z, a = symbols('z a', positive=True)
    >>> slater_expansion(meijerg([1 - a], [], [0], [], z))
    gamma(a)/(z + 1)**a
    >>> slater_expansion(meijerg([1], [], [Rational(1, 2)], [0], z))
    sqrt(pi)*erf(sqrt(z))
    """
    if not isinstance(g, MeijerGFunction):
        return g
    value = expand_meijerg(MeijerG(S.One, g.an, g.aother, g.bm, g.bother, g.argument), assumptions)
    return g if value is None else value.value


def expand_meijerg(g: MeijerG, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``g`` (with its prefactor) expanded by Slater's theorem, see
    :func:`slater_expansion`; ``None`` when no expansion is found.

    >>> from sympy_extras.core import Integer, Symbol
    >>> from sympy_extras.integrals.slater import MeijerG, expand_meijerg
    >>> z = Symbol('z', positive=True)
    >>> expand_meijerg(MeijerG(Integer(3), [], [], [Integer(0)], [], z))
    ConditionalValue(3*exp(-z))
    """
    try:
        return _expand_meijerg(g, assumptions)
    except AttributeError:
        # SymPy 1.14 (through the bridge): the numerical evaluation of a
        # meijerg with polar arguments raises AttributeError inside the
        # assumptions system
        return None


def _expand_meijerg(g: MeijerG, assumptions: Assumptions) -> Optional[ConditionalValue]:
    if g.p < g.q:
        series = _slater_series(g)
    elif g.p > g.q:
        series = _slater_series(g.reflected())
    else:
        inside = ask(abs_(g.z) < 1, assumptions)
        if inside is True:
            series = _slater_series(g)
        elif inside is False and ask(abs_(g.z) > 1, assumptions) is True:
            series = _slater_series(g.reflected())
        elif inside is False:
            # |z| = 1: the series of the |z| < 1 case, continued analytically
            # by hyperexpand when it evaluates the hypergeometric functions;
            # only where every term is regular: at z = 1 itself a 2F1 with
            # Re(sum b - sum a) <= 0 is singular, the Gauss summation of it
            # meaningless, and the singularities of the terms cancel in the
            # sum (the bug: Integral(airyai(x)**2, (x, 0, oo)) came out as zoo)
            series = _slater_series(g)
            if series is not None:
                if not all(_converges_on_the_circle(h, assumptions) for h in series.atoms(Hyper)):
                    return None
                series = _hyperexpand(series)
                if series.has(Hyper) or series.has(S.ComplexInfinity, S.NaN):
                    return None
        else:
            small = _slater_series(g)
            large = _slater_series(g.reflected())
            if small is None or large is None:
                # the logarithmic case on one side (the Parseval integrand
                # of exp(-s*t)*erf(sqrt(t)) has b's differing by an integer):
                # the expansion of the G-function, one formula on both sides
                try:
                    expanded = _hyperexpand(g.as_expr())
                except ValueError:
                    return None
                if expanded.has(MeijerGFunction, Hyper):
                    return None
                return ConditionalValue(g.prefactor * expanded)
            small, large = _hyperexpand(small), _hyperexpand(large)
            if _same(small, large, assumptions):
                # one formula on both sides of |z| = 1, hence on it as well:
                # the one real as written at a sample of the parameters (the
                # small side's (pi/2 - asin(s/k))/sqrt(k**2 - s**2) is real by
                # a cancellation of imaginary parts for s > k); neither
                # real as written, the sides stay apart for the assumptions
                # to choose from
                choice = _real_choice(small, large, assumptions)
                if choice is not None:
                    return ConditionalValue(g.prefactor * choice)
            return ConditionalValue(g.prefactor * piecewise((small, abs_(g.z) < 1), (large, true)),
                                    or_(abs_(g.z) > 1, abs_(g.z) < 1))
    if series is None:
        try:
            expanded = _hyperexpand(g.as_expr())
        except ValueError:
            # meijerg refuses a_j - b_k a positive integer
            return None
        if expanded.has(MeijerGFunction):
            return None
        return ConditionalValue(g.prefactor * expanded)
    return ConditionalValue(g.prefactor * _hyperexpand(series))


def _converges_on_the_circle(h: Basic, assumptions: Assumptions) -> bool:
    """Whether the hypergeometric function is regular at its argument on
    ``|z| = 1``, so that the value ``hyperexpand`` gives there is its
    analytic continuation: ``pFq`` with ``p = q + 1`` is analytic off
    ``[1, oo)``, and at ``z = 1`` itself needs ``Re(sum(b) - sum(a)) > 0``
    (a polynomial, one of the ``a`` a non-positive integer, always is)."""
    if not isinstance(h, Hyper):
        return True
    ap = list(h.ap)
    bq = list(h.bq)
    if any(a.is_integer and a.is_nonpositive for a in ap):
        return True
    if len(ap) < len(bq) + 1:
        return True
    if len(ap) > len(bq) + 1:
        return False
    if h.argument != 1:
        return True
    excess = re(add(*bq) - add(*ap))
    return excess.is_positive is True or ask(excess > 0, assumptions) is True


def _is_function(node: Basic) -> bool:
    """Whether the node is a function application (SymPy's ``Function``,
    the hypergeometric functions included)."""
    return isinstance(node, (Application, Hyper, MeijerGFunction))


def _real_choice(small: Expr, large: Expr, assumptions: Assumptions) -> Optional[Expr]:
    """``small``, the expansion for ``|z| < 1``, as the one formula for
    both sides when it is real as written at a sample of the parameters
    (every function in it evaluates real) or the value itself is complex
    (complex scales); ``None`` when the value is real by a cancellation of
    imaginary parts only, so that the caller keeps the two sides apart
    for the assumptions to choose from (``(pi/2 - asin(s/k))/sqrt(k**2 - s**2)``
    for ``s > k``, where the other side gives ``acosh(s/k)/sqrt(s**2 - k**2)``)."""
    parameters: list[Symbol] = sorted(small.free_symbols | large.free_symbols, key=lambda s: s.name)
    values = sample_values(parameters, assumptions, random.Random(str((small, large)))) if parameters else {}
    if values is None:
        return small
    number = attempt(lambda: reliable_value(small, 20, values), settings.timeout, native=True)
    if number is None:
        return small
    try:
        # a zero of unknown precision (0.e-26) is not comparable: as floats
        imaginary, size = abs(float(im(number))), abs(complex(number))
    except (TypeError, ValueError):
        return small
    if imaginary >= 1e-12 * (1 + size):
        return small
    rule: dict[Basic, Basic] = {s: v for s, v in values.items()}
    for atom in small.find(_is_function):
        if not isinstance(atom, Expr):
            continue
        if evalf(_polar_as_exp(_xreplace(atom, rule)), 20).is_extended_real is False:
            return None
    return small


def _is_principal_polar(node: Basic) -> bool:
    if not isinstance(node, ExpPolar):
        return False
    part = im(_expr_arg(node))
    return isinstance(lt(-pi, part), BooleanTrue) and isinstance(le(part, pi), BooleanTrue)


def _expr_arg(node: Basic) -> Expr:
    argument = node.args[0]
    if not isinstance(argument, Expr):
        raise TypeError("an expression is expected, got %s" % (argument,))
    return argument


def _as_exp(node: Basic) -> Basic:
    return exp(_expr_arg(node))


def _polar_as_exp(e: Expr) -> Expr:
    """``exp_polar(z)`` with ``-pi < im(z) <= pi`` written ``exp(z)``, as
    SymPy's ``exp_polar._eval_evalf`` evaluates it: ``exp(I*pi)`` is
    exactly ``-1``, where the numerical exponential of ``I*pi`` leaves an
    imaginary part of ``1e-28`` which makes ``elliptic_k(4*exp_polar(I*pi)/a**2)``
    a non-real number."""
    replaced = e.replace(_is_principal_polar, _as_exp)
    if not isinstance(replaced, Expr):
        raise TypeError("an expression is expected, got %s" % (replaced,))
    return replaced


def _same(a: Expr, b: Expr, assumptions: Assumptions) -> bool:
    """Whether two closed forms are the same function (by simplification,
    then numerically)."""
    if a.has(Hyper) or b.has(Hyper):
        return False
    difference = attempt(lambda: simplify(a - b), settings.timeout, native=True)
    if difference is not None and difference == 0:
        return True
    return numerically_equal(a, b, assumptions)


def line_conditions(g: MeijerG, lower: Expr, upper: Expr) -> Optional[Boolean]:
    """The condition for the Mellin–Barnes integral along a vertical line
    in the strip ``lower < Re s < upper`` to converge absolutely (DLMF
    16.17.1): ``delta > 0`` and ``|arg z| < delta*pi``, or ``|arg z| =
    delta*pi`` with the integrand decaying faster than ``1/|t|``; ``None``
    when it fails.

    >>> from sympy_extras.core import Integer, Symbol, oo
    >>> from sympy_extras.integrals.slater import MeijerG, line_conditions
    >>> z = Symbol('z', positive=True)
    >>> line_conditions(MeijerG(Integer(1), [], [], [Integer(0)], [], z), Integer(0), oo)
    True
    """
    delta = g.delta
    positive_delta = delta.p > 0
    argument = arg_(g.z)
    if positive_delta:
        strict = abs_(argument) < delta * pi
        if isinstance(strict, BooleanTrue) or argument == 0 or _positive_argument(g.z):
            return true
    if delta.p < 0:
        return None
    # the power of |t| in the modulus of the integrand at Re s = sigma:
    # sum over the numerator of (Re a + b sigma - 1/2), minus the denominator
    sigma = _sigma(g, lower, upper)
    if sigma is None:
        return None
    exponent: Expr = S.Zero
    for a in g.bm:
        exponent = exponent + (a - sigma - S.Half)
    for a in g.an:
        exponent = exponent + (1 - a + sigma - S.Half)
    for b in g.bq:
        exponent = exponent - (1 - b + sigma - S.Half)
    for a in g.ap:
        exponent = exponent - (a - sigma - S.Half)
    boundary = and_(abs_(argument) <= delta * pi, exponent < -1)
    if positive_delta:
        return or_(abs_(argument) < delta * pi, boundary)
    return boundary


def _positive_argument(z: Expr) -> bool:
    """Whether ``z`` is a product of powers of positive numbers and of
    absolute values (of nonzero scales), hence positive."""
    factors = z.args if isinstance(z, Mul) else (z,)
    for factor in factors:
        if not isinstance(factor, Expr):
            return False
        base = factor.base if isinstance(factor, Pow) else factor
        if not (base.is_positive or isinstance(base, Abs)):
            return False
    return True


def _sigma(g: MeijerG, lower: Expr, upper: Expr) -> Optional[Expr]:
    """A point of the strip at which the decay of the integrand is best:
    the exponent decreases with ``sigma`` when ``p < q``."""
    if g.p == g.q:
        if not isinstance(lower, NegativeInfinity):
            return lower
        if not isinstance(upper, Infinity):
            return upper
        return S.Zero
    if g.p < g.q:
        return None if isinstance(upper, Infinity) else upper
    return None if isinstance(lower, NegativeInfinity) else lower

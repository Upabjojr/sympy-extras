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
functions by :func:`sympy.simplify.hyperexpand.hyperexpand`. When two of
the `b_j` differ by an integer (a logarithmic case) the expansion is
left to SymPy's ``hyperexpand`` of the G-function, which takes the limit.

Examples
========

>>> from sympy import symbols, meijerg, S
>>> from sympy_extras.integrals.slater import slater_expansion
>>> z = symbols('z', positive=True)
>>> slater_expansion(meijerg([], [], [0], [], z))
exp(-z)
>>> slater_expansion(meijerg([1], [], [S.Half], [0], z))
sqrt(pi)*erf(sqrt(z))

References
==========

.. [Slater] L. J. Slater, *Generalized hypergeometric functions*,
   Cambridge University Press, 1966, section 5.2.
.. [DLMF] NIST Digital Library of Mathematical Functions, chapter 16.17,
   https://dlmf.nist.gov/16.17.
"""
from __future__ import annotations

from math import lcm
from typing import Optional, Sequence

from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.add import Add
from sympy.core.numbers import Integer, Rational, nan, pi, oo, zoo
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.series.limits import Limit, limit
from sympy.functions.elementary.complexes import Abs, arg as arg_, re
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import gamma, lowergamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.logic.boolalg import And, Boolean, true
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import simplify

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.settings import settings
from .conditions import ConditionalValue, numerically_equal
from .mellin import GammaFactor, GammaQuotient

__all__ = ['MeijerG', 'mellin_barnes', 'slater_expansion', 'expand_meijerg']


class MeijerG:
    """A G-function `G^{m,n}_{p,q}(z \\mid a_n, a_p; b_m, b_q)` with a
    multiplicative prefactor."""

    def __init__(self, prefactor: Expr, an: Sequence[Expr], ap: Sequence[Expr],
                 bm: Sequence[Expr], bq: Sequence[Expr], z: Expr) -> None:
        self.prefactor = prefactor
        self.an = tuple(an)
        self.ap = tuple(ap)
        self.bm = tuple(bm)
        self.bq = tuple(bq)
        self.z = z

    def __repr__(self) -> str:
        return "MeijerG(%s, %s)" % (self.prefactor, self.as_sympy())

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

    def as_sympy(self) -> Expr:
        """The :class:`~sympy.functions.special.hyper.meijerg` expression
        (without the prefactor)."""
        return as_expr(meijerg(list(self.an), list(self.ap), list(self.bm), list(self.bq), self.z))

    def reflected(self) -> MeijerG:
        """The same function written with argument ``1/z``."""
        return MeijerG(self.prefactor, [1 - b for b in self.bm], [1 - b for b in self.bq],
                       [1 - a for a in self.an], [1 - a for a in self.ap], 1 / self.z)


def _multiplication(shift: Expr, k: int) -> tuple[Expr, Expr, list[GammaFactor]]:
    """Gauss's multiplication formula for ``Gamma(shift + k*u)``: the
    constant, the base of the power ``|k|**(k*u)`` and the factors
    ``Gamma((shift + j)/|k| + sign(k)*u)``."""
    size = abs(k)
    constant = as_expr((2 * pi)**Rational(1 - size, 2) * Integer(size)**(shift - S.Half))
    sign = S.One if k > 0 else S.NegativeOne
    factors: list[GammaFactor] = [(as_expr((shift + j) / size), Rational(sign)) for j in range(size)]
    return constant, Integer(size), factors


def mellin_barnes(quotient: GammaQuotient) -> Optional[MeijerG]:
    """The G-function equal to ``(1/2 pi i) Integral(quotient(t), t)``
    along a vertical line inside the strip; ``None`` when the quotient has
    an ``extra`` factor.

    Examples
    ========

    >>> from sympy import S
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
            k = int(scale * lam)
            if k == 0:
                constant = constant * gamma(shift) if numerator else constant / gamma(shift)
                continue
            unit: list[GammaFactor] = [(shift, Rational(k))]
            if abs(k) > 1:
                c, base, unit = _multiplication(shift, k)
                if numerator:
                    constant = constant * c
                    z = z * base**k
                else:
                    constant = constant / c
                    z = z * base**(-k)
            for a, b in unit:
                if numerator and b > 0:
                    an.append(1 - a)
                elif numerator:
                    bm.append(a)
                elif b > 0:
                    bq.append(1 - a)
                else:
                    ap.append(a)
    return MeijerG(as_expr(constant), an, ap, bm, bq, as_expr(z))


def _integer_difference(a: Expr, b: Expr) -> Optional[bool]:
    d = as_expr(a - b)
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
    sign = S.NegativeOne**(g.p - g.m - g.n)
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
    return as_expr(total)


def _logarithmic_case(g: MeijerG, coincident: list[int]) -> Optional[Expr]:
    # an irrational perturbation, so that the moved parameters do not
    # differ by integers any more
    epsilon = Dummy('epsilon', positive=True, rational=False)
    bm = [b + (coincident.index(i) + 1) * epsilon if i in coincident else b for i, b in enumerate(g.bm)]
    perturbed = MeijerG(g.prefactor, g.an, g.ap, bm, g.bq, g.z)
    series = _slater_series(perturbed)
    if series is None:
        return None
    expanded = _hyperexpand(series)
    if expanded.has(hyper):
        return None
    value = attempt(lambda: as_expr(limit(expanded, epsilon, 0)), settings.timeout)
    if value is None or value.has(oo, zoo, nan) or value.has(Limit):
        return None
    return value


def _hyperexpand(e: Expr) -> Expr:
    try:
        expanded = attempt(lambda: as_expr(hyperexpand(e)), settings.timeout)
    except AttributeError:
        # SymPy 1.14: meijerg._eval_evalf fails on some arguments
        # (AttributeError: 'NoneType' object has no attribute 'has')
        return e
    result = e if expanded is None else expanded
    if result.has(hyper):
        result = _incomplete_gamma_forms(result)
    return result


def _incomplete_gamma_forms(e: Expr) -> Expr:
    """``1F1(1; b; z)``, which ``hyperexpand`` leaves alone, written
    ``(b - 1) z**(1 - b) exp(z) lowergamma(b - 1, z)`` (DLMF 8.5.1) for
    ``b`` not an integer below 2 and ``z`` not negative.

    >>> from sympy import hyper, symbols, Rational
    >>> from sympy_extras.integrals.slater import _incomplete_gamma_forms
    >>> z = symbols('z')
    >>> _incomplete_gamma_forms(hyper((1,), (Rational(5, 3),), z))
    2*exp(z)*lowergamma(2/3, z)/(3*z**(2/3))
    """
    replacement: dict[Expr, Expr] = {}
    for h in e.atoms(hyper):
        ap = [as_expr(a) for a in h.ap]
        bq = [as_expr(b) for b in h.bq]
        if ap == [S.One] and len(bq) == 1:
            b = bq[0]
            z = as_expr(h.argument)
            if (b - 1).is_integer and (b - 1).is_nonpositive or z.is_negative:
                # a negative argument gives lowergamma of a polar number,
                # which no branch resolves: the series stays as it is
                continue
            replacement[as_expr(h)] = (b - 1) * z**(1 - b) * exp(z) * lowergamma(b - 1, z)
    if not replacement:
        return e
    return as_expr(e.xreplace(replacement))


def slater_expansion(g: Expr, assumptions: Assumptions = None) -> Expr:
    """A :class:`~sympy.functions.special.hyper.meijerg` expression as a
    sum of hypergeometric functions by Slater's theorem, each then
    expanded by ``hyperexpand``; with ``p = q``, the assumptions decide
    whether ``|z| < 1`` and otherwise a ``Piecewise`` on it is returned.
    The expression is returned unchanged when the theorem does not apply
    (a logarithmic case SymPy cannot expand either, or ``|z| = 1``).

    Examples
    ========

    >>> from sympy import symbols, meijerg, S
    >>> from sympy_extras.integrals.slater import slater_expansion
    >>> z, a = symbols('z a', positive=True)
    >>> slater_expansion(meijerg([1 - a], [], [0], [], z))
    gamma(a)/(z + 1)**a
    >>> slater_expansion(meijerg([1], [], [S.Half], [0], z))
    sqrt(pi)*erf(sqrt(z))
    """
    if not isinstance(g, meijerg):
        return g
    an = [as_expr(a) for a in g.an]
    ap = [as_expr(a) for a in g.aother]
    bm = [as_expr(b) for b in g.bm]
    bq = [as_expr(b) for b in g.bother]
    value = expand_meijerg(MeijerG(S.One, an, ap, bm, bq, as_expr(g.argument)), assumptions)
    return g if value is None else value.value


def expand_meijerg(g: MeijerG, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``g`` (with its prefactor) expanded by Slater's theorem, see
    :func:`slater_expansion`; ``None`` when no expansion is found."""
    try:
        return _expand_meijerg(g, assumptions)
    except AttributeError:
        # SymPy 1.14: the numerical evaluation of a meijerg with polar
        # arguments raises AttributeError inside the assumptions system
        return None


def _expand_meijerg(g: MeijerG, assumptions: Assumptions) -> Optional[ConditionalValue]:
    if g.p < g.q:
        series = _slater_series(g)
    elif g.p > g.q:
        series = _slater_series(g.reflected())
    else:
        inside = ask(Abs(g.z) < 1, assumptions)
        if inside is True:
            series = _slater_series(g)
        elif inside is False and ask(Abs(g.z) > 1, assumptions) is True:
            series = _slater_series(g.reflected())
        elif inside is False:
            # |z| = 1: the series of the |z| < 1 case, continued analytically
            # by hyperexpand when it evaluates the hypergeometric functions;
            # only when every series converges there (Re(sum b - sum a) > 0):
            # the Gauss summation of a divergent 2F1 at 1 is meaningless, and
            # the singularities of the terms cancel in the sum (the bug:
            # Integral(airyai(x)**2, (x, 0, oo)) came out as zoo)
            series = _slater_series(g)
            if series is not None:
                if not all(_converges_on_the_circle(h, assumptions) for h in series.atoms(hyper)):
                    return None
                series = _hyperexpand(series)
                if series.has(hyper) or series.has(zoo, nan):
                    return None
        else:
            small = _slater_series(g)
            large = _slater_series(g.reflected())
            if small is None or large is None:
                return None
            small, large = _hyperexpand(small), _hyperexpand(large)
            if _same(small, large, assumptions):
                # one formula on both sides of |z| = 1, hence on it as well
                return ConditionalValue(g.prefactor * small)
            return ConditionalValue(g.prefactor * Piecewise((small, Abs(g.z) < 1), (large, True)),
                                    as_boolean(Abs(g.z) > 1) | as_boolean(Abs(g.z) < 1))
    if series is None:
        try:
            expanded = _hyperexpand(g.as_sympy())
        except ValueError:
            # SymPy refuses a_j - b_k a positive integer
            return None
        if expanded.has(meijerg):
            return None
        return ConditionalValue(g.prefactor * expanded)
    return ConditionalValue(g.prefactor * _hyperexpand(series))


def _converges_on_the_circle(h: hyper, assumptions: Assumptions) -> bool:
    """Whether the hypergeometric series converges at its argument on
    ``|z| = 1``: for ``pFq`` with ``p = q + 1``, ``Re(sum(b) - sum(a)) > 0``
    at ``z = 1`` and ``> -1`` elsewhere on the circle (a polynomial, one of
    the ``a`` a non-positive integer, always converges)."""
    ap = [as_expr(a) for a in h.ap]
    bq = [as_expr(b) for b in h.bq]
    if any(a.is_integer and a.is_nonpositive for a in ap):
        return True
    if len(ap) < len(bq) + 1:
        return True
    if len(ap) > len(bq) + 1:
        return False
    bound = S.Zero if as_expr(h.argument) == 1 else S.NegativeOne
    excess = as_expr(re(Add(*bq) - Add(*ap)) - bound)
    return excess.is_positive is True or ask(as_boolean(excess > 0), assumptions) is True


def _same(a: Expr, b: Expr, assumptions: Assumptions) -> bool:
    """Whether two closed forms are the same function (by simplification,
    then by SymPy's numerical test ``equals``)."""
    if a.has(hyper) or b.has(hyper):
        return False
    difference = attempt(lambda: as_expr(simplify(a - b)), settings.timeout)
    if difference is not None and difference == 0:
        return True
    return numerically_equal(a, b, assumptions)


def line_conditions(g: MeijerG, lower: Expr, upper: Expr) -> Optional[Boolean]:
    """The condition for the Mellin–Barnes integral along a vertical line
    in the strip ``lower < Re s < upper`` to converge absolutely (DLMF
    16.17.1): ``delta > 0`` and ``|arg z| < delta*pi``, or ``|arg z| =
    delta*pi`` with the integrand decaying faster than ``1/|t|``; ``None``
    when it fails."""
    delta = g.delta
    argument = as_expr(arg_(g.z))
    if delta > 0:
        strict = as_boolean(Abs(argument) < delta * pi)
        if strict is true or argument == 0 or _positive_argument(g.z):
            return true
    if delta < 0:
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
    boundary = as_boolean(And(Abs(argument) <= delta * pi, exponent < -1))
    if delta > 0:
        return as_boolean((Abs(argument) < delta * pi) | boundary)
    return boundary


def _positive_argument(z: Expr) -> bool:
    """Whether ``z`` is a product of powers of positive numbers and of
    absolute values (of nonzero scales), hence positive."""
    factors = z.args if isinstance(z, Mul) else (z,)
    for factor in factors:
        base = as_expr(factor.base) if isinstance(factor, Pow) else as_expr(factor)
        if not (base.is_positive or isinstance(base, Abs)):
            return False
    return True


def _sigma(g: MeijerG, lower: Expr, upper: Expr) -> Optional[Expr]:
    """A point of the strip at which the decay of the integrand is best:
    the exponent decreases with ``sigma`` when ``p < q``."""
    if g.p == g.q:
        if lower != -oo:
            return lower
        if upper != oo:
            return upper
        return S.Zero
    if g.p < g.q:
        return None if upper == oo else upper
    return None if lower == -oo else lower

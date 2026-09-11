"""Mellin transforms written as quotients of gamma functions, the
representation behind the Marichev–Adamchik method of definite
integration.

The Mellin transform of a function `f` on `(0, \\infty)`,

.. math::

    F(s) = \\int_0^\\infty x^{s-1} f(x)\\, dx,

converges in a vertical strip `\\alpha < \\operatorname{Re} s < \\beta`, and for
the functions of mathematical physics it is a **gamma quotient**: a
constant times powers `\\rho^{-s}` times a quotient of products of
`\\Gamma(a + b s)` with rational `b`. For instance

.. math::

    \\int_0^\\infty x^{s-1} e^{-x}\\, dx = \\Gamma(s), \\qquad
    \\int_0^\\infty \\frac{x^{s-1}}{1 + x}\\, dx = \\Gamma(s)\\Gamma(1 - s), \\qquad
    \\int_0^\\infty x^{s-1} \\sin x\\, dx = \\Gamma(s) \\sin\\frac{\\pi s}{2}.

Such a quotient is exactly the integrand of the Mellin–Barnes integral
defining a Meijer G-function: this is Marichev's observation [Marichev]_,
turned into an algorithm by Adamchik and Marichev [Adamchik]_. The
transforms are closed under the operations an integrand undergoes --
multiplication by a power of `x` (a shift of `s`), scaling and powering of
the argument (an affine change of `s`), products of two functions (the
Mellin convolution, which turns the product of two quotients into a
G-function evaluated at the ratio of the scales) -- so that a definite
integral of a product of such functions is a G-function, which Slater's
theorem then writes as hypergeometric series (:mod:`.slater`).

This module holds the representation, :class:`GammaQuotient`, the
operations on it, and the table of transforms of the elementary and
special functions (:func:`mellin_kernel`), each with its strip of
convergence and the conditions on its parameters. The entries are the
classical ones of [Erdelyi]_, chapter VI, and of [PBM]_, chapter 8.4, in
the gamma-quotient form; :func:`mellin_transform` looks up a product of
them. The transforms are checked in the tests against numerical
integration and against :func:`sympy.mellin_transform`.

Examples
========

>>> from sympy import exp, symbols, sin, besselj
>>> from sympy_extras.integrals.mellin import mellin_transform
>>> x, s = symbols('x s')
>>> mellin_transform(exp(-2*x), x, s)
MellinTransform(gamma(s)/2**s, (0, oo))
>>> mellin_transform(sin(x)/x, x, s)
MellinTransform(2**s*sqrt(pi)*gamma(s/2)/(4*gamma(3/2 - s/2)), (0, 2))
>>> mellin_transform(besselj(0, 3*x)/(x**2 + 4), x, s) is None
True

References
==========

.. [Marichev] O. I. Marichev, *Handbook of integral transforms of higher
   transcendental functions: theory and algorithmic tables*, Ellis
   Horwood, 1983.
.. [Adamchik] V. S. Adamchik, O. I. Marichev, *The algorithm for
   calculating integrals of hypergeometric type functions and its
   realization in REDUCE system*, ISSAC 1990, pp. 212-224.
.. [Erdelyi] A. Erdélyi, W. Magnus, F. Oberhettinger, F. G. Tricomi,
   *Tables of integral transforms*, vol. I, McGraw-Hill, 1954, chapter VI.
.. [PBM] A. P. Prudnikov, Yu. A. Brychkov, O. I. Marichev, *Integrals and
   series*, vol. 3: More special functions, Gordon and Breach, 1990,
   chapter 8.4.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.add import Add
from sympy.core.exprtools import factor_terms
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, oo, pi
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs, arg as arg_
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import sinh, cosh
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.trigonometric import sin, cos, atan
from sympy.functions.special.bessel import besselj, bessely, besseli, besselk
from sympy.functions.special.error_functions import erf, erfc, Ei, expint, Si, Ci
from sympy.functions.special.gamma_functions import gamma, polygamma
from sympy.functions.special.zeta_functions import zeta, dirichlet_eta, lerchphi
from sympy.functions.special.delta_functions import Heaviside
from sympy.logic.boolalg import And, Boolean, true

from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element

__all__ = ['GammaFactor', 'GammaQuotient', 'MellinTransform', 'Kernel', 'mellin_kernel',
           'mellin_transform', 'monomial', 'KERNELS', 'Match', 'Product', 'decompose_integrand', 'poles_separated']

#: `\\Gamma(\\text{shift} + \\text{scale} \\cdot s)`
GammaFactor = tuple[Expr, Rational]

#: the variable of the ``extra`` factor of a quotient
_S = Dummy('s')


def _gamma_product(factors: Sequence[GammaFactor], s: Expr) -> Expr:
    result: Expr = S.One
    for shift, scale in factors:
        result = result * gamma(shift + scale * s)
    return result


class GammaQuotient:
    """A function of ``s`` of the form

    .. math::

        C \\prod_k \\rho_k^{c_k s}\\, E(s)
        \\frac{\\prod_i \\Gamma(a_i + b_i s)}{\\prod_j \\Gamma(c_j + d_j s)},

    holding in the strip ``lower < Re s < upper`` under ``condition``.

    ``E(s)`` (``extra``) is a factor which is not a gamma quotient (a zeta
    function, a power of ``s``) allowed for the transforms which are used
    alone; a quotient with such a factor cannot be turned into a
    G-function.

    Examples
    ========

    >>> from sympy import S, symbols
    >>> from sympy_extras.integrals.mellin import GammaQuotient
    >>> s = symbols('s')
    >>> q = GammaQuotient(1, [(2, -1)], [(0, S.One)], [], 0)
    >>> q.as_expr(s)
    gamma(s)/2**s
    >>> q.compose(1, -2).as_expr(s)
    2**(2*s)*gamma(1 - 2*s)/2
    >>> q.compose(1, -2).strip
    (-oo, 1/2)
    """

    def __init__(self, constant: ExprLike, bases: Sequence[tuple[ExprLike, ExprLike]],
                 numerator: Sequence[tuple[ExprLike, ExprLike]],
                 denominator: Sequence[tuple[ExprLike, ExprLike]],
                 lower: ExprLike = -oo, upper: ExprLike = oo, condition: object = True,
                 extra: ExprLike = 1) -> None:
        self.constant: Expr = as_expr(constant)
        self.bases: tuple[tuple[Expr, Expr], ...] = tuple(
            (as_expr(b), as_expr(c)) for b, c in bases if as_expr(c) != 0)
        self.numerator: tuple[GammaFactor, ...] = tuple(_factor(a, b) for a, b in numerator)
        self.denominator: tuple[GammaFactor, ...] = tuple(_factor(a, b) for a, b in denominator)
        self.lower: Expr = as_expr(lower)
        self.upper: Expr = as_expr(upper)
        self.condition: Boolean = as_boolean(condition)
        self.extra: Expr = as_expr(extra)

    @property
    def strip(self) -> tuple[Expr, Expr]:
        """The strip of convergence ``(lower, upper)``."""
        return (self.lower, self.upper)

    def __repr__(self) -> str:
        return "GammaQuotient(%s, %s)" % (self.as_expr(Symbol('s')), self.strip)

    def as_expr(self, s: ExprLike) -> Expr:
        """The value at ``s`` as a SymPy expression."""
        s_ = as_expr(s)
        result = self.constant * self.extra.subs(_S, s_)
        for base, coefficient in self.bases:
            result = result * base**(coefficient * s_)
        return as_expr(result * _gamma_product(self.numerator, s_)
                       / _gamma_product(self.denominator, s_))

    def scaled(self, factor: ExprLike) -> GammaQuotient:
        """The quotient multiplied by a constant."""
        return GammaQuotient(self.constant * as_expr(factor), self.bases, self.numerator,
                             self.denominator, self.lower, self.upper, self.condition, self.extra)

    def times(self, other: GammaQuotient) -> GammaQuotient:
        """The product of two quotients on the intersection of their strips."""
        return GammaQuotient(self.constant * other.constant, self.bases + other.bases,
                             self.numerator + other.numerator,
                             self.denominator + other.denominator,
                             _max(self.lower, other.lower), _min(self.upper, other.upper),
                             And(self.condition, other.condition), self.extra * other.extra)

    def compose(self, p: ExprLike, q: ExprLike) -> GammaQuotient:
        """The quotient at ``p + q*s`` (``q`` a nonzero rational number)."""
        p_, q_ = as_expr(p), as_expr(q)
        if not isinstance(q_, Rational) or q_ == 0:
            raise ValueError("the scale must be a nonzero rational number, got %s" % (q_,))
        constant = self.constant
        bases: list[tuple[Expr, Expr]] = []
        for base, coefficient in self.bases:
            constant = constant * base**(coefficient * p_)
            bases.append((base, coefficient * q_))
        numerator = [(a + b * p_, b * q_) for a, b in self.numerator]
        denominator = [(a + b * p_, b * q_) for a, b in self.denominator]
        if q_ > 0:
            lower, upper = (self.lower - p_) / q_, (self.upper - p_) / q_
        else:
            lower, upper = (self.upper - p_) / q_, (self.lower - p_) / q_
        return GammaQuotient(constant, bases, numerator, denominator, lower, upper,
                             self.condition, self.extra.subs(_S, p_ + q_ * _S))

    def cancelled(self) -> GammaQuotient:
        """The quotient with the gamma factors common to the numerator and
        the denominator removed."""
        numerator = list(self.numerator)
        denominator: list[GammaFactor] = []
        for factor in self.denominator:
            if factor in numerator:
                numerator.remove(factor)
            else:
                denominator.append(factor)
        return GammaQuotient(self.constant, self.bases, numerator, denominator, self.lower,
                             self.upper, self.condition, self.extra)

    def strip_condition(self, s: ExprLike) -> Boolean:
        """The condition for ``s`` (a real number) to lie in the strip."""
        s_ = as_expr(s)
        parts: list[Boolean] = []
        if self.lower != -oo:
            parts.append(as_boolean(s_ > self.lower))
        if self.upper != oo:
            parts.append(as_boolean(s_ < self.upper))
        return as_boolean(And(*parts))

    def is_gamma_quotient(self) -> bool:
        """Whether there is no ``extra`` factor."""
        return self.extra == 1


def poles_separated(quotient: GammaQuotient) -> Optional[bool]:
    """Whether the poles of every ``Gamma(a + b s)`` with ``b > 0`` lie
    to the left of the strip and those with ``b < 0`` to the right, as
    the Mellin–Barnes representation of a G-function requires: ``True``,
    ``False``, or ``None`` when a comparison with symbolic parameters is
    undecided. A transform of the table must satisfy it: a factor
    ``1/s`` is ``Gamma(s)/Gamma(1 + s)`` when the strip lies right of 0
    and ``-Gamma(-s)/Gamma(1 - s)`` when it lies left of it.

    >>> from sympy import S
    >>> from sympy_extras.integrals.mellin import GammaQuotient, poles_separated
    >>> poles_separated(GammaQuotient(1, [], [(0, 1)], [(1, 1)], 0, 1))
    True
    >>> poles_separated(GammaQuotient(1, [], [(0, 1)], [(1, 1)], -1, 0))
    False
    """
    verdict: Optional[bool] = True
    for shift, scale in quotient.numerator:
        if scale == 0:
            continue
        # the rightmost pole of Gamma(shift + scale s) is at s = -shift/scale
        pole = as_expr(-shift / scale)
        if scale > 0:
            ok = _compare(pole, quotient.lower)
        else:
            ok = _compare(quotient.upper, pole)
        if ok is False:
            return False
        if ok is None:
            verdict = None
    return verdict


def _compare(a: Expr, b: Expr) -> Optional[bool]:
    """``a <= b``, ``None`` if undecided."""
    if b == oo or a == -oo:
        return True
    if b == -oo or a == oo:
        return False
    d = as_expr(b - a)
    if d.is_nonnegative:
        return True
    if d.is_negative:
        return False
    return None


def _factor(shift: ExprLike, scale: ExprLike) -> GammaFactor:
    scale_ = as_expr(scale)
    if not isinstance(scale_, Rational):
        raise ValueError("the coefficient of s in a gamma factor must be rational, got %s" % (scale_,))
    return (as_expr(shift), scale_)


def _max(a: Expr, b: Expr) -> Expr:
    if a == -oo:
        return b
    if b == -oo:
        return a
    return as_expr(Max(a, b))


def _min(a: Expr, b: Expr) -> Expr:
    if a == oo:
        return b
    if b == oo:
        return a
    return as_expr(Min(a, b))


class MellinTransform:
    """A Mellin transform: the transform as an expression in ``s``, its
    strip of convergence and the conditions on the parameters."""

    def __init__(self, transform: Expr, strip: tuple[Expr, Expr], condition: Boolean = true) -> None:
        self.transform = transform
        self.strip = strip
        self.condition = condition

    def __repr__(self) -> str:
        if self.condition is true:
            return "MellinTransform(%s, (%s, %s))" % (self.transform, self.strip[0], self.strip[1])
        return "MellinTransform(%s, (%s, %s), %s)" % (self.transform, self.strip[0], self.strip[1],
                                                       self.condition)


# ---------------------------------------------------------------------------
# The table

class Kernel:
    """One entry of the table: a function ``f(x)`` with its Mellin transform.

    Attributes
    ==========

    name : str
    quotient : GammaQuotient
        The transform of ``f(x)`` itself, in ``s``.
    parameters : tuple of Expr
        The parameters of the kernel (the order of a Bessel function).
    scale_condition : Boolean
        The condition on the scale ``beta`` (``|arg beta| < pi/2`` for
        ``exp(-beta*x)``) under which the transform of ``f(beta*x)`` is
        ``beta**(-s)`` times the transform of ``f(x)``; written with the
        symbol :data:`BETA`.
    """

    def __init__(self, name: str, quotient: GammaQuotient, parameters: Sequence[Expr] = (),
                 scale_condition: object = True, parity: int = 0) -> None:
        self.name = name
        self.quotient = quotient
        self.parameters = tuple(parameters)
        self.scale_condition: Boolean = as_boolean(scale_condition)
        #: ``1`` for an even function, ``-1`` for an odd one, ``0`` otherwise:
        #: ``f(beta x) = f(|beta| x)`` or ``sign(beta) f(|beta| x)`` for a real
        #: ``beta``, so that a real scale of unknown sign is allowed
        self.parity = parity

    def __repr__(self) -> str:
        return "Kernel(%s)" % self.name


#: the scale of the argument in the scale conditions of the kernels
BETA = Dummy('beta')

_HALF = S.Half
_ONE = S.One


def _positive_real(b: Expr) -> Boolean:
    return as_boolean(b > 0)


def _right_half_plane(b: Expr) -> Boolean:
    """``Re b > 0`` written as the argument condition (the condition an
    inequality on a real parameter reduces to)."""
    return as_boolean(Abs(arg_(b)) < pi / 2)


def _cut_plane(b: Expr) -> Boolean:
    """``b`` off the negative real axis."""
    return as_boolean(Abs(arg_(b)) < pi)


def _exp_kernel() -> Kernel:
    # e^{-x}: Gamma(s), Re s > 0
    return Kernel('exp', GammaQuotient(1, [], [(0, 1)], [], 0, oo), (), _right_half_plane(BETA))


def _exp_minus_one_kernel() -> Kernel:
    # e^{-x} - 1: Gamma(s) = -Gamma(1 + s) Gamma(-s) / Gamma(1 - s), -1 < Re s < 0
    # (the analytic continuation; the pole at 0 lies to the right of the strip)
    return Kernel('exp - 1', GammaQuotient(-1, [], [(1, 1), (0, -1)], [(1, -1)], -1, 0), (), _right_half_plane(BETA))


def _power_kernel(a: Expr) -> Kernel:
    # (1 + x)^{-a}: Gamma(s) Gamma(a - s) / Gamma(a), 0 < Re s < Re a
    return Kernel('(1 + x)**(-a)', GammaQuotient(1, [], [(0, 1), (a, -1)], [(a, 0)], 0, a), (a,),
                  _cut_plane(BETA))


def _beta_kernel(b: Expr) -> Kernel:
    # (1 - x)^{b-1} on (0, 1): Gamma(s) Gamma(b) / Gamma(s + b), Re s > 0, Re b > 0
    return Kernel('(1 - x)**(b - 1) theta(1 - x)',
                  GammaQuotient(1, [], [(0, 1), (b, 0)], [(b, 1)], 0, oo, b > 0), (b,),
                  _positive_real(BETA))


def _beta_upper_kernel(b: Expr) -> Kernel:
    # (x - 1)^{b-1} on (1, oo): Gamma(b) Gamma(1 - b - s) / Gamma(1 - s), Re s < 1 - Re b
    return Kernel('(x - 1)**(b - 1) theta(x - 1)',
                  GammaQuotient(1, [], [(b, 0), (1 - b, -1)], [(1, -1)], -oo, 1 - b, b > 0), (b,),
                  _positive_real(BETA))


def _log1p_kernel() -> Kernel:
    # log(1 + x): pi / (s sin(pi s)) = Gamma(1 + s) Gamma(-s)^2 / Gamma(1 - s), -1 < Re s < 0
    # (the poles at 0, 1, 2, ... to the right, at -1, -2, ... to the left)
    return Kernel('log(1 + x)', GammaQuotient(1, [], [(1, 1), (0, -1), (0, -1)], [(1, -1)], -1, 0), (),
                  _cut_plane(BETA))


def _sin_kernel() -> Kernel:
    # sin x: Gamma(s) sin(pi s / 2) = sqrt(pi) 2^{s-1} Gamma((1 + s)/2) / Gamma(1 - s/2), -1 < Re s < 1
    return Kernel('sin', GammaQuotient(sqrt(pi) / 2, [(2, 1)], [(_HALF, _HALF)], [(1, -_HALF)], -1, 1),
                  (), _positive_real(BETA), -1)


def _sin_minus_x_kernel() -> Kernel:
    # sin x - x: Gamma(s) sin(pi s/2) in -3 < Re s < -1, with the pole at -1 to the right:
    # -sqrt(pi) 2^(s-1) Gamma(3/2 + s/2) Gamma(-1/2 - s/2) / (Gamma(1 - s/2) Gamma(1/2 - s/2))
    return Kernel('sin - x', GammaQuotient(-sqrt(pi) / 2, [(2, 1)], [(Rational(3, 2), _HALF), (-_HALF, -_HALF)],
                                            [(1, -_HALF), (_HALF, -_HALF)], -3, -1),
                  (), _positive_real(BETA), -1)


def _cos_kernel() -> Kernel:
    # cos x: Gamma(s) cos(pi s / 2) = sqrt(pi) 2^{s-1} Gamma(s/2) / Gamma(1/2 - s/2), 0 < Re s < 1
    return Kernel('cos', GammaQuotient(sqrt(pi) / 2, [(2, 1)], [(0, _HALF)], [(_HALF, -_HALF)], 0, 1),
                  (), _positive_real(BETA), 1)


def _cos_minus_one_kernel() -> Kernel:
    # cos x - 1: Gamma(s) cos(pi s/2) in -2 < Re s < 0, with the pole at 0 to the right:
    # -sqrt(pi) 2^(s-1) Gamma(1 + s/2) Gamma(-s/2) / (Gamma(1 - s/2) Gamma(1/2 - s/2))
    return Kernel('cos - 1', GammaQuotient(-sqrt(pi) / 2, [(2, 1)], [(1, _HALF), (0, -_HALF)],
                                            [(1, -_HALF), (_HALF, -_HALF)], -2, 0),
                  (), _positive_real(BETA), 1)


def _atan_kernel() -> Kernel:
    # atan x: -pi / (2 s cos(pi s/2)) = Gamma(-s) Gamma(1/2 + s/2) Gamma(1/2 - s/2) / (2 Gamma(1 - s)),
    # -1 < Re s < 0 (the pole at 0 to the right of the strip)
    return Kernel('atan', GammaQuotient(_HALF, [], [(0, -1), (_HALF, _HALF), (_HALF, -_HALF)], [(1, -1)], -1, 0),
                  (), _positive_real(BETA), -1)


def _atan_minus_kernel() -> Kernel:
    # atan x - pi/2: the same quotient in 0 < Re s < 1
    return Kernel('atan - pi/2', GammaQuotient(-_HALF, [], [(0, 1), (_HALF, _HALF), (_HALF, -_HALF)], [(1, 1)], 0, 1),
                  (), _positive_real(BETA))


def _erfc_kernel() -> Kernel:
    # erfc x: Gamma((1 + s)/2) / (sqrt(pi) s), Re s > 0
    return Kernel('erfc', GammaQuotient(1 / sqrt(pi), [], [(_HALF, _HALF), (0, 1)], [(1, 1)], 0, oo), (),
                  _positive_real(BETA))


def _erf_kernel() -> Kernel:
    # erf x: -Gamma((1 + s)/2) / (sqrt(pi) s) = Gamma((1 + s)/2) Gamma(-s) / (sqrt(pi) Gamma(1 - s)),
    # -1 < Re s < 0
    return Kernel('erf', GammaQuotient(1 / sqrt(pi), [], [(_HALF, _HALF), (0, -1)], [(1, -1)], -1, 0), (),
                  _positive_real(BETA), -1)


def _e1_kernel() -> Kernel:
    # E_1(x) = -Ei(-x): Gamma(s) / s, Re s > 0
    return Kernel('E1', GammaQuotient(1, [], [(0, 1), (0, 1)], [(1, 1)], 0, oo), (), _right_half_plane(BETA))


def _expint_kernel(n: Expr) -> Kernel:
    # E_n(x): Gamma(s) / (s + n - 1), Re s > 0 (n not a nonpositive integer... Re s > max(0, 1 - Re n))
    return Kernel('expint', GammaQuotient(1, [], [(0, 1), (n - 1, 1)], [(n, 1)], Max(0, 1 - n), oo), (n,),
                  _right_half_plane(BETA))


def _besselj_kernel(nu: Expr) -> Kernel:
    # J_nu(x): 2^{s-1} Gamma((nu + s)/2) / Gamma(1 + (nu - s)/2), -Re nu < Re s < 3/2
    return Kernel('besselj', GammaQuotient(_HALF, [(2, 1)], [(nu / 2, _HALF)], [(1 + nu / 2, -_HALF)], -nu, Rational(3, 2)),
                  (nu,), _positive_real(BETA))


def _bessely_kernel(nu: Expr) -> Kernel:
    # Y_nu(x): -2^{s-1} Gamma((s + nu)/2) Gamma((s - nu)/2) cos(pi (s - nu)/2) / pi, |Re nu| < Re s < 3/2
    # cos(pi (s - nu)/2) = pi / (Gamma(1/2 + (s - nu)/2) Gamma(1/2 - (s - nu)/2))
    return Kernel('bessely', GammaQuotient(-_HALF, [(2, 1)], [(nu / 2, _HALF), (-nu / 2, _HALF)],
                                            [(_HALF - nu / 2, _HALF), (_HALF + nu / 2, -_HALF)], Abs(nu), Rational(3, 2)),
                  (nu,), _positive_real(BETA))


def _besselk_kernel(nu: Expr) -> Kernel:
    # K_nu(x): 2^{s-2} Gamma((s - nu)/2) Gamma((s + nu)/2), Re s > |Re nu|
    return Kernel('besselk', GammaQuotient(Rational(1, 4), [(2, 1)], [(-nu / 2, _HALF), (nu / 2, _HALF)], [], Abs(nu), oo),
                  (nu,), _right_half_plane(BETA))


def _exp_besseli_kernel(nu: Expr) -> Kernel:
    # e^{-x} I_nu(x): Gamma(s + nu) Gamma(1/2 - s) / (2^s sqrt(pi) Gamma(1 + nu - s)), -Re nu < Re s < 1/2
    return Kernel('exp(-x)*besseli', GammaQuotient(1 / sqrt(pi), [(_HALF, 1)], [(nu, 1), (_HALF, -1)], [(1 + nu, -1)], -nu, _HALF),
                  (nu,), _positive_real(BETA))


def _si_kernel() -> Kernel:
    # Si(x): -Gamma(s) sin(pi s/2) / s, -1 < Re s < 0, with the double pole at 0 to the right:
    # -pi Gamma(1 + s) Gamma(-s)^2 / (Gamma(s/2) Gamma(1 - s/2) Gamma(1 - s)^2)
    return Kernel('Si', GammaQuotient(-pi, [], [(1, 1), (0, -1), (0, -1)],
                                       [(0, _HALF), (1, -_HALF), (1, -1), (1, -1)], -1, 0),
                  (), _positive_real(BETA), -1)


def _si_minus_kernel() -> Kernel:
    # Si(x) - pi/2: the same quotient in 0 < Re s < 1
    return Kernel('Si - pi/2', GammaQuotient(-sqrt(pi) / 2, [(2, 1)], [(_HALF, _HALF), (0, 1)], [(1, -_HALF), (1, 1)], 0, 1),
                  (), _positive_real(BETA))


def _ci_kernel() -> Kernel:
    # Ci(x): -Gamma(s) cos(pi s/2) / s, 0 < Re s < 1
    return Kernel('Ci', GammaQuotient(-sqrt(pi) / 2, [(2, 1)], [(0, _HALF), (0, 1)], [(_HALF, -_HALF), (1, 1)], 0, 1),
                  (), _positive_real(BETA))


def _theta_lower_kernel() -> Kernel:
    # theta(1 - x): 1/s = Gamma(s) / Gamma(1 + s), Re s > 0
    return Kernel('theta(1 - x)', GammaQuotient(1, [], [(0, 1)], [(1, 1)], 0, oo), (), _positive_real(BETA))


def _theta_upper_kernel() -> Kernel:
    # theta(x - 1): -1/s = Gamma(-s) / Gamma(1 - s), Re s < 0
    return Kernel('theta(x - 1)', GammaQuotient(1, [], [(0, -1)], [(1, -1)], -oo, 0), (), _positive_real(BETA))


def _log_power_lower_kernel(k: Expr) -> Kernel:
    # (-log x)^k theta(1 - x): Gamma(k + 1) s^{-k-1}, Re s > 0, Re k > -1
    return Kernel('(-log(x))**k theta(1 - x)',
                  GammaQuotient(1, [], [(k + 1, 0)], [], 0, oo, k > -1, _S**(-k - 1)), (k,), _positive_real(BETA))


def _log_one_minus_kernel() -> Kernel:
    # log(1 - x) on (0, 1): -(psi(1 + s) + EulerGamma)/s, Re s > 0 (Erdelyi 6.6 (7))
    return Kernel('log(1 - x) theta(1 - x)',
                  GammaQuotient(-1, [], [], [], 0, oo, True, (polygamma(0, _S + 1) + S.EulerGamma) / _S), (),
                  _positive_real(BETA))


def _bose_kernel() -> Kernel:
    # 1/(e^x - 1): Gamma(s) zeta(s), Re s > 1
    return Kernel('1/(exp(x) - 1)', GammaQuotient(1, [], [(0, 1)], [], 1, oo, True, zeta(_S)), (), _positive_real(BETA))


def _fermi_kernel() -> Kernel:
    # 1/(e^x + 1): (1 - 2^{1-s}) Gamma(s) zeta(s) = Gamma(s) eta(s) (Dirichlet's eta, entire), Re s > 0
    return Kernel('1/(exp(x) + 1)', GammaQuotient(1, [], [(0, 1)], [], 0, oo, True, dirichlet_eta(_S)), (),
                  _positive_real(BETA))


def _csch_kernel() -> Kernel:
    # 1/sinh x: 2 (1 - 2^{-s}) Gamma(s) zeta(s), Re s > 1
    return Kernel('1/sinh(x)', GammaQuotient(2, [], [(0, 1)], [], 1, oo, True, (1 - 2**(-_S)) * zeta(_S)), (),
                  _positive_real(BETA), -1)


def _sech_kernel() -> Kernel:
    # 1/cosh x: 2 Gamma(s) beta(s) with Dirichlet's beta(s) = 2^{-s} Phi(-1, s, 1/2) (Lerch's transcendent,
    # entire in s), Re s > 0
    return Kernel('1/cosh(x)', GammaQuotient(2, [], [(0, 1)], [], 0, oo, True,
                                             2**(-_S) * lerchphi(-1, _S, S.Half)), (),
                  _positive_real(BETA), 1)


#: the kernels without parameters, by the head of the expression they match
KERNELS: dict[str, Kernel] = {k.name: k for k in [
    _exp_kernel(), _exp_minus_one_kernel(), _log1p_kernel(), _sin_kernel(), _sin_minus_x_kernel(),
    _cos_kernel(), _cos_minus_one_kernel(), _atan_kernel(), _atan_minus_kernel(), _erfc_kernel(),
    _erf_kernel(), _e1_kernel(), _si_kernel(), _si_minus_kernel(), _ci_kernel(), _theta_lower_kernel(),
    _theta_upper_kernel(), _bose_kernel(), _fermi_kernel(), _csch_kernel(), _sech_kernel()]}


# ---------------------------------------------------------------------------
# Matching

class Match:
    """A factor of an integrand recognised as ``constant * f(beta * x**gamma)``
    for a kernel ``f``."""

    def __init__(self, kernel: Kernel, beta: Expr, gamma_: Expr, constant: Expr = S.One,
                 cutoff: bool = False) -> None:
        self.kernel = kernel
        self.beta = beta
        self.gamma = gamma_
        self.constant = constant
        #: whether the kernel includes the cutoff at 1 of a finite range
        self.cutoff = cutoff

    def __repr__(self) -> str:
        return "Match(%s, beta=%s, gamma=%s, constant=%s)" % (self.kernel.name, self.beta, self.gamma, self.constant)

    def quotient(self, assumptions: Assumptions = None) -> GammaQuotient:
        """The Mellin transform of the factor: for ``gamma > 0``,
        ``M[f(beta x^gamma)](s) = beta^{-s/gamma} F(s/gamma) / gamma``, and
        for ``gamma < 0`` the same with ``1/|gamma|`` and the strip
        reflected. A real scale of unknown sign is written ``|beta|`` for
        an even or odd kernel (with the factor ``beta/|beta|`` for an odd
        one); otherwise the scale condition of the kernel is part of the
        condition."""
        g = self.gamma
        if not isinstance(g, Rational) or g == 0:
            raise ValueError("the power of x must be a nonzero rational number, got %s" % (g,))
        beta, constant = self.beta, self.constant
        condition = self.kernel.scale_condition.xreplace({BETA: beta})
        if self.kernel.parity != 0 and ask(beta > 0, assumptions) is not True \
                and (beta.is_extended_real or ask(element(beta, S.Reals), assumptions) is True):
            if self.kernel.parity < 0:
                constant = constant * beta / Abs(beta)
            beta = as_expr(Abs(beta))
            condition = true
        q = self.kernel.quotient.compose(0, 1 / g)
        q = GammaQuotient(q.constant * constant / Abs(g), q.bases + ((beta, -1 / g),),
                          q.numerator, q.denominator, q.lower, q.upper,
                          And(q.condition, condition), q.extra)
        return q


def monomial(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(beta, gamma)`` when ``e == beta * x**gamma`` with ``beta`` free of
    ``x`` and ``gamma`` a nonzero rational number, ``None`` otherwise.

    >>> from sympy import sqrt, symbols
    >>> from sympy_extras.integrals.mellin import monomial
    >>> x, a = symbols('x a')
    >>> monomial(3*a*sqrt(x), x)
    (3*a, 1/2)
    >>> monomial(x + 1, x) is None
    True
    """
    if isinstance(e, Add):
        # -a*x - I*b*x is x*(-a - I*b)
        e = as_expr(factor_terms(e))
        if isinstance(e, Add):
            return None
    coefficient, rest = e.as_independent(x, as_Add=False)
    coefficient_, rest_ = as_expr(coefficient), as_expr(rest)
    if rest_ == x:
        return (coefficient_, S.One)
    if isinstance(rest_, Pow) and rest_.base == x:
        exponent = as_expr(rest_.exp)
        if isinstance(exponent, Rational) and exponent != 0:
            return (coefficient_, exponent)
    return None


def _positive_monomial(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``monomial`` restricted to a coefficient which may be positive (not
    a negative number)."""
    found = monomial(e, x)
    if found is None:
        return None
    beta, g = found
    if beta.is_negative or beta == 0:
        return None
    return (beta, g)


def _negated_monomial(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(beta, gamma)`` when ``e == -beta * x**gamma``."""
    found = monomial(-e, x)
    if found is None:
        return None
    beta, g = found
    if beta.is_negative or beta == 0:
        return None
    return (beta, g)


def _match_function(f: Expr, x: Symbol, cutoff: Optional[str] = None) -> Optional[Match]:
    """A function application recognised as a kernel."""
    if not isinstance(f, Function):
        return None
    args = [as_expr(a) for a in f.args]
    if isinstance(f, exp) and len(args) == 1:
        found = _negated_monomial(args[0], x)
        if found is not None:
            return Match(KERNELS['exp'], found[0], found[1])
        return None
    one_argument: dict[type[Function], str] = {
        sin: 'sin', cos: 'cos', atan: 'atan', erf: 'erf', erfc: 'erfc', Si: 'Si', Ci: 'Ci'}
    for cls, name in one_argument.items():
        if isinstance(f, cls) and len(args) == 1:
            found = _positive_monomial(args[0], x)
            if found is not None:
                return Match(KERNELS[name], found[0], found[1])
            return None
    if isinstance(f, log) and len(args) == 1:
        # log(1 + beta x^gamma), possibly written log(c + d x^gamma) = log c + log(1 + (d/c) x^gamma)
        inner = args[0]
        if isinstance(inner, Add):
            constant, rest = inner.as_independent(x, as_Add=True)
            constant_, rest_ = as_expr(constant), as_expr(rest)
            if constant_ == 1:
                found = _positive_monomial(rest_, x)
                if found is not None:
                    return Match(KERNELS['log(1 + x)'], found[0], found[1])
                found = _negated_monomial(rest_, x)
                if found is not None and cutoff == 'lower' and found[0] == 1 and found[1].is_positive:
                    # log(1 - x^gamma) on (0, 1)
                    return Match(_log_one_minus_kernel(), S.One, found[1], S.One, True)
        return None
    if isinstance(f, Ei) and len(args) == 1:
        found = _negated_monomial(args[0], x)
        if found is not None:
            return Match(KERNELS['E1'], found[0], found[1], S.NegativeOne)
        return None
    if isinstance(f, expint) and len(args) == 2:
        found = _positive_monomial(args[1], x)
        if found is not None and not args[0].has(x):
            return Match(_expint_kernel(args[0]), found[0], found[1])
        return None
    two_arguments: dict[type[Function], str] = {besselj: 'J', bessely: 'Y', besselk: 'K'}
    for cls, name in two_arguments.items():
        if isinstance(f, cls) and len(args) == 2 and not args[0].has(x):
            found = _positive_monomial(args[1], x)
            if found is None:
                return None
            nu = args[0]
            if name == 'J':
                return Match(_besselj_kernel(nu), found[0], found[1])
            if name == 'Y':
                return Match(_bessely_kernel(nu), found[0], found[1])
            return Match(_besselk_kernel(nu), found[0], found[1])
    if isinstance(f, Heaviside) and len(args) >= 1:
        inner = args[0]
        # theta(c - beta x^gamma) = theta(1 - (beta/c) x^gamma) for c > 0
        if isinstance(inner, Add):
            constant, rest = inner.as_independent(x, as_Add=True)
            constant_, rest_ = as_expr(constant), as_expr(rest)
            if constant_.is_positive:
                found = _negated_monomial(rest_, x)
                if found is not None:
                    return Match(KERNELS['theta(1 - x)'], found[0] / constant_, found[1])
                found = _positive_monomial(rest_, x)
                if found is not None and constant_.is_negative:
                    return Match(KERNELS['theta(x - 1)'], found[0] / (-constant_), found[1])
            if constant_.is_negative:
                found = _positive_monomial(rest_, x)
                if found is not None:
                    return Match(KERNELS['theta(x - 1)'], found[0] / (-constant_), found[1])
        return None
    return None


def _match_power(f: Pow, x: Symbol, cutoff: Optional[str]) -> Optional[Match]:
    """``(c + d x^gamma)**e`` as ``c**e (1 + (d/c) x^gamma)**e``; on
    ``(0, 1)`` (``cutoff='lower'``) also ``(1 - x^gamma)**e`` and
    ``(-log(x))**e``, and on ``(1, oo)`` (``cutoff='upper'``)
    ``(x^gamma - 1)**e``; the reciprocals ``1/(exp(x) - 1)``,
    ``1/(exp(x) + 1)``, ``1/sinh(x)`` and ``1/cosh(x)``."""
    base, exponent = as_expr(f.base), as_expr(f.exp)
    if exponent.has(x):
        return None
    if isinstance(base, exp) and len(base.args) == 1:
        found = _negated_monomial(as_expr(base.args[0]) * exponent, x)
        if found is not None:
            return Match(KERNELS['exp'], found[0], found[1])
        return None
    if exponent == -1:
        if isinstance(base, Add) and len(base.args) == 2:
            terms = [as_expr(t) for t in base.args]
            for constant_term, other in ((terms[0], terms[1]), (terms[1], terms[0])):
                if not constant_term.has(x) and isinstance(other, exp):
                    found = _positive_monomial(as_expr(other.args[0]), x)
                    if found is not None:
                        if constant_term == -1:
                            return Match(KERNELS['1/(exp(x) - 1)'], found[0], found[1])
                        if constant_term == 1:
                            return Match(KERNELS['1/(exp(x) + 1)'], found[0], found[1])
        if isinstance(base, sinh):
            found = _positive_monomial(as_expr(base.args[0]), x)
            if found is not None:
                return Match(KERNELS['1/sinh(x)'], found[0], found[1])
        if isinstance(base, cosh):
            found = _positive_monomial(as_expr(base.args[0]), x)
            if found is not None:
                return Match(KERNELS['1/cosh(x)'], found[0], found[1])
    if cutoff == 'lower' and base == -log(x):
        return Match(_log_power_lower_kernel(exponent), S.One, S.One, S.One, True)
    if isinstance(base, Add):
        constant, rest = base.as_independent(x, as_Add=True)
        constant_, rest_ = as_expr(constant), as_expr(rest)
        if constant_ == 0:
            return None
        found = monomial(rest_, x)
        if found is None:
            return None
        beta, g = as_expr(found[0] / constant_), found[1]
        if beta == 0:
            return None
        if beta.is_negative:
            if beta != -1 or not (g.is_positive):
                return None
            if cutoff == 'lower' and constant_.is_positive:
                # c^e (1 - x^gamma)^e on (0, 1)
                return Match(_beta_kernel(exponent + 1), S.One, g, as_expr(constant_**exponent), True)
            if cutoff == 'upper' and constant_.is_negative:
                # (-c)^e (x^gamma - 1)^e on (1, oo)
                return Match(_beta_upper_kernel(exponent + 1), S.One, g, as_expr((-constant_)**exponent), True)
            return None
        return Match(_power_kernel(-exponent), beta, g, as_expr(constant_**exponent))
    return None


def _match_add(f: Add, x: Symbol) -> Optional[Match]:
    """The kernels which are differences: ``exp(-b x^g) - 1``,
    ``cos(b x^g) - 1``, ``sin(b x^g) - b x^g``, ``atan(b x^g) - pi/2`` and
    ``Si(b x^g) - pi/2``, each also with the opposite sign."""
    if len(f.args) != 2:
        return None
    for sign_ in (S.One, S.NegativeOne):
        terms = [as_expr(sign_ * t) for t in f.args]
        for constant, function in ((terms[0], terms[1]), (terms[1], terms[0])):
            if function.has(x) and not constant.has(x):
                found = _match_function(function, x)
                if found is None:
                    continue
                shifted: dict[str, tuple[str, Expr]] = {
                    'exp': ('exp - 1', S.NegativeOne), 'cos': ('cos - 1', S.NegativeOne),
                    'atan': ('atan - pi/2', -pi / 2), 'Si': ('Si - pi/2', -pi / 2)}
                name = found.kernel.name
                if name in shifted and found.constant == 1 and constant == shifted[name][1]:
                    return Match(KERNELS[shifted[name][0]], found.beta, found.gamma, sign_)
            elif function.has(x) and constant.has(x):
                # sin(b x^g) - b x^g
                found = _match_function(function, x)
                if found is not None and found.kernel.name == 'sin' and found.constant == 1 \
                        and constant == -found.beta * x**found.gamma:
                    return Match(KERNELS['sin - x'], found.beta, found.gamma, sign_)
    return None


def mellin_kernel(f: Expr, x: Symbol, cutoff: Optional[str] = None) -> Optional[Match]:
    """``f`` recognised as ``constant * k(beta * x**gamma)`` for a kernel
    ``k`` of the table, or ``None``. With ``cutoff='lower'`` the factor is
    read on ``(0, 1)``, so that ``(1 - x)**e`` is a kernel (the Beta
    integrand); with ``cutoff='upper'`` on ``(1, oo)``, where
    ``(x - 1)**e`` is one.

    Examples
    ========

    >>> from sympy import exp, symbols, besselj, sqrt
    >>> from sympy_extras.integrals.mellin import mellin_kernel
    >>> x, a = symbols('x a')
    >>> mellin_kernel(exp(-3*x**2), x)
    Match(exp, beta=3, gamma=2, constant=1)
    >>> mellin_kernel(1/(x + 2)**a, x)
    Match((1 + x)**(-a), beta=1/2, gamma=1, constant=2**(-a))
    >>> mellin_kernel(besselj(0, sqrt(x)), x)
    Match(besselj, beta=1, gamma=1/2, constant=1)
    >>> mellin_kernel(exp(x), x) is None
    True
    """
    if isinstance(f, Pow):
        return _match_power(f, x, cutoff)
    if isinstance(f, Add):
        return _match_add(f, x)
    return _match_function(f, x, cutoff)


def mellin_transform(f: ExprLike, x: Symbol, s: Symbol) -> Optional[MellinTransform]:
    """The Mellin transform of ``f`` looked up in the table, with its strip
    of convergence: ``f`` must be a constant times a power of ``x`` times
    at most two kernels of the table (with the argument ``beta * x**gamma``).
    ``None`` when ``f`` is not of this form.

    Examples
    ========

    >>> from sympy import exp, symbols, sqrt
    >>> from sympy_extras.integrals.mellin import mellin_transform
    >>> x, s, a = symbols('x s a')
    >>> mellin_transform(exp(-x**2)/sqrt(x), x, s)
    MellinTransform(gamma(s/2 - 1/4)/2, (1/2, oo))
    >>> mellin_transform(1/(1 + x)**a, x, s)
    MellinTransform(gamma(s)*gamma(a - s)/gamma(a), (0, a))
    """
    product = decompose_integrand(as_expr(f), x)
    if product is None:
        return None
    quotient = product.quotient()
    if quotient is None:
        return None
    return MellinTransform(quotient.as_expr(s), quotient.strip, quotient.condition)


class Product:
    """An integrand ``constant * x**alpha * log(x)**n * k_1(...) * k_2(...)``."""

    def __init__(self, constant: Expr, alpha: Expr, log_power: int, matches: Sequence[Match]) -> None:
        self.constant = constant
        self.alpha = alpha
        self.log_power = log_power
        self.matches = tuple(matches)

    def __repr__(self) -> str:
        return "Product(%s, x**%s, log**%s, %s)" % (self.constant, self.alpha, self.log_power,
                                                  list(self.matches))

    def quotient(self, assumptions: Assumptions = None) -> Optional[GammaQuotient]:
        """The Mellin transform of the integrand without the logarithm, as
        a gamma quotient: ``None`` with two kernels (the product is a
        G-function, not a quotient) or with an ``extra`` factor in a
        product."""
        if len(self.matches) == 0:
            return None
        if len(self.matches) == 1:
            return self.matches[0].quotient(assumptions).compose(self.alpha, 1).scaled(self.constant)
        return None


def _exp_besseli(factors: list[Expr], x: Symbol) -> list[Expr]:
    """``exp(-a x^g) besseli(nu, b x^g)`` rewritten as
    ``exp(-(a - b) x^g) * [exp(-b x^g) besseli(nu, b x^g)]``, the bracket
    being the kernel of the table (a symbol standing for it)."""
    for i, factor in enumerate(factors):
        if isinstance(factor, besseli) and len(factor.args) == 2 and not as_expr(factor.args[0]).has(x):
            inner = _positive_monomial(as_expr(factor.args[1]), x)
            if inner is None:
                continue
            for j, other in enumerate(factors):
                if isinstance(other, exp) and len(other.args) == 1:
                    outer = _negated_monomial(as_expr(other.args[0]), x)
                    if outer is None or outer[1] != inner[1]:
                        continue
                    difference = as_expr(outer[0] - inner[0])
                    rest = [f for k, f in enumerate(factors) if k not in (i, j)]
                    combined = _CombinedBesseli(as_expr(factor.args[0]), inner[0], inner[1])
                    if difference != 0:
                        rest.append(exp(-difference * x**inner[1]))
                    return rest + [combined]
    return factors


class _CombinedBesseli(Expr):
    """A placeholder for ``exp(-b x^g) besseli(nu, b x^g)`` in a list of
    factors (never part of a returned expression)."""

    def __new__(cls, nu: Expr, beta: Expr, gamma_: Expr) -> _CombinedBesseli:
        obj = Expr.__new__(cls, nu, beta, gamma_)
        return obj

    @property
    def nu(self) -> Expr:
        return as_expr(self.args[0])

    @property
    def beta(self) -> Expr:
        return as_expr(self.args[1])

    @property
    def power(self) -> Expr:
        return as_expr(self.args[2])


def decompose_integrand(f: Expr, x: Symbol, cutoff: Optional[str] = None) -> Optional[Product]:
    """``f`` split into a constant, a power of ``x``, a power of ``log(x)``
    and kernel factors; ``None`` when a factor is not recognised or there
    are more than two kernels. With ``cutoff='lower'`` the integrand lives
    on ``(0, 1)``: the step function ``theta(1 - x)`` is added as a kernel
    unless a factor ``(1 - x**gamma)**e`` already carries the cutoff;
    ``cutoff='upper'`` likewise for ``(1, oo)``."""
    constant: Expr = S.One
    alpha: Expr = S.Zero
    log_power = 0
    matches: list[Match] = []
    factors: list[Expr] = []
    for factor in (f.args if isinstance(f, Mul) else [f]):
        e = as_expr(factor)
        # an integer power of a function is that many factors
        if isinstance(e, Pow) and isinstance(e.exp, Integer) and e.exp > 1 and isinstance(e.base, Function) \
                and e.has(x):
            factors.extend([as_expr(e.base)] * int(e.exp))
        else:
            factors.append(e)
    if any(isinstance(e, besseli) for e in factors):
        factors = _exp_besseli(factors, x)
    for e in factors:
        if isinstance(e, _CombinedBesseli):
            matches.append(Match(_exp_besseli_kernel(e.nu), e.beta, e.power))
            continue
        if not e.has(x):
            constant = constant * e
            continue
        if e == x:
            alpha = alpha + 1
            continue
        if isinstance(e, Pow) and e.base == x and not as_expr(e.exp).has(x):
            alpha = alpha + as_expr(e.exp)
            continue
        if isinstance(e, log) and e.args[0] == x:
            log_power += 1
            continue
        if isinstance(e, Pow) and isinstance(e.base, log) and e.base.args[0] == x \
                and isinstance(e.exp, Integer) and e.exp > 0:
            log_power += int(e.exp)
            continue
        found = mellin_kernel(e, x, cutoff)
        if found is None:
            return None
        matches.append(found)
    if cutoff is not None and not any(m.cutoff for m in matches):
        name = 'theta(1 - x)' if cutoff == 'lower' else 'theta(x - 1)'
        matches.append(Match(KERNELS[name], S.One, S.One, S.One, True))
    if len(matches) > 2:
        return None
    return Product(constant, alpha, log_power, matches)

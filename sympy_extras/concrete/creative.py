r"""Definite summation by creative telescoping in $\Pi\Sigma$-fields.

For a summand $f(n, k)$ the definite sum $S(n) = \sum_{k=a(n)}^{b(n)} f(n, k)$
is evaluated in three steps.

**Creative telescoping.** The sequences $f(n, k), f(n + 1, k), \ldots,
f(n + r, k)$ are written as elements of one $\Pi\Sigma$-field of $k$ over
the constants $\mathbb{Q}(n)$ (the upper limit $n$ is a parameter of the
constant field): a hypergeometric part $T(n, k)$ of the summand is a
$\Pi$-extension, and $T(n + i, k) = \rho_i(n, k)\, T(n, k)$ with $\rho_i$
rational, harmonic numbers and nested sums are $\Sigma$-extensions. Karr's
parameterized telescoping (:meth:`~.PiSigmaField.solve`) then finds
constants $c_0(n), \ldots, c_r(n)$ of $\mathbb{Q}(n)$, not all zero, and a
certificate $g$ in the field with

.. math::

    c_0(n) f(n, k) + \cdots + c_r(n) f(n + r, k) = g(n, k + 1) - g(n, k),

trying the orders $r = 0, 1, 2, \ldots$ in turn. For hypergeometric
summands this is Zeilberger's algorithm; with harmonic numbers or other
indefinite sums in the summand it is Schneider's extension of creative
telescoping to $\Pi\Sigma$-fields ([Schneider2007]_, section 2;
[Schneider2008]_).

**The recurrence.** Summing the relation over $k$ gives a recurrence
$\sum_i c_i(n) S(n + i) = b(n)$. The limits depend on $n$, so the relation
is summed over the range common to all the shifted sums, and the terms of
each $S(n + i)$ outside it are added explicitly. The certificate is only
evaluated where its representation in the field is regular: the linear
factors of its denominators in $k$ are located, and the ends of the
telescoped range are moved inwards past them (the certificate of
$\binom{n}{k}$ has the factor $1/(k - n - 1)$, which vanishes just past the
upper limit), the terms left out being summed explicitly too.

**The closed form.** The recurrence is solved in d'Alembertian terms
(Abramov and Petkovšek): a hypergeometric solution of the homogeneous
recurrence (:func:`~sympy_extras.solvers.recurr.rsolve_hypergeometric`)
reduces the order by one, a first order recurrence is solved by a product
and a sum, and the sums are evaluated with Karr's algorithm
(:func:`~.karr.karr_sum`) or left as unevaluated sums when they have no
closed form in their field (they are then part of the answer, as
$\sum_{j=1}^n 1/(j 2^j)$ in
$\sum_k \binom{n}{k} H_k = 2^n (H_n - \sum_{j=1}^n 1/(j 2^j))$).
The constants are fixed by values of the sum computed directly, from the
first $n$ from which the recurrence holds and determines the next value;
the closed form is then compared with the directly computed sum at a few
more values of $n$ (exactly, or numerically at sample values of the other
parameters), and at the small values of $n$ below the start, where it may
fail and the sum is written as a ``Piecewise``.

The module is written on the core (:mod:`sympy_extras.core`, the native
functions, polynomials, simplifications and sums, the native
$\Pi\Sigma$-fields of :mod:`~sympy_extras.concrete.pisigma` built by the
builder of :mod:`~sympy_extras.concrete.karr`) and takes and returns its
expressions. Two steps still go through SymPy for the call, and only for
summands with parameters other than the limit: the cases of the isolated
values of the parameters (:mod:`sympy_extras._special_values`, through
:mod:`~sympy_extras.concrete.karr`) and the numerical comparison of the
closed form with the sum at sample values of the parameters
(:func:`~sympy_extras._numeric.reliable_value`, through
:mod:`~sympy_extras.concrete.karr_sympy`).

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import binomial, harmonic
>>> from sympy_extras.concrete.creative import creative_telescoping, definite_sum
>>> n, k = symbols('n k')
>>> Z = creative_telescoping(binomial(n, k)**2, k, n)
>>> Z.coefficients
[-4*n - 2, n + 1]
>>> definite_sum(binomial(n, k)**2, (k, 0, n))
binomial(2*n, n)
>>> definite_sum(harmonic(k)/(n + 1 - k), (k, 1, n))
harmonic(n + 1)**2 - harmonic(n + 1, 2)

References
==========

.. [Schneider2007] C. Schneider, Symbolic summation assists combinatorics,
   Sém. Lothar. Combin. 56 (2007), B56b.
.. [Schneider2008] C. Schneider, A refined difference field theory for
   symbolic summation, J. Symbolic Comput. 43 (2008) 611-644.
.. [Zeilberger1991] D. Zeilberger, The method of creative telescoping,
   J. Symbolic Comput. 11 (1991) 195-204.
.. [AbramovPetkovsek1994] S. A. Abramov, M. Petkovšek, D'Alembertian
   solutions of linear differential and difference equations, ISSAC 1994.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Callable, Final, Optional, Sequence

import mpmath

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.function import Function, count_ops, expand_func
from sympy_extras.core.logic import Boolean, true
from sympy_extras.core.mod import mod
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import ComplexInfinity, Infinity, Integer, NaN, NegativeInfinity, Rational, pi
from sympy_extras.core.power import sqrt
from sympy_extras.core.relational import eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import Binomial, Factorial, RisingFactorial, factorial, ff, rf
from sympy_extras.functions.combinatorial.numbers import Harmonic, harmonic
from sympy_extras.functions.elementary.piecewise import piecewise
from sympy_extras.functions.special.gamma_functions import Gamma, gamma
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import Poly, cancel, factor_list, gcd_list, lcm_list
from sympy_extras.polys.rationaltools import together
from sympy_extras.settings import settings
from sympy_extras.simplify.gammasimp import combsimp
from sympy_extras.simplify.powsimp import powsimp
from sympy_extras.simplify.simplify import simplify
from sympy_extras.solvers.linsolve import solve_linear_unique
from sympy_extras.solvers.recurr import rsolve_hypergeometric

from .expr_with_limits import Product, Sum
from .gosper import hypersimp
from .karr import SumPoint, _atoms, _auto_extensions, _Builder, _with_special_values, karr_sum
from .pisigma import PiSigmaElement, PiSigmaTower
from .products import product

__all__ = ['CreativeTelescoper', 'creative_telescoping', 'definite_sum']

#: the seconds given to one simplification or one call of a solver
_STEP_SECONDS = 10.0
#: the values of n beyond the initial values at which the closed form is
#: compared with the sum
_CHECKS = 5

ZERO = Integer(0)
ONE = Integer(1)

#: the errors of the polynomial and simplification routines on an
#: expression outside their scope
_OUT_OF_SCOPE = (BasePolynomialError, ValueError, TypeError, NotImplementedError)


def _within(f: Callable[[], Expr], seconds: float) -> Optional[Expr]:
    """``f()`` within ``seconds`` (a time limit which does not load SymPy),
    ``None`` when the time is up or the routines give up."""
    return attempt(f, seconds, native=True)


def _sorted_symbols(symbols: set[Symbol]) -> list[Symbol]:
    return sorted(symbols, key=lambda s: s.name)


def _int(e: Basic) -> int:
    if not isinstance(e, Integer):
        raise TypeError("%s is not an integer" % (e,))
    return e.p


def _arg(node: Basic, i: int) -> Expr:
    return as_expr(node.args[i])


def _ops(e: Expr) -> int:
    n = count_ops(e)
    return n.p if isinstance(n, Integer) else 0


def _negative(e: Expr) -> bool:
    """Whether ``e`` is a negative rational number."""
    return isinstance(e, Rational) and e.p < 0


class CreativeTelescoper:
    r"""The result of creative telescoping for a summand ``f(n, k)``.

    Attributes
    ==========

    term, k, n
        The input.
    coefficients : list of Expr
        Polynomials $c_0(n), \ldots, c_r(n)$ in ``n`` (and the other
        parameters) without a common factor.
    certificate : Expr
        $g(n, k)$ with $\sum_i c_i(n) f(n + i, k) = g(n, k + 1) - g(n, k)$.
    field : PiSigmaTower
        The $\Pi\Sigma$-field of ``k`` over $\mathbb{Q}(n)$ in which the
        relation holds.
    """

    def __init__(self, term: Expr, k: Symbol, n: Symbol, coefficients: list[Expr], certificate: Expr,
                 field: PiSigmaTower, elements: list[PiSigmaElement], g: PiSigmaElement, factor: Expr) -> None:
        self.term: Final[Expr] = term
        self.k: Final[Symbol] = k
        self.n: Final[Symbol] = n
        self.coefficients: Final[list[Expr]] = coefficients
        self.certificate: Final[Expr] = certificate
        self.field: Final[PiSigmaTower] = field
        self._elements: Final[list[PiSigmaElement]] = elements
        self._g: Final[PiSigmaElement] = g
        self._factor: Final[Expr] = factor

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def recurrence(self, name: str = 'S') -> Expr:
        """The left hand side $\\sum_i c_i(n) S(n + i)$ of the recurrence
        of the sums over ``k``, as an expression in the function ``S``."""
        f = Function(name)
        return add(*[c*f(self.n + i) for i, c in enumerate(self.coefficients)])

    def shifted_certificate(self) -> Expr:
        """$g(n, k + 1)$ written with the sequences of the field at ``k``
        (so that it can be evaluated where $g$ itself is not, just past
        the support of a binomial coefficient)."""
        return self._factor*self.field.to_expr(self.field.sigma(self._g), positive_denominator=True)

    def check(self) -> bool:
        """Whether the telescoping relation holds in the field (exact
        arithmetic in $\\mathbb{Q}(n)(k)(t_1, \\ldots, t_m)$)."""
        F = self.field
        lhs = F.zero
        for c, f in zip(self.coefficients, self._elements):
            lhs += F.from_expr(c)*f
        # the difference is compared with zero: the numerator and the
        # denominator of an element over Q(n) are not normalised, so equal
        # elements may compare different
        return bool(F.sigma(self._g) - self._g - lhs == F.zero)

    def __repr__(self) -> str:
        return "CreativeTelescoper(%s, %s)" % (self.coefficients, self.certificate)


# ----------------------------------------------------------------------
# the summand as an element of a PiSigma-field over Q(n)

def _has_sums(expr: Expr, k: Symbol) -> bool:
    """Whether ``expr`` contains harmonic numbers, sums or products
    depending on ``k``."""
    return any(isinstance(a, (Harmonic, Sum, Product)) or a.has(Harmonic, Sum, Product)
               for a in _atoms(expr, k))


def _split(f: Expr, k: Symbol) -> tuple[Expr, Expr, Expr]:
    """``(u, T, R)`` with ``f = u*T*R``: ``u`` the factors free of ``k``,
    ``T`` the hypergeometric factors and ``R`` the rest (rational in ``k``
    or containing sums)."""
    u: Expr = ONE
    T: Expr = ONE
    R: Expr = ONE
    for factor in Mul.make_args(f):
        if not factor.has(k):
            u = u*factor
        elif _has_sums(factor, k) or not _atoms(factor, k):
            R = R*factor
        else:
            T = T*factor
    return u, T, R


def _rational_ratio(num: Expr, den: Expr, symbols: Sequence[Symbol]) -> Optional[Expr]:
    """``num/den`` as a rational function of the ``symbols``, or ``None``."""
    if num == den:
        return ONE
    ratio = _within(lambda: cancel(combsimp(num/den)), _STEP_SECONDS)
    if ratio is None or ratio.is_rational_function(*symbols) is not True:
        second = _within(lambda: cancel(combsimp((num/den).rewrite(Gamma))), _STEP_SECONDS)
        if second is None or second.is_rational_function(*symbols) is not True:
            return None
        ratio = second
    return ratio


def _normalized(values: list[Expr], n: Symbol) -> tuple[list[Expr], Expr]:
    """The coefficients scaled to polynomials without a common factor,
    the leading one with a positive leading coefficient, and the scale."""
    denominators = [together(v).as_numer_denom()[1] for v in values if v != ZERO]
    scale = lcm_list(denominators) if denominators else ONE
    scaled = [cancel(v*scale) for v in values]
    numerators = [v for v in scaled if v != ZERO]
    if numerators:
        common = gcd_list(numerators)
        if common != ZERO:
            scale = cancel(scale/common)
            scaled = [cancel(v/common) for v in scaled]
    leading = next((v for v in reversed(scaled) if v != ZERO), ONE)
    symbols = _sorted_symbols(leading.free_symbols) or [n]
    try:
        if _negative(Poly(leading, *symbols).LC()):
            scale = -scale
            scaled = [-v for v in scaled]
    except _OUT_OF_SCOPE:
        pass
    return scaled, scale


def creative_telescoping(f: ExprLike, k: Symbol, n: Symbol, order: int = 4,
                         extensions: Sequence[Expr] = ()) -> Optional[CreativeTelescoper]:
    r"""Creative telescoping of ``f(n, k)`` in a $\Pi\Sigma$-field.

    Finds polynomials $c_0(n), \ldots, c_r(n)$, not all zero, and a
    certificate $g(n, k)$ with

    .. math::

        \sum_{i=0}^{r} c_i(n) f(n + i, k) = g(n, k + 1) - g(n, k),

    for the least order $r \le$ ``order``, or returns ``None`` when there
    is no such relation of order at most ``order`` in the field built from
    the summand (and the ``extensions``). Summing over ``k`` gives a
    recurrence for the definite sums of ``f``, see :func:`definite_sum`.

    The summand may contain hypergeometric terms in ``n`` and ``k``
    (binomials, factorials, powers), rational functions, and harmonic
    numbers or sums with upper limit ``k + c`` (the ``n`` of these must
    only appear in the hypergeometric and rational parts); other symbols
    are parameters. ``ValueError`` is raised for a summand outside this
    class.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import binomial, harmonic
    >>> from sympy_extras.concrete.creative import creative_telescoping
    >>> n, k = symbols('n k')
    >>> Z = creative_telescoping(binomial(n, k), k, n)
    >>> Z.coefficients, Z.certificate
    ([-2, 1], k*binomial(n, k)/(k - n - 1))
    >>> Z.recurrence()
    -2*S(n) + S(n + 1)
    >>> Z = creative_telescoping(binomial(n, k)*harmonic(k), k, n)
    >>> Z.coefficients
    [4*n + 4, -4*n - 6, n + 2]
    >>> Z.check()
    True
    """
    f_ = to_expr(f)
    if not isinstance(k, Symbol) or not isinstance(n, Symbol):
        raise TypeError("the index and the parameter must be symbols")
    result = _creative_telescoping(f_, k, n, order, list(extensions))
    if result is None:
        u, T, R = _split(f_, k)
        if _has_sums(R, k):
            extra = [e for e in _auto_extensions(R, k) if e not in extensions]
            if extra:
                result = _creative_telescoping(f_, k, n, order, list(extensions) + extra)
    return result


def _creative_telescoping(f: Expr, k: Symbol, n: Symbol, order: int,
                          extensions: list[Expr]) -> Optional[CreativeTelescoper]:
    u, T, R = _split(f, k)
    others = _sorted_symbols(f.free_symbols - {k, n})
    symbols = f.free_symbols | {n}
    for e in extensions:
        symbols |= e.free_symbols
    params = _sorted_symbols(symbols - {k})
    builder = _Builder(k, params)
    tau: Expr = ONE
    if T != ONE:
        alpha = hypersimp(T, k)
        if alpha is None or alpha.is_rational_function(k) is not True:
            raise ValueError("%s is not hypergeometric in %s" % (T, k))
        try:
            value = builder.field.from_expr(alpha)
        except (ZeroDivisionError, BasePolynomialError) as error:
            raise ValueError("%s is not hypergeometric in %s: %s" % (T, k, error))
        builder.field, tau = builder.field.extend('pi', value, T)
    for e in extensions:
        builder.convert(e)
    converted: list[Expr] = []
    for r in range(order + 1):
        mu = _rational_ratio(u.subs(n, n + r), u, [n, *others])
        rho = _rational_ratio(T.subs(n, n + r), T, [n, k, *others])
        if mu is None or rho is None:
            raise ValueError("the shifts of %s in %s are not rational multiples of it" % (f, n))
        element = builder.convert(mu*rho*tau*R.subs(n, n + r))
        converted.append(builder.field.to_expr(element, substitute=False))
        field = builder.field
        fs = [field.from_expr(e) for e in converted]
        sols = field.solve_embedded(field.one, fs)
        if sols is None:
            return None
        for c, g in sols:
            values = [field.to_expr(ci, substitute=False) for ci in c]
            if all(v == ZERO for v in values):
                continue
            coefficients, scale = _normalized(values, n)
            g = g*field.from_expr(scale)
            certificate = u*field.to_expr(g, positive_denominator=True)
            return CreativeTelescoper(f, k, n, coefficients, certificate, field, fs, g, u)
    return None


# ----------------------------------------------------------------------
# definite sums

def _linear(expr: Expr, n: Symbol) -> Optional[tuple[int, int]]:
    """``(m, s)`` with ``expr == m*n + s``, integers ``m >= 0`` and ``s``."""
    try:
        p = Poly(expr, n)
    except _OUT_OF_SCOPE:
        return None
    if p.degree() > 1:
        return None
    coeffs = p.all_coeffs()
    m, s = (coeffs[0], coeffs[1]) if len(coeffs) == 2 else (ZERO, coeffs[0])
    if not (isinstance(m, Integer) and isinstance(s, Integer)) or m.p < 0:
        return None
    return m.p, s.p


def _finite(value: Expr) -> bool:
    return not value.has(NaN, ComplexInfinity, Infinity, NegativeInfinity)


def _direct(f: Expr, k: Symbol, lo: int, hi: int) -> Optional[Expr]:
    """``Sum(f, (k, lo, hi))`` term by term with Karr's convention for
    ``hi < lo``, or ``None`` when a term is not defined."""
    if hi < lo - 1:
        value = _direct(f, k, hi + 1, lo - 1)
        return None if value is None else -value
    total: Expr = ZERO
    for i in range(lo, hi + 1):
        term = f.subs(k, Integer(i))
        if term.has(Sum, Product):
            term = term.doit()
        if not _finite(term):
            return None
        total = total + term
    return total


def _tidy(expr: Expr) -> Expr:
    """``expr`` with the binomials and harmonic numbers of shifted
    arguments expanded and the hypergeometric terms simplified."""
    value = _within(lambda: combsimp(expand_func(expr)), _STEP_SECONDS)
    return expr if value is None else value


def _unshifted_harmonics(expr: Expr) -> Expr:
    """``harmonic(x + c, r)`` for integers ``c`` written through
    ``harmonic(x, r)`` and ``c`` reciprocal powers."""
    def rule(h: Basic) -> Basic:
        x = _arg(h, 0)
        order = _arg(h, 1) if len(h.args) > 1 else ONE
        c, y = x.as_coeff_Add()
        if not isinstance(c, Integer) or c.p == 0 or y.is_number:
            return h
        if c.p > 0:
            return harmonic(y, order) + add(*[1/(y + i)**order for i in range(1, c.p + 1)])
        return harmonic(y, order) - add(*[1/(y - i)**order for i in range(0, -c.p)])
    return as_expr(expr.replace(lambda e: isinstance(e, Harmonic), rule))


def _falling(expr: Expr) -> Expr:
    """``expr`` with ``RisingFactorial(-x, n)`` written as ``(-1)**n*
    FallingFactorial(x, n)``, and falling factorials over factorials as
    binomials: the product of Vandermonde's recurrence is
    ``(-1)**n*RisingFactorial(-a - b, n)/factorial(n)``, which is
    ``binomial(a + b, n)``."""
    def rule(e: Basic) -> Basic:
        x, m = _arg(e, 0), _arg(e, 1)
        return (-1)**m*ff(-x, m)
    replaced = powsimp(as_expr(expr.replace(lambda e: isinstance(e, RisingFactorial), rule)))
    return replaced.rewrite(Binomial)


def _shifted_harmonics(expr: Expr, n: Symbol, shift: int) -> Expr:
    """``expr`` with ``harmonic(n, r)`` written through ``harmonic(n +
    shift, r)``, expanded."""
    def rule(h: Basic) -> Basic:
        order = _arg(h, 1) if len(h.args) > 1 else ONE
        if h.args[0] != n or not isinstance(order, Integer):
            return h
        # harmonic(n + shift) minus the reciprocals between n and n + shift
        return harmonic(n + shift, order) - (_unshifted_harmonics(harmonic(n + shift, order)) - as_expr(h))
    return cancel(as_expr(expr.replace(lambda e: isinstance(e, Harmonic), rule)))


def _nicer(solution: Expr, n: Symbol) -> Expr:
    """The closed form simplified, the gamma functions of half-integer
    arguments removed with the duplication formula and factorials or
    binomials preferred when smaller."""
    half_ = Rational(1, 2)
    third = Rational(1, 3)

    def duplicate(e: Expr) -> Expr:
        def rule(g: Basic) -> Basic:
            half = _arg(g, 0) - half_
            if half.is_integer or (half.is_polynomial(n) and Poly(half, n).all_coeffs()[-1].is_integer
                                   and half.is_integer is not False):
                return gamma(2*half)*sqrt(pi)*2**(1 - 2*half)/gamma(half)
            return g
        return as_expr(e.replace(lambda g: isinstance(g, Gamma), rule))

    def triplicate(e: Expr) -> Expr:
        # gamma(z + 1/3) gamma(z + 2/3) = 2 pi 3**(1/2 - 3z) gamma(3z)/gamma(z)
        gammas = [_arg(g, 0) for g in e.atoms(Gamma)]
        for a in gammas:
            for b in gammas:
                if b - a == third and (a - third).is_polynomial(n):
                    z = a - third
                    formula = 2*pi*3**(half_ - 3*z)*gamma(3*z)/(gamma(z)*gamma(b))
                    return e.subs({gamma(a): formula})
        return e

    candidates = [simplify(combsimp(solution))]
    candidates.append(combsimp(duplicate(candidates[0])))
    candidates.append(combsimp(triplicate(candidates[-1])))
    candidates.append(combsimp(candidates[-1].rewrite(Factorial)))
    candidates.append(candidates[-1].rewrite(Binomial))
    candidates.append(simplify(candidates[-1]))
    return min(candidates, key=lambda e: (_ops(e), len(str(e))))


def _simplest(expr: Expr, n: Symbol) -> Expr:
    """The shortest of a few simplifications of the closed form."""
    candidates = [expr]
    starts = [expr]
    for transformation in (_unshifted_harmonics, _falling):
        value = _within(lambda: transformation(expr), _STEP_SECONDS)
        if value is not None and value != expr:
            starts.append(value)
    if expr.has(Harmonic) and not expr.has(Sum):
        for shift in (1, -1):
            value = _within(lambda: _shifted_harmonics(starts[1] if len(starts) > 1 else expr, n, shift),
                            _STEP_SECONDS)
            if value is not None:
                starts.append(value)

    def to_factorials(e: Expr) -> Expr:
        return as_expr(e.replace(lambda g: isinstance(g, Gamma) and (_arg(g, 0) - 1).is_nonnegative is True,
                                 lambda g: factorial(_arg(g, 0) - 1)))

    simplifications: list[Callable[[Expr], Expr]] = [
        lambda e: factor_terms(combsimp(e)),
        lambda e: factor_terms(cancel(e)) if not e.has(Sum) else e,
        lambda e: factor_terms(combsimp(e.expand())),
        to_factorials,
    ]
    if not expr.has(Sum, Harmonic):
        simplifications.append(lambda e: _nicer(e, n))
    for start in starts:
        candidates.append(start)
        for simplification in simplifications:
            value = _within(lambda: simplification(start), _STEP_SECONDS)
            if value is not None:
                candidates.append(value)
    return min(candidates, key=lambda e: (_ops(e), len(str(e))))


def _integer_roots(expr: Expr, n: Symbol) -> list[int]:
    """The integer roots of the factors of ``expr`` which are polynomials
    in ``n`` alone."""
    roots: list[int] = []
    try:
        factors = factor_list(expr)[1]
    except _OUT_OF_SCOPE:
        return roots
    for factor, _ in factors:
        if factor.free_symbols != {n} or factor.is_polynomial(n) is not True:
            continue
        p = Poly(factor, n)
        if p.degree() == 1:
            a1, a0 = p.all_coeffs()
            root = -a0/a1
            if isinstance(root, Integer):
                roots.append(root.p)
    return roots


class _Unsupported(Exception):
    """A pole of the certificate inside the summation range."""


def _fraction(c: Rational) -> Fraction:
    return Fraction(c.p, c.q)


def _poles(Z: CreativeTelescoper) -> tuple[list[tuple[Fraction, Fraction]], list[int]]:
    """The poles ``k = alpha*n + beta`` of the representations in the
    field of the summands, the certificate, its shift and the generators'
    shift quotients, and the integer poles in ``n`` of their
    coefficients."""
    F, k, n = Z.field, Z.k, Z.n
    elements = list(Z._elements) + [Z._g, F.sigma(Z._g)]
    elements += [F.from_expr(e.value) for e in F.extensions]
    gens: set[Symbol] = {e.symbol for e in F.extensions}
    in_k: list[tuple[Fraction, Fraction]] = []
    in_n: list[int] = []
    for element in elements:
        for is_numer, part in ((True, F.numer_expr(element)), (False, F.denom_expr(element))):
            numerator, denominator = together(part).as_numer_denom()
            # the coefficients of the numerator and the denominator in Q(n)
            in_n.extend(_integer_roots(denominator, n))
            if is_numer:
                continue
            try:
                factors = factor_list(numerator)[1]
            except _OUT_OF_SCOPE:
                continue
            for factor, _ in factors:
                symbols = factor.free_symbols
                if symbols & gens or k not in symbols:
                    if not symbols & gens and k not in symbols and symbols == {n}:
                        in_n.extend(_integer_roots(factor, n))
                    continue
                p = Poly(factor, k)
                if p.degree() != 1:
                    continue
                a1, a0 = p.all_coeffs()
                root = cancel(-a0/a1)
                if not root.free_symbols <= {n}:
                    continue
                try:
                    q = Poly(root, n)
                except _OUT_OF_SCOPE:
                    continue
                if q.degree() > 1:
                    continue
                coeffs = q.all_coeffs()
                alpha, beta = (coeffs[0], coeffs[1]) if len(coeffs) == 2 else (ZERO, coeffs[0])
                if not (isinstance(alpha, Rational) and isinstance(beta, Rational)):
                    continue
                in_k.append((_fraction(alpha), _fraction(beta)))
    return in_k, in_n


def _ceil(x: Fraction) -> int:
    return -((-x.numerator)//x.denominator)


def _window(poles: list[tuple[Fraction, Fraction]], lower: tuple[int, int], upper: tuple[int, int],
            r: int) -> tuple[int, int, int]:
    """``(j0, j1, n_min)``: the telescoped range is ``[a(n + r) + j0,
    b(n) - j1]``, free of poles for ``n >= n_min``."""
    m1, s1 = lower
    m2, s2 = upper
    bottom = s1 + m1*r
    j0 = j1 = 0
    n_min = 0
    for alpha, beta in poles:
        if alpha.denominator == 1 and beta.denominator != 1:
            continue
        if alpha == m1 and m1 < m2:
            if beta >= bottom + j0:
                j0 = int(beta) - bottom + 1
        elif alpha == m2 and m1 < m2:
            if beta <= s2 - j1:
                j1 = s2 - int(beta) + 1
        elif m1 == m2 == alpha:
            if bottom <= beta <= s2:
                raise _Unsupported()
        elif m1 < alpha < m2:
            raise _Unsupported()
    for alpha, beta in poles:
        if alpha.denominator == 1 and beta.denominator != 1:
            continue
        if alpha < m1:
            # alpha*n + beta < m1*n + bottom + j0
            n_min = max(n_min, _ceil((beta - bottom - j0 + 1)/(m1 - alpha)))
        elif alpha > m2:
            # alpha*n + beta > m2*n + s2 - j1
            n_min = max(n_min, _ceil((s2 - j1 - beta + 1)/(alpha - m2)))
    # a nonempty telescoped range: bottom + j0 <= b(n) - j1 + 1
    if m2 > m1:
        n_min = max(n_min, _ceil(Fraction(bottom + j0 + j1 - 1 - s2, m2 - m1)))
    elif bottom + j0 > s2 - j1 + 1:
        raise _Unsupported()
    return j0, j1, n_min


def _index(expr: Expr) -> Symbol:
    """A summation index which does not occur in ``expr``."""
    names = {s.name for s in expr.free_symbols}
    for name in ('j', 'i', 'l', 'm', 'p', 'q'):
        if name not in names:
            return Symbol(name, integer=True)
    return Dummy('j', integer=True)


def _prettier_sum(term: Expr, j: Symbol, lo: Expr, hi: Expr) -> Expr:
    """``Sum(term, (j, lo, hi))`` with the index shifted to the simplest
    summand."""
    candidates: list[Sum] = []
    for d in range(-2, 3):
        shifted = cancel(term.subs(j, j - d)) if term.is_rational_function(j) is True else \
            combsimp(term.subs(j, j - d))
        candidates.append(Sum(shifted, (j, lo + d, hi + d)))
    return min(candidates, key=lambda e: (_ops(e.function), len(str(e.function))))


def _harmonic_extensions(term: Expr, j: Symbol) -> list[Expr]:
    """``harmonic(d*j)`` for the leading coefficients ``d > 1`` of the
    linear factors of the denominator of ``term``."""
    result: list[Expr] = []
    denominator = together(term).as_numer_denom()[1]
    try:
        factors = factor_list(denominator)[1]
    except _OUT_OF_SCOPE:
        return result
    for factor, _ in factors:
        if factor.has(j) and factor.is_polynomial(j) is True and Poly(factor, j).degree() == 1:
            d = Poly(factor, j).LC()
            if isinstance(d, Integer) and abs(d.p) > 1:
                result.append(harmonic(abs(d.p)*j))
    return result


def _sum(term: Expr, j: Symbol, lo: Expr, hi: Expr) -> Expr:
    r"""$\sum_{j=lo}^{hi}$ ``term``: a closed form by Karr's algorithm,
    else the sum of the closed forms of the terms, with the terms which
    have none left as unevaluated sums."""
    term = _tidy(term)
    if term == ZERO:
        return ZERO

    def karr(t: Expr) -> Optional[Expr]:
        def compute() -> Expr:
            value = karr_sum(t, (j, lo, hi), special_values=False)
            if value is None:
                extra = _harmonic_extensions(t, j)
                if extra:
                    value = karr_sum(t, (j, lo, hi), extra, special_values=False)
            if value is None:
                raise NotImplementedError("no closed form")
            # g(n) - g(0) for the sum up to n - 1 has harmonic(n - 1),
            # undefined at n = 0
            return cancel(_unshifted_harmonics(value))
        return _within(compute, _STEP_SECONDS)

    whole = karr(term)
    if whole is not None:
        return whole
    total: Expr = ZERO
    rest: Expr = ZERO
    expanded = _within(lambda: term.expand(), _STEP_SECONDS)
    for t in Add.make_args(expanded if expanded is not None else term):
        t = _tidy(powsimp(t))
        value = karr(t)
        if value is None:
            rest = rest + t
        else:
            total = total + value
    if rest != ZERO:
        total = total + _prettier_sum(_tidy(rest), j, lo, hi)
    return total


def _product(ratio: Expr, n: Symbol, n0: int) -> Optional[Expr]:
    r"""$\prod_{j=n_0}^{n-1}$ ``ratio(j)`` for a rational ``ratio``, or
    ``None`` when a factor vanishes or has a pole for some $j \ge n_0$.

    The ratio is split into linear factors $j + a$, whose products are
    $\Gamma(n + a)/\Gamma(n_0 + a)$; the factors whose $a$ differ by
    integers are collected, so that the gamma functions of a class cancel
    when their exponents add up to zero (the product of $j + 1/4$ is
    $\Gamma(n + 5/4)/\Gamma(n + 1/4)$, which ``gammasimp`` does not reduce
    to $n + 1/4$).
    """
    j = Dummy('j', integer=True)
    numerator, denominator = together(ratio.subs(n, j)).as_numer_denom()
    constant: Expr = ONE
    exponents: dict[Expr, int] = {}
    for part, sign in ((numerator, 1), (denominator, -1)):
        try:
            coefficient, factors = factor_list(part, j)
        except _OUT_OF_SCOPE:
            return None
        constant = constant*coefficient**sign
        for factor, multiplicity in factors:
            p = Poly(factor, j)
            if p.degree() != 1:
                value = _within(lambda: product(ratio.subs(n, j), (j, Integer(n0), n - 1)), _STEP_SECONDS)
                if value is None or value.has(Product):
                    return None
                return _tidy(value)
            lead, tail = p.all_coeffs()
            constant = constant*lead**(sign*_int(multiplicity))
            a = cancel(tail/lead)
            if isinstance(a, Integer) and n0 + a.p <= 0:
                return None
            exponents[a] = exponents.get(a, 0) + sign*_int(multiplicity)
    result: Expr = constant**(n - n0)
    classes: list[list[Expr]] = []
    for a in exponents:
        for members in classes:
            if isinstance(cancel(a - members[0]), Integer):
                members.append(a)
                break
        else:
            classes.append([a])
    for members in classes:
        first = members[0]
        base = min(members, key=lambda a: _int(cancel(a - first)))
        total = 0
        for a in members:
            e = exponents[a]
            total += e
            shift = _int(cancel(a - base))
            # Gamma(x + a)/Gamma(x + base) = (x + base)*...*(x + a - 1)
            rising = mul(*[(n + base + i)/(n0 + base + i) for i in range(shift)])
            result = result*rising**e
        if total and base.is_number:
            result = result*(gamma(n + base)/gamma(n0 + base))**total
        elif total:
            result = result*rf(n0 + base, n - n0)**total
    return cancel(result) if not result.has(Gamma, RisingFactorial) else result


def _hypergeometric_solution(c: list[Expr], n: Symbol) -> Optional[Expr]:
    """A hypergeometric solution of the recurrence with coefficients ``c``
    (the first of the basis of
    :func:`~sympy_extras.solvers.recurr.rsolve_hypergeometric`), in its
    nicest form."""
    def first() -> Expr:
        for term in rsolve_hypergeometric(c, n):
            if term != ZERO:
                return term
        raise NotImplementedError("no hypergeometric solution")
    value = _within(first, 4*_STEP_SECONDS)
    if value is None:
        return None
    hypergeometric = value
    nicer = _within(lambda: _nicer(hypergeometric, n), _STEP_SECONDS)
    return value if nicer is None else nicer


def _general_solution(c: list[Expr], rhs: Expr, n: Symbol, n0: int) -> Optional[tuple[Expr, list[Symbol]]]:
    r"""The general solution of $\sum_i c_i(n) S(n + i) = rhs(n)$ for
    $n \ge n_0$ in d'Alembertian terms, with its constants, or ``None``."""
    r = len(c) - 1
    if r == 0:
        return cancel(rhs/c[0]), []
    C = Dummy('C')
    if r == 1:
        ratio = cancel(-c[0]/c[1])
        P = _product(ratio, n, n0)
        if P is None:
            return None
        beta = cancel(rhs/c[1])
        if beta == ZERO:
            return C*P, [C]
        j = _index(rhs + P)
        term = _tidy(beta.subs(n, j)/P.subs(n, j + 1))
        return P*(C + _sum(term, j, Integer(n0), n - 1)), [C]
    h = _hypergeometric_solution(c, n)
    if h is None:
        return None
    hypergeometric = h

    def quotient() -> Expr:
        value = hypersimp(hypergeometric, n)
        if value is None:
            raise NotImplementedError("not hypergeometric")
        return value
    q = _within(quotient, _STEP_SECONDS)
    rho: Optional[Expr] = q if q is not None and q.is_rational_function(n) is True else \
        _rational_ratio(h.subs(n, n + 1), h, [n])
    if rho is None:
        return None
    # S = h*y, z = y(n + 1) - y(n): sum_t d_t z(n + t) = rhs/h with
    # d_t = sum_{i > t} c_i Q_i, Q_i = h(n + i)/h(n)
    Q: list[Expr] = [ONE]
    for i in range(1, r + 1):
        Q.append(cancel(Q[-1]*rho.subs(n, n + i - 1)))
    d = [cancel(add(*[c[i]*Q[i] for i in range(t + 1, r + 1)])) for t in range(r)]
    reduced = _general_solution(d, _tidy(rhs/h), n, n0)
    if reduced is None:
        return None
    z, constants = reduced
    j = _index(z + rhs)
    return h*(C + _sum(z.subs(n, j), j, Integer(n0), n - 1)), constants + [C]


def _numeric(e: Expr, digits: int, values: SumPoint) -> Optional[complex]:
    """The value of ``e`` at the ``values`` of its symbols as a complex
    number, ``None`` when it is not a reliable finite number. With
    parameters the value comes from
    :func:`~sympy_extras._numeric.reliable_value` (through SymPy, which
    tells the hidden zeros); without, it is evaluated natively."""
    if values:
        from .karr_sympy import reliable_value_of_core
        return reliable_value_of_core(e, digits, values)
    if e.free_symbols:
        return None
    z = numeric_value(e, int(3.33*digits) + 20)
    if z is None or not mpmath.isfinite(z):
        return None
    return complex(z)


def _agree(direct: Expr, value: Expr) -> Optional[bool]:
    """Whether ``direct == value``: exactly when it is proved, else
    numerically at sample values of the symbols; ``None`` when neither
    is decided."""
    difference = direct - value
    exact = _within(lambda: cancel(combsimp(expand_func(difference))), _STEP_SECONDS)
    if exact == ZERO:
        return True
    digits = settings.precision
    symbols = _sorted_symbols(difference.free_symbols)
    samples: list[SumPoint] = [{}] if not symbols else [
        {s: Rational(p, q) for s, p, q in zip(symbols, primes, [7, 11, 13, 17, 19, 23, 29, 31])}
        for primes in ([31, 37, 41, 43, 47, 53, 59, 61], [67, 71, 73, 79, 83, 89, 97, 101])]
    for values in samples:
        error = _numeric(difference, digits, values)
        size = _numeric(direct, digits, values)
        if error is None or size is None:
            return None
        if abs(error) > 10.0**(-(digits - 10))*max(1.0, abs(size)):
            return False
    return True


def definite_sum(f: ExprLike, limits: Sequence[ExprLike], n: Optional[Symbol] = None,
                 order: int = 4, extensions: Sequence[Expr] = (), special_values: bool = True) -> Optional[Expr]:
    r"""The definite sum $\sum_{k=a}^{b} f(n, k)$ by creative telescoping
    in a $\Pi\Sigma$-field, or ``None``.

    ``limits`` is ``(k, a, b)`` with ``a`` and ``b`` of the form
    ``m*n + s`` (``m`` a nonnegative integer, ``s`` an integer), for
    integers ``n >= 0``; ``n`` is the symbol of the limits unless given.
    The recurrence of :func:`creative_telescoping` (orders up to
    ``order``) is completed with the boundary terms and solved in terms
    of products and (possibly unevaluated) sums; the constants are fixed
    by the sum computed directly, and the result is checked against the
    directly computed sum at further values of ``n``. Where it does not
    hold for a few small ``n`` the result is a ``Piecewise``. ``None``
    means that no closed form was found and certified this way.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import binomial, harmonic
    >>> from sympy_extras.concrete.creative import definite_sum
    >>> n, k, a, b = symbols('n k a b')
    >>> definite_sum(binomial(a, k)*binomial(b, n - k), (k, 0, n))
    binomial(a + b, n)
    >>> definite_sum(k*binomial(n, k), (k, 0, n))
    2**n*n/2
    >>> definite_sum(binomial(n, k)*harmonic(k), (k, 0, n))
    2**n*(harmonic(n) - Sum(1/(2**j*j), (j, 1, n)))
    >>> definite_sum(binomial(n, k)**2*harmonic(k), (k, 0, n))
    (2*harmonic(n) - harmonic(2*n))*binomial(2*n, n)

    The isolated values of the other parameters at which the closed form
    is undefined get cases of their own, the sum computed again there
    (unless ``special_values`` is false):

    >>> y = symbols('y')
    >>> definite_sum(y**k, (k, 0, n))
    Piecewise((n + 1, Eq(y, 1)), ((y*y**n - 1)/(y - 1), True))
    """
    f_ = to_expr(f)
    if len(limits) != 3:
        raise ValueError("the limits are (k, a, b), not %s" % (limits,))
    k_, a_, b_ = limits
    if not isinstance(k_, Symbol):
        raise TypeError("%s is not a symbol" % (k_,))
    k = k_
    a, b = to_expr(a_), to_expr(b_)
    if n is None:
        candidates = _sorted_symbols((a.free_symbols | b.free_symbols) - {k})
        if len(candidates) != 1:
            return None
        n = candidates[0]
    # work with an integer n and k, for which binomial(n, n) and the like
    # evaluate
    N = Dummy(n.name, integer=True, nonnegative=True)
    K = Dummy(k.name, integer=True)
    g = f_.subs({n: N, k: K})
    lower, upper = _linear(a.subs({n: N}), N), _linear(b.subs({n: N}), N)
    if lower is None or upper is None or lower[0] > upper[0] or (lower[0] == upper[0] == 0):
        return None
    u, T, R = _split(g, K)
    h = T*R
    result = _definite_sum(h, K, N, lower, upper, order, extensions)
    if result is None and not isinstance(h, Add):
        expanded = _within(lambda: h.expand(), _STEP_SECONDS)
        if isinstance(expanded, Add):
            h = expanded
    if result is None and isinstance(h, Add):
        total: Expr = ZERO
        for term in Add.make_args(h):
            value = _definite_sum(term, K, N, lower, upper, order, extensions)
            if value is None:
                return None
            total = total + value
        result = total
    if result is None:
        return None
    restored: dict[Basic, Basic] = {N: n, K: k}
    closed = (u*result).subs(restored)
    if not special_values:
        return closed
    limit_symbol = n

    def at_point(point: SumPoint) -> Optional[Expr]:
        return definite_sum(f_.subs(point), (k, a, b), limit_symbol, order,
                            [e.subs(point) for e in extensions])

    return _with_special_values(closed, f_, [k, n], [a, b], at_point)


def _definite_sum(f: Expr, k: Symbol, n: Symbol, lower: tuple[int, int], upper: tuple[int, int],
                  order: int, extensions: Sequence[Expr]) -> Optional[Expr]:
    try:
        Z = creative_telescoping(f, k, n, order, extensions)
    except ValueError:
        return None
    if Z is None:
        return None
    r = Z.order
    c = Z.coefficients
    (m1, s1), (m2, s2) = lower, upper
    try:
        poles, poles_in_n = _poles(Z)
        j0, j1, n_min = _window(poles, lower, upper, r)
    except _Unsupported:
        return None

    def a(v: Expr) -> Expr:
        return m1*v + s1

    def b(v: Expr) -> Expr:
        return m2*v + s2

    bottom = a(n + r) + j0
    top = b(n) - j1
    # sum_i c_i S(n + i) = g(n, top + 1) - g(n, bottom) + the terms of
    # each S(n + i) outside [bottom, top]
    rhs = Z.shifted_certificate().subs(k, top) - Z.certificate.subs(k, bottom)
    for i, ci in enumerate(c):
        if ci == ZERO:
            continue
        fi = f.subs(n, n + i)
        below = m1*(r - i) + j0
        above = m2*i + j1
        extra = add(*[fi.subs(k, a(n + i) + t) for t in range(below)],
                    *[fi.subs(k, top + 1 + t) for t in range(above)])
        rhs = rhs + ci*extra
    rhs = _tidy(rhs)
    if not _finite(rhs):
        return None
    starts = [n_min, 0] + [v + 1 for v in poles_in_n]
    starts += [v + 1 for v in _integer_roots(c[-1], n)]
    starts += [v + 1 for v in _integer_roots(together(rhs).as_numer_denom()[1], n)]
    if r == 1:
        starts += [v + 1 for v in _integer_roots(c[0], n)]
    n0 = max(starts)

    def direct(v: int) -> Optional[Expr]:
        return _direct(f.subs(n, Integer(v)), k, _int(a(Integer(v))), _int(b(Integer(v))))

    # a recurrence in steps of d > 1 (sum (-1)**k binomial(n, k)**2 gives
    # S(n + 2) in terms of S(n)) is one recurrence for each residue class
    # of n modulo d
    d = 0
    for i, ci in enumerate(c):
        if ci != ZERO:
            d = gcd(d, i)
    if d <= 1:
        solved = _solve_and_check(c, rhs, n, n0, direct)
        if solved is None:
            return None
        solution, exceptions = solved
        pieces = [(value, eq(n, v)) for v, value in exceptions]
        return piecewise(*pieces, (solution, true)) if pieces else solution
    m = Dummy('m', integer=True, nonnegative=True)
    pieces = []
    branches: list[tuple[Expr, Boolean]] = []
    for residue in range(d):
        class_c = [c[d*i].subs(n, d*m + residue) for i in range(r//d + 1)]
        class_rhs = rhs.subs(n, d*m + residue)
        m0 = max(0, -((residue - n0)//d))

        def class_direct(v: int, residue: int = residue) -> Optional[Expr]:
            return direct(d*v + residue)
        solved = _solve_and_check(class_c, class_rhs, m, m0, class_direct)
        if solved is None:
            return None
        solution, exceptions = solved
        pieces += [(value, eq(n, d*v + residue)) for v, value in exceptions]
        branch = _simplest(solution.subs(m, (n - residue)/d), n)
        condition: Boolean = true if residue == d - 1 else eq(mod(n, d), residue)
        branches.append((branch, condition))
    return piecewise(*pieces, *branches)


def _reversed_at(expr: Expr, n: Symbol, v: int) -> bool:
    """Whether an unevaluated ``Sum`` of ``expr`` has, at ``n = v``, an
    upper limit below its lower limit minus one: a sum is read by Karr's
    convention (``Sum(t, (j, 2, 0)) = -t(1)``), where the empty-sum
    reading gives 0, so the closed form is not given there."""
    for node in expr.atoms(Sum):
        if not isinstance(node, Sum):
            continue
        for limit in node.limits:
            length = (limit[2] - limit[1] + 1).subs(n, Integer(v))
            if length.is_number and length.is_negative is True:
                return True
    return False


def _sums_ordered_from(expr: Expr, n: Symbol, start: int) -> Optional[Expr]:
    r"""``expr`` with the lower limit ``a`` of every ``Sum(t, (j, a,
    b(n)))`` (``a`` a constant, ``b`` increasing in ``n``) lowered to
    ``b(start) + 1`` when it is above, by $\sum_{j=a}^{b} t(j) =
    \sum_{j=b(start)+1}^{b} t(j) - \sum_{j=b(start)+1}^{a-1} t(j)$ (an
    identity under Karr's convention), so that the sums are ordered for
    every ``n >= start``; ``None`` when a term taken out is not
    defined."""
    replacements: dict[Basic, Basic] = {}
    for node in expr.atoms(Sum):
        if not isinstance(node, Sum) or len(node.limits) != 1:
            continue
        j, lo, hi = node.limits[0]
        if not isinstance(lo, Integer) or hi.is_polynomial(n) is not True:
            continue
        p = Poly(hi, n)
        if p.degree() > 1 or _negative(p.LC()):
            continue
        bottom = hi.subs(n, Integer(start)) + 1
        if not isinstance(bottom, Integer) or bottom.p >= lo.p:
            continue
        term = node.function
        taken: list[Expr] = []
        for t in range(bottom.p, lo.p):
            value = term.subs(j, Integer(t))
            if not _finite(value):
                return None
            taken.append(value)
        replacements[node] = Sum(term, (j, bottom, hi)) - add(*taken)
    return expr.subs(replacements) if replacements else expr


def _solve_and_check(c: list[Expr], rhs: Expr, n: Symbol, n0: int, direct: Callable[[int], Optional[Expr]]
                     ) -> Optional[tuple[Expr, list[tuple[int, Expr]]]]:
    """The solution of the recurrence with the initial values of the sum
    at ``n0``, ``n0 + 1``, ..., checked against the sum at further values
    and below ``n0``: ``(solution, [(v, sum at v), ...])`` with the small
    values ``v`` at which the solution does not hold, or ``None``."""
    r = len(c) - 1
    general = None
    # the products of the solution may vanish at the first values: the
    # start is moved on
    for start in range(n0, n0 + 4):
        general = _general_solution(c, rhs, n, start)
        if general is not None:
            n0 = start
            break
    if general is None:
        return None
    solution, constants = general
    if constants:
        # the equations at the first values from n0 on at which the
        # general solution is defined
        equations: list[Expr] = []
        for v in range(n0, n0 + r + _CHECKS):
            value = direct(v)
            at = _within(lambda: solution.subs(n, Integer(v)).doit(), _STEP_SECONDS)
            if value is None or at is None or not _finite(at):
                continue
            equations.append(at - value)
            if len(equations) == len(constants):
                break
        if len(equations) < len(constants):
            return None
        unknowns = constants

        def solve() -> list[Expr]:
            values = solve_linear_unique(equations, unknowns)
            if values is None:
                raise NotImplementedError("no unique solution")
            return values
        solved = attempt(solve, _STEP_SECONDS, native=True)
        if solved is None or any(v.has(*constants) for v in solved):
            return None
        solution = solution.subs(dict(zip(constants, solved)))
    solution = _simplest(solution, n)
    last = n0 + r + _CHECKS
    if solution.has(Sum):
        # the unevaluated sums ordered from the first n possible on (the
        # bug: sum k binomial(n, k) H_k had Sum(..., (j, 2, n - 1)), from
        # 2 to 0 at n = 1, a value by Karr's convention only); the n
        # where one is still reversed are cases of the Piecewise
        for start in range(0, last):
            lowered = _sums_ordered_from(solution, n, start)
            if lowered is not None:
                if lowered != solution:
                    solution = _simplest(lowered, n)
                break
        if _reversed_at(solution, n, last) or _reversed_at(solution, n, last + 100):
            return None
    # the check: from n0 on the recurrence determines the values, below
    # n0 the closed form may fail
    exceptions: list[tuple[int, Expr]] = []
    for v in range(0, last):
        expected = direct(v)
        if _reversed_at(solution, n, v):
            if expected is None:
                return None
            exceptions.append((v, expected))
            continue
        at = _within(lambda: solution.subs(n, Integer(v)).doit(), _STEP_SECONDS)
        agree = None if expected is None or at is None or not _finite(at) else _agree(expected, at)
        if v >= n0:
            if agree is not True:
                return None
        elif agree is not True and expected is not None:
            exceptions.append((v, expected))
    return solution, exceptions

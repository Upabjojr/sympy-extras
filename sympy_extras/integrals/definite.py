"""Definite integration: the driver.

:func:`definite_integral` computes ``Integral(f, (x, a, b))`` with
assumptions on the parameters. It does not compute antiderivatives: the
value of a definite integral with a singular integrand, an infinite
range or parameters is obtained from the structure of the integrand,
the way Mathematica's ``Integrate`` and the DEFINT package of REDUCE
work, and the conditions on the parameters under which the integral
converges come out of the computation. The steps are

1. **Splitting.** The range is cut at the points where the integrand is
   not analytic: the zeros of the arguments of ``Abs``, ``sign``,
   ``Heaviside`` and ``Max``/``Min``, the conditions of a ``Piecewise``,
   the zeros of the factors of a radicand or of the argument of a
   logarithm, and the singularities of the integrand inside the range
   (poles, branch points), found with :func:`sympy_extras.assumptions.solve`
   and :func:`sympy.calculus.singularities.singularities`. On each piece
   the integrand is replaced by its analytic branch, and a power or a
   logarithm of a product is split over the factors whose sign on the
   piece is known (``(u*v)**r`` is ``u**r * v**r`` for ``u > 0``):
   ``sqrt(x**2 + 2*x + 1)`` is ``x + 1`` or ``-x - 1``, ``(1 - cos(x))**(3/2)``
   is ``2*sqrt(2)*sin(x/2)**3`` on ``(0, 2*pi)``, ``log(sin(x)/x)`` is
   ``log(sin(x)) - log(x)`` on ``(0, pi/2)``. The integral over a piece
   with a singular endpoint is an improper integral, whose convergence is
   part of the conditions.

2. **The Marichev–Adamchik method** (:mod:`.marichev`): the range is
   mapped onto `(0, \\infty)`, `(0, 1)` or `(1, \\infty)` by a linear change
   of variable, and the integral of a power of ``x`` times at most two
   functions of the Mellin table is a Meijer G-function, written as
   hypergeometric functions by Slater's theorem. This is the method for
   integrals with singularities: the strips of the Mellin transforms
   are the convergence conditions, so ``Integral(x**k/(x + 3), (x, 0,
   oo))`` is ``-3**k*pi/sin(pi*k)`` for ``-1 < k < 0`` and nothing else.

3. **Changes of variable** on the way there: ``x = log(u)`` for
   integrands in ``exp(x)`` over the real line; ``x = 1/u`` between
   `(0, 1)` and `(1, \\infty)`; ``x = asin(sqrt(u))`` for trigonometric
   integrands over `(0, \\pi/2)`, which turns powers of ``sin`` and
   ``cos`` into a Beta integral, after cutting a range with endpoints
   at multiples of `\\pi/2` into such pieces.

4. **Residues** (:mod:`.residues`) for rational functions times
   powers, exponentials or trigonometric functions over `(0, \\infty)`
   and over the real line, and for rational functions of ``sin`` and
   ``cos`` over a period.

5. **SymPy's** ``integrate`` as the last resort, under the time limit
   of the settings, and only when its result passes a numerical check
   (``settings.numerical_checks``; an answer the quadrature cannot
   confirm is dropped): ``integrate`` evaluates an antiderivative at
   the endpoints and sometimes misses a singularity in between, a
   condition on the parameters, or a branch cut.

The conditions are decided against the assumptions
(:func:`sympy_extras.assumptions.ask`); what stays undecided is
returned in a ``Piecewise`` with the unevaluated integral as the other
branch, as ``integrate`` does.

Examples
========

>>> from sympy import symbols, exp, sin, log, sqrt, Abs, oo, pi, cos
>>> from sympy_extras.integrals import definite_integral
>>> x, k = symbols('x k')
>>> a, s = symbols('a s', positive=True)
>>> definite_integral(x**k/(x + 3), (x, 0, oo))
Piecewise((-3**k*pi/sin(pi*k), (k > -1) & (k < 0)), (Integral(x**k/(x + 3), (x, 0, oo)), True))
>>> definite_integral(x**k/(x + 3), (x, 0, oo), (k > -1) & (k < 0))
-3**k*pi/sin(pi*k)
>>> definite_integral(exp(-s*x)*sin(a*x)/x, (x, 0, oo))
atan(a/s)
>>> definite_integral(Abs(x - 1)/sqrt(x), (x, 0, 2))
2*(4 - sqrt(2))/3
>>> definite_integral(log(x)**2/(1 + x**2), (x, 0, oo))
pi**3/8
>>> definite_integral(sqrt(sin(x)), (x, 0, pi/2))
2*sqrt(pi)*gamma(3/4)/gamma(1/4)
>>> definite_integral(1/(x*sqrt((x + 1)**2)), (x, -oo, -2))
-log(2)
"""
from __future__ import annotations

import random
from math import gcd, lcm
from typing import Callable, Optional, Sequence

import mpmath

from sympy.calculus.singularities import singularities
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, nan, oo, pi, zoo
from sympy.core.power import Pow
from sympy.core.relational import Eq, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs, sign, re, im
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor, frac
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (TrigonometricFunction, asin, sin, cos, tan, cot, sec,
                                                      csc)
from sympy.functions.elementary.hyperbolic import HyperbolicFunction, InverseHyperbolicFunction
from sympy.functions.special.polynomials import OrthogonalPolynomial
from sympy.core.exprtools import factor_terms
from sympy.core.function import count_ops, expand, expand_func, expand_log
from sympy.simplify.fu import TR8, TR9
from sympy.simplify.trigsimp import trigsimp
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.integrals.integrals import Integral, integrate
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import cancel, degree, factor
from sympy.polys.rationaltools import together
from sympy.series.limits import limit
from sympy.logic.boolalg import And, Boolean, true
from sympy.sets.sets import FiniteSet, Interval, Set
from sympy.simplify.powsimp import powdenest
from sympy.utilities.lambdify import lambdify

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, as_set, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.solve import solve
from sympy_extras.settings import settings
from .conditions import ConditionalValue, _items, decide, sample_values
from .marichev import mellin_integrate, tidy
from .mellin import monomial

__all__ = ['definite_integral', 'conditional_integral', 'verify_numerically', 'Limits']

#: ``(x, a, b)``
Limits = tuple[Symbol, ExprLike, ExprLike]

#: the depth of the recursive splitting
_MAX_DEPTH = 6
#: the most terms a sum is integrated term by term
_TERMWISE_LIMIT = 8


def definite_integral(f: ExprLike, limits: Limits, assumptions: Assumptions = None,
                      conds: str = 'piecewise', recognize: bool = False,
                      principal_value: bool = False, finite_part: bool = False,
                      numeric: bool = False, digits: int = 15, regularize: bool = False,
                      summability: Optional[str] = None) -> Expr:
    """``Integral(f, (x, a, b))`` under assumptions on the parameters.

    Parameters
    ==========

    f : Expr
        The integrand.
    limits : (Symbol, Expr, Expr)
        The variable and the bounds, ``-oo`` and ``oo`` allowed.
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters as statements (``a > 0``), see
        :mod:`sympy_extras.assumptions`.
    conds : ``'piecewise'`` or ``'none'``
        What to do with the conditions the assumptions do not settle:
        return a ``Piecewise`` with the unevaluated integral as the
        other branch (the default), or assume that they hold.
    recognize : bool
        When every method fails and the integral has no parameters,
        guess a closed form from a high-precision numerical value with
        PSLQ (:mod:`.recognize`): the result is a conjecture checked to
        forty-five digits, not a proof, hence off by default.
    principal_value : bool
        Cauchy's principal value when the integrand has a singularity
        inside the range at which the integral diverges
        (:func:`~sympy_extras.integrals.antiderivative.principal_value_integral`).
    finite_part : bool
        Hadamard's finite part of a divergent integral, the divergent
        terms of the excision at each singularity dropped
        (:func:`~sympy_extras.integrals.antiderivative.finite_part_integral`).
    numeric : bool
        When every symbolic method fails and the integral has no
        parameters, a ``Float`` with ``digits`` correct digits, proved by
        validated integration in interval arithmetic
        (:func:`~sympy_extras.integrals.validated.validated_integral`);
        ``None`` there leaves the integral unevaluated.
    regularize : bool
        Analytic (Riesz) regularisation of the Mellin method: the value
        of the formula, analytic in the exponents, beyond the strips of
        convergence, where the integral itself diverges
        (``Integral(x**(-3/2)*exp(-x), (x, 0, oo))`` gives ``gamma(-1/2)``).
    summability : str or None
        ``'abel'``, ``'cesaro'`` or ``'gaussian'``: the integral taken in
        the sense of that summability method (:mod:`.summability`),
        ``Integral(sin(x), (x, 0, oo))`` being 1 in each; a convergent
        integral keeps its value.

    Returns
    =======

    The value, a ``Piecewise`` on the conditions, ``oo`` or ``-oo`` when
    the integral diverges to a signed infinity (an infinite one-sided
    limit of an antiderivative at an endpoint or a singularity, as for
    ``1/x`` over ``(0, 1)``; an oscillatory divergence, ``cos(x)`` over
    ``(0, oo)``, and infinities of both signs, ``1/x`` over ``(-1, 1)``,
    are left unevaluated), or the unevaluated ``Integral`` when no
    method applies.

    Examples
    ========

    >>> from sympy import symbols, exp, cos, oo, log, pi, S
    >>> from sympy_extras.assumptions import element
    >>> from sympy_extras.integrals import definite_integral
    >>> x, a, b = symbols('x a b')
    >>> definite_integral(exp(-a*x)*cos(b*x), (x, 0, oo), (a > 0) & element(b, S.Reals))
    a/(a**2 + b**2)
    >>> definite_integral(exp(-a*x)*cos(b*x), (x, 0, oo), element(a, S.Reals) & element(b, S.Reals))
    Piecewise((a/(a**2 + b**2), a > 0), (Integral(exp(-a*x)*cos(b*x), (x, 0, oo)), True))
    >>> definite_integral(x**a*log(x), (x, 0, 1), a > -1)
    -1/(a + 1)**2
    >>> definite_integral(log(1 - x)/x, (x, 0, 1))
    -pi**2/6
    >>> definite_integral(1/x, (x, 0, 1))
    oo
    """
    x, a, b = limits[0], as_expr(limits[1]), as_expr(limits[2])
    f_ = as_expr(f)
    if conds not in ('piecewise', 'none'):
        raise ValueError("conds must be 'piecewise' or 'none', got %r" % (conds,))
    if summability is not None:
        from .summability import summable_integral
        found = summable_integral(f_, x, a, b, summability, assumptions)
    else:
        found = conditional_integral(f_, x, a, b, assumptions, principal_value, finite_part, regularize)
    if (found is None or found.value.has(nan)) and recognize:
        from .recognize import recognize_integral
        guessed = recognize_integral(f_, (x, a, b), assumptions)
        if guessed is not None:
            return guessed
    if (found is None or found.value.has(nan)) and numeric:
        from .validated import validated_integral
        enclosed = validated_integral(f_, x, a, b, digits, assumptions)
        if enclosed is not None:
            return enclosed[0]
    if found is None or found.value.has(nan):
        return as_expr(Integral(f_, (x, a, b)))
    found = ConditionalValue(tidy(found.value, assumptions, found.condition), found.condition)
    if conds == 'none':
        return found.value
    return found.as_piecewise(Integral(f_, (x, a, b)))


def conditional_integral(f: Expr, x: Symbol, a: Expr, b: Expr,
                         assumptions: Assumptions = None,
                         principal_value: bool = False,
                         finite_part: bool = False,
                         regularize: bool = False) -> Optional[ConditionalValue]:
    """The value of ``Integral(f, (x, a, b))`` with the condition on the
    parameters under which it holds, or ``None``.

    The methods of this package run under half of the time limit of the
    settings, so that SymPy's ``integrate`` gets the other half when they
    fail; a value they find under a condition the assumptions do not
    settle is kept unless SymPy finds an unconditional one."""
    f, a, b = _with_equalities(f, a, b, x, assumptions)
    integrator = _Integrator(assumptions, principal_value=principal_value, finite_part=finite_part,
                             regularize=regularize)
    if not _real_bounds(a, b) or _nested_complex_powers(f):
        return integrator._sympy(f, x, a, b)
    budget = None if settings.timeout is None else settings.timeout / 2
    found = attempt(lambda: integrator.integrate(f, x, a, b, 0, False), budget)
    if found is not None and found.condition is true:
        return found
    fallback = integrator._sympy(f, x, a, b)
    if fallback is not None and (found is None or fallback.condition is true):
        return fallback
    return found


def _with_equalities(f: Expr, a: Expr, b: Expr, x: Symbol, assumptions: Assumptions) -> tuple[Expr, Expr, Expr]:
    """The integrand and bounds with the equalities among the assumptions
    (``Eq(n, m)``, a symbol equal to an expression) substituted: a
    formula valid for generic parameters may fail on the equality (the
    bug: ``sin(m x) sin(n x)`` over a period came out as ``-pi m/(2 n)``
    under ``Eq(n, m)``, from the generic antiderivative ``sin((m - n) x)
    /(m - n)``; the value is ``pi``)."""
    items: list[Boolean] = []
    if isinstance(assumptions, (Basic, bool)):
        items.append(as_boolean(assumptions))
    elif assumptions is not None:
        items.extend(as_boolean(s) for s in assumptions)
    replacement: dict[Expr, Expr] = {}
    for item in items:
        for atom in ([item] if isinstance(item, Eq) else list(item.atoms(Eq)) if isinstance(item, And) else []):
            lhs, rhs = as_expr(atom.lhs), as_expr(atom.rhs)
            if isinstance(lhs, Symbol) and lhs != x and not rhs.has(lhs, x):
                replacement[lhs] = rhs
            elif isinstance(rhs, Symbol) and rhs != x and not lhs.has(rhs, x):
                replacement[rhs] = lhs
    if not replacement:
        return f, a, b
    return as_expr(f.xreplace(replacement)), as_expr(a.xreplace(replacement)), as_expr(b.xreplace(replacement))


def _nested_complex_powers(f: Expr) -> bool:
    """Whether ``f`` has a power of a power with a non-real exponent,
    ``(x**(I/2))**I``: SymPy combines the exponents of a positive base
    (``x**(-1/2)``), which the principal branches do not allow when the
    inner argument leaves ``(-pi, pi]``, so the methods which substitute
    a positive variable are not used (the bug: the integral of
    ``(z**(I/2))**I`` over ``(0, 1)`` came out as 2, the value of
    ``z**(-1/2)``; Mathematica's NIntegrate disagrees)."""
    for node in f.atoms(Pow):
        base, exponent = as_expr(node.base), as_expr(node.exp)
        if exponent.is_extended_real is False and isinstance(base, Pow) \
                and as_expr(base.exp).is_extended_real is False:
            return True
    return False


def _real_bounds(a: Expr, b: Expr) -> bool:
    """Whether both bounds are real (or infinite), so that the range is a
    real interval; a complex bound means a segment in the complex plane,
    which only SymPy's antiderivatives handle."""
    for bound in (a, b):
        if bound in (oo, -oo):
            continue
        if bound.is_extended_real is False:
            return False
        if bound.is_extended_real is None and not bound.free_symbols:
            return False
    return True


# ---------------------------------------------------------------------------
# Numerical verification

def _quadrature(f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[complex]:
    """``Integral(f, (x, a, b))`` by mpmath's quadrature, trusted only when
    two rules agree; a complex bound means the straight segment from
    ``a`` to ``b``, parametrised by ``x = a + (b - a) t``."""
    if not _real_bounds(a, b):
        if a in (-oo, oo) or b in (-oo, oo) or a.free_symbols or b.free_symbols:
            return None
        t = Dummy('t', real=True)
        return _quadrature(as_expr(f.subs(x, a + (b - a) * t) * (b - a)), t, S.Zero, S.One)
    g: Callable[[mpmath.mpf], mpmath.mpf] = lambdify(x, f, 'mpmath')
    lo = mpmath.mpf('-inf') if a == -oo else mpmath.mpf(str(as_expr(a).evalf(20)))
    hi = mpmath.mpf('inf') if b == oo else mpmath.mpf(str(as_expr(b).evalf(20)))
    points: list[mpmath.mpf] = [lo]
    if lo == mpmath.mpf('-inf') and hi == mpmath.mpf('inf'):
        points = [lo, mpmath.mpf(-10), mpmath.mpf(-1), mpmath.mpf(0), mpmath.mpf(1), mpmath.mpf(10), hi]
    elif hi == mpmath.mpf('inf'):
        points = [lo, lo + 1, lo + 10, lo + 100, hi]
    elif lo == mpmath.mpf('-inf'):
        points = [lo, hi - 100, hi - 10, hi - 1, hi]
    else:
        points = [lo, hi]
    finer: list[mpmath.mpf] = []
    for p, q in zip(points[:-1], points[1:]):
        finer.append(p)
        if p != mpmath.mpf('-inf') and q != mpmath.mpf('inf'):
            finer.append((p + q) / 2)
    finer.append(points[-1])
    try:
        with mpmath.workdps(20):
            first = mpmath.quad(g, points, method='tanh-sinh', error=True)
            # the same rule on a finer subdivision: an endpoint singularity
            # (a logarithm, a fractional power) which the Gauss-Legendre
            # rule cannot resolve is confirmed this way
            second = mpmath.quad(g, finer, method='tanh-sinh', error=True)
            third = mpmath.quad(g, points, method='gauss-legendre', error=True)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, NameError, AttributeError,
            NotImplementedError, mpmath.libmp.NoConvergence):
        return None
    v1, e1 = complex(first[0]), float(abs(first[1]))
    v2, v3 = complex(second[0]), complex(third[0])
    scale = 1 + max(abs(v1), abs(v2))
    if abs(v1 - v2) > 1e-8 * scale or e1 > 1e-6 * scale:
        return _oscillatory_quadrature(f, x, lo, hi)
    if abs(v1 - v3) > 1e-6 * scale and abs(v2 - v3) > 1e-6 * scale and e1 > 1e-12 * scale:
        return _oscillatory_quadrature(f, x, lo, hi)
    return v1


def _oscillatory_quadrature(f: Expr, x: Symbol, lo: mpmath.mpf, hi: mpmath.mpf) -> Optional[complex]:
    """``Integral(f, (x, lo, hi))`` over a half-line by mpmath's
    ``quadosc`` when ``f`` oscillates (a sine or cosine of ``k*x``): the
    integral summed period by period with Richardson extrapolation,
    trusted when two choices of the period agree (``sin(x)/x`` over
    ``(2, oo)``, where the plain rules do not converge)."""
    if lo == mpmath.mpf('-inf') and hi == mpmath.mpf('inf'):
        left = _oscillatory_quadrature(as_expr(f.subs(x, -x)), x, mpmath.mpf(0), hi)
        right = _oscillatory_quadrature(f, x, mpmath.mpf(0), hi)
        return None if left is None or right is None else left + right
    if lo == mpmath.mpf('-inf'):
        return _oscillatory_quadrature(as_expr(f.subs(x, -x)), x, -hi, mpmath.mpf('inf'))
    if hi != mpmath.mpf('inf'):
        return None
    frequencies: list[Rational] = []
    for node in f.atoms(sin, cos):
        found = monomial(as_expr(node.args[0]), x)
        if found is None or found[1] != 1 or not isinstance(found[0], Rational):
            return None
        frequencies.append(abs(found[0]))
    if not frequencies:
        return None
    numerator, denominator = 0, 1
    for k in frequencies:
        numerator = gcd(numerator, int(k.p))
        denominator = lcm(denominator, int(k.q))
    period = 2 * mpmath.pi * denominator / numerator
    g: Callable[[mpmath.mpf], mpmath.mpf] = lambdify(x, f, 'mpmath')
    try:
        with mpmath.workdps(20):
            first = complex(mpmath.quadosc(g, [lo, hi], period=period))
            second = complex(mpmath.quadosc(g, [lo, hi], period=2 * period))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, NameError, AttributeError,
            NotImplementedError, mpmath.libmp.NoConvergence):
        return None
    if abs(first - second) > 1e-8 * (1 + abs(first)):
        return None
    return first


def _quadrature_converges(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions = None) -> bool:
    """Whether the quadrature of ``Integral(f, (x, a, b))`` gives a
    number of a plausible size at some sample of the parameters: the
    check of a claimed divergence (``1/x`` over ``(0, 1)`` gives none;
    ``exp(x)`` over ``(0, oo)`` gives ``inf`` and ``x**2*log(x + 1)`` a
    number of the order of ``1e74``, which are no convergence)."""
    if f.has(DiracDelta):
        return False
    parameters = sorted_symbols((free_symbols(f) | free_symbols(a) | free_symbols(b)) - {x})
    rng = random.Random(str((f, a, b)))
    for _ in range(2 if parameters else 1):
        values = sample_values(parameters, assumptions, rng)
        if values is None:
            return False
        found = _quadrature(f.xreplace(values), x, as_expr(a.xreplace(values)), as_expr(b.xreplace(values)))
        if found is not None and abs(found) < 1e8:
            return True
    return False


def verify_numerically(value: Expr, f: Expr, x: Symbol, a: Expr, b: Expr,
                       assumptions: Assumptions = None, samples: int = 2) -> Optional[bool]:
    """Whether ``value`` agrees with a numerical quadrature of
    ``Integral(f, (x, a, b))`` at ``samples`` random values of the
    parameters satisfying the assumptions: ``False`` when they disagree
    where the quadrature is trusted, ``True`` when they agree there,
    ``None`` when nothing could be checked."""
    if f.has(DiracDelta):
        return None
    parameters = sorted_symbols((free_symbols(f) | free_symbols(a) | free_symbols(b)) - {x})
    rng = random.Random(str((f, a, b)))
    verdict: Optional[bool] = None
    for _ in range(samples if parameters else 1):
        values = sample_values(parameters, assumptions, rng)
        if values is None:
            return None
        expected = _quadrature(f.xreplace(values), x, as_expr(a.xreplace(values)),
                               as_expr(b.xreplace(values)))
        if expected is None:
            continue
        try:
            ours = complex(as_expr(value.xreplace(values)).evalf(20))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            # mpmath raises ZeroDivisionError on a hypergeometric series at
            # a pole (SymPy's value of Maxima's specint 174 at an integer
            # sample of the order): nothing is checked
            return None
        if abs(ours - expected) > 1e-6 * (1 + abs(expected)):
            return False
        verdict = True
    return verdict


# ---------------------------------------------------------------------------
# The integrator

class _Integrator:
    """The strategies, sharing the assumptions; ``parametric`` allows
    differentiation under the integral sign (off inside that method)."""

    def __init__(self, assumptions: Assumptions, parametric: bool = True,
                 principal_value: bool = False, finite_part: bool = False,
                 regularize: bool = False) -> None:
        self.assumptions = assumptions
        self.parametric = parametric
        self.principal_value = principal_value
        self.finite_part = finite_part
        self.regularize = regularize
        #: the factored arguments of the powers and logarithms, by argument
        self._factored: dict[Expr, Optional[tuple[Expr, list[tuple[Expr, Rational]], bool]]] = {}
        #: the zeros of the factors inside the ranges seen
        self._zero_cache: dict[tuple[Expr, Expr, Expr], Optional[list[Expr]]] = {}

    def ask(self, query: Boolean) -> Optional[bool]:
        return ask(query, self.assumptions)

    def _factors_of(self, base: Expr, x: Symbol) -> Optional[tuple[Expr, list[tuple[Expr, Rational]], bool]]:
        if base not in self._factored:
            self._factored[base] = _factored(base, x)
        return self._factored[base]

    def _nodes(self, f: Expr, x: Symbol, full: bool) -> list[tuple[Expr, Expr, list[tuple[Expr, Rational]]]]:
        """The powers and logarithms of ``f`` to split, with the constant
        and the factors of their arguments; without ``full`` only those
        whose argument has a square (``sqrt((x - 1)**2)``, ``(1 -
        cos(x))**(3/2)``), which are rewritten before the other methods;
        with it every one, for the late splitting."""
        found: list[tuple[Expr, Expr, list[tuple[Expr, Rational]]]] = []
        for node in _signed_nodes(f, x):
            base = as_expr(node.base if isinstance(node, Pow) else node.args[0])
            factored = self._factors_of(base, x)
            if factored is not None and (full or factored[2]):
                found.append((node, factored[0], factored[1]))
        return found

    # -- entry point --------------------------------------------------------

    def integrate(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int,
                  fallback: bool = True, mapped: bool = False) -> Optional[ConditionalValue]:
        """``Integral(f, (x, a, b))``; without ``fallback`` SymPy's
        ``integrate`` is not tried (the caller tries it on its own form),
        and inside a ``mapped`` range the slow methods which work on the
        original form (creative telescoping, differentiation under the
        integral sign) are skipped."""
        if depth > _MAX_DEPTH:
            return None
        if a == b:
            return ConditionalValue(S.Zero)
        if self.ask(as_boolean(a > b)) is True:
            found = self.integrate(f, x, b, a, depth, fallback, mapped)
            return None if found is None else found.scaled(S.NegativeOne)
        if f == 0:
            return ConditionalValue(S.Zero)
        if not f.has(x):
            if a in (-oo, oo) or b in (-oo, oo):
                return None
            return ConditionalValue(f * (b - a))
        f = as_expr(piecewise_fold(f)) if f.has(Piecewise) else f
        if f.has(DiracDelta):
            return self._sympy(f, x, a, b)
        f = _combined_exponentials(f)
        f = _sums_to_products(f, x)
        # constants out, the common factors of sums pulled first
        # (log(t + 1)/(a**2*t**2 + a**2) is log(t + 1)/(t**2 + 1) over a**2)
        constant, rest = as_expr(factor_terms(f)).as_independent(x, as_Add=False)
        constant_, rest_ = as_expr(constant), as_expr(rest)
        if constant_ != 1:
            found = self.integrate(rest_, x, a, b, depth, fallback, mapped)
            return None if found is None else found.scaled(constant_)
        found = self._split_branches(f, x, a, b, depth)
        if found is not None:
            return found
        if not mapped:
            # the identities which change the integrand act on the whole
            # range, before it is cut at singularities (Glasser's map has
            # poles inside)
            found = self._transformations(f, x, a, b, depth)
            if found is not None:
                return found
        singular, found = self._split_singularities(f, x, a, b, depth)
        if found is not None:
            return found
        if singular:
            # a singularity inside the range whose pieces could not be
            # integrated: an antiderivative evaluated at the endpoints
            # would be wrong, so SymPy is not asked; the principal value
            # is computed when asked for
            if self.finite_part and depth == 0:
                from .antiderivative import finite_part_integral
                return self._finish(finite_part_integral(f, x, a, b, self.assumptions))
            if self.principal_value and depth == 0:
                from .antiderivative import principal_value_integral
                return self._finish(principal_value_integral(f, x, a, b, self.assumptions))
            return None
        allowed = (free_symbols(f) | free_symbols(a) | free_symbols(b)) - {x}
        strategies = [self._table, self._dirichlet, self._radicals, self._canonical, self._mean_value, self._elliptic,
                      self._trigonometric, self._mapped, self._inversion, self._residues, self._contours,
                      self._algebraic]
        if a == -oo and b == oo and f.has(HyperbolicFunction):
            # the rectangular contour gives pi**3/4 for x**2/cosh(x) where
            # the Mellin table gives polylogarithms at +-I
            strategies.remove(self._contours)
            strategies.insert(0, self._contours)
        strategies.append(self._split_powers)
        # the antiderivative, bounded by a quarter of the time limit and
        # checked numerically, before the slow methods: 1/(cosh(n*t)**2 + 1)
        # over (0, 1) has one in a second, and differentiation under the
        # integral sign spent the whole budget first
        strategies.append(self._antiderivative)
        if not mapped:
            # the methods which work on the original form only, and the
            # slow ones: not inside a mapped range
            strategies += [self._holonomic, self._laplace, self._parametric, self._series, self._dfinite]
        strategies.append(self._termwise)
        if depth == 0:
            # the heuristics last, on the integral as given only (a quarter
            # of the limit on every piece of every range ate the budget of
            # log(sin(x)/x) over (0, pi/2) before the splitting of the log)
            strategies.append(self._late_antiderivative)
        for strategy in strategies:
            try:
                found = strategy(f, x, a, b, depth)
            except (ZeroDivisionError, AttributeError, AssertionError, OverflowError):
                # SymPy's internals fail on some inputs (a division by zero
                # in mpmath inside evalf, an AttributeError in the cache
                # wrapper of meijerint, an assertion in the LRA solver):
                # the method is skipped
                found = None
            if found is not None:
                if free_symbols(found.value) - allowed or free_symbols(found.condition) - allowed:
                    # a constant of integration or a dummy leaked from a method
                    continue
                return found
        if fallback:
            return self._sympy(f, x, a, b, depth)
        return None

    # -- splitting ----------------------------------------------------------

    def _points_in(self, points: Set, a: Expr, b: Expr) -> Optional[list[Expr]]:
        """The points of a finite set which lie inside ``(a, b)``, sorted;
        ``None`` when a point cannot be placed."""
        if not isinstance(points, FiniteSet):
            return None
        inside: list[Expr] = []
        for p in points:
            p_ = as_expr(p)
            if not p_.is_extended_real and self.ask(as_boolean(p_ - p_ >= 0)) is not True:
                if p_.is_real is False:
                    continue
                return None
            below = self.ask(as_boolean(p_ > a)) if a != -oo else True
            above = self.ask(as_boolean(p_ < b)) if b != oo else True
            if below is None or above is None:
                return None
            if below and above:
                inside.append(p_)
        for i in range(len(inside)):
            for j in range(i + 1, len(inside)):
                if self.ask(as_boolean(inside[i] < inside[j])) is None:
                    return None
        return sorted(inside, key=lambda p: [0 if self.ask(as_boolean(p < q)) else 1 for q in inside])

    def _zeros(self, u: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[list[Expr]]:
        """The zeros of ``u`` in ``(a, b)``."""
        key = (u, a, b)
        if key not in self._zero_cache:
            self._zero_cache[key] = self._solve_zeros(u, x, a, b)
        return self._zero_cache[key]

    def _solve_zeros(self, u: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[list[Expr]]:
        zeros = _trigonometric_zeros(u, x, a, b)
        if zeros is not None:
            return zeros
        bounds: list[Boolean] = []
        if a != -oo:
            bounds.append(as_boolean(x > a))
        if b != oo:
            bounds.append(as_boolean(x < b))
        # the assumptions on the parameters of u only: an unrelated one
        # (a > 0 for the zeros of sin(t/2)) turns the solver's answer from
        # the empty set into a ConditionSet on the bounds
        relevant = [item for item in _items(self.assumptions) if free_symbols(item) & free_symbols(u)]
        assumptions = _with(relevant, bounds)
        try:
            found = attempt(lambda: solve(u, x, assumptions, domain=S.Reals), settings.timeout)
        except (ValueError, TypeError):
            return None
        if found is None:
            return None
        if found is S.EmptySet:
            return []
        return self._points_in(as_set(found), a, b)

    def _sample(self, a: Expr, b: Expr) -> Expr:
        if a == -oo and b == oo:
            return S.Zero
        if a == -oo:
            return as_expr(b - 1)
        if b == oo:
            return as_expr(a + 1)
        return as_expr((a + b) / 2)

    def _branch(self, f: Expr, x: Symbol, a: Expr, b: Expr, unplaced: frozenset[Expr] = frozenset(),
                full: bool = False) -> Optional[Expr]:
        """``f`` with ``Abs``, ``sign``, ``Heaviside``, ``Max``, ``Min`` and
        ``Piecewise`` of arguments depending on ``x`` replaced by their
        branch on ``(a, b)``, decided at a sample point, and the powers
        and logarithms of products split over the factors whose sign on
        ``(a, b)`` is known (:meth:`_signed`); ``None`` when a sign of the
        former is undecided. The factors in ``unplaced`` are those whose
        zeros could not be found, so their sample sign says nothing."""
        sample = self._sample(a, b)
        result = f

        def sign_of(u: Expr) -> Optional[int]:
            value = as_expr(u.subs(x, sample))
            if self.ask(as_boolean(value > 0)) is True:
                return 1
            if self.ask(as_boolean(value < 0)) is True:
                return -1
            return None

        for node, constant, factors in self._nodes(result, x, full):
            rewritten = self._signed(node, constant, factors, sign_of, unplaced, full)
            if rewritten is not None:
                result = as_expr(result.xreplace({node: rewritten}))
        for node in sorted(result.atoms(Abs, sign, Heaviside), key=lambda n: str(n)):
            u = as_expr(node.args[0])
            if not u.has(x):
                continue
            s = sign_of(u)
            if s is None:
                return None
            if isinstance(node, Abs):
                result = as_expr(result.xreplace({node: s * u}))
            elif isinstance(node, sign):
                result = as_expr(result.xreplace({node: Integer(s)}))
            else:
                result = as_expr(result.xreplace({node: S.One if s > 0 else S.Zero}))
        for node in sorted(result.atoms(Max, Min), key=lambda n: str(n)):
            if not node.has(x) or len(node.args) != 2:
                continue
            u, v = as_expr(node.args[0]), as_expr(node.args[1])
            s = sign_of(u - v)
            if s is None:
                return None
            larger, smaller = (u, v) if s > 0 else (v, u)
            result = as_expr(result.xreplace({node: larger if isinstance(node, Max) else smaller}))
        for node in result.atoms(Piecewise):
            if not node.has(x):
                continue
            chosen: Optional[Expr] = None
            for value, cond in _pairs(node):
                if cond is true:
                    chosen = value
                    break
                holds = self.ask(as_boolean(cond.subs(x, sample)))
                if holds is None:
                    return None
                if holds:
                    chosen = value
                    break
            if chosen is None:
                return None
            result = as_expr(result.xreplace({node: chosen}))
        return result

    def _signed(self, node: Expr, constant: Expr, factors: list[tuple[Expr, Rational]],
                sign_of: Callable[[Expr], Optional[int]], unplaced: frozenset[Expr],
                full: bool) -> Optional[Expr]:
        """A power ``P**r`` or a logarithm ``log(P)`` with ``P`` a product
        of factors of known sign on the piece split over the factors, or
        ``None`` when nothing is known.

        ``(u*v)**r = u**r * v**r`` and ``log(u*v) = log(u) + log(v)`` hold
        for ``u > 0`` and any ``v`` (the argument of the product is the
        argument of ``v``), so the factors known positive on the piece,
        with their multiplicities, and the positive constant come out one
        by one, a factor known negative as ``(-g)**e`` with the sign
        ``(-1)**e`` kept with the remainder, and the factors whose sign is
        unknown stay together in the remainder, which is ``(-1)**r`` at
        most when every sign is known: ``sqrt(c*u**2)`` is ``sqrt(c)*(s*u)``
        with the sign ``s`` of ``u`` on the piece, ``sqrt(x - 2 + 1/x)`` on
        ``(0, 1)`` is ``(1 - x)/sqrt(x)``, ``log(x**2)`` is ``2*log(-x)`` on
        the negative axis. Without ``full`` only the factors of even
        multiplicity and the positive constant come out (the square roots
        of squares), the other factors staying together for the methods
        which read a radicand as a whole."""
        exponent = as_expr(node.exp) if isinstance(node, Pow) else S.One
        positive: list[tuple[Expr, Rational]] = []
        remainder: list[Expr] = []
        total_sign = S.One
        for g, e in factors:
            if g in unplaced or (not full and not _even(e)):
                remainder.append(g**e)
                continue
            s = sign_of(g)
            if s is None:
                remainder.append(g**e)
                continue
            positive.append((s * g, e))
            total_sign *= Integer(s)**e
        if constant != 1:
            if self.ask(as_boolean(constant > 0)) is True:
                positive.append((constant, S.One))
            elif self.ask(as_boolean(constant < 0)) is True:
                positive.append((-constant, S.One))
                total_sign = -total_sign
            else:
                remainder.append(constant)
        if not positive:
            return None
        rest = as_expr(total_sign * Mul(*remainder))
        if isinstance(node, Pow):
            rewritten = as_expr(Mul(*[Pow(g, e * exponent) for g, e in positive]) * (rest**exponent if rest != 1 else 1))
        else:
            rewritten = as_expr(Add(*[e * log(g) for g, e in positive]) + (log(rest) if rest != 1 else 0))
        return None if rewritten == node else rewritten

    def _breakpoints(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                     full: bool = False) -> Optional[tuple[list[Expr], frozenset[Expr]]]:
        """The points of ``(a, b)`` where a branch of ``f`` changes: the
        zeros of the arguments of ``Abs`` and the like, and of the factors
        of the powers and logarithms to split (:meth:`_signed`); with the
        factors whose zeros could not be found, which are not split.
        ``None`` when a zero of the former cannot be placed."""
        points: list[Expr] = []
        arguments: list[Expr] = []
        for node in f.atoms(Abs, sign, Heaviside):
            arguments.append(as_expr(node.args[0]))
        for node in f.atoms(Max, Min):
            if len(node.args) == 2:
                arguments.append(as_expr(node.args[0]) - as_expr(node.args[1]))
        for node in f.atoms(Piecewise):
            for _, condition in _pairs(node):
                for atom in condition.atoms(Relational):
                    arguments.append(as_expr(atom.lhs) - as_expr(atom.rhs))
        for u in arguments:
            if not u.has(x):
                continue
            zeros = self._zeros(u, x, a, b)
            if zeros is None:
                return None
            points.extend(zeros)
        unplaced: set[Expr] = set()
        for _, _, factors in self._nodes(f, x, full):
            for g, e in factors:
                if not full and not _even(e):
                    continue
                zeros = self._zeros(g, x, a, b)
                if zeros is None:
                    unplaced.add(g)
                else:
                    points.extend(zeros)
        placed = self._points_in(FiniteSet(*points), a, b) if points else []
        if placed is None:
            return None
        return placed, frozenset(unplaced)

    def _split_at(self, f: Expr, x: Symbol, a: Expr, b: Expr, points: Sequence[Expr], depth: int,
                  branch: bool, unplaced: frozenset[Expr] = frozenset(),
                  full: bool = False) -> Optional[ConditionalValue]:
        bounds = [a] + list(points) + [b]
        total = ConditionalValue(S.Zero)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            piece = self._branch(f, x, lo, hi, unplaced, full) if branch else f
            if piece is None:
                return None
            # the late splitting is a rewriting: its pieces skip the slow
            # methods and SymPy, which the original form gets afterwards
            found = self.integrate(piece, x, lo, hi, depth + 1, not full, full)
            if found is None or found.value.has(zoo, nan):
                return None
            # a piece divergent to a signed infinity makes the whole so,
            # unless another cancels it (oo - oo below)
            total = total.add(found)
        if total.value.has(zoo, nan):
            return None
        return total

    def _split_signs(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int,
                     full: bool) -> Optional[ConditionalValue]:
        """The range cut where a branch of ``f`` changes, each piece
        integrated in its branch; ``None`` when there is nothing to cut or
        to rewrite."""
        if not f.atoms(Abs, sign, Heaviside, Max, Min, Piecewise) and not self._nodes(f, x, full):
            return None
        found = self._breakpoints(f, x, a, b, full)
        if found is None:
            return None
        points, unplaced = found
        if not points:
            piece = self._branch(f, x, a, b, unplaced, full)
            if piece is None or piece == f:
                return None
            return self.integrate(piece, x, a, b, depth + 1, not full, full)
        return self._split_at(f, x, a, b, points, depth, True, unplaced, full)

    def _signed_form(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> Expr:
        """``f`` with its powers and logarithms split by the signs of
        their factors on ``(a, b)`` when no factor changes sign inside,
        else ``f`` itself: ``(-cos(t))**(2/3)`` on ``(0, pi/2)`` is
        ``(-1)**(2/3)*cos(t)**(2/3)``, which the substitution of the Beta
        integral reads."""
        if not self._nodes(f, x, True):
            return f
        found = self._breakpoints(f, x, a, b, True)
        if found is None or found[0]:
            return f
        piece = self._branch(f, x, a, b, found[1], True)
        return f if piece is None else piece

    def _split_branches(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """The branches of ``Abs`` and the like, and the square roots of
        squares, before the methods."""
        return self._split_signs(f, x, a, b, depth, False)

    def _split_powers(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """The powers and logarithms of products split over their factors
        of known sign (``sqrt(x*(4 - x))`` as ``sqrt(x)*sqrt(4 - x)`` on
        ``(0, 4)``, ``log(sin(x)/x)`` as ``log(sin(x)) - log(x)`` on
        ``(0, pi/2)``), after the methods which read them whole."""
        return self._split_signs(f, x, a, b, depth, True)

    def _split_singularities(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                             depth: int) -> tuple[bool, Optional[ConditionalValue]]:
        """Whether singularities were found inside the range, and the sum
        of the integrals over the pieces between them."""
        if f.atoms(Abs, sign, Heaviside, Max, Min, Piecewise):
            return False, None
        found = attempt(lambda: singularities(f, x, Interval.open(a, b)), settings.timeout)
        if found is None:
            return False, None
        points = self._points_in(as_set(found), a, b)
        if not points:
            return False, None
        return True, self._split_at(f, x, a, b, points, depth, False)

    # -- the methods on a plain piece ---------------------------------------

    def _finish(self, found: Optional[ConditionalValue]) -> Optional[ConditionalValue]:
        if found is None or found.value.has(zoo, nan):
            return None
        condition = decide(found.condition, self.assumptions)
        if condition is None:
            return None
        return ConditionalValue(found.value, condition)

    def _mellin_on(self, g: Expr, t: Symbol, cutoff: Optional[str]) -> Optional[ConditionalValue]:
        """The Marichev–Adamchik method on ``g``, tried as it is and
        expanded."""
        for candidate in _forms(g, t, self.assumptions):
            found = mellin_integrate(candidate, t, self.assumptions, cutoff, self.regularize)
            if found is not None:
                return found
        return None

    def _canonical(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """The Marichev–Adamchik method on the ranges it works on."""
        t = Dummy('t', positive=True)
        if a == 0 and b == oo:
            found = self._mellin_on(f.subs(x, t), t, None)
            if found is None and f.has(exp):
                found = self._exponential_substitution(f, x, a, b)
            if found is None:
                found = self._series_mellin(as_expr(f.subs(x, t)), t)
            return self._finish(found)
        if a == 0 and b == 1:
            return self._finish(self._mellin_on(f.subs(x, t), t, 'lower'))
        if a == 1 and b == oo:
            return self._finish(self._mellin_on(f.subs(x, t), t, 'upper'))
        if a == -oo and b == oo and f.has(exp):
            return self._finish(self._exponential_substitution(f, x, a, b))
        return None

    def _mapped(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Other ranges mapped linearly onto `(0, oo)` and `(0, 1)`, and
        the real line cut at 0."""
        t = Dummy('t', positive=True)
        if a == -oo and b == oo:
            right = self.integrate(f, x, S.Zero, oo, depth + 1, False, True)
            if right is None:
                return None
            left = self.integrate(f, x, -oo, S.Zero, depth + 1, False, True)
            return None if left is None else left.add(right)
        if a == -oo:
            return self.integrate(as_expr(f.subs(x, b - t)), t, S.Zero, oo, depth + 1, False, True)
        if b == oo:
            if a == 0:
                return None
            return self.integrate(_shift(f, x, t, a), t, S.Zero, oo, depth + 1, False, True)
        if a == 0:
            if b == 1:
                return None
            # (0, b): x = b t
            if self.ask(as_boolean(b > 0)) is not True:
                return None
            g = as_expr(f.subs(x, b * t) * b)
            return self.integrate(g, t, S.Zero, S.One, depth + 1, False, True)
        # (a, b): x = a + (b - a) t, and reflected, x = b - (b - a) t, when
        # that fails: the singular end of log(-x)/sqrt(1 - x**2) over
        # (-1, 0) is at 0, which the reflection sends to t = 0 where the
        # Mellin method reads it
        length = as_expr(b - a)
        g = as_expr(f.subs(x, a + length * t) * length)
        found = self.integrate(g, t, S.Zero, S.One, depth + 1, False, True)
        if found is not None:
            return found
        g = as_expr(f.subs(x, b - length * t) * length)
        return self.integrate(g, t, S.Zero, S.One, depth + 1, False, True)

    def _series_mellin(self, g: Expr, t: Symbol) -> Optional[ConditionalValue]:
        """``Integral(g, (t, 0, oo))`` as the Mellin transform at ``s = 1``
        found by Ramanujan's master theorem or the method of brackets
        (:mod:`.brackets`), for factors outside the table."""
        from .brackets import mellin_transform_series
        s = Dummy('s')
        found = mellin_transform_series(g, t, s)
        if found is None:
            return None
        value = as_expr(found.transform.subs(s, 1))
        if value.has(nan, zoo, oo, -oo):
            value_ = attempt(lambda: as_expr(limit(found.transform, s, 1)), settings.timeout)
            if value_ is None or value_.has(nan, zoo, oo, -oo):
                return None
            value = value_
        lower, upper = found.strip
        parts: list[Boolean] = []
        if lower != -oo:
            parts.append(as_boolean(lower < 1))
        if upper != oo:
            parts.append(as_boolean(upper > 1))
        return ConditionalValue(value, as_boolean(And(*parts, found.condition)))

    def _exponential_substitution(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[ConditionalValue]:
        """``t = exp(c x)`` for an integrand in ``exp(c_i x)`` with all
        ``c_i`` integer multiples of ``c``: the real line becomes
        ``(0, oo)``, and ``(0, oo)`` becomes ``(0, 1)`` with ``c < 0``."""
        c = _exponential_generator(f, x, negative=(a == 0))
        if c is None:
            return None
        u = Dummy('u', positive=True)
        g = _denest(as_expr(f.subs(x, log(u) / c) / (c * u)), self.assumptions)
        if g.has(log(u)) and not _log_powers_only(g, u):
            return None
        if a == 0:
            cutoff = 'lower' if c < 0 else 'upper'
            found = self._mellin_on(g, u, cutoff)
            return None if found is None else found.scaled(S.NegativeOne if c < 0 else S.One)
        found = self._mellin_on(g, u, None)
        return None if found is None else found.scaled(S.NegativeOne if c < 0 else S.One)

    def _inversion(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """``x = 1/u`` between ``(1, oo)`` and ``(0, 1)``."""
        if (a, b) not in ((S.One, oo), (S.Zero, S.One)):
            return None
        u = Dummy('u', positive=True)
        g = _denest(as_expr(f.subs(x, 1 / u) / u**2), self.assumptions)
        cutoff = 'lower' if b == oo else 'upper'
        return self._finish(self._mellin_on(g, u, cutoff))

    def _trigonometric(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Trigonometric integrands over ranges with endpoints at multiples
        of ``pi/2``: cut into quarter periods, each mapped onto
        ``(0, pi/2)`` and integrated as a Beta integral through
        ``x = asin(sqrt(u))``."""
        if not f.has(TrigonometricFunction) or a in (-oo, oo) or b in (-oo, oo):
            return None
        ka, kb = _quarter(a), _quarter(b)
        if ka is None or kb is None:
            return None
        if kb - ka > 1:
            t = Dummy('t')
            total = ConditionalValue(S.Zero)
            for k in range(ka, kb):
                piece = self.integrate(as_expr(f.subs(x, t + k * pi / 2)), t, S.Zero, pi / 2, depth + 1, False, True)
                if piece is None:
                    return None
                total = total.add(piece)
            return total
        if ka != 0:
            t = Dummy('t')
            return self.integrate(as_expr(f.subs(x, t + ka * pi / 2)), t, S.Zero, pi / 2, depth + 1, False, True)
        u = Dummy('u', positive=True)
        f = self._signed_form(f, x, S.Zero, pi / 2)
        g = as_expr(f.subs(x, asin(sqrt(u))) / (2 * sqrt(u) * sqrt(1 - u)))
        g = _denest(g, self.assumptions)
        # and after the substitution: -log(cos(x)) is -log(sqrt(1 - u)),
        # which is -log(1 - u)/2 on (0, 1)
        g = self._signed_form(g, u, S.Zero, S.One)
        if g.has(asin):
            return None
        found = self._finish(self._mellin_on(g, u, 'lower'))
        if found is not None:
            return found
        # the other methods on the substituted form: -log(1 - u)/(4*u), from
        # log(1/cos(x))*cos(x)/sin(x), has a dilogarithm antiderivative
        return self.integrate(g, u, S.Zero, S.One, depth + 1, False, True)

    def _mean_value(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """A trigonometric integrand over whole periods as `2 pi k` times
        the constant Laurent coefficient of its form in ``exp(I x)``
        (:mod:`.periodic`)."""
        from .periodic import mean_value_integral
        if not f.has(TrigonometricFunction) or a in (-oo, oo) or b in (-oo, oo):
            return None
        return self._finish(mean_value_integral(f, x, a, b, self.assumptions))

    def _residues(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        from .residues import residue_integral
        return self._finish(residue_integral(f, x, a, b, self.assumptions))

    def _real_parts(self, condition: Boolean) -> Boolean:
        """SymPy's conditions with ``re(p)`` replaced by ``p`` and ``im(p)``
        by 0 for the parameters known to be real."""
        replacement: dict[Expr, Expr] = {}
        for node in condition.atoms(re, im):
            p = as_expr(node.args[0])
            if isinstance(p, Symbol) and (p.is_extended_real or self.ask(element(p, S.Reals)) is True):
                replacement[as_expr(node)] = p if isinstance(node, re) else S.Zero
        return as_boolean(condition.xreplace(replacement)) if replacement else condition

    def _radicals(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """``x**n * Q**(m/2)`` with ``Q`` quadratic (:mod:`.radicals`): the
        real antiderivative of the table evaluated at the endpoints, cheap
        and early, before the methods which spend the budget on the sums
        of radicals the cells of a region leave (the bug: the slices of
        the cylinders reached the antiderivative route, the last one, with
        no time left)."""
        from .radicals import quadratic_radical_antiderivative
        if a.has(x) or b.has(x) or quadratic_radical_antiderivative(f, x) is None:
            return None
        return self._antiderivative(f, x, a, b, depth)

    def _antiderivative(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int,
                        late: bool = False) -> Optional[ConditionalValue]:
        """An antiderivative evaluated by one-sided limits at the
        endpoints and at its discontinuities (:mod:`.antiderivative`):
        by the exact methods first, by the heuristics ``late``."""
        from .antiderivative import antiderivative_integral
        if a.has(x) or b.has(x):
            return None
        found = antiderivative_integral(f, x, a, b, self.assumptions, late)
        if found is None:
            return None
        if found.value in (oo, -oo):
            # a divergence to a signed infinity, from an infinite one-sided
            # limit of the antiderivative: refused when the quadrature
            # converges (a wrong antiderivative, a branch cut taken for a
            # pole), kept otherwise
            if settings.numerical_checks and _quadrature_converges(f, x, a, b, self.assumptions):
                return None
            return self._finish(found)
        if settings.numerical_checks and verify_numerically(found.value, f, x, a, b, self.assumptions) is not True:
            # the antiderivatives sometimes hold for positive parameters
            # or principal branches only: kept when confirmed numerically
            return None
        return self._finish(found)

    def _late_antiderivative(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                             depth: int) -> Optional[ConditionalValue]:
        """The antiderivative by the heuristic methods (the Risch–Norman
        method, the substitutions, SymPy's manual and Meijer routes), last
        of all: ``t*exp(-sqrt(t))*log(t)`` over ``(x, oo)`` has one in
        exponential integrals from SymPy in seconds."""
        return self._antiderivative(f, x, a, b, depth, True)

    def _termwise(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """A sum integrated term by term when the whole defeats every
        method: the antiderivative of ``sqrt(x)*sqrt(2 - x) - sqrt(1 -
        x)*sqrt(x + 1)`` is a complex ``Piecewise`` whose branches give
        ``nan`` at the endpoints, while each term is elementary. The value
        is kept only when every term has a finite value (terms which
        diverge separately may cancel in the sum, ``1/x - 1/(x + x**2)``
        near 0) and the sum passes the numerical check; a product which
        expands to a few terms is tried in the expanded form."""
        terms = [as_expr(term) for term in Add.make_args(f)]
        if len(terms) < 2:
            expanded = as_expr(expand(f))
            terms = [as_expr(term) for term in Add.make_args(expanded)]
            if len(terms) < 2 or len(terms) > _TERMWISE_LIMIT:
                return None
        elif len(terms) > _TERMWISE_LIMIT:
            return None
        total: Optional[ConditionalValue] = None
        for term in terms:
            found = self.integrate(term, x, a, b, depth + 1, False, False)
            if found is None or found.value.has(oo, -oo, zoo, nan):
                return None
            total = found if total is None else total.add(found)
        if total is None:
            return None
        if settings.numerical_checks and verify_numerically(total.value, f, x, a, b, self.assumptions) is False:
            return None
        return self._finish(total)

    def _dirichlet(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Trigonometric sums over powers of ``x`` on the half-lines and
        the real line (:mod:`.dirichlet`): the Dirichlet, Frullani and
        Borwein integrals."""
        from .dirichlet import dirichlet_integral
        if a not in (S.Zero, -oo) or b not in (S.Zero, oo) or not f.has(sin, cos, exp):
            return None
        return self._finish(dirichlet_integral(f, x, a, b, self.assumptions))

    def _table(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """The table of Gradshteyn and Ryzhik (:mod:`.tables`), cheap and
        first."""
        from .tables import table_integral
        return self._finish(table_integral(f, x, a, b, self.assumptions))

    def _elliptic(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Square roots of cubics and quartics reduced to Legendre's
        elliptic integrals (:mod:`.elliptic`)."""
        from .elliptic import elliptic_integral
        if not any(isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q == 2
                   for node in f.atoms(Pow)):
            return None
        return self._finish(elliptic_integral(f, x, a, b, self.assumptions))

    def _algebraic(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Rationalising substitutions for algebraic integrands of genus
        zero (:mod:`.algebraic`): Euler's substitutions, Möbius roots,
        Chebyshev's binomial differentials."""
        from .algebraic import algebraic_integral
        if depth > 1 or not any(isinstance(node, Pow) and isinstance(node.exp, Rational) and not node.exp.is_integer
                                and node.base.has(x) for node in f.atoms(Pow)):
            return None
        return self._finish(algebraic_integral(f, x, a, b, self.assumptions))

    def _series(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Series expansion of a factor and termwise integration, the
        series summed in closed form (:mod:`.series`)."""
        from .series import series_integral
        if depth > 0:
            return None
        if a != 0:
            # a Fourier series on a bounded range, (-pi, pi) or (0, t)
            if a in (oo, -oo) or b in (oo, -oo) or not f.has(log, TrigonometricFunction):
                return None
        elif b not in (S.One, oo) and not b.is_positive:
            return None
        return self._finish(series_integral(f, x, a, b, self.assumptions))

    def _laplace(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """The operational rules of the Laplace transform
        (:mod:`.laplace`) for ``g(x)*exp(-s*x)`` over ``(0, oo)``."""
        from .laplace import laplace_integral
        if depth > 1 or a != 0 or b != oo or not f.has(exp):
            return None
        return self._finish(laplace_integral(f, x, a, b, self.assumptions))

    def _contours(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Rectangular, sector and indented contours (:mod:`.contours`)."""
        from .contours import contour_integral
        return self._finish(contour_integral(f, x, a, b, self.assumptions))

    def _transformations(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Frullani's theorem and Glasser's master theorem
        (:mod:`.transformations`)."""
        from .transformations import transformation_integral
        if depth > 1:
            return None
        return self._finish(transformation_integral(f, x, a, b, self.assumptions))

    def _holonomic(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Creative telescoping (:mod:`.telescoping`) for a hyperexponential
        integrand with one parameter: the integral satisfies a linear
        ODE in the parameter, solved with initial conditions."""
        from .reduction import reduction_integral
        from .telescoping import holonomic_integral, is_hyperexponential
        if not self.parametric or depth > 1:
            return None
        parameters = sorted_symbols(free_symbols(f) - {x} - free_symbols(a) - free_symbols(b))
        if len(parameters) != 1 or not is_hyperexponential(f, x, parameters[0]):
            return None
        # the reduction-based telescoper first (minimal order, no bound
        # on the certificate), the ansatz as the fallback
        found = reduction_integral(f, x, a, b, parameters[0], self.assumptions)
        if found is None:
            found = holonomic_integral(f, x, a, b, parameters[0], self.assumptions)
        return self._finish(found)

    def _dfinite(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Chyzak's algorithm (:mod:`.dfinite`): creative telescoping for
        a D-finite integrand with one parameter which is not
        hyperexponential."""
        from .dfinite import dfinite_integral
        from .telescoping import is_hyperexponential
        if not self.parametric or depth > 1:
            return None
        parameters = sorted_symbols(free_symbols(f) - {x} - free_symbols(a) - free_symbols(b))
        if len(parameters) != 1 or is_hyperexponential(f, x, parameters[0]):
            return None
        return self._finish(dfinite_integral(f, x, a, b, parameters[0], self.assumptions))

    def _parametric(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Differentiation under the integral sign (:mod:`.parametric`),
        for each parameter of the integrand in turn."""
        from .parametric import candidate_parameters, parametric_integral
        if not self.parametric or depth > 1:
            return None
        for p in candidate_parameters(f, x):
            found = parametric_integral(f, x, a, b, p, self.assumptions, depth)
            if found is not None:
                return self._finish(found)
        return None

    def _sympy(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int = 0) -> Optional[ConditionalValue]:
        """SymPy's ``integrate`` under the time limit, checked numerically."""
        limits = (x, a, b)
        # a quarter of the time limit for each of the two forms: SymPy's
        # half of the limit in all (each ran under the whole limit, and an
        # integral no method takes cost 75 s under a limit of 30)
        budget = None if settings.timeout is None else settings.timeout / 4
        for keywords in ({}, {'meijerg': True}):
            try:
                value = attempt(lambda: as_expr(integrate(f, limits, **keywords)), budget)
            except AttributeError:
                # SymPy 1.14: meijerg._eval_evalf raises AttributeError
                # ('NoneType' object has no attribute 'has') on some
                # arguments, e.g. integrate(exp(-x)*sin(x)/x**2, (x, 0, oo))
                value = None
            if value is None:
                continue
            found = _from_sympy(value, x)
            if found is None:
                continue
            found = self._finish(ConditionalValue(found.value, self._real_parts(found.condition)))
            if found is None:
                continue
            if settings.numerical_checks and not f.has(DiracDelta):
                if found.value.has(oo, -oo, zoo, nan):
                    continue
                if verify_numerically(found.value, f, x, a, b, self.assumptions) is not True:
                    # an answer of integrate which the quadrature cannot
                    # confirm is not kept: the branch cuts of the
                    # antiderivative make such answers wrong too often
                    # (a delta function cannot be integrated numerically
                    # and is handled by integrate symbolically)
                    continue
            return ConditionalValue(tidy(found.value, self.assumptions), found.condition)
        return None


def _pairs(node: Piecewise) -> list[tuple[Expr, Boolean]]:
    """The ``(value, condition)`` pairs of a ``Piecewise``."""
    return [(as_expr(pair.args[0]), as_boolean(pair.args[1])) for pair in node.args]


def _with(assumptions: Assumptions, extra: Sequence[Boolean]) -> Assumptions:
    """The assumptions together with ``extra``."""
    if not extra:
        return assumptions
    items: list[Boolean] = list(extra)
    if isinstance(assumptions, (Basic, bool)):
        items.append(as_boolean(assumptions))
    elif assumptions is not None:
        items.extend(as_boolean(a) for a in assumptions)
    return items


def _from_sympy(value: Expr, x: Symbol) -> Optional[ConditionalValue]:
    """A result of ``integrate`` as a conditional value: a ``Piecewise``
    whose last branch is the unevaluated integral gives the first branch
    with its condition."""
    if value.has(Integral):
        if isinstance(value, Piecewise) and len(value.args) == 2:
            first, condition = _pairs(value)[0]
            if not first.has(Integral) and not condition.has(x):
                return ConditionalValue(first, condition)
        return None
    if value.has(x):
        return None
    return ConditionalValue(value)


def _forms(g: Expr, t: Symbol, assumptions: Assumptions) -> list[Expr]:
    """The integrand as it is and in the forms the Mellin table reads:
    expanded, denested, products and powers of ``sin`` and ``cos``
    written as sums (``sin(x)**2`` is ``1/2 - cos(2 x)/2``), hyperbolic
    functions as exponentials, inverse hyperbolic functions as
    logarithms, orthogonal polynomials expanded."""
    forms = [g]

    def add(e: Expr) -> None:
        if e not in forms:
            forms.append(e)

    # the exponentials kept whole: expand writes exp(-(a + s)*t) as
    # exp(-a*t)*exp(-s*t), two kernels for Parseval's formula with the
    # condition a > 0 on each where a + s > 0 is the one
    add(as_expr(g.expand(power_exp=False)))
    add(_denest(g, assumptions))
    if g.has(sin, cos):
        add(as_expr(TR8(g).expand()))
    if g.has(HyperbolicFunction):
        add(as_expr(g.rewrite(exp).expand()))
    if g.has(InverseHyperbolicFunction):
        rewritten = as_expr(g.rewrite(log).expand())
        add(rewritten)
        add(as_expr(expand_log(rewritten, force=True)))
    if g.has(OrthogonalPolynomial):
        add(as_expr(expand_func(g).expand()))
    return forms


def _denest(g: Expr, assumptions: Assumptions) -> Expr:
    """``powdenest(g, force=True)`` done honestly: the forced denesting
    takes every symbol positive (``sqrt(c**2)`` becomes ``c``), so a
    parameter known negative is first written ``-d`` with ``d`` positive,
    and a parameter of unknown sign blocks the forced form (the bug: the
    area of the disc ``x**2 + y**2 < c**2`` came out as 0 for ``c < 0``)."""
    replacement: dict[Expr, Expr] = {}
    back: dict[Expr, Expr] = {}
    for p in sorted_symbols(free_symbols(g)):
        if p.is_positive:
            continue
        if ask(as_boolean(p > 0), assumptions) is True:
            d = Dummy(p.name, positive=True)
            replacement[p] = d
            back[d] = p
        elif ask(as_boolean(p < 0), assumptions) is True:
            d = Dummy(p.name, positive=True)
            replacement[p] = -d
            back[d] = -p
        else:
            return as_expr(powdenest(g))
    denested = as_expr(powdenest(g.xreplace(replacement), force=True))
    return as_expr(denested.xreplace(back))


def _shift(f: Expr, x: Symbol, t: Symbol, a: Expr) -> Expr:
    return as_expr(f.subs(x, t + a))


def _quarter(e: Expr) -> Optional[int]:
    """``k`` when ``e == k*pi/2``."""
    k = as_expr(2 * e / pi)
    if isinstance(k, Integer):
        return int(k)
    return None


def _exponential_generator(f: Expr, x: Symbol, negative: bool) -> Optional[Expr]:
    """``c`` such that every ``exp(c_i x)`` of ``f`` has ``c_i`` an integer
    multiple of ``c`` (the rational ``c_i`` reduced to their common
    fraction), with the sign of the ``c_i`` when they share one and
    ``negative`` deciding otherwise; ``None`` without such exponentials."""
    coefficients: list[Rational] = []
    symbolic = 0
    for node in f.atoms(exp):
        argument = as_expr(node.args[0])
        found = monomial(argument, x)
        if found is None or found[1] != 1:
            return None
        if isinstance(found[0], Rational):
            coefficients.append(found[0])
        else:
            symbolic += 1
    if not coefficients and not symbolic:
        return None
    numerator = 0
    denominator = 1
    for c in coefficients:
        numerator = gcd(numerator, int(c.p))
        denominator = lcm(denominator, int(c.q))
    c = Rational(numerator, denominator) if coefficients else S.One
    if coefficients and all(k < 0 for k in coefficients):
        return -c
    if coefficients and all(k > 0 for k in coefficients):
        return c
    return -c if negative else c


def _log_powers_only(g: Expr, u: Symbol) -> bool:
    """Whether ``log(u)`` occurs only as ``log(u)**n``."""
    for node in g.atoms(log):
        if node.has(u) and node != log(u):
            return False
    return True


def _trigonometric_zeros(u: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[list[Expr]]:
    """The zeros of ``sin(w*x + c)`` or ``cos(w*x + c)`` with numeric
    ``w`` and ``c`` inside a bounded numeric range ``(a, b)``, enumerated
    (``solve`` spends seconds on them, and answers with a ConditionSet
    under an assumption on another symbol); ``None`` for any other
    ``u`` or range."""
    if not isinstance(u, (sin, cos)) or not (a.is_number and b.is_number) or a in (-oo, oo) or b in (-oo, oo):
        return None
    argument = as_expr(u.args[0])
    if not argument.is_polynomial(x) or degree(argument, x) != 1:
        return None
    w, c = as_expr(argument.coeff(x)), as_expr(argument.subs(x, 0))
    if not (w.is_number and c.is_number and w.is_real and c.is_real) or w == 0:
        return None
    # the zeros are at w x + c = k pi, or k pi + pi/2 for the cosine
    shift = S.Zero if isinstance(u, sin) else pi / 2
    ends = sorted([float(as_expr((w * a + c - shift) / pi).evalf()), float(as_expr((w * b + c - shift) / pi).evalf())])
    if ends[1] - ends[0] > 1000:
        return None
    zeros: list[Expr] = []
    for k in range(int(ends[0]) - 1, int(ends[1]) + 2):
        point = as_expr((k * pi + shift - c) / w)
        above, below = as_expr(point - a).is_positive, as_expr(b - point).is_positive
        if above is None or below is None:
            return None
        if above and below:
            zeros.append(point)
    return sorted(zeros, key=lambda p: float(p.evalf()))


def _combined_exponentials(f: Expr) -> Expr:
    """The exponential factors of each product combined into one,
    ``exp(-a*t)*exp(-s*t)`` into ``exp(-(a + s)*t)``: the identity
    ``exp(u)*exp(v) = exp(u + v)`` holds everywhere, and the methods read
    one exponential (the bug: the product went through Parseval's formula
    as two kernels, with the condition ``a > 0`` on each instead of
    ``a + s > 0`` on the sum)."""
    def combine(node: Basic) -> Basic:
        exponentials = [as_expr(part) for part in node.args if isinstance(part, exp)]
        if len(exponentials) < 2:
            return node
        rest = [as_expr(part) for part in node.args if not isinstance(part, exp)]
        return Mul(*rest) * exp(Add(*[as_expr(e.args[0]) for e in exponentials]))

    if not f.has(exp):
        return f
    return as_expr(f.replace(lambda n: isinstance(n, Mul), combine))


def _sums_to_products(f: Expr, x: Symbol) -> Expr:
    """A quotient of sums of sines and cosines of multiples of ``x`` with
    the sums written as products (the sum-to-product formulas, ``TR9``)
    and cancelled: ``(sin(19*x) + sin(20*x))/(cos(19*x) + cos(20*x))`` is
    ``tan(39*x/2)``; ``f`` itself when nothing cancels."""
    numerator, denominator = f.as_numer_denom()
    numerator_, denominator_ = as_expr(numerator), as_expr(denominator)

    def trigonometric_sum(e: Expr) -> bool:
        if not isinstance(e, Add):
            return False
        for term in e.args:
            _, rest = as_expr(term).as_independent(x, as_Add=False)
            if not isinstance(rest, (sin, cos)) or not as_expr(rest.args[0]).is_polynomial(x):
                return False
        return True

    if not (trigonometric_sum(numerator_) or trigonometric_sum(denominator_)):
        return f
    rewritten = as_expr(cancel(as_expr(TR9(numerator_)) / as_expr(TR9(denominator_))))
    return rewritten if count_ops(rewritten) < count_ops(f) else f


def _trigonometric_squares(base: Expr) -> Expr:
    """The trigonometric sums which are squares written as such, so that
    a power of the sum splits over the factors: ``c - c*cos(u)`` is
    ``2*c*sin(u/2)**2`` and ``c + c*cos(u)`` is ``2*c*cos(u/2)**2`` (the
    half-angle formulas), and a sum which ``trigsimp`` turns into a product
    or a power is taken in that form (``tan(u)**2 + 1`` is ``cos(u)**(-2)``,
    the cardioid's ``a**2*(1 - cos(t))**2 + a**2*sin(t)**2`` is
    ``2*a**2*(1 - cos(t))``)."""
    def half_angle(node: Basic) -> Basic:
        if not isinstance(node, Add) or len(node.args) != 2:
            return node
        terms = [as_expr(t) for t in node.args]
        for constant, other in ((terms[0], terms[1]), (terms[1], terms[0])):
            if constant.has(TrigonometricFunction):
                continue
            cosines = [as_expr(part) for part in Mul.make_args(other) if isinstance(part, cos)]
            if len(cosines) != 1:
                continue
            coefficient = as_expr(other / cosines[0])
            u = as_expr(cosines[0].args[0])
            if coefficient == constant:
                return 2 * constant * cos(u / 2)**2
            if coefficient == -constant:
                return 2 * constant * sin(u / 2)**2
        return node

    def squares(node: Expr) -> Expr:
        # the outer sums first, the constants factored out or not (the
        # cardioid's a**2*((1 - cos(t))**2 + sin(t)**2) as well as the
        # expanded sum): trigsimp gives 2 - 2*cos(t), a square by the
        # half-angle formula, only while its 1 - cos(t) is not yet written
        # as one
        if isinstance(node, Add) and node.has(TrigonometricFunction):
            found = as_expr(trigsimp(node))
            found = as_expr(found.replace(lambda n: isinstance(n, Add), half_angle))
            if not isinstance(found, Add):
                return found
        if not node.args:
            return node
        return as_expr(node.func(*[squares(as_expr(argument)) for argument in node.args]))

    return as_expr(squares(base).replace(lambda n: isinstance(n, Add), half_angle))


#: the largest base of a power factored for its signs
_FACTOR_LIMIT = 200


def _factored(base: Expr, x: Symbol) -> Optional[tuple[Expr, list[tuple[Expr, Rational]], bool]]:
    """``base`` as a constant times powers of factors depending on ``x``
    with integer or rational exponents (a radical factor, ``sqrt(1 - v)``
    in ``log(sqrt(1 - v))``, keeps its base and its exponent): the
    trigonometric squares written as such, the numerator and the
    denominator factored over the rationals; and whether a square
    appeared (a factor of integer multiplicity two or more, a
    trigonometric square). Each factor is continuous, so that it keeps
    one sign between its zeros; ``None`` when a factor has poles
    (``tan``, a negative power of ``x`` inside it) or the base is too
    large to factor."""
    if count_ops(base) > _FACTOR_LIMIT:
        return None
    g = _trigonometric_squares(base) if base.has(TrigonometricFunction) else base
    squares = g != base
    try:
        factored = attempt(lambda: as_expr(factor(together(g))), settings.timeout)
    except (PolynomialError, ValueError, TypeError):
        return None
    if factored is None:
        return None
    constant, rest = factored.as_independent(x, as_Add=False)
    factors: list[tuple[Expr, Rational]] = []
    for part in Mul.make_args(as_expr(rest)):
        b, e = part.as_base_exp()
        b_, e_ = as_expr(b), as_expr(e)
        if not isinstance(e_, Rational):
            # a symbolic power, continuous where it is real, as one factor
            b_, e_ = as_expr(part), S.One
        if b_.has(tan, cot, sec, csc, Piecewise, floor, ceiling, frac, sign, Heaviside) or any(
                isinstance(node, Pow) and node.base.has(x) and as_expr(node.exp).is_negative
                for node in b_.atoms(Pow)):
            return None
        if not isinstance(e_, Rational):
            return None
        factors.append((b_, e_))
        squares = squares or (isinstance(e_, Integer) and abs(int(e_)) >= 2)
    return as_expr(constant), factors, squares


def _even(e: Rational) -> bool:
    """Whether a multiplicity is an even integer (the square roots of
    squares are rewritten before the other methods)."""
    return isinstance(e, Integer) and int(e) % 2 == 0


def _signed_nodes(f: Expr, x: Symbol) -> list[Expr]:
    """The powers with a non-integer exponent and the logarithms whose
    argument depends on ``x``, the candidates of :meth:`_Integrator._signed`."""
    found: list[Expr] = []
    for node in sorted(f.atoms(Pow, log), key=lambda n: str(n)):
        if isinstance(node, Pow):
            base, exponent = as_expr(node.base), as_expr(node.exp)
            if isinstance(exponent, Integer) or not base.has(x):
                continue
        elif not node.args[0].has(x):
            continue
        found.append(node)
    return found

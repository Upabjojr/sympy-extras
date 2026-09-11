"""Asymptotic expansions of parametric definite integrals for a large
parameter: Watson's lemma, Laplace's method and the method of stationary
phase.

For `t \\to \\infty` the integrals

.. math::

    \\int_0^\\infty e^{-tx} \\varphi(x)\\, dx, \\qquad
    \\int_a^b \\varphi(x) e^{t h(x)}\\, dx, \\qquad
    \\int_a^b \\varphi(x) e^{i t h(x)}\\, dx

are dominated by the neighbourhood of one point: the origin, the maximum
of `h`, the stationary point of `h`. The expansions are

* **Watson's lemma** ([Bleistein]_ 4.1, [DLMF]_ 2.3(ii)): with
  `\\varphi(x) \\sim \\sum_k c_k x^{a_k}` at `x = 0`, `a_k > -1`,

  .. math:: \\int_0^\\infty e^{-tx} \\varphi(x)\\, dx \\sim \\sum_k c_k \\Gamma(a_k + 1)\\, t^{-a_k - 1};

* **Laplace's method** ([Bleistein]_ 5.1, [Olver]_ 3.7, [DLMF]_ 2.3(iii)):
  at an interior maximum `x_0` of `h`, with `\\alpha = -h''(x_0) > 0` and
  `u = x - x_0`, the expansion of `\\varphi(x_0 + u)\\, e^{t(h(x_0 + u) - h(x_0) + \\alpha u^2/2)}`
  in powers of `u` integrated against the Gaussian moments
  `\\int u^{2k} e^{-t\\alpha u^2/2} du = \\sqrt{2\\pi/(t\\alpha)}\\,(2k - 1)!!\\,(t\\alpha)^{-k}`;
  the first two terms are

  .. math::

      e^{t h(x_0)} \\sqrt{\\frac{2\\pi}{t \\alpha}} \\left( \\varphi_0 + \\frac{1}{t}\\left[
      \\frac{\\varphi_0 h''''}{8\\alpha^2} + \\frac{5 \\varphi_0 h'''^2}{24 \\alpha^3}
      + \\frac{\\varphi' h'''}{2\\alpha^2} + \\frac{\\varphi''}{2\\alpha} \\right] \\right),

  and at an endpoint maximum (`h'(a) < 0`) the expansion of
  `\\varphi(a + u) e^{t(h(a + u) - h(a) - h'(a) u)}` against
  `\\int_0^\\infty u^k e^{-t|h'(a)| u} du = k!\\,(t|h'(a)|)^{-k-1}` (integration
  by parts, [Bleistein]_ 3.3);

* **stationary phase** ([Bleistein]_ 6.1, [DLMF]_ 2.3(iv)): at a
  stationary point `x_0` of `h` the leading term is
  `\\varphi(x_0) \\sqrt{2\\pi/(t|h''(x_0)|)}\\, e^{i t h(x_0) \\pm i\\pi/4}`, the
  sign that of `h''(x_0)`; for `\\cos(t h)` and `\\sin(t h)` the real and
  imaginary parts. Only the leading term is computed.

Laplace's method sums the contributions of all the global maxima of `h`
on the closed range (maxima of lower height are exponentially smaller
and dropped): an interior maximum of order `2m`,
`h(x_0 + u) - h(x_0) = -c u^{2m} + \\dots`, contributes
`\\varphi(x_0) e^{t h(x_0)}\\, \\Gamma(1/2m)/(m (tc)^{1/2m})` and its
corrections in powers of `t^{-1/2m}` ([Bleistein]_ 5.1, [Olver]_ 3.8),
an endpoint where `h` decreases into the range with a vanishing slope
half of it (`e^{t\\cos x}` on `(0, 2\\pi)`: the ends `0` and `2\\pi` give
`\\sqrt{2\\pi/t}\\, e^t`, the asymptotics of `2\\pi I_0(t)`). On the real
line without a real maximum the **saddle points** in the complex plane
are used ([Bleistein]_ 7.2, [Olver]_ 4.7): each saddle `z_0` on the
deformed contour contributes `\\varphi(z_0) e^{t h(z_0)}
\\sqrt{2\\pi/(-t h''(z_0))}`; which saddles lie on the contour is decided
by a quadrature except for a quadratic `h`. The **uniform expansions**
for coalescing stationary points ([Bleistein]_ 9.2-9.4, [DLMF]_ 2.4(v),
[Chester]_, [Wong]_ VII) write `h` as a cubic or a quadratic in a new
variable and give the Airy and error function forms of
:func:`uniform_expansion`, valid as the stationary points merge with
each other or with an endpoint.

The expansions come with an :class:`~sympy.series.order.Order` term in the
parameter at infinity; an expansion which terminates (a polynomial
`\\varphi` under Watson's lemma) is returned exactly.

Examples
========

>>> from sympy import symbols, exp, sqrt, oo, log, cos, sin, pi
>>> from sympy_extras.integrals.asymptotic import asymptotic_integral
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> asymptotic_integral(exp(-t*x)/(1 + x), x, 0, oo, t)
2/t**3 - 1/t**2 + 1/t + O(t**(-4), (t, oo))
>>> asymptotic_integral(exp(-t*x)*sqrt(x), x, 0, oo, t)
sqrt(pi)/(2*t**(3/2))
>>> asymptotic_integral(exp(t*(log(x) - x)), x, 0, oo, t, order=2)
sqrt(2)*sqrt(pi)*(t + 1/12)*exp(-t)/t**(3/2) + O(exp(-t)/t**(5/2), (t, oo))
>>> asymptotic_integral(cos(t*sin(x))/pi, x, 0, pi, t)
sqrt(2)*sin(t + pi/4)/(sqrt(pi)*sqrt(t)) + O(t**(-3/2), (t, oo))

References
==========

.. [Bleistein] N. Bleistein, R. A. Handelsman, *Asymptotic Expansions of
   Integrals*, Dover, 1986, chapters 3–7 and 9.
.. [Olver] F. W. J. Olver, *Asymptotics and Special Functions*, Academic
   Press, 1974, chapter 3.
.. [DLMF] NIST Digital Library of Mathematical Functions, sections 2.3
   and 2.4, https://dlmf.nist.gov/2.3; A. Erdélyi, *Asymptotic
   Expansions*, Dover, 1956, chapter 2.
.. [Chester] C. Chester, B. Friedman, F. Ursell, *An extension of the
   method of steepest descents*, Proc. Cambridge Philos. Soc. 53 (1957),
   599-611.
.. [Wong] R. Wong, *Asymptotic Approximations of Integrals*, SIAM, 2001,
   chapters II and VII.
"""
from __future__ import annotations

from typing import Callable, Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import I, Integer, Rational, oo, pi, nan, zoo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.elementary.complexes import re, Abs
from sympy.functions.special.bessel import airyai, airyaiprime
from sympy.functions.special.gamma_functions import gamma
from sympy.series.order import Order
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.logic.boolalg import Boolean

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, as_set
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.solve import solve
from sympy_extras.settings import settings

__all__ = ['asymptotic_integral', 'watson_lemma', 'laplace_method', 'stationary_phase',
           'steepest_descent', 'uniform_expansion', 'exponential_part', 'oscillatory_part']


# ---------------------------------------------------------------------------
# Recognising the shapes

def exponential_part(f: Expr, x: Symbol, t: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(phi, h)`` with ``f == phi(x) * exp(t*h(x))``, ``phi`` and ``h``
    free of ``t``; powers ``g(x)**t`` count as ``exp(t*log(g))``. ``None``
    when the dependence on ``t`` is not of this form.

    >>> from sympy import symbols, exp, log
    >>> from sympy_extras.integrals.asymptotic import exponential_part
    >>> x, t = symbols('x t')
    >>> exponential_part(x**t*exp(-t*x)/(1 + x), x, t)
    (1/(x + 1), -x + log(x))
    """
    rewritten = as_expr(f.replace(lambda e: isinstance(e, Pow) and as_expr(e.exp).has(t),
                                  lambda e: exp(as_expr(e.exp) * log(as_expr(e.base)))))
    phi: Expr = S.One
    argument: Expr = S.Zero
    for factor in Mul.make_args(rewritten):
        factor_ = as_expr(factor)
        if isinstance(factor_, exp) and factor_.has(t):
            argument = argument + as_expr(factor_.args[0])
        elif factor_.has(t):
            return None
        else:
            phi = phi * factor_
    if argument == 0:
        return None
    polynomial = argument.as_poly(t)
    if polynomial is None or polynomial.degree() != 1:
        return None
    h = as_expr(polynomial.coeff_monomial(t))
    rest = as_expr(polynomial.coeff_monomial(1))
    if h.has(t) or rest.has(t):
        return None
    return (as_expr(phi * exp(rest)), h)


def oscillatory_part(f: Expr, x: Symbol, t: Symbol) -> Optional[tuple[Expr, Expr, str]]:
    """``(phi, h, kind)`` with ``f == phi(x) * K(t*h(x))`` for ``K`` one of
    ``exp(I*...)`` (``kind='exp'``), ``cos`` or ``sin``; ``None`` otherwise."""
    phi: Expr = S.One
    found: Optional[tuple[Expr, str]] = None
    for factor in Mul.make_args(f):
        factor_ = as_expr(factor)
        if not factor_.has(t):
            phi = phi * factor_
            continue
        if found is not None:
            return None
        if isinstance(factor_, (cos, sin, exp)) and len(factor_.args) == 1:
            argument = as_expr(factor_.args[0])
            if isinstance(factor_, exp):
                argument = as_expr(argument / I)
            polynomial = argument.as_poly(t)
            if polynomial is None or polynomial.degree() != 1 or polynomial.coeff_monomial(1) != 0:
                return None
            h = as_expr(polynomial.coeff_monomial(t))
            if h.has(t) or h.has(I):
                return None
            kind = 'cos' if isinstance(factor_, cos) else 'sin' if isinstance(factor_, sin) else 'exp'
            found = (h, kind)
        else:
            return None
    if found is None:
        return None
    return (phi, found[0], found[1])


# ---------------------------------------------------------------------------
# Watson's lemma

def _puiseux_terms(phi: Expr, x: Symbol, count: int) -> Optional[tuple[list[tuple[Expr, Expr]], Optional[Expr]]]:
    """The first ``count`` terms ``(c_k, a_k)`` of the expansion of ``phi``
    at ``0`` and the exponent of the next one (``None`` when the expansion
    terminates)."""
    if phi.is_polynomial(x):
        polynomial = phi.as_poly(x)
        if polynomial is None:
            return None
        monomials: list[tuple[Expr, Expr]] = [(as_expr(c), Integer(k)) for (k,), c in sorted(polynomial.terms())]
        return (monomials, None)
    n = count + 2
    for _ in range(6):
        expansion = attempt(lambda: as_expr(phi.series(x, 0, n)), settings.timeout)
        if expansion is None:
            return None
        tail = expansion.getO()
        order = tail if isinstance(tail, Order) else None
        body = as_expr(expansion.removeO())
        terms: list[tuple[Expr, Expr]] = []
        for term in Add.make_args(body):
            term_ = as_expr(term)
            coefficient, rest = term_.as_independent(x, as_Add=False)
            rest_ = as_expr(rest)
            if rest_ == 1:
                exponent: Expr = S.Zero
            elif rest_ == x:
                exponent = S.One
            elif isinstance(rest_, Pow) and rest_.base == x:
                exponent = as_expr(rest_.exp)
            else:
                return None
            if as_expr(coefficient) != 0:
                terms.append((as_expr(coefficient), exponent))
        terms.sort(key=lambda item: float(item[1]))
        if order is None:
            return (terms, None)
        next_exponent = as_expr(order.expr.as_independent(x, as_Add=False)[1])
        next_ = S.Zero if next_exponent == 1 else as_expr(next_exponent.exp) if isinstance(next_exponent, Pow) \
            else S.One if next_exponent == x else None
        if next_ is None:
            return None
        if len(terms) > count:
            return (terms[:count], terms[count][1])
        n += count + 2
    return None


def watson_lemma(phi: Expr, x: Symbol, t: Symbol, order: int = 3) -> Optional[Expr]:
    """The expansion of ``Integral(exp(-t*x)*phi(x), (x, 0, oo))`` for
    ``t -> oo`` by Watson's lemma, with ``order`` terms; exact when the
    expansion of ``phi`` terminates.

    >>> from sympy import symbols, sin
    >>> from sympy_extras.integrals.asymptotic import watson_lemma
    >>> x, t = symbols('x t', positive=True)
    >>> watson_lemma(sin(x), x, t, 2)
    -1/t**4 + t**(-2) + O(t**(-6), (t, oo))
    """
    found = _puiseux_terms(phi, x, order)
    if found is None:
        return None
    terms, next_exponent = found
    total: Expr = S.Zero
    for coefficient, exponent in terms:
        if not (as_expr(exponent + 1)).is_positive:
            return None
        total = total + coefficient * gamma(exponent + 1) / t**(exponent + 1)
    if next_exponent is None:
        return total
    return as_expr(total + Order(t**(-next_exponent - 1), (t, oo)))


# ---------------------------------------------------------------------------
# Laplace's method

def _points_in(points: list[Expr], a: Expr, b: Expr, assumptions: Assumptions) -> Optional[list[Expr]]:
    inside: list[Expr] = []
    for p in points:
        if p.is_extended_real is False:
            continue
        below = True if a == -oo else ask(as_boolean(p > a), assumptions)
        above = True if b == oo else ask(as_boolean(p < b), assumptions)
        if below is None or above is None:
            return None
        if below and above:
            inside.append(p)
    return inside


def _stationary_points(h: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[list[Expr]]:
    """The zeros of ``h'`` inside ``(a, b)``."""
    facts: list[Boolean] = []
    if a != -oo:
        facts.append(as_boolean(x > a))
    if b != oo:
        facts.append(as_boolean(x < b))
    if isinstance(assumptions, (Boolean, bool)):
        facts.append(as_boolean(assumptions))
    elif assumptions is not None:
        facts.extend(as_boolean(item) for item in assumptions)
    zeros = attempt(lambda: solve(as_expr(h.diff(x)), x, facts, domain=S.Reals), settings.timeout)
    if zeros is None:
        return None
    if zeros is S.EmptySet:
        return []
    zero_set = as_set(zeros)
    if not isinstance(zero_set, FiniteSet):
        return None
    return _points_in([as_expr(p) for p in zero_set], a, b, assumptions)


def _first_nonzero_derivative(h: Expr, x: Symbol, x0: Expr, start: int,
                              assumptions: Assumptions) -> Optional[tuple[int, int]]:
    """``(n, sign)`` with ``h^(n)(x0)`` the first derivative of order at
    least ``start`` which is not zero, and its sign; ``None`` when a sign
    cannot be decided or the first eight derivatives vanish."""
    for n in range(start, 9):
        value = as_expr(simplify(h.diff(x, n).subs(x, x0)))
        if value == 0:
            continue
        if ask(as_boolean(value < 0), assumptions) is True:
            return (n, -1)
        if ask(as_boolean(value > 0), assumptions) is True:
            return (n, 1)
        return None
    return None


def _power_expansion(phi: Expr, h: Expr, x: Symbol, x0: Expr, t: Symbol, corrections: int, n: int,
                     side: str) -> Optional[tuple[Expr, Expr]]:
    """The series of the Laplace integral at a maximum of ``h`` of order
    ``n`` at ``x0`` (``h(x0 + u) - h(x0) = -c u**n + ...``, ``c > 0``):
    the expansion of ``phi(x0 + u) exp(t (h(x0 + u) - h(x0) + c u**n))``
    in powers of ``u`` integrated against the moments
    ``Integral(u**k exp(-t c u**n), (u, 0, oo)) = Gamma((k + 1)/n)/(n (t c)**((k + 1)/n))``
    over the half line (``side`` ``'right'``: ``u > 0``, ``'left'``:
    ``x = x0 - u``) or, for ``'both'`` with ``n`` even, over the line
    (twice the even moments). The result is the series without the
    factor ``exp(t h(x0))``, in powers of ``t**(-1/n)`` down to
    ``t**(-corrections)`` times the leading one, and the exponent of
    ``t`` of the next term."""
    u = Dummy('u')
    orientation = S.NegativeOne if side == 'left' else S.One
    point = as_expr(x0 + orientation * u)
    c = as_expr(-orientation**n * h.diff(x, n).subs(x, x0) / factorial(n))
    remainder = as_expr(h.subs(x, point) - h.subs(x, x0) + c * u**n)
    integrand = as_expr(phi.subs(x, point) * exp(t * remainder))
    kept_steps = n * corrections         # down to t**(-corrections) below the leading term
    depth = (n + 1) * kept_steps + 1
    expansion = attempt(lambda: as_expr(integrand.series(u, 0, depth).removeO()), settings.timeout)
    if expansion is None:
        return None
    total: Expr = S.Zero
    for term in Add.make_args(as_expr(expansion.expand())):
        coefficient, rest = as_expr(term).as_independent(u, as_Add=False)
        rest_ = as_expr(rest)
        k = 0 if rest_ == 1 else 1 if rest_ == u else int(as_expr(rest_.exp)) if isinstance(rest_, Pow) else -1
        if k < 0:
            return None
        if side == 'both' and k % 2:
            continue
        moment = as_expr(gamma(Rational(k + 1, n)) / (n * (t * c)**Rational(k + 1, n)))
        if side == 'both':
            moment = 2 * moment
        total = total + as_expr(coefficient) * moment
    # keep the powers of t from the leading t**(-1/n) down kept_steps steps of t**(-1/n)
    kept: Expr = S.Zero
    for term in Add.make_args(as_expr(total.expand())):
        coefficient, rest = as_expr(term).as_independent(t, as_Add=False)
        rest_ = as_expr(rest)
        if rest_ == 1:
            power: Optional[Rational] = Rational(0)
        elif rest_ == t:
            power = Rational(1)
        elif isinstance(rest_, Pow) and rest_.base == t and isinstance(rest_.exp, Rational):
            power = rest_.exp
        else:
            power = None
        if power is None:
            return None
        step = -(power * n + 1)          # steps of t**(-1/n) below the leading term
        if 0 <= step <= kept_steps:
            kept = kept + as_expr(term)
    next_step = kept_steps + 2 if side == 'both' else kept_steps + 1
    return (as_expr(simplify(kept)), Rational(-1 - next_step, n))


def _edges_below(h: Expr, x: Symbol, value: Expr, a: Expr, b: Expr, assumptions: Assumptions) -> bool:
    """Whether ``h`` stays below ``value`` at the infinite ends of the
    range (its limits there are smaller)."""
    from sympy_extras.assumptions.limits import limit
    for endpoint in (a, b):
        if endpoint not in (-oo, oo):
            continue
        edge = attempt(lambda: limit(h, x, endpoint, assumptions=assumptions), settings.timeout)
        if edge is None or edge.has(nan, zoo):
            return False
        if edge == -oo:
            continue
        if edge == oo or ask(as_boolean(edge < value), assumptions) is not True:
            return False
    return True


def _maxima(phi: Expr, h: Expr, x: Symbol, a: Expr, b: Expr,
            assumptions: Assumptions) -> Optional[list[tuple[Expr, int, str]]]:
    """The local maxima of ``h`` on the closed range as ``(x0, n, side)``,
    ``n`` the order of the maximum (``h - h(x0) ~ -c (x - x0)**n``) and
    ``side`` ``'both'`` for an interior point, ``'right'`` and ``'left'``
    for the endpoints ``a`` and ``b``; ``None`` when undecided."""
    points = _stationary_points(h, x, a, b, assumptions)
    if points is None:
        return None
    found: list[tuple[Expr, int, str]] = []
    for x0 in points:
        order = _first_nonzero_derivative(h, x, x0, 2, assumptions)
        if order is None:
            return None
        n, sign = order
        if n % 2 == 0 and sign < 0:
            found.append((x0, n, 'both'))
        # a minimum or an inflection: nothing
    for endpoint, side in ((a, 'right'), (b, 'left')):
        if endpoint in (-oo, oo):
            continue
        derivatives = [as_expr(h.diff(x, n).subs(x, endpoint)) for n in (1, 2)]
        if any(d.has(zoo, nan, oo, -oo) for d in derivatives):
            # a singular endpoint (log(x) at 0): no maximum when h falls to -oo there
            from sympy_extras.assumptions.limits import limit
            edge = attempt(lambda: limit(h, x, endpoint, '+' if side == 'right' else '-', assumptions=assumptions),
                           settings.timeout)
            if edge == -oo:
                continue
            return None
        order = _first_nonzero_derivative(h, x, endpoint, 1, assumptions)
        if order is None:
            return None
        n, sign = order
        orientation = 1 if side == 'right' else -1
        if orientation**n * sign < 0:
            # h decreases into the range: a maximum at the endpoint
            found.append((endpoint, n, side))
    return found


def _global_maxima(candidates: list[tuple[Expr, int, str]], h: Expr, x: Symbol,
                   assumptions: Assumptions) -> Optional[list[tuple[Expr, int, str]]]:
    """The candidates where ``h`` is largest: the ones of the greatest
    height (several when they share it); ``None`` when a comparison is
    undecided."""
    if not candidates:
        return None
    best = [candidates[0]]
    height = as_expr(h.subs(x, candidates[0][0]))
    for candidate in candidates[1:]:
        value = as_expr(h.subs(x, candidate[0]))
        difference = as_expr(simplify(value - height))
        if difference == 0:
            best.append(candidate)
        elif ask(as_boolean(difference > 0), assumptions) is True:
            # the earlier maxima are exponentially smaller: dropped
            best, height = [candidate], value
        elif ask(as_boolean(difference < 0), assumptions) is not True:
            return None
    return best


def laplace_method(phi: Expr, h: Expr, x: Symbol, a: Expr, b: Expr, t: Symbol, order: int = 2,
                   assumptions: Assumptions = None) -> Optional[Expr]:
    """The expansion of ``Integral(phi*exp(t*h), (x, a, b))`` for ``t -> oo``
    by Laplace's method, ``order`` terms: the sum of the contributions of
    the global maxima of ``h`` on the closed range, interior maxima of any
    even order (``h - h(x0) ~ -c (x - x0)**(2m)``, the leading term
    ``phi(x0) exp(t h(x0)) Gamma(1/(2m))/(m (t c)**(1/(2m)))``) and
    endpoint maxima of any order (a nonzero slope, or a vanishing slope
    with half of the interior contribution). Maxima of lower height are
    exponentially smaller and dropped. On the real line without a real
    maximum the complex saddle points are tried
    (:func:`steepest_descent`).

    >>> from sympy import symbols, cosh, oo, cos, pi
    >>> from sympy_extras.integrals.asymptotic import laplace_method
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> laplace_method(1, -cosh(x), x, -oo, oo, t, 2)
    sqrt(2)*sqrt(pi)*(t - 1/8)*exp(-t)/t**(3/2) + O(exp(-t)/t**(5/2), (t, oo))
    >>> laplace_method(1, -x**4, x, -oo, oo, t, 1)
    gamma(1/4)/(2*t**(1/4)) + O(t**(-3/4), (t, oo))
    >>> laplace_method(1, cos(x), x, 0, 2*pi, t, 1)
    sqrt(2)*sqrt(pi)*exp(t)/sqrt(t) + O(exp(t)/t, (t, oo))
    """
    phi_, h_ = as_expr(phi), as_expr(h)
    corrections = max(order - 1, 0)
    candidates = _maxima(phi_, h_, x, a, b, assumptions)
    if candidates is None:
        return None
    if not candidates:
        if a == -oo and b == oo and not _stationary_points(h_, x, a, b, assumptions):
            return steepest_descent(phi_, h_, x, t, assumptions)
        return None
    best = _global_maxima(candidates, h_, x, assumptions)
    if best is None:
        return None
    height = as_expr(h_.subs(x, best[0][0]))
    if not _edges_below(h_, x, height, a, b, assumptions):
        return None
    total: Expr = S.Zero
    scale: Optional[Expr] = None
    for x0, n, side in best:
        found = _power_expansion(phi_, h_, x, x0, t, corrections, n, side)
        if found is None:
            return None
        series, next_power = found
        if series == 0:
            return None
        total = total + exp(t * height) * series
        candidate_scale = as_expr(exp(t * height) * t**next_power)
        if scale is None or as_expr(next_power - _power_of(scale, t)).is_positive:
            scale = candidate_scale
    if scale is None:
        return None
    return as_expr(as_expr(total) + Order(scale, (t, oo)))


def _power_of(e: Expr, t: Symbol) -> Expr:
    """The exponent of ``t`` in ``e``, a product with one power of ``t``."""
    for factor in Mul.make_args(e):
        factor_ = as_expr(factor)
        if factor_ == t:
            return S.One
        if isinstance(factor_, Pow) and factor_.base == t:
            return as_expr(factor_.exp)
    return S.Zero


# ---------------------------------------------------------------------------
# Saddle points in the complex plane

def _saddle_points(h: Expr, x: Symbol) -> Optional[list[Expr]]:
    """The complex zeros of ``h'`` (the saddle points of ``exp(t h)``),
    ``None`` when they cannot be found or are not isolated."""
    from sympy.solvers.solvers import solve as sympy_solve
    derivative = as_expr(h.diff(x))
    found = attempt(lambda: sympy_solve(derivative, x), settings.timeout)
    if not isinstance(found, list) or not found:
        return None
    points: list[Expr] = []
    for item in found:
        if not isinstance(item, Expr) or item.has(oo, -oo, zoo, nan):
            return None
        points.append(item)
    return points


def _leading_saddle_term(phi: Expr, h: Expr, x: Symbol, z0: Expr, t: Symbol) -> Optional[Expr]:
    """``phi(z0) exp(t h(z0)) sqrt(2 pi/(-t h''(z0)))``, the leading term of
    the saddle at ``z0`` on the path of steepest descent traversed in the
    direction of increasing real part (the principal branch of the root
    gives that direction, [Bleistein]_ 7.2); ``None`` for a degenerate
    saddle or one whose descent path is vertical."""
    second = as_expr(simplify(h.diff(x, 2).subs(x, z0)))
    if second == 0 or second.is_extended_real and second.is_positive:
        return None
    height = as_expr(h.subs(x, z0).expand())
    return as_expr(as_expr(phi.subs(x, z0)).expand() * exp(t * height) * sqrt(2 * pi / (-t * second)))


def _saddle_check(phi: Expr, h: Expr, x: Symbol, t: Symbol, terms: list[Expr], level: float) -> bool:
    """Whether the sum of the leading ``terms`` agrees with the quadrature
    of ``Integral(phi exp(t h), (x, -oo, oo))`` at two large values of
    ``t``: the error relative to the envelope (the sum of the moduli of
    the terms, since the sum itself oscillates through zero) small and
    decreasing. The integral is ``exp(t level)`` times smaller than the
    integrand, which cancels: the quadrature runs with the digits that
    cancellation costs."""
    import mpmath
    from sympy.utilities.lambdify import lambdify
    errors: list[float] = []
    for value in (30, 120):
        digits = 25 + int(value * abs(level) / 2.3)
        g: Callable[[mpmath.mpf], mpmath.mpc] = lambdify(x, as_expr((phi * exp(t * h)).subs(t, value)), 'mpmath')
        try:
            with mpmath.workdps(digits):
                points = [mpmath.mpf(p) / 10 for p in range(-30, 31)]
                exact = mpmath.quad(g, [mpmath.mpf('-inf')] + points + [mpmath.mpf('inf')])
                exact_ = complex(exact)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, NameError, AttributeError,
                NotImplementedError, mpmath.libmp.NoConvergence):
            return False
        values = [complex(as_expr(term.subs(t, value)).evalf(digits)) for term in terms]
        envelope = sum(abs(v) for v in values)
        if envelope == 0:
            return False
        errors.append(abs(sum(values) - exact_) / envelope)
    return errors[1] < 0.02 and errors[1] <= errors[0]


def steepest_descent(phi: Expr, h: Expr, x: Symbol, t: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """The leading term of ``Integral(phi*exp(t*h), (x, -oo, oo))`` for
    ``t -> oo`` from the saddle points of ``h`` in the complex plane
    ([Bleistein]_ 7.2, [Olver]_ 4.7): the real line is deformed onto the
    paths of steepest descent through the saddles ``z0`` (``h'(z0) = 0``)
    and each contributes ``phi(z0) exp(t h(z0)) sqrt(2 pi/(-t h''(z0)))``.

    The deformation is justified without a check for a quadratic ``h``
    whose leading coefficient has a negative real part (``phi`` a
    polynomial): a single saddle and a Gaussian decay along every
    horizontal line. Otherwise the saddles are grouped by the real part
    of ``h(z0)``, and, level by level from the top, the sum over a level
    and then the single saddles are kept when the quadrature confirms
    them at two large values of ``t`` (a saddle of larger ``Re h`` may
    lie off the deformed contour, as the lower one of ``-x**4 + I x``
    does); without the numerical checks of the settings only the
    quadratic case is answered.

    >>> from sympy import symbols, I
    >>> from sympy_extras.integrals.asymptotic import steepest_descent
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> steepest_descent(1, -x**2 + I*x, x, t)
    sqrt(pi)*exp(-t/4)/sqrt(t) + O(exp(-t/4)/t**(3/2), (t, oo))
    """
    phi_, h_ = as_expr(phi), as_expr(h)
    saddles = _saddle_points(h_, x)
    if saddles is None:
        return None
    terms: list[tuple[Expr, Expr]] = []
    for z0 in saddles:
        term = _leading_saddle_term(phi_, h_, x, z0, t)
        if term is None:
            if as_expr(simplify(h_.diff(x, 2).subs(x, z0))) == 0:
                # a degenerate saddle (cubic or worse): an Airy-type
                # contribution, not a Gaussian one
                return None
            # a vertical descent path: not on a contour from -oo to oo
            continue
        terms.append((z0, term))
    if not terms:
        return None
    polynomial = h_.as_poly(x)
    if polynomial is not None and polynomial.degree() == 2 and phi_.is_polynomial(x):
        leading = as_expr(polynomial.LC())
        if ask(as_boolean(re(leading) < 0), assumptions) is True and len(terms) == 1:
            return _with_order(terms[0][1], _saddle_scale(h_, x, terms[0][0], t), t)
        return None
    if not settings.numerical_checks or any(as_expr(z0).free_symbols for z0, _ in terms):
        return None
    levels: dict[float, list[Expr]] = {}
    for z0, term in terms:
        height = as_expr(re(h_.subs(x, z0)).evalf(15))
        if not height.is_number:
            return None
        key = round(float(height), 8)
        levels.setdefault(key, []).append(term)
    for key in sorted(levels, reverse=True):
        group = levels[key]
        candidates = [group] + ([[term] for term in group] if len(group) > 1 else [])
        for candidate in candidates:
            if _saddle_check(phi_, h_, x, t, candidate, key):
                scale = _saddle_scale(h_, x, [z0 for z0, term in terms if term in candidate][0], t)
                return _with_order(as_expr(Add(*candidate)), scale, t)
    return None


def _saddle_scale(h: Expr, x: Symbol, z0: Expr, t: Symbol) -> Expr:
    """The size of the next term of the saddle at ``z0``:
    ``exp(t Re h(z0)) t**(-3/2)``."""
    height = as_expr(simplify(re(h.subs(x, z0))))
    return as_expr(exp(t * height) * t**Rational(-3, 2))


def _with_order(leading: Expr, scale: Expr, t: Symbol) -> Expr:
    """``leading + O(scale)``; the sum is left unevaluated when SymPy
    cannot compare the terms with the order (an ``exp(I t c)`` or an
    ``erfc`` of a complex argument stops ``gruntz``)."""
    order = Order(scale, (t, oo))
    try:
        return as_expr(leading + order)
    except (NotImplementedError, ValueError, TypeError):
        return as_expr(Add(leading, order, evaluate=False))


# ---------------------------------------------------------------------------
# Stationary phase

def stationary_phase(phi: Expr, h: Expr, x: Symbol, a: Expr, b: Expr, t: Symbol, kind: str = 'exp',
                     assumptions: Assumptions = None) -> Optional[Expr]:
    """The leading term of ``Integral(phi*exp(I*t*h), (x, a, b))`` (or with
    ``cos(t*h)``, ``sin(t*h)`` for ``kind='cos'``, ``'sin'``) for
    ``t -> oo``: the sum over the interior stationary points of
    ``phi(x0) sqrt(2 pi/(t |h''(x0)|)) exp(I t h(x0) +/- I pi/4)``.

    >>> from sympy import symbols, pi
    >>> from sympy_extras.integrals.asymptotic import stationary_phase
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> stationary_phase(1, x**2, x, -1, 1, t)
    sqrt(pi)*exp(I*pi/4)/sqrt(t) + O(t**(-3/2), (t, oo))
    """
    phi_, h_ = as_expr(phi), as_expr(h)
    points = _stationary_points(h_, x, a, b, assumptions)
    if not points:
        return None
    total: Expr = S.Zero
    for x0 in points:
        second = as_expr(h_.diff(x, 2).subs(x, x0))
        positive = ask(as_boolean(second > 0), assumptions)
        negative = ask(as_boolean(second < 0), assumptions)
        if positive is not True and negative is not True:
            return None
        sign = S.One if positive else S.NegativeOne
        phase = as_expr(t * h_.subs(x, x0) + sign * pi / 4)
        amplitude = as_expr(phi_.subs(x, x0) * sqrt(2 * pi / (t * sign * second)))
        if kind == 'exp':
            total = total + amplitude * exp(I * phase)
        elif kind == 'cos':
            total = total + amplitude * cos(phase)
        else:
            total = total + amplitude * sin(phase)
    return as_expr(total + Order(t**(-Rational(3, 2)), (t, oo)))


# ---------------------------------------------------------------------------
# Uniform expansions: coalescing stationary points

def _real_stationary_points(h: Expr, x: Symbol, assumptions: Assumptions) -> Optional[list[Expr]]:
    """The real zeros of ``h'`` on the whole line, sorted when their order
    is decided."""
    points = _stationary_points(h, x, -oo, oo, assumptions)
    if points is None:
        return None
    ordered = list(points)
    for i in range(len(ordered)):
        for j in range(len(ordered) - 1 - i):
            if ask(as_boolean(ordered[j] > ordered[j + 1]), assumptions) is True:
                ordered[j], ordered[j + 1] = ordered[j + 1], ordered[j]
            elif ask(as_boolean(ordered[j] < ordered[j + 1]), assumptions) is not True:
                return None
    return ordered


def _in_range(p: Expr, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[bool]:
    below = True if a == -oo else ask(as_boolean(p > a), assumptions)
    above = True if b == oo else ask(as_boolean(p < b), assumptions)
    if below is None or above is None:
        return None
    return bool(below and above)


def _oscillatory_form(value: Expr, kind: str) -> Expr:
    """``value`` (the expansion for ``exp(I t h)``) as the one for ``cos(t h)``
    or ``sin(t h)``: the real or imaginary part, ``phi`` real."""
    if kind == 'exp':
        return value
    from sympy.functions.elementary.complexes import im
    return as_expr(simplify(re(value) if kind == 'cos' else im(value)))


def uniform_expansion(phi: Expr, h: Expr, x: Symbol, a: Expr, b: Expr, t: Symbol, kind: str = 'exp',
                      assumptions: Assumptions = None) -> Optional[Expr]:
    """The leading term of ``Integral(phi*exp(I*t*h), (x, a, b))`` (or with
    ``cos(t*h)``, ``sin(t*h)`` for ``kind='cos'``, ``'sin'``, ``phi``
    real) for ``t -> oo``, uniform in the parameters of ``h`` as its
    stationary points coalesce ([Bleistein]_ 9.2-9.4, [DLMF]_ 2.4(v),
    Chester, Friedman and Ursell 1957).

    Two stationary points ``x1 < x2`` in the range: the cubic
    transformation ``h(x) = s (zeta**3/3 - alpha zeta) + rho`` with
    ``rho = (h(x1) + h(x2))/2``, ``(4/3) alpha**(3/2) = |h(x1) - h(x2)|``
    and ``s`` the sign of ``h(x1) - h(x2)``, maps them onto ``-+sqrt(alpha)``;
    with ``G(zeta) = phi(x(zeta)) x'(zeta)``, ``G(-+sqrt(alpha)) =
    phi(x_j) sqrt(2 sqrt(alpha)/|h''(x_j)|)``, ``p0 = (G(+) + G(-))/2``
    and ``q0 = (G(+) - G(-))/(2 sqrt(alpha))``, the Airy form

    .. math::

        2\\pi e^{i t \\rho} \\left[ p_0 t^{-1/3} \\mathrm{Ai}(-\\alpha t^{2/3})
        - i s q_0 t^{-2/3} \\mathrm{Ai}'(-\\alpha t^{2/3}) \\right] + O(t^{-4/3}),

    exact for the Airy integral ``h = x**3/3 - a x``, ``phi = 1``.

    One stationary point ``x0`` and the endpoint ``a`` (the other
    endpoint far from ``x0`` or infinite; with ``b`` the roles are
    mirrored): the quadratic transformation ``h(x) = h(x0) + sigma
    zeta**2/2``, ``sigma`` the sign of ``h''(x0)``, the endpoint at
    ``zeta_a = -sqrt(2 |h(a) - h(x0)|)`` when ``a < x0`` and ``+`` when
    the stationary point lies outside; with ``p0 = phi(x0)/sqrt(|h''(x0)|)``
    and ``q0 = (phi(a) sigma zeta_a/h'(a) - p0)/zeta_a`` the error function
    form

    .. math::

        e^{i t h(x_0)} p_0 \\sqrt{\\frac{\\pi}{2t}}\\, e^{i\\sigma\\pi/4}
        \\operatorname{erfc}\\!\\left(e^{-i\\sigma\\pi/4} \\sqrt{t/2}\\, \\zeta_a\\right)
        + \\frac{i \\sigma q_0}{t} e^{i t h(a)} + O(t^{-3/2}),

    which is the stationary phase term when ``zeta_a -> -oo`` and half of
    it when the point reaches the endpoint.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.asymptotic import uniform_expansion
    >>> x = symbols('x')
    >>> a, t = symbols('a t', positive=True)
    >>> uniform_expansion(1, x**3/3 - a*x, x, -oo, oo, t)
    2*pi*airyai(-a*t**(2/3))/t**(1/3) + O(t**(-4/3), (t, oo))
    """
    phi_, h_ = as_expr(phi), as_expr(h)
    points = _real_stationary_points(h_, x, assumptions)
    if points is None:
        return None
    inside = [p for p in points if _in_range(p, a, b, assumptions) is True]
    if any(_in_range(p, a, b, assumptions) is None for p in points):
        return None
    if len(inside) == 2:
        return _airy_form(phi_, h_, x, inside[0], inside[1], t, kind, assumptions)
    if len(points) == 1 and len(inside) <= 1:
        x0 = points[0]
        if a != -oo and (b == oo or ask(as_boolean(x0 - a < b - x0), assumptions) is True):
            return _error_function_form(phi_, h_, x, x0, a, t, 1, kind, assumptions)
        if b != oo:
            return _error_function_form(phi_, h_, x, x0, b, t, -1, kind, assumptions)
    return None


def _airy_form(phi: Expr, h: Expr, x: Symbol, x1: Expr, x2: Expr, t: Symbol, kind: str,
               assumptions: Assumptions) -> Optional[Expr]:
    h1, h2 = as_expr(h.subs(x, x1)), as_expr(h.subs(x, x2))
    difference = as_expr(simplify(h1 - h2))
    if ask(as_boolean(difference > 0), assumptions) is True:
        s: Expr = S.One
    elif ask(as_boolean(difference < 0), assumptions) is True:
        s = S.NegativeOne
    else:
        return None
    rho = as_expr((h1 + h2) / 2)
    alpha = as_expr(simplify((Rational(3, 4) * s * difference)**Rational(2, 3)))
    values: list[Expr] = []
    for point in (x1, x2):
        second = as_expr(simplify(h.diff(x, 2).subs(x, point)))
        if second == 0:
            return None
        values.append(as_expr(phi.subs(x, point) * sqrt(2 * sqrt(alpha) / Abs(second))))
    p0 = as_expr(simplify((values[1] + values[0]) / 2))
    q0 = as_expr(simplify((values[1] - values[0]) / (2 * sqrt(alpha))))
    argument = as_expr(-alpha * t**Rational(2, 3))
    value = as_expr(2 * pi * exp(I * t * rho) * (p0 * t**Rational(-1, 3) * airyai(argument)
                                                - I * s * q0 * t**Rational(-2, 3) * airyaiprime(argument)))
    return _with_order(_oscillatory_form(value, kind), t**Rational(-4, 3), t)


def _error_function_form(phi: Expr, h: Expr, x: Symbol, x0: Expr, endpoint: Expr, t: Symbol, orientation: int,
                         kind: str, assumptions: Assumptions) -> Optional[Expr]:
    from sympy.functions.special.error_functions import erfc
    second = as_expr(simplify(h.diff(x, 2).subs(x, x0)))
    if ask(as_boolean(second > 0), assumptions) is True:
        sigma: Expr = S.One
    elif ask(as_boolean(second < 0), assumptions) is True:
        sigma = S.NegativeOne
    else:
        return None
    slope = as_expr(h.diff(x).subs(x, endpoint))
    if slope == 0:
        return None
    # x = endpoint + orientation*u runs into the range: zeta increases with u
    gap = as_expr(simplify(sigma * (h.subs(x, endpoint) - h.subs(x, x0))))
    if ask(as_boolean(gap >= 0), assumptions) is not True:
        return None
    side = ask(as_boolean(orientation * (x0 - endpoint) > 0), assumptions)
    if side is None:
        return None
    zeta = as_expr((S.NegativeOne if side else S.One) * sqrt(2 * gap))
    p0 = as_expr(phi.subs(x, x0) / sqrt(Abs(second)))
    endpoint_value = as_expr(phi.subs(x, endpoint) * sigma * zeta / (orientation * slope))
    q0 = as_expr(simplify((endpoint_value - p0) / zeta))
    phase = exp(I * t * h.subs(x, x0))
    value = as_expr(phase * p0 * sqrt(pi / (2 * t)) * exp(I * sigma * pi / 4)
                    * erfc(exp(-I * sigma * pi / 4) * sqrt(t / 2) * zeta)
                    + I * sigma * q0 * exp(I * t * h.subs(x, endpoint)) / t)
    return _with_order(_oscillatory_form(value, kind), t**Rational(-3, 2), t)


# ---------------------------------------------------------------------------
# The entry point

def asymptotic_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, parameter: Symbol,
                        order: int = 3, assumptions: Assumptions = None, uniform: bool = False) -> Optional[Expr]:
    """The first ``order`` terms of the expansion of ``Integral(f, (x, a, b))``
    as ``parameter -> oo``, with an ``Order`` term, or ``None`` when no
    method applies: Watson's lemma for ``exp(-c*t*x)`` over ``(0, b)``,
    Laplace's method for ``exp(t*h(x))`` (the saddle points in the
    complex plane on the real line without a real maximum), the leading
    term of the stationary phase for ``exp(I*t*h)``, ``cos(t*h)`` and
    ``sin(t*h)``, or with ``uniform=True`` the expansion uniform in the
    parameters of ``h`` (:func:`uniform_expansion`).

    Examples
    ========

    >>> from sympy import symbols, exp, oo, sqrt, I
    >>> from sympy_extras.integrals.asymptotic import asymptotic_integral
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> asymptotic_integral(exp(-t*x)*sqrt(1 + x), x, 0, oo, t, order=2)
    1/(2*t**2) + 1/t + O(t**(-3), (t, oo))
    >>> asymptotic_integral(exp(-t*(x**2 - 1)**2), x, -oo, oo, t, order=1)
    sqrt(pi)/sqrt(t) + O(t**(-3/2), (t, oo))
    >>> asymptotic_integral(exp(-t*x**2 + I*t*x), x, -oo, oo, t)
    sqrt(pi)*exp(-t/4)/sqrt(t) + O(exp(-t/4)/t**(3/2), (t, oo))
    """
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    t = parameter
    if order < 1:
        raise ValueError("order must be at least 1")
    oscillatory = oscillatory_part(f_, x, t)
    if oscillatory is not None:
        phi, h, kind = oscillatory
        if uniform:
            return uniform_expansion(phi, h, x, a_, b_, t, kind, assumptions)
        return stationary_phase(phi, h, x, a_, b_, t, kind, assumptions)
    found = exponential_part(f_, x, t)
    if found is None:
        return None
    phi, h = found
    # Watson's lemma: h = -c x on (0, b)
    slope = as_expr(-h / x)
    if a_ == 0 and not slope.has(x) and slope.is_positive:
        scaled = watson_lemma(phi, x, t, order)
        if scaled is not None:
            return as_expr(scaled.subs(t, slope * t)) if slope != 1 else scaled
    return laplace_method(phi, h, x, a_, b_, t, order, assumptions)

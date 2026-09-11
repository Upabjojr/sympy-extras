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
   Integrals*, Dover, 1986, chapters 3–6.
.. [Olver] F. W. J. Olver, *Asymptotics and Special Functions*, Academic
   Press, 1974, chapter 3.
.. [DLMF] NIST Digital Library of Mathematical Functions, section 2.3,
   https://dlmf.nist.gov/2.3; A. Erdélyi, *Asymptotic Expansions*, Dover,
   1956, chapter 2.
"""
from __future__ import annotations

from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import I, Integer, Rational, oo, pi, nan, zoo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import factorial, factorial2
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
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
           'exponential_part', 'oscillatory_part']


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


def _gaussian_expansion(phi: Expr, h: Expr, x: Symbol, x0: Expr, t: Symbol, corrections: int) -> Optional[Expr]:
    """The series of the Laplace integral at an interior maximum: the
    leading factor times ``1 + c_1/t + ... + c_N/t**N``."""
    u = Dummy('u')
    alpha = as_expr(-h.diff(x, 2).subs(x, x0))
    remainder = as_expr(h.subs(x, x0 + u) - h.subs(x, x0) + alpha * u**2 / 2)
    integrand = as_expr(phi.subs(x, x0 + u) * exp(t * remainder))
    depth = 6 * corrections + 1
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
        if k % 2:
            continue
        total = total + as_expr(coefficient) * factorial2(k - 1) / (t * alpha)**(k // 2)
    # keep the powers of t down to t**(-corrections)
    kept: Expr = S.Zero
    for term in Add.make_args(as_expr(total.expand())):
        coefficient, rest = as_expr(term).as_independent(t, as_Add=False)
        rest_ = as_expr(rest)
        power = 0 if rest_ == 1 else 1 if rest_ == t else int(as_expr(rest_.exp)) if isinstance(rest_, Pow) \
            and isinstance(rest_.exp, Integer) else None
        if power is None:
            return None
        if -corrections <= power <= 0:
            kept = kept + as_expr(term)
    return as_expr(simplify(kept / phi.subs(x, x0))) if phi.subs(x, x0) != 0 else None


def _endpoint_expansion(phi: Expr, h: Expr, x: Symbol, a: Expr, t: Symbol, sign: int,
                        corrections: int) -> Optional[Expr]:
    """The series of the Laplace integral at an endpoint maximum, ``sign``
    ``+1`` for the left endpoint (``h`` decreasing) and ``-1`` for the
    right one: the terms ``k! (t |h'|)^(-k-1)`` times the coefficients."""
    u = Dummy('u')
    slope = as_expr(h.diff(x).subs(x, a))
    rate = as_expr(-sign * slope)         # positive
    remainder = as_expr(h.subs(x, a + sign * u) - h.subs(x, a) - slope * sign * u)
    integrand = as_expr(phi.subs(x, a + sign * u) * exp(t * remainder))
    depth = 2 * corrections + 1
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
        total = total + as_expr(coefficient) * factorial(k) / (t * rate)**(k + 1)
    kept: Expr = S.Zero
    for term in Add.make_args(as_expr(total.expand())):
        coefficient, rest = as_expr(term).as_independent(t, as_Add=False)
        rest_ = as_expr(rest)
        power = -1 if rest_ == 1 / t else int(as_expr(rest_.exp)) if isinstance(rest_, Pow) \
            and isinstance(rest_.exp, Integer) else None
        if power is None:
            return None
        if -1 - corrections <= power <= -1:
            kept = kept + as_expr(term)
    return as_expr(simplify(kept))


def laplace_method(phi: Expr, h: Expr, x: Symbol, a: Expr, b: Expr, t: Symbol, order: int = 2,
                   assumptions: Assumptions = None) -> Optional[Expr]:
    """The expansion of ``Integral(phi*exp(t*h), (x, a, b))`` for ``t -> oo``
    by Laplace's method, ``order`` terms, at the single interior maximum
    of ``h`` or at an endpoint where ``h`` is largest and monotone.

    >>> from sympy import symbols, cosh, oo
    >>> from sympy_extras.integrals.asymptotic import laplace_method
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> laplace_method(1, -cosh(x), x, -oo, oo, t, 2)
    sqrt(2)*sqrt(pi)*(t - 1/8)*exp(-t)/t**(3/2) + O(exp(-t)/t**(5/2), (t, oo))
    """
    phi_, h_ = as_expr(phi), as_expr(h)
    points = _stationary_points(h_, x, a, b, assumptions)
    if points is None:
        return None
    corrections = max(order - 1, 0)
    if len(points) == 1:
        x0 = points[0]
        second = as_expr(h_.diff(x, 2).subs(x, x0))
        if ask(as_boolean(second < 0), assumptions) is not True:
            return None
        if not _global_maximum(h_, x, x0, a, b, assumptions):
            return None
        factor = as_expr(simplify(_gaussian_expansion(phi_, h_, x, x0, t, corrections) or S.Zero))
        if factor == 0:
            return None
        leading = as_expr(phi_.subs(x, x0) * exp(t * h_.subs(x, x0)) * sqrt(2 * pi / (t * (-second))))
        scale = as_expr(exp(t * h_.subs(x, x0)) * t**(-Rational(1, 2) - corrections - 1))
        return as_expr(leading * factor + Order(scale, (t, oo)))
    if points:
        return None
    # no stationary point: the maximum is at an endpoint where h is monotone
    for endpoint, sign in ((a, 1), (b, -1)):
        if endpoint in (-oo, oo):
            continue
        slope = as_expr(h_.diff(x).subs(x, endpoint))
        if slope.has(zoo, nan, oo, -oo) or ask(as_boolean(sign * slope < 0), assumptions) is not True:
            continue
        other = b if sign == 1 else a
        if other not in (-oo, oo) and ask(as_boolean(h_.subs(x, other) < h_.subs(x, endpoint)), assumptions) is not True:
            continue
        series = _endpoint_expansion(phi_, h_, x, endpoint, t, sign, corrections)
        if series is None:
            return None
        value = as_expr(exp(t * h_.subs(x, endpoint)) * series)
        scale = as_expr(exp(t * h_.subs(x, endpoint)) * t**(-corrections - 2))
        return as_expr(value + Order(scale, (t, oo)))
    return None


def _global_maximum(h: Expr, x: Symbol, x0: Expr, a: Expr, b: Expr, assumptions: Assumptions) -> bool:
    """Whether the interior critical point is the maximum on the range:
    ``h`` is larger there than at the endpoints (limits at infinity)."""
    from sympy_extras.assumptions.limits import limit
    value = as_expr(h.subs(x, x0))
    for endpoint, direction in ((a, '+'), (b, '-')):
        if endpoint in (-oo, oo):
            edge = attempt(lambda: limit(h, x, endpoint, assumptions=assumptions), settings.timeout)
        else:
            edge = attempt(lambda: limit(h, x, endpoint, direction, assumptions=assumptions), settings.timeout)
        if edge is None or edge.has(nan, zoo):
            return False
        if edge == -oo:
            continue
        if edge == oo or ask(as_boolean(edge < value), assumptions) is not True:
            return False
    return True


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
# The entry point

def asymptotic_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, parameter: Symbol,
                        order: int = 3, assumptions: Assumptions = None) -> Optional[Expr]:
    """The first ``order`` terms of the expansion of ``Integral(f, (x, a, b))``
    as ``parameter -> oo``, with an ``Order`` term, or ``None`` when no
    method applies: Watson's lemma for ``exp(-c*t*x)`` over ``(0, b)``,
    Laplace's method for ``exp(t*h(x))``, the leading term of the
    stationary phase for ``exp(I*t*h)``, ``cos(t*h)`` and ``sin(t*h)``.

    Examples
    ========

    >>> from sympy import symbols, exp, oo, sqrt
    >>> from sympy_extras.integrals.asymptotic import asymptotic_integral
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> asymptotic_integral(exp(-t*x)*sqrt(1 + x), x, 0, oo, t, order=2)
    1/(2*t**2) + 1/t + O(t**(-3), (t, oo))
    """
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    t = parameter
    if order < 1:
        raise ValueError("order must be at least 1")
    oscillatory = oscillatory_part(f_, x, t)
    if oscillatory is not None:
        phi, h, kind = oscillatory
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

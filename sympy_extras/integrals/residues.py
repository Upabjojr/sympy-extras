"""Definite integrals by the residue theorem.

The residue theorem, `\\oint_\\gamma f(z)\\, dz = 2\\pi i \\sum \\operatorname{Res} f`
over the poles inside a closed contour `\\gamma` [Ahlfors]_, chapter 4.5,
evaluates the classical families of definite integrals of rational
functions; Mathematica's ``Integrate`` uses it for the same families
[Wolfram]_. Four of them are implemented here, each with the conditions
under which the contour integral equals the real integral:

1. **Rational functions over the real line**, `\\int_{-\\infty}^\\infty R(x)\\, dx`
   with `R = P/Q`, `\\deg Q \\ge \\deg P + 2` and no real poles: the
   contour is the real segment closed by a large semicircle in the upper
   half plane, whose contribution vanishes, so the integral is `2\\pi i`
   times the sum of the residues of `R` in the upper half plane
   ([Ahlfors]_, 4.5.3, example 1).

2. **Fourier integrals** `\\int_{-\\infty}^\\infty R(x) e^{ikx}\\, dx` with
   `\\deg Q \\ge \\deg P + 1` and `k > 0`: Jordan's lemma makes the
   semicircle vanish again, the value is `2\\pi i` times the residues of
   `R(z) e^{ikz}` in the upper half plane, and `k < 0` uses the lower one
   with the opposite sign. The integrals with `\\cos kx` and `\\sin kx` are
   the half sum and half difference of the two, so they stay exact for
   parametric `R`. The same integrals over `(0, \\infty)` are computed when
   the integrand is even.

3. **Powers of** `x` **times rational functions over** `(0, \\infty)`,
   `\\int_0^\\infty x^\\alpha R(x)\\, dx` with `\\alpha` not an integer, the
   integral converging at both ends and `R` without poles on the positive
   real axis: the keyhole contour around the branch cut of `z^\\alpha`
   along the positive axis gives

   .. math::

       \\int_0^\\infty x^\\alpha R(x)\\, dx =
       \\frac{2\\pi i}{1 - e^{2\\pi i \\alpha}} \\sum_{z_k \\ne 0}
       \\operatorname{Res}_{z = z_k} z^\\alpha R(z),
       \\qquad 0 < \\arg z < 2\\pi,

   ([Ahlfors]_, 4.5.3, example 3; [Marsden]_, section 4.3). For an
   integer `\\alpha` the integrand is rational and the same contour with
   `R(z) \\log z` gives `\\int_0^\\infty R(x)\\, dx = -\\sum \\operatorname{Res} R(z) \\log z`
   with the branch `0 < \\arg z < 2\\pi`. A factor `\\log^n x` is the `n`-th
   derivative of the power formula with respect to `\\alpha`.

4. **Rational functions of** `\\sin x` **and** `\\cos x` **over a period**:
   with `z = e^{ix}`, `\\sin x = (z - 1/z)/2i`, `\\cos x = (z + 1/z)/2` and
   `dx = dz/iz`, the integral over `(a, a + 2\\pi)` is the integral over
   the unit circle of a rational function of `z`, that is `2\\pi i` times
   the residues at the poles inside the circle ([Ahlfors]_, 4.5.3,
   example 2); over `(a, a + 2\\pi k)` it is `k` times that.

The poles are the roots of the denominators (:func:`sympy.roots`, or
:class:`~sympy.polys.rootoftools.ComplexRootOf` for numerical
coefficients), and their position -- in the upper half plane, inside the
unit circle, off the positive real axis -- is decided exactly for numbers
and by :func:`sympy_extras.assumptions.ask` under the assumptions for
parameters; a pole whose position cannot be decided makes the function
give up, except that, with the numerical checks of the settings on, the
position may be read at a sample of the parameters satisfying the
assumptions and the value is then verified numerically. The residues at
simple poles are `P(z_k)/Q'(z_k)`, at multiple poles the derivative
formula, with the branch of the logarithm kept as an explicit function
so that the derivatives are branch-independent.

Examples
========

>>> from sympy import symbols, cos, oo, pi, sqrt
>>> from sympy_extras.integrals.residues import residue_integral
>>> x = symbols('x')
>>> a, b = symbols('a b', positive=True)
>>> residue_integral(1/(x**4 + 1), x, -oo, oo)
ConditionalValue(sqrt(2)*pi/2)
>>> residue_integral(cos(b*x)/(x**2 + a**2), x, -oo, oo)
ConditionalValue(pi*exp(-a*b)/a)
>>> residue_integral(1/(2 + cos(x)), x, 0, 2*pi)
ConditionalValue(2*sqrt(3)*pi/3)
>>> residue_integral(sqrt(x)/(x**2 + 1), x, 0, oo)
ConditionalValue(sqrt(2)*pi/2)

References
==========

.. [Ahlfors] L. V. Ahlfors, *Complex analysis*, 3rd edition, McGraw-Hill,
   1979, chapter 4.5 (The calculus of residues).
.. [Marsden] J. E. Marsden, M. J. Hoffman, *Basic complex analysis*, 3rd
   edition, W. H. Freeman, 1999, section 4.3 (Evaluation of definite
   integrals).
.. [Wolfram] Wolfram Research, *Some notes on internal implementation*,
   https://reference.wolfram.com/language/tutorial/SomeNotesOnInternalImplementation.html
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function, expand, expand_trig
from sympy.core.mul import Mul
from sympy.core.numbers import I, Integer, Rational, nan, oo, pi, zoo
from sympy.core.relational import Ne
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import Abs, arg, im, re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import TrigonometricFunction, cos, sin
from sympy.logic.boolalg import And, Boolean, true, false
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, cancel, factor_list
from sympy.polys.rootoftools import ComplexRootOf
from sympy.series.limits import limit
from sympy.simplify.simplify import simplify

from sympy_extras._timeout import TimeLimitExceeded, attempt, time_limit
from sympy_extras._typing import as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import Facts, element
from sympy_extras.assumptions.sat import find_instance
from sympy_extras.settings import settings
from .conditions import ConditionalValue

__all__ = ['residue_integral', 'rational_function', 'poles', 'real_line_integral',
           'fourier_integral', 'keyhole_integral', 'period_integral']

#: the branch of the logarithm, kept as an undefined function while
#: differentiating, with ``L'(z) = 1/z``
_L = Function('L')


class _Fraction:
    """A rational function ``numerator/denominator`` of ``x``, as
    polynomials with coefficients free of ``x``."""

    def __init__(self, numerator: Poly, denominator: Poly, x: Symbol) -> None:
        self.numerator = numerator
        self.denominator = denominator
        self.x = x

    def as_expr(self) -> Expr:
        return as_expr(self.numerator.as_expr() / self.denominator.as_expr())

    @property
    def excess(self) -> int:
        """``deg Q - deg P``."""
        return self.denominator.degree() - self.numerator.degree()


def rational_function(f: Expr, x: Symbol) -> Optional[_Fraction]:
    """``f`` as a quotient of polynomials in ``x`` with coefficients free of
    ``x`` and the common factors cancelled, or ``None``."""
    if not f.has(x):
        return None
    if not f.is_rational_function(x):
        return None
    reduced = attempt(lambda: as_expr(cancel(f, x)), settings.timeout)
    if reduced is None:
        return None
    numerator, denominator = reduced.as_numer_denom()
    try:
        p = Poly(numerator, x)
        q = Poly(denominator, x)
    except PolynomialError:
        return None
    if p.has(x) and any(as_expr(c).has(x) for c in p.coeffs()):
        return None
    if any(as_expr(c).has(x) for c in q.coeffs()):
        return None
    if q.is_zero:
        return None
    return _Fraction(p, q, x)


class _Pole:
    """A pole with its multiplicity."""

    def __init__(self, point: Expr, multiplicity: int) -> None:
        self.point = point
        self.multiplicity = multiplicity

    def __repr__(self) -> str:
        return "_Pole(%s, %s)" % (self.point, self.multiplicity)


def poles(q: Poly) -> Optional[list[_Pole]]:
    """The roots of ``q`` with multiplicities, or ``None`` when they cannot
    all be found (``roots`` for parametric coefficients, ``ComplexRootOf``
    for numerical ones)."""
    if q.degree() <= 0:
        return []
    parametric = any(free_symbols(as_expr(c)) for c in q.coeffs())
    found = attempt(lambda: roots(q), settings.timeout)
    if found is not None and sum(found.values()) == q.degree():
        return [_Pole(as_expr(r), int(m)) for r, m in found.items()]
    if parametric:
        return None
    result: list[_Pole] = []
    _, factors = factor_list(q.as_expr(), q.gens[0])
    for g, e in factors:
        g_ = Poly(g, q.gens[0])
        if g_.degree() <= 0:
            continue
        all_roots = attempt(lambda: g_.all_roots(), settings.timeout)
        if all_roots is None:
            return None
        result.extend(_Pole(as_expr(r), int(e)) for r in all_roots)
    if sum(p.multiplicity for p in result) != q.degree():
        return None
    return result


# ---------------------------------------------------------------------------
# Deciding where a pole lies

class _Locator:
    """Answers ``query`` about a pole: exactly, or under the assumptions,
    or (with the numerical checks on) at a sample of the parameters."""

    def __init__(self, assumptions: Assumptions) -> None:
        self.assumptions = assumptions
        self.sampled = False
        self._sample: Optional[dict[Symbol, Expr]] = None
        self._no_sample = False
        self._slow = False

    def decide(self, query: Boolean) -> Optional[bool]:
        q = as_boolean(query)
        if q is true:
            return True
        if q is false:
            return False
        # a question about radicals of the parameters is expensive for the
        # exact decision procedures: the sample is consulted first, and
        # the value found with it is verified numerically afterwards
        if settings.numerical_checks and _has_radicals(q):
            verdict = self._at_sample(q)
            if verdict is not None:
                return verdict
        if self._slow:
            return None
        try:
            with time_limit(settings.timeout):
                verdict = ask(q, self.assumptions)
        except TimeLimitExceeded:
            self._slow = True
            return None
        if verdict is not None:
            return verdict
        if not settings.numerical_checks:
            return None
        return self._at_sample(q)

    def _at_sample(self, q: Boolean) -> Optional[bool]:
        sample = self.sample(sorted_symbols(free_symbols(q)))
        if sample is None:
            return None
        at_sample = as_boolean(q.xreplace(dict(sample.items())))
        if at_sample is true:
            self.sampled = True
            return True
        if at_sample is false:
            self.sampled = True
            return False
        verdict = ask(at_sample)
        if verdict is not None:
            self.sampled = True
        return verdict

    def sample(self, symbols: Sequence[Symbol]) -> Optional[dict[Symbol, Expr]]:
        """A point of the parameter space satisfying the assumptions."""
        if self._no_sample:
            return None
        if self._sample is not None and all(s in self._sample for s in symbols):
            return self._sample
        facts = Facts(self.assumptions, symbols=symbols)
        wanted = sorted_symbols(set(symbols) | free_symbols(facts.formula))
        # plain symbols, so that the flags of the symbols (positive=True)
        # are relations the SAT solver sees
        plain = {s: Dummy(s.name) for s in wanted}
        flags: list[Boolean] = []
        for s in wanted:
            if s.is_positive:
                flags.append(as_boolean(plain[s] > 0))
            elif s.is_negative:
                flags.append(as_boolean(plain[s] < 0))
            elif s.is_nonnegative:
                flags.append(as_boolean(plain[s] >= 0))
            elif s.is_nonzero:
                flags.append(as_boolean(Ne(plain[s], 0)))
        formula = as_boolean(And(as_boolean(facts.formula.xreplace(dict(plain.items()))), *flags))
        if formula is true:
            # nothing constrains the parameters: a sample says nothing
            self._no_sample = True
            return None
        witnesses = find_instance(formula, [plain[s] for s in wanted])
        if not witnesses:
            self._no_sample = True
            return None
        self._sample = {s: as_expr(witnesses[0].get(plain[s], Rational(3, 2))) for s in wanted}
        return self._sample

    def imaginary_sign(self, point: Expr) -> Optional[int]:
        """The sign of the imaginary part of a pole: ``1``, ``-1``, ``0``
        (real) or ``None``."""
        if isinstance(point, ComplexRootOf):
            if point.is_real:
                return 0
            value = im(point.evalf(30))
            return 1 if value > 0 else -1
        imaginary = as_expr(im(point))
        if self.decide(as_boolean(imaginary > 0)):
            return 1
        if self.decide(as_boolean(imaginary < 0)):
            return -1
        real = self.decide(element(point, S.Reals))
        if real is True:
            return 0
        return None

    def inside_unit_circle(self, point: Expr) -> Optional[bool]:
        """Whether ``|point| < 1``; ``None`` on the circle or undecided."""
        if isinstance(point, ComplexRootOf) or not free_symbols(point):
            value = Abs(point.evalf(30))
            distance = value - 1
            if abs(distance) < Rational(1, 10**20):
                return None
            return bool(distance < 0)
        modulus = as_expr(Abs(point))
        inside = self.decide(as_boolean(modulus < 1))
        if inside is not None:
            return inside if inside else (None if self.decide(as_boolean(modulus > 1)) is not True else False)
        if self.decide(element(point, S.Reals)):
            square = as_expr(point**2)
            inside = self.decide(as_boolean(square < 1))
            if inside is True:
                return True
            if self.decide(as_boolean(square > 1)) is True:
                return False
        return None

    def positive_real(self, point: Expr) -> Optional[bool]:
        """Whether the pole lies on the positive real axis."""
        s = self.imaginary_sign(point)
        if s is None:
            return None
        if s != 0:
            return False
        return self.decide(as_boolean(re(point) > 0))


def _has_radicals(q: Boolean) -> bool:
    """Whether the question involves non-polynomial functions of the
    parameters (radicals, absolute values, ...)."""
    for node in q.atoms(Pow):
        if not isinstance(node.exp, Integer):
            return True
    return bool(q.atoms(Function, Abs))


# ---------------------------------------------------------------------------
# Residues

def _residue(fraction: _Fraction, pole: _Pole, factor: Expr, z: Symbol) -> Optional[Expr]:
    """The residue of ``fraction(z) * factor(z)`` at the pole, ``factor``
    holomorphic there: ``P(z_k) factor(z_k)/Q'(z_k)`` at a simple pole,
    the derivative formula otherwise (the branch of a logarithm written
    as ``L(z)`` with ``L'(z) = 1/z`` is left in place)."""
    p = fraction.numerator.as_expr()
    x = fraction.x
    point = pole.point
    if pole.multiplicity == 1:
        derivative = as_expr(fraction.denominator.diff().as_expr())
        value = as_expr((p * factor.subs(z, x)).subs(x, point) / derivative.subs(x, point))
        return value
    m = pole.multiplicity
    reduced = _divide_out(fraction, point, m)
    if reduced is None:
        return None
    h: Expr = reduced * factor.subs(z, x)
    for _ in range(m - 1):
        h = as_expr(h.diff(x))
        h = as_expr(h.subs(Derivative(_L(x), x), 1 / x))
    value = as_expr(h.subs(x, point) / factorial(m - 1))
    if value.has(nan, zoo):
        found = attempt(lambda: as_expr(limit(h, x, point)), settings.timeout)
        if found is None or found.has(nan, zoo, oo):
            return None
        value = as_expr(found / factorial(m - 1))
    return value


def _divide_out(fraction: _Fraction, point: Expr, m: int) -> Optional[Expr]:
    """``(x - point)**m * P/Q`` with the pole divided out of ``Q`` exactly."""
    x = fraction.x
    divisor = Poly((x - point)**m, x, domain='EX')
    try:
        quotient, remainder = Poly(fraction.denominator.as_expr(), x, domain='EX').div(divisor)
    except PolynomialError:
        return None
    if not remainder.is_zero:
        simplified = attempt(lambda: as_expr(cancel((x - point)**m * fraction.as_expr())), settings.timeout)
        return simplified
    return as_expr(fraction.numerator.as_expr() / quotient.as_expr())


def _branch_log(point: Expr, imaginary_sign: int) -> Expr:
    """``log|point| + i theta`` with ``0 < theta < 2 pi`` (``theta = pi``
    for a negative real pole)."""
    if imaginary_sign == 0:
        return as_expr(log(-point) + I * pi)
    theta = as_expr(arg(point))
    if imaginary_sign < 0:
        theta = as_expr(theta + 2 * pi)
    return as_expr(log(Abs(point)) + I * theta)


def _tidy(value: Expr, assumptions: Assumptions) -> Expr:
    """The value simplified, with the imaginary parts which cancel for
    real parameters removed."""
    result = value
    parameters = sorted_symbols(free_symbols(result))
    reals = {s: Dummy(s.name, real=True) for s in parameters
             if s.is_extended_real or ask(element(s, S.Reals), assumptions)}
    if reals and result.has(I):
        back = {d: s for s, d in reals.items()}
        expanded = attempt(lambda: as_expr(expand(result.xreplace(dict(reals.items())), complex=True)),
                           settings.timeout)
        if expanded is not None and not expanded.has(I):
            result = as_expr(expanded.xreplace(dict(back.items())))
    simpler = attempt(lambda: as_expr(simplify(cancel(result))), settings.timeout)
    if simpler is not None and _size(simpler) <= _size(result):
        result = simpler
    if result.has(I) and not reals and not parameters:
        simpler = attempt(lambda: as_expr(simplify(expand(result, complex=True))), settings.timeout)
        if simpler is not None and not simpler.has(I):
            result = simpler
    return result


def _size(e: Expr) -> int:
    return int(e.count_ops(visual=False))


# ---------------------------------------------------------------------------
# 1. and 2.: the real line

def _sum_residues(fraction: _Fraction, factor: Expr, z: Symbol, locator: _Locator,
                  upper: bool) -> Optional[Expr]:
    """The sum of the residues of ``fraction * factor`` in the upper (or
    lower) half plane; ``None`` with a real pole or an undecided one."""
    found = poles(fraction.denominator)
    if found is None:
        return None
    total: Expr = S.Zero
    for pole in found:
        s = locator.imaginary_sign(pole.point)
        if s is None or s == 0:
            return None
        if (s > 0) == upper:
            value = _residue(fraction, pole, factor, z)
            if value is None:
                return None
            total = total + value
    return total


def real_line_integral(f: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(R(x), (x, -oo, oo))`` for a rational function with
    ``deg Q >= deg P + 2`` and no real poles.

    Examples
    ========

    >>> from sympy import symbols, oo
    >>> from sympy_extras.integrals.residues import real_line_integral
    >>> x = symbols('x')
    >>> real_line_integral(1/(x**2 + 1)**2, x)
    ConditionalValue(pi/2)
    """
    fraction = rational_function(f, x)
    if fraction is None or fraction.excess < 2:
        return None
    locator = _Locator(assumptions)
    z = Dummy('z')
    total = _sum_residues(fraction, S.One, z, locator, True)
    if total is None:
        return None
    return _checked(ConditionalValue(_tidy(as_expr(2 * pi * I * total), assumptions)),
                    f, x, -oo, oo, assumptions, locator)


def _fourier_factor(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, str]]:
    """``(R, k, kind)`` when ``f == R(x) * g(k x)`` with ``g`` one of
    ``exp(I*.)``, ``cos``, ``sin`` (``kind`` ``'exp'``, ``'cos'``,
    ``'sin'``) and ``R`` rational."""
    factors = [as_expr(t) for t in Mul.make_args(f)]
    rest: list[Expr] = []
    found: Optional[tuple[Expr, str]] = None
    for e in factors:
        kind: Optional[str] = None
        if isinstance(e, exp):
            kind = 'exp'
        elif isinstance(e, cos):
            kind = 'cos'
        elif isinstance(e, sin):
            kind = 'sin'
        if kind is None or found is not None or not e.has(x):
            rest.append(e)
            continue
        inner = as_expr(e.args[0])
        coefficient, variable = inner.as_independent(x, as_Add=False)
        if as_expr(variable) != x:
            return None
        k = as_expr(coefficient)
        if kind == 'exp':
            k = as_expr(k / I)
            if k.is_extended_real is not True and not (k.is_number and im(k) == 0 and k != 0):
                return None
        found = (k, kind)
    if found is None:
        return None
    return (as_expr(Mul(*rest)), found[0], found[1])


def _real_parameters(e: Expr, assumptions: Assumptions) -> Optional[dict[Symbol, Dummy]]:
    """Real dummies for the parameters of ``e``, or ``None`` when one is
    not known to be real."""
    reals: dict[Symbol, Dummy] = {}
    for s in sorted_symbols(free_symbols(e)):
        if s.is_extended_real or ask(element(s, S.Reals), assumptions):
            reals[s] = Dummy(s.name, real=True)
        else:
            return None
    return reals


def _part(e: Expr, real: bool, assumptions: Assumptions) -> Optional[Expr]:
    """The real (or imaginary) part of ``e`` for real parameters, when it
    comes out free of ``re``, ``im`` and ``I``."""
    reals = _real_parameters(e, assumptions)
    if reals is None:
        return None
    back = {d: s for s, d in reals.items()}
    replaced = as_expr(e.xreplace(dict(reals.items())))
    found = attempt(lambda: as_expr(expand(re(replaced) if real else im(replaced), complex=True)),
                    settings.timeout)
    if found is None or found.has(re, im, I):
        return None
    return as_expr(found.xreplace(dict(back.items())))


def fourier_integral(f: Expr, x: Symbol, assumptions: Assumptions = None,
                     half: bool = False) -> Optional[ConditionalValue]:
    """``Integral(R(x)*exp(I*k*x), (x, -oo, oo))``, and with ``cos(k*x)`` or
    ``sin(k*x)``, for a rational ``R`` with ``deg Q >= deg P + 1`` and no
    real poles, by Jordan's lemma; with ``half=True`` the integral over
    ``(0, oo)`` of an even integrand.

    Examples
    ========

    >>> from sympy import symbols, cos, sin
    >>> from sympy_extras.integrals.residues import fourier_integral
    >>> x, k = symbols('x k')
    >>> fourier_integral(cos(2*x)/(x**2 + 1), x)
    ConditionalValue(pi*exp(-2))
    >>> fourier_integral(x*sin(k*x)/(x**2 + 4), x)
    ConditionalValue(pi*exp(-2*k), k > 0)
    """
    found = _fourier_factor(f, x)
    if found is None:
        return None
    r, k, kind = found
    fraction = rational_function(r, x)
    if fraction is None or fraction.excess < 1:
        return None
    if half:
        even = attempt(lambda: as_expr(cancel(f - f.subs(x, -x))), settings.timeout)
        if even is None or even != 0:
            return None
    locator = _Locator(assumptions)
    z = Dummy('z')
    condition: Boolean = true
    sign_k = 0
    if k.is_positive:
        sign_k = 1
    elif k.is_negative:
        sign_k = -1
    else:
        verdict = ask(as_boolean(k > 0), assumptions)
        if verdict is True:
            sign_k = 1
        elif verdict is False and ask(as_boolean(k < 0), assumptions) is True:
            sign_k = -1
        elif verdict is None:
            condition = as_boolean(k > 0)
            sign_k = 1
        else:
            return None
    kappa = k if sign_k > 0 else -k
    # E(kappa) = 2 pi i sum_upper Res R e^{i kappa z}, E(-kappa) = -2 pi i sum_lower Res R e^{-i kappa z}
    upper = _sum_residues(fraction, exp(I * kappa * z), z, locator, True)
    if upper is None:
        return None
    plus = as_expr(2 * pi * I * upper)
    value: Optional[Expr] = None
    if kind == 'exp':
        if sign_k > 0:
            value = plus
        else:
            lower = _sum_residues(fraction, exp(-I * kappa * z), z, locator, False)
            if lower is None:
                return None
            value = as_expr(-2 * pi * I * lower)
    else:
        # the real and imaginary parts of E(kappa) when the coefficients are real
        value = _part(plus, kind == 'cos', assumptions)
        if value is None:
            lower = _sum_residues(fraction, exp(-I * kappa * z), z, locator, False)
            if lower is None:
                return None
            minus = as_expr(-2 * pi * I * lower)
            if kind == 'cos':
                value = as_expr((plus + minus) / 2)
            else:
                value = as_expr((plus - minus) / (2 * I))
        if kind == 'sin' and sign_k < 0:
            value = -value
    if half:
        value = as_expr(value / 2)
    return _checked(ConditionalValue(_tidy(value, assumptions), condition), f, x,
                    S.Zero if half else -oo, oo, assumptions, locator)


# ---------------------------------------------------------------------------
# 3.: the keyhole contour

def _power_and_logs(f: Expr, x: Symbol) -> Optional[tuple[Expr, int, Expr]]:
    """``(alpha, n, R)`` when ``f == x**alpha * log(x)**n * R(x)``."""
    alpha: Expr = S.Zero
    n = 0
    rest: list[Expr] = []
    for e in (as_expr(t) for t in Mul.make_args(f)):
        if e == x:
            alpha = alpha + 1
        elif isinstance(e, Pow) and e.base == x and not as_expr(e.exp).has(x):
            alpha = alpha + as_expr(e.exp)
        elif isinstance(e, log) and e.args[0] == x:
            n += 1
        elif isinstance(e, Pow) and isinstance(e.base, log) and e.base.args[0] == x \
                and isinstance(e.exp, Integer) and e.exp > 0:
            n += int(e.exp)
        else:
            rest.append(e)
    return (alpha, n, as_expr(Mul(*rest)))


def _keyhole_sum(fraction: _Fraction, alpha: Expr, locator: _Locator, z: Symbol) -> Optional[Expr]:
    """``sum Res z^alpha R(z)`` over the poles off the positive real axis,
    with ``z^alpha = exp(alpha L(z))``, ``0 < arg z < 2 pi``."""
    found = poles(fraction.denominator)
    if found is None:
        return None
    total: Expr = S.Zero
    for pole in found:
        if pole.point == 0:
            return None
        s = locator.imaginary_sign(pole.point)
        if s is None:
            return None
        if s == 0 and locator.decide(as_boolean(re(pole.point) > 0)) is not False:
            return None
        value = _residue(fraction, pole, exp(alpha * _L(z)), z)
        if value is None:
            return None
        total = total + value.subs(_L(pole.point), _branch_log(pole.point, s))
    return total


def _log_sum(fraction: _Fraction, locator: _Locator, z: Symbol) -> Optional[Expr]:
    """``-sum Res R(z) L(z)`` over the poles, ``0 < arg z < 2 pi``."""
    found = poles(fraction.denominator)
    if found is None:
        return None
    total: Expr = S.Zero
    for pole in found:
        if pole.point == 0:
            return None
        s = locator.imaginary_sign(pole.point)
        if s is None:
            return None
        if s == 0 and locator.decide(as_boolean(re(pole.point) > 0)) is not False:
            return None
        value = _residue(fraction, pole, _L(z), z)
        if value is None:
            return None
        total = total + value.subs(_L(pole.point), _branch_log(pole.point, s))
    return as_expr(-total)


def _absorb_zero(fraction: _Fraction, alpha: Expr) -> tuple[_Fraction, Expr]:
    """The factor ``x**m`` of the numerator or the denominator moved into
    the power."""
    x = fraction.x
    p, q = fraction.numerator, fraction.denominator
    while p.degree() > 0 and p.eval(0) == 0:
        p = Poly(cancel(p.as_expr() / x), x)
        alpha = as_expr(alpha + 1)
    while q.degree() > 0 and q.eval(0) == 0:
        q = Poly(cancel(q.as_expr() / x), x)
        alpha = as_expr(alpha - 1)
    return _Fraction(p, q, x), alpha


def keyhole_integral(f: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(x**alpha * log(x)**n * R(x), (x, 0, oo))`` for a rational
    ``R`` without poles on the positive real axis, by the keyhole contour
    (a plain rational function, ``alpha`` an integer, by the logarithm).

    Examples
    ========

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.residues import keyhole_integral
    >>> x, a = symbols('x a')
    >>> keyhole_integral(x**a/(x + 1), x)
    ConditionalValue(-pi/sin(pi*a), (a > -1) & (a < 0))
    >>> keyhole_integral(1/(x**3 + 1), x)
    ConditionalValue(2*sqrt(3)*pi/9)
    >>> keyhole_integral(log(x)/(x**2 + 1), x)
    ConditionalValue(0)
    """
    found = _power_and_logs(f, x)
    if found is None:
        return None
    alpha, n, r = found
    fraction = rational_function(r, x)
    if fraction is None:
        return None
    fraction, alpha = _absorb_zero(fraction, alpha)
    locator = _Locator(assumptions)
    z = Dummy('z')
    lower_bound = as_boolean(alpha > -1)
    upper_bound = as_boolean(alpha < fraction.excess - 1)
    condition = as_boolean(And(lower_bound, upper_bound))
    if condition is false:
        return None
    if n > 0:
        return _keyhole_with_logs(fraction, alpha, n, locator, z, condition, f, x, assumptions)
    if alpha.is_integer:
        p, q = fraction.numerator, fraction.denominator
        shift = int(alpha)
        if shift >= 0:
            p = Poly(p.as_expr() * x**shift, x)
        else:
            q = Poly(q.as_expr() * x**(-shift), x)
        folded = _Fraction(p, q, x)
        if folded.excess < 2 or shift < 0:
            return None
        total = _log_sum(folded, locator, z)
        if total is None:
            return None
        return _checked(ConditionalValue(_tidy(total, assumptions)), f, x, S.Zero, oo, assumptions, locator)
    total = _keyhole_sum(fraction, alpha, locator, z)
    if total is None:
        return None
    value = as_expr(2 * pi * I * total / (1 - exp(2 * pi * I * alpha)))
    return _checked(ConditionalValue(_tidy(value, assumptions), condition), f, x, S.Zero, oo,
                    assumptions, locator)


def _keyhole_with_logs(fraction: _Fraction, alpha: Expr, n: int, locator: _Locator, z: Symbol,
                       condition: Boolean, f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """The power formula with a symbolic exponent, differentiated ``n``
    times; the limit at an integer exponent."""
    a = Dummy('alpha', real=True)
    total = _keyhole_sum(fraction, a, locator, z)
    if total is None:
        return None
    value = as_expr(2 * pi * I * total / (1 - exp(2 * pi * I * a)))
    for _ in range(n):
        value = as_expr(value.diff(a))
    if alpha.is_integer:
        at = attempt(lambda: as_expr(limit(value, a, alpha)), settings.timeout)
        if at is None or at.has(oo, -oo):
            return None
        value = at
    else:
        value = as_expr(value.subs(a, alpha))
    return _checked(ConditionalValue(_tidy(value, assumptions), condition), f, x, S.Zero, oo,
                    assumptions, locator)


# ---------------------------------------------------------------------------
# 4.: a period

def _periods(a: Expr, b: Expr) -> Optional[int]:
    """``k`` when ``b - a == 2*pi*k`` for a positive integer ``k``."""
    k = as_expr((b - a) / (2 * pi))
    if isinstance(k, Integer) and k > 0:
        return int(k)
    return None


def period_integral(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` over ``k`` periods, ``b - a = 2*pi*k``,
    for a rational function of ``sin(x)`` and ``cos(x)`` (also of
    multiple angles), through ``z = exp(I*x)`` and the residues inside
    the unit circle.

    Examples
    ========

    >>> from sympy import symbols, cos, pi
    >>> from sympy_extras.integrals.residues import period_integral
    >>> x = symbols('x')
    >>> a = symbols('a', positive=True)
    >>> period_integral(1/(1 - 2*a*cos(x) + a**2), x, 0, 2*pi, a < 1)
    ConditionalValue(-2*pi/(a**2 - 1))
    """
    k = _periods(a, b)
    if k is None or not f.has(TrigonometricFunction):
        return None
    z = Dummy('z')
    g = f.rewrite(cos).rewrite(sin) if f.has(TrigonometricFunction) else f
    g = as_expr(expand_trig(g))
    for node in g.atoms(TrigonometricFunction):
        if node.has(x) and not (isinstance(node, (sin, cos)) and node.args[0] == x):
            return None
    substituted = as_expr(g.xreplace({sin(x): (z - 1 / z) / (2 * I), cos(x): (z + 1 / z) / 2}) / (I * z))
    if substituted.has(x):
        return None
    fraction = rational_function(substituted, z)
    if fraction is None:
        return None
    found = poles(fraction.denominator)
    if found is None:
        return None
    locator = _Locator(assumptions)
    total: Expr = S.Zero
    for pole in found:
        inside = locator.inside_unit_circle(pole.point)
        if inside is None:
            return None
        if inside:
            value = _residue(fraction, pole, S.One, z)
            if value is None:
                return None
            total = total + value
    value = as_expr(2 * pi * I * k * total)
    return _checked(ConditionalValue(_tidy(value, assumptions)), f, x, a, b, assumptions, locator)


# ---------------------------------------------------------------------------
# Verification and the entry point

def _checked(result: ConditionalValue, f: Expr, x: Symbol, a: Expr, b: Expr,
             assumptions: Assumptions, locator: _Locator) -> Optional[ConditionalValue]:
    """The result, verified numerically when a pole was placed by
    sampling (the numerical checks are then on)."""
    if not locator.sampled:
        return result
    from .definite import verify_numerically
    if verify_numerically(result.value, f, x, a, b, assumptions) is False:
        return None
    return result


def residue_integral(f: Expr, x: Symbol, a: Expr, b: Expr,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` by the residue theorem when ``f`` is one
    of the families of the module documentation; ``None`` otherwise.

    Parameters
    ==========

    f : Expr
    x : Symbol
    a, b : Expr
        The bounds: ``(-oo, oo)``, ``(0, oo)`` or a range of length
        ``2*pi*k``.
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters, used to place the poles.

    Returns
    =======

    A :class:`~.conditions.ConditionalValue`: the value and the condition
    on the parameters (the sign of ``k`` in a Fourier integral, the
    convergence conditions on the exponent of a power).

    Examples
    ========

    >>> from sympy import symbols, sin, oo, pi, log
    >>> from sympy_extras.integrals.residues import residue_integral
    >>> x = symbols('x')
    >>> residue_integral(x*sin(x)/(x**2 + 1), x, -oo, oo)
    ConditionalValue(pi*exp(-1))
    >>> residue_integral(log(x)**2/(x**2 + 1), x, 0, oo)
    ConditionalValue(pi**3/8)
    >>> residue_integral(1/(5 + 3*sin(x)), x, 0, 2*pi)
    ConditionalValue(pi/2)
    >>> residue_integral(1/(x**2 + 1), x, 0, 1) is None
    True
    """
    a, b = as_expr(a), as_expr(b)
    if not f.has(x):
        return None
    constant, rest = f.as_independent(x, as_Add=False)
    constant_, rest_ = as_expr(constant), as_expr(rest)
    if constant_ != 1:
        found = residue_integral(rest_, x, a, b, assumptions)
        return None if found is None else found.scaled(constant_)
    if a == -oo and b == oo:
        if rest_.has(exp, sin, cos):
            return fourier_integral(rest_, x, assumptions)
        return real_line_integral(rest_, x, assumptions)
    if a == 0 and b == oo:
        if rest_.has(exp, sin, cos):
            return fourier_integral(rest_, x, assumptions, half=True)
        if rest_.has(TrigonometricFunction):
            return None
        return keyhole_integral(rest_, x, assumptions)
    if a.is_number and b.is_number and _periods(a, b) is not None:
        return period_integral(rest_, x, a, b, assumptions)
    return None

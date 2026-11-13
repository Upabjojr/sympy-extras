"""The Marichev–Adamchik method: definite integrals over `(0, \\infty)`,
`(0, 1)` and `(1, \\infty)` through Mellin transforms.

An integrand `x^\\alpha \\log^n x\\, f(\\beta_1 x^{\\gamma_1})\\, g(\\beta_2 x^{\\gamma_2})`
whose factors `f` and `g` are in the table of :mod:`.mellin` is
integrated over `(0, \\infty)` as follows [Marichev]_, [Adamchik]_:

* with one factor, the integral is the Mellin transform of `f` at
  `s = \\alpha + 1`, valid when `\\alpha + 1` lies in the strip of the
  transform (the condition under which the integral converges at `0` and
  at `\\infty`);
* with two factors, Parseval's formula for the Mellin transform,

  .. math::

      \\int_0^\\infty x^\\alpha f(x) g(x)\\, dx =
      \\frac{1}{2\\pi i} \\int_{c - i\\infty}^{c + i\\infty} F(t)\\, G(\\alpha + 1 - t)\\, dt,

  valid on a line `\\operatorname{Re} t = c` inside both strips, turns the
  integral into a Mellin–Barnes integral of a quotient of gamma functions,
  that is a Meijer G-function evaluated at the ratio of the scales
  (:func:`.slater.mellin_barnes`), which Slater's theorem writes as
  hypergeometric functions (:func:`.slater.expand_meijerg`);
* a logarithm `\\log^n x` is a derivative with respect to `\\alpha`: the
  integral is computed with a symbolic exponent and differentiated.

A finite range `(0, 1)` is the range `(0, \\infty)` with the step function
`\\theta(1 - x)` as one more factor (its transform is `1/s`), or with the
factor `(1 - x)^{b-1}` of a Beta integral when the integrand has one;
`(1, \\infty)` likewise with `\\theta(x - 1)`.

The conditions of validity -- the strip of each transform, the
convergence of the Mellin–Barnes integral, the parameters of the kernels
-- are inequalities on the parameters; they are returned with the value
as a :class:`~.conditions.ConditionalValue` and decided against the
assumptions by the caller (:func:`~sympy_extras.integrals.definite.definite_integral`).

The module works on the objects of the core (:mod:`sympy_extras.core`, the
functions of :mod:`sympy_extras.functions`) with the native ``ask``,
``refine``, ``hyperexpand``, ``simplify`` and limits, and imports no
SymPy: :mod:`sympy_extras.integrals.marichev_sympy` is its face on SymPy
objects, used by the integrators which are still written on SymPy.

Examples
========

>>> from sympy_extras.core import Symbol, symbols
>>> from sympy_extras.functions import exp, log, sin
>>> from sympy_extras.integrals.marichev import mellin_integrate
>>> x, k = symbols('x k')
>>> a, b = symbols('a b', positive=True)
>>> mellin_integrate(exp(-a*x)*sin(b*x), x)
ConditionalValue(b/(a**2 + b**2))
>>> mellin_integrate(x**k/(x + 3), x)
ConditionalValue(-3**k*pi/sin(pi*k), (k > -1) & (k < 0))
>>> mellin_integrate(exp(-x**2)*log(x), x)
ConditionalValue(sqrt(pi)*(-log(4) - EulerGamma)/4)
>>> mellin_integrate(x**(b - 1)*(1 - x)**(a - 1), x, cutoff='lower')
ConditionalValue(gamma(a)*gamma(b)/gamma(a + b))

References
==========

.. [Marichev] O. I. Marichev, *Handbook of integral transforms of higher
   transcendental functions: theory and algorithmic tables*, Ellis
   Horwood, 1983, chapter 3.
.. [Adamchik] V. S. Adamchik, O. I. Marichev, *The algorithm for
   calculating integrals of hypergeometric type functions and its
   realization in REDUCE system*, ISSAC 1990, pp. 212-224.
"""
from __future__ import annotations

from typing import Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.refine import refine
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import evalf
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.function import Derivative, Subs, count_ops, expand_complex, expand_func
from sympy_extras.core.limits import Limit, limit
from sympy_extras.core.logic import Boolean, BooleanFalse, BooleanTrue, and_, to_boolean, true
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Float, I, Integer, pi
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import lt
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Im, PolarLift, PrincipalBranch, Re, re, unpolarify
from sympy_extras.functions.elementary.exponential import Exp, ExpPolar, Log, exp, log
from sympy_extras.functions.elementary.hyperbolic import Cosh, Sinh
from sympy_extras.functions.elementary.integers import ceiling
from sympy_extras.functions.elementary.trigonometric import Cos, Sin, sin
from sympy_extras.functions.special.bessel import BesselI, besseli, besselk
from sympy_extras.functions.special.elliptic_integrals import EllipticE, EllipticF, EllipticK, EllipticPi
from sympy_extras.functions.special.gamma_functions import LowerGamma, PolyGamma
from sympy_extras.functions.special.hyper import MeijerG as MeijerGFunction, hyper
from sympy_extras.functions.special.zeta_functions import DirichletEta, LerchPhi
from sympy_extras.polys.polytools import cancel, factor
from sympy_extras.sets.sets import Reals
from sympy_extras.settings import settings
from sympy_extras.simplify.gammasimp import gammasimp
from sympy_extras.simplify.simplify import simplify
from .conditions import ConditionalValue, decide, numerical_verdict
from .mellin import BETA_LOWER, GammaQuotient, Match, Product, _beta_kernel, decompose_integrand, monomial
from .slater import expand_meijerg, line_conditions, mellin_barnes

__all__ = ['mellin_integrate', 'integrate_product', 'evaluate_quotient', 'tidy', 'checked_tidy', 'real_logarithms',
           'polar_elliptic', 'polar_lowergamma', 'projected_sums', 'real_constant', 'right_half_plane_powers', 'bessel_k_forms', 'exponential_form',
           'reduce_positive_powers']


def _xreplace(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced, evaluated (SymPy's ``xreplace``)."""
    replaced = e._xreplace_evaluated(rule)
    if not isinstance(replaced, Expr):
        raise TypeError("an expression is expected, got %s" % (replaced,))
    return replaced


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


def _has_infinity(e: Expr) -> bool:
    """``e.has(nan, zoo, oo)``."""
    return e.has(S.NaN, S.ComplexInfinity, S.Infinity)


def evaluate_quotient(quotient: GammaQuotient, point: Expr) -> Optional[Expr]:
    """The value of the quotient at ``point``, by a limit when a pole of
    the numerator meets a zero of the denominator (``sin(x)/x`` at
    ``s = 0``); ``None`` when the limit is not found.

    >>> from sympy_extras.core import Integer
    >>> from sympy_extras.integrals.mellin import GammaQuotient
    >>> from sympy_extras.integrals.marichev import evaluate_quotient
    >>> evaluate_quotient(GammaQuotient(1, [], [(0, 1)], [(1, 1)], 0), Integer(2))
    1/2
    """
    value = quotient.as_expr(point)
    if not _has_infinity(value):
        return value
    s = Dummy('s')
    found = attempt(lambda: limit(quotient.as_expr(s), s, point), settings.timeout, native=True)
    if found is None or _has_infinity(found) or found.has(Limit):
        return None
    return found


def _facts(assumptions: Assumptions, condition: Boolean) -> list[Boolean]:
    """The condition and the assumptions as a list of statements."""
    facts: list[Boolean] = [] if isinstance(condition, BooleanTrue) else [condition]
    if isinstance(assumptions, (Basic, bool)):
        facts.append(to_boolean(assumptions))
    elif assumptions is not None:
        facts.extend(to_boolean(a) for a in assumptions)
    return facts


def tidy(value: Expr, assumptions: Assumptions = None, condition: Boolean = true) -> Expr:
    """The value with polar numbers removed and simplified, under the
    time limit of the settings; ``refine`` uses the assumptions and the
    condition under which the value holds.

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.functions import gamma
    >>> from sympy_extras.integrals.marichev import tidy
    >>> a = Symbol('a')
    >>> tidy(gamma(a + 1)/gamma(a) + sqrt(a**2), a > 0)
    2*a
    """
    result = value
    # a quarter of the budget for each simplification: ``trigsimp`` of a
    # sine of a complex argument factors over Q(i) for half a minute
    budget = None if settings.timeout is None else settings.timeout / 4
    if result.has(LowerGamma) and result.has(ExpPolar):
        result = polar_lowergamma(result)
    if result.has(ExpPolar) and result.has(EllipticK, EllipticE, EllipticF, EllipticPi):
        result = polar_elliptic(result)
    if result.has(PolarLift, ExpPolar):
        result = projected_sums(result)
    if result.has(ExpPolar, PolarLift, PrincipalBranch):
        current = result
        unpolar = attempt(lambda: unpolarify(current), budget, native=True)
        if unpolar is not None:
            result = unpolar
    if result.has(MeijerGFunction):
        return result
    if result.has(BesselI):
        result = bessel_k_forms(result)
    if result.has(PolyGamma):
        current = result
        expanded = attempt(lambda: expand_func(current), budget, native=True)
        if expanded is not None and _size(expanded) < _size(result):
            result = expanded
    if result.has(LerchPhi, DirichletEta):
        # the Dirichlet functions at integers are logarithms and constants
        current = result
        expanded = attempt(lambda: simplify(expand_func(current)), budget, native=True)
        if expanded is not None and not expanded.has(LerchPhi, DirichletEta):
            result = expanded
    current = result
    simpler = attempt(lambda: simplify(gammasimp(current)), budget, native=True)
    if simpler is not None and _size(simpler) <= _size(result):
        result = simpler
    current = result
    factored = attempt(lambda: factor_terms(factor(current)), budget, native=True)
    if factored is not None and _size(factored) < _size(result):
        result = factored
    facts = _facts(assumptions, condition)
    if result.has(Log):
        result = real_logarithms(result, facts)
    if result.has(I):
        result = right_half_plane_powers(result, facts)
    if not result.free_symbols and _has_complex_unit(result):
        current = result
        real = attempt(lambda: real_constant(current), budget, native=True)
        if real is not None:
            result = real
    if facts:
        current = result
        refined = attempt(lambda: _expr(refine(current, facts)), budget, native=True)
        if refined is not None and _size(refined) <= _size(result):
            result = refined
    if result.has(LowerGamma) and result.has(ExpPolar):
        # simplify's hyperexpand brings the polar form back
        result = polar_lowergamma(result)
    return result


def checked_tidy(value: Expr, assumptions: Assumptions = None, condition: Boolean = true) -> Expr:
    """:func:`tidy` with a safety net: the tidied value when it cannot be
    told from ``value`` numerically, ``value`` itself when the two differ
    at values of the parameters satisfying the assumptions and the
    condition (:func:`~.conditions.numerical_verdict`).

    The simplifications of :func:`tidy` (``unpolarify``, ``simplify``,
    ``refine``, the rewriting of logarithms and powers) move values across
    branch cuts when they go wrong, and they come after the value was
    checked against a quadrature: this is the check of that last step.
    Only a difference which was seen rejects the tidied form; when nothing
    could be compared, or not within a quarter of the time limit of the
    settings, it is kept, and so it is when the numerical checks of the
    settings are off.

    Examples
    ========

    >>> from sympy_extras.core import Integer, Symbol, sqrt
    >>> from sympy_extras.functions import exp_polar, log
    >>> from sympy_extras.integrals.marichev import checked_tidy
    >>> a = Symbol('a')
    >>> checked_tidy(sqrt(a**2) + log(a*exp_polar(Integer(0))), a > 0)
    a + log(a)
    """
    tidied = tidy(value, assumptions, condition)
    if tidied == value or not settings.numerical_checks:
        return tidied
    # under a time limit, as the simplifications are: the sampling asks
    # the solver for an instance of the assumptions and the condition
    budget = None if settings.timeout is None else settings.timeout / 4
    verdict = attempt(lambda: numerical_verdict(value, tidied, assumptions, condition), budget, native=True)
    if verdict is False:
        return value
    return tidied


def _projected(e: Expr) -> Expr:
    """A summand with its polar factors replaced by their projections on
    the complex plane (the projection of a product is the product of the
    projections, of an integer power the power of the projection)."""
    if isinstance(e, PolarLift):
        return _projected(_expr(e.args[0]))
    if isinstance(e, ExpPolar):
        return exp(_projected(_expr(e.args[0])))
    if isinstance(e, Mul):
        return mul(*[_projected(f) for f in Mul.make_args(e)])
    if isinstance(e, Pow) and e.exp.is_integer:
        return _projected(e.base)**e.exp
    return e


def _is_polar_sum(node: Basic) -> bool:
    return isinstance(node, Add) and node.has(PolarLift, ExpPolar)


def _project_sum(node: Basic) -> Basic:
    if not isinstance(node, Add):
        return node
    return add(*[_projected(t) for t in Add.make_args(node)])


def projected_sums(value: Expr) -> Expr:
    """The polar numbers in the terms of sums replaced by their projections:
    a sum is an ordinary number, which depends on the projections of its
    terms only, so that ``1 - polar_lift(w)`` is ``1 - w``, but
    ``unpolarify`` leaves the polar number in the base of a power
    (``hyperexpand`` writes ``1/sqrt(1 - polar_lift(3/25 - 4*I/25))`` for
    ``1F0(1/2;; 3/25 - 4*I/25)``, which the Marichev–Adamchik value of
    ``exp(-x)*cos(2*x)*besselj(0, x)`` over ``(0, oo)`` has; the value was
    refused as a branch which could not be resolved).

    >>> from sympy_extras.core import I, Rational, sqrt
    >>> from sympy_extras.functions import polar_lift
    >>> from sympy_extras.integrals.marichev import projected_sums
    >>> projected_sums(1/sqrt(1 - polar_lift(Rational(3, 25) - 4*I/25)))
    1/sqrt(22/25 + 4*I/25)
    """
    return _expr(value.replace(_is_polar_sum, _project_sum))


def _split_polar(m: Expr) -> tuple[Expr, Expr]:
    """``m.as_independent(exp_polar, as_Add=False)``: the factors free of
    ``exp_polar`` and the others."""
    factors = Mul.make_args(m)
    free = [f for f in factors if not f.has(ExpPolar)]
    dependent = [f for f in factors if f.has(ExpPolar)]
    return mul(*free), mul(*dependent)


def _is_polar_minus_one(e: Expr) -> bool:
    """Whether ``e`` is ``exp_polar(I*pi)``."""
    return isinstance(e, ExpPolar) and e.args[0] == I * pi


def polar_elliptic(value: Expr) -> Expr:
    """The parameter ``m*exp_polar(I*pi)`` of a Legendre elliptic function
    (``hyperexpand`` writes ``elliptic_k(4*exp_polar(I*pi)/a**2)`` for the
    Laplace transform of ``J_0(x)**2``) written ``-m``: the functions are
    analytic in the parameter off ``[1, oo)``, so the negative axis needs
    no branch.

    >>> from sympy_extras.core import I, Symbol, pi
    >>> from sympy_extras.functions import elliptic_k, exp_polar
    >>> from sympy_extras.integrals.marichev import polar_elliptic
    >>> a = Symbol('a', positive=True)
    >>> polar_elliptic(elliptic_k(4*exp_polar(I*pi)/a**2))
    elliptic_k(-4/a**2)
    """
    replacement: dict[Basic, Basic] = {}
    for node in value.atoms(EllipticK, EllipticE, EllipticF, EllipticPi):
        m = _expr(node.args[-1])
        polar, rest = _split_polar(m)
        if _is_polar_minus_one(rest) and polar.is_extended_real:
            replacement[m] = -polar
    if not replacement:
        return value
    return _xreplace(value, replacement)


def polar_lowergamma(value: Expr) -> Expr:
    """``lowergamma(a, w*exp_polar(I*pi))``, which ``hyperexpand`` writes
    for ``1F1(1; a + 1; -w)`` and no branch resolves, as the real form
    ``(-1)**a * w**a * hyper((a,), (a + 1,), w) / a`` (the series of the
    incomplete gamma function, DLMF 8.7.1, at the polar argument; the
    phase ``(-1)**a`` combines with the one of the coefficient).

    >>> from sympy_extras.core import I, Rational, Symbol, pi
    >>> from sympy_extras.functions import exp_polar, lowergamma
    >>> from sympy_extras.integrals.marichev import polar_lowergamma
    >>> w = Symbol('w', positive=True)
    >>> polar_lowergamma(lowergamma(Rational(1, 3), w*exp_polar(I*pi)))
    3*(-1)**(1/3)*w**(1/3)*hyper((1/3,), (4/3,), w)
    """
    replacement: dict[Basic, Basic] = {}
    for node in value.atoms(LowerGamma):
        a, argument = _expr(node.args[0]), _expr(node.args[1])
        # SymPy's as_independent(exp_polar): the free factors first
        polar, w = _split_polar(argument)
        if _is_polar_minus_one(polar) and w.is_positive:
            replacement[node] = Integer(-1)**a * w**a * hyper([a], [a + 1], w) / a
        elif _is_polar_minus_one(w) and polar.is_positive:
            replacement[node] = Integer(-1)**a * polar**a * hyper([a], [a + 1], polar) / a
    if not replacement:
        return value
    return _xreplace(value, replacement)


def right_half_plane_powers(value: Expr, assumptions: Assumptions = None) -> Expr:
    """``(w**2)**p`` with ``w`` a non-real number of positive real part
    under the assumptions written ``w**(2*p)`` (``sqrt((a + I*b)**2)`` is
    ``a + I*b`` for ``a > 0``): the principal argument of ``w**2`` is
    twice the one of ``w`` when ``|arg w| < pi/2``.

    >>> from sympy_extras.core import I, sqrt, symbols
    >>> from sympy_extras.integrals.marichev import right_half_plane_powers
    >>> a, b = symbols('a b', positive=True)
    >>> right_half_plane_powers(sqrt((a + I*b)**2)/(a + I*b))
    1
    """
    replacement: dict[Basic, Basic] = {}
    for node in value.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        base, exponent = node.base, node.exp
        if not (isinstance(base, Pow) and base.exp == 2) or exponent.is_integer:
            continue
        w = base.base
        if w.is_extended_real or ask(element(w, Reals), assumptions):
            continue
        real_part = re(w)
        if real_part.has(Re, Im):
            continue
        if ask(real_part > 0, assumptions) is True:
            replacement[node] = w**(2 * exponent)
    if not replacement:
        return value
    return _xreplace(value, replacement)


def _order_derivative_at_zero(node: Basic) -> bool:
    """``Subs(Derivative(besseli(nu, z), nu), nu, 0)``."""
    if not isinstance(node, Subs) or node.point != (S.Zero,):
        return False
    derivative = node.expr
    if not isinstance(derivative, Derivative) or not isinstance(derivative.expr, BesselI):
        return False
    variables = derivative.variables
    return len(variables) == 1 and variables[0] == derivative.expr.args[0] and node.variables == (variables[0],)


def _minus_k0(node: Basic) -> Basic:
    """``-besselk(0, z)`` for ``Subs(Derivative(besseli(nu, z), nu), nu, 0)``."""
    if not isinstance(node, Subs) or not isinstance(node.expr, Derivative):
        return node
    return -besselk(0, _expr(node.expr.expr.args[1]))


def bessel_k_forms(value: Expr) -> Expr:
    """``besseli(-n, z)`` written as ``besseli(n, z) + 2*sin(pi*n)*besselk(n, z)/pi``
    where ``besseli(n, z)`` occurs too (the definition of the modified
    Bessel function of the second kind for a non-integer order), so that
    the ``sin(pi*n)`` of the Mellin method's value cancels: the value is then right at integer orders
    too, where the difference is `0/0` (the bug: the census's numerical
    check of `t**(n - 1)*exp(-a/t - s*t)` over `(0, oo)` divided by
    ``sin(pi*n)`` at an integer sample of ``n``).

    >>> from sympy_extras.core import pi, symbols
    >>> from sympy_extras.functions import besseli, sin
    >>> from sympy_extras.integrals.marichev import bessel_k_forms
    >>> n, z = symbols('n z')
    >>> bessel_k_forms(pi*(besseli(-n, z) - besseli(n, z))/sin(pi*n))
    2*besselk(n, z)
    """
    # the derivative in the order at 0, which the limit of the logarithmic
    # case of Slater's theorem leaves: d/dnu I_nu(z) at nu = 0 is -K_0(z)
    # (DLMF 10.38.6); exp(-k**2/(4*t) - s*t)/t over (0, oo) is 2*K_0(k*sqrt(s))
    value = _expr(value.replace(_order_derivative_at_zero, _minus_k0))
    nodes = [node for node in value.atoms(BesselI)]
    orders = {(_expr(node.args[0]), _expr(node.args[1])) for node in nodes}
    replacement: dict[Basic, Basic] = {}
    for node in nodes:
        order, argument = _expr(node.args[0]), _expr(node.args[1])
        if (-order, argument) in orders and order.could_extract_minus_sign():
            # I_{-nu} = I_nu + (2/pi) sin(nu pi) K_nu for the order written
            # with the minus sign
            replacement[node] = besseli(-order, argument) + 2 * sin(-pi * order) * besselk(-order, argument) / pi
    if not replacement:
        return value
    rewritten = cancel(_xreplace(value, replacement))
    simpler = attempt(lambda: simplify(gammasimp(rewritten)), settings.timeout, native=True)
    return rewritten if simpler is None else simpler


def real_logarithms(value: Expr, assumptions: Assumptions = None) -> Expr:
    """``log(u)`` with ``u`` negative under the assumptions written
    ``log(-u) + I*pi``, so that the imaginary parts of a real value cancel
    (the bug: ``I*c**2*(log(c**2) - log(-c**2))``, SymPy's value of the
    area of a disc of radius ``-c``, was left as it was and then refined
    to 0 under ``c < 0``).

    >>> from sympy_extras.core import I, Symbol
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.marichev import real_logarithms
    >>> c = Symbol('c')
    >>> real_logarithms(I*c**2*(log(c**2) - log(-c**2)), c < 0)
    pi*c**2
    """
    replacement: dict[Basic, Basic] = {}
    for node in value.atoms(Log):
        argument = _expr(node.args[0])
        if argument.is_extended_real is False:
            # a complex argument, 1 + 43*sqrt(771)*I/2313 (the crash: the
            # relation argument < 0 cannot be formed)
            continue
        if not argument.free_symbols:
            # a number: its sign numerically (the relation expands an
            # algebraic number with thousands of digits, and ran out of memory)
            numeric = attempt(lambda: evalf(argument, 50), settings.timeout, native=True)
            if (numeric is not None and isinstance(numeric, Float)
                    and isinstance(lt(numeric, Float(-1e-40)), BooleanTrue)):
                replacement[node] = log(-argument) + I * pi
            continue
        try:
            negative = ask(argument < 0, assumptions)
        except TypeError:
            continue
        if negative is True:
            replacement[node] = log(-argument) + I * pi
    if not replacement:
        return value
    rewritten = _xreplace(value, replacement)
    simpler = attempt(lambda: simplify(rewritten), settings.timeout, native=True)
    return rewritten if simpler is None else simpler


def _is_root_of_minus_one(node: Basic) -> bool:
    """``(-1)**q`` for a non-integer ``q``."""
    return isinstance(node, Pow) and node.base == -1 and node.exp.is_integer is not True


def _has_complex_unit(e: Expr) -> bool:
    return e.has(I) or bool(e.find(_is_root_of_minus_one))


def real_constant(value: Expr) -> Expr:
    """A constant written with ``I`` or roots of ``-1`` in real terms, by
    its real and imaginary parts simplified, when that removes them and
    is not longer; ``value`` itself otherwise (the Marichev–Adamchik value
    of ``cbrt(x)/(1 + x)`` over ``(0, 1)``, through the polar numbers of
    ``hyperexpand`` projected by :func:`projected_sums`, is
    ``-log(2) + 3 - (-1)**(2/3)*log(1 - (-1)**(1/3)) + (-1)**(1/3)*log(1 + (-1)**(2/3))``).

    >>> from sympy_extras.core import Integer, Rational
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.marichev import real_constant
    >>> w = Integer(-1)**Rational(1, 3)
    >>> real_constant(3 - log(2) - w**2*log(1 - w) + w*log(1 + w**2))
    -sqrt(3)*pi/3 - log(2) + 3
    """
    if value.free_symbols or not _has_complex_unit(value):
        return value
    candidate = simplify(expand_complex(value))
    if _has_complex_unit(candidate) or _size(candidate) > _size(value):
        return value
    return candidate


def _size(e: Expr) -> int:
    n = count_ops(e, visual=False)
    return int(n) if isinstance(n, Integer) else 0


def _one_kernel(product: Product, assumptions: Assumptions) -> Optional[ConditionalValue]:
    quotient = product.matches[0].quotient(assumptions)
    point = product.alpha + 1
    value = evaluate_quotient(quotient, point)
    if value is None:
        return None
    return ConditionalValue(product.constant * value, and_(quotient.strip_condition(point), quotient.condition))


def _two_kernels(product: Product, assumptions: Assumptions) -> Optional[ConditionalValue]:
    first = product.matches[0].quotient(assumptions)
    second = product.matches[1].quotient(assumptions)
    if not first.is_gamma_quotient() or not second.is_gamma_quotient():
        return None
    point = product.alpha + 1
    # Parseval: (1/2 pi i) Integral(F(t) G(alpha + 1 - t), t) on a line in both strips
    integrand = first.times(second.compose(point, -1)).cancelled()
    nonempty: Boolean = true
    if integrand.lower != -S.Infinity and integrand.upper != S.Infinity:
        nonempty = integrand.lower < integrand.upper
        if isinstance(nonempty, BooleanFalse):
            return None
    g = mellin_barnes(integrand)
    if g is None:
        return None
    line = line_conditions(g, integrand.lower, integrand.upper)
    if line is None:
        return None
    expanded = expand_meijerg(g, assumptions)
    if expanded is None:
        return None
    return ConditionalValue(product.constant * expanded.value,
                            and_(nonempty, integrand.condition, line, expanded.condition))


def integrate_product(product: Product, assumptions: Assumptions = None,
                      regularize: bool = False) -> Optional[ConditionalValue]:
    """``Integral(product, (x, 0, oo))`` for a decomposed integrand, see
    the module documentation; ``None`` when the method does not apply.
    With ``regularize`` the strip conditions are dropped (the analytic
    continuation of the value).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.mellin import decompose_integrand
    >>> from sympy_extras.integrals.marichev import integrate_product
    >>> x = Symbol('x')
    >>> integrate_product(decompose_integrand(x**2*exp(-x), x))
    ConditionalValue(2)
    """
    if not product.matches:
        return None
    if product.one_minus_power > 0:
        # log(1 - x)**m is the m-th derivative of (1 - x)**(b - 1) in b:
        # the Beta kernel's parameter made a symbol, the integral
        # differentiated, the parameter put back (t**2*(1 - t)**2*log(t)**2
        # *log(1 - t)**2 over (0, 1), Maxima's rtestint 206)
        beta = Dummy('beta', positive=True)
        matches: list[Match] = []
        parameter: Optional[Expr] = None
        for m in product.matches:
            if m.kernel.name == BETA_LOWER and parameter is None:
                parameter = m.kernel.parameters[0]
                matches.append(Match(_beta_kernel(beta), m.beta, m.gamma, m.constant, m.cutoff))
            else:
                matches.append(m)
        if parameter is None:
            return None
        shifted = Product(product.constant, product.alpha, product.log_power, matches)
        found = integrate_product(shifted, assumptions, regularize)
        if found is None:
            return None
        value = found.value
        for _ in range(product.one_minus_power):
            value = value.diff(beta)
        return ConditionalValue(value.subs(beta, parameter), to_boolean(found.condition.subs(beta, parameter)))
    if product.log_power > 0:
        alpha = Dummy('alpha', real=True)
        shifted = Product(product.constant, alpha, 0, product.matches, product.one_minus_power)
        found = integrate_product(shifted, assumptions, regularize)
        if found is None:
            return None
        value = found.value
        for _ in range(product.log_power):
            value = value.diff(alpha)
        return ConditionalValue(value.subs(alpha, product.alpha),
                                to_boolean(found.condition.subs(alpha, product.alpha)))
    if len(product.matches) == 1:
        found = _one_kernel(product, assumptions)
    else:
        found = _two_kernels(product, assumptions)
    if found is None or not regularize:
        return found
    # keep the conditions on the parameters of the kernels, drop the strips
    kept = [m.quotient(assumptions).condition for m in product.matches]
    return ConditionalValue(found.value, and_(*kept))


def exponential_form(term: Expr, x: Symbol) -> Expr:
    """The trigonometric and hyperbolic factors of ``term`` written as
    exponentials, so that a product of three kernels (``exp(-a x) sin(b x)
    J_0(c x)``) becomes a sum of products of two (the exponential with
    the complex scale ``a - I b``, whose condition ``Re(a - I b) > 0`` is
    decided by the real part). The exponentials of one term are combined
    into one (``exp(-a x)*exp(I b x)`` is ``exp((-a + I b) x)``).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import cosh, exp
    >>> from sympy_extras.integrals.marichev import exponential_form
    >>> x, a = symbols('x a')
    >>> exponential_form(exp(-x)*cosh(a*x), x)
    exp(x*(a - 1))/2 + exp(-x*(a + 1))/2
    """
    factors = list(Mul.make_args(term))
    sums: list[Expr] = []
    others: list[Expr] = []
    for piece in factors:
        if piece.has(x) and isinstance(piece, (Sin, Cos, Sinh, Cosh)):
            sums.append(_expr(piece.rewrite(Exp)))
        else:
            others.append(piece)
    if not sums:
        return term
    rest = mul(*others)
    summands = Add.make_args(mul(*sums).expand())
    return add(*[_combine_linear_exponentials(u * rest, x) for u in summands])


def _combine_linear_exponentials(term: Expr, x: Symbol) -> Expr:
    """The exponentials of ``term`` whose arguments are linear in ``x``
    combined into one (``exp(-a x)*exp(I b x)`` is ``exp((-a + I b) x)``),
    the others (``exp(-x**2)``, a kernel of its own) left alone.

    >>> from sympy_extras.core import I, symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.marichev import _combine_linear_exponentials
    >>> x, a, b = symbols('x a b')
    >>> _combine_linear_exponentials(exp(-a*x)*exp(I*b*x)*exp(-x**2), x)
    exp(-x**2)*exp(x*(-a + I*b))
    """
    linear: list[Expr] = []
    others: list[Expr] = []
    for factor_ in Mul.make_args(term):
        if isinstance(factor_, Exp) and factor_.has(x) and _expr(factor_.args[0]).diff(x).has(x) is False:
            linear.append(_expr(factor_.args[0]))
        else:
            others.append(factor_)
    if len(linear) < 2:
        return term
    argument = factor_terms(add(*linear))
    return mul(*others) * exp(argument)


def reduce_positive_powers(f: Expr, x: Symbol) -> Expr:
    """``(1 + b*x**g)**e`` with a positive non-integer ``e``, which has no
    Mellin transform, written as ``(1 + b*x**g)**(e - n) * (1 + b*x**g)**n``
    with ``n = ceiling(e)`` and the polynomial expanded, so that every
    term has the kernel ``(1 + b*x**g)**(e - n)`` of the table.

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.integrals.marichev import reduce_positive_powers
    >>> x = Symbol('x')
    >>> reduce_positive_powers(sqrt(1 + x), x)
    x/sqrt(x + 1) + 1/sqrt(x + 1)
    """
    def rewrite(node: Expr) -> Expr:
        if not isinstance(node, Pow) or not isinstance(node.base, Add):
            return node
        exponent = node.exp
        if not exponent.is_positive or exponent.is_integer is not False or exponent.has(x):
            return node
        constant, rest = node.base.as_independent(x, as_Add=True)
        if constant == 0 or monomial(rest, x) is None:
            return node
        n = ceiling(exponent)
        if not isinstance(n, Integer):
            return node
        expanded = (node.base**n).expand()
        # term by term, so that the powers of the base are not merged back
        return add(*[term * node.base**(exponent - n) for term in Add.make_args(expanded)])

    result = f
    for node in f.atoms(Pow):
        replacement = rewrite(_expr(node))
        if replacement != node:
            result = _xreplace(result, {node: replacement})
    if result == f:
        return f
    return result.expand(mul=True, multinomial=False, power_exp=False, log=False)


def mellin_integrate(f: ExprLike, x: Symbol, assumptions: Assumptions = None,
                     cutoff: Optional[str] = None, regularize: bool = False) -> Optional[ConditionalValue]:
    """``Integral(f, (x, 0, oo))`` by the Marichev–Adamchik method, or
    over ``(0, 1)`` with ``cutoff='lower'`` and over ``(1, oo)`` with
    ``cutoff='upper'``.

    Parameters
    ==========

    f : Expr
        The integrand: a sum of terms, each a constant times a power of
        ``x``, a power of ``log(x)`` and at most two kernels of the
        table of :mod:`.mellin` with arguments ``beta * x**gamma``.
    x : Symbol
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters, used to decide the conditions and
        to simplify the result.
    cutoff : ``'lower'``, ``'upper'`` or None
    regularize : bool
        Analytic (Riesz) regularisation: the value of the formula, which
        is analytic in the exponents, is returned beyond the strips of
        convergence too (the integral diverges there, and the value is
        its analytic continuation, e.g. ``Integral(x**(-3/2)*exp(-x),
        (x, 0, oo))`` is ``gamma(-1/2)``); the conditions on the
        parameters of the kernels are kept.

    Returns
    =======

    A :class:`~.conditions.ConditionalValue` with the value and the
    condition on the parameters (what the assumptions do not settle), or
    ``None`` when a term of the integrand is not of the form above or
    the conditions are refuted by the assumptions.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import besselk, cos, exp, log
    >>> from sympy_extras.integrals.marichev import mellin_integrate
    >>> x, s = symbols('x s')
    >>> a, nu = symbols('a nu', positive=True)
    >>> mellin_integrate(exp(-s*x)*cos(a*x), x, s > 0)
    ConditionalValue(s/(a**2 + s**2))
    >>> mellin_integrate(x**(2*nu)*besselk(nu, x), x)
    ConditionalValue(2**(2*nu)*gamma((nu + 1)/2)*gamma((3*nu + 1)/2)/2)
    >>> mellin_integrate(log(x)**2, x, cutoff='lower')
    ConditionalValue(2)
    """
    terms = list(Add.make_args(reduce_positive_powers(to_expr(f), x)))
    total: Optional[ConditionalValue] = None
    for term in terms:
        pieces = [term]
        if decompose_integrand(term, x, cutoff) is None:
            # three kernels with trigonometric or hyperbolic factors:
            # exponentials with complex scales, two kernels per piece
            pieces = list(Add.make_args(exponential_form(term, x)))
        for piece in pieces:
            product = decompose_integrand(piece, x, cutoff)
            if product is None:
                return None
            found = integrate_product(product, assumptions, regularize)
            if found is None:
                return None
            total = found if total is None else total.add(found)
    if total is None:
        return None
    condition = decide(total.condition, assumptions)
    if condition is None:
        return None
    value = tidy(total.value, assumptions, condition)
    if value.has(ExpPolar, PolarLift, PrincipalBranch, Limit):
        # a branch which could not be resolved: not an answer
        return None
    return ConditionalValue(value, condition)

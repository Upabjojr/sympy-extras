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
assumptions by the caller (:func:`~sympy_extras.integrals.definite_integral`).

Examples
========

>>> from sympy import symbols, exp, sin, besselj, sqrt, log, oo
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

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import nan, oo, zoo
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import polar_lift, unpolarify, principal_branch
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.special.gamma_functions import polygamma
from sympy.functions.special.zeta_functions import lerchphi, dirichlet_eta
from sympy.functions.special.hyper import meijerg
from sympy.core.function import expand_func
from sympy.core.power import Pow
from sympy.core.numbers import Integer
from sympy.functions.elementary.integers import ceiling
from .mellin import monomial
from sympy.series.limits import Limit, limit
from sympy.simplify.gammasimp import gammasimp
from sympy.polys.polytools import factor
from sympy.core.exprtools import factor_terms
from sympy.simplify.simplify import simplify
from sympy.logic.boolalg import And, Boolean, true

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.refine import refine
from sympy_extras.settings import settings
from .conditions import ConditionalValue, decide
from .mellin import GammaQuotient, Product, decompose_integrand
from .slater import expand_meijerg, line_conditions, mellin_barnes

__all__ = ['mellin_integrate', 'integrate_product', 'evaluate_quotient', 'tidy']


def evaluate_quotient(quotient: GammaQuotient, point: Expr) -> Optional[Expr]:
    """The value of the quotient at ``point``, by a limit when a pole of
    the numerator meets a zero of the denominator (``sin(x)/x`` at
    ``s = 0``); ``None`` when SymPy cannot take the limit."""
    value = quotient.as_expr(point)
    if not value.has(nan, zoo, oo):
        return value
    s = Dummy('s')
    found = attempt(lambda: as_expr(limit(quotient.as_expr(s), s, point)), settings.timeout)
    if found is None or found.has(nan, zoo, oo, Limit):
        return None
    return found


def tidy(value: Expr, assumptions: Assumptions = None, condition: Boolean = true) -> Expr:
    """The value with polar numbers removed and simplified, under the
    time limit of the settings; ``refine`` uses the assumptions and the
    condition under which the value holds."""
    result = value
    if result.has(exp_polar, polar_lift, principal_branch):
        unpolar = attempt(lambda: as_expr(unpolarify(result)), settings.timeout)
        if unpolar is not None:
            result = unpolar
    if result.has(meijerg):
        return result
    if result.has(polygamma):
        expanded = attempt(lambda: as_expr(expand_func(result)), settings.timeout)
        if expanded is not None and _size(expanded) < _size(result):
            result = expanded
    if result.has(lerchphi, dirichlet_eta):
        # the Dirichlet functions at integers are logarithms and constants
        expanded = attempt(lambda: as_expr(simplify(expand_func(result))), settings.timeout)
        if expanded is not None and not expanded.has(lerchphi, dirichlet_eta):
            result = expanded
    simpler = attempt(lambda: as_expr(simplify(gammasimp(result))), settings.timeout)
    if simpler is not None and _size(simpler) <= _size(result):
        result = simpler
    factored = attempt(lambda: as_expr(factor_terms(factor(result))), settings.timeout)
    if factored is not None and _size(factored) < _size(result):
        result = factored
    facts: list[Boolean] = [] if condition is true else [condition]
    if isinstance(assumptions, (Basic, bool)):
        facts.append(as_boolean(assumptions))
    elif assumptions is not None:
        facts.extend(as_boolean(a) for a in assumptions)
    if facts:
        refined = attempt(lambda: as_expr(refine(result, facts)), settings.timeout)
        if refined is not None and _size(refined) <= _size(result):
            result = refined
    return result


def _size(e: Expr) -> int:
    return int(e.count_ops(visual=False)) if isinstance(e, Expr) else 0


def _one_kernel(product: Product, assumptions: Assumptions) -> Optional[ConditionalValue]:
    quotient = product.matches[0].quotient(assumptions)
    point = as_expr(product.alpha + 1)
    value = evaluate_quotient(quotient, point)
    if value is None:
        return None
    return ConditionalValue(product.constant * value,
                            And(quotient.strip_condition(point), quotient.condition))


def _two_kernels(product: Product, assumptions: Assumptions) -> Optional[ConditionalValue]:
    first = product.matches[0].quotient(assumptions)
    second = product.matches[1].quotient(assumptions)
    if not first.is_gamma_quotient() or not second.is_gamma_quotient():
        return None
    point = as_expr(product.alpha + 1)
    # Parseval: (1/2 pi i) Integral(F(t) G(alpha + 1 - t), t) on a line in both strips
    integrand = first.times(second.compose(point, -1)).cancelled()
    if integrand.lower != -oo and integrand.upper != oo:
        nonempty: Boolean = as_boolean(integrand.lower < integrand.upper)
        if nonempty is S.false:
            return None
    else:
        nonempty = true
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
                            And(nonempty, integrand.condition, line, expanded.condition))


def integrate_product(product: Product, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(product, (x, 0, oo))`` for a decomposed integrand, see
    the module documentation; ``None`` when the method does not apply."""
    if not product.matches:
        return None
    if product.log_power > 0:
        alpha = Dummy('alpha', real=True)
        shifted = Product(product.constant, alpha, 0, product.matches)
        found = integrate_product(shifted, assumptions)
        if found is None:
            return None
        value = found.value
        for _ in range(product.log_power):
            value = as_expr(value.diff(alpha))
        return ConditionalValue(value.subs(alpha, product.alpha),
                                as_boolean(found.condition.subs(alpha, product.alpha)))
    if len(product.matches) == 1:
        return _one_kernel(product, assumptions)
    return _two_kernels(product, assumptions)


def reduce_positive_powers(f: Expr, x: Symbol) -> Expr:
    """``(1 + b*x**g)**e`` with a positive non-integer ``e``, which has no
    Mellin transform, written as ``(1 + b*x**g)**(e - n) * (1 + b*x**g)**n``
    with ``n = ceiling(e)`` and the polynomial expanded, so that every
    term has the kernel ``(1 + b*x**g)**(e - n)`` of the table.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.marichev import reduce_positive_powers
    >>> x = symbols('x')
    >>> reduce_positive_powers(sqrt(1 + x), x)
    x/sqrt(x + 1) + 1/sqrt(x + 1)
    """
    def rewrite(node: Expr) -> Expr:
        if not isinstance(node, Pow) or not isinstance(node.base, Add):
            return node
        exponent = as_expr(node.exp)
        if not exponent.is_positive or exponent.is_integer is not False or exponent.has(x):
            return node
        constant, rest = node.base.as_independent(x, as_Add=True)
        if as_expr(constant) == 0 or monomial(as_expr(rest), x) is None:
            return node
        n = as_expr(ceiling(exponent))
        if not isinstance(n, Integer):
            return node
        expanded = as_expr((as_expr(node.base)**n).expand())
        # term by term, so that the powers of the base are not merged back
        return as_expr(Add(*[as_expr(term) * node.base**(exponent - n) for term in Add.make_args(expanded)]))

    result = f
    for node in f.atoms(Pow):
        replacement = rewrite(as_expr(node))
        if replacement != node:
            result = as_expr(result.xreplace({node: replacement}))
    if result == f:
        return f
    return as_expr(result.expand(mul=True, multinomial=False, power_exp=False, log=False))


def mellin_integrate(f: ExprLike, x: Symbol, assumptions: Assumptions = None,
                     cutoff: Optional[str] = None) -> Optional[ConditionalValue]:
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

    Returns
    =======

    A :class:`~.conditions.ConditionalValue` with the value and the
    condition on the parameters (what the assumptions do not settle), or
    ``None`` when a term of the integrand is not of the form above or
    the conditions are refuted by the assumptions.

    Examples
    ========

    >>> from sympy import symbols, exp, cos, besselk, log
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
    terms = [as_expr(t) for t in Add.make_args(reduce_positive_powers(as_expr(f), x))]
    total: Optional[ConditionalValue] = None
    for term in terms:
        product = decompose_integrand(term, x, cutoff)
        if product is None:
            return None
        found = integrate_product(product, assumptions)
        if found is None:
            return None
        total = found if total is None else total.add(found)
    if total is None:
        return None
    condition = decide(total.condition, assumptions)
    if condition is None:
        return None
    value = tidy(total.value, assumptions, condition)
    if value.has(exp_polar, polar_lift, principal_branch, Limit):
        # a branch which could not be resolved: not an answer
        return None
    return ConditionalValue(value, condition)

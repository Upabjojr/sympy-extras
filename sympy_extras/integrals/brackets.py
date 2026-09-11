"""Ramanujan's master theorem and the method of brackets: Mellin
transforms from power series.

If `f(x) = \\sum_{k \\ge 0} \\varphi(k) (-x)^k / k!`, **Ramanujan's master
theorem** states that

.. math::

    \\int_0^\\infty x^{s-1} f(x)\\, dx = \\Gamma(s)\\, \\varphi(-s),

where `\\varphi(-s)` is the analytic continuation of the coefficient
function to `k = -s` [Hardy]_, [Amdeberhan]_. Hardy's proof needs growth
conditions on `\\varphi`; in practice the formula holds for `s` in the
strip where the integral converges, and this is what is returned: the
strip is read off the behaviour of `f` at `0` and at `\\infty`.

The **method of brackets** [Gonzalez]_ extends this to products. A series
`\\sum_n \\varphi(n) (-1)^n/n!\\, x^{a n + b}` integrated against `x^{s-1}`
on `(0, \\infty)` is written with the *bracket* `\\langle a n + b + s
\\rangle`, and the rule for one bracket replaces the sum by
`\\varphi(n^*) \\Gamma(-n^*) / |a|` with `n^*` the solution of `a n + b + s =
0`: this is the master theorem again. For a product of two series there
are two indices and one bracket; eliminating one index leaves a series
in the other, which is hypergeometric and summed in closed form. Which
index to eliminate is decided by convergence: the rule of the method is
to keep the series which converge, and the method is a heuristic --
its rules are not proved in general, and a result is only as good as
the check made of it. Here the series left after the bracket is summed
exactly by SymPy (as a hypergeometric function, expanded when
possible), so the value is right whenever the summation converges.

The coefficient functions come from SymPy's formal power series
(:func:`sympy.series.formal.fps`), which gives a closed form of the
`k`-th coefficient with factorials and rising factorials, rewritten with
the gamma function so that `k` can be replaced by `-s`.

Examples
========

>>> from sympy import symbols, exp, sin, atan, besselj
>>> from sympy_extras.integrals.brackets import ramanujan_master_theorem, mellin_transform_series
>>> x, s = symbols('x s')
>>> ramanujan_master_theorem(exp(-x), x, s)
MellinTransform(gamma(s), (0, oo))
>>> ramanujan_master_theorem(atan(x), x, s)
MellinTransform(-pi/(2*s*cos(pi*s/2)), (-1, 0))
>>> mellin_transform_series(exp(-x)*sin(x), x, s)
MellinTransform(2**(s + 1)*gamma(s/4 + 1/2)*gamma(s/2 + 1/2)/(8*gamma(1 - s/4)), (-1, oo))

References
==========

.. [Hardy] G. H. Hardy, *Ramanujan: Twelve lectures on subjects
   suggested by his life and work*, Cambridge University Press, 1940,
   chapter XI.
.. [Amdeberhan] T. Amdeberhan, O. Espinosa, I. Gonzalez, M. Harrison,
   V. H. Moll, A. Straub, *Ramanujan's Master Theorem*, The Ramanujan
   Journal 29 (2012), pp. 103-120.
.. [Gonzalez] I. Gonzalez, V. H. Moll, *Definite integrals by the method
   of brackets. Part 1*, Advances in Applied Mathematics 45 (2010),
   pp. 50-73.
"""
from __future__ import annotations

from typing import Optional

from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, PoleError
from sympy.series.limits import limit
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, oo
from sympy.core.power import Pow
from sympy.core.relational import Equality
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.integrals.transforms import mellin_transform as sympy_mellin, MellinTransform as SympyMellin
from sympy.logic.boolalg import true
from sympy.series.formal import fps, FormalPowerSeries
from sympy.series.order import Order
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.powsimp import powsimp
from sympy.simplify.simplify import simplify

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.settings import settings
from .mellin import MellinTransform, monomial

__all__ = ['SeriesCoefficient', 'taylor_coefficient', 'ramanujan_master_theorem', 'method_of_brackets',
           'mellin_transform_series']


class SeriesCoefficient:
    """The power series of a function about `0` in the form
    `f(x) = x^r \\sum_{j \\ge 0} c(j)\\, (x^m)^j`.

    Attributes
    ==========

    coefficient : Expr
        `c(j)`, a closed form in the index.
    index : Dummy
        The index `j` (a nonnegative integer).
    period : int
        `m`, the step of the exponents (2 for an even or odd function).
    offset : Expr
        `r`, the exponent of the first term.
    """

    def __init__(self, coefficient: Expr, index: Dummy, period: int, offset: Expr) -> None:
        self.coefficient = coefficient
        self.index = index
        self.period = period
        self.offset = offset

    def __repr__(self) -> str:
        return "SeriesCoefficient(%s, %s, period=%d, offset=%s)" % (self.coefficient, self.index,
                                                                    self.period, self.offset)

    def phi(self) -> Optional[Expr]:
        """`\\varphi(j) = (-1)^j j!\\, c(j)` of the master theorem, with the
        gamma function in place of the factorials; ``None`` when a factor
        `(-1)^j` remains (no Mellin transform: the series has positive
        coefficients growing like those of `e^x`)."""
        j = self.index
        value = as_expr(powsimp(as_expr(S.NegativeOne**j * factorial(j) * self.coefficient), combine='exp'))
        value = as_expr(value.rewrite(gamma))
        value = as_expr(gammasimp(value))
        if value.has(S.NegativeOne**j) or _has_sign_power(value, j):
            return None
        return value


def _has_sign_power(e: Expr, j: Dummy) -> bool:
    """Whether ``e`` still has a power of a negative number with the
    index in the exponent."""
    for node in e.atoms(Pow):
        base = as_expr(node.base)
        if base.is_negative and node.exp.has(j):
            return True
    return False


def _residue_class(formula: Expr, k: Symbol) -> Optional[tuple[Expr, int, int]]:
    """A ``Piecewise`` coefficient of ``fps`` read as ``(value, m, r)``:
    ``value`` for ``k = r mod m`` and zero otherwise; a plain formula is
    ``(formula, 1, 0)``. ``None`` when two residue classes are nonzero."""
    if not isinstance(formula, Piecewise):
        return (formula, 1, 0)
    found: Optional[tuple[Expr, int, int]] = None
    for pair in formula.args:
        value, condition = as_expr(pair.args[0]), as_boolean(pair.args[1])
        if value == 0:
            continue
        if condition is true and found is None:
            return (value, 1, 0)
        if not isinstance(condition, Equality):
            return None
        lhs, rhs = as_expr(condition.lhs), as_expr(condition.rhs)
        if not (lhs.func.__name__ == 'Mod' and len(lhs.args) == 2 and lhs.args[0] == k):
            return None
        modulus, remainder = as_expr(lhs.args[1]), rhs
        if not isinstance(modulus, Integer) or not isinstance(remainder, Integer):
            return None
        if found is not None:
            return None
        found = (value, int(modulus), int(remainder))
    return found


def taylor_coefficient(f: Expr, x: Symbol) -> Optional[SeriesCoefficient]:
    """The power series of ``f`` about 0 as a :class:`SeriesCoefficient`,
    from SymPy's formal power series; ``None`` when there is none or its
    coefficients have no single closed form.

    Examples
    ========

    >>> from sympy import symbols, exp, sin
    >>> from sympy_extras.integrals.brackets import taylor_coefficient
    >>> x = symbols('x')
    >>> c = taylor_coefficient(exp(-x), x)
    >>> c.coefficient.subs(c.index, 3), c.period, c.offset
    (-1/6, 1, 0)
    >>> c = taylor_coefficient(sin(x), x)
    >>> c.period, c.offset, c.coefficient.subs(c.index, 1)
    (2, 1, -1/6)
    """
    if f.is_polynomial(x):
        return None
    series = attempt(lambda: fps(f, x), settings.timeout)
    if not isinstance(series, FormalPowerSeries):
        return None
    k = series.ak.variables[0]
    if not isinstance(k, Symbol):
        return None
    formula = as_expr(series.ak.formula)
    found = _residue_class(formula, k)
    if found is None:
        return None
    value, m, r = found
    # the independent terms must agree with the formula
    independent = as_expr(series.ind)
    for term in Add.make_args(independent):
        if term == 0:
            continue
        coefficient, rest = as_expr(term).as_independent(x, as_Add=False)
        exponent = monomial(as_expr(rest), x)
        if exponent is None:
            if as_expr(rest) == 1:
                power: Expr = S.Zero
            else:
                return None
        else:
            power = exponent[1]
        if not isinstance(power, Integer) or (int(power) - r) % m != 0:
            return None
        if simplify(as_expr(value.subs(k, power)) - as_expr(coefficient)) != 0:
            return None
    j = Dummy('j', integer=True, nonnegative=True)
    try:
        coefficient = as_expr(value.subs(k, m * j + r))
    except (ValueError, TypeError):
        return None
    return SeriesCoefficient(coefficient, j, m, Integer(r))


def _order_at_infinity(f: Expr, x: Symbol) -> Optional[Expr]:
    """``e`` such that ``f ~ C x**(-e)`` as ``x -> oo`` (the exponent of
    the leading term), from the series of ``f(1/y)`` at ``y = 0``; ``oo``
    when ``f`` decays faster than every power (an exponential)."""
    y = Dummy('y', positive=True)
    g = as_expr(f.subs(x, 1 / y))
    try:
        leading = attempt(lambda: g.series(y, 0, 1).removeO(), settings.timeout)
    except PoleError:
        # an essential singularity at infinity: exponential decay or oscillation
        decays = attempt(lambda: as_expr(limit(f * x**50, x, oo)), settings.timeout)
        if decays is not None and decays == 0:
            return oo
        return None
    if leading is None or leading == 0:
        # the leading term is beyond the first order: ask for the leading term itself
        try:
            found = attempt(lambda: g.leadterm(y), settings.timeout)
        except PoleError:
            return None
        if found is None:
            return None
        coefficient, exponent = found
        return as_expr(exponent)
    if isinstance(leading, Order):
        return None
    terms = Add.make_args(as_expr(leading))
    exponents: list[Expr] = []
    for term in terms:
        found_ = monomial(as_expr(term), x) if as_expr(term).has(x) else None
        if found_ is not None:
            return None
        coefficient, rest = as_expr(term).as_independent(y, as_Add=False)
        if not as_expr(rest).has(y):
            exponents.append(S.Zero)
            continue
        mono = monomial(as_expr(rest), y)
        if mono is None:
            return None
        exponents.append(mono[1])
    if not exponents:
        return None
    return min(exponents, key=lambda e: float(e))


def _strip(f: Expr, x: Symbol, s: Symbol, offset: Expr) -> Optional[tuple[Expr, Expr]]:
    """The strip of convergence: ``-offset < Re s < e`` with ``e`` the
    order of decay at infinity; SymPy's own strip when it knows one and
    the decay cannot be found."""
    upper = _order_at_infinity(f, x)
    if upper is None or upper.has(x):
        known = attempt(lambda: sympy_mellin(f, x, s), settings.timeout)
        if isinstance(known, tuple) and len(known) == 3:
            strip = known[1]
            return (as_expr(strip[0]), as_expr(strip[1]))
        if isinstance(known, SympyMellin) or known is None:
            return None
        return None
    return (-offset, upper)


def ramanujan_master_theorem(f: Expr, x: Symbol, s: Symbol,
                             assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """The Mellin transform of ``f`` by Ramanujan's master theorem, with
    the strip in which the integral converges; ``None`` when ``f`` has no
    usable power series or no decay at infinity.

    Examples
    ========

    >>> from sympy import symbols, exp, log, cos
    >>> from sympy_extras.integrals.brackets import ramanujan_master_theorem
    >>> x, s = symbols('x s')
    >>> ramanujan_master_theorem(1/(1 + x), x, s)
    MellinTransform(pi/sin(pi*s), (0, 1))
    >>> ramanujan_master_theorem(exp(-x**2), x, s)
    MellinTransform(gamma(s/2)/2, (0, oo))
    >>> ramanujan_master_theorem(exp(x), x, s) is None
    True
    """
    c = taylor_coefficient(f, x)
    if c is None:
        return None
    phi = c.phi()
    if phi is None:
        return None
    # f = x^r g(x^m): M[f](s) = M[g]((s + r)/m) / m
    sigma = as_expr((s + c.offset) / c.period)
    value = as_expr(gamma(sigma) * phi.subs(c.index, -sigma) / c.period)
    value = as_expr(gammasimp(value))
    strip = _strip(f, x, s, c.offset)
    if strip is None:
        return None
    lower, upper = strip
    if upper != oo and lower != -oo and (upper - lower).is_positive is False:
        return None
    return MellinTransform(value, (lower, upper))


def _factors(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, list[tuple[Expr, Expr, Expr]]]]:
    """``f`` as ``constant * x**alpha * prod g_i(beta_i x**gamma_i)``:
    the constant, ``alpha`` and the list of ``(g_i(y), beta_i, gamma_i)``
    with ``y`` a fresh symbol per factor."""
    constant: Expr = S.One
    alpha: Expr = S.Zero
    factors: list[tuple[Expr, Expr, Expr]] = []
    for factor in Mul.make_args(f):
        e = as_expr(factor)
        if not e.has(x):
            constant = constant * e
            continue
        mono = monomial(e, x)
        if mono is not None:
            constant = constant * mono[0]
            alpha = alpha + mono[1]
            continue
        argument: Optional[Expr] = None
        if isinstance(e, Function) and len(e.args) >= 1:
            argument = as_expr(e.args[-1])
        elif isinstance(e, Pow) and isinstance(e.base, Add):
            argument = as_expr(e.base)
        if argument is None:
            return None
        inner_constant, rest = argument.as_independent(x, as_Add=True)
        mono = monomial(as_expr(rest), x)
        if mono is None:
            return None
        beta, g_ = mono
        # a positive scale, so that g is read on (0, oo)
        if beta.is_negative:
            beta = -beta
        y = Dummy('y', positive=True)
        factors.append((as_expr(e.subs(x, (y / beta)**(1 / g_))), beta, g_))
    return constant, alpha, factors


def _scaled(transform: MellinTransform, beta: Expr, g_: Expr, s: Symbol) -> MellinTransform:
    """The transform of ``g(beta x**gamma)`` from that of ``g``."""
    value = as_expr(transform.transform.subs(s, s / g_) * beta**(-s / g_) / Abs(g_))
    lower, upper = transform.strip
    if g_ > 0:
        return MellinTransform(value, (lower * g_, upper * g_), transform.condition)
    return MellinTransform(value, (upper * g_, lower * g_), transform.condition)


def _shifted(transform: MellinTransform, alpha: Expr, constant: Expr, s: Symbol) -> MellinTransform:
    value = as_expr(constant * transform.transform.subs(s, s + alpha))
    lower, upper = transform.strip
    return MellinTransform(value, (lower - alpha, upper - alpha), transform.condition)


def method_of_brackets(f: Expr, x: Symbol, s: Symbol, assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """The Mellin transform of a product of two series-expandable factors
    by the method of brackets: one index is eliminated by the bracket
    rule and the remaining series is summed by SymPy as a hypergeometric
    function, which is expanded when possible. Both eliminations are
    tried and the one whose series converges is kept. The strip is the
    intersection of the strips of the factors' own transforms.

    Examples
    ========

    >>> from sympy import symbols, exp, sin, besselj
    >>> from sympy_extras.integrals.brackets import method_of_brackets
    >>> x, s = symbols('x s')
    >>> method_of_brackets(exp(-x)*sin(x), x, s)
    MellinTransform(2**(s + 1)*gamma(s/4 + 1/2)*gamma(s/2 + 1/2)/(8*gamma(1 - s/4)), (-1, oo))
    """
    decomposed = _factors(f, x)
    if decomposed is None or len(decomposed[2]) != 2:
        return None
    constant, alpha, factors = decomposed
    series: list[tuple[SeriesCoefficient, Expr, Expr, Expr]] = []
    strips: list[tuple[Expr, Expr]] = []
    for g, beta, g_ in factors:
        variable = _variable(g)
        if variable is None:
            return None
        c = taylor_coefficient(g, variable)
        if c is None or c.phi() is None:
            return None
        single = ramanujan_master_theorem(g, variable, s)
        if single is None:
            return None
        scaled = _scaled(single, beta, g_, s)
        series.append((c, beta, g_, variable))
        strips.append(scaled.strip)
    lower = _max(strips[0][0], strips[1][0])
    upper = _min(strips[0][1], strips[1][1])
    if lower != -oo and upper != oo and (upper - lower).is_positive is False:
        return None
    # the double series: sum_{n, k} phi1(n) (-1)^n/n! phi2(k) (-1)^k/k! beta1^(...) beta2^(...)
    # x^(g1 (m1 n + r1) + g2 (m2 k + r2) + alpha); the bracket <A n + B k + C + s>
    (c1, beta1, g1, _), (c2, beta2, g2, _) = series
    phi1, phi2 = c1.phi(), c2.phi()
    if phi1 is None or phi2 is None:
        return None
    for first, second in ((0, 1), (1, 0)):
        ca, cb = (c1, c2) if first == 0 else (c2, c1)
        phia, phib = (phi1, phi2) if first == 0 else (phi2, phi1)
        betaa, betab = (beta1, beta2) if first == 0 else (beta2, beta1)
        ga, gb = (g1, g2) if first == 0 else (g2, g1)
        n = ca.index
        # the exponent of x: ga (ma n + ra) + gb (mb k + rb) + alpha; eliminate k
        A = ga * ca.period
        B = gb * cb.period
        C = ga * ca.offset + gb * cb.offset + alpha
        k_star = as_expr(-(A * n + C + s) / B)
        term = (phia * S.NegativeOne**n / factorial(n) * betaa**(ca.period * n + ca.offset)
                * as_expr(phib.subs(cb.index, k_star)) * betab**(cb.period * k_star + cb.offset)
                * gamma(-k_star) / Abs(B))
        term = as_expr(gammasimp(as_expr(term)))
        total = attempt(lambda: as_expr(Sum(term, (n, 0, oo)).doit()), settings.timeout)
        if total is None:
            continue
        if isinstance(total, Piecewise):
            # the closed form where the series converges, continued
            # analytically to the strip (the bracket rule keeps it)
            total = _closed_branch(total)
            if total is None:
                continue
        if total.has(Sum):
            continue
        if total.has(hyper) and not _converges(total, assumptions):
            continue
        expanded = attempt(lambda: as_expr(hyperexpand(total)), settings.timeout)
        value = as_expr(gammasimp(constant * (expanded if expanded is not None else total)))
        decay = _order_at_infinity(f, x)
        if decay is not None and not decay.has(x):
            # the true strip of the product: from its order at 0 and its decay
            lower = -(g1 * c1.offset + g2 * c2.offset + alpha)
            upper = decay - alpha
            return MellinTransform(value, (lower, upper))
        return MellinTransform(value, (lower - alpha, upper - alpha))
    return None


def _closed_branch(total: Piecewise) -> Optional[Expr]:
    """The first branch of a ``Piecewise`` returned by ``Sum.doit`` whose
    other branch is the unevaluated sum."""
    for pair in total.args:
        value = as_expr(pair.args[0])
        if not value.has(Sum):
            return value
    return None


def _variable(g: Expr) -> Optional[Symbol]:
    symbols = [v for v in g.free_symbols if isinstance(v, Dummy)]
    return symbols[0] if len(symbols) == 1 else None


def _converges(e: Expr, assumptions: Assumptions) -> bool:
    """Whether every hypergeometric series in ``e`` converges: ``p <= q``,
    or ``p = q + 1`` with ``|z| < 1`` (decided with the assumptions)."""
    for node in e.atoms(hyper):
        p, q = len(node.ap), len(node.bq)
        if p <= q:
            continue
        if p == q + 1:
            z = as_expr(node.argument)
            if ask(as_boolean(Abs(z) < 1), assumptions) is True:
                continue
        return False
    return True


def _max(a: Expr, b: Expr) -> Expr:
    if a == -oo:
        return b
    if b == -oo:
        return a
    d = as_expr(a - b)
    if d.is_nonnegative:
        return a
    if d.is_negative:
        return b
    from sympy.functions.elementary.miscellaneous import Max
    return as_expr(Max(a, b))


def _min(a: Expr, b: Expr) -> Expr:
    if a == oo:
        return b
    if b == oo:
        return a
    d = as_expr(a - b)
    if d.is_nonpositive:
        return a
    if d.is_positive:
        return b
    from sympy.functions.elementary.miscellaneous import Min
    return as_expr(Min(a, b))


def mellin_transform_series(f: Expr, x: Symbol, s: Symbol, assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """The Mellin transform of ``f`` from its power series: Ramanujan's
    master theorem for one factor (times a constant and a power of ``x``),
    the method of brackets for a product of two; ``None`` otherwise.

    Examples
    ========

    >>> from sympy import symbols, exp, cos, sqrt
    >>> from sympy_extras.integrals.brackets import mellin_transform_series
    >>> x, s = symbols('x s')
    >>> mellin_transform_series(x**2*exp(-3*x), x, s)
    MellinTransform(3**(-s - 2)*gamma(s + 2), (-2, oo))
    >>> mellin_transform_series(cos(sqrt(x)), x, s)
    MellinTransform(2**(2*s)*sqrt(pi)*gamma(s)/gamma(1/2 - s), (0, 1/2))
    """
    decomposed = _factors(f, x)
    if decomposed is None:
        return None
    constant, alpha, factors = decomposed
    if len(factors) == 1:
        g, beta, g_ = factors[0]
        variable = _variable(g)
        if variable is None:
            return None
        single = ramanujan_master_theorem(g, variable, s, assumptions)
        if single is None:
            return None
        return _shifted(_scaled(single, beta, g_, s), alpha, constant, s)
    if len(factors) == 2:
        return method_of_brackets(f, x, s, assumptions)
    return None

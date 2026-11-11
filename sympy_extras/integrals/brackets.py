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
exactly (:func:`sympy_extras.concrete.summations.summation`, as a
hypergeometric function, expanded when possible), so the value is right
whenever the summation converges.

The coefficient functions come from SymPy's formal power series
(``sympy.series.formal.fps``), which gives a closed form of the `k`-th
coefficient with factorials and rising factorials, rewritten with the
gamma function so that `k` can be replaced by `-s`.

The module works on the objects of the core (:mod:`sympy_extras.core`, the
functions of :mod:`sympy_extras.functions`) with the native ``ask``,
series, limits, summation, ``gammasimp`` and ``hyperexpand``, and imports
no SymPy. Two steps stay on SymPy and go through the face
:mod:`sympy_extras.integrals.brackets_sympy` when they are reached (the
functions ending in ``_of_core``): the formal power series
(:func:`~sympy_extras.integrals.brackets_sympy.fps_of_core`, SymPy's
``fps``, not ported: its rational and hypergeometric algorithms are a
module of their own) and, for a function whose decay at infinity the
series do not give, the strip of SymPy's ``mellin_transform``
(:func:`~sympy_extras.integrals.brackets_sympy.mellin_strip_of_core`).

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import exp, sin, atan
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

from typing import Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.concrete.expr_with_limits import Sum
from sympy_extras.concrete.summations import summation
from sympy_extras.core.add import Add
from sympy_extras.core.coreerrors import PoleError
from sympy_extras.core.expr import Expr
from sympy_extras.core.limits import limit
from sympy_extras.core.logic import BooleanTrue
from sympy_extras.core.mod import Mod
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, oo
from sympy_extras.core.order import Order
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Equality
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.complexes import abs_ as Abs
from sympy_extras.functions.elementary.miscellaneous import max_ as Max, min_ as Min
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.special.gamma_functions import Gamma, gamma
from sympy_extras.functions.special.hyper import Hyper
from sympy_extras.settings import settings
from sympy_extras.simplify.gammasimp import gammasimp
from sympy_extras.simplify.hyperexpand import hyperexpand
from sympy_extras.simplify.powsimp import powsimp
from sympy_extras.simplify.simplify import simplify
from .mellin import MellinTransform, monomial

__all__ = ['SeriesCoefficient', 'FormalSeriesTerms', 'taylor_coefficient', 'ramanujan_master_theorem',
           'method_of_brackets', 'mellin_transform_series']

#: ``(g_i(y), beta_i, gamma_i)``: a factor ``g_i(beta_i x**gamma_i)`` of a product
Factor = tuple[Expr, Expr, Expr]


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

    __slots__ = ('coefficient', 'index', 'period', 'offset')

    def __init__(self, coefficient: Expr, index: Dummy, period: int, offset: Expr) -> None:
        self.coefficient: Final[Expr] = coefficient
        self.index: Final[Dummy] = index
        self.period: Final[int] = period
        self.offset: Final[Expr] = offset

    def __repr__(self) -> str:
        return "SeriesCoefficient(%s, %s, period=%d, offset=%s)" % (self.coefficient, self.index,
                                                                    self.period, self.offset)

    def phi(self) -> Optional[Expr]:
        """`\\varphi(j) = (-1)^j j!\\, c(j)` of the master theorem, with the
        gamma function in place of the factorials; ``None`` when a factor
        `(-1)^j` remains (no Mellin transform: the series has positive
        coefficients growing like those of `e^x`).

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.functions import log
        >>> from sympy_extras.integrals.brackets import taylor_coefficient
        >>> x = symbols('x')
        >>> c = taylor_coefficient(log(1 + x), x)
        >>> c.phi().subs(c.index, 3)
        -2
        """
        j = self.index
        value = powsimp(S.NegativeOne**j * factorial(j) * self.coefficient, combine='exp')
        value = gammasimp(value.rewrite(Gamma))
        if value.has(S.NegativeOne**j) or _has_sign_power(value, j):
            return None
        return value


class FormalSeriesTerms:
    """SymPy's formal power series of a function read on the core (see
    :func:`~sympy_extras.integrals.brackets_sympy.fps_of_core`): the
    ``formula`` of the coefficient of ``x**index``, a ``Piecewise`` on the
    residue classes of the index or a single expression, valid from
    ``start`` on, and the ``independent`` terms, the polynomial part which
    the formula does not give."""

    __slots__ = ('formula', 'index', 'start', 'independent')

    def __init__(self, formula: Expr, index: Symbol, start: Expr, independent: Expr) -> None:
        self.formula: Final[Expr] = formula
        self.index: Final[Symbol] = index
        self.start: Final[Expr] = start
        self.independent: Final[Expr] = independent

    def __repr__(self) -> str:
        return "FormalSeriesTerms(%s, %s, start=%s, independent=%s)" % (self.formula, self.index, self.start,
                                                                         self.independent)


def _has_sign_power(e: Expr, j: Dummy) -> bool:
    """Whether ``e`` still has a power of a negative number with the
    index in the exponent."""
    for node in e.atoms(Pow):
        if isinstance(node, Pow) and node.base.is_negative is True and node.exp.has(j):
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
        value, condition = pair.expr, pair.cond
        if value == 0:
            continue
        if isinstance(condition, BooleanTrue) and found is None:
            return (value, 1, 0)
        if not isinstance(condition, Equality):
            return None
        lhs, rhs = condition.lhs, condition.rhs
        if not (isinstance(lhs, Mod) and lhs.args[0] == k):
            return None
        modulus, remainder = lhs.args[1], rhs
        if not isinstance(modulus, Integer) or not isinstance(remainder, Integer):
            return None
        if found is not None:
            return None
        found = (value, int(modulus), int(remainder))
    return found


def _replaced(e: Expr, old: Symbol, new: Expr) -> Optional[Expr]:
    """``e.subs(old, new)``, ``None`` when the substitution fails."""
    try:
        return e.subs(old, new)
    except (ValueError, TypeError):
        return None


def taylor_coefficient(f: Expr, x: Symbol) -> Optional[SeriesCoefficient]:
    """The power series of ``f`` about 0 as a :class:`SeriesCoefficient`,
    from SymPy's formal power series; ``None`` when there is none or its
    coefficients have no single closed form.

    Examples
    ========

    >>> from sympy_extras.core import Integer, symbols
    >>> from sympy_extras.functions import exp, sin
    >>> from sympy_extras.integrals.brackets import taylor_coefficient
    >>> x = symbols('x')
    >>> c = taylor_coefficient(exp(-x), x)
    >>> c.coefficient.subs(c.index, Integer(3)), c.period, c.offset
    (-1/6, 1, 0)
    >>> c = taylor_coefficient(sin(x), x)
    >>> c.period, c.offset, c.coefficient.subs(c.index, Integer(1))
    (2, 1, -1/6)
    """
    if f.is_polynomial(x) is True:
        return None
    from .brackets_sympy import fps_of_core
    series = fps_of_core(f, x)
    if series is None:
        return None
    k = series.index
    found = _residue_class(series.formula, k)
    if found is None:
        return None
    value, m, r = found
    # the independent terms must agree with the formula
    for term in Add.make_args(series.independent):
        if term == 0:
            continue
        coefficient, rest = term.as_independent(x, as_Add=False)
        exponent = monomial(rest, x)
        if exponent is None:
            if rest == 1:
                power: Expr = S.Zero
            else:
                return None
        else:
            power = exponent[1]
        if not isinstance(power, Integer) or (int(power) - r) % m != 0:
            return None
        at_power = _replaced(value, k, power)
        if at_power is None or simplify(at_power - coefficient) != 0:
            return None
    # and the whole must agree with the Taylor expansion by the other
    # algorithm for the first terms: fps is not to be trusted alone. Its
    # formula for (exp(4*x) - exp(-4*x))**2 holds from k = 3 and gives 2
    # at k = 0, where the series has no term (the bug: the integral over
    # (0, 1) came out as sinh(8)/4, 2 too much), and its formula for
    # 1/(x**2 + x + 1) is (-1)**k, the series of 1/(x + 1) (the bug: the
    # integral of (1 - x)/(x**2 + x + 1) over (0, 1) came out as
    # log(4) - 1, an eighth too much)
    first = series.start
    if not isinstance(first, Integer):
        return None
    terms = max(6, int(first) + 2 * m)
    try:
        truncated = attempt(lambda: f.series(x, S.Zero, terms).removeO(), settings.timeout, native=True)
    except PoleError:
        return None
    if truncated is None:
        return None
    # the one formula returned must give every coefficient of its residue
    # class, the ones below the start of fps included; where it is
    # infinite (-x**k/k for log(1 - x) at k = 0) the callers start the sum
    # after it, so the coefficient must be zero there
    for k0 in range(terms):
        actual = truncated.coeff(x, k0) if k0 else truncated.subs(x, S.Zero)
        if (k0 - r) % m:
            predicted: Expr = S.Zero
        else:
            at_k0 = _replaced(value, k, Integer(k0))
            if at_k0 is None:
                return None
            predicted = S.Zero if at_k0.is_finite is False else at_k0
        if simplify(predicted - actual) != 0:
            return None
    j = Dummy('j', integer=True, nonnegative=True)
    coefficient_ = _replaced(value, k, m * j + r)
    if coefficient_ is None:
        return None
    return SeriesCoefficient(coefficient_, j, m, Integer(r))


def _order_at_infinity(f: Expr, x: Symbol) -> Optional[Expr]:
    """``e`` such that ``f ~ C x**(-e)`` as ``x -> oo`` (the exponent of
    the leading term), from the series of ``f(1/y)`` at ``y = 0``; ``oo``
    when ``f`` decays faster than every power (an exponential)."""
    y = Dummy('y', positive=True)
    g = f.subs(x, 1 / y)
    try:
        leading = attempt(lambda: g.series(y, S.Zero, 1).removeO(), settings.timeout, native=True)
    except PoleError:
        # an essential singularity at infinity: exponential decay or oscillation
        decays = attempt(lambda: limit(f * x**50, x, oo), settings.timeout, native=True)
        if decays is not None and decays == 0:
            return oo
        return None
    if leading is None or leading == 0:
        # the leading term is beyond the first order: ask for the leading term itself
        try:
            found = attempt(lambda: g.leadterm(y), settings.timeout, native=True)
        except PoleError:
            return None
        if found is None:
            return None
        return found[1]
    if isinstance(leading, Order):
        return None
    exponents: list[Expr] = []
    for term in Add.make_args(leading):
        found_ = monomial(term, x) if term.has(x) else None
        if found_ is not None:
            return None
        rest = term.as_independent(y, as_Add=False)[1]
        if not rest.has(y):
            exponents.append(S.Zero)
            continue
        mono = monomial(rest, y)
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
        from .brackets_sympy import mellin_strip_of_core
        return mellin_strip_of_core(f, x, s)
    return (-offset, upper)


def ramanujan_master_theorem(f: Expr, x: Symbol, s: Symbol,
                             assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """The Mellin transform of ``f`` by Ramanujan's master theorem, with
    the strip in which the integral converges; ``None`` when ``f`` has no
    usable power series or no decay at infinity.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
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
    sigma = (s + c.offset) / c.period
    value = gammasimp(gamma(sigma) * phi.subs(c.index, -sigma) / c.period)
    strip = _strip(f, x, s, c.offset)
    if strip is None:
        return None
    lower, upper = strip
    if upper != oo and lower != -oo and (upper - lower).is_positive is False:
        return None
    return MellinTransform(value, (lower, upper))


def _factors(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, list[Factor]]]:
    """``f`` as ``constant * x**alpha * prod g_i(beta_i x**gamma_i)``:
    the constant, ``alpha`` and the list of ``(g_i(y), beta_i, gamma_i)``
    with ``y`` a fresh symbol per factor."""
    constant: Expr = S.One
    alpha: Expr = S.Zero
    factors: list[Factor] = []
    for e in Mul.make_args(f):
        if not e.has(x):
            constant = constant * e
            continue
        mono = monomial(e, x)
        if mono is not None:
            constant = constant * mono[0]
            alpha = alpha + mono[1]
            continue
        argument: Optional[Expr] = None
        if e.is_Function and len(e.args) >= 1:
            last = e.args[-1]
            if isinstance(last, Expr):
                argument = last
        elif isinstance(e, Pow) and isinstance(e.base, Add):
            argument = e.base
        if argument is None:
            return None
        rest = argument.as_independent(x, as_Add=True)[1]
        mono = monomial(rest, x)
        if mono is None:
            return None
        beta, g_ = mono
        # a positive scale, so that g is read on (0, oo)
        if beta.is_negative is True:
            beta = -beta
        y = Dummy('y', positive=True)
        factors.append((e.subs(x, (y / beta)**(1 / g_)), beta, g_))
    return constant, alpha, factors


def _scaled(transform: MellinTransform, beta: Expr, g_: Expr, s: Symbol) -> MellinTransform:
    """The transform of ``g(beta x**gamma)`` from that of ``g``."""
    value = transform.transform.subs(s, s / g_) * beta**(-s / g_) / Abs(g_)
    lower, upper = transform.strip
    if g_.is_positive is True:
        return MellinTransform(value, (lower * g_, upper * g_), transform.condition)
    return MellinTransform(value, (upper * g_, lower * g_), transform.condition)


def _shifted(transform: MellinTransform, alpha: Expr, constant: Expr, s: Symbol) -> MellinTransform:
    value = constant * transform.transform.subs(s, s + alpha)
    lower, upper = transform.strip
    return MellinTransform(value, (lower - alpha, upper - alpha), transform.condition)


def method_of_brackets(f: Expr, x: Symbol, s: Symbol, assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """The Mellin transform of a product of two series-expandable factors
    by the method of brackets: one index is eliminated by the bracket
    rule and the remaining series is summed as a hypergeometric function,
    which is expanded when possible. Both eliminations are tried and the
    one whose series converges is kept. The strip is the intersection of
    the strips of the factors' own transforms.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, sin
    >>> from sympy_extras.integrals.brackets import method_of_brackets
    >>> x, s = symbols('x s')
    >>> method_of_brackets(exp(-x)*sin(x), x, s)
    MellinTransform(2**(s + 1)*gamma(s/4 + 1/2)*gamma(s/2 + 1/2)/(8*gamma(1 - s/4)), (-1, oo))
    """
    decomposed = _factors(f, x)
    if decomposed is None or len(decomposed[2]) != 2:
        return None
    constant, alpha, factors = decomposed
    series: list[tuple[SeriesCoefficient, Expr, Expr]] = []
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
        series.append((c, beta, g_))
        strips.append(scaled.strip)
    lower = _max(strips[0][0], strips[1][0])
    upper = _min(strips[0][1], strips[1][1])
    if lower != -oo and upper != oo and (upper - lower).is_positive is False:
        return None
    # the double series: sum_{n, k} phi1(n) (-1)^n/n! phi2(k) (-1)^k/k! beta1^(...) beta2^(...)
    # x^(g1 (m1 n + r1) + g2 (m2 k + r2) + alpha); the bracket <A n + B k + C + s>
    (c1, beta1, g1), (c2, beta2, g2) = series
    phi1, phi2 = c1.phi(), c2.phi()
    if phi1 is None or phi2 is None:
        return None
    for first in (0, 1):
        ca, cb = (c1, c2) if first == 0 else (c2, c1)
        phia, phib = (phi1, phi2) if first == 0 else (phi2, phi1)
        betaa, betab = (beta1, beta2) if first == 0 else (beta2, beta1)
        ga, gb = (g1, g2) if first == 0 else (g2, g1)
        n = ca.index
        # the exponent of x: ga (ma n + ra) + gb (mb k + rb) + alpha; eliminate k
        A = ga * ca.period
        B = gb * cb.period
        C = ga * ca.offset + gb * cb.offset + alpha
        k_star = -(A * n + C + s) / B
        term = (phia * S.NegativeOne**n / factorial(n) * betaa**(ca.period * n + ca.offset)
                * phib.subs(cb.index, k_star) * betab**(cb.period * k_star + cb.offset)
                * gamma(-k_star) / Abs(B))
        term = gammasimp(term)
        total = attempt(lambda: summation(term, (n, S.Zero, oo)), settings.timeout, native=True)
        if total is None:
            continue
        if isinstance(total, Piecewise):
            # the closed form where the series converges, continued
            # analytically to the strip (the bracket rule keeps it)
            closed = _closed_branch(total)
            if closed is None:
                continue
            total = closed
        if total.has(Sum):
            continue
        if total.has(Hyper) and not _converges(total, assumptions):
            continue
        summed = total
        expanded = attempt(lambda: hyperexpand(summed), settings.timeout, native=True)
        value = gammasimp(constant * (expanded if expanded is not None else total))
        decay = _order_at_infinity(f, x)
        if decay is not None and not decay.has(x):
            # the true strip of the product: from its order at 0 and its decay
            lower = -(g1 * c1.offset + g2 * c2.offset + alpha)
            upper = decay - alpha
            return MellinTransform(value, (lower, upper))
        return MellinTransform(value, (lower - alpha, upper - alpha))
    return None


def _closed_branch(total: Piecewise) -> Optional[Expr]:
    """The first branch of a ``Piecewise`` returned by the summation whose
    other branch is the unevaluated sum."""
    for pair in total.args:
        if not pair.expr.has(Sum):
            return pair.expr
    return None


def _variable(g: Expr) -> Optional[Symbol]:
    symbols = [v for v in g.free_symbols if isinstance(v, Dummy)]
    return symbols[0] if len(symbols) == 1 else None


def _converges(e: Expr, assumptions: Assumptions) -> bool:
    """Whether every hypergeometric series in ``e`` converges: ``p <= q``,
    or ``p = q + 1`` with ``|z| < 1`` (decided with the assumptions)."""
    for node in e.atoms(Hyper):
        if not isinstance(node, Hyper):
            continue
        p, q = len(node.ap), len(node.bq)
        if p <= q:
            continue
        if p == q + 1 and ask(Abs(node.argument) < 1, assumptions) is True:
            continue
        return False
    return True


def _max(a: Expr, b: Expr) -> Expr:
    if a == -oo:
        return b
    if b == -oo:
        return a
    d = a - b
    if d.is_nonnegative is True:
        return a
    if d.is_negative is True:
        return b
    return Max(a, b)


def _min(a: Expr, b: Expr) -> Expr:
    if a == oo:
        return b
    if b == oo:
        return a
    d = a - b
    if d.is_nonpositive is True:
        return a
    if d.is_positive is True:
        return b
    return Min(a, b)


def mellin_transform_series(f: Expr, x: Symbol, s: Symbol, assumptions: Assumptions = None) -> Optional[MellinTransform]:
    """The Mellin transform of ``f`` from its power series: Ramanujan's
    master theorem for one factor (times a constant and a power of ``x``),
    the method of brackets for a product of two; ``None`` otherwise.

    Examples
    ========

    >>> from sympy_extras.core import sqrt, symbols
    >>> from sympy_extras.functions import exp, cos
    >>> from sympy_extras.integrals.brackets import mellin_transform_series
    >>> x, s = symbols('x s')
    >>> mellin_transform_series(x**2*exp(-3*x), x, s)
    MellinTransform(3**(-s - 2)*gamma(s + 2), (-2, oo))
    >>> mellin_transform_series(cos(sqrt(x)), x, s)
    MellinTransform(2**(2*s)*sqrt(pi)*gamma(s)/gamma(1/2 - s), (0, 1/2))
    """
    if not isinstance(f, Expr):
        raise TypeError("expected an expression of the core, got %r" % (f,))
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

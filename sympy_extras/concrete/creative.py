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
recurrence (SymPy's ``rsolve_hyper``) reduces the order by one, a first
order recurrence is solved by a product and a sum, and the sums are
evaluated with Karr's algorithm (:func:`~.karr.karr_sum`) or left as
unevaluated sums when they have no closed form in their field (they are
then part of the answer, as $\sum_{j=1}^n 1/(j 2^j)$ in
$\sum_k \binom{n}{k} H_k = 2^n (H_n - \sum_{j=1}^n 1/(j 2^j))$).
The constants are fixed by values of the sum computed directly, from the
first $n$ from which the recurrence holds and determines the next value;
the closed form is then compared with the directly computed sum at a few
more values of $n$ (exactly, or numerically with
:func:`~sympy_extras._numeric.reliable_value` at sample values of the other
parameters), and at the small values of $n$ below the start, where it may
fail and the sum is written as a ``Piecewise``.

Examples
========

>>> from sympy import binomial, harmonic, symbols
>>> from sympy_extras.concrete import creative_telescoping, definite_sum
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
from typing import Callable, Optional, Sequence, Union

from sympy.concrete.products import Product, product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, count_ops, expand_func
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import FallingFactorial, RisingFactorial, binomial, factorial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import Boolean, true
from sympy.polys.fields import FracElement
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel, factor_list, gcd_list, lcm_list
from sympy.polys.rationaltools import together
from sympy.simplify.combsimp import combsimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.simplify import hypersimp
from sympy.solvers.recurr import rsolve_hyper
from sympy.solvers.solveset import linsolve

from sympy_extras._numeric import reliable_value
from sympy_extras._timeout import attempt
from sympy_extras._typing import as_boolean, as_expr, as_symbol, free_symbols, sorted_symbols
from sympy_extras.settings import settings

from .karr import _Builder, _atoms, _auto_extensions, karr_sum
from .pisigma import PiSigmaField

__all__ = ['CreativeTelescoper', 'creative_telescoping', 'definite_sum']

#: the seconds given to one simplification or one call of a SymPy solver
_STEP_SECONDS = 10.0
#: the values of n beyond the initial values at which the closed form is
#: compared with the sum
_CHECKS = 5


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
    field : PiSigmaField
        The $\Pi\Sigma$-field of ``k`` over $\mathbb{Q}(n)$ in which the
        relation holds.
    """

    def __init__(self, term: Expr, k: Symbol, n: Symbol, coefficients: list[Expr], certificate: Expr,
                 field: PiSigmaField, elements: list[FracElement], g: FracElement, factor: Expr) -> None:
        self.term = term
        self.k = k
        self.n = n
        self.coefficients = coefficients
        self.certificate = certificate
        self.field = field
        self._elements = elements
        self._g = g
        self._factor = factor

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def recurrence(self, name: str = 'S') -> Expr:
        """The left hand side $\\sum_i c_i(n) S(n + i)$ of the recurrence
        of the sums over ``k``, as an expression in the function ``S``."""
        f = Function(name)
        return as_expr(Add(*[c*f(self.n + i) for i, c in enumerate(self.coefficients)]))

    def shifted_certificate(self) -> Expr:
        """$g(n, k + 1)$ written with the sequences of the field at ``k``
        (so that it can be evaluated where $g$ itself is not, just past
        the support of a binomial coefficient)."""
        return as_expr(self._factor*self.field.to_expr(self.field.sigma(self._g)))

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
    return any(isinstance(a, (harmonic, Sum, Product)) or a.has(harmonic, Sum, Product)
               for a in _atoms(expr, k))


def _split(f: Expr, k: Symbol) -> tuple[Expr, Expr, Expr]:
    """``(u, T, R)`` with ``f = u*T*R``: ``u`` the factors free of ``k``,
    ``T`` the hypergeometric factors and ``R`` the rest (rational in ``k``
    or containing sums)."""
    u: Expr = S.One
    T: Expr = S.One
    R: Expr = S.One
    for factor in Mul.make_args(f):
        if not factor.has(k):
            u = as_expr(u*factor)
        elif _has_sums(factor, k) or not _atoms(factor, k):
            R = as_expr(R*factor)
        else:
            T = as_expr(T*factor)
    return u, T, R


def _rational_ratio(num: Expr, den: Expr, symbols: Sequence[Symbol]) -> Optional[Expr]:
    """``num/den`` as a rational function of the ``symbols``, or ``None``."""
    if num == den:
        return S.One
    ratio = attempt(lambda: as_expr(cancel(combsimp(as_expr(num/den)))), _STEP_SECONDS)
    if ratio is None or not ratio.is_rational_function(*symbols):
        second = attempt(lambda: as_expr(cancel(combsimp(as_expr((num/den).rewrite('gamma'))))), _STEP_SECONDS)
        if second is None or not second.is_rational_function(*symbols):
            return None
        ratio = second
    return ratio


def _normalized(values: list[Expr], n: Symbol) -> tuple[list[Expr], Expr]:
    """The coefficients scaled to polynomials without a common factor,
    the leading one with a positive leading coefficient, and the scale."""
    denominators = [as_expr(together(v).as_numer_denom()[1]) for v in values if v != 0]
    scale = as_expr(lcm_list(denominators)) if denominators else S.One
    scaled = [as_expr(cancel(v*scale)) for v in values]
    numerators = [v for v in scaled if v != 0]
    if numerators:
        common = as_expr(gcd_list(numerators))
        if common != 0:
            scale = as_expr(cancel(scale/common))
            scaled = [as_expr(cancel(v/common)) for v in scaled]
    leading = next((v for v in reversed(scaled) if v != 0), S.One)
    symbols = sorted_symbols(free_symbols(leading)) or [n]
    try:
        if Poly(leading, *symbols).LC() < 0:
            scale = -scale
            scaled = [-v for v in scaled]
    except (PolynomialError, TypeError):
        pass
    return scaled, scale


def creative_telescoping(f: Union[Expr, int], k: Symbol, n: Symbol, order: int = 4,
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

    >>> from sympy import binomial, harmonic, symbols
    >>> from sympy_extras.concrete import creative_telescoping
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
    f_ = as_expr(f)
    k_, n_ = as_symbol(k), as_symbol(n)
    result = _creative_telescoping(f_, k_, n_, order, list(extensions))
    if result is None:
        u, T, R = _split(f_, k_)
        if _has_sums(R, k_):
            extra = [e for e in _auto_extensions(R, k_) if e not in extensions]
            if extra:
                result = _creative_telescoping(f_, k_, n_, order, list(extensions) + extra)
    return result


def _creative_telescoping(f: Expr, k: Symbol, n: Symbol, order: int,
                          extensions: list[Expr]) -> Optional[CreativeTelescoper]:
    u, T, R = _split(f, k)
    others = sorted_symbols(free_symbols(f) - {k, n})
    symbols = free_symbols(f) | {n}
    for e in extensions:
        symbols |= free_symbols(e)
    params = sorted_symbols(symbols - {k})
    builder = _Builder(k, params)
    tau: Expr = S.One
    if T != 1:
        alpha = hypersimp(T, k)
        if alpha is None or not as_expr(alpha).is_rational_function(k):
            raise ValueError("%s is not hypergeometric in %s" % (T, k))
        tau = builder.field.add_pi(as_expr(alpha), T)
    for e in extensions:
        builder.convert(e)
    converted: list[Expr] = []
    for r in range(order + 1):
        mu = _rational_ratio(as_expr(u.subs(n, n + r)), u, [n, *others])
        rho = _rational_ratio(as_expr(T.subs(n, n + r)), T, [n, k, *others])
        if mu is None or rho is None:
            raise ValueError("the shifts of %s in %s are not rational multiples of it" % (f, n))
        element = builder.convert(as_expr(mu*rho*tau*R.subs(n, n + r)))
        converted.append(builder.field.to_expr(element, substitute=False))
        field = builder.field
        fs = [field.from_expr(e) for e in converted]
        sols = field.solve(field.one, fs)
        if sols is None:
            return None
        for c, g in sols:
            if not any(c):
                continue
            values = [as_expr(field.C.to_sympy(ci)) for ci in c]
            coefficients, scale = _normalized(values, n)
            g = g*field.from_expr(scale)
            certificate = as_expr(u*field.to_expr(g))
            return CreativeTelescoper(f, k, n, coefficients, certificate, field, fs, g, u)
    return None


# ----------------------------------------------------------------------
# definite sums

def _linear(expr: Expr, n: Symbol) -> Optional[tuple[int, int]]:
    """``(m, s)`` with ``expr == m*n + s``, integers ``m >= 0`` and ``s``."""
    try:
        p = Poly(expr, n)
    except PolynomialError:
        return None
    if p.degree() > 1:
        return None
    coeffs = p.all_coeffs()
    m, s = (coeffs[0], coeffs[1]) if len(coeffs) == 2 else (S.Zero, coeffs[0])
    if not (isinstance(m, Integer) and isinstance(s, Integer)) or m < 0:
        return None
    return int(m), int(s)


def _finite(value: Expr) -> bool:
    return not value.has(S.NaN, S.ComplexInfinity, S.Infinity, S.NegativeInfinity)


def _direct(f: Expr, k: Symbol, lo: int, hi: int) -> Optional[Expr]:
    """``Sum(f, (k, lo, hi))`` term by term with Karr's convention for
    ``hi < lo``, or ``None`` when a term is not defined."""
    if hi < lo - 1:
        value = _direct(f, k, hi + 1, lo - 1)
        return None if value is None else -value
    total: Expr = S.Zero
    for i in range(lo, hi + 1):
        term = as_expr(f.subs(k, i))
        if term.has(Sum, Product):
            term = as_expr(term.doit())
        if not _finite(term):
            return None
        total = as_expr(total + term)
    return total


def _tidy(expr: Expr) -> Expr:
    """``expr`` with the binomials and harmonic numbers of shifted
    arguments expanded and the hypergeometric terms simplified."""
    value = attempt(lambda: as_expr(combsimp(expand_func(expr))), _STEP_SECONDS)
    return expr if value is None else value


def _unshifted_harmonics(expr: Expr) -> Expr:
    """``harmonic(x + c, r)`` for integers ``c`` written through
    ``harmonic(x, r)`` and ``c`` reciprocal powers."""
    def rule(h: Expr) -> Expr:
        x = as_expr(h.args[0])
        order = as_expr(h.args[1]) if len(h.args) > 1 else S.One
        c, y = x.as_coeff_Add()
        if not isinstance(c, Integer) or c == 0 or y.is_number:
            return h
        if c > 0:
            return as_expr(harmonic(y, order) + Add(*[1/(y + i)**order for i in range(1, int(c) + 1)]))
        return as_expr(harmonic(y, order) - Add(*[1/(y - i)**order for i in range(0, -int(c))]))
    return as_expr(expr.replace(lambda e: isinstance(e, harmonic), rule))


def _falling(expr: Expr) -> Expr:
    """``expr`` with ``RisingFactorial(-x, n)`` written as ``(-1)**n*
    FallingFactorial(x, n)``, and falling factorials over factorials as
    binomials: the product of Vandermonde's recurrence is
    ``(-1)**n*RisingFactorial(-a - b, n)/factorial(n)``, which is
    ``binomial(a + b, n)``."""
    def rule(e: Expr) -> Expr:
        x, m = as_expr(e.args[0]), as_expr(e.args[1])
        return as_expr((-1)**m*FallingFactorial(-x, m))
    replaced = as_expr(powsimp(expr.replace(lambda e: isinstance(e, RisingFactorial), rule)))
    return as_expr(replaced.rewrite(binomial))


def _shifted_harmonics(expr: Expr, n: Symbol, shift: int) -> Expr:
    """``expr`` with ``harmonic(n, r)`` written through ``harmonic(n +
    shift, r)``, expanded."""
    def rule(h: Expr) -> Expr:
        order = as_expr(h.args[1]) if len(h.args) > 1 else S.One
        if h.args[0] != n or not isinstance(order, Integer):
            return h
        # harmonic(n + shift) minus the reciprocals between n and n + shift
        return as_expr(harmonic(n + shift, order) - (_unshifted_harmonics(harmonic(n + shift, order)) - h))
    return as_expr(cancel(expr.replace(lambda e: isinstance(e, harmonic), rule)))


def _simplest(expr: Expr, n: Symbol) -> Expr:
    """The shortest of a few simplifications of the closed form."""
    from .zeilberger import _nicer
    candidates = [expr]
    starts = [expr]
    for transformation in (_unshifted_harmonics, _falling):
        value = attempt(lambda: transformation(expr), _STEP_SECONDS)
        if value is not None and value != expr:
            starts.append(value)
    if expr.has(harmonic) and not expr.has(Sum):
        for shift in (1, -1):
            value = attempt(lambda: _shifted_harmonics(starts[1] if len(starts) > 1 else expr, n, shift),
                            _STEP_SECONDS)
            if value is not None:
                starts.append(value)
    simplifications: list[Callable[[Expr], Expr]] = [
        lambda e: as_expr(factor_terms(combsimp(e))),
        lambda e: as_expr(factor_terms(cancel(e))) if not e.has(Sum) else e,
        lambda e: as_expr(factor_terms(combsimp(e.expand()))),
        lambda e: as_expr(e.replace(lambda g: isinstance(g, gamma) and as_expr(g.args[0] - 1).is_nonnegative is True,
                                    lambda g: factorial(g.args[0] - 1))),
    ]
    if not expr.has(Sum, harmonic):
        simplifications.append(lambda e: _nicer(e, n))
    for start in starts:
        candidates.append(start)
        for simplify in simplifications:
            value = attempt(lambda: simplify(start), _STEP_SECONDS)
            if value is not None:
                candidates.append(value)
    return min(candidates, key=lambda e: (count_ops(e), len(str(e))))


def _integer_roots(expr: Expr, n: Symbol) -> list[int]:
    """The integer roots of the factors of ``expr`` which are polynomials
    in ``n`` alone."""
    roots: list[int] = []
    try:
        factors = factor_list(expr)[1]
    except (PolynomialError, TypeError):
        return roots
    for factor, _ in factors:
        if free_symbols(factor) != {n} or not factor.is_polynomial(n):
            continue
        p = Poly(factor, n)
        if p.degree() == 1:
            a1, a0 = p.all_coeffs()
            root = -a0/a1
            if isinstance(root, Integer):
                roots.append(int(root))
    return roots


class _Unsupported(Exception):
    """A pole of the certificate inside the summation range."""


def _poles(Z: CreativeTelescoper) -> tuple[list[tuple[Fraction, Fraction]], list[int]]:
    """The poles ``k = alpha*n + beta`` of the representations in the
    field of the summands, the certificate, its shift and the generators'
    shift quotients, and the integer poles in ``n`` of their
    coefficients."""
    F, k, n = Z.field, Z.k, Z.n
    elements = list(Z._elements) + [Z._g, F.sigma(Z._g)]
    elements += [F.from_expr(e.value) for e in F.extensions]
    gens = set(F.symbols[1:])
    in_k: list[tuple[Fraction, Fraction]] = []
    in_n: list[int] = []
    for element in elements:
        for part in (element.numer, element.denom):
            expr = as_expr(together(part.as_expr()))
            numerator, denominator = expr.as_numer_denom()
            # the coefficients of the numerator and the denominator in Q(n)
            in_n.extend(_integer_roots(as_expr(denominator), n))
            if part is element.numer:
                continue
            try:
                factors = factor_list(as_expr(numerator))[1]
            except (PolynomialError, TypeError):
                continue
            for factor, _ in factors:
                symbols = free_symbols(factor)
                if symbols & gens or k not in symbols:
                    if not symbols & gens and k not in symbols and symbols == {n}:
                        in_n.extend(_integer_roots(as_expr(factor), n))
                    continue
                p = Poly(factor, k)
                if p.degree() != 1:
                    continue
                a1, a0 = p.all_coeffs()
                root = as_expr(cancel(-a0/a1))
                if not free_symbols(root) <= {n}:
                    continue
                try:
                    q = Poly(root, n)
                except PolynomialError:
                    continue
                if q.degree() > 1:
                    continue
                coeffs = q.all_coeffs()
                alpha, beta = (coeffs[0], coeffs[1]) if len(coeffs) == 2 else (S.Zero, coeffs[0])
                if not (isinstance(alpha, Rational) and isinstance(beta, Rational)):
                    continue
                in_k.append((Fraction(int(alpha.p), int(alpha.q)), Fraction(int(beta.p), int(beta.q))))
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
    names = {s.name for s in free_symbols(expr)}
    for name in ('j', 'i', 'l', 'm', 'p', 'q'):
        if name not in names:
            return Symbol(name, integer=True)
    return Dummy('j', integer=True)


def _prettier_sum(term: Expr, j: Symbol, lo: Expr, hi: Expr) -> Expr:
    """``Sum(term, (j, lo, hi))`` with the index shifted to the simplest
    summand."""
    candidates = []
    for d in range(-2, 3):
        shifted = as_expr(cancel(term.subs(j, j - d))) if term.is_rational_function(j) else as_expr(
            combsimp(term.subs(j, j - d)))
        candidates.append(Sum(shifted, (j, lo + d, hi + d)))
    return min(candidates, key=lambda e: (count_ops(e.function), len(str(e.function))))


def _harmonic_extensions(term: Expr, j: Symbol) -> list[Expr]:
    """``harmonic(d*j)`` for the leading coefficients ``d > 1`` of the
    linear factors of the denominator of ``term``."""
    result: list[Expr] = []
    denominator = as_expr(together(term).as_numer_denom()[1])
    try:
        factors = factor_list(denominator)[1]
    except (PolynomialError, TypeError):
        return result
    for factor, _ in factors:
        if factor.has(j) and factor.is_polynomial(j) and Poly(factor, j).degree() == 1:
            d = Poly(factor, j).LC()
            if isinstance(d, Integer) and abs(int(d)) > 1:
                result.append(harmonic(abs(int(d))*j))
    return result


def _sum(term: Expr, j: Symbol, lo: Expr, hi: Expr) -> Expr:
    r"""$\sum_{j=lo}^{hi}$ ``term``: a closed form by Karr's algorithm,
    else the sum of the closed forms of the terms, with the terms which
    have none left as unevaluated sums."""
    term = _tidy(term)
    if term == 0:
        return S.Zero

    def karr(t: Expr) -> Optional[Expr]:
        def compute() -> Optional[Expr]:
            value = karr_sum(t, (j, lo, hi))
            if value is None:
                extra = _harmonic_extensions(t, j)
                if extra:
                    value = karr_sum(t, (j, lo, hi), extra)
            if value is None:
                return None
            # g(n) - g(0) for the sum up to n - 1 has harmonic(n - 1),
            # undefined at n = 0
            return as_expr(cancel(_unshifted_harmonics(value)))
        return attempt(compute, _STEP_SECONDS)

    whole = karr(term)
    if whole is not None:
        return whole
    total: Expr = S.Zero
    rest: Expr = S.Zero
    expanded = attempt(lambda: as_expr(term.expand()), _STEP_SECONDS)
    for t in Add.make_args(expanded if expanded is not None else term):
        t = _tidy(as_expr(powsimp(t)))
        value = karr(t)
        if value is None:
            rest = as_expr(rest + t)
        else:
            total = as_expr(total + value)
    if rest != 0:
        total = as_expr(total + _prettier_sum(_tidy(rest), j, lo, hi))
    return total


def _product(ratio: Expr, n: Symbol, n0: int) -> Optional[Expr]:
    r"""$\prod_{j=n_0}^{n-1}$ ``ratio(j)`` for a rational ``ratio``, or
    ``None`` when a factor vanishes or has a pole for some $j \ge n_0$.

    The ratio is split into linear factors $j + a$, whose products are
    $\Gamma(n + a)/\Gamma(n_0 + a)$; the factors whose $a$ differ by
    integers are collected, so that the gamma functions of a class cancel
    when their exponents add up to zero (SymPy's ``product`` gives
    $\Gamma(n + 5/4)/\Gamma(n + 1/4)$ for $\prod (j + 1/4)$, which
    ``gammasimp`` does not reduce to $n + 1/4$).
    """
    j = Dummy('j', integer=True)
    numerator, denominator = together(as_expr(ratio.subs(n, j))).as_numer_denom()
    constant: Expr = S.One
    exponents: dict[Expr, int] = {}
    for part, sign in ((as_expr(numerator), 1), (as_expr(denominator), -1)):
        try:
            coefficient, factors = factor_list(part, j)
        except (PolynomialError, TypeError):
            return None
        constant = as_expr(constant*coefficient**sign)
        for factor, e in factors:
            p = Poly(factor, j)
            if p.degree() != 1:
                value = attempt(lambda: product(as_expr(ratio.subs(n, j)), (j, n0, n - 1)), _STEP_SECONDS)
                if not isinstance(value, Expr) or value.has(Product):
                    return None
                return _tidy(value)
            lead, tail = p.all_coeffs()
            constant = as_expr(constant*lead**(sign*e))
            a = as_expr(cancel(tail/lead))
            if isinstance(a, Integer) and n0 + int(a) <= 0:
                return None
            exponents[a] = exponents.get(a, 0) + sign*int(e)
    result: Expr = as_expr(constant**(n - n0))
    classes: list[list[Expr]] = []
    for a in exponents:
        for members in classes:
            if isinstance(cancel(a - members[0]), Integer):
                members.append(a)
                break
        else:
            classes.append([a])
    for members in classes:
        base = min(members, key=lambda a: as_expr(cancel(a - members[0])))
        total = 0
        for a in members:
            e = exponents[a]
            total += e
            shift = int(as_expr(cancel(a - base)))
            # Gamma(x + a)/Gamma(x + base) = (x + base)*...*(x + a - 1)
            rising = Mul(*[(n + base + i)/(n0 + base + i) for i in range(shift)])
            result = as_expr(result*rising**e)
        if total and base.is_number:
            result = as_expr(result*(gamma(n + base)/gamma(n0 + base))**total)
        elif total:
            result = as_expr(result*RisingFactorial(n0 + base, n - n0)**total)
    return as_expr(cancel(result)) if not result.has(gamma, RisingFactorial) else result


def _hypergeometric_solution(c: list[Expr], n: Symbol) -> Optional[Expr]:
    """A hypergeometric solution of the recurrence with coefficients ``c``
    (SymPy's ``rsolve_hyper``, a term of its general solution)."""
    general = attempt(lambda: rsolve_hyper(c, 0, n), 4*_STEP_SECONDS)
    if not isinstance(general, Expr) or general == 0:
        return None
    for term in Add.make_args(general.expand()):
        constants = [s for s in free_symbols(as_expr(term)) if s.name.startswith('C') and s != n]
        value = as_expr(term)
        for s in constants:
            value = as_expr(value.subs(s, 1))
        if value != 0:
            from .zeilberger import _nicer
            nicer = attempt(lambda: _nicer(value, n), _STEP_SECONDS)
            return value if nicer is None else nicer
    return None


def _general_solution(c: list[Expr], rhs: Expr, n: Symbol, n0: int) -> Optional[tuple[Expr, list[Symbol]]]:
    r"""The general solution of $\sum_i c_i(n) S(n + i) = rhs(n)$ for
    $n \ge n_0$ in d'Alembertian terms, with its constants, or ``None``."""
    r = len(c) - 1
    if r == 0:
        return as_expr(cancel(rhs/c[0])), []
    C = Dummy('C')
    if r == 1:
        ratio = as_expr(cancel(-c[0]/c[1]))
        P = _product(ratio, n, n0)
        if P is None:
            return None
        beta = as_expr(cancel(rhs/c[1]))
        if beta == 0:
            return as_expr(C*P), [C]
        j = _index(as_expr(rhs + P))
        term = _tidy(as_expr(beta.subs(n, j)/P.subs(n, j + 1)))
        return as_expr(P*(C + _sum(term, j, Integer(n0), n - 1))), [C]
    h = _hypergeometric_solution(c, n)
    if h is None:
        return None
    hypergeometric = h
    quotient = attempt(lambda: hypersimp(hypergeometric, n), _STEP_SECONDS)
    rho: Optional[Expr] = as_expr(quotient) if isinstance(quotient, Expr) and quotient.is_rational_function(n) else \
        _rational_ratio(as_expr(h.subs(n, n + 1)), h, [n])
    if rho is None:
        return None
    # S = h*y, z = y(n + 1) - y(n): sum_t d_t z(n + t) = rhs/h with
    # d_t = sum_{i > t} c_i Q_i, Q_i = h(n + i)/h(n)
    Q: list[Expr] = [S.One]
    for i in range(1, r + 1):
        Q.append(as_expr(cancel(Q[-1]*rho.subs(n, n + i - 1))))
    d = [as_expr(cancel(Add(*[c[i]*Q[i] for i in range(t + 1, r + 1)]))) for t in range(r)]
    reduced = _general_solution(d, _tidy(as_expr(rhs/h)), n, n0)
    if reduced is None:
        return None
    z, constants = reduced
    j = _index(as_expr(z + rhs))
    return as_expr(h*(C + _sum(as_expr(z.subs(n, j)), j, Integer(n0), n - 1))), constants + [C]


def _agree(direct: Expr, value: Expr) -> Optional[bool]:
    """Whether ``direct == value``: exactly when SymPy proves it, else
    numerically at sample values of the symbols; ``None`` when neither
    is decided."""
    difference = as_expr(direct - value)
    exact = attempt(lambda: as_expr(cancel(combsimp(expand_func(difference)))), _STEP_SECONDS)
    if exact == 0:
        return True
    digits = settings.precision
    symbols = sorted_symbols(free_symbols(difference))
    samples: list[dict[Symbol, Expr]] = [{}] if not symbols else [
        {s: Rational(p, q) for s, p, q in zip(symbols, primes, [7, 11, 13, 17, 19, 23, 29, 31])}
        for primes in ([31, 37, 41, 43, 47, 53, 59, 61], [67, 71, 73, 79, 83, 89, 97, 101])]
    for values in samples:
        error = reliable_value(difference, digits, values)
        size = reliable_value(direct, digits, values)
        if error is None or size is None:
            return None
        if abs(complex(error)) > 10.0**(-(digits - 10))*max(1.0, abs(complex(size))):
            return False
    return True


def definite_sum(f: Union[Expr, int], limits: Sequence[Union[Symbol, Expr, int]], n: Optional[Symbol] = None,
                 order: int = 4, extensions: Sequence[Expr] = ()) -> Optional[Expr]:
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

    >>> from sympy import binomial, harmonic, symbols
    >>> from sympy_extras.concrete import definite_sum
    >>> n, k, a, b = symbols('n k a b')
    >>> definite_sum(binomial(a, k)*binomial(b, n - k), (k, 0, n))
    binomial(a + b, n)
    >>> definite_sum(k*binomial(n, k), (k, 0, n))
    2**n*n/2
    >>> definite_sum(binomial(n, k)*harmonic(k), (k, 0, n))
    2**n*(harmonic(n) - Sum(1/(2**j*j), (j, 1, n)))
    >>> definite_sum(binomial(n, k)**2*harmonic(k), (k, 0, n))
    (2*harmonic(n) - harmonic(2*n))*binomial(2*n, n)
    """
    f_ = as_expr(f)
    k_, a_, b_ = limits
    k = as_symbol(k_)
    a, b = as_expr(a_), as_expr(b_)
    if n is None:
        candidates = sorted_symbols((free_symbols(a) | free_symbols(b)) - {k})
        if len(candidates) != 1:
            return None
        n = candidates[0]
    # work with an integer n and k, for which SymPy evaluates
    # binomial(n, n) and the like
    N = Dummy(n.name, integer=True, nonnegative=True)
    K = Dummy(k.name, integer=True)
    g = as_expr(f_.xreplace({n: N, k: K}))
    lower, upper = _linear(as_expr(a.xreplace({n: N})), N), _linear(as_expr(b.xreplace({n: N})), N)
    if lower is None or upper is None or lower[0] > upper[0] or (lower[0] == upper[0] == 0):
        return None
    u, T, R = _split(g, K)
    h = as_expr(T*R)
    result = _definite_sum(h, K, N, lower, upper, order, extensions)
    if result is None and not isinstance(h, Add):
        expanded = attempt(lambda: as_expr(h.expand()), _STEP_SECONDS)
        if isinstance(expanded, Add):
            h = expanded
    if result is None and isinstance(h, Add):
        total: Expr = S.Zero
        for term in h.args:
            value = _definite_sum(as_expr(term), K, N, lower, upper, order, extensions)
            if value is None:
                return None
            total = as_expr(total + value)
        result = total
    if result is None:
        return None
    return as_expr((u*result).xreplace({N: n, K: k}))


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
    a: Callable[[Expr], Expr] = lambda v: as_expr(m1*v + s1)
    b: Callable[[Expr], Expr] = lambda v: as_expr(m2*v + s2)
    bottom = as_expr(a(n + r) + j0)
    top = as_expr(b(n) - j1)
    # sum_i c_i S(n + i) = g(n, top + 1) - g(n, bottom) + the terms of
    # each S(n + i) outside [bottom, top]
    rhs = as_expr(Z.shifted_certificate().subs(k, top) - Z.certificate.subs(k, bottom))
    for i, ci in enumerate(c):
        if ci == 0:
            continue
        fi = as_expr(f.subs(n, n + i))
        below = m1*(r - i) + j0
        above = m2*i + j1
        extra = Add(*[fi.subs(k, a(n + i) + t) for t in range(below)],
                    *[fi.subs(k, top + 1 + t) for t in range(above)])
        rhs = as_expr(rhs + ci*extra)
    rhs = _tidy(rhs)
    if not _finite(rhs):
        return None
    starts = [n_min, 0] + [v + 1 for v in poles_in_n]
    starts += [v + 1 for v in _integer_roots(c[-1], n)]
    starts += [v + 1 for v in _integer_roots(as_expr(together(rhs).as_numer_denom()[1]), n)]
    if r == 1:
        starts += [v + 1 for v in _integer_roots(c[0], n)]
    n0 = max(starts)

    def direct(v: int) -> Optional[Expr]:
        return _direct(as_expr(f.subs(n, v)), k, int(a(Integer(v))), int(b(Integer(v))))

    # a recurrence in steps of d > 1 (sum (-1)**k binomial(n, k)**2 gives
    # S(n + 2) in terms of S(n)) is one recurrence for each residue class
    # of n modulo d
    d = 0
    for i, ci in enumerate(c):
        if ci != 0:
            d = gcd(d, i)
    if d <= 1:
        solved = _solve_and_check(c, rhs, n, n0, direct)
        if solved is None:
            return None
        solution, exceptions = solved
        pieces = [(value, as_boolean(Eq(n, v))) for v, value in exceptions]
        return as_expr(Piecewise(*pieces, (solution, true))) if pieces else solution
    m = Dummy('m', integer=True, nonnegative=True)
    pieces = []
    branches: list[tuple[Expr, Boolean]] = []
    for residue in range(d):
        class_c = [as_expr(c[d*i].subs(n, d*m + residue)) for i in range(r//d + 1)]
        class_rhs = as_expr(rhs.subs(n, d*m + residue))
        m0 = max(0, -((residue - n0)//d))
        def class_direct(v: int, residue: int = residue) -> Optional[Expr]:
            return direct(d*v + residue)
        solved = _solve_and_check(class_c, class_rhs, m, m0, class_direct)
        if solved is None:
            return None
        solution, exceptions = solved
        pieces += [(value, as_boolean(Eq(n, d*v + residue))) for v, value in exceptions]
        branch = _simplest(as_expr(solution.subs(m, (n - residue)/d)), n)
        condition = true if residue == d - 1 else as_boolean(Eq(Mod(n, d), residue))
        branches.append((branch, condition))
    return as_expr(Piecewise(*pieces, *branches))


def _reversed_at(expr: Expr, n: Symbol, v: int) -> bool:
    """Whether an unevaluated ``Sum`` of ``expr`` has, at ``n = v``, an
    upper limit below its lower limit minus one: SymPy reads such a sum
    by Karr's convention (``Sum(t, (j, 2, 0)) = -t(1)``), where the
    empty-sum reading gives 0, so the closed form is not given there."""
    for node in expr.atoms(Sum):
        assert isinstance(node, Sum)
        for limit in node.limits:
            length = as_expr(as_expr(limit[2] - limit[1] + 1).subs(n, v))
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
    replacements: dict[Sum, Expr] = {}
    for node in expr.atoms(Sum):
        assert isinstance(node, Sum)
        if len(node.limits) != 1:
            continue
        j, lo, hi = node.limits[0]
        if not isinstance(j, Symbol) or not isinstance(lo, Integer) or not as_expr(hi).is_polynomial(n):
            continue
        upper = as_expr(hi)
        if Poly(upper, n).degree() > 1 or Poly(upper, n).LC() < 0:
            continue
        bottom = as_expr(upper.subs(n, start)) + 1
        if not isinstance(bottom, Integer) or bottom >= lo:
            continue
        term = as_expr(node.function)
        taken: list[Expr] = []
        for t in range(int(bottom), int(lo)):
            value = as_expr(term.subs(j, t))
            if not _finite(value):
                return None
            taken.append(value)
        replacements[node] = as_expr(Sum(term, (j, bottom, upper)) - Add(*taken))
    return as_expr(expr.xreplace(replacements)) if replacements else expr


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
        equations = []
        for v in range(n0, n0 + r + _CHECKS):
            value = direct(v)
            at = attempt(lambda: as_expr(solution.subs(n, v).doit()), _STEP_SECONDS)
            if value is None or at is None or not _finite(at):
                continue
            equations.append(as_expr(at - value))
            if len(equations) == len(constants):
                break
        if len(equations) < len(constants):
            return None
        solved = attempt(lambda: linsolve(equations, constants), _STEP_SECONDS)
        if solved is None or not solved.args:
            return None
        values = solved.args[0]
        if any(v.has(*constants) for v in values):
            return None
        solution = as_expr(solution.xreplace(dict(zip(constants, values))))
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
        at = attempt(lambda: as_expr(solution.subs(n, v).doit()), _STEP_SECONDS)
        agree = None if expected is None or at is None or not _finite(at) else _agree(expected, at)
        if v >= n0:
            if agree is not True:
                return None
        elif agree is not True and expected is not None:
            exceptions.append((v, expected))
    return solution, exceptions

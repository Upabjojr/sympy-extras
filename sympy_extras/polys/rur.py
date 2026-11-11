r"""The rational univariate representation of a zero-dimensional ideal
(Rouillier).

Let $I \subset \mathbb{Q}[x_1, \ldots, x_n]$ be zero-dimensional, $A =
\mathbb{Q}[x]/I$ its quotient algebra of dimension $D$ and $Z$ its set of
complex zeros, $\mu(\alpha)$ the multiplicity of $\alpha \in Z$. A linear
form $u = x_1 + c_2 x_2 + \cdots + c_n x_n$ is *separating* when it takes
distinct values at the points of $Z$; then $Z$ is in bijection with the
roots of

$$\bar f(T) = \prod_{\alpha \in Z} (T - u(\alpha)),$$

the squarefree part of the characteristic polynomial $\chi_u$ of the
multiplication by $u$ on $A$, and the coordinates of the points are
rational functions of the root:

$$x_i(\alpha) = \frac{g_{x_i}(u(\alpha))}{g_1(u(\alpha))}, \qquad
  g_v(T) = \sum_{\alpha \in Z} \mu(\alpha) v(\alpha)
           \prod_{\beta \ne \alpha} (T - u(\beta)).$$

Everything is computed from traces: with $\bar f = \sum_{i=0}^{\delta} a_i
T^{\delta - i}$ and the Horner polynomials $H_j = \sum_{i \le j} a_i
T^{j - i}$,

$$g_v(T) = \sum_{j=0}^{\delta - 1} \operatorname{Tr}(v u^j)\, H_{\delta - 1 - j}(T),$$

since $\sum_j u(\alpha)^j H_{\delta - 1 - j}(T) = \bar f(T)/(T - u(\alpha))$
at a root. The traces $\operatorname{Tr}(v u^j)$ are the linear forms
$h \mapsto \operatorname{Tr}(u^j h)$ applied to $v$, each obtained from the
previous one through the transposed multiplication matrix of $u$; the
power sums $\operatorname{Tr}(u^j)$ give $\chi_u$ by Newton's identities;
$u$ is separating exactly when $\deg \bar f$ equals the number of distinct
zeros, the rank of Hermite's quadratic form
(:mod:`sympy_extras.polys.hermite`). The candidates are the first forms of
:func:`~sympy_extras.polys.ideals.separating_forms`, then forms with random
coefficients, and each is tested modulo a large prime: the squarefree part
of $\chi_u$ modulo $p$ has at most the degree of the one over the
rationals, itself at most the number of distinct zeros, so that equality
modulo $p$ proves the form separating (the exact computation over the
rationals is done once, for the form found). The representation is exact
and the real zeros of $I$ correspond to the real roots of $\bar f$.

SymPy solves zero-dimensional systems through a lexicographic Gröbner
basis (``solve_poly_system``), which fails or grows large when the basis
is not in shape position, and expresses the solutions with nested
``CRootOf`` objects; the rational univariate representation gives every
coordinate as a rational function of one algebraic number.

Examples
========

>>> from sympy.abc import x, y, t
>>> from sympy_extras.polys.rur import rur
>>> R = rur([x**2 + y**2 - 1, x - y], x, y, t=t)
>>> R.polynomial
Poly(t**2 - 1/2, t, domain='QQ')
>>> R.polynomials()
[Poly(t, t, domain='QQ'), Poly(t, t, domain='QQ')]
>>> R.solutions()
[{x: -sqrt(2)/2, y: -sqrt(2)/2}, {x: sqrt(2)/2, y: sqrt(2)/2}]

References
==========

.. [1] F. Rouillier, *Solving zero-dimensional systems through the
       rational univariate representation*, Applicable Algebra in
       Engineering, Communication and Computing 9 (1999), 433-461.
.. [2] S. Basu, R. Pollack, M.-F. Roy, *Algorithms in Real Algebraic
       Geometry*, 2nd ed., Springer, 2006, Section 12.4 (the univariate
       representation).
"""
from __future__ import annotations

import random
from typing import Iterator, Optional, Sequence, Union

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polytools import Poly, invert

from sympy_extras._typing import DomainElement, as_expr, as_symbol

# the module, not its names: hermite and ideals_sympy import this module back
from . import hermite as _hermite
from .ideals_sympy import Ideal, separating_forms

__all__ = ['rur', 'RationalUnivariateRepresentation']


class RationalUnivariateRepresentation:
    """The rational univariate representation of a zero-dimensional ideal:
    the separating form ``form`` in the variables ``symbols``, the
    squarefree polynomial ``polynomial`` in ``t`` whose roots are the
    values of the form at the zeros, the characteristic polynomial
    ``characteristic`` of the form (its roots with the multiplicities of
    the zeros), and the coordinates ``numerators[i]/denominator`` of the
    zero at a root.

    Examples
    ========

    >>> from sympy.abc import x, y, t
    >>> from sympy_extras.polys.rur import rur
    >>> R = rur([x**2 - 2, y**2 - 3], x, y, t=t)
    >>> R.form, R.polynomial
    (x + y, Poly(t**4 - 10*t**2 + 1, t, domain='QQ'))
    >>> R.denominator, R.numerators
    (Poly(4*t**3 - 20*t, t, domain='QQ'), [Poly(8*t**2 + 8, t, domain='QQ'), Poly(12*t**2 - 12, t, domain='QQ')])
    >>> R.coordinates(2)
    [-5, -9/2]
    """

    def __init__(self, symbols: Sequence[Symbol], t: Symbol, form: Expr, polynomial: Poly, characteristic: Poly,
                 denominator: Poly, numerators: Sequence[Poly]) -> None:
        self.symbols: list[Symbol] = list(symbols)
        self.t: Symbol = t
        self.form: Expr = form
        self.polynomial: Poly = polynomial
        self.characteristic: Poly = characteristic
        self.denominator: Poly = denominator
        self.numerators: list[Poly] = list(numerators)

    def __repr__(self) -> str:
        return 'RationalUnivariateRepresentation(%s, %s, %s)' % (self.form, self.polynomial.as_expr(),
                                                                 [g.as_expr() for g in self.numerators])

    def coordinates(self, root: Union[Expr, int]) -> list[Expr]:
        """The coordinates of the zero at a root of ``polynomial`` (the
        rational functions evaluated at the root).

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.rur import rur
        >>> rur([x**2 - 2, y - x - 1], x, y, t=t).coordinates(sqrt(2))
        [sqrt(2), 1 + sqrt(2)]
        """
        value = as_expr(root)
        d = as_expr(self.denominator.as_expr().subs(self.t, value))
        return [as_expr(as_expr((g.as_expr().subs(self.t, value)/d).radsimp()).expand()) for g in self.numerators]

    def polynomials(self) -> list[Poly]:
        """The coordinates as polynomials in ``t`` of degree less than that
        of ``polynomial``: ``numerators[i]`` times the inverse of
        ``denominator`` modulo ``polynomial`` (the form of a lexicographic
        Gröbner basis in shape position).

        Examples
        ========

        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.rur import rur
        >>> rur([x**2 - 2, y**2 - 3], x, y, t=t).polynomials()
        [Poly(1/2*t**3 - 9/2*t, t, domain='QQ'), Poly(-1/2*t**3 + 11/2*t, t, domain='QQ')]
        """
        if self.polynomial.degree() <= 0:
            return [Poly(0, self.t, domain=QQ) for _ in self.numerators]
        inverse = invert(self.denominator, self.polynomial)
        return [(g*inverse).rem(self.polynomial) for g in self.numerators]

    def solutions(self, real: bool = False) -> list[dict[Symbol, Expr]]:
        """The zeros of the ideal, each as a dictionary of the variables:
        the roots of ``polynomial`` (:class:`~sympy.CRootOf` objects, or
        radicals when SymPy finds them) substituted in :meth:`polynomials`;
        with ``real=True`` the real zeros only.

        Examples
        ========

        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.rur import rur
        >>> rur([x**2 - 2, y - x**3], x, y, t=t).solutions()
        [{x: -sqrt(2), y: -2*sqrt(2)}, {x: sqrt(2), y: 2*sqrt(2)}]
        >>> rur([x**2 + 1, y - x], x, y, t=t).solutions(real=True)
        []
        """
        if self.polynomial.degree() <= 0:
            return []
        polys = self.polynomials()
        roots = self.polynomial.real_roots() if real else self.polynomial.all_roots()
        result = []
        for r in roots:
            root = as_expr(r)
            result.append({s: as_expr(as_expr(g.as_expr().subs(self.t, root)).expand()) for s, g in zip(self.symbols, polys)})
        return result

    def numerical_solutions(self, digits: int = 15) -> list[dict[Symbol, Expr]]:
        """The zeros numerically: the roots of ``polynomial`` to
        ``digits`` digits substituted in :meth:`polynomials` (much faster
        than evaluating :meth:`solutions`, whose ``CRootOf`` objects are
        refined at every evaluation).

        Examples
        ========

        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.rur import rur
        >>> rur([x**2 - 2, y - x**3], x, y, t=t).numerical_solutions(6)
        [{x: -1.41421, y: -2.82843}, {x: 1.41421, y: 2.82843}]
        """
        if self.polynomial.degree() <= 0:
            return []
        polys = self.polynomials()
        # the coordinate polynomials have large coefficients: the roots are
        # found with as many extra digits as the coefficients have
        height = max((abs(int(c.numerator)) for g in polys for c in g.coeffs()), default=1)
        extra = len(str(height)) + self.polynomial.degree()
        result = []
        for root in self.polynomial.nroots(n=digits + extra):
            value = as_expr(root)
            result.append({s: as_expr(as_expr(g.as_expr().subs(self.t, value)).evalf(digits)) for s, g in zip(self.symbols, polys)})
        return result

    def multiplicity(self, root: Union[Expr, int]) -> int:
        """The multiplicity of the zero at a root of ``polynomial``: the
        multiplicity of the root in ``characteristic``.

        Examples
        ========

        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.rur import rur
        >>> rur([(x - 1)**2, y - x], x, y, t=t).multiplicity(1)
        2
        """
        value = as_expr(root)
        count = 0
        p = self.characteristic
        linear = Poly(self.t - value, self.t)
        while p.degree() > 0:
            q, r = p.div(linear)
            if not r.is_zero:
                break
            p, count = q, count + 1
        return count


def _newton_characteristic(power_sums: _hermite.Vector, t: Symbol) -> Poly:
    """The characteristic polynomial of an endomorphism in dimension ``D``
    from the power sums ``Tr(u), ..., Tr(u**D)`` (Newton's identities, in
    characteristic zero)."""
    D = len(power_sums)
    e: list[DomainElement] = [QQ.one]
    for k in range(1, D + 1):
        total = QQ.zero
        for i in range(1, k + 1):
            term = e[k - i]*power_sums[i - 1]
            total += term if i % 2 else -term
        e.append(total/QQ(k))
    return Poly({D - k: e[k] if k % 2 == 0 else -e[k] for k in range(D + 1)}, t, domain=QQ)


def _horner(coefficients: Sequence[DomainElement], j: int, t: Symbol) -> Poly:
    """``H_j = sum_{i <= j} a_i t**(j - i)`` for ``f = sum a_i t**(delta - i)``."""
    return Poly({j - i: coefficients[i] for i in range(j + 1)}, t, domain=QQ)


def rur(system: _hermite.System, *gens: Symbol, t: Optional[Symbol] = None) -> RationalUnivariateRepresentation:
    """The rational univariate representation of a zero-dimensional
    system of polynomial equations with rational coefficients (Rouillier's
    algorithm), with the first separating form among the candidates of
    :func:`~sympy_extras.polys.ideals.separating_forms`.

    Parameters
    ==========

    system : Ideal or list of Expr or Poly
        The ideal, or the polynomials (equated to zero).
    gens : Symbols
        The variables, when ``system`` is a list of polynomials (all its
        free symbols by default).
    t : Symbol
        The variable of the representation (``t`` by default).

    Raises
    ======

    NotImplementedError
        If the system has infinitely many complex solutions or its
        coefficients are not rational.

    Examples
    ========

    >>> from sympy.abc import x, y, z, t
    >>> from sympy_extras.polys.rur import rur
    >>> R = rur([x**2 + y**2 - 1, x - y], x, y, t=t)
    >>> R.form, R.polynomial, R.denominator, R.numerators
    (x, Poly(t**2 - 1/2, t, domain='QQ'), Poly(2*t, t, domain='QQ'), [Poly(1, t, domain='QQ'), Poly(1, t, domain='QQ')])

    Neither ``x`` nor ``x + y`` separates the zeros of the next system
    (two of them have ``x = 0``, two have ``x + y = 1``), ``x + 2y`` does:

    >>> R = rur([x**2 + y**2 - 1, x*y], x, y, t=t)
    >>> R.form, R.polynomial
    (x + 2*y, Poly(t**4 - 5*t**2 + 4, t, domain='QQ'))
    >>> R.polynomials()
    [Poly(-1/3*t**3 + 4/3*t, t, domain='QQ'), Poly(1/6*t**3 - 1/6*t, t, domain='QQ')]
    >>> R.solutions()
    [{x: 0, y: -1}, {x: -1, y: 0}, {x: 1, y: 0}, {x: 0, y: 1}]

    A system with a zero of multiplicity three:

    >>> R = rur([x**3, y - x - 1, z - y**2], x, y, z, t=t)
    >>> R.polynomial, R.characteristic, R.multiplicity(0)
    (Poly(t, t, domain='QQ'), Poly(t**3, t, domain='QQ'), 3)
    >>> R.solutions()
    [{x: 0, y: 1, z: 1}]
    """
    T = Symbol('t') if t is None else as_symbol(t)
    A = _hermite._algebra(system, gens)
    ideal: Ideal = A.ideal
    symbols = list(ideal.symbols)
    if A.empty:
        one = Poly(1, T, domain=QQ)
        return RationalUnivariateRepresentation(symbols, T, as_expr(symbols[0]) if symbols else S.Zero, one, one,
                                                Poly(0, T, domain=QQ), [Poly(0, T, domain=QQ) for _ in symbols])
    D = len(A.basis)
    # the number of distinct zeros: the rank of Hermite's quadratic form
    count = DomainMatrix(A.hermite(A.trace), (D, D), QQ).rank()
    coordinates = [A.coordinates(A.element(s)) for s in symbols]
    one_index = A.index[tuple(0 for _ in symbols)]
    for form in _candidates(symbols, count):
        transposed = A.transposed_multiplication(A.element(form))
        if not _is_separating(A.trace, transposed, one_index, count, T):
            continue
        # the linear forms h -> Tr(u**j h), j = 0, ..., D
        forms: list[_hermite.Vector] = [list(A.trace)]
        for _ in range(D):
            forms.append(A.compose(forms[-1], transposed))
        characteristic = _newton_characteristic([f[one_index] for f in forms[1:]], T)
        squarefree = characteristic.quo(characteristic.gcd(characteristic.diff(T)))
        delta = squarefree.degree()
        if delta != count:
            raise AssertionError('the modular separating test is a proof')
        coefficients = squarefree.all_coeffs()
        horner = [_horner(coefficients, j, T) for j in range(delta)]

        def g_of(values: _hermite.Vector) -> Poly:
            total = Poly(0, T, domain=QQ)
            for j in range(delta):
                total += horner[delta - 1 - j]*values[j]
            return total

        denominator = g_of([f[one_index] for f in forms])
        numerators = [g_of([sum((a*b for a, b in zip(f, v)), QQ.zero) for f in forms]) for v in coordinates]
        return RationalUnivariateRepresentation(symbols, T, form, squarefree, characteristic, denominator, numerators)
    raise AssertionError('no separating form among the candidates')


#: the prime of the modular separating test
_PRIME = 2**61 - 1
#: how many of the small forms of ``separating_forms`` are tried before
#: random forms (the small forms fail in turn on symmetric systems: cyclic-5
#: needs the 124th)
_SMALL_FORMS = 6


def _candidates(symbols: list[Symbol], count: int) -> Iterator[Expr]:
    """Candidate separating forms: the first small forms, then forms with
    random coefficients bounded by the number of pairs of zeros in two
    variables and by the number of zeros otherwise (Schwartz–Zippel: a
    random form fails with probability at most 1/2 and 1/8)."""
    for form, _ in zip(separating_forms(symbols, count), range(_SMALL_FORMS)):
        yield form
    pairs = count*(count - 1)//2
    bound = max(2, pairs if len(symbols) == 2 else count)
    rng = random.Random(count)
    while True:
        candidate: Expr = symbols[0]
        for s in symbols[1:]:
            candidate += rng.randint(-bound, bound)*s
        yield candidate


def _is_separating(trace: _hermite.Vector, transposed: _hermite.Rows, one_index: int, count: int, t: Symbol) -> bool:
    """Whether the form with the given transposed multiplication matrix is
    separating, decided modulo a prime: the squarefree part of its
    characteristic polynomial modulo ``p`` has at most the degree of the
    one over the rationals, which is at most the number of distinct zeros,
    so that equality modulo ``p`` is a proof."""
    p = _PRIME

    def mod(q: DomainElement) -> int:
        return int(q.numerator)*pow(int(q.denominator), -1, p) % p

    try:
        current = [mod(c) for c in trace]
        rows = [[mod(c) for c in row] for row in transposed]
    except ValueError:
        # a denominator divisible by the prime: no decision, the next form
        return False
    D = len(trace)
    power_sums: list[int] = []
    for _ in range(D):
        current = [sum(a*b for a, b in zip(row, current)) % p for row in rows]
        power_sums.append(current[one_index])
    e = [1]
    for k in range(1, D + 1):
        total = 0
        for i in range(1, k + 1):
            term = e[k - i]*power_sums[i - 1]
            total += term if i % 2 else -term
        e.append(total*pow(k, -1, p) % p)
    f = Poly({D - k: e[k] if k % 2 == 0 else -e[k] for k in range(D + 1)}, t, modulus=p)
    return D - f.gcd(f.diff(t)).degree() == count

r"""The rational univariate representation of a zero-dimensional ideal
(Rouillier), on the typed polynomial layer (no SymPy).

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

The systems are the ideals of :mod:`sympy_extras.polys.ideals` over ``QQ``
or lists of core expressions; the polynomials of the representation are
:class:`~sympy_extras.polys.polyclasses.DensePoly` objects over ``QQ``.
The exact solutions (:meth:`RationalUnivariateRepresentation.solutions`)
are written with the roots of the univariate polynomial in SymPy's order:
radicals for its linear and quadratic factors and binomials, the real
roots of the other factors as
:class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot` and the others as
:class:`~sympy_extras.polys.complexrootof.ComplexRootOf` (SymPy's
``CRootOf``). :mod:`sympy_extras.polys.rur_sympy` is the SymPy face of
this module.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.polys.rur import rur
>>> x, y, t = symbols('x y t')
>>> R = rur([x**2 + y**2 - 1, x - y], x, y, t=t)
>>> R.polynomial.to_expr()
t**2 - 1/2
>>> [p.to_expr() for p in R.polynomials()]
[t, t]
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
from fractions import Fraction
from typing import Final, Iterator, Optional, Sequence, Union

import mpmath

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import expand
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow as power
from sympy_extras.core.symbol import Symbol

from .complexrootof import polynomial_roots, real_roots_indexed
from .domains.finitefield import FiniteField
from .domains.rationalfield import QQ
# the module, not its names: hermite imports ideals, which imports this module back
from . import hermite as _hermite
from .ideals import separating_forms
from .numberfields.minpoly import minimal_polynomial_coeffs
from .polyclasses import DensePoly
# the module, not its names: it imports ideals, which imports this module
from .regularchains import regularchain as _chains

__all__ = ['rur', 'RationalUnivariateRepresentation', 'Point', 'exact_solutions']

#: a zero of the ideal: the values of the variables
Point = dict[Symbol, Expr]

_ZERO = Fraction(0)
_ONE = Fraction(1)


def exact_solutions(polynomial: DensePoly[Fraction], polys: Sequence[DensePoly[Fraction]],
                    symbols: Sequence[Symbol], real: bool = False) -> list[Point]:
    """The zeros of a representation: the roots of the univariate
    ``polynomial`` in SymPy's order (as SymPy's ``Poly.all_roots``, with
    ``real`` ``Poly.real_roots``: radicals for the linear and quadratic
    factors and the binomials, the real roots of the other factors as
    ``IndexedRoot``, the others as ``CRootOf``) substituted in the
    coordinate polynomials ``polys`` of the ``symbols``, expanded.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.domains.rationalfield import QQ
    >>> from sympy_extras.polys.polyclasses import DensePoly
    >>> from sympy_extras.polys.rur import exact_solutions
    >>> x, t = symbols('x t')
    >>> f = DensePoly([Fraction(1), Fraction(0), Fraction(-1), Fraction(-1)], QQ, t)
    >>> exact_solutions(f, [DensePoly([Fraction(1), Fraction(0), Fraction(0)], QQ, t)], [x])[:2]
    [{x: IndexedRoot(t**3 - t - 1, t, 0)**2}, {x: CRootOf(t**3 - t - 1, 1)**2}]
    """
    t = polynomial.gen
    if not isinstance(t, Symbol):
        raise TypeError("the polynomial is not in a symbol: %s" % (t,))
    roots = [real_roots_indexed(r) for r in polynomial_roots(polynomial.rep, t, real=real)]
    exprs = [g.to_expr() for g in polys]
    return [{s: expand(as_expr(e.subs(t, root))) for s, e in zip(symbols, exprs)} for root in roots]


def _constant(c: Fraction, t: Symbol) -> DensePoly[Fraction]:
    return DensePoly([c] if c else [], QQ, t)


class RationalUnivariateRepresentation:
    """The rational univariate representation of a zero-dimensional ideal:
    the separating form ``form`` in the variables ``symbols``, the
    squarefree polynomial ``polynomial`` in ``t`` whose roots are the
    values of the form at the zeros, the characteristic polynomial
    ``characteristic`` of the form (its roots with the multiplicities of
    the zeros), and the coordinates ``numerators[i]/denominator`` of the
    zero at a root. The polynomials are dense polynomials over ``QQ`` in
    ``t``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.rur import rur
    >>> x, y, t = symbols('x y t')
    >>> R = rur([x**2 - 2, y**2 - 3], x, y, t=t)
    >>> R.form, R.polynomial.to_expr()
    (x + y, t**4 - 10*t**2 + 1)
    >>> R.denominator.to_expr(), [g.to_expr() for g in R.numerators]
    (4*t**3 - 20*t, [8*t**2 + 8, 12*t**2 - 12])
    >>> R.coordinates(2)
    [-5, -9/2]
    """

    def __init__(self, symbols: Sequence[Symbol], t: Symbol, form: Expr, polynomial: DensePoly[Fraction],
                 characteristic: DensePoly[Fraction], denominator: DensePoly[Fraction],
                 numerators: Sequence[DensePoly[Fraction]]) -> None:
        self.symbols: Final[list[Symbol]] = list(symbols)
        self.t: Final[Symbol] = t
        self.form: Final[Expr] = form
        self.polynomial: Final[DensePoly[Fraction]] = polynomial
        self.characteristic: Final[DensePoly[Fraction]] = characteristic
        self.denominator: Final[DensePoly[Fraction]] = denominator
        self.numerators: Final[list[DensePoly[Fraction]]] = list(numerators)

    def __repr__(self) -> str:
        return 'RationalUnivariateRepresentation(%s, %s, %s)' % (self.form, self.polynomial.to_expr(),
                                                                 [g.to_expr() for g in self.numerators])

    def coordinates(self, root: Union[Expr, int]) -> list[Expr]:
        """The coordinates of the zero at a root of ``polynomial`` (the
        rational functions evaluated at the root, the denominators
        rationalised).

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.functions import sqrt
        >>> from sympy_extras.polys.rur import rur
        >>> x, y, t = symbols('x y t')
        >>> rur([x**2 - 2, y - x - 1], x, y, t=t).coordinates(sqrt(2))
        [sqrt(2), 1 + sqrt(2)]
        """
        value = Integer(root) if isinstance(root, int) else root
        d = self.denominator.to_expr().xreplace({self.t: value})
        result: list[Expr] = []
        for g in self.numerators:
            quotient = mul(g.to_expr().xreplace({self.t: value}), power(d, Integer(-1)))
            result.append(quotient.radsimp().expand())
        return result

    def polynomials(self) -> list[DensePoly[Fraction]]:
        """The coordinates as polynomials in ``t`` of degree less than that
        of ``polynomial``: ``numerators[i]`` times the inverse of
        ``denominator`` modulo ``polynomial`` (the form of a lexicographic
        Gröbner basis in shape position).

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.rur import rur
        >>> x, y, t = symbols('x y t')
        >>> [g.to_expr() for g in rur([x**2 - 2, y**2 - 3], x, y, t=t).polynomials()]
        [t**3/2 - 9*t/2, -t**3/2 + 11*t/2]
        """
        if self.polynomial.degree <= 0:
            return [_constant(_ZERO, self.t) for _ in self.numerators]
        inverse = self.denominator.invert(self.polynomial)
        return [(g*inverse).rem(self.polynomial) for g in self.numerators]

    def solutions(self, real: bool = False) -> list[Point]:
        """The zeros of the ideal, each as a dictionary of the variables:
        the roots of ``polynomial`` in SymPy's order (radicals for the
        quadratic factors and binomials, ``IndexedRoot`` for the other real
        roots, ``CRootOf`` for the others; see :func:`exact_solutions`)
        substituted in :meth:`polynomials`; with ``real=True`` the real
        zeros only.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.rur import rur
        >>> x, y, t = symbols('x y t')
        >>> rur([x**2 - 2, y - x**3], x, y, t=t).solutions()
        [{x: -sqrt(2), y: -2*sqrt(2)}, {x: sqrt(2), y: 2*sqrt(2)}]
        >>> rur([x**2 + 1, y - x], x, y, t=t).solutions(real=True)
        []
        """
        if self.polynomial.degree <= 0:
            return []
        return exact_solutions(self.polynomial, self.polynomials(), self.symbols, real)

    def numerical_solutions(self, digits: int = 15) -> list[Point]:
        """The zeros numerically: the roots of ``polynomial`` (real ones
        first, as SymPy's ``nroots`` sorts them) substituted in
        :meth:`polynomials`, to ``digits`` digits, the real or imaginary
        parts below the precision dropped.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.rur import rur
        >>> x, y, t = symbols('x y t')
        >>> rur([x**2 - 2, y - x**3], x, y, t=t).numerical_solutions(6)
        [{x: -1.41421, y: -2.82843}, {x: 1.41421, y: 2.82843}]
        """
        if self.polynomial.degree <= 0:
            return []
        polys = self.polynomials()
        # the coordinate polynomials have large coefficients: the roots are
        # found with as many extra digits as the coefficients have
        height = max((abs(c.numerator) for g in polys for c in g.rep), default=1)
        working = digits + len(str(height)) + self.polynomial.degree
        result: list[Point] = []
        with mpmath.workdps(working):
            coefficients = [mpmath.mpc(mpmath.mpf(c.numerator)/c.denominator) for c in self.polynomial.rep]
            for root in _chains._nroots(coefficients, working):
                point: Point = {}
                for s, g in zip(self.symbols, polys):
                    value = mpmath.mpc(0)
                    for c in g.rep:
                        value = value*root + mpmath.mpf(c.numerator)/c.denominator
                    point[s] = _chains._number(value, digits)
                result.append(point)
        return result

    def multiplicity(self, root: Union[Expr, int]) -> int:
        """The multiplicity of the zero at a root of ``polynomial``: the
        multiplicity of the root in ``characteristic`` (the largest power
        of its minimal polynomial over the rationals which divides it).

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.rur import rur
        >>> x, y, t = symbols('x y t')
        >>> rur([(x - 1)**2, y - x], x, y, t=t).multiplicity(1)
        2
        """
        value = Integer(root) if isinstance(root, int) else root
        minimal = DensePoly([Fraction(c) for c in minimal_polynomial_coeffs(value)], QQ, self.t)
        count = 0
        p = self.characteristic
        while p.degree > 0:
            q, r = p.div(minimal)
            if not r.is_zero:
                break
            p, count = q, count + 1
        return count


def _newton_characteristic(power_sums: _hermite.Vector, t: Symbol) -> DensePoly[Fraction]:
    """The characteristic polynomial of an endomorphism in dimension ``D``
    from the power sums ``Tr(u), ..., Tr(u**D)`` (Newton's identities, in
    characteristic zero)."""
    D = len(power_sums)
    e: list[Fraction] = [_ONE]
    for k in range(1, D + 1):
        total = _ZERO
        for i in range(1, k + 1):
            term = e[k - i]*power_sums[i - 1]
            total += term if i % 2 else -term
        e.append(total/k)
    return DensePoly([e[k] if k % 2 == 0 else -e[k] for k in range(D + 1)], QQ, t)


def _horner(coefficients: Sequence[Fraction], j: int, t: Symbol) -> DensePoly[Fraction]:
    """``H_j = sum_{i <= j} a_i t**(j - i)`` for ``f = sum a_i t**(delta - i)``."""
    return DensePoly(list(coefficients[:j + 1]), QQ, t)


def rur(system: _hermite.System, *gens: Symbol, t: Optional[Symbol] = None) -> RationalUnivariateRepresentation:
    """The rational univariate representation of a zero-dimensional
    system of polynomial equations with rational coefficients (Rouillier's
    algorithm), with the first separating form among the candidates of
    :func:`~sympy_extras.polys.ideals.separating_forms`.

    Parameters
    ==========

    system : Ideal or list of Expr
        The ideal over ``QQ``, or the polynomials (equated to zero).
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

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.rur import rur
    >>> x, y, z, t = symbols('x y z t')
    >>> R = rur([x**2 + y**2 - 1, x - y], x, y, t=t)
    >>> R.form, R.polynomial.to_expr(), R.denominator.to_expr(), [g.to_expr() for g in R.numerators]
    (x, t**2 - 1/2, 2*t, [1, 1])

    Neither ``x`` nor ``x + y`` separates the zeros of the next system
    (two of them have ``x = 0``, two have ``x + y = 1``), ``x + 2y`` does:

    >>> R = rur([x**2 + y**2 - 1, x*y], x, y, t=t)
    >>> R.form, R.polynomial.to_expr()
    (x + 2*y, t**4 - 5*t**2 + 4)
    >>> R.solutions()
    [{x: 0, y: -1}, {x: -1, y: 0}, {x: 1, y: 0}, {x: 0, y: 1}]

    A system with a zero of multiplicity three:

    >>> R = rur([x**3, y - x - 1, z - y**2], x, y, z, t=t)
    >>> R.polynomial.to_expr(), R.characteristic.to_expr(), R.multiplicity(0)
    (t, t**3, 3)
    """
    T = Symbol('t') if t is None else t
    try:
        A = _hermite._algebra(system, gens)
    except ValueError as error:
        # a coefficient outside QQ (a symbol which is not a variable)
        raise NotImplementedError("the coefficients of the system are not rational: %s" % error)
    symbols = list(A.ideal.symbols)
    if A.empty:
        one = _constant(_ONE, T)
        return RationalUnivariateRepresentation(symbols, T, symbols[0] if symbols else Integer(0), one, one,
                                                _constant(_ZERO, T), [_constant(_ZERO, T) for _ in symbols])
    D = A.dimension
    # the number of distinct zeros: the rank of Hermite's quadratic form
    positive, negative = _hermite._inertia(A.hermite(A.trace))
    count = positive + negative
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
        squarefree = characteristic.quo(characteristic.gcd(characteristic.diff()))
        delta = squarefree.degree
        if delta != count:
            raise AssertionError('the modular separating test is a proof')
        horner = [_horner(squarefree.rep, j, T) for j in range(delta)]

        def g_of(values: _hermite.Vector) -> DensePoly[Fraction]:
            total = _constant(_ZERO, T)
            for j in range(delta):
                total = total + horner[delta - 1 - j].mul_ground(values[j])
            return total

        denominator = g_of([f[one_index] for f in forms])
        numerators = [g_of([sum((a*b for a, b in zip(f, v)), _ZERO) for f in forms]) for v in coordinates]
        return RationalUnivariateRepresentation(symbols, T, form, squarefree, characteristic, denominator, numerators)
    raise AssertionError('no separating form among the candidates')


#: the prime of the modular separating test
_PRIME: Final = 2**61 - 1
#: how many of the small forms of ``separating_forms`` are tried before
#: random forms (the small forms fail in turn on symmetric systems: cyclic-5
#: needs the 124th)
_SMALL_FORMS: Final = 6


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
        terms: list[Expr] = [symbols[0]]
        for s in symbols[1:]:
            terms.append(mul(Integer(rng.randint(-bound, bound)), s))
        yield add(*terms)


def _is_separating(trace: _hermite.Vector, transposed: _hermite.Rows, one_index: int, count: int, t: Symbol) -> bool:
    """Whether the form with the given transposed multiplication matrix is
    separating, decided modulo a prime: the squarefree part of its
    characteristic polynomial modulo ``p`` has at most the degree of the
    one over the rationals, which is at most the number of distinct zeros,
    so that equality modulo ``p`` is a proof."""
    p = _PRIME

    def mod(q: Fraction) -> int:
        return q.numerator*_inverse_mod(q.denominator, p) % p

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
        e.append(total*_inverse_mod(k, p) % p)
    coefficients = [(e[k] if k % 2 == 0 else -e[k]) % p for k in range(D + 1)]
    f = DensePoly(coefficients, FiniteField(p), t)
    return D - f.gcd(f.diff()).degree == count


def _inverse_mod(a: int, p: int) -> int:
    """The inverse of ``a`` modulo ``p`` (``ValueError`` when there is none)."""
    return pow(a, -1, p)

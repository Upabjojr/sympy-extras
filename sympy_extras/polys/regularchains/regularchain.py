"""Regular chains and triangular decompositions of polynomial systems.

A *triangular decomposition* writes the solutions of a system of polynomial
equations as a union of triangular systems, each of which is solved one
variable after the other, like a linear system in row echelon form:

>>> from sympy.abc import x, y, z
>>> from sympy_extras.polys.regularchains import triangularize
>>> for chain in triangularize([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1], x, y, z):
...     print(chain.polys)
[z**2 + 2*z - 1, y - z, x - z]
[z, y, x - 1]
[z, y - 1, x]
[z - 1, y, x]

The variables are ordered, the first one being the greatest (as for the
lexicographic Gröbner bases of SymPy). A polynomial is seen as a polynomial
in its greatest variable, its *main variable*, whose leading coefficient is
its *initial*. A *regular chain* `T` is a set of polynomials with distinct
main variables such that the initial of each of them is not a zero divisor
modulo the *saturated ideal* `\\operatorname{sat}(T') = (T') : h^\\infty` of
the polynomials `T'` below it, `h` being the product of their initials. The
solutions it stands for are its *quasi-component* `W(T) = V(T) \\setminus
V(h)`: the zeros of the chain where no initial vanishes. Its variables which
are not main variables are free: the dimension of the quasi-component is
their number.

Unlike a lexicographic Gröbner basis, the decomposition works in any
dimension, separates the components of different dimensions, and with the
parameters as smallest variables it discusses them:

>>> from sympy.abc import a, b, c
>>> for chain in triangularize([a*x**2 + b*x + c], x, a, b, c):
...     print(chain.polys, chain.initials)
[a*x**2 + b*x + c] [a]
[a, b*x + c] [1, b]
[c, b, a] [1, 1, 1]

Two decompositions are computed ([ChenMorenoMaza]_):

* the one *in the sense of Lazard* (``mode='lazard'``, the default):
  the zeros of the system are exactly the union of the quasi-components;
* the one *in the sense of Kalkbrener* (``mode='kalkbrener'``): the zeros
  are the union of the *closures* of the quasi-components, that is the
  radical of the ideal of the system is the intersection of the saturated
  ideals. It describes the generic points of every irreducible component,
  and is smaller and faster.

The chains are *squarefree*: their saturated ideals are radical, and a
polynomial belongs to the saturated ideal exactly when its pseudo-remainder
by the chain is zero.

References
==========

.. [ChenMorenoMaza] C. Chen, M. Moreno Maza, Algorithms for computing
   triangular decomposition of polynomial systems, Journal of Symbolic
   Computation 47 (2012), 610-642.
.. [ALM] P. Aubry, D. Lazard, M. Moreno Maza, On the theories of
   triangular sets, Journal of Symbolic Computation 28 (1999), 105-124.
.. [Lazard] D. Lazard, A new method for solving algebraic systems of
   positive dimension, Discrete Applied Mathematics 33 (1991), 147-160.
.. [Kalkbrener] M. Kalkbrener, A generalized Euclidean algorithm for
   computing triangular representations of algebraic varieties, Journal of
   Symbolic Computation 15 (1993), 143-167.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly
from sympy.polys.rings import PolyElement, PolyRing

from sympy_extras._typing import as_expr, as_symbol, free_symbols, sorted_symbols
from sympy_extras.polys.ideals import Ideal

from .operations import Chain, Decomposer
from .recursive import degree, initial, main_variable, normalize
from .solutions import exact_solutions

__all__ = ['RegularChain', 'triangularize', 'regular_gcd']

#: an equation: a polynomial expression which vanishes, or an equality
Equation = Union[Expr, Equality, int]


def _polynomial(ring: PolyRing, symbols: Sequence[Symbol], equation: Equation) -> PolyElement:
    """The polynomial over the integers of an equation (the numerator of a
    polynomial with rational coefficients)."""
    given = sympify(equation)
    expr = as_expr(given.lhs - given.rhs) if isinstance(given, Equality) else as_expr(given)
    unknown = free_symbols(expr) - set(symbols)
    if unknown:
        raise ValueError("symbols which are not variables: %s" % sorted_symbols(unknown))
    try:
        poly = Poly(expr, *symbols, domain='QQ')
    except PolynomialError:
        raise ValueError("not a polynomial in the variables: %s" % expr)
    except Exception:
        raise ValueError("not a polynomial with rational coefficients: %s" % expr)
    result: PolyElement = ring.from_expr(poly.clear_denoms()[1].as_expr())
    return result


def _variables(equations: Sequence[Equation], symbols: Sequence[Symbol]) -> tuple[Symbol, ...]:
    """The given variables followed by the other symbols, as parameters."""
    variables = [as_symbol(s) for s in symbols]
    if len(set(variables)) != len(variables):
        raise ValueError("repeated variables")
    found: set[Symbol] = set()
    for equation in equations:
        given = sympify(equation)
        if isinstance(given, Equality):
            found |= free_symbols(as_expr(given.lhs)) | free_symbols(as_expr(given.rhs))
        else:
            found |= free_symbols(as_expr(given))
    variables.extend(sorted_symbols(found - set(variables)))
    if not variables:
        raise ValueError("no variables")
    return tuple(variables)


class RegularChain:
    """A squarefree regular chain of polynomials with rational coefficients.

    Parameters
    ==========

    polys : iterable of Expr
        Polynomials with distinct main variables.
    symbols : Symbols
        The variables, the greatest first. The other symbols of the
        polynomials are put after them, as the smallest variables.

    Raises
    ======

    ValueError
        if two polynomials have the same main variable, if an initial is a
        zero divisor modulo the polynomials below it, or if a polynomial is
        not squarefree modulo them.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.regularchains import RegularChain
    >>> T = RegularChain([z**2 - 2, y**2 - z, (y + 1)*x - 1], x, y, z)
    >>> T.main_variables, T.dimension, T.degree
    ([z, y, x], 0, 4)
    >>> T.initials
    [1, 1, y + 1]

    Membership in the saturated ideal is decided by pseudo-division:

    >>> T.contains(y**4 - 2), T.contains(x*z - x - y + 1)
    (True, True)
    >>> T.reduce(y**3 + x)
    y*z + 3

    A polynomial is regular, zero, or splits the chain:

    >>> T.is_regular(y - z)
    True
    >>> S = RegularChain([y**2 - 1, x**2 - y], x, y)
    >>> S.regularize(y - 1)
    ([RegularChain([y - 1, x**2 - y], x, y)], [RegularChain([y + 1, x**2 - y], x, y)])

    The initial of ``(y - z)*x - 1`` vanishes at one of the two roots of the
    polynomial below it:

    >>> RegularChain([y**2 - z*y, (y - z)*x - 1], x, y, z)
    Traceback (most recent call last):
    ...
    ValueError: the initial y - z is a zero divisor modulo the polynomials below
    """

    def __init__(self, polys: Iterable[Equation], *symbols: Symbol) -> None:
        equations = list(polys)
        self.symbols: tuple[Symbol, ...] = _variables(equations, symbols)
        self._ring: PolyRing = PolyRing(self.symbols, ZZ)
        self._decomposer: Decomposer = Decomposer(self._ring)
        elements = [normalize(_polynomial(self._ring, self.symbols, p)) for p in equations]
        levels = [main_variable(p) for p in elements]
        if any(level == self._ring.ngens for level in levels):
            raise ValueError("a constant polynomial")
        if len(set(levels)) != len(levels):
            raise ValueError("two polynomials with the same main variable")
        chain = self._decomposer.empty
        for level, p in sorted(zip(levels, elements), key=lambda pair: -pair[0]):
            if not self._decomposer.is_regular(initial(p), chain):
                raise ValueError("the initial %s is a zero divisor modulo the polynomials below"
                    % initial(p).as_expr())
            added = [t for t in (T.get(level) for T in self._decomposer.add(chain, p)) if t is not None]
            if len(added) != 1 or degree(added[0], level) != degree(p, level):
                raise ValueError("%s is not squarefree modulo the polynomials below" % p.as_expr())
            chain = chain.with_poly(level, p)
        self._chain: Chain = chain

    @classmethod
    def _new(cls, chain: Chain, symbols: tuple[Symbol, ...], ring: PolyRing, decomposer: Decomposer) -> RegularChain:
        self = object.__new__(cls)
        self.symbols = symbols
        self._ring = ring
        self._decomposer = decomposer
        self._chain = chain
        return self

    def with_symbols(self, *symbols: Symbol) -> RegularChain:
        """The same chain in a ring with more variables, which are free: the
        variables of the chain must come in the same order among the new ones.

        Examples
        ========

        >>> from sympy.abc import t, x, y
        >>> from sympy_extras.polys.regularchains import RegularChain
        >>> T = RegularChain([y**2 - 2, x - y], x, y).with_symbols(x, t, y)
        >>> T.free_variables, T.contains(t*x**2 - 2*t)
        ([t], True)
        """
        variables = tuple(as_symbol(s) for s in symbols)
        if len(set(variables)) != len(variables):
            raise ValueError("repeated variables")
        if [s for s in variables if s in self.symbols] != list(self.symbols):
            raise ValueError("the variables %s must be kept, in this order" % (self.symbols,))
        ring = PolyRing(variables, ZZ)
        decomposer = Decomposer(ring)
        polys = [ring.from_expr(p) for p in self.polys]
        return RegularChain._new(decomposer.empty.union(polys), variables, ring, decomposer)

    def _wrap(self, chain: Chain) -> RegularChain:
        return RegularChain._new(chain, self.symbols, self._ring, self._decomposer)

    def _element(self, f: Equation) -> PolyElement:
        return _polynomial(self._ring, self.symbols, f)

    # ------------------------------------------------------------------
    # description

    @property
    def polys(self) -> list[Expr]:
        """The polynomials, by increasing main variable: the order in which
        the triangular system is solved."""
        return [as_expr(p.as_expr()) for p in self._chain.elements()]

    @property
    def main_variables(self) -> list[Symbol]:
        """The main variables of the polynomials, in the same order."""
        return [self.symbols[main_variable(p)] for p in self._chain.elements()]

    @property
    def free_variables(self) -> list[Symbol]:
        """The variables which are the main variable of no polynomial."""
        return [s for s, p in zip(self.symbols, self._chain.polys) if p is None]

    @property
    def initials(self) -> list[Expr]:
        """The leading coefficients in the main variables: the solutions of
        the chain are its zeros where none of them vanishes."""
        return [as_expr(initial(p).as_expr()) for p in self._chain.elements()]

    @property
    def height(self) -> int:
        """The number of polynomials."""
        return self._chain.height

    @property
    def dimension(self) -> int:
        """The dimension of the quasi-component: the number of free variables."""
        return len(self.symbols) - self._chain.height

    @property
    def degree(self) -> int:
        """The product of the degrees in the main variables: the number of
        solutions for generic values of the free variables."""
        result = 1
        for p in self._chain.elements():
            result *= degree(p, main_variable(p))
        return result

    def __repr__(self) -> str:
        return 'RegularChain(%s, %s)' % (self.polys, ', '.join(str(s) for s in self.symbols))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RegularChain) and self.symbols == other.symbols and self._chain == other._chain

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((self.symbols, self._chain))

    # ------------------------------------------------------------------
    # the saturated ideal

    def reduce(self, f: Equation) -> Expr:
        """The pseudo-remainder of ``f`` by the chain, primitive over the
        integers: `h f = r` modulo the chain for a product `h` of divisors of
        the initials, the degree of `r` in every main variable being lower
        than the one of the polynomial of the chain."""
        return as_expr(self._decomposer.reduce(self._element(f), self._chain).as_expr())

    def contains(self, f: Equation) -> bool:
        """Whether ``f`` is in the saturated ideal, that is vanishes on the
        quasi-component."""
        return not self._decomposer.reduce(self._element(f), self._chain)

    def __contains__(self, f: Equation) -> bool:
        return self.contains(f)

    def saturated_ideal(self) -> Ideal:
        """The saturated ideal, by a Gröbner basis computation.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.regularchains import RegularChain
        >>> RegularChain([z*y - 1, y*x - z], x, y, z).saturated_ideal().reduced().exprs
        [x*y - z, y*z - 1, -x + z**2]
        """
        ideal = Ideal(self.polys if self.polys else [Integer(0)], *self.symbols)
        if not self.polys:
            return ideal
        return ideal.saturate(as_expr(Mul(*self.initials)))

    def numerical_solutions(self, n: int = 15) -> list[dict[Symbol, Expr]]:
        """The points of a chain without free variables, numerically.

        The triangular system is solved from the bottom: the roots of the
        first polynomial, then for each of them the roots of the second
        one, and so on. The chain being squarefree, all these roots are
        simple, and no initial vanishes on the way.

        Parameters
        ==========

        n : int
            The number of digits; the computation uses twice as many, and
            as many more as the coefficients of the chain have.

        Returns
        =======

        list of dict
            The `\\deg` solutions, complex in general.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.regularchains import triangularize
        >>> [chain] = triangularize([x**2 + y**2 - 5, x*y - 1], x, y)
        >>> chain.polys
        [y**4 - 5*y**2 + 1, x*y - 1]
        >>> for solution in chain.numerical_solutions(5):
        ...     print(solution)
        {y: -2.1889, x: -0.45685}
        {y: -0.45685, x: -2.1889}
        {y: 0.45685, x: 2.1889}
        {y: 2.1889, x: 0.45685}
        """
        if self.dimension:
            raise ValueError("the chain has the free variables %s: its solutions are not isolated"
                % self.free_variables)
        # the coefficients of a chain are large, and as many digits cancel
        digits = 2*n + 10 + sum(len(str(p.max_norm())) for p in self._chain.elements())
        solutions: list[dict[Symbol, Expr]] = [{}]
        for p, v in zip(self.polys, self.main_variables):
            following: list[dict[Symbol, Expr]] = []
            for solution in solutions:
                poly = Poly(p.xreplace(solution), v)
                for root in poly.nroots(n=digits, maxsteps=200):
                    extended = dict(solution)
                    extended[v] = as_expr(root)
                    following.append(extended)
            solutions = following
        return [{v: as_expr(value.n(n, chop=True)) for v, value in solution.items()} for solution in solutions]

    def solutions(self, real: bool = False) -> list[dict[Symbol, Expr]]:
        """The points of a chain without free variables, exactly.

        The coordinates are rational numbers, root objects (``CRootOf``,
        in radicals where the package writes them so) and rational
        functions of them, see :mod:`.solutions`: the chain is solved from
        the bottom; a polynomial of degree one in its main variable gives
        its coordinate by a division (written as a root object of its own
        polynomial when it involves root objects), and the roots of another one, whose
        coefficients are algebraic numbers, are root objects of a
        polynomial with rational coefficients (an iterated resultant)
        which are told from its other roots numerically.

        Parameters
        ==========

        real : bool
            The real points only; whether a root object is real is decided
            exactly.

        Raises
        ======

        ValueError
            When the chain has free variables.
        NotImplementedError
            When the roots are not told apart at the working precision.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.regularchains import triangularize
        >>> [chain] = triangularize([x**2 + y**2 - 5, x*y - 1], x, y)
        >>> chain.solutions()[0]
        {y: CRootOf(x**4 - 5*x**2 + 1, 0), x: CRootOf(x**4 - 5*x**2 + 1, 1)}
        >>> [chain] = triangularize([x**5 - x - 1 - y, y**2 - 2], x, y)
        >>> len(chain.solutions()), chain.solutions(real=True)[0]
        (10, {y: -sqrt(2), x: CRootOf(x**10 - 2*x**6 - 2*x**5 + x**2 + 2*x - 1, 0)})
        """
        if self.dimension:
            raise ValueError("the chain has the free variables %s: its solutions are not isolated"
                % self.free_variables)
        return exact_solutions(self.polys, self.main_variables, real,
            lambda v: self.saturated_ideal().univariate(v))

    # ------------------------------------------------------------------
    # operations

    def is_regular(self, f: Equation) -> bool:
        """Whether ``f`` is not a zero divisor modulo the saturated ideal:
        it vanishes identically on no irreducible component of the closure
        of the quasi-component."""
        return self._decomposer.is_regular(self._element(f), self._chain)

    def regularize(self, f: Equation) -> tuple[list[RegularChain], list[RegularChain]]:
        """Split the chain into chains modulo which ``f`` is zero and chains
        modulo which it is regular.

        Returns
        =======

        (zero, regular)
            The quasi-components of all the chains cover the one of this
            chain, and lie in its closure. Chains of smaller dimension may
            be among them: the points where a leading coefficient met by
            the computation vanishes.
        """
        zero: list[RegularChain] = []
        regular: list[RegularChain] = []
        for chain, vanishes in self._decomposer.regularize(self._element(f), self._chain):
            (zero if vanishes else regular).append(self._wrap(chain))
        return zero, regular

    def intersect(self, f: Equation) -> list[RegularChain]:
        """Chains whose quasi-components cover the zeros of ``f`` in the
        quasi-component of this chain, and lie in the zeros of ``f`` in its
        closure.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.regularchains import RegularChain
        >>> RegularChain([x**2 + y**2 - 1], x, y).intersect(x - y)
        [RegularChain([2*y**2 - 1, x - y], x, y)]
        """
        decomposer = self._decomposer
        chains = decomposer.irredundant(decomposer.intersect(self._element(f), self._chain))
        return [self._wrap(chain) for chain in chains]


def triangularize(equations: Iterable[Equation], *symbols: Symbol, inequations: Iterable[Equation] = (),
                  mode: str = 'lazard') -> list[RegularChain]:
    """A triangular decomposition of the zeros of polynomial equations.

    Parameters
    ==========

    equations : iterable of Expr or Eq
        Polynomials with rational coefficients (which vanish), or equalities.
    symbols : Symbols
        The variables, the greatest first: the last ones are solved first,
        as with a lexicographic Gröbner basis. The other symbols of the
        system are put after them as the smallest variables, which makes
        them parameters whose values are discussed. All the symbols in
        their sorted order if omitted.
    inequations : iterable of Expr
        Polynomials which must not vanish.
    mode : ``'lazard'`` or ``'kalkbrener'``
        With ``'lazard'`` the zeros of the system are the union of the
        quasi-components of the chains. With ``'kalkbrener'`` they are the
        union of their closures: the chains describe the generic points of
        every irreducible component, and the points where an initial
        vanishes are not told apart.

    Returns
    =======

    list of RegularChain
        Squarefree regular chains, by increasing height (decreasing
        dimension); the empty list if there is no solution. With
        inequations, the solutions are the points of the quasi-components
        where no inequation vanishes, and every inequation is regular
        modulo every chain (it vanishes on a subset of smaller dimension).

    Examples
    ========

    A plane and a line, which a lexicographic Gröbner basis does not
    separate (the system is its own basis), then three lines:

    >>> from sympy.abc import x, y, z
    >>> from sympy_extras.polys.regularchains import triangularize
    >>> triangularize([x*y, x*z], x, y, z)
    [RegularChain([x], x, y, z), RegularChain([z, y], x, y, z)]
    >>> triangularize([x*z - y, y*z - x], x, y, z)
    [RegularChain([y, x], x, y, z), RegularChain([z + 1, x + y], x, y, z), RegularChain([z - 1, x - y], x, y, z)]

    The double roots of a cubic, in the sense of Kalkbrener: the discriminant
    and the root as a rational function of the coefficients.

    >>> from sympy.abc import p, q
    >>> f = x**3 + p*x + q
    >>> triangularize([f, f.diff(x)], x, p, q, mode='kalkbrener')
    [RegularChain([4*p**3 + 27*q**2, 2*p*x + 3*q], x, p, q)]

    In the sense of Lazard the case where the initial `p` vanishes is there
    too:

    >>> triangularize([f, f.diff(x)], x, p, q)
    [RegularChain([4*p**3 + 27*q**2, 2*p*x + 3*q], x, p, q), RegularChain([q, p, x], x, p, q)]

    An inequation:

    >>> triangularize([x**2 - y**2, x*(y - 1)], x, y, inequations=[x])
    [RegularChain([y - 1, x + 1], x, y), RegularChain([y - 1, x - 1], x, y)]
    """
    given = list(equations)
    excluded = list(inequations)
    if mode not in ('lazard', 'kalkbrener'):
        raise ValueError("mode must be 'lazard' or 'kalkbrener'")
    variables = _variables(given + excluded, symbols)
    ring = PolyRing(variables, ZZ)
    polys = [_polynomial(ring, variables, p) for p in given]
    nonzero = [_polynomial(ring, variables, p) for p in excluded]
    if any(not h for h in nonzero):
        return []
    bound: Optional[int] = None
    if mode == 'kalkbrener':
        bound = sum(1 for p in polys if p)
    chains = Decomposer(ring, bound).triangularize(polys, nonzero)
    # the chains which are returned keep every chain of their own operations
    decomposer = Decomposer(ring)
    return [RegularChain._new(chain, variables, ring, decomposer) for chain in chains]


def regular_gcd(p: Equation, q: Equation, x: Symbol, chain: RegularChain) -> list[tuple[Expr, RegularChain]]:
    """Regular greatest common divisors of two polynomials modulo a chain.

    Parameters
    ==========

    p, q : Expr
        Polynomials of main variable ``x`` whose initials are regular
        modulo the chain.
    x : Symbol
        A variable greater than the ones of the polynomials of the chain.
    chain : RegularChain

    Returns
    =======

    list of (Expr, RegularChain)
        Chains whose quasi-components cover the one of the chain, each
        with a regular gcd `g` of ``p`` and ``q`` modulo its saturated
        ideal: the initial of `g` is regular, `g` is in the ideal of ``p``
        and ``q`` and it pseudo-divides both. On the chains of smaller
        dimension than the given one, where the initials of ``p`` and ``q``
        need not be regular, the gcd is given as ``0``: nothing is known
        there.

    Examples
    ========

    Over the two roots of `y^2 - 1` the polynomials have different gcds:

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.regularchains import RegularChain, regular_gcd
    >>> T = RegularChain([y**2 - 1], x, y)
    >>> regular_gcd(x**2 - y, x**2 - 3*x + 2*y, x, T)
    [(1, RegularChain([y + 1], x, y)), (x - 1, RegularChain([y - 1], x, y))]
    """
    decomposer = chain._decomposer
    if x not in chain.symbols:
        raise ValueError("%s is not a variable of the chain" % x)
    level = chain.symbols.index(as_symbol(x))
    first, second = chain._element(p), chain._element(q)
    if main_variable(first) != level or main_variable(second) != level:
        raise ValueError("the main variable of the polynomials must be %s" % x)
    if any(t is not None for t in chain._chain.polys[:level + 1]):
        raise ValueError("%s must be greater than the main variables of the chain" % x)
    for f in (first, second):
        if not decomposer.is_regular(initial(f), chain._chain):
            raise ValueError("the initial %s is not regular modulo the chain" % initial(f).as_expr())
    result: list[tuple[Expr, RegularChain]] = []
    for g, part, _ in decomposer.regular_gcd(first, second, level, chain._chain):
        result.append((Integer(0) if g is None else as_expr(g.as_expr()), chain._wrap(part)))
    return result

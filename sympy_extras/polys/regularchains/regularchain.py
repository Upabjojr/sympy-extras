"""Regular chains and triangular decompositions of polynomial systems.

A *triangular decomposition* writes the solutions of a system of polynomial
equations as a union of triangular systems, each of which is solved one
variable after the other, like a linear system in row echelon form:

>>> from sympy_extras.core import symbols
>>> from sympy_extras.polys.regularchains import triangularize
>>> x, y, z = symbols('x y z')
>>> for chain in triangularize([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1], x, y, z):
...     print(chain.polys)
[z**2 + 2*z - 1, y - z, x - z]
[z, y, x - 1]
[z, y - 1, x]
[z - 1, y, x]

The variables are ordered, the first one being the greatest (as for the
lexicographic Gröbner bases). A polynomial is seen as a polynomial in its
greatest variable, its *main variable*, whose leading coefficient is its
*initial*. A *regular chain* `T` is a set of polynomials with distinct
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

>>> a, b, c = symbols('a b c')
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

The module works on objects of the core and imports no SymPy
(:mod:`sympy_extras.polys.regularchains_sympy` is its face on SymPy
objects). One step goes through SymPy at run time: the exact solutions of
a chain without free variables (:meth:`RegularChain.solutions`), whose
coordinates are SymPy's root objects ``CRootOf``, which are not ported
(:func:`~sympy_extras.polys.regularchains_sympy.exact_solutions_of_core`).
The numerical solutions are those of mpmath's ``polyroots``, as SymPy's
``Poly.nroots`` computes them (ported).

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
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/polytools.py (Poly.nroots), sympy/core/evalf.py
# (chop_parts) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: the numerical roots of the triangular system computed on mpmath
# numbers directly; typed.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Final, Iterable, Optional, Sequence, Union

import mpmath
from mpmath.libmp.libhyper import NoConvergence

from sympy_extras.core.add import add
from sympy_extras.core.evalf import mpf_to_fraction
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Float, ImaginaryUnit, Integer, dps_to_prec, number_from_fraction
from sympy_extras.core.power import pow
from sympy_extras.core.relational import Equality
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.domains.integerring import ZZ
from sympy_extras.polys.domains.rationalfield import QQ
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.polyerrors import PolynomialError
from sympy_extras.polys.polytools import Poly
from sympy_extras.polys.rings import PolyRing

from .operations import Chain, Decomposer
from .recursive import ZPoly, degree, initial, main_variable, max_norm, normalize

__all__ = ['RegularChain', 'triangularize', 'regular_gcd', 'Equation', 'Point']

#: an equation: a polynomial expression which vanishes, or an equality
Equation = Union[Expr, Equality, int]
#: a point: values of the variables
Point = dict[Symbol, Expr]


def _as_expr(equation: Equation) -> Expr:
    if isinstance(equation, bool):
        raise TypeError("an equation is expected, got %s" % (equation,))
    if isinstance(equation, int):
        return Integer(equation)
    if isinstance(equation, Equality):
        return equation.lhs - equation.rhs
    if isinstance(equation, Expr):
        return equation
    raise TypeError("an equation is expected, got %s" % (equation,))


def _polynomial(ring: PolyRing[int], symbols: Sequence[Symbol], equation: Equation) -> ZPoly:
    """The polynomial over the integers of an equation (the numerator of a
    polynomial with rational coefficients)."""
    expr = _as_expr(equation)
    unknown = expr.free_symbols - set(symbols)
    if unknown:
        raise ValueError("symbols which are not variables: %s" % sorted(unknown, key=lambda s: s.name))
    try:
        poly = Poly(expr, *symbols)
    except PolynomialError:
        raise ValueError("not a polynomial in the variables: %s" % expr)
    except Exception:
        raise ValueError("not a polynomial with rational coefficients: %s" % expr)
    n = len(symbols)
    terms = {m[:n]: c for m, c in poly.rep.terms.items()}
    if any(any(m[n:]) for m in poly.rep.terms):
        raise ValueError("not a polynomial with rational coefficients: %s" % expr)
    denominator = 1
    for c in terms.values():
        denominator = lcm(denominator, c.denominator)
    return ring.from_dict({m: int(c*denominator) for m, c in terms.items()})


def _expr(p: ZPoly, symbols: Sequence[Symbol]) -> Expr:
    """The expression of a polynomial in the actual symbols."""
    parts: list[Expr] = []
    for m, c in p.terms.items():
        factors: list[Expr] = [Integer(c)]
        for s, e in zip(symbols, m):
            if e:
                factors.append(pow(s, Integer(e)) if e > 1 else s)
        parts.append(mul(*factors))
    return add(*parts) if parts else Integer(0)


def _variables(equations: Sequence[Equation], symbols: Sequence[Symbol]) -> tuple[Symbol, ...]:
    """The given variables followed by the other symbols, as parameters."""
    variables = [_as_symbol(s) for s in symbols]
    if len(set(variables)) != len(variables):
        raise ValueError("repeated variables")
    found: set[Symbol] = set()
    for equation in equations:
        found |= _as_expr(equation).free_symbols
    variables.extend(sorted(found - set(variables), key=lambda s: s.name))
    if not variables:
        raise ValueError("no variables")
    return tuple(variables)


def _as_symbol(s: object) -> Symbol:
    if not isinstance(s, Symbol):
        raise TypeError("a symbol is expected, got %s" % (s,))
    return s


def _ring(symbols: Sequence[Symbol]) -> PolyRing[int]:
    return PolyRing(ZZ, [s.name for s in symbols])


class _Parts:
    """What a chain is made of, for the chains built by the operations."""

    def __init__(self, chain: Chain, ring: PolyRing[int], decomposer: Decomposer) -> None:
        self.chain: Final[Chain] = chain
        self.ring: Final[PolyRing[int]] = ring
        self.decomposer: Final[Decomposer] = decomposer


def _validated(equations: list[Equation], symbols: Sequence[Symbol]) -> tuple[_Parts, tuple[Symbol, ...]]:
    """The chain of the polynomials, checked to be a squarefree regular chain."""
    variables = _variables(equations, symbols)
    ring = _ring(variables)
    decomposer = Decomposer(ring)
    elements = [normalize(_polynomial(ring, variables, p)) for p in equations]
    levels = [main_variable(p) for p in elements]
    if any(level == ring.ngens for level in levels):
        raise ValueError("a constant polynomial")
    if len(set(levels)) != len(levels):
        raise ValueError("two polynomials with the same main variable")
    chain = decomposer.empty
    for level, p in sorted(zip(levels, elements), key=lambda pair: -pair[0]):
        if not decomposer.is_regular(initial(p), chain):
            raise ValueError("the initial %s is a zero divisor modulo the polynomials below"
                % _expr(initial(p), variables))
        added = [t for t in (T.get(level) for T in decomposer.add(chain, p)) if t is not None]
        if len(added) != 1 or degree(added[0], level) != degree(p, level):
            raise ValueError("%s is not squarefree modulo the polynomials below" % _expr(p, variables))
        chain = chain.with_poly(level, p)
    return _Parts(chain, ring, decomposer), variables


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

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.regularchains import RegularChain
    >>> x, y, z = symbols('x y z')
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

    def __init__(self, polys: Iterable[Equation], *symbols: Symbol, parts: Optional[_Parts] = None) -> None:
        if parts is None:
            parts, symbols = _validated(list(polys), symbols)
        self.symbols: Final[tuple[Symbol, ...]] = tuple(symbols)
        self._ring: Final[PolyRing[int]] = parts.ring
        self._decomposer: Final[Decomposer] = parts.decomposer
        self._chain: Final[Chain] = parts.chain

    def with_symbols(self, *symbols: Symbol) -> RegularChain:
        """The same chain in a ring with more variables, which are free: the
        variables of the chain must come in the same order among the new ones.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.regularchains import RegularChain
        >>> t, x, y = symbols('t x y')
        >>> T = RegularChain([y**2 - 2, x - y], x, y).with_symbols(x, t, y)
        >>> T.free_variables, T.contains(t*x**2 - 2*t)
        ([t], True)
        """
        variables = tuple(_as_symbol(s) for s in symbols)
        if len(set(variables)) != len(variables):
            raise ValueError("repeated variables")
        if [s for s in variables if s in self.symbols] != list(self.symbols):
            raise ValueError("the variables %s must be kept, in this order" % (self.symbols,))
        ring = _ring(variables)
        decomposer = Decomposer(ring)
        polys = [_polynomial(ring, variables, p) for p in self.polys]
        return RegularChain([], *variables, parts=_Parts(decomposer.empty.union(polys), ring, decomposer))

    def _wrap(self, chain: Chain) -> RegularChain:
        return RegularChain([], *self.symbols, parts=_Parts(chain, self._ring, self._decomposer))

    def _element(self, f: Equation) -> ZPoly:
        return _polynomial(self._ring, self.symbols, f)

    def _expr(self, p: ZPoly) -> Expr:
        return _expr(p, self.symbols)

    # ------------------------------------------------------------------
    # description

    @property
    def chain(self) -> Chain:
        """The chain of polynomials over the integers (of :mod:`.operations`)."""
        return self._chain

    @property
    def polys(self) -> list[Expr]:
        """The polynomials, by increasing main variable: the order in which
        the triangular system is solved."""
        return [self._expr(p) for p in self._chain.elements()]

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
        return [self._expr(initial(p)) for p in self._chain.elements()]

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
        return self._expr(self._decomposer.reduce(self._element(f), self._chain))

    def contains(self, f: Equation) -> bool:
        """Whether ``f`` is in the saturated ideal, that is vanishes on the
        quasi-component."""
        return not self._decomposer.reduce(self._element(f), self._chain)

    def __contains__(self, f: Equation) -> bool:
        return self.contains(f)

    def saturated_ideal(self) -> Ideal[Fraction]:
        """The saturated ideal, by a Gröbner basis computation.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.regularchains import RegularChain
        >>> x, y, z = symbols('x y z')
        >>> RegularChain([z*y - 1, y*x - z], x, y, z).saturated_ideal().reduced().exprs
        [x*y - z, y*z - 1, -x + z**2]
        """
        polys = self.polys
        ideal = Ideal(polys if polys else [Integer(0)], *self.symbols, domain=QQ)
        if not polys:
            return ideal
        return ideal.saturate(mul(*self.initials))

    def _univariate(self, v: Symbol) -> Expr:
        """The generator of the saturated ideal in ``v`` alone."""
        p = self.saturated_ideal().univariate(v)
        d = len(p.rep) - 1
        return add(*[mul(number_from_fraction(c), pow(v, Integer(d - k))) for k, c in enumerate(p.rep)])

    def numerical_solutions(self, n: int = 15) -> list[Point]:
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

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.regularchains import triangularize
        >>> x, y = symbols('x y')
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
        elements = self._chain.elements()
        # the coefficients of a chain are large, and as many digits cancel
        digits = 2*n + 10 + sum(len(str(max_norm(p))) for p in elements)
        solutions: list[dict[int, mpmath.mpc]] = [{}]
        for p in elements:
            i = main_variable(p)
            following: list[dict[int, mpmath.mpc]] = []
            for solution in solutions:
                for root in _nroots(_coefficients(p, i, solution, digits), digits):
                    extended = dict(solution)
                    extended[i] = root
                    following.append(extended)
            solutions = following
        levels = [main_variable(p) for p in elements]
        return [{self.symbols[i]: _number(solution[i], n) for i in levels} for solution in solutions]

    def solutions(self, real: bool = False) -> list[Point]:
        """The points of a chain without free variables, exactly.

        The coordinates are rational numbers, root objects (SymPy's
        ``CRootOf``, in radicals where the package writes them so) and
        rational functions of them, see
        :func:`~sympy_extras.polys.regularchains_sympy.solutions.exact_solutions`
        (on SymPy, called through the face at run time): the chain is
        solved from the bottom; a polynomial of degree one in its main
        variable gives its coordinate by a division, and the roots of
        another one, whose coefficients are algebraic numbers, are root
        objects of a polynomial with rational coefficients (an iterated
        resultant) which are told from its other roots numerically.

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

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.regularchains import triangularize
        >>> x, y = symbols('x y')
        >>> [chain] = triangularize([y**2 - 2, x*y - 1], x, y)
        >>> chain.solutions()
        [{y: -sqrt(2), x: -sqrt(2)/2}, {y: sqrt(2), x: sqrt(2)/2}]
        """
        if self.dimension:
            raise ValueError("the chain has the free variables %s: its solutions are not isolated"
                % self.free_variables)
        from sympy_extras.polys.regularchains_sympy import exact_solutions_of_core
        return exact_solutions_of_core(self.polys, self.main_variables, real, self._univariate)

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

        >>> from sympy_extras.core import symbols
        >>> from sympy_extras.polys.regularchains import RegularChain
        >>> x, y = symbols('x y')
        >>> RegularChain([x**2 + y**2 - 1], x, y).intersect(x - y)
        [RegularChain([2*y**2 - 1, x - y], x, y)]
        """
        decomposer = self._decomposer
        chains = decomposer.irredundant(decomposer.intersect(self._element(f), self._chain))
        return [self._wrap(chain) for chain in chains]


def _coefficients(p: ZPoly, i: int, values: dict[int, mpmath.mpc], digits: int) -> list[mpmath.mpc]:
    """The coefficients of ``p`` in the variable ``i`` (highest degree
    first) at the values of the smaller variables, at ``digits`` digits:
    integers for the first polynomial, complex numbers otherwise."""
    d = degree(p, i)
    with mpmath.workdps(digits):
        if not values:
            integers = [0]*(d + 1)
            for m, c in p.terms.items():
                integers[d - m[i]] += c
            return [mpmath.mpf(c) for c in integers]
        result = [mpmath.mpc(0)]*(d + 1)
        for m, c in p.terms.items():
            term = mpmath.mpc(c)
            for j, e in enumerate(m):
                if e and j != i:
                    term *= values[j]**e
            result[d - m[i]] += term
        return result


def _root_key(r: mpmath.mpc) -> tuple[int, mpmath.mpf, mpmath.mpf, int]:
    imag = mpmath.im(r)
    return (1 if imag else 0, mpmath.re(r), abs(imag), (imag > 0) - (imag < 0))


# PORTED FROM SYMPY: sympy/polys/polytools.py:Poly.nroots
def _nroots(coefficients: list[mpmath.mpc], digits: int) -> list[mpmath.mpc]:
    """The numerical roots of a polynomial, as SymPy's ``Poly.nroots(n=digits,
    maxsteps=200)``: real ones first."""
    degree_ = len(coefficients) - 1
    if degree_ <= 0:
        return []
    with mpmath.workdps(digits):
        try:
            roots = mpmath.polyroots(coefficients, maxsteps=200, cleanup=True, error=False,
                                     extraprec=degree_*10)
        except NoConvergence:
            roots = mpmath.polyroots(coefficients, maxsteps=200, cleanup=True, error=False,
                                     extraprec=degree_*15)
        found: list[mpmath.mpc] = [mpmath.mpc(r) for r in roots]
        return sorted(found, key=_root_key)


def _float(x: mpmath.mpf, n: int) -> Float:
    return Float(number_from_fraction(mpf_to_fraction(x)), dps=n)


# PORTED FROM SYMPY: sympy/core/evalf.py:chop_parts (for numbers of full accuracy)
def _number(z: mpmath.mpc, n: int) -> Expr:
    """``z`` to ``n`` digits, its real or imaginary part dropped when it is
    below the precision, as SymPy's ``evalf(n, chop=True)`` does."""
    prec = dps_to_prec(n)
    re, im = mpmath.re(z), mpmath.im(z)
    if re and abs(re) < mpmath.ldexp(1, -prec):
        re = mpmath.mpf(0)
    if im and abs(im) < mpmath.ldexp(1, -prec):
        im = mpmath.mpf(0)
    if not im:
        return _float(re, n)
    imaginary = mul(_float(im, n), ImaginaryUnit())
    if not re:
        return imaginary
    return add(_float(re, n), imaginary)


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

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.regularchains import triangularize
    >>> x, y, z = symbols('x y z')
    >>> triangularize([x*y, x*z], x, y, z)
    [RegularChain([x], x, y, z), RegularChain([z, y], x, y, z)]
    >>> triangularize([x*z - y, y*z - x], x, y, z)
    [RegularChain([y, x], x, y, z), RegularChain([z + 1, x + y], x, y, z), RegularChain([z - 1, x - y], x, y, z)]

    The double roots of a cubic, in the sense of Kalkbrener: the discriminant
    and the root as a rational function of the coefficients.

    >>> p, q = symbols('p q')
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
    ring = _ring(variables)
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
    return [RegularChain([], *variables, parts=_Parts(chain, ring, decomposer)) for chain in chains]


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

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.regularchains import RegularChain, regular_gcd
    >>> x, y = symbols('x y')
    >>> T = RegularChain([y**2 - 1], x, y)
    >>> regular_gcd(x**2 - y, x**2 - 3*x + 2*y, x, T)
    [(1, RegularChain([y + 1], x, y)), (x - 1, RegularChain([y - 1], x, y))]
    """
    decomposer = chain._decomposer
    if x not in chain.symbols:
        raise ValueError("%s is not a variable of the chain" % x)
    level = chain.symbols.index(x)
    first, second = chain._element(p), chain._element(q)
    if main_variable(first) != level or main_variable(second) != level:
        raise ValueError("the main variable of the polynomials must be %s" % x)
    if any(t is not None for t in chain.chain.polys[:level + 1]):
        raise ValueError("%s must be greater than the main variables of the chain" % x)
    for f in (first, second):
        if not decomposer.is_regular(initial(f), chain.chain):
            raise ValueError("the initial %s is not regular modulo the chain" % chain._expr(initial(f)))
    result: list[tuple[Expr, RegularChain]] = []
    for g, part, _ in decomposer.regular_gcd(first, second, level, chain.chain):
        result.append((Integer(0) if g is None else chain._expr(g), chain._wrap(part)))
    return result


"""The SymPy face of the regular chains (:mod:`sympy_extras.polys.regularchains`),
and the step which stays on SymPy.

:class:`RegularChain`, :func:`triangularize` and :func:`regular_gcd`
take and return SymPy objects, as before the native port: the equations
are SymPy expressions or equalities, the symbols SymPy symbols, and the
polynomials, initials, reduced forms and numerical solutions SymPy
expressions. The work is done by the native module, the objects being
converted by :mod:`sympy_extras.interop`; the saturated ideal is an ideal
of :mod:`sympy_extras.polys.ideals_sympy`.

What stays here, on SymPy: the exact solutions of a chain without free
variables (:mod:`.solutions`), whose coordinates are SymPy's root objects
``CRootOf`` (not ported), told apart by SymPy's numerical roots and
isolating intervals. The native :meth:`RegularChain.solutions
<sympy_extras.polys.regularchains.RegularChain.solutions>` calls
:func:`exact_solutions_of_core` at run time.

Examples
========

>>> from sympy.abc import x, y, z
>>> from sympy_extras.polys.regularchains_sympy import triangularize
>>> for chain in triangularize([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1], x, y, z):
...     print(chain.polys)
[z**2 + 2*z - 1, y - z, x - z]
[z, y, x - 1]
[z, y - 1, x]
[z - 1, y, x]
>>> from sympy.abc import a, b, c
>>> for chain in triangularize([a*x**2 + b*x + c], x, a, b, c):
...     print(chain.polys, chain.initials)
[a*x**2 + b*x + c] [a]
[a, b*x + c] [1, b]
[c, b, a] [1, 1, 1]
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.polys.polytools import Poly

from sympy_extras._typing_sympy import as_expr, as_symbol
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.relational import Equality as CoreEquality
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.polys.ideals_sympy import Ideal
from sympy_extras.polys.regularchains import regularchain as native

from .solutions import exact_solutions

__all__ = ['RegularChain', 'triangularize', 'regular_gcd', 'exact_solutions', 'exact_solutions_of_core',
    'core_chain', 'sympy_chain', 'Equation']

#: an equation: a polynomial expression which vanishes, or an equality
Equation = Union[Expr, Equality, int]


def _core_expr(e: Basic) -> CoreExpr:
    converted = from_sympy(e)
    if not isinstance(converted, CoreExpr):
        raise TypeError("expected an expression, got %s" % (e,))
    return converted


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(as_symbol(s))
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy_expr(e: CoreBasic) -> Expr:
    return as_expr(to_sympy(e))


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    return as_symbol(to_sympy(s))


def _equation(equation: Equation) -> native.Equation:
    """An equation as one of the core (an expression or an equality)."""
    given = sympify(equation)
    if isinstance(given, Equality):
        converted = from_sympy(given)
        if not isinstance(converted, CoreEquality):
            raise TypeError("expected an equality, got %s" % (given,))
        return converted
    return _core_expr(as_expr(given))


def _symbols(symbols: Iterable[Basic]) -> list[CoreSymbol]:
    return [_core_symbol(s) for s in symbols]


def _point(point: native.Point) -> dict[Symbol, Expr]:
    return {_sympy_symbol(v): _sympy_expr(value) for v, value in point.items()}


class RegularChain:
    """A squarefree regular chain of polynomials with rational coefficients
    (:class:`sympy_extras.polys.regularchains.RegularChain` on SymPy
    objects).

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
    >>> from sympy_extras.polys.regularchains_sympy import RegularChain
    >>> T = RegularChain([z**2 - 2, y**2 - z, (y + 1)*x - 1], x, y, z)
    >>> T.main_variables, T.dimension, T.degree
    ([z, y, x], 0, 4)
    >>> T.initials
    [1, 1, y + 1]
    >>> T.contains(y**4 - 2), T.contains(x*z - x - y + 1)
    (True, True)
    >>> T.reduce(y**3 + x)
    y*z + 3
    >>> S = RegularChain([y**2 - 1, x**2 - y], x, y)
    >>> S.regularize(y - 1)
    ([RegularChain([y - 1, x**2 - y], x, y)], [RegularChain([y + 1, x**2 - y], x, y)])
    >>> RegularChain([y**2 - z*y, (y - z)*x - 1], x, y, z)
    Traceback (most recent call last):
    ...
    ValueError: the initial y - z is a zero divisor modulo the polynomials below
    """

    def __init__(self, polys: Iterable[Equation], *symbols: Symbol) -> None:
        self._native: native.RegularChain = native.RegularChain([_equation(p) for p in polys],
                                                                 *_symbols(symbols))
        self.symbols: tuple[Symbol, ...] = tuple(_sympy_symbol(s) for s in self._native.symbols)

    @classmethod
    def _new(cls, chain: native.RegularChain) -> RegularChain:
        self = object.__new__(cls)
        self._native = chain
        self.symbols = tuple(_sympy_symbol(s) for s in chain.symbols)
        return self

    @property
    def native(self) -> native.RegularChain:
        """The native chain."""
        return self._native

    def with_symbols(self, *symbols: Symbol) -> RegularChain:
        """The same chain in a ring with more variables, which are free: the
        variables of the chain must come in the same order among the new ones.

        >>> from sympy.abc import t, x, y
        >>> from sympy_extras.polys.regularchains_sympy import RegularChain
        >>> T = RegularChain([y**2 - 2, x - y], x, y).with_symbols(x, t, y)
        >>> T.free_variables, T.contains(t*x**2 - 2*t)
        ([t], True)
        """
        return RegularChain._new(self._native.with_symbols(*_symbols(symbols)))

    # ------------------------------------------------------------------
    # description

    @property
    def polys(self) -> list[Expr]:
        """The polynomials, by increasing main variable."""
        return [_sympy_expr(p) for p in self._native.polys]

    @property
    def main_variables(self) -> list[Symbol]:
        """The main variables of the polynomials, in the same order."""
        return [_sympy_symbol(s) for s in self._native.main_variables]

    @property
    def free_variables(self) -> list[Symbol]:
        """The variables which are the main variable of no polynomial."""
        return [_sympy_symbol(s) for s in self._native.free_variables]

    @property
    def initials(self) -> list[Expr]:
        """The leading coefficients in the main variables."""
        return [_sympy_expr(p) for p in self._native.initials]

    @property
    def height(self) -> int:
        """The number of polynomials."""
        return self._native.height

    @property
    def dimension(self) -> int:
        """The dimension of the quasi-component: the number of free variables."""
        return self._native.dimension

    @property
    def degree(self) -> int:
        """The product of the degrees in the main variables."""
        return self._native.degree

    def __repr__(self) -> str:
        return 'RegularChain(%s, %s)' % (self.polys, ', '.join(str(s) for s in self.symbols))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RegularChain) and self._native == other._native

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self._native)

    # ------------------------------------------------------------------
    # the saturated ideal

    def reduce(self, f: Equation) -> Expr:
        """The pseudo-remainder of ``f`` by the chain, primitive over the
        integers."""
        return _sympy_expr(self._native.reduce(_equation(f)))

    def contains(self, f: Equation) -> bool:
        """Whether ``f`` is in the saturated ideal."""
        return self._native.contains(_equation(f))

    def __contains__(self, f: Equation) -> bool:
        return self.contains(f)

    def saturated_ideal(self) -> Ideal:
        """The saturated ideal, by a Gröbner basis computation.

        >>> from sympy.abc import x, y, z
        >>> from sympy_extras.polys.regularchains_sympy import RegularChain
        >>> RegularChain([z*y - 1, y*x - z], x, y, z).saturated_ideal().reduced().exprs
        [x*y - z, y*z - 1, -x + z**2]
        """
        polys = self.polys
        ideal = Ideal(polys if polys else [Integer(0)], *self.symbols)
        if not polys:
            return ideal
        return ideal.saturate(as_expr(Mul(*self.initials)))

    def numerical_solutions(self, n: int = 15) -> list[dict[Symbol, Expr]]:
        """The points of a chain without free variables, numerically.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.regularchains_sympy import triangularize
        >>> [chain] = triangularize([x**2 + y**2 - 5, x*y - 1], x, y)
        >>> for solution in chain.numerical_solutions(5):
        ...     print(solution)
        {y: -2.1889, x: -0.45685}
        {y: -0.45685, x: -2.1889}
        {y: 0.45685, x: 2.1889}
        {y: 2.1889, x: 0.45685}
        """
        return [_point(point) for point in self._native.numerical_solutions(n)]

    def solutions(self, real: bool = False) -> list[dict[Symbol, Expr]]:
        """The points of a chain without free variables, exactly
        (:func:`.solutions.exact_solutions`).

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.regularchains_sympy import triangularize
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
        """Whether ``f`` is not a zero divisor modulo the saturated ideal."""
        return self._native.is_regular(_equation(f))

    def regularize(self, f: Equation) -> tuple[list[RegularChain], list[RegularChain]]:
        """Split the chain into chains modulo which ``f`` is zero and chains
        modulo which it is regular."""
        zero, regular = self._native.regularize(_equation(f))
        return [RegularChain._new(T) for T in zero], [RegularChain._new(T) for T in regular]

    def intersect(self, f: Equation) -> list[RegularChain]:
        """Chains whose quasi-components cover the zeros of ``f`` in the
        quasi-component of this chain.

        >>> from sympy.abc import x, y
        >>> from sympy_extras.polys.regularchains_sympy import RegularChain
        >>> RegularChain([x**2 + y**2 - 1], x, y).intersect(x - y)
        [RegularChain([2*y**2 - 1, x - y], x, y)]
        """
        return [RegularChain._new(T) for T in self._native.intersect(_equation(f))]


def core_chain(chain: RegularChain) -> native.RegularChain:
    """The native chain of a chain of the face."""
    return chain.native


def sympy_chain(chain: native.RegularChain) -> RegularChain:
    """The chain of the face of a native chain."""
    return RegularChain._new(chain)


def triangularize(equations: Iterable[Equation], *symbols: Symbol, inequations: Iterable[Equation] = (),
                  mode: str = 'lazard') -> list[RegularChain]:
    """A triangular decomposition of the zeros of polynomial equations
    (:func:`sympy_extras.polys.regularchains.triangularize` on SymPy
    objects).

    >>> from sympy.abc import x, y, z, p, q
    >>> from sympy_extras.polys.regularchains_sympy import triangularize
    >>> triangularize([x*y, x*z], x, y, z)
    [RegularChain([x], x, y, z), RegularChain([z, y], x, y, z)]
    >>> f = x**3 + p*x + q
    >>> triangularize([f, f.diff(x)], x, p, q, mode='kalkbrener')
    [RegularChain([4*p**3 + 27*q**2, 2*p*x + 3*q], x, p, q)]
    >>> triangularize([x**2 - y**2, x*(y - 1)], x, y, inequations=[x])
    [RegularChain([y - 1, x + 1], x, y), RegularChain([y - 1, x - 1], x, y)]
    """
    given = [_equation(e) for e in equations]
    excluded = [_equation(e) for e in inequations]
    chains = native.triangularize(given, *_symbols(symbols), inequations=excluded, mode=mode)
    return [RegularChain._new(chain) for chain in chains]


def regular_gcd(p: Equation, q: Equation, x: Symbol, chain: RegularChain) -> list[tuple[Expr, RegularChain]]:
    """Regular greatest common divisors of two polynomials modulo a chain
    (:func:`sympy_extras.polys.regularchains.regular_gcd` on SymPy objects).

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.regularchains_sympy import RegularChain, regular_gcd
    >>> T = RegularChain([y**2 - 1], x, y)
    >>> regular_gcd(x**2 - y, x**2 - 3*x + 2*y, x, T)
    [(1, RegularChain([y + 1], x, y)), (x - 1, RegularChain([y - 1], x, y))]
    """
    found = native.regular_gcd(_equation(p), _equation(q), _core_symbol(x), chain.native)
    return [(_sympy_expr(g), RegularChain._new(part)) for g, part in found]


def exact_solutions_of_core(polys: Sequence[CoreExpr], variables: Sequence[CoreSymbol], real: bool,
                            fallback: Callable[[CoreSymbol], CoreExpr]) -> list[native.Point]:
    """:func:`.solutions.exact_solutions` of a native chain: its polynomials
    by increasing main variable, the main variables, the real points only
    with ``real``, and ``fallback(v)``, the polynomial in ``v`` alone of the
    saturated ideal."""
    symbols = {_sympy_symbol(v): v for v in variables}

    def sympy_fallback(v: Symbol) -> Poly:
        poly: Poly = Poly(_sympy_expr(fallback(symbols[v])), v, domain='QQ')
        return poly

    found = exact_solutions([_sympy_expr(p) for p in polys], list(symbols), real, sympy_fallback)
    return [{symbols[v]: _core_expr(value) for v, value in point.items()} for point in found]

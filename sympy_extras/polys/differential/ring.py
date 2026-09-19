"""Differential polynomial rings: derivatives, rankings and the conversion
of SymPy expressions.

A *differential polynomial ring* `K\\{u_1, \\ldots, u_m\\}` with the
commuting derivations `\\partial_1, \\ldots, \\partial_n` is the polynomial
ring over `K` in the infinitely many *derivatives*
`\\partial^J u_a = \\partial_1^{j_1} \\cdots \\partial_n^{j_n} u_a`. Here
`K = \\mathbb{Q}(p_1, \\ldots)(x_1, \\ldots, x_n)` is the field of the
rational functions of the independent variables (and of constant
parameters), the derivations are `\\partial / \\partial x_i`, and a
derivative is stored as a pair ``(a, J)`` of the index of the function and
the multi-index of the orders (a :data:`Jet`).

A *ranking* is a total order of the derivatives with

.. math:: v < \\partial_i v, \\qquad v < w \\implies \\partial_i v < \\partial_i w,

and it is a well-order [Kolchin]_. The rankings of this module are given by
blocks of functions, as in the systems which implement differential
elimination [Boulier]_: every derivative of a function of an earlier
block is higher than every derivative of a function of a later block (an
*elimination ranking*, which eliminates the functions of the earlier
blocks), and inside a block the ranking is *orderly*: the derivatives are
compared by their total order, then by the function (the earlier one is
higher), then lexicographically by the orders in `x_1, x_2, \\ldots` (the
first variable weighs most). One block with all the functions is the
default; it is an orderly ranking.

The *leader* of a differential polynomial is its highest derivative.

References
==========

.. [Kolchin] E. R. Kolchin, Differential Algebra and Algebraic Groups,
   Academic Press 1973, chapter I.
.. [Boulier] F. Boulier, D. Lazard, F. Ollivier, M. Petitot, Computing
   representations for radicals of finitely generated differential ideals,
   Appl. Algebra Engrg. Comm. Comput. 20 (2009), 73-121.
"""
from __future__ import annotations

from math import gcd
from typing import Optional, Sequence, Union

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative
from sympy.core.numbers import Integer
from sympy.core.relational import Equality
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.domains import QQ, ZZ
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError, PolynomialError
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import fraction
from sympy.polys.rationaltools import together

from sympy_extras._typing import DomainElement, Monomial, as_expr, free_symbols

__all__ = ['DifferentialRing', 'Jet', 'JetMonomial', 'Terms', 'Equation', 'unit_jet']

#: a derivative: the index of the function and the orders in the variables
Jet = tuple[int, Monomial]
#: a product of powers of derivatives, sorted by derivative
JetMonomial = tuple[tuple[Jet, int], ...]
#: a differential polynomial as a map from monomials to coefficients
Terms = dict[JetMonomial, DomainElement]
#: an equation: an expression which is to vanish, or an equality
Equation = Union[Expr, Equality]


def unit_jet(n: int) -> Jet:
    """The pseudo derivative standing for the constant `1` in a linear
    form (the lowest of the ranking; its derivatives are zero)."""
    return (-1, (0,)*n)


class DifferentialRing:
    """A differential polynomial ring with a ranking.

    Parameters
    ==========

    functions : sequence of applied undefined functions
        The differential indeterminates, such as ``u(x, y)``. A function
        of some of the variables only is a function which is constant in
        the others (see :meth:`constancy`).
    variables : sequence of Symbol, optional
        The independent variables, in the order which the ranking uses;
        the arguments of the functions in the order of their appearance
        by default.
    ranking : sequence of sequences of functions, optional
        The blocks of the ranking, the highest first. One block with all
        the functions in the given order by default.
    parameters : sequence of Symbol
        The constants which the coefficients may contain.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.polys.differential import DifferentialRing
    >>> x, y = symbols('x y')
    >>> u, v = Function('u'), Function('v')
    >>> R = DifferentialRing([u(x, y), v(x, y)])
    >>> R.leader(u(x, y).diff(x) + v(x, y).diff(y, 2))
    Derivative(v(x, y), (y, 2))
    >>> R = DifferentialRing([u(x, y), v(x, y)], ranking=[[u(x, y)], [v(x, y)]])
    >>> R.leader(u(x, y).diff(x) + v(x, y).diff(y, 2))
    Derivative(u(x, y), x)
    """

    def __init__(self, functions: Sequence[AppliedUndef], variables: Optional[Sequence[Symbol]] = None,
                 ranking: Optional[Sequence[Sequence[AppliedUndef]]] = None,
                 parameters: Sequence[Symbol] = ()) -> None:
        if not functions:
            raise ValueError("at least one function is expected")
        found: list[Symbol] = []
        heads: set[str] = set()
        for f in functions:
            if not isinstance(f, AppliedUndef):
                raise TypeError("an applied undefined function such as u(x, y) is expected, got %s" % (f,))
            if f.func.__name__ in heads:
                raise ValueError("the function %s is given twice" % (f.func.__name__,))
            heads.add(f.func.__name__)
            for arg in f.args:
                if not isinstance(arg, Symbol):
                    raise ValueError("the arguments of %s must be symbols" % (f,))
                if arg not in found:
                    found.append(arg)
            if len(set(f.args)) != len(f.args):
                raise ValueError("the arguments of %s must be distinct" % (f,))
        if variables is None:
            chosen = list(found)
        else:
            chosen = list(variables)
            for s in chosen:
                if not isinstance(s, Symbol):
                    raise TypeError("the variables must be symbols, got %s" % (s,))
            missing = [s for s in found if s not in chosen]
            if missing:
                raise ValueError("%s is not among the variables" % (missing[0],))
        if len(set(chosen)) != len(chosen):
            raise ValueError("the variables must be distinct")
        self.functions: tuple[AppliedUndef, ...] = tuple(functions)
        self.variables: tuple[Symbol, ...] = tuple(chosen)
        self.parameters: tuple[Symbol, ...] = tuple(p for p in parameters if p not in chosen)
        self.domain: Domain = QQ.frac_field(*self.parameters, *self.variables)
        #: the polynomials with integer coefficients in the parameters and
        #: the variables: the coefficients of the differential polynomials
        self.integers: Domain = ZZ.poly_ring(*self.parameters, *self.variables)
        self._rationals: Domain = QQ.poly_ring(*self.parameters, *self.variables)
        self.blocks: tuple[tuple[int, ...], ...] = self._blocks(ranking)
        self._level: dict[int, tuple[int, int]] = {}
        for b, block in enumerate(self.blocks):
            for position, a in enumerate(block):
                self._level[a] = (len(self.blocks) - b, len(block) - position)
        self._keys: dict[Jet, tuple[int, ...]] = {}
        self._generators: list[DomainElement] = [
            self.domain.from_sympy(s) for s in self.variables]
        self._integer_generators: list[DomainElement] = [
            self.integers.from_sympy(s) for s in self.variables]

    def _blocks(self, ranking: Optional[Sequence[Sequence[AppliedUndef]]]) -> tuple[tuple[int, ...], ...]:
        if ranking is None:
            return (tuple(range(len(self.functions))),)
        blocks: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for block in ranking:
            indices: list[int] = []
            for f in block:
                if f not in self.functions:
                    raise ValueError("%s of the ranking is not a function of the ring" % (f,))
                a = self.functions.index(f)
                if a in seen:
                    raise ValueError("%s occurs twice in the ranking" % (f,))
                seen.add(a)
                indices.append(a)
            if indices:
                blocks.append(tuple(indices))
        if len(seen) != len(self.functions):
            absent = [f for a, f in enumerate(self.functions) if a not in seen]
            raise ValueError("%s is not in the ranking" % (absent[0],))
        return tuple(blocks)

    def __repr__(self) -> str:
        return "DifferentialRing(%s, %s)" % (list(self.functions), list(self.variables))

    # ------------------------------------------------------------------
    # derivatives and the ranking

    @property
    def n(self) -> int:
        """The number of derivations."""
        return len(self.variables)

    @property
    def is_orderly(self) -> bool:
        """Whether a derivative of higher order is always higher."""
        return len(self.blocks) == 1

    def rank_key(self, jet: Jet) -> tuple[int, ...]:
        """The key which sorts the derivatives by the ranking."""
        key = self._keys.get(jet)
        if key is None:
            a, J = jet
            if a < 0:
                key = (-1, 0, 0) + J
            else:
                block, position = self._level[a]
                key = (block, sum(J), position) + J
            self._keys[jet] = key
        return key

    def highest(self, jets: Sequence[Jet]) -> Jet:
        """The highest of the derivatives."""
        return max(jets, key=self.rank_key)

    def differentiate(self, jet: Jet, i: int) -> Jet:
        """The derivative of a derivative with respect to the variable of
        index ``i``."""
        a, J = jet
        return (a, J[:i] + (J[i] + 1,) + J[i + 1:])

    def is_derivative_of(self, jet: Jet, other: Jet) -> bool:
        """Whether ``jet`` is a derivative of ``other`` (or equal to it)."""
        return jet[0] == other[0] and all(j >= k for j, k in zip(jet[1], other[1]))

    def constancy(self) -> list[Jet]:
        """The derivatives which vanish because a function does not depend
        on a variable: `\\partial u / \\partial y` for ``u(x)`` in a ring
        with the variables `x, y`."""
        result: list[Jet] = []
        for a, f in enumerate(self.functions):
            for i, s in enumerate(self.variables):
                if s not in f.args:
                    result.append((a, (0,)*i + (1,) + (0,)*(self.n - i - 1)))
        return result

    # ------------------------------------------------------------------
    # conversion

    def jet(self, expr: Basic) -> Optional[Jet]:
        """The derivative which an applied function or a ``Derivative`` of
        one stands for, ``None`` for another expression."""
        if isinstance(expr, AppliedUndef):
            if expr in self.functions:
                return (self.functions.index(expr), (0,)*self.n)
            return None
        if isinstance(expr, Derivative) and expr.expr in self.functions:
            orders = [0]*self.n
            for s, count in expr.variable_count:
                if s not in self.variables or not isinstance(count, Integer):
                    return None
                orders[self.variables.index(s)] += int(count)
            return (self.functions.index(expr.expr), tuple(orders))
        return None

    def to_expr(self, jet: Jet) -> Expr:
        """The SymPy expression of a derivative."""
        a, J = jet
        if a < 0:
            return S.One
        counts = [(s, j) for s, j in zip(self.variables, J) if j]
        if not counts:
            return self.functions[a]
        return Derivative(self.functions[a], *counts)

    def coefficient(self, expr: Expr) -> DomainElement:
        """An expression as an element of the field of coefficients."""
        try:
            return self.domain.from_sympy(expr)
        except (CoercionFailed, GeneratorsError, PolynomialError, ValueError, TypeError):
            raise ValueError("%s is not a rational function of the variables %s and the parameters %s"
                             % (expr, list(self.variables), list(self.parameters))) from None

    def coefficient_derivative(self, c: DomainElement, i: int) -> DomainElement:
        """The derivative of a coefficient with respect to the variable of
        index ``i``."""
        return c.diff(self._generators[i])

    def integer_derivative(self, c: DomainElement, i: int) -> DomainElement:
        """The derivative of an element of :attr:`integers`."""
        return c.diff(self._integer_generators[i])

    def integral(self, terms: Terms) -> Terms:
        """The terms multiplied by an element of the field which makes the
        coefficients elements of :attr:`integers`."""
        if not terms:
            return {}
        values = list(terms.values())
        multiple = values[0].denom
        for c in values[1:]:
            multiple = multiple.lcm(c.denom)
        factor = self.domain.convert(multiple, self._rationals)
        cleared: dict[JetMonomial, DomainElement] = {}
        denominator = 1
        for m, c in terms.items():
            scaled = c*factor
            q = scaled.numer.mul_ground(1/scaled.denom.coeffs()[0])
            cleared[m] = q
            for e in q.coeffs():
                denominator = denominator*int(e.denominator)//gcd(denominator, int(e.denominator))
        result: Terms = {}
        for m, q in cleared.items():
            result[m] = self.integers.convert(q.mul_ground(denominator), self._rationals)
        return result

    def terms(self, equation: Equation) -> Terms:
        """The monomials and coefficients of a differential polynomial
        given as an expression.

        Raises ``ValueError`` when the expression is not a polynomial in
        the derivatives of the functions with coefficients rational in the
        variables and the parameters.
        """
        expr = equation.lhs - equation.rhs if isinstance(equation, Equality) else as_expr(equation)
        expr = expr.doit()
        symbols: dict[Basic, Dummy] = {}
        jets: list[Jet] = []
        atoms = sorted(expr.atoms(Derivative) | expr.atoms(AppliedUndef), key=lambda e: e.sort_key())
        for atom in atoms:
            jet = self.jet(atom)
            if jet is None:
                if isinstance(atom, AppliedUndef) or any(f in self.functions for f in atom.atoms(AppliedUndef)):
                    if isinstance(atom, Derivative) and atom.expr in self.functions:
                        raise ValueError("%s is not a derivative in the variables of the ring" % (atom,))
                    if isinstance(atom, AppliedUndef):
                        raise ValueError("%s is not a function of the ring" % (atom,))
                raise ValueError("%s is not a derivative of a function of the ring" % (atom,))
            symbols[atom] = Dummy('jet%d' % len(jets))
            jets.append(jet)
        extra = free_symbols(expr) - set(self.variables) - set(self.parameters)
        if extra:
            raise ValueError("%s is neither a variable nor a parameter of the ring"
                             % (sorted(extra, key=lambda s: s.name)[0],))
        replaced = as_expr(expr.xreplace(symbols))
        numerator, denominator = fraction(together(replaced))
        gens = [symbols[atom] for atom in atoms]
        if any(denominator.has(g) for g in gens):
            raise ValueError("%s is not a polynomial in the derivatives" % (expr,))
        if not gens:
            c = self.coefficient(replaced)
            return {(): c} if c else {}
        try:
            poly = Poly(numerator, *gens, domain=self.domain)
        except (PolynomialError, CoercionFailed, GeneratorsError):
            raise ValueError("%s is not a polynomial in the derivatives with rational coefficients"
                             % (expr,)) from None
        scale = self.coefficient(denominator)
        result: Terms = {}
        for exponents, c in poly.as_dict(native=True).items():
            powers: dict[Jet, int] = {}
            for jet, e in zip(jets, exponents):
                if e:
                    powers[jet] = powers.get(jet, 0) + e
            monomial = tuple(sorted(powers.items()))
            value = result.get(monomial, self.domain.zero) + c/scale
            if value:
                result[monomial] = value
            else:
                result.pop(monomial, None)
        return result

    def from_terms(self, terms: Terms) -> Expr:
        """The expression of a differential polynomial."""
        parts: list[Expr] = []
        for monomial, c in terms.items():
            term = as_expr(self.domain.to_sympy(c))
            for jet, e in monomial:
                term = term*self.to_expr(jet)**e
            parts.append(term)
        return Add(*parts)

    def leader(self, equation: Equation) -> Expr:
        """The highest derivative of a differential polynomial.

        >>> from sympy import Function, symbols
        >>> from sympy_extras.polys.differential import DifferentialRing
        >>> x, y = symbols('x y')
        >>> u = Function('u')(x, y)
        >>> DifferentialRing([u]).leader(u.diff(x, y) + u.diff(y, 2)*u)
        Derivative(u(x, y), x, y)
        """
        jets = [jet for monomial in self.terms(equation) for jet, _ in monomial]
        if not jets:
            raise ValueError("the differential polynomial has no derivative")
        return self.to_expr(self.highest(jets))

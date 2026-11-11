"""The SymPy face of the differential algebra
(:mod:`sympy_extras.polys.differential`): differential rings, Janet bases
and the Rosenfeld-Groebner algorithm on SymPy objects.

:class:`DifferentialRing`, :class:`DifferentialPolynomial`,
:func:`janet_basis` (:class:`JanetBasis`) and :func:`rosenfeld_groebner`
(:class:`RadicalDifferentialIdeal`, :class:`RegularDifferentialSystem`)
take and return SymPy objects, as before the native port: the functions
are SymPy's applied undefined functions, the equations SymPy expressions
or equalities, the coefficients of :meth:`DifferentialRing.terms`
elements of SymPy's field ``QQ(p, x)`` (:attr:`DifferentialRing.domain`)
and those of a :class:`DifferentialPolynomial` elements of SymPy's ring
``ZZ[p, x]`` (:attr:`DifferentialRing.integers`); the regular chains of
the components are those of :mod:`sympy_extras.polys.regularchains_sympy`.
The work is done by the native modules, the objects being converted by
:mod:`sympy_extras.interop` and the coefficients term by term. Nothing of
the computation stays on SymPy.

Examples
========

>>> from sympy import Function, symbols
>>> from sympy_extras.polys.differential_sympy import janet_basis, rosenfeld_groebner
>>> x, y, z = symbols('x y z')
>>> u = Function('u')(x, y, z)
>>> J = janet_basis([u.diff(z, 2) + y*u.diff(x, 2), u.diff(y, 2)], [u])
>>> J.dimension, J.contains(u.diff(z, 4))
(12, True)
>>> p = Function('p')(x)
>>> I = rosenfeld_groebner([p.diff(x)**2 - 4*p], [p])
>>> [(c.equations, c.inequations) for c in I.components]
[([-4*p(x) + Derivative(p(x), x)**2], [Derivative(p(x), x)]), ([p(x)], [])]
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Equality
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.polys.domains import QQ, ZZ
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.polynomialring import PolynomialRing

from sympy_extras._typing import DomainElement, ExprLike, as_expr, as_symbol
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.core.relational import Equality as CoreEquality
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.polys.differential import janet as native_janet
from sympy_extras.polys.differential import polynomial as native_polynomial
from sympy_extras.polys.differential import ring as native_ring
from sympy_extras.polys.differential.janet import janet_multiplicative_variables
from sympy_extras.polys.differential.rosenfeld_groebner import (
    JetSymbols as NativeJetSymbols, RadicalDifferentialIdeal as NativeIdeal,
    RegularDifferentialSystem as NativeSystem, _characteristic_set as native_characteristic_set,
    _delta_polynomials as native_delta_polynomials, rosenfeld_groebner as native_rosenfeld_groebner)
from sympy_extras.polys.regularchains_sympy import RegularChain, sympy_chain

__all__ = ['DifferentialRing', 'DifferentialPolynomial', 'JanetBasis', 'janet_basis',
           'janet_multiplicative_variables', 'RadicalDifferentialIdeal', 'RegularDifferentialSystem',
           'rosenfeld_groebner', 'JetSymbols', 'Jet', 'JetMonomial', 'Terms', 'LinearForm', 'Equation', 'Rank']

Jet = native_ring.Jet
JetMonomial = native_ring.JetMonomial
Rank = native_polynomial.Rank
#: a differential polynomial with coefficients in a SymPy domain
Terms = dict[JetMonomial, DomainElement]
#: a linear form with coefficients in SymPy's ``ZZ[p, x]``
LinearForm = dict[Jet, DomainElement]
#: an equation: an expression which is to vanish, or an equality
Equation = Union[Expr, Equality]


# ----------------------------------------------------------------------
# conversions

def _core_expr(e: Basic) -> CoreExpr:
    converted = from_sympy(e)
    if not isinstance(converted, CoreExpr):
        raise TypeError("expected an expression, got %s" % (e,))
    return converted


def _core_basic(e: Basic) -> CoreBasic:
    return from_sympy(e)


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(as_symbol(s))
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _equation(equation: Equation) -> native_ring.Equation:
    """An equation as one of the core (an expression or an equality)."""
    given = sympify(equation)
    if isinstance(given, Equality):
        converted = from_sympy(given)
        if not isinstance(converted, CoreEquality):
            raise TypeError("expected an equality, got %s" % (given,))
        return converted
    return _core_expr(as_expr(given))


def _sympy_expr(e: CoreBasic) -> Expr:
    return as_expr(to_sympy(e))


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    return as_symbol(to_sympy(s))


def _sympy_function(f: CoreAppliedUndef) -> AppliedUndef:
    converted = to_sympy(f)
    if not isinstance(converted, AppliedUndef):
        raise TypeError("expected an applied function, got %s" % (converted,))
    return converted


def _rational(c: DomainElement) -> Fraction:
    return Fraction(int(QQ.numer(c)), int(QQ.denom(c)))


# ----------------------------------------------------------------------
# the ring

class DifferentialRing:
    """A differential polynomial ring with a ranking
    (:class:`sympy_extras.polys.differential.DifferentialRing` on SymPy
    objects).

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
    >>> from sympy_extras.polys.differential_sympy import DifferentialRing
    >>> x, y = symbols('x y')
    >>> u, v = Function('u'), Function('v')
    >>> R = DifferentialRing([u(x, y), v(x, y)])
    >>> R.leader(u(x, y).diff(x) + v(x, y).diff(y, 2))
    Derivative(v(x, y), (y, 2))
    >>> R = DifferentialRing([u(x, y), v(x, y)], ranking=[[u(x, y)], [v(x, y)]])
    >>> R.leader(u(x, y).diff(x) + v(x, y).diff(y, 2))
    Derivative(u(x, y), x)
    """

    def __init__(self, functions: Sequence[Basic], variables: Optional[Sequence[Basic]] = None,
                 ranking: Optional[Sequence[Sequence[Basic]]] = None, parameters: Sequence[Symbol] = ()) -> None:
        native = native_ring.DifferentialRing(
            [_core_basic(f) for f in functions], None if variables is None else [_core_basic(s) for s in variables],
            None if ranking is None else [[_core_basic(f) for f in block] for block in ranking],
            [_core_symbol(p) for p in parameters])
        self._init(native)

    def _init(self, native: native_ring.DifferentialRing) -> None:
        self.native: native_ring.DifferentialRing = native
        self.functions: tuple[AppliedUndef, ...] = tuple(_sympy_function(f) for f in native.functions)
        self.variables: tuple[Symbol, ...] = tuple(_sympy_symbol(s) for s in native.variables)
        self.parameters: tuple[Symbol, ...] = tuple(_sympy_symbol(s) for s in native.parameters)
        domain = QQ.frac_field(*self.parameters, *self.variables)
        integers = ZZ.poly_ring(*self.parameters, *self.variables)
        if not isinstance(domain, FractionField) or not isinstance(integers, PolynomialRing):
            raise TypeError("unexpected SymPy domains %s, %s" % (domain, integers))
        #: SymPy's field ``QQ(p, x)`` of the coefficients
        self.domain: FractionField = domain
        #: SymPy's polynomials with integer coefficients in the parameters
        #: and the variables: the coefficients of the differential polynomials
        self.integers: PolynomialRing = integers
        self.blocks: tuple[tuple[int, ...], ...] = native.blocks

    @classmethod
    def _new(cls, native: native_ring.DifferentialRing) -> DifferentialRing:
        self = object.__new__(cls)
        self._init(native)
        return self

    def __repr__(self) -> str:
        return "DifferentialRing(%s, %s)" % (list(self.functions), list(self.variables))

    # ------------------------------------------------------------------
    # coefficients

    def integer_in(self, c: DomainElement) -> native_ring.IntegerCoefficient:
        """An element of :attr:`integers` as a native one."""
        return self.native.integers.ring.from_dict({tuple(m): int(v) for m, v in c.items()})

    def integer_out(self, c: native_ring.IntegerCoefficient) -> DomainElement:
        """A native element of ``ZZ[p, x]`` as one of :attr:`integers`."""
        return self.integers.ring.from_dict({m: ZZ(v) for m, v in c.terms.items()})

    def field_in(self, c: DomainElement) -> native_ring.Coefficient:
        """An element of :attr:`domain` as a native one."""
        Q = self.native.domain.ring.ring
        numer = Q.from_dict({tuple(m): _rational(v) for m, v in c.numer.items()})
        denom = Q.from_dict({tuple(m): _rational(v) for m, v in c.denom.items()})
        return self.native.domain.new(numer, denom)

    def field_out(self, c: native_ring.Coefficient) -> DomainElement:
        """A native element of ``QQ(p, x)`` as one of :attr:`domain`."""
        R = self.domain.field.ring
        numer = R.from_dict({m: QQ(v.numerator, v.denominator) for m, v in c.numer.terms.items()})
        denom = R.from_dict({m: QQ(v.numerator, v.denominator) for m, v in c.denom.terms.items()})
        return self.domain.field.new(numer, denom)

    def terms_in(self, terms: Terms) -> native_ring.Terms:
        return {m: self.integer_in(c) for m, c in terms.items()}

    def terms_out(self, terms: native_ring.Terms) -> Terms:
        return {m: self.integer_out(c) for m, c in terms.items()}

    # ------------------------------------------------------------------
    # derivatives and the ranking

    @property
    def n(self) -> int:
        """The number of derivations."""
        return self.native.n

    @property
    def is_orderly(self) -> bool:
        """Whether a derivative of higher order is always higher."""
        return self.native.is_orderly

    def rank_key(self, jet: Jet) -> tuple[int, ...]:
        """The key which sorts the derivatives by the ranking."""
        return self.native.rank_key(jet)

    def highest(self, jets: Sequence[Jet]) -> Jet:
        """The highest of the derivatives."""
        return self.native.highest(jets)

    def differentiate(self, jet: Jet, i: int) -> Jet:
        """The derivative of a derivative with respect to the variable of
        index ``i``."""
        return self.native.differentiate(jet, i)

    def is_derivative_of(self, jet: Jet, other: Jet) -> bool:
        """Whether ``jet`` is a derivative of ``other`` (or equal to it)."""
        return self.native.is_derivative_of(jet, other)

    def constancy(self) -> list[Jet]:
        """The derivatives which vanish because a function does not depend
        on a variable."""
        return self.native.constancy()

    # ------------------------------------------------------------------
    # conversion

    def jet(self, expr: Basic) -> Optional[Jet]:
        """The derivative which an applied function or a ``Derivative`` of
        one stands for, ``None`` for another expression."""
        return self.native.jet(from_sympy(expr))

    def to_expr(self, jet: Jet) -> Expr:
        """The SymPy expression of a derivative."""
        return _sympy_expr(self.native.to_expr(jet))

    def coefficient(self, expr: Expr) -> DomainElement:
        """An expression as an element of the field of coefficients."""
        return self.field_out(self.native.coefficient(_core_expr(expr)))

    def coefficient_derivative(self, c: DomainElement, i: int) -> DomainElement:
        """The derivative of a coefficient with respect to the variable of
        index ``i``."""
        return self.field_out(self.native.coefficient_derivative(self.field_in(c), i))

    def integer_derivative(self, c: DomainElement, i: int) -> DomainElement:
        """The derivative of an element of :attr:`integers`."""
        return self.integer_out(self.native.integer_derivative(self.integer_in(c), i))

    def integral(self, terms: Terms) -> Terms:
        """The terms multiplied by an element of the field which makes the
        coefficients elements of :attr:`integers`."""
        return self.terms_out(self.native.integral({m: self.field_in(c) for m, c in terms.items()}))

    def terms(self, equation: Equation) -> Terms:
        """The monomials and coefficients (elements of :attr:`domain`) of a
        differential polynomial given as an expression.

        Raises ``ValueError`` when the expression is not a polynomial in
        the derivatives of the functions with coefficients rational in the
        variables and the parameters.
        """
        return {m: self.field_out(c) for m, c in self.native.terms(_equation(equation)).items()}

    def from_terms(self, terms: Terms) -> Expr:
        """The expression of a differential polynomial with coefficients in
        :attr:`domain`."""
        return _sympy_expr(self.native.from_terms({m: self.field_in(c) for m, c in terms.items()}))

    def leader(self, equation: Equation) -> Expr:
        """The highest derivative of a differential polynomial.

        >>> from sympy import Function, symbols
        >>> from sympy_extras.polys.differential_sympy import DifferentialRing
        >>> x, y = symbols('x y')
        >>> u = Function('u')(x, y)
        >>> DifferentialRing([u]).leader(u.diff(x, y) + u.diff(y, 2)*u)
        Derivative(u(x, y), x, y)
        """
        return _sympy_expr(self.native.leader(_equation(equation)))


# ----------------------------------------------------------------------
# differential polynomials

class DifferentialPolynomial:
    """A polynomial in the derivatives of the functions of a
    :class:`DifferentialRing`
    (:class:`sympy_extras.polys.differential.DifferentialPolynomial` on
    SymPy objects; the coefficients are elements of the ring's
    ``integers``).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.polys.differential_sympy import DifferentialRing, DifferentialPolynomial
    >>> x = symbols('x')
    >>> u = Function('u')(x)
    >>> R = DifferentialRing([u])
    >>> p = DifferentialPolynomial.from_expr(R, u.diff(x)**2 - 4*u)
    >>> p.leader_expr(), p.degree(), p.initial().to_expr(), p.separant().to_expr()
    (Derivative(u(x), x), 2, 1, 2*Derivative(u(x), x))
    >>> p.derivative(0).to_expr()
    2*Derivative(u(x), x)*Derivative(u(x), (x, 2)) - 4*Derivative(u(x), x)
    """

    def __init__(self, ring: DifferentialRing, terms: Terms) -> None:
        self.ring: DifferentialRing = ring
        self.native: native_polynomial.DifferentialPolynomial = native_polynomial.DifferentialPolynomial(
            ring.native, ring.terms_in(terms))

    @classmethod
    def _new(cls, ring: DifferentialRing, native: native_polynomial.DifferentialPolynomial) -> DifferentialPolynomial:
        self = object.__new__(cls)
        self.ring = ring
        self.native = native
        return self

    def _wrap(self, native: native_polynomial.DifferentialPolynomial) -> DifferentialPolynomial:
        return DifferentialPolynomial._new(self.ring, native)

    @classmethod
    def from_expr(cls, ring: DifferentialRing, equation: Equation) -> DifferentialPolynomial:
        """The differential polynomial of an expression or an equality."""
        return cls._new(ring, native_polynomial.DifferentialPolynomial.from_expr(ring.native, _equation(equation)))

    @property
    def terms(self) -> Terms:
        """The monomials and their coefficients (elements of the ring's
        ``integers``)."""
        return self.ring.terms_out(self.native.terms)

    def to_expr(self) -> Expr:
        """The SymPy expression."""
        return _sympy_expr(self.native.to_expr())

    def __repr__(self) -> str:
        return "DifferentialPolynomial(%s)" % (self.to_expr(),)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DifferentialPolynomial):
            return NotImplemented
        return self.native == other.native

    def __hash__(self) -> int:
        return hash(self.native)

    def __bool__(self) -> bool:
        return bool(self.native)

    # ------------------------------------------------------------------
    # arithmetic

    def constant(self, c: DomainElement) -> DifferentialPolynomial:
        """The element ``c`` of the ring of coefficients."""
        return self._wrap(self.native.constant(self.ring.integer_in(c)))

    def __neg__(self) -> DifferentialPolynomial:
        return self._wrap(-self.native)

    def __add__(self, other: DifferentialPolynomial) -> DifferentialPolynomial:
        return self._wrap(self.native + other.native)

    def __sub__(self, other: DifferentialPolynomial) -> DifferentialPolynomial:
        return self._wrap(self.native - other.native)

    def __mul__(self, other: DifferentialPolynomial) -> DifferentialPolynomial:
        return self._wrap(self.native*other.native)

    def __pow__(self, n: int) -> DifferentialPolynomial:
        return self._wrap(self.native**n)

    def primitive(self) -> DifferentialPolynomial:
        """The polynomial divided by the greatest common divisor of its
        coefficients, with a positive leading coefficient."""
        return self._wrap(self.native.primitive())

    # ------------------------------------------------------------------
    # structure

    @property
    def is_ground(self) -> bool:
        """Whether the polynomial lies in the field of coefficients."""
        return self.native.is_ground

    def jets(self) -> set[Jet]:
        """The derivatives which occur."""
        return self.native.jets()

    def leader(self) -> Jet:
        """The highest derivative."""
        return self.native.leader()

    def leader_expr(self) -> Expr:
        """The leader as an expression."""
        return _sympy_expr(self.native.leader_expr())

    def degree(self, jet: Optional[Jet] = None) -> int:
        """The degree in a derivative (the leader by default)."""
        return self.native.degree(jet)

    def rank(self) -> Rank:
        """The rank: the leader and the degree in it."""
        return self.native.rank()

    def coefficients(self, jet: Jet) -> dict[int, DifferentialPolynomial]:
        """The coefficients as a polynomial in one derivative, by degree."""
        return {e: self._wrap(c) for e, c in self.native.coefficients(jet).items()}

    def initial(self) -> DifferentialPolynomial:
        """The coefficient of the highest power of the leader."""
        return self._wrap(self.native.initial())

    def separant(self) -> DifferentialPolynomial:
        """The partial derivative with respect to the leader."""
        return self._wrap(self.native.separant())

    def scaled(self, c: DomainElement) -> DifferentialPolynomial:
        """The product with an element of the ring of coefficients."""
        return self._wrap(self.native.scaled(self.ring.integer_in(c)))

    def power(self, jet: Jet, e: int) -> DifferentialPolynomial:
        """The power of a derivative."""
        return self._wrap(self.native.power(jet, e))

    def derivative(self, i: int) -> DifferentialPolynomial:
        """The derivative with respect to the variable of index ``i``."""
        return self._wrap(self.native.derivative(i))

    def prolonged(self, theta: Sequence[int]) -> DifferentialPolynomial:
        """The derivative of the given orders."""
        return self._wrap(self.native.prolonged(theta))

    # ------------------------------------------------------------------
    # reduction

    def pseudo_remainder(self, divisor: DifferentialPolynomial, jet: Jet) -> DifferentialPolynomial:
        """The pseudo-remainder of the division by ``divisor`` as
        polynomials in the derivative ``jet``."""
        return self._wrap(self.native.pseudo_remainder(divisor.native, jet))

    def is_partially_reduced(self, other: DifferentialPolynomial) -> bool:
        """Whether no proper derivative of the leader of ``other``
        occurs."""
        return self.native.is_partially_reduced(other.native)

    def is_reduced(self, other: DifferentialPolynomial) -> bool:
        """Whether the polynomial is partially reduced with respect to
        ``other`` and of lower degree in its leader."""
        return self.native.is_reduced(other.native)

    def partial_remainder(self, chain: Sequence[DifferentialPolynomial]) -> DifferentialPolynomial:
        """Ritt's partial remainder by the chain."""
        return self._wrap(self.native.partial_remainder([a.native for a in chain]))

    def remainder(self, chain: Sequence[DifferentialPolynomial]) -> DifferentialPolynomial:
        """Ritt's full remainder by an autoreduced set."""
        return self._wrap(self.native.remainder([a.native for a in chain]))


# ----------------------------------------------------------------------
# Janet bases

class JanetBasis:
    """The Janet basis of a linear system of differential equations
    (:class:`sympy_extras.polys.differential.JanetBasis` on SymPy objects);
    see :func:`janet_basis`.

    Attributes
    ==========

    ring : DifferentialRing
        The functions, the variables and the ranking.
    multiplicative : list of tuples of bool
        For every equation, the flags of the multiplicative variables.
    """

    def __init__(self, native: native_janet.JanetBasis) -> None:
        self.native: native_janet.JanetBasis = native
        self.ring: DifferentialRing = DifferentialRing._new(native.ring)
        self.multiplicative: list[tuple[bool, ...]] = native.multiplicative

    @property
    def forms(self) -> list[LinearForm]:
        """The equations, with coprime polynomial coefficients (elements of
        the ring's ``integers``), by increasing leader."""
        return [{jet: self.ring.integer_out(c) for jet, c in form.items()} for form in self.native.forms]

    def __repr__(self) -> str:
        return "JanetBasis(%s)" % (self.equations,)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JanetBasis):
            return NotImplemented
        return self.native == other.native

    def __hash__(self) -> int:
        return hash(self.native)

    @property
    def is_consistent(self) -> bool:
        """Whether the system has solutions: the basis is not `\\{1\\}`."""
        return self.native.is_consistent

    @property
    def equations(self) -> list[Expr]:
        """The expressions which vanish, each with the coefficient one at
        its leader, by increasing leader."""
        return [_sympy_expr(e) for e in self.native.equations]

    @property
    def leaders(self) -> list[Expr]:
        """The leaders of the equations."""
        return [_sympy_expr(e) for e in self.native.leaders]

    @property
    def multiplicative_variables(self) -> list[tuple[Symbol, ...]]:
        """The multiplicative variables of the equations."""
        return [tuple(_sympy_symbol(s) for s in flags) for flags in self.native.multiplicative_variables]

    def reduce(self, expr: Equation) -> Expr:
        """The Janet normal form of a linear differential expression.

        >>> from sympy import Function, symbols
        >>> from sympy_extras.polys.differential_sympy import janet_basis
        >>> x, y = symbols('x y')
        >>> u = Function('u')(x, y)
        >>> J = janet_basis([u.diff(x, 2) - u, u.diff(y) - u.diff(x)], [u])
        >>> J.reduce(u.diff(x, y, y) + u.diff(y))
        2*Derivative(u(x, y), y)
        """
        return _sympy_expr(self.native.reduce(_equation(expr)))

    def contains(self, expr: Equation) -> bool:
        """Whether the expression is a linear differential consequence of
        the equations."""
        return self.native.contains(_equation(expr))

    def __contains__(self, expr: Equation) -> bool:
        return self.contains(expr)

    def is_parametric(self, derivative: Expr) -> bool:
        """Whether the derivative is not a derivative of a leader."""
        return self.native.is_parametric(_core_expr(derivative))

    def parametric_derivatives(self, order: int) -> list[Expr]:
        """The parametric derivatives up to the given order, by increasing
        rank."""
        return [_sympy_expr(e) for e in self.native.parametric_derivatives(order)]

    def hilbert_function(self, order: int) -> int:
        """The number of the parametric derivatives of exactly the given
        order."""
        return self.native.hilbert_function(order)

    def hilbert_series(self, t: Symbol) -> Expr:
        """The generating function of :meth:`hilbert_function`."""
        return _sympy_expr(self.native.hilbert_series(_core_symbol(t)))

    def hilbert_polynomial(self, s: Symbol) -> Expr:
        """The polynomial which :meth:`hilbert_function` equals for the
        large orders `s`; zero when the solution space has a finite
        dimension."""
        return _sympy_expr(self.native.hilbert_polynomial(_core_symbol(s)))

    @property
    def dimension(self) -> Expr:
        """The dimension of the solution space, ``oo`` when it is
        infinite."""
        return _sympy_expr(self.native.dimension)

    def series_solution(self, order: int, point: Optional[Sequence[ExprLike]] = None,
                        constant: str = 'C') -> dict[AppliedUndef, Expr]:
        """The Taylor polynomials of degree ``order`` of the general
        solution at a point (the origin by default).

        >>> from sympy import Function, symbols
        >>> from sympy_extras.polys.differential_sympy import janet_basis
        >>> x = symbols('x')
        >>> u = Function('u')(x)
        >>> janet_basis([u.diff(x, 2) + u], [u]).series_solution(4)[u]
        C0*x**4/24 - C0*x**2/2 + C0 - C1*x**3/6 + C1*x
        """
        values: Optional[list[CoreExpr]] = None
        if point is not None:
            values = [_core_expr(as_expr(sympify(v))) for v in point]
        found = self.native.series_solution(order, values, constant)
        return {f: _sympy_expr(found[g]) for f, g in zip(self.ring.functions, self.native.ring.functions)}


def _ring_arguments(functions: Sequence[Basic], variables: Optional[Sequence[Basic]],
                    ranking: Optional[Sequence[Sequence[Basic]]]
                    ) -> tuple[list[CoreBasic], Optional[list[CoreBasic]], Optional[list[list[CoreBasic]]]]:
    return ([_core_basic(f) for f in functions], None if variables is None else [_core_basic(s) for s in variables],
            None if ranking is None else [[_core_basic(f) for f in block] for block in ranking])


def janet_basis(equations: Sequence[Equation], functions: Sequence[Basic],
                variables: Optional[Sequence[Basic]] = None,
                ranking: Optional[Sequence[Sequence[Basic]]] = None) -> JanetBasis:
    """The Janet basis of a linear system of differential equations with
    coefficients rational in the variables
    (:func:`sympy_extras.polys.differential.janet_basis` on SymPy
    objects).

    Examples
    ========

    Janet's example:

    >>> from sympy import Function, symbols
    >>> from sympy_extras.polys.differential_sympy import janet_basis
    >>> x, y, z = symbols('x y z')
    >>> u = Function('u')(x, y, z)
    >>> J = janet_basis([u.diff(z, 2) + y*u.diff(x, 2), u.diff(y, 2)], [u])
    >>> J.equations[:2]
    [Derivative(u(x, y, z), (y, 2)), Derivative(u(x, y, z), (x, 2)) + Derivative(u(x, y, z), (z, 2))/y]
    >>> len(J.equations)
    7
    >>> J.dimension
    12
    >>> t = symbols('t')
    >>> J.hilbert_series(t)
    t**4 + 3*t**3 + 4*t**2 + 3*t + 1

    An inconsistent system:

    >>> v = Function('v')(x, y)
    >>> janet_basis([v.diff(x) - y, v.diff(y)], [v]).equations
    [1]

    The compatibility condition of `u_x = f`, `u_y = g`, by the
    elimination of `u`:

    >>> w, f, g = Function('w')(x, y), Function('f')(x, y), Function('g')(x, y)
    >>> J = janet_basis([w.diff(x) - f, w.diff(y) - g], [w, f, g], ranking=[[w], [f, g]])
    >>> J.equations[0]
    Derivative(f(x, y), y) - Derivative(g(x, y), x)
    """
    core_functions, core_variables, core_ranking = _ring_arguments(functions, variables, ranking)
    return JanetBasis(native_janet.janet_basis([_equation(e) for e in equations], core_functions, core_variables,
                                               core_ranking))


# ----------------------------------------------------------------------
# the Rosenfeld-Groebner algorithm

class JetSymbols:
    """The symbols standing for the derivatives in the polynomial rings of
    a component, as SymPy dummies."""

    def __init__(self, ring: DifferentialRing, native: NativeJetSymbols) -> None:
        self.ring: DifferentialRing = ring
        self.native: NativeJetSymbols = native

    def symbol(self, jet: Jet) -> Dummy:
        """The symbol of a derivative."""
        s = to_sympy(self.native.symbol(jet))
        if not isinstance(s, Dummy):
            raise TypeError("expected a dummy, got %s" % (s,))
        return s

    def expr(self, p: DifferentialPolynomial) -> Expr:
        """The polynomial as an expression in the symbols."""
        return _sympy_expr(self.native.expr(p.native))


class RegularDifferentialSystem:
    """A regular differential system `A = 0`, `H \\ne 0`: a component of
    the decomposition computed by :func:`rosenfeld_groebner`
    (:class:`sympy_extras.polys.differential.RegularDifferentialSystem`
    on SymPy objects)."""

    def __init__(self, ring: DifferentialRing, native: NativeSystem) -> None:
        self.ring: DifferentialRing = ring
        self.native: NativeSystem = native
        self.chain: list[DifferentialPolynomial] = [DifferentialPolynomial._new(ring, p) for p in native.chain]
        self.nonzero: list[DifferentialPolynomial] = [DifferentialPolynomial._new(ring, p) for p in native.nonzero]
        self.symbols: JetSymbols = JetSymbols(ring, native.symbols)

    def __repr__(self) -> str:
        return "RegularDifferentialSystem(%s, %s)" % (self.equations, self.inequations)

    @property
    def equations(self) -> list[Expr]:
        """The expressions which vanish (a coherent autoreduced set), by
        increasing leader."""
        return [_sympy_expr(e) for e in self.native.equations]

    @property
    def inequations(self) -> list[Expr]:
        """The expressions which do not vanish."""
        return [_sympy_expr(e) for e in self.native.inequations]

    @property
    def leaders(self) -> list[Expr]:
        """The leaders of the equations."""
        return [_sympy_expr(e) for e in self.native.leaders]

    def saturation(self) -> list[Expr]:
        """A Gröbner basis of the ideal `(A) : H^\\infty` in symbols
        standing for the derivatives."""
        return [_sympy_expr(e) for e in self.native.saturation()]

    def regular_chains(self) -> list[RegularChain]:
        """Squarefree regular chains `C_i` with `(A) : H^\\infty = \\bigcap
        \\operatorname{sat}(C_i)`, chains of
        :mod:`sympy_extras.polys.regularchains_sympy`."""
        return [sympy_chain(T) for T in self.native.regular_chains()]

    @property
    def is_consistent(self) -> bool:
        """Whether the system has solutions: `1 \\notin (A) : H^\\infty`."""
        return self.native.is_consistent

    def remainder(self, expr: Equation) -> Expr:
        """Ritt's remainder of a differential polynomial by the
        equations."""
        return _sympy_expr(self.native.remainder(_equation(expr)))

    def contains(self, expr: Equation) -> bool:
        """Whether the differential polynomial lies in `[A] : H^\\infty`."""
        return self.native.contains(_equation(expr))

    def __contains__(self, expr: Equation) -> bool:
        return self.contains(expr)


class RadicalDifferentialIdeal:
    """The radical differential ideal of a system, as the intersection of
    the regular differential ideals of its components; see
    :func:`rosenfeld_groebner`."""

    def __init__(self, native: NativeIdeal) -> None:
        self.native: NativeIdeal = native
        self.ring: DifferentialRing = DifferentialRing._new(native.ring)
        self.components: list[RegularDifferentialSystem] = [
            RegularDifferentialSystem(self.ring, c) for c in native.components]

    def __repr__(self) -> str:
        return "RadicalDifferentialIdeal(%s)" % (self.components,)

    def __len__(self) -> int:
        return len(self.components)

    @property
    def is_consistent(self) -> bool:
        """Whether the system has a solution (in some differential
        extension field)."""
        return self.native.is_consistent

    def contains(self, expr: Equation) -> bool:
        """Whether the differential polynomial vanishes on every solution
        of the system."""
        return self.native.contains(_equation(expr))

    def __contains__(self, expr: Equation) -> bool:
        return self.contains(expr)


def _characteristic_set(polynomials: Sequence[DifferentialPolynomial]) -> list[DifferentialPolynomial]:
    """An autoreduced subset of lowest rank."""
    if not polynomials:
        return []
    ring = polynomials[0].ring
    return [DifferentialPolynomial._new(ring, p) for p in native_characteristic_set([q.native for q in polynomials])]


def _delta_polynomials(chain: Sequence[DifferentialPolynomial]) -> list[DifferentialPolynomial]:
    """The cross derivatives of the pairs with leaders which are
    derivatives of the same function."""
    if not chain:
        return []
    ring = chain[0].ring
    return [DifferentialPolynomial._new(ring, p) for p in native_delta_polynomials([q.native for q in chain])]


def rosenfeld_groebner(equations: Sequence[Equation], functions: Sequence[Basic],
                       variables: Optional[Sequence[Basic]] = None,
                       ranking: Optional[Sequence[Sequence[Basic]]] = None,
                       inequations: Sequence[Expr] = ()) -> RadicalDifferentialIdeal:
    """The radical of the differential ideal of a polynomial system of
    differential equations, as an intersection of regular differential
    ideals (:func:`sympy_extras.polys.differential.rosenfeld_groebner` on
    SymPy objects).

    Examples
    ========

    The equation `u'^2 = 4u` has the general solution `(x + c)^2` and the
    singular solution `0`, on which the separant `2u'` vanishes:

    >>> from sympy import Function, symbols
    >>> from sympy_extras.polys.differential_sympy import rosenfeld_groebner
    >>> x = symbols('x')
    >>> u = Function('u')(x)
    >>> I = rosenfeld_groebner([u.diff(x)**2 - 4*u], [u])
    >>> [(c.equations, c.inequations) for c in I.components]
    [([-4*u(x) + Derivative(u(x), x)**2], [Derivative(u(x), x)]), ([u(x)], [])]
    >>> I.contains(u.diff(x)*(u.diff(x, 2) - 2))
    True
    >>> I.contains(u.diff(x, 2) - 2)
    False
    >>> I.components[0].contains(u.diff(x, 2) - 2)
    True

    A system without solutions:

    >>> rosenfeld_groebner([u.diff(x) - u, u.diff(x, 2) - u - 1], [u]).is_consistent
    False
    """
    core_functions, core_variables, core_ranking = _ring_arguments(functions, variables, ranking)
    return RadicalDifferentialIdeal(native_rosenfeld_groebner(
        [_equation(e) for e in equations], core_functions, core_variables, core_ranking,
        [_core_expr(as_expr(sympify(h))) for h in inequations]))

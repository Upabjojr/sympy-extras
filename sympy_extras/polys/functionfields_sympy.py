r"""The SymPy face of the algebraic function fields
(:mod:`sympy_extras.polys.functionfields`).

:class:`FunctionField`, :class:`Element` and :func:`integral_basis` take
and return SymPy objects as the module did before it was native: the
field of constants ``K`` is SymPy's ``QQ``, an
:class:`~sympy.polys.domains.AlgebraicField` or the rational functions
``QQ.frac_field(a, b, ...)`` of the parameters, the coordinates of an
element are elements of SymPy's ``K.frac_field(x)`` (``Kx``), and
``R = K.poly_ring(x)``. The algorithms (the integral bases by Trager's
algorithm, the local bases at infinity, the normalization at infinity,
the products, inverses, derivatives, traces, discriminants, coordinates
and the integrality test) are native: the face converts the
coordinates to the native field of the same curve over the native field
of constants (the number fields through
:func:`sympy_extras.polys.puiseux_sympy.native_field`, so that they are
the fields of the Puiseux expansions) and back.

Examples
========

>>> from sympy import symbols
>>> from sympy_extras.polys.functionfields_sympy import FunctionField, integral_basis
>>> x, y = symbols('x y')
>>> integral_basis(y**3 - x**2*(x + 1), x, y)
[1, y, y**2/x]
>>> E = FunctionField(y**2 - x**3 - 1, x, y)
>>> [w.as_expr() for w in E.normal_basis()], E.pole_orders_at_infinity(E.normal_basis())
([1, y], [0, 2])
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Generic, Optional, TypeVar

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly, cancel

from sympy_extras._typing import DomainElement, ExprLike, as_expr, free_symbols, sorted_symbols
from sympy_extras.assumptions.intervals_sympy import core_expr, core_symbol, sympy_expr
from sympy_extras.polys.domains import QQ as NativeQQ
from sympy_extras.polys.domains.fractionfield import FracElement
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing

from . import functionfields as native
from .puiseux_sympy import native_field, native_number, sympy_number

__all__ = ['FunctionField', 'Element', 'integral_basis']

T = TypeVar('T')

#: the coordinates of an element of the field (elements of SymPy's ``Kx``)
Coordinates = list[DomainElement]


class Element:
    """An element of a :class:`FunctionField`, by its coordinates in the
    power basis `1, y, \\ldots, y^{n-1}` over `K(x)` (``coefficients[i]``
    the coefficient of `y^i`, elements of SymPy's fraction field
    ``field.Kx``)."""

    def __init__(self, field: FunctionField, coefficients: list[DomainElement]) -> None:
        self.field = field
        n = field.n
        Kx = field.Kx
        padded = list(coefficients) + [Kx.zero] * (n - len(coefficients))
        self.coefficients: list[DomainElement] = padded[:n]

    def __add__(self, other: Element) -> Element:
        return Element(self.field, [a + b for a, b in zip(self.coefficients, other.coefficients)])

    def __sub__(self, other: Element) -> Element:
        return Element(self.field, [a - b for a, b in zip(self.coefficients, other.coefficients)])

    def __neg__(self) -> Element:
        return Element(self.field, [-a for a in self.coefficients])

    def scaled(self, c: DomainElement) -> Element:
        """The element times ``c`` of `K(x)`."""
        return Element(self.field, [c * a for a in self.coefficients])

    def __mul__(self, other: Element) -> Element:
        return Element(self.field, self.field.case.mul(self.coefficients, other.coefficients))

    def power(self, k: int) -> Element:
        result = self.field.one()
        base = self
        while k > 0:
            if k % 2:
                result = result * base
            k //= 2
            if k:
                base = base * base
        return result

    def is_zero(self) -> bool:
        return all(a == self.field.Kx.zero for a in self.coefficients)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Element) and other.field is self.field and other.coefficients == self.coefficients

    def __hash__(self) -> int:
        return hash(tuple(self.coefficients))

    def inverse(self) -> Element:
        return self.field.invert(self)

    def __truediv__(self, other: Element) -> Element:
        return self * other.inverse()

    def derivative(self) -> Element:
        """The derivative with respect to `x` (`y' = -F_x/F_y`)."""
        return Element(self.field, self.field.case.derivative(self.coefficients))

    def trace(self) -> DomainElement:
        return self.field.trace(self)

    def as_expr(self) -> Expr:
        """The element as an expression in `x` and in the generator of
        the field as given (the original `y`, before the leading
        coefficient was absorbed)."""
        field = self.field
        Kx = field.Kx
        y_integral = field.unit * field.y  # the integral generator in terms of the original one
        return as_expr(cancel(Add(*[Kx.to_sympy(c) * y_integral**i for i, c in enumerate(self.coefficients)])))

    def __repr__(self) -> str:
        return f"Element({self.as_expr()})"


# ---------------------------------------------------------------------------
# the native field of a curve, with the conversions of the coordinates

class _Case:
    """The native function field of a :class:`FunctionField`, seen
    through the coordinates of SymPy (``Kx`` elements): every operation
    converts its arguments, runs natively and converts back."""

    def mul(self, a: Coordinates, b: Coordinates) -> Coordinates:
        raise NotImplementedError

    def reduce(self, a: Coordinates) -> Coordinates:
        raise NotImplementedError

    def invert(self, a: Coordinates) -> Coordinates:
        raise NotImplementedError

    def derivative(self, a: Coordinates) -> Coordinates:
        raise NotImplementedError

    def derivative_of_y(self) -> Coordinates:
        raise NotImplementedError

    def power_traces(self) -> Coordinates:
        raise NotImplementedError

    def trace(self, a: Coordinates) -> DomainElement:
        raise NotImplementedError

    def element(self, e: Expr) -> Coordinates:
        raise NotImplementedError

    def coordinates(self, a: Coordinates, basis: list[Coordinates]) -> Coordinates:
        raise NotImplementedError

    def discriminant(self, basis: list[Coordinates]) -> DomainElement:
        raise NotImplementedError

    def is_integral(self, a: Coordinates) -> bool:
        raise NotImplementedError

    def integral_basis(self) -> list[Coordinates]:
        raise NotImplementedError

    def local_basis_at_infinity(self) -> list[Coordinates]:
        raise NotImplementedError

    def normal_basis(self) -> list[Coordinates]:
        raise NotImplementedError

    def pole_order_at_infinity(self, a: Coordinates) -> int:
        raise NotImplementedError


class _CaseOf(_Case, Generic[T]):
    """:class:`_Case` for a native field of constants with elements of
    type ``T``, converted by ``to_native`` and ``from_native`` (the
    elements of SymPy's ``K`` and of the native one)."""

    def __init__(self, field: native.FunctionField[T], Kx: FractionField,
                 to_native: Callable[[DomainElement], T], from_native: Callable[[T], DomainElement]) -> None:
        self.native = field
        self.Kx = Kx
        self.to_native = to_native
        self.from_native = from_native
        # the native bases met, by the identity of the SymPy list of
        # coordinates, with that list as it was (it may change)
        self._bases: dict[int, tuple[list[Coordinates], list[native.Element[T]]]] = {}

    # the coordinates, both ways

    def _fraction(self, c: DomainElement) -> FracElement[T]:
        numerator = [self.to_native(a) for a in c.numer.to_dense()]
        denominator = [self.to_native(a) for a in c.denom.to_dense()]
        return self.native.Kx.new(numerator, denominator)

    def _sympy_fraction(self, c: FracElement[T]) -> DomainElement:
        field = self.Kx.field
        ring = field.ring
        numerator = ring.from_list([self.from_native(a) for a in c.numer])
        denominator = ring.from_list([self.from_native(a) for a in c.denom])
        return field.new(numerator, denominator)

    def _element(self, a: Coordinates) -> native.Element[T]:
        return native.Element(self.native, [self._fraction(c) for c in a])

    def _coordinates(self, e: native.Element[T]) -> Coordinates:
        return [self._sympy_fraction(c) for c in e.coefficients]

    def _basis(self, basis: list[Coordinates]) -> list[native.Element[T]]:
        key = id(basis)
        found = self._bases.get(key)
        if found is not None and found[0] == basis:
            return found[1]
        converted = [self._element(w) for w in basis]
        self._bases[key] = ([list(w) for w in basis], converted)
        return converted

    # the operations

    def mul(self, a: Coordinates, b: Coordinates) -> Coordinates:
        return self._coordinates(self._element(a) * self._element(b))

    def reduce(self, a: Coordinates) -> Coordinates:
        return [self._sympy_fraction(c) for c in self.native.reduce([self._fraction(c) for c in a])]

    def invert(self, a: Coordinates) -> Coordinates:
        return self._coordinates(self.native.invert(self._element(a)))

    def derivative(self, a: Coordinates) -> Coordinates:
        return self._coordinates(self._element(a).derivative())

    def derivative_of_y(self) -> Coordinates:
        return self._coordinates(self.native.derivative_of_y())

    def power_traces(self) -> Coordinates:
        return [self._sympy_fraction(c) for c in self.native.power_traces()]

    def trace(self, a: Coordinates) -> DomainElement:
        return self._sympy_fraction(self.native.trace(self._element(a)))

    def element(self, e: Expr) -> Coordinates:
        return self._coordinates(self.native.element(core_expr(e)))

    def coordinates(self, a: Coordinates, basis: list[Coordinates]) -> Coordinates:
        return [self._sympy_fraction(c) for c in self.native.coordinates(self._element(a), self._basis(basis))]

    def discriminant(self, basis: list[Coordinates]) -> DomainElement:
        return self._sympy_fraction(self.native.discriminant([self._element(w) for w in basis]))

    def is_integral(self, a: Coordinates) -> bool:
        return self.native.is_integral(self._element(a))

    def integral_basis(self) -> list[Coordinates]:
        return [self._coordinates(w) for w in self.native.integral_basis()]

    def local_basis_at_infinity(self) -> list[Coordinates]:
        return [self._coordinates(w) for w in self.native.local_basis_at_infinity()]

    def normal_basis(self) -> list[Coordinates]:
        return [self._coordinates(w) for w in self.native.normal_basis()]

    def pole_order_at_infinity(self, a: Coordinates) -> int:
        return self.native.pole_order_at_infinity(self._element(a))


def _rational_case(F: Expr, x: Symbol, y: Symbol, Kx: FractionField) -> _Case:
    field: native.FunctionField[Fraction] = native.function_field(core_expr(F), core_symbol(x), core_symbol(y),
                                                                  NativeQQ)

    def to_native(c: DomainElement) -> Fraction:
        return Fraction(int(c.numerator), int(c.denominator))

    def from_native(c: Fraction) -> DomainElement:
        return QQ(c.numerator, c.denominator)

    return _CaseOf(field, Kx, to_native, from_native)


def _algebraic_case(F: Expr, x: Symbol, y: Symbol, K: AlgebraicField, Kx: FractionField) -> _Case:
    L = native_field(K)
    field: native.FunctionField[list[Fraction]] = native.function_field(core_expr(F), core_symbol(x),
                                                                        core_symbol(y), L)

    def to_native(c: DomainElement) -> list[Fraction]:
        return native_number(c, K)

    def from_native(c: list[Fraction]) -> DomainElement:
        return sympy_number(c, K)

    return _CaseOf(field, Kx, to_native, from_native)


def _parametric_case(F: Expr, x: Symbol, y: Symbol, K: FractionField, Kx: FractionField) -> _Case:
    ground = K.dom
    symbols = K.symbols
    if ground is None or symbols is None or not (ground.is_QQ or ground.is_ZZ):
        raise ValueError(f"constants in {K} are not supported")
    L = SparseFractionField(RationalPolynomialRing([str(s) for s in symbols]))
    field = native.function_field(core_expr(F), core_symbol(x), core_symbol(y), L)

    def to_native(c: DomainElement) -> SparseFracElement[Fraction]:
        return L.from_expr(core_expr(as_expr(K.to_sympy(c))))

    def from_native(c: SparseFracElement[Fraction]) -> DomainElement:
        return K.from_sympy(sympy_expr(L.to_expr(c)))

    return _CaseOf(field, Kx, to_native, from_native)


class FunctionField:
    """The algebraic function field `K(x)[y]/(F)` of the curve `F(x, y) =
    0` (`F` squarefree in `y` with polynomial coefficients; when its
    leading coefficient `a_n(x)` is not constant the generator is `a_n
    y`, which is integral, and ``unit`` records `a_n`).

    >>> from sympy import symbols
    >>> from sympy_extras.polys.functionfields_sympy import FunctionField
    >>> x, y = symbols('x y')
    >>> E = FunctionField(y**3 - x**2*(x + 1), x, y)
    >>> [w.as_expr() for w in E.integral_basis()]
    [1, y, y**2/x]
    >>> E.element(y**2/x) * E.element(y)
    Element(x**2 + x)
    """

    def __init__(self, F: ExprLike, x: Symbol, y: Symbol, domain: Optional[Domain] = None) -> None:
        F_ = as_expr(F)
        self.x = x
        self.y = y
        P = Poly(F_, y)
        n = P.degree()
        if n < 1:
            raise ValueError(f"{F_} has no root y")
        parameters = sorted_symbols(free_symbols(F_) - {x, y})
        K: Domain
        if domain is not None:
            K = domain
        else:
            K = QQ.frac_field(*parameters) if parameters else QQ
        self.K = K
        self.R: PolynomialRing = K.poly_ring(x)
        self.Kx: FractionField = K.frac_field(x)
        self.n = n
        leading = as_expr(P.LC())
        coefficients = [as_expr(c) for c in P.all_coeffs()]  # high to low
        # a_n y^n + ... + a_0 = 0: Y = a_n y satisfies Y^n + a_{n-1} Y^{n-1} +
        # a_n a_{n-2} Y^{n-2} + ... + a_n^{n-1} a_0 = 0
        self.unit: Expr = leading
        monic: list[Expr] = []
        for k, c in enumerate(coefficients):
            # coefficient of Y^(n-k): a_{n-k} a_n^(k-1) for k >= 1, 1 for k = 0
            monic.append(as_expr(cancel(c * leading**(k - 1))) if k >= 1 else as_expr(1))
        self.F: list[DomainElement] = [self.Kx.from_sympy(c) for c in monic]  # high to low, monic
        for c in monic:
            try:
                self.R.from_sympy(c)
            except CoercionFailed:
                raise ValueError(f"{F_}: polynomial coefficients expected") from None
        self.original = F_
        self._case: Optional[_Case] = None
        self._integral_basis: Optional[list[Element]] = None
        self._local_at_infinity: Optional[list[Element]] = None
        self._normal: Optional[list[Element]] = None
        self._dy: Optional[Element] = None

    @property
    def case(self) -> _Case:
        """The native field of the curve, with the conversions."""
        if self._case is None:
            K = self.K
            if isinstance(K, AlgebraicField):
                self._case = _algebraic_case(self.original, self.x, self.y, K, self.Kx)
            elif isinstance(K, FractionField):
                self._case = _parametric_case(self.original, self.x, self.y, K, self.Kx)
            elif K.is_QQ:
                self._case = _rational_case(self.original, self.x, self.y, self.Kx)
            else:
                raise ValueError(f"constants in {K} are not supported")
        return self._case

    # Construction of elements

    def convert(self, e: Element) -> Element:
        """The element ``e`` of the field of the same curve over a smaller
        coefficient field, here."""
        if e.field is self:
            return e
        Kx = self.Kx
        return Element(self, [Kx.from_sympy(as_expr(e.field.Kx.to_sympy(c))) for c in e.coefficients])

    def zero(self) -> Element:
        return Element(self, [])

    def one(self) -> Element:
        return Element(self, [self.Kx.one])

    def generator(self) -> Element:
        """The integral generator `Y = a_n y`."""
        return Element(self, [self.Kx.zero, self.Kx.one])

    def constant(self, c: ExprLike) -> Element:
        return Element(self, [self.Kx.from_sympy(as_expr(c))])

    def element(self, expr: ExprLike) -> Element:
        """The element given by a rational expression in `x` and `y`."""
        e = as_expr(expr)
        if self.y not in e.free_symbols:
            return self.constant(e)
        return Element(self, self.case.element(e))

    # Arithmetic

    def reduce(self, coefficients: list[DomainElement]) -> list[DomainElement]:
        """The coefficients (low degree first) reduced modulo `F`."""
        return self.case.reduce(coefficients)

    def invert(self, e: Element) -> Element:
        """The inverse of ``e`` (the extended Euclidean algorithm modulo
        `F`)."""
        if e.is_zero():
            raise ZeroDivisionError("inverse of zero")
        return Element(self, self.case.invert(e.coefficients))

    def derivative_of_y(self) -> Element:
        """`Y' = -F_x(Y)/F_Y(Y)` for the integral generator."""
        if self._dy is None:
            self._dy = Element(self, self.case.derivative_of_y())
        return self._dy

    def power_traces(self) -> list[DomainElement]:
        """`\\operatorname{Tr}(Y^k)` for `k < 2n - 1` by Newton's identities."""
        return self.case.power_traces()

    def trace(self, e: Element) -> DomainElement:
        return self.case.trace(e.coefficients)

    # Bases

    def coordinates(self, e: Element, basis: list[Element]) -> list[DomainElement]:
        """The coordinates of ``e`` in ``basis`` (elements of `K(x)`)."""
        return self.case.coordinates(e.coefficients, [w.coefficients for w in basis])

    def discriminant(self, basis: list[Element]) -> DomainElement:
        """`\\det(\\operatorname{Tr}(w_i w_j))`, an element of `K(x)` (a
        polynomial for integral elements)."""
        return self.case.discriminant([w.coefficients for w in basis])

    def is_integral(self, e: Element) -> bool:
        """Whether ``e`` is integral over `K[x]`: the characteristic
        polynomial of the multiplication by ``e`` has polynomial
        coefficients."""
        return self.case.is_integral(e.coefficients)

    def integral_basis(self) -> list[Element]:
        """An integral basis, triangular: the `i`-th element is a
        polynomial of degree `i` in `Y` whose leading coefficient is a
        monic polynomial in `x`, over a polynomial denominator (Trager's
        algorithm; the basis is in Hermite normal form).

        >>> from sympy import symbols
        >>> from sympy_extras.polys.functionfields_sympy import FunctionField
        >>> x, y = symbols('x y')
        >>> [w.as_expr() for w in FunctionField(y**4 + (x**2 + x)*y - x**2, x, y).integral_basis()]
        [1, y, y**2, y**3/x]
        """
        if self._integral_basis is None:
            self._integral_basis = [Element(self, w) for w in self.case.integral_basis()]
        return self._integral_basis

    def local_basis_at_infinity(self) -> list[Element]:
        """A basis of the elements integral at infinity over the local
        ring at `x = \\infty`: the integral basis of the curve in `z =
        1/x`, with `Y = z^m y` integral (`m` the least integer with `\\deg
        a_i \\le m (n - i)`), written back in `x` and `Y`.

        >>> from sympy import symbols
        >>> from sympy_extras.polys.functionfields_sympy import FunctionField
        >>> x, y = symbols('x y')
        >>> [w.as_expr() for w in FunctionField(y**2 - x**3 - 1, x, y).local_basis_at_infinity()]
        [1, y/x**2]
        """
        if self._local_at_infinity is None:
            self._local_at_infinity = [Element(self, w) for w in self.case.local_basis_at_infinity()]
        return self._local_at_infinity

    def pole_order_at_infinity(self, e: Element) -> int:
        """The least `k` with `x^{-k} e` integral at infinity (negative
        for a zero at infinity)."""
        return self.case.pole_order_at_infinity(e.coefficients)

    def normal_basis(self) -> list[Element]:
        """An integral basis normal at infinity (Trager): `(x^{-k_i}
        w_i)`, `k_i` the pole orders at infinity, is a local basis at
        infinity, so that `\\sum a_i w_i` with polynomial `a_i` has pole
        order `\\max(\\deg a_i + k_i)` there.

        >>> from sympy import symbols
        >>> from sympy_extras.polys.functionfields_sympy import FunctionField
        >>> x, y = symbols('x y')
        >>> E = FunctionField(y**2 - x**3 - 1, x, y)
        >>> [w.as_expr() for w in E.normal_basis()], E.pole_orders_at_infinity(E.normal_basis())
        ([1, y], [0, 2])
        """
        if self._normal is None:
            self._normal = [Element(self, w) for w in self.case.normal_basis()]
        return self._normal

    def pole_orders_at_infinity(self, basis: list[Element]) -> list[int]:
        return [self.pole_order_at_infinity(w) for w in basis]

    def __repr__(self) -> str:
        return f"FunctionField({self.original}, {self.x}, {self.y})"


def integral_basis(F: ExprLike, x: Symbol, y: Symbol) -> list[Expr]:
    """An integral basis of the function field of the curve `F(x, y) = 0`
    (`F` squarefree in `y`): `n = \\deg_y F` elements `p_i(x, y)/d_i(x)`,
    `p_i` monic of degree `i` in the integral generator `a_n y`, which
    generate the integral closure of `K[x]` in `K(x)[y]/(F)` as a
    `K[x]`-module; Trager's algorithm.

    >>> from sympy import symbols
    >>> from sympy_extras.polys.functionfields_sympy import integral_basis
    >>> x, y = symbols('x y')
    >>> integral_basis(y**3 - x**2*(x + 1), x, y)
    [1, y, y**2/x]
    >>> integral_basis(y**2 - x**3, x, y)
    [1, y/x]
    >>> integral_basis(y**4 + (x**2 + x)*y - x**2, x, y)
    [1, y, y**2, y**3/x]

    A curve which is not monic in `y`: the basis is written in `y` with
    the leading coefficient absorbed in the generator (`x y` for `x y^2 =
    1`):

    >>> integral_basis(x*y**2 - 1, x, y)
    [1, x*y]
    """
    E = FunctionField(F, x, y)
    return [w.as_expr() for w in E.integral_basis()]


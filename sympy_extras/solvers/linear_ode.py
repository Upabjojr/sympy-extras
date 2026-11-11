"""Linear ordinary differential equations with polynomial coefficients:
polynomial, rational and hyperexponential solutions, first order right
factors and reduction of order.

For an operator `L = a_n(x) D^n + \\cdots + a_1(x) D + a_0(x)` with
polynomial coefficients:

* **Polynomial solutions** (Abramov, Bronstein, Petkovšek): the degree of
  a polynomial solution is a nonnegative integer root of the *indicial
  polynomial at infinity* `\\lambda(d) = \\sum_i \\mathrm{lc}(a_i)\\, d^{\\underline{i}}`
  over the `i` with `\\deg a_i - i` maximal, so a polynomial ansatz up to
  the largest such root and a linear system give all of them.
* **Rational solutions** (Abramov, Singer): the multiplicity of a root
  `c` of `a_n` in the denominator of a rational solution is bounded by
  the negative integer roots of the indicial polynomial at `c`
  (computed through the norm over the field of `c` when `c` is not
  rational); with the denominator `Q` so bounded, `y = P/Q` and the
  polynomial solutions `P` of the transformed operator are found.
* **Hyperexponential solutions** `y = \\exp(\\int \\rho)`, `\\rho` rational:
  the exponential part at infinity from the Newton polygon, those at the
  irregular finite points from :mod:`sympy_extras.solvers.irregular`, the
  local exponents at the finite singularities and at infinity from the
  roots of the indicial polynomials; the rational combinations of them
  bound the ansatz `y = \\prod (x - c)^{e_c} P(x)`, and `P` is again a
  polynomial solution. Each such solution gives a first order right factor
  `D - \\rho` of the operator (Beke's method for first order factors).
* **Reduction of order**: a known solution `y_1` turns `L` into an
  operator of order `n - 1` for `v = (y/y_1)'`.
* **Operator arithmetic** in the Ore ring `K[D]`, `D a = a D + a'`:
  composition, Euclidean division on the right, ``gcrd`` and ``lclm``,
  the adjoint, and the exterior and symmetric powers written as scalar
  operators through a cyclic vector (the associated operators of
  :mod:`sympy_extras.solvers.factorization`). For order three and more
  ``dsolve_linear`` factors the operator there and solves through the
  factors.

SymPy's ``dsolve`` solves linear equations with constant coefficients,
Euler equations and a few named second order equations, and has no
general algorithm for polynomial coefficients.

The module works on the objects of the core and imports no SymPy: the
polynomials (in one variable, with any constant coefficients: the
irrational numbers, ``I`` and the functions of the parameters which the
typed ``Poly`` refuses are taken as generators of the coefficients, as
SymPy's ``Poly`` does), their roots and factors, the linear algebra and
the simplifications are the native ones, and
:mod:`sympy_extras.solvers.linear_ode_sympy` is its face on SymPy objects.
Three steps are not ported and go through SymPy at run time: the
antiderivatives of the reduction of order and of the first order
equations (:func:`~sympy_extras.solvers.kovacic_sympy.integrate_of_core`),
the unevaluated integrals of what SymPy cannot integrate
(:func:`~sympy_extras.solvers.kovacic_sympy.integral_of_core`, an
``Integral`` of SymPy wrapped in a node of the core), and SymPy's
``dsolve``, the last resort on the reduced equations
(:func:`~sympy_extras.solvers.linear_ode_sympy.dsolve_of_core`).

References
==========

.. [Abramov] S. A. Abramov, M. Bronstein, M. Petkovšek, On polynomial
   solutions of linear operator equations, ISSAC 1995.
.. [Singer] M. F. Singer, Liouvillian solutions of n-th order homogeneous
   linear differential equations, American Journal of Mathematics 103
   (1981).
.. [Bronstein] M. Bronstein, On solutions of linear ordinary differential
   equations in their coefficient field, Journal of Symbolic Computation
   13 (1992).

Examples
========

>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.solvers.linear_ode import LinearOperator, polynomial_solutions
>>> x = Symbol('x')
>>> y = Function('y')(x)
>>> L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
>>> polynomial_solutions(L)
[x**2 + 2*x + 2]
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product as cartesian
from math import comb
from typing import Final, Optional, Sequence

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Derivative, Function, expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, exp
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import cancel, factor_list, gcd, gcd_list, lcm_list, resultant
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

from .kovacic import _coefficients, _roots

__all__ = ['LinearOperator', 'polynomial_solutions', 'rational_solutions',
           'hyperexponential_solutions', 'reduce_order_linear', 'dsolve_linear',
           'gcrd', 'lclm', 'log_derivative', 'hyperexponential_search']

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

#: what the polynomial routines raise on coefficients they cannot take
_FAILURES = (BasePolynomialError, NotImplementedError, ValueError, TypeError, ZeroDivisionError)


def _sum(terms: Sequence[Expr]) -> Expr:
    return add(*terms) if terms else _ZERO


def _product(factors: Sequence[Expr]) -> Expr:
    return mul(*factors) if factors else _ONE


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced, evaluated (SymPy's
    ``xreplace``)."""
    value = e._xreplace_evaluated(rule)
    if not isinstance(value, Expr):
        raise TypeError("%s is not an expression" % (value,))
    return value


def _normal(e: Expr) -> Expr:
    """The normal form ``p/q`` of a rational function (``e`` itself when
    the native ``cancel`` cannot take its coefficients)."""
    try:
        return cancel(e)
    except _FAILURES:
        return e


def has_integral(e: Basic) -> bool:
    """Whether ``e`` has an unevaluated ``Integral`` (of SymPy, wrapped in
    a node of the core: :func:`~sympy_extras.solvers.kovacic_sympy.integral_of_core`)."""
    return any(node._is_foreign_node() and node._compare_name() == 'Integral' for node in preorder_traversal(e))


# ---------------------------------------------------------------------------
# polynomials in one variable with any constant coefficients

class _Generic:
    """The coefficients which the typed ``Poly`` refuses (irrational
    numbers, ``I``, functions of the parameters) replaced by dummies, as
    SymPy's ``Poly`` takes them as generators of its domain, and back."""

    def __init__(self, x: Symbol) -> None:
        self.x: Final[Symbol] = x
        self.forward: Final[dict[Expr, Expr]] = {}
        self.back: Final[dict[Basic, Basic]] = {}

    def to(self, e: Expr) -> Expr:
        x = self.x

        def walk(node: Expr) -> Expr:
            if isinstance(node, (Symbol, Rational)):
                return node
            if isinstance(node, Add):
                return _sum([walk(a) for a in node.args if isinstance(a, Expr)])
            if isinstance(node, Mul):
                return _product([walk(a) for a in node.args if isinstance(a, Expr)])
            if isinstance(node, Pow) and isinstance(node.exp, Integer) and int(node.exp) >= 0:
                return walk(node.base)**node.exp
            if node.has(x):
                return node   # no polynomial in x: Poly refuses it
            if node not in self.forward:
                d = Dummy('g%d' % len(self.forward))
                self.forward[node] = d
                self.back[d] = node
            return self.forward[node]

        return walk(expand(e))

    def restore(self, e: Expr) -> Expr:
        return _replaced(e, self.back) if self.back else e


class UPoly:
    """A polynomial in ``x`` whose coefficients are any expressions free of
    ``x``: what the algorithms ask of SymPy's ``Poly`` (the degree, the
    coefficients, the division with remainder), with the coefficients
    normalised by the native ``cancel``."""

    def __init__(self, coefficients: dict[int, Expr], x: Symbol) -> None:
        self.terms_: Final[dict[int, Expr]] = {k: c for k, c in coefficients.items() if c != _ZERO}
        self.x: Final[Symbol] = x

    @staticmethod
    def of(e: Expr, x: Symbol) -> UPoly:
        """The polynomial of an expression; ``ValueError`` when it is no
        polynomial in ``x``."""
        return UPoly(_coefficients(e, x), x)

    @property
    def is_zero(self) -> bool:
        return not self.terms_

    def degree(self) -> int:
        """The degree (``-1`` for zero)."""
        return max(self.terms_) if self.terms_ else -1

    def LC(self) -> Expr:
        return self.terms_[self.degree()] if self.terms_ else _ZERO

    def coeff(self, k: int) -> Expr:
        return self.terms_.get(k, _ZERO)

    def all_coeffs(self) -> list[Expr]:
        """The coefficients from the leading one down (SymPy's order)."""
        return [self.coeff(k) for k in range(self.degree(), -1, -1)]

    def as_expr(self) -> Expr:
        x = self.x
        return _sum([c*x**k if k else c for k, c in sorted(self.terms_.items(), reverse=True)])

    def diff(self) -> UPoly:
        return UPoly({k - 1: k*c for k, c in self.terms_.items() if k}, self.x)

    def scaled(self, c: Expr) -> UPoly:
        return UPoly({k: _normal(c*a) for k, a in self.terms_.items()}, self.x)

    def __sub__(self, other: UPoly) -> UPoly:
        keys = set(self.terms_) | set(other.terms_)
        return UPoly({k: _normal(self.coeff(k) - other.coeff(k)) for k in keys}, self.x)

    def div(self, other: UPoly) -> tuple[UPoly, UPoly]:
        """``(q, r)`` with ``self = q other + r``, ``deg r < deg other``,
        over the field of the coefficients."""
        if other.is_zero:
            raise ZeroDivisionError("polynomial division by zero")
        m = other.degree()
        lc = other.LC()
        rest = dict(self.terms_)
        quotient: dict[int, Expr] = {}
        while rest and max(rest) >= m:
            top = max(rest)
            c = _normal(rest.pop(top)/lc)
            if c == _ZERO:
                continue
            quotient[top - m] = c
            for k, b in other.terms_.items():
                if k == m:
                    continue
                key = k + top - m
                value = _normal(rest.get(key, _ZERO) - c*b)
                if value == _ZERO:
                    rest.pop(key, None)
                else:
                    rest[key] = value
        return UPoly(quotient, self.x), UPoly(rest, self.x)

    def rem(self, other: UPoly) -> UPoly:
        return self.div(other)[1]

    def monic(self) -> UPoly:
        if self.is_zero:
            return self
        return self.scaled(_ONE/self.LC())

    def gcd(self, other: UPoly) -> UPoly:
        """The monic greatest common divisor (the native ``gcd``, Euclid's
        algorithm on coefficients it cannot take)."""
        x = self.x
        try:
            return UPoly.of(_polynomial_gcd(self.as_expr(), other.as_expr(), x), x).monic()
        except _FAILURES:
            pass
        a, b = self, other
        while not b.is_zero:
            a, b = b, a.rem(b)
        return a.monic()

    def __repr__(self) -> str:
        return "UPoly(%s, %s)" % (self.as_expr(), self.x)


def _polynomial_gcd(f: Expr, g: Expr, x: Symbol) -> Expr:
    """The native ``gcd`` of two polynomials in ``x``, with the
    coefficients it cannot take as generators."""
    try:
        return gcd(f, g, x)
    except _FAILURES:
        pass
    generic = _Generic(x)
    return generic.restore(gcd(generic.to(f), generic.to(g), x))


def _factor_list(e: Expr, x: Symbol) -> list[tuple[Expr, int]]:
    """The irreducible factors of a polynomial in ``x`` with their
    multiplicities, without the constant (the native ``factor_list``; the
    coefficients it cannot take as generators; the polynomial itself when
    both fail)."""
    try:
        _, found = factor_list(e, x)
    except _FAILURES:
        generic = _Generic(x)
        try:
            _, found = factor_list(generic.to(e), x)
            found = [(generic.restore(f), m) for f, m in found]
        except _FAILURES:
            return [(e, 1)]
    return [(f, int(m)) for f, m in found if isinstance(m, Integer)]


def _resultant(f: Expr, g: Expr, c: Symbol) -> Expr:
    """The resultant of two polynomials in ``c`` (the native one; the
    Sylvester determinant on coefficients it cannot take)."""
    try:
        return resultant(f, g, c)
    except _FAILURES:
        pass
    from ._special_values import determinant
    p, q = UPoly.of(f, c).all_coeffs(), UPoly.of(g, c).all_coeffs()
    m, n = len(p) - 1, len(q) - 1
    size = m + n
    rows: list[list[Expr]] = []
    for i in range(n):
        rows.append([_ZERO]*i + p + [_ZERO]*(size - m - 1 - i))
    for i in range(m):
        rows.append([_ZERO]*i + q + [_ZERO]*(size - n - 1 - i))
    return expand(determinant(rows))


def _lcm(exprs: Sequence[Expr]) -> Expr:
    """The least common multiple of polynomials (the native one; the
    product of the different ones on coefficients it cannot take)."""
    try:
        return lcm_list(list(exprs))
    except _FAILURES:
        pass
    result: Expr = _ONE
    seen: list[Expr] = []
    for e in exprs:
        if e not in seen:
            seen.append(e)
            result = result*e
    return expand(result)


def _gcd_list(exprs: Sequence[Expr]) -> Expr:
    try:
        return gcd_list(list(exprs))
    except _FAILURES:
        return _ONE


def _root_multiplicities(p: UPoly) -> dict[Expr, int]:
    """The roots which the native ``roots`` finds, with their
    multiplicities."""
    if p.degree() < 1:
        return {}
    try:
        return _roots(p.as_expr(), p.x)
    except _FAILURES:
        return {}


# ---------------------------------------------------------------------------

class LinearOperator:
    """``a_n D**n + ... + a_1 D + a_0`` with coefficients ``[a_0, ...,
    a_n]`` (rational functions) in ``x``.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    >>> L.coefficients, L.order
    ((2, -x - 2, x), 2)
    >>> L(x**2 + 2*x + 2)
    0
    """

    def __init__(self, coefficients: Sequence[Expr], x: Symbol) -> None:
        coefficients_ = list(coefficients)
        while len(coefficients_) > 1 and coefficients_[-1] == _ZERO:
            coefficients_.pop()
        if not coefficients_:
            coefficients_ = [_ZERO]
        self.coefficients: Final[tuple[Expr, ...]] = tuple(coefficients_)
        self.x: Final[Symbol] = x

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @classmethod
    def from_equation(cls, equation: Basic, f: AppliedUndef) -> LinearOperator:
        """The operator of a homogeneous linear equation in ``f``, the
        coefficients made polynomial by clearing denominators."""
        x_ = f.args[0]
        if not isinstance(x_, Symbol):
            raise ValueError("the function must depend on a symbol")
        x = x_
        if isinstance(equation, Eq):
            lhs = equation.lhs - equation.rhs
        elif isinstance(equation, Expr):
            lhs = equation
        else:
            raise TypeError("%s is not an equation" % (equation,))
        expanded = expand(lhs)
        n = 0
        for d in expanded.atoms(Derivative):
            if isinstance(d, Derivative) and d.expr == f:
                n = max(n, sum(count for v, count in d.variable_count if v == x))
        coefficients: list[Expr] = []
        rest = expanded
        for i in range(n, -1, -1):
            term = f.diff(x, i) if i > 0 else f
            c = rest.coeff(term)
            rest = expand(rest - c*term)
            coefficients.append(c)
        if rest != _ZERO or any(c.has(f) for c in coefficients):
            raise ValueError("%s is not a homogeneous linear equation in %s" % (equation, f))
        coefficients.reverse()
        denominators = [_normal(c).as_numer_denom()[1] for c in coefficients]
        common = _lcm(denominators)
        cleared = [_normal(c*common) for c in coefficients]
        for c in cleared:
            if not c.is_polynomial(x):
                raise ValueError("the coefficients are not rational functions of %s" % (x,))
        return cls(cleared, x)

    def __call__(self, y: Expr) -> Expr:
        x = self.x
        return expand(_sum([c*y.diff(x, i) if i else c*y for i, c in enumerate(self.coefficients)]))

    def __repr__(self) -> str:
        return "LinearOperator(%s, %s)" % (list(self.coefficients), self.x)

    def transformed(self, factor: Expr) -> LinearOperator:
        """The operator ``M`` with ``L(factor * u) = factor * M(u)``,
        denominators cleared (``factor`` may be rational or
        hyperexponential: only ``factor'/factor`` needs to be
        rational)."""
        x = self.x
        rho = log_derivative(factor, x)
        if rho.is_rational_function(x):
            # L(f u)/f = sum_i a_i sum_l binomial(i, l) (f^(i-l)/f) u^(l),
            # with f^(j)/f from the recurrence T_{j+1} = T_j' + rho T_j
            ratios: list[Expr] = [_ONE]
            for _ in range(self.order):
                ratios.append(_normal(ratios[-1].diff(x) + rho*ratios[-1]))
            shifted: list[Expr] = [_ZERO]*(self.order + 1)
            for i, a in enumerate(self.coefficients):
                if a == _ZERO:
                    continue
                for l in range(i + 1):
                    shifted[l] = shifted[l] + comb(i, l)*a*ratios[i - l]
            return LinearOperator([_normal(c) for c in shifted], x).cleared()
        u = Function('u')(x)
        image = expand(_normal(self(factor*u)/factor))
        coefficients: list[Expr] = []
        rest = image
        for i in range(self.order, -1, -1):
            term = u.diff(x, i) if i > 0 else u
            c = rest.coeff(term)
            rest = expand(rest - c*term)
            coefficients.append(_normal(c))
        coefficients.reverse()
        denominators = [_normal(c).as_numer_denom()[1] for c in coefficients]
        common = _lcm(denominators)
        return LinearOperator([_normal(c*common) for c in coefficients], x)

    # -- arithmetic in the Ore ring K[D], K = Q(x) (or its extension by
    # the constants of the coefficients), D a = a D + a'

    @property
    def leading_coefficient(self) -> Expr:
        return self.coefficients[-1]

    @property
    def is_zero(self) -> bool:
        return len(self.coefficients) == 1 and self.coefficients[0] == _ZERO

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinearOperator):
            return NotImplemented
        if self.x != other.x or self.order != other.order:
            return False
        return all(_normal(a - b) == _ZERO for a, b in zip(self.coefficients, other.coefficients))

    def __hash__(self) -> int:
        return hash((self.x, self.order))

    def __add__(self, other: LinearOperator) -> LinearOperator:
        n = max(self.order, other.order) + 1
        a = list(self.coefficients) + [_ZERO]*(n - len(self.coefficients))
        b = list(other.coefficients) + [_ZERO]*(n - len(other.coefficients))
        return LinearOperator([_normal(p + q) for p, q in zip(a, b)], self.x)

    def __neg__(self) -> LinearOperator:
        return LinearOperator([-c for c in self.coefficients], self.x)

    def __sub__(self, other: LinearOperator) -> LinearOperator:
        return self + (-other)

    def __mul__(self, other: LinearOperator) -> LinearOperator:
        """The composition ``self(other(y))``: ``a D^i b D^j = a sum_l
        binomial(i, l) b^(l) D^(i - l + j)`` (Leibniz's rule).

        Examples
        ========

        >>> from sympy_extras.core import Integer, Symbol
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> x = Symbol('x')
        >>> D = LinearOperator([Integer(0), Integer(1)], x)
        >>> D*LinearOperator([x], x)
        LinearOperator([1, x], x)
        >>> (D - LinearOperator([1/x], x))*(D + LinearOperator([1/x], x))
        LinearOperator([-2/x**2, 0, 1], x)
        """
        x = self.x
        top = max((i for i, a in enumerate(self.coefficients) if a != _ZERO), default=0)
        derivatives: list[list[Expr]] = []
        for b in other.coefficients:
            row = [b]
            for _ in range(top):
                row.append(_normal(row[-1].diff(x)))
            derivatives.append(row)
        result: list[Expr] = [_ZERO]*(self.order + other.order + 1)
        for i, a in enumerate(self.coefficients):
            if a == _ZERO:
                continue
            for l in range(i + 1):
                weight = comb(i, l)*a
                for j in range(len(other.coefficients)):
                    b = derivatives[j][l]
                    if b != _ZERO:
                        result[i - l + j] = result[i - l + j] + weight*b
        return LinearOperator([_normal(c) for c in result], x)

    def scaled(self, c: Expr) -> LinearOperator:
        """The operator ``c L`` (left multiplication by the function
        ``c``)."""
        return LinearOperator([_normal(c*a) for a in self.coefficients], self.x)

    def monic(self) -> LinearOperator:
        """The operator divided by its leading coefficient.

        Examples
        ========

        >>> from sympy_extras.core import Integer, Symbol
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> x = Symbol('x')
        >>> LinearOperator([Integer(1), x, x**2], x).monic()
        LinearOperator([x**(-2), 1/x, 1], x)
        """
        if self.is_zero:
            raise ValueError("the zero operator has no leading coefficient")
        return self.scaled(_ONE/self.leading_coefficient)

    def primitive(self) -> LinearOperator:
        """The operator times a rational function so that the
        coefficients are polynomials in ``x`` without common factor (the
        leading coefficient keeps the sign it had).

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> x = Symbol('x')
        >>> LinearOperator([2/x**2, 2/x, 2 + 0*x], x).primitive()
        LinearOperator([1, x, x**2], x)
        """
        if self.is_zero:
            return self
        x = self.x
        parts = [_normal(c).as_numer_denom() for c in self.coefficients]
        common = _lcm([q for _, q in parts])
        numerators = [_normal(p*common/q) for p, q in parts]
        content = _gcd_list([c for c in numerators if c != _ZERO])
        if content.has(x) or content.is_number:
            numerators = [_normal(c/content) for c in numerators]
        return LinearOperator(numerators, x)

    def cleared(self) -> LinearOperator:
        """The operator times the least common multiple of the
        denominators of its coefficients (polynomial coefficients, their
        common factors kept)."""
        if self.is_zero:
            return self
        parts = [_normal(c).as_numer_denom() for c in self.coefficients]
        common = _lcm([q for _, q in parts])
        return LinearOperator([_normal(p*common/q) for p, q in parts], self.x)

    def right_divmod(self, other: LinearOperator) -> tuple[LinearOperator, LinearOperator]:
        """``(Q, R)`` with ``self = Q*other + R`` and ``R`` of order less
        than ``other`` (Euclidean division on the right in the Ore
        ring).

        Examples
        ========

        >>> from sympy_extras.core import Integer, Symbol
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> x = Symbol('x')
        >>> L = LinearOperator([Integer(0), Integer(0), Integer(0), Integer(1)], x)
        >>> L.right_divmod(LinearOperator([Integer(-1), x], x))
        (LinearOperator([0, -1/x**2, 1/x], x), LinearOperator([0], x))
        """
        if other.is_zero:
            raise ZeroDivisionError("division by the zero operator")
        x = self.x
        quotient = LinearOperator([_ZERO], x)
        rest = self
        m = other.order
        lc = other.leading_coefficient
        while not rest.is_zero and rest.order >= m:
            d = rest.order - m
            term = LinearOperator([_ZERO]*d + [_normal(rest.leading_coefficient/lc)], x)
            quotient = quotient + term
            difference = rest - term*other
            # the leading coefficient cancels exactly: drop it even when
            # the normal form of its algebraic constants does not show it
            coefficients = difference.coefficients[:rest.order] if difference.order >= rest.order \
                else difference.coefficients
            rest = LinearOperator(coefficients or (_ZERO,), x)
        return quotient, rest

    def right_quotient(self, other: LinearOperator) -> LinearOperator:
        return self.right_divmod(other)[0]

    def right_remainder(self, other: LinearOperator) -> LinearOperator:
        return self.right_divmod(other)[1]

    def adjoint(self) -> LinearOperator:
        """The adjoint ``L* = sum (-1)**i D**i a_i``: the right factors of
        ``L*`` are the adjoints of the left factors of ``L``, and
        ``(A B)* = B* A*``.

        Examples
        ========

        >>> from sympy_extras.core import Integer, Symbol
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> x = Symbol('x')
        >>> LinearOperator([Integer(0), x, Integer(1)], x).adjoint()
        LinearOperator([-1, -x, 1], x)
        """
        x = self.x
        n = self.order
        derivatives: list[list[Expr]] = []
        for i, a in enumerate(self.coefficients):
            row = [a]
            for _ in range(i):
                row.append(_normal(row[-1].diff(x)))
            derivatives.append(row)
        result: list[Expr] = []
        for l in range(n + 1):
            total: Expr = _ZERO
            for i in range(l, n + 1):
                total = total + (-1)**i*comb(i, l)*derivatives[i][i - l]
            result.append(_normal(total))
        return LinearOperator(result, x)

    def exterior_power(self, k: int) -> LinearOperator:
        """The ``k``-th associated operator: the monic operator of least
        order annihilating the Wronskian of any ``k`` solutions (of
        order ``binomial(n, k)`` at most).

        Examples
        ========

        >>> from sympy_extras.core import Integer, Symbol
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> x = Symbol('x')
        >>> LinearOperator([Integer(-1), Integer(0), Integer(0), Integer(1)], x).exterior_power(2)
        LinearOperator([1, 0, 0, 1], x)
        """
        if not 1 <= k <= self.order:
            raise ValueError("the power must be between 1 and the order")
        subsets, system = _exterior_system(self.monic(), k)
        start: list[Expr] = [_ONE if i == 0 else _ZERO for i in range(len(subsets))]
        return _scalar_operator(system, start, self.x)[0]

    def symmetric_power(self, m: int) -> LinearOperator:
        """The ``m``-th symmetric power: the monic operator of least order
        annihilating the products of ``m`` solutions.

        Examples
        ========

        >>> from sympy_extras.core import Integer, Symbol
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> x = Symbol('x')
        >>> LinearOperator([Integer(1), Integer(0), Integer(1)], x).symmetric_power(2)
        LinearOperator([0, 4, 0, 1], x)
        """
        if m < 1:
            raise ValueError("the power must be positive")
        monomials, system = _symmetric_system(self.monic(), m)
        start: list[Expr] = [_ONE if i == 0 else _ZERO for i in range(len(monomials))]
        return _scalar_operator(system, start, self.x)[0]


# ---------------------------------------------------------------------------
# helpers of the operator arithmetic

def log_derivative(h: Expr, x: Symbol) -> Expr:
    """``h'/h``, computed factor by factor so that it comes out rational
    for a hyperexponential ``h`` written as a product of powers and
    exponentials.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers.linear_ode import log_derivative
    >>> x = Symbol('x')
    >>> log_derivative(sqrt(x - 1)*exp(x**2)/x, x)
    (4*x**3 - 4*x**2 - x + 2)/(2*x**2 - 2*x)
    """
    if not h.has(x):
        return _ZERO
    if isinstance(h, Mul):
        return _normal(_sum([log_derivative(f, x) for f in h.args if isinstance(f, Expr)]))
    if isinstance(h, Pow) and not h.exp.has(x):
        return _normal(h.exp*h.base.diff(x)/h.base)
    if isinstance(h, Exp):
        argument = h.args[0]
        if isinstance(argument, Expr):
            return _normal(argument.diff(x))
    return _normal(h.diff(x)/h)


class _RationalRows:
    """A matrix of rational functions over the native field of rational
    functions of its symbols (renamed), and the way back."""

    def __init__(self, rows: Sequence[Sequence[Expr]]) -> None:
        symbols = sorted({s for row in rows for e in row for s in e.free_symbols if isinstance(s, Symbol)},
                         key=lambda s: s.sort_key())
        names = ['_ln%d' % i for i in range(len(symbols))] or ['_ln']
        renamed: dict[Basic, Basic] = {s: Symbol(name) for s, name in zip(symbols, names)}
        self.restored: Final[dict[Basic, Basic]] = {Symbol(name): s for s, name in zip(symbols, names)}
        self.field: Final[SparseFractionField[Fraction]] = SparseFractionField(RationalPolynomialRing(names))
        self.rows: Final[list[list[SparseFracElement[Fraction]]]] = [
            [self.field.from_expr(_replaced(e, renamed)) for e in row] for row in rows]

    def expr(self, v: SparseFracElement[Fraction]) -> Expr:
        return _replaced(self.field.to_expr(v), self.restored)


_FIELD_FAILURES = (ValueError, TypeError, NotImplementedError, AttributeError, BasePolynomialError)


def _nullspace(rows: Sequence[Sequence[Expr]], columns: int) -> list[list[Expr]]:
    """A basis of the vectors annihilated by the rows, over the field of
    rational functions of the entries, read from the reduced row echelon
    form with the free coordinate one (SymPy's ``Matrix.nullspace``); over
    the expressions when an entry is not a rational function."""
    if not rows:
        return [[_ONE if k == n else _ZERO for k in range(columns)] for n in range(columns)]
    try:
        matrix = _RationalRows(rows)
    except _FIELD_FAILURES:
        return _expression_nullspace(rows, columns)
    return [[matrix.expr(v) for v in vector] for vector in matrix.field.nullspace(matrix.rows, columns)]


def _inverse(rows: Sequence[Sequence[Expr]]) -> Optional[list[list[Expr]]]:
    """The inverse of a square matrix of rational functions (the reduced
    row echelon form of ``[M | 1]``), ``None`` when it is singular."""
    n = len(rows)
    augmented = [list(row) + [_ONE if i == j else _ZERO for j in range(n)] for i, row in enumerate(rows)]
    try:
        matrix = _RationalRows(augmented)
    except _FIELD_FAILURES:
        from .recurr import _row_reduce
        reduced, pivots = _row_reduce(augmented)
        if pivots[:n] != list(range(n)):
            return None
        return [[_normal(reduced[i][n + j]) for j in range(n)] for i in range(n)]
    field = matrix.field
    sparse = [{j: c for j, c in enumerate(row) if not field.is_zero(c)} for row in matrix.rows]
    echelon = field.sparse_echelon([r for r in sparse if r])
    pivot_rows = {min(r): r for r in echelon}
    if sorted(pivot_rows)[:n] != list(range(n)):
        return None
    return [[matrix.expr(pivot_rows[i].get(n + j, field.zero)) for j in range(n)] for i in range(n)]


def _expression_nullspace(rows: Sequence[Sequence[Expr]], columns: int) -> list[list[Expr]]:
    from .recurr import _row_reduce
    reduced, pivots = _row_reduce([list(row) for row in rows])
    basis: list[list[Expr]] = []
    for free in range(columns):
        if free in pivots:
            continue
        vector: list[Expr] = [_ZERO]*columns
        vector[free] = _ONE
        for row, pivot in enumerate(pivots):
            vector[pivot] = -reduced[row][free]
        basis.append(vector)
    return basis


def _exterior_system(L: LinearOperator, k: int) -> tuple[list[tuple[int, ...]], list[list[Expr]]]:
    """The system ``w' = A w`` of the ``k x k`` minors ``w_I`` (``I`` a set
    of rows, rows ``0, ..., n - 1`` the derivatives) of the Wronskian
    matrix of ``k`` solutions of the monic ``L``: the derivative of a
    minor replaces one row ``i`` by ``i + 1``, and the row ``n`` is
    ``-sum a_m y^(m)``."""
    n = L.order
    a = L.coefficients
    subsets = list(combinations(range(n), k))
    position = {I: p for p, I in enumerate(subsets)}
    system: list[list[Expr]] = [[_ZERO]*len(subsets) for _ in subsets]
    for p, I in enumerate(subsets):
        for t, i in enumerate(I):
            if i + 1 < n:
                if i + 1 in I:
                    continue
                J = I[:t] + (i + 1,) + I[t + 1:]
                system[p][position[J]] = system[p][position[J]] + 1
            else:
                base = I[:-1]
                for m in range(n):
                    if m in base or a[m] == _ZERO:
                        continue
                    J = tuple(sorted(base + (m,)))
                    sign = (-1)**sum(1 for b in base if b > m)
                    system[p][position[J]] = system[p][position[J]] - sign*a[m]
    return subsets, system


def _symmetric_system(L: LinearOperator, m: int) -> tuple[list[tuple[int, ...]], list[list[Expr]]]:
    """The system of the monomials of degree ``m`` in ``y, ..., y^(n-1)``
    for a solution ``y`` of the monic ``L``, ``y**m`` first."""
    n = L.order
    a = L.coefficients
    monomials: list[tuple[int, ...]] = [e for e in cartesian(range(m + 1), repeat=n) if sum(e) == m]
    monomials.sort(key=lambda e: tuple(-c for c in e))
    position = {e: p for p, e in enumerate(monomials)}
    system: list[list[Expr]] = [[_ZERO]*len(monomials) for _ in monomials]
    for p, e in enumerate(monomials):
        for i in range(n):
            if e[i] == 0:
                continue
            lowered = list(e)
            lowered[i] -= 1
            if i + 1 < n:
                raised = list(lowered)
                raised[i + 1] += 1
                q = position[tuple(raised)]
                system[p][q] = system[p][q] + e[i]
            else:
                for j in range(n):
                    if a[j] == _ZERO:
                        continue
                    raised = list(lowered)
                    raised[j] += 1
                    q = position[tuple(raised)]
                    system[p][q] = system[p][q] - e[i]*a[j]
    return monomials, system


def _scalar_operator(system: list[list[Expr]], start: list[Expr], x: Symbol
                     ) -> tuple[LinearOperator, list[list[Expr]]]:
    """The monic operator of least order annihilating ``u = start . w``
    for the solutions of ``w' = A w``, with the rows ``v_j`` such that
    ``u^(j) = v_j . w`` below its order: ``v_{j+1} = v_j' + v_j A``."""
    size = len(start)
    rows: list[list[Expr]] = [[_normal(c) for c in start]]
    while True:
        v = rows[-1]
        following = [_normal(v[j].diff(x) + _sum([v[i]*system[i][j] for i in range(size)
                                                   if v[i] != _ZERO and system[i][j] != _ZERO]))
                     for j in range(size)]
        # sum_j c_j v_j + v_N = 0
        columns = rows + [following]
        relations = _nullspace([[columns[j][i] for j in range(len(columns))] for i in range(size)], len(columns))
        for relation in relations:
            if relation[-1] != _ZERO:
                coefficients = [_normal(c/relation[-1]) for c in relation]
                return LinearOperator(coefficients, x), rows
        rows.append(following)


def gcrd(A: LinearOperator, B: LinearOperator) -> LinearOperator:
    """The monic greatest common right divisor of ``A`` and ``B``, by the
    Euclidean algorithm on the right; its solutions are the common
    solutions of ``A`` and ``B``.

    Examples
    ========

    >>> from sympy_extras.core import Integer, Symbol
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, gcrd
    >>> x = Symbol('x')
    >>> D = LinearOperator([Integer(0), Integer(1)], x)
    >>> R = D - LinearOperator([1/x], x)
    >>> gcrd((D*D + LinearOperator([Integer(1)], x))*R, (D - LinearOperator([x], x))*R)
    LinearOperator([-1/x, 1], x)
    """
    while not B.is_zero:
        A, B = B, A.right_remainder(B)
    return A.monic() if not A.is_zero else A


def lclm(A: LinearOperator, B: LinearOperator) -> LinearOperator:
    """The monic least common left multiple of ``A`` and ``B``: its
    solutions are the sums of solutions of ``A`` and of ``B``. The powers
    ``D**i`` are reduced modulo ``A`` and ``B`` until the pairs of
    remainders become linearly dependent.

    Examples
    ========

    >>> from sympy_extras.core import Integer, Symbol
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.polys.polytools import cancel
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, lclm
    >>> x = Symbol('x')
    >>> M = lclm(LinearOperator([Integer(-1), Integer(1)], x), LinearOperator([Integer(-1), x], x))
    >>> M.order, cancel(M(exp(x))), M(x)
    (2, 0, 0)
    """
    x = A.x
    A, B = A.monic(), B.monic()
    remainders: list[tuple[LinearOperator, LinearOperator]] = []
    power = LinearOperator([_ONE], x)
    D = LinearOperator([_ZERO, _ONE], x)
    a, b = A.order, B.order
    while True:
        remainders.append((power.right_remainder(A), power.right_remainder(B)))
        vectors = [list(r.coefficients) + [_ZERO]*(a - len(r.coefficients)) + list(s.coefficients)
                   + [_ZERO]*(b - len(s.coefficients)) for r, s in remainders]
        size = a + b
        relations = _nullspace([[v[i] for v in vectors] for i in range(size)], len(vectors))
        for relation in relations:
            if relation[-1] != _ZERO:
                return LinearOperator([_normal(c/relation[-1]) for c in relation], x)
        power = D*power


def _falling(d: Expr, i: int) -> Expr:
    """The falling factorial ``d (d - 1) ... (d - i + 1)``."""
    return _product([d - j for j in range(i)])


def _indicial_at_infinity(L: LinearOperator) -> UPoly:
    """The polynomial in ``d`` whose nonnegative integer roots bound the
    degree of the polynomial solutions."""
    x = L.x
    d = Dummy('d')
    data: list[tuple[int, int, Expr]] = []
    for i, c in enumerate(L.coefficients):
        if c == _ZERO:
            continue
        p = UPoly.of(c, x)
        data.append((i, p.degree() - i, p.LC()))
    top = max(shift for _, shift, _ in data)
    expression = _sum([lc*_falling(d, i) for i, shift, lc in data if shift == top])
    return UPoly.of(expression, d)


def _nonnegative_integer_roots(p: UPoly) -> list[int]:
    found = _root_multiplicities(p)
    return sorted(int(r) for r in found if isinstance(r, Integer) and int(r) >= 0)


def _negative_integer_roots(p: UPoly) -> list[int]:
    found = _root_multiplicities(p)
    return sorted(int(r) for r in found if isinstance(r, Integer) and int(r) < 0)


def polynomial_solutions(L: LinearOperator) -> list[Expr]:
    """A basis of the polynomial solutions of ``L(y) = 0``.

    Examples
    ========

    >>> from sympy_extras.core import Function, Integer, Symbol
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, polynomial_solutions
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    >>> polynomial_solutions(L)
    [x**2 + 2*x + 2]
    >>> polynomial_solutions(LinearOperator([Integer(-2), Integer(0), Integer(1)], x))
    []
    """
    L = _polynomial_operator(L)
    x = L.x
    indicial = _indicial_at_infinity(L)
    if indicial.is_zero:
        return []
    degrees = _nonnegative_integer_roots(indicial)
    if not degrees:
        return []
    bound = degrees[-1]
    unknowns: list[Symbol] = [Dummy('c%d' % i) for i in range(bound + 1)]
    ansatz = _sum([c*x**i for i, c in enumerate(unknowns)])
    image = expand(L(ansatz))
    try:
        rows = UPoly.of(image, x).all_coeffs()
    except ValueError:
        return []
    matrix = [[row.coeff(u) for u in unknowns] for row in rows]
    solutions: list[Expr] = []
    for vector in _nullspace(matrix, len(unknowns)):
        solutions.append(_sum([v*x**i for i, v in enumerate(vector)]))
    return solutions


def _order_at(p: UPoly, q: UPoly) -> tuple[int, UPoly]:
    """The order of ``p`` at the roots of ``q`` and the cofactor
    ``p/q**order``."""
    order = 0
    while not p.is_zero:
        quotient, remainder = p.div(q)
        if not remainder.is_zero:
            break
        p = quotient
        order += 1
    return order, p


def _indicial_at(L: LinearOperator, q: UPoly) -> UPoly:
    """The indicial polynomial (in ``m``) at the roots ``c`` of the
    irreducible ``q``, as a polynomial with rational coefficients whose
    integer roots contain those of the true indicial polynomial (the norm
    from ``Q(c)`` when ``q`` is not linear)."""
    x = L.x
    m = Dummy('m')
    c = Dummy('c')
    q_c = UPoly.of(_replaced(q.as_expr(), {x: c}), c)
    derivative = q.diff().as_expr()
    orders: list[tuple[int, int, Expr]] = []
    for i, coefficient in enumerate(L.coefficients):
        if coefficient == _ZERO:
            continue
        order, p = _order_at(UPoly.of(coefficient, x), q)
        # the leading coefficient of a_i = q**order p at c in powers of
        # x - c: q = (x - c) q~ with q~(c) = q'(c), so p(c) q'(c)**order,
        # reduced modulo q(c) (the factor q'(c)**order was missing, and
        # the exponents at the roots of a factor like 2*x + 1 came out
        # wrong)
        cofactor = p.as_expr()*derivative**order
        if q.degree() > 1:
            value = UPoly.of(_replaced(cofactor, {x: c}), c).rem(q_c).as_expr()
        else:
            coefficients = q.all_coeffs()
            value = _normal(_replaced(cofactor, {x: -coefficients[1]/coefficients[0]}))
        orders.append((i, order - i, value))
    lowest = min(shift for _, shift, _ in orders)
    expression = _sum([value*_falling(m, i) for i, shift, value in orders if shift == lowest])
    if q.degree() > 1:
        expression = _resultant(expression, q_c.as_expr(), c)
    return UPoly.of(expression, m)


def _places(leading: Expr, x: Symbol) -> list[tuple[UPoly, int]]:
    """The irreducible factors of positive degree of the leading
    coefficient, with their multiplicities."""
    result: list[tuple[UPoly, int]] = []
    for f, multiplicity in _factor_list(leading, x):
        q = UPoly.of(f, x)
        if q.degree() > 0:
            result.append((q, multiplicity))
    return result


def _denominator_bound(L: LinearOperator) -> Expr:
    x = L.x
    result: Expr = _ONE
    for q, _ in _places(L.coefficients[-1], x):
        indicial = _indicial_at(L, q)
        if indicial.is_zero:
            continue
        negative = _negative_integer_roots(indicial)
        if negative:
            result = result*q.as_expr()**(-negative[0])
    return result


def rational_solutions(L: LinearOperator) -> list[Expr]:
    """A basis of the rational solutions of ``L(y) = 0``.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, rational_solutions
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    >>> rational_solutions(L)
    [x**(-2), 1/x]
    """
    L = _polynomial_operator(L)
    Q = _denominator_bound(L)
    if Q == _ONE:
        return polynomial_solutions(L)
    M = L.transformed(1/Q)
    return [_normal(P/Q) for P in polynomial_solutions(M)]


# ---------------------------------------------------------------------------
# hyperexponential solutions

def _is_fuchsian(L: LinearOperator) -> bool:
    """Whether every finite singular point is regular singular:
    ``ord_c(a_i) >= ord_c(a_n) - (n - i)``."""
    x = L.x
    n = L.order
    for q, multiplicity in _places(L.coefficients[-1], x):
        for i, c in enumerate(L.coefficients[:-1]):
            if c == _ZERO:
                continue
            order, _ = _order_at(UPoly.of(c, x), q)
            if order < multiplicity - (n - i):
                return False
    return True


def _exponents(L: LinearOperator, q: UPoly) -> list[Rational]:
    """The rational local exponents at the roots of ``q``."""
    return _exponents_status(L, q)[0]


def _exponents_status(L: LinearOperator, q: UPoly) -> tuple[list[Rational], bool]:
    """The rational local exponents at the roots of ``q``, and whether
    they are all the exponents there: every root of the indicial
    polynomial found and rational, and all of them congruent modulo the
    integers when ``q`` is not linear (conjugate points could otherwise
    take exponents of different classes; integer differences are taken
    by the polynomial of the ansatz)."""
    indicial = _indicial_at(L, q)
    if indicial.is_zero:
        return [], False
    found = _root_multiplicities(indicial)
    rational = sorted(set(r for r in found if isinstance(r, Rational)), key=lambda r: (r.p, r.q))
    complete = sum(found.values()) == indicial.degree() and len(rational) == len(found)
    if q.degree() > 1 and any(not _integer_difference(e, rational[0]) for e in rational):
        complete = False
    return rational, complete


def _integer_difference(a: Rational, b: Rational) -> bool:
    return Fraction(a.p, a.q) - Fraction(b.p, b.q) == int(Fraction(a.p, a.q) - Fraction(b.p, b.q))


def _newton_polygon_parts(L: LinearOperator) -> tuple[list[Expr], bool]:
    """Candidate leading terms ``c x**s`` (``s`` a nonnegative integer)
    of ``rho = y'/y`` at infinity, from the edges of the Newton polygon
    of the points ``(i, deg a_i)`` with slope ``-s``: on such an edge the
    terms ``a_i rho**i`` balance, ``sum lc(a_i) c**i = 0``; and whether
    every root of those polynomials was found."""
    x = L.x
    points: list[tuple[int, int, Expr]] = []
    for i, c in enumerate(L.coefficients):
        if c != _ZERO:
            p = UPoly.of(c, x)
            points.append((i, p.degree(), p.LC()))
    if len(points) < 2:
        return [], True
    # upper convex hull from the left
    hull: list[tuple[int, int, Expr]] = []
    for point in points:
        while len(hull) >= 2:
            (i1, d1, _), (i2, d2, _) = hull[-2], hull[-1]
            i3, d3 = point[0], point[1]
            # keep the hull concave: drop the middle point when it lies below the chord
            if (d2 - d1)*(i3 - i1) <= (d3 - d1)*(i2 - i1):
                hull.pop()
            else:
                break
        hull.append(point)
    candidates: list[Expr] = []
    complete = True
    t = Dummy('t')
    for (i1, d1, _), (i2, d2, _) in zip(hull, hull[1:]):
        slope = Fraction(d2 - d1, i2 - i1)
        s = -slope
        if s < 0 or s.denominator != 1:
            continue
        on_edge = [(i, lc) for i, d, lc in points if d - d1 == slope*(i - i1)]
        characteristic = UPoly.of(_sum([lc*t**i for i, lc in on_edge]), t)
        found = _root_multiplicities(characteristic)
        if sum(found.values()) != characteristic.degree():
            complete = False
        for root in found:
            if root != _ZERO and not root.has(x):
                candidates.append(root*x**int(s))
            elif root != _ZERO:
                complete = False
    return candidates, complete


def _polynomial_integral(p: Expr, x: Symbol) -> Expr:
    """The antiderivative of a polynomial in ``x`` without constant."""
    terms = UPoly.of(p, x).terms_
    return _sum([c*x**(k + 1)/(k + 1) for k, c in sorted(terms.items(), reverse=True)])


def _exponential_parts(L: LinearOperator, depth: int = 0, below: Optional[int] = None) -> tuple[list[Expr], bool]:
    """The polynomial parts ``P`` of ``rho`` at infinity (solutions
    ``exp(Integral(P)) * ...``), found term by term from the Newton
    polygon, each term of lower degree than the one before (``below``);
    ``0`` is always a candidate (no exponential part). The flag says
    whether the candidates are all the possible parts."""
    x = L.x
    parts: list[Expr] = [_ZERO]
    leading_terms, complete = _newton_polygon_parts(L)
    if below is not None:
        leading_terms = [t for t in leading_terms if UPoly.of(t, x).degree() < below]
    if depth > 4:
        return parts, complete and not leading_terms
    for leading in leading_terms:
        exponent = _polynomial_integral(leading, x)
        M = L.transformed(exp(exponent))
        lower_parts, lower_complete = _exponential_parts(M, depth + 1, UPoly.of(leading, x).degree())
        complete = complete and lower_complete
        for lower in lower_parts:
            candidate = leading + lower
            if candidate not in parts:
                parts.append(candidate)
    return parts, complete


def _polynomial_operator(L: LinearOperator) -> LinearOperator:
    """``L`` with polynomial coefficients (the solvers read the degrees
    of the coefficients)."""
    if all(c.is_polynomial(L.x) for c in L.coefficients):
        return L
    return L.primitive()


def hyperexponential_solutions(L: LinearOperator, max_combinations: int = 200) -> list[Expr]:
    """Solutions ``exp(Integral(P) + F) * prod (x - c)**e_c * Q(x)`` with
    ``P`` a polynomial (the exponential part at infinity, from the Newton
    polygon), ``F`` a rational function (the exponential parts at the
    irregular finite singular points, from the local Newton polygons:
    :mod:`sympy_extras.solvers.irregular`), rational exponents at the
    finite singular points and a polynomial ``Q``. Solutions with
    distinct exponent combinations are independent.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, hyperexponential_solutions
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(4*x**2*y.diff(x, 2) + 4*x*y.diff(x) - y, y)
    >>> hyperexponential_solutions(L)
    [1/sqrt(x), sqrt(x)]
    >>> hyperexponential_solutions(LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y))
    [x**2 + 2*x + 2, exp(x)]
    >>> hyperexponential_solutions(LinearOperator.from_equation(x**4*y.diff(x, 2) + 2*x**3*y.diff(x) - y, y))
    [exp(1/x), exp(-1/x)]
    """
    return hyperexponential_search(L, max_combinations)[0]


def hyperexponential_search(L: LinearOperator, max_combinations: int = 200) -> tuple[list[Expr], bool]:
    """The hyperexponential solutions of :func:`hyperexponential_solutions`
    and whether the search was exhaustive: ``True`` when every
    hyperexponential solution (over the algebraic closure of the
    constants) is a linear combination of the solutions returned. It is
    not when a local exponent is not rational or not found, when
    conjugate singular points could take different exponents, when an
    exponential part at a finite point has a leading coefficient outside
    the field of the point, or when an exponential part has more terms
    than the search follows.

    Examples
    ========

    >>> from sympy_extras.core import Integer, Symbol
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, hyperexponential_search
    >>> x = Symbol('x')
    >>> hyperexponential_search(LinearOperator([-x, Integer(0), Integer(1)], x))
    ([], True)
    >>> hyperexponential_search(LinearOperator([Integer(-1), Integer(0), x**2], x))
    ([], False)

    The second equation has the solutions ``x**((1 +- sqrt(5))/2)``,
    whose exponents are not rational.
    """
    from .irregular import finite_exponential_parts
    x = L.x
    L = _polynomial_operator(L)
    solutions: list[Expr] = []
    parts, complete = _exponential_parts(L)
    # the exponential parts at the irregular finite singular points do
    # not depend on the one at infinity
    finite_parts, finite_complete = finite_exponential_parts(L, max_combinations)
    complete = complete and finite_complete
    for part in parts:
        for finite in finite_parts:
            if part == _ZERO and finite == _ZERO:
                M = L
                factor_part: Expr = _ONE
            else:
                factor_part = exp(_polynomial_integral(part, x) + finite)
                M = _polynomial_operator(L.transformed(factor_part))
            found, found_complete = _hyperexponential_regular(M, max_combinations)
            complete = complete and found_complete
            for s in found:
                candidate = factor_part*s
                if candidate not in solutions:
                    solutions.append(candidate)
    return _independent(solutions, x), complete


def _hyperexponential_regular(L: LinearOperator, max_combinations: int) -> tuple[list[Expr], bool]:
    """The ansatz with the local exponents at the finite singular points:
    the roots of the indicial polynomials, which at an irregular point
    are the exponents of the solutions without exponential part there
    (the balance of the terms of lowest order when ``rho`` has a pole of
    order at most one; the solutions with an exponential part are reached
    through :mod:`sympy_extras.solvers.irregular`). One exponent per
    class modulo the integers is enough at each point (the smallest: the
    polynomial ``Q`` takes the integer shifts), and a combination is kept
    only when the exponents at infinity allow a polynomial ``Q``."""
    x = L.x
    places: list[tuple[Expr, int, list[Rational]]] = []
    complete = True
    for q, _ in _places(L.coefficients[-1], x):
        # a common exponent for conjugate roots: the factor itself
        exponents, exponents_complete = _exponents_status(L, q)
        complete = complete and exponents_complete
        if not exponents:
            return [], complete
        representatives: list[Rational] = []
        for e in exponents:
            if not any(_integer_difference(e, r) for r in representatives):
                representatives.append(e)
        places.append((q.as_expr(), q.degree(), representatives))
    at_infinity = _indicial_at_infinity(L)
    d = at_infinity.x
    t = Dummy('t')
    solutions: list[Expr] = []
    combinations_ = list(cartesian(*[e for _, _, e in places]))
    if len(combinations_) > max_combinations:
        combinations_ = combinations_[:max_combinations]
        complete = False
    for combination in combinations_:
        degree = _sum([e*k for (_, k, _), e in zip(places, combination)])
        # y ~ x**(degree + deg Q) at infinity
        if not at_infinity.is_zero and not _nonnegative_integer_roots(
                UPoly.of(_replaced(at_infinity.as_expr(), {d: degree + t}), t)):
            continue
        factor = _product([f**e for (f, _, _), e in zip(places, combination)])
        M = L.transformed(factor)
        for P in polynomial_solutions(M):
            solutions.append(factor*P)
    return _independent(solutions, x), complete


# ---------------------------------------------------------------------------
# reduction of order

def reduce_order_linear(L: LinearOperator, y1: Expr) -> LinearOperator:
    """The operator ``M`` of order ``n - 1`` with ``L(y1 * w) = y1 * M(w')``
    for a solution ``y1`` of ``L``.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, reduce_order_linear
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(y.diff(x, 2) - 3*y.diff(x) + 2*y, y)
    >>> reduce_order_linear(L, exp(x))
    LinearOperator([-1, 1], x)
    """
    x = L.x
    T = L.transformed(y1)
    if _normal(T.coefficients[0]) != _ZERO:
        raise ValueError("%s is not a solution" % (y1,))
    return LinearOperator(T.coefficients[1:], x)


def _independent(solutions: list[Expr], x: Symbol) -> list[Expr]:
    """Drop the solutions which are constant multiples of earlier ones."""
    kept: list[Expr] = []
    for s in solutions:
        if s == _ZERO:
            continue
        if any(_proportional(s, k, x) for k in kept):
            continue
        kept.append(s)
    return kept


def _proportional(s: Expr, k: Expr, x: Symbol) -> bool:
    """Whether ``s/k`` is constant; a ratio with unevaluated integrals
    is taken as not constant (deciding it would call ``simplify`` on the
    integrals, which may not return)."""
    ratio = _normal(s/k)
    if not ratio.has(x):
        return True
    if has_integral(ratio):
        return False
    try:
        # is_constant substitutes values such as 0 for x; SymPy's
        # evalf of meijerg(..., zoo) raises AttributeError
        # (sympy-extras#25), and a ratio it cannot decide is taken as not
        # constant
        return bool(attempt(lambda: ratio.is_constant(x), settings.timeout, native=True))
    except AttributeError:
        return False


def dsolve_linear(equation: Basic, f: AppliedUndef, use_kovacic: bool = True,
                  use_dsolve: bool = True, special_values: bool = True) -> list[Expr]:
    """Independent solutions of a homogeneous linear equation with
    rational coefficients: rational and hyperexponential solutions,
    Kovacic's algorithm for the second order, and reduction of order by
    the solutions found (the reduced equations handed to the same
    solvers and to SymPy's ``dsolve``). The list is empty when nothing
    is found and shorter than the order when the solution space is not
    exhausted.

    With ``special_values``, for an equation with parameters, each
    solution becomes a ``Piecewise`` with a case, first, for every
    isolated value of the parameters at which the solutions found are
    undefined or dependent (their Wronskian vanishes: two roots of the
    indicial or characteristic polynomial coincide), the equation solved
    again there (:mod:`sympy_extras.solvers._special_values`); the
    ``i``-th solution of the list at the point is the ``i``-th solution of
    the equation there.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's integrate and dsolve (the *_of_core functions)
    >>> from sympy_extras.core import Function, Symbol, symbols
    >>> from sympy_extras.solvers.linear_ode import dsolve_linear
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> dsolve_linear(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    [x**2 + 2*x + 2, exp(x)]
    >>> dsolve_linear(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    [x**(-2), 1/x]
    >>> a, b = symbols('a b')
    >>> dsolve_linear(y.diff(x, 2) - (a + b)*y.diff(x) + a*b*y, y)
    [exp(b*x), Piecewise((x*exp(b*x), Eq(a, b)), (exp(a*x), True))]
    """
    L = LinearOperator.from_equation(equation, f)
    found = _solve_operator(L, f, use_kovacic, use_dsolve)
    if not special_values:
        return found
    from ._special_values import as_zero, basis_cases

    def solve_at(at: Expr) -> Optional[list[Expr]]:
        return dsolve_linear(at, f, use_kovacic, use_dsolve)

    return basis_cases(found, as_zero(equation), f, solve_at)


def _equation(L: LinearOperator) -> Expr:
    """The equation ``L(y) = 0`` in the function ``y`` of ``x``."""
    x = L.x
    y = Function('y')(x)
    return _sum([c*y.diff(x, i) if i else c*y for i, c in enumerate(L.coefficients)])


def _solve_operator(L: LinearOperator, f: AppliedUndef, use_kovacic: bool, use_dsolve: bool,
                    factorize: bool = True) -> list[Expr]:
    x = L.x
    n = L.order
    if n == 0:
        return []
    if n == 1:
        # SymPy's integrate, imported here only: the other orders reach it
        # through the reduction of order alone
        from .kovacic_sympy import integral_of_core, integrate_of_core
        # y' = -a_0/a_1 y
        rho = _normal(-L.coefficients[0]/L.coefficients[1])
        integral = integrate_of_core(rho, x, 'piecewise', False)
        if integral is None:
            return [exp(integral_of_core(rho, x))]
        return [exp(integral)]
    found: list[Expr] = []
    found.extend(rational_solutions(L))
    if len(found) < n:
        found.extend(hyperexponential_solutions(L))
    found = _independent(found, x)
    if len(found) < n and n >= 3 and factorize:
        # the right factor's solutions, then variation of parameters for
        # the solutions of the left factor
        by_factors = _solve_by_factorization(L, f, use_kovacic, use_dsolve)
        if len(by_factors) > len(found):
            found = by_factors
    if len(found) == 1 and n == 2:
        # the reduction of order by a known solution gives the second one
        # by quadratures; when they are evaluated Kovacic's algorithm
        # (which can take the whole time limit) is not needed
        reduced_found = _by_reduction(L, f, found, use_kovacic, use_dsolve, factorize)
        if reduced_found is not None and len(reduced_found) == 2 and not has_integral(reduced_found[1]):
            return reduced_found
    if len(found) < n and n == 2 and use_kovacic:
        from .kovacic import dsolve_kovacic
        equation = _equation(L)
        y = Function('y')(x)
        liouvillian = attempt(lambda: dsolve_kovacic(equation, y, special_values=False), settings.timeout,
                              native=True)
        if liouvillian:
            found = _independent(found + liouvillian, x)
    if len(found) < n and n == 2:
        from .special import special_solutions
        equation = _equation(L)
        y = Function('y')(x)
        special = attempt(lambda: special_solutions(equation, y), settings.timeout, native=True)
        if special:
            found = _independent(found + special, x)
    reduced_by_constant = False
    if found and len(found) < n:
        reduced_by_constant = not found[0].has(x)
        reduced_found = _by_reduction(L, f, found, use_kovacic, use_dsolve, factorize)
        if reduced_found is None:
            return found
        found = reduced_found
    # an equation without y reduced by the constant solution is the
    # equation in y' which SymPy's dsolve would reduce it to (its hint
    # nth_order_reducible), and which the reduction has already handed to
    # dsolve: asking again is wasted
    if len(found) < n and use_dsolve and not (reduced_by_constant and L.coefficients[0] == _ZERO):
        from .linear_ode_sympy import dsolve_of_core
        parts = dsolve_of_core(_equation(L), Function('y')(x))
        for part in parts or []:
            found = _independent(found + [part], x)
    return found[:n]


def _by_reduction(L: LinearOperator, f: AppliedUndef, found: list[Expr], use_kovacic: bool, use_dsolve: bool,
                  factorize: bool) -> Optional[list[Expr]]:
    """``found`` extended by the solutions of the operator reduced by the
    first solution (``y1 * Integral(v)`` for its solutions ``v``), ``None``
    when the first one is not a solution."""
    from .kovacic_sympy import integral_of_core, integrate_of_core
    x = L.x
    y1 = found[0]
    try:
        M = reduce_order_linear(L, y1)
    except ValueError:
        return None
    reduced = _solve_operator(M, f, use_kovacic, use_dsolve, factorize)
    for v in reduced:
        w = integrate_of_core(v, x, 'none', True)
        if w is None:
            # the quadrature is left unevaluated
            w = integral_of_core(v, x)
        product = y1*w
        found = _independent(found + [_normal(product) if product.is_rational_function(x) else product], x)
    return found


def _solve_by_factorization(L: LinearOperator, f: AppliedUndef, use_kovacic: bool, use_dsolve: bool) -> list[Expr]:
    """Independent solutions of ``L`` through a factorisation into at
    least two factors (none when ``L`` does not factor)."""
    from .factorization import factor_operator, solve_factored
    limit = None if settings.timeout is None else 4*settings.timeout
    factorization = attempt(lambda: factor_operator(L), limit, native=True)
    if factorization is None or len(factorization.factors) < 2:
        return []

    def solver(M: LinearOperator) -> list[Expr]:
        return _solve_operator(M.primitive(), f, use_kovacic, use_dsolve, False)

    return _independent(solve_factored(factorization.factors, solver), L.x)

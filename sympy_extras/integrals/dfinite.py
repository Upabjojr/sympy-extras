"""Chyzak's algorithm: creative telescoping for definite integrals of
D-finite integrands, the general case of the Almkvist–Zeilberger
algorithm of :mod:`sympy_extras.integrals.telescoping`.

A function `F(x, t)` is *D-finite* in `x` and in `t` when it satisfies
linear differential equations with polynomial coefficients in each
variable,

.. math::

    L_x F = \\sum_{i=0}^{r} c_i(x, t)\\, \\partial_x^i F = 0, \\qquad
    L_t F = \\sum_{j=0}^{s} d_j(x, t)\\, \\partial_t^j F = 0:

exponentials, Bessel functions, orthogonal polynomials, hypergeometric
functions and their products and compositions with linear arguments.
Every mixed derivative `\\partial_x^i \\partial_t^j F` is then a combination,
with rational coefficients, of finitely many functions. Two ways to
that closure are implemented: the *normal forms* modulo the two
annihilators (`i \\ge r` reduced with `L_x`, `j \\ge s` with `L_t`,
Leibniz's rule taking the derivatives of the coefficients; the
annihilators come from :func:`sympy.holonomic.expr_to_holonomic` with
the other variable in the coefficient field, a scaled argument which
it refuses, `J_0(t x)`, through the substitution `x = y/t` and back),
which is exact when the ideal of the two operators is rectangular; and,
for the algorithm itself, the closure of a *product of D-finite
factors* `f_i(\\text{argument}_i)` with polynomial arguments: the basis
is the set of products `\\prod_i f_i^{(m_i)}(\\text{argument}_i)`,
`m_i < n_i`, and the chain rule with the equation of each factor gives
the derivatives, so that the rank is right even when a factor depends on
`x t` (where the two annihilators overcount).

Chyzak's algorithm [Chyzak]_ looks for a *telescoper*, an operator
`P = \\sum_{k \\le J} a_k(t)\\, \\partial_t^k` in `t` alone, and a *certificate*
`G = \\sum_b g_b(x, t)\\, b` over the basis `b` of the closure, with
rational `g_b`, such that

.. math::

    P F = \\partial_x G.

Both sides are reduced to the normal forms, and the coefficients of the
basis derivatives are compared: with the ansatz of Koutschan
[Koutschan]_, `g_b = p_b(x)/D(x)^e` with `D` the product of the
irreducible factors of the denominators met in the derivatives of the
basis, `e = 1, 2`, and `p_b` polynomials of degree at most `N`, the
comparison is a linear system over `\\mathbb{Q}(t)` for the coefficients
of the `p_b` and the `a_k`; `J` and `N` are increased
until it has a solution with some `a_k \\ne 0` (a solution with all
`a_k = 0` is a certificate of nothing). Integrating over `x`, the
definite integral `I(t) = \\int_a^b F\\, dx` satisfies

.. math::

    \\sum_k a_k(t)\\, I^{(k)}(t) = \\bigl[G\\bigr]_{x=a}^{x=b},

which ``dsolve`` solves, the constants fixed by the values of the
integral at a point (:func:`~sympy_extras.integrals.definite_integral`),
as in :func:`~sympy_extras.integrals.telescoping.holonomic_integral`.
For a hyperexponential `F` (`r = s = 1`) the certificate is the single
rational function of the Almkvist–Zeilberger algorithm.

Examples
========

>>> from sympy import symbols, exp, besselj, oo, sqrt
>>> from sympy_extras.integrals.dfinite import chyzak, dfinite_ode, dfinite_integral
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> chyzak(exp(-t*x)*besselj(0, x), x, t)
DFiniteTelescoper([t, t**2 + 1], {exp(-t*x)*besselj(0, x): t*x, -exp(-t*x)*besselj(1, x): x})
>>> dfinite_ode(exp(-t*x)*besselj(0, x), x, 0, oo, t)
Eq(t*I(t) + (t**2 + 1)*Derivative(I(t), t), 0)
>>> dfinite_integral(exp(-t*x)*besselj(0, x), x, 0, oo, t)
ConditionalValue(1/sqrt(t**2 + 1))

References
==========

.. [Chyzak] F. Chyzak, An extension of Zeilberger's fast algorithm to
   general holonomic functions, Discrete Mathematics 217 (2000),
   115–134.
.. [Koutschan] C. Koutschan, A fast approach to creative telescoping,
   Mathematics in Computer Science 4 (2010), 259–266.
.. [AZ] G. Almkvist, D. Zeilberger, The method of differentiating under
   the integral sign, Journal of Symbolic Computation 10 (1990),
   571–591 (the hyperexponential case).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional, Sequence

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef, diff, expand_func
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.numbers import Integer, nan, oo, zoo
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.holonomic.holonomic import expr_to_holonomic
from sympy.matrices.dense import Matrix
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import PolynomialError, CoercionFailed, GeneratorsError
from sympy.polys.polytools import Poly, cancel, factor_list
from sympy.polys.rationaltools import together
from sympy.simplify.simplify import simplify
from sympy.solvers.ode import dsolve
from sympy.solvers.solveset import linsolve
from sympy.sets.sets import FiniteSet

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_expr, free_symbols
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.settings import settings
from .conditions import ConditionalValue
from .telescoping import _boundary, _constants, _initial_values, _order

__all__ = ['Annihilator', 'DFiniteTelescoper', 'Factor', 'annihilators', 'normal_form', 'closure', 'chyzak',
           'dfinite_ode', 'dfinite_integral']

#: a mixed derivative ``d^i/dx^i d^j/dt^j F`` by its orders
Derivative_ = tuple[int, int]
#: a combination of the basis derivatives with rational coefficients
NormalForm = dict[Derivative_, Expr]


class Annihilator:
    """A linear differential operator ``sum_i c_i(x, t) d^i/dv^i`` in the
    variable ``v`` with coefficients polynomial in ``v``, ``c_r != 0``.

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.dfinite import annihilators
    >>> x, t = symbols('x t')
    >>> L_x, L_t = annihilators(exp(-t*x**2), x, t)
    >>> L_x
    Annihilator(x, [2*t*x, 1])
    >>> L_x.order
    1
    """

    def __init__(self, variable: Symbol, coefficients: Sequence[Expr]) -> None:
        self.variable = variable
        self.coefficients = [as_expr(c) for c in coefficients]
        while self.coefficients and self.coefficients[-1] == 0:
            self.coefficients.pop()
        if len(self.coefficients) < 2:
            raise ValueError("an annihilator has order at least one")

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @property
    def leading(self) -> Expr:
        return self.coefficients[-1]

    def reduction(self) -> list[Expr]:
        """The coefficients ``e_i`` with ``d^r F = sum_{i < r} e_i d^i F``."""
        return [as_expr(cancel(-c / self.leading)) for c in self.coefficients[:-1]]

    def apply(self, f: Expr) -> Expr:
        """``sum_i c_i d^i f/dv^i``."""
        return as_expr(Add(*[c * diff(f, self.variable, i) for i, c in enumerate(self.coefficients)]))

    def __repr__(self) -> str:
        return "Annihilator(%s, %s)" % (self.variable, self.coefficients)


class DFiniteTelescoper:
    """The output of Chyzak's algorithm for ``F(x, t)``: the operator
    ``P = sum_k a_k(t) d^k/dt^k`` and the certificate ``G``, a combination
    of the basis of the closure with rational coefficients, with
    ``P F = dG/dx``.

    Attributes
    ==========

    coefficients : list of Expr
        ``a_0(t), ..., a_J(t)``, polynomials without common factors.
    certificate : dict
        The basis functions of the closure (products of derivatives of
        the D-finite factors) with their rational coefficients in ``G``.
    term, x, t
        The input.
    """

    def __init__(self, term: Expr, x: Symbol, t: Symbol, coefficients: Sequence[Expr],
                 certificate: dict[Expr, Expr]) -> None:
        self.term = term
        self.x = x
        self.t = t
        self.coefficients = [as_expr(c) for c in coefficients]
        self.certificate = dict(certificate)

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def operator(self, name: str = 'I') -> Expr:
        """``sum_k a_k(t) I^(k)(t)`` in the function ``I(t)``."""
        f = Function(name)(self.t)
        return as_expr(Add(*[a * diff(f, self.t, k) for k, a in enumerate(self.coefficients)]))

    def function(self) -> Expr:
        """The certificate ``G`` as an expression."""
        return as_expr(Add(*[g * b for b, g in self.certificate.items()]))

    def check(self) -> bool:
        """Verify ``P F = dG/dx`` by simplification."""
        lhs = Add(*[a * diff(self.term, self.t, k) for k, a in enumerate(self.coefficients)])
        difference = as_expr(lhs - diff(self.function(), self.x))
        difference = as_expr(simplify(difference.doit()))
        return difference == 0

    def __repr__(self) -> str:
        return "DFiniteTelescoper(%s, %s)" % (self.coefficients, self.certificate)


# ---------------------------------------------------------------------------
# Annihilators

def _holonomic(F: Expr, v: Symbol, parameter: Symbol) -> Optional[Annihilator]:
    """The annihilator of ``F`` in ``v`` from SymPy's ``expr_to_holonomic``
    with ``parameter`` in the coefficient field, ``None`` when it fails."""
    domain = QQ.frac_field(parameter) if F.has(parameter) else QQ
    try:
        h = attempt(lambda: expr_to_holonomic(F, v, domain=domain), settings.timeout)
    except (PolynomialError, CoercionFailed, GeneratorsError, AttributeError, KeyError, ZeroDivisionError,
            AssertionError):
        return None
    if h is None:
        return None
    operator = h.annihilator
    ring = operator.parent.base
    coefficients = [as_expr(ring.to_sympy(p)) for p in operator.listofpoly]
    if not coefficients or coefficients[-1] == 0:
        return None
    try:
        return Annihilator(v, coefficients)
    except ValueError:
        return None


def _scales(F: Expr, v: Symbol) -> list[Expr]:
    """The factors ``c`` of arguments ``c*v`` of the functions of ``F``,
    ``c`` free of ``v`` and not a number."""
    found: list[Expr] = []
    for node in F.atoms(Function):
        for argument in node.args:
            argument_ = as_expr(argument)
            coefficient, rest = argument_.as_independent(v, as_Add=False)
            if as_expr(rest) == v and not as_expr(coefficient).is_number and as_expr(coefficient) not in found:
                found.append(as_expr(coefficient))
    return found


def _annihilator(F: Expr, v: Symbol, parameter: Symbol) -> Optional[Annihilator]:
    """The annihilator in ``v``, with the substitution ``v = y/c`` and back
    when a scaled argument ``c*v`` stops ``expr_to_holonomic``."""
    found = _holonomic(F, v, parameter)
    if found is not None:
        return found
    for c in _scales(F, v):
        y = Dummy('y')
        scaled = _holonomic(as_expr(F.subs(v, y / c)), y, parameter)
        if scaled is None:
            continue
        # d/dy = (1/c) d/dv and p_i(y) = p_i(c v)
        coefficients = [as_expr(cancel(p.subs(y, c * v) * c**(-i))) for i, p in enumerate(scaled.coefficients)]
        try:
            return Annihilator(v, coefficients)
        except ValueError:
            continue
    return None


def annihilators(F: ExprLike, x: Symbol, t: Symbol) -> Optional[tuple[Annihilator, Annihilator]]:
    """The annihilators ``L_x`` and ``L_t`` of ``F`` (see the module
    documentation), or ``None`` when SymPy finds none.

    Examples
    ========

    >>> from sympy import symbols, exp, besselj
    >>> from sympy_extras.integrals.dfinite import annihilators
    >>> x, t = symbols('x t')
    >>> L_x, L_t = annihilators(exp(-x**2)*besselj(0, t*x), x, t)
    >>> L_x.order, L_t.order
    (2, 2)
    >>> annihilators(exp(exp(x))*t, x, t) is None
    True
    """
    F_ = as_expr(sympify(F))
    L_x = _annihilator(F_, x, t)
    if L_x is None:
        return None
    L_t = _annihilator(F_, t, x)
    if L_t is None:
        return None
    return L_x, L_t


# ---------------------------------------------------------------------------
# Normal forms modulo the two annihilators

def _add(total: NormalForm, key: Derivative_, value: Expr) -> None:
    total[key] = as_expr(total.get(key, S.Zero) + value)


def _leibniz(coefficient: Expr, i: int, m: int, base: Derivative_, x: Symbol, t: Symbol,
             reduce: Callable[[int, int], NormalForm]) -> NormalForm:
    """``d^i/dx^i d^m/dt^m (coefficient * d^base F)`` in normal form."""
    result: NormalForm = {}
    for a in range(i + 1):
        for b in range(m + 1):
            factor = as_expr(binomial(i, a) * binomial(m, b) * diff(coefficient, x, i - a, t, m - b)
                             if (i - a or m - b) else binomial(i, a) * binomial(m, b) * coefficient)
            if factor == 0:
                continue
            inner = reduce(base[0] + a, base[1] + b)
            for key, value in inner.items():
                _add(result, key, factor * value)
    return result


def normal_form(L_x: Annihilator, L_t: Annihilator, x: Symbol, t: Symbol) -> Callable[[int, int], NormalForm]:
    """The reduction ``(i, j) -> d^i/dx^i d^j/dt^j F`` as a combination of
    the basis derivatives ``i < r``, ``j < s``, memoised.

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.dfinite import annihilators, normal_form
    >>> x, t = symbols('x t')
    >>> L_x, L_t = annihilators(exp(-t*x**2), x, t)
    >>> reduce = normal_form(L_x, L_t, x, t)
    >>> reduce(2, 0)
    {(0, 0): 4*t**2*x**2 - 2*t}
    """
    r, s = L_x.order, L_t.order
    e_x = L_x.reduction()
    e_t = L_t.reduction()

    @lru_cache(maxsize=None)
    def reduce(i: int, j: int) -> NormalForm:
        if i < r and j < s:
            return {(i, j): S.One}
        result: NormalForm = {}
        if j >= s:
            # d^i_x d^j_t F = d^i_x d^(j-s)_t sum_k e_k d^k_t F
            for k, e in enumerate(e_t):
                if e == 0:
                    continue
                for key, value in _leibniz(e, i, j - s, (0, k), x, t, reduce).items():
                    _add(result, key, value)
        else:
            # i >= r: d^i_x d^j_t F = d^(i-r)_x d^j_t sum_k e_k d^k_x F
            for k, e in enumerate(e_x):
                if e == 0:
                    continue
                for key, value in _leibniz(e, i - r, j, (k, 0), x, t, reduce).items():
                    _add(result, key, value)
        return {key: as_expr(cancel(value)) for key, value in result.items() if cancel(value) != 0}

    return reduce


# ---------------------------------------------------------------------------
# The closure of the derivatives of a product of D-finite factors

class Factor:
    """A D-finite factor ``f(u)`` of the integrand at ``u = argument``, a
    polynomial in ``x`` and ``t``: its order ``n`` and the reduction
    ``f^(n)(u) = sum_{m < n} e_m(u) f^(m)(u)``.

    The basis of the closure of ``F`` under differentiation is the set
    of products of the ``f_i^(m_i)(argument_i)``, ``m_i < n_i``, with
    coefficients rational in ``x`` and ``t``: the derivative of a product
    is computed by the chain rule and the reductions.
    """

    def __init__(self, function: Expr, u: Symbol, argument: Expr, reduction: Sequence[Expr]) -> None:
        self.function = function
        self.u = u
        self.argument = argument
        self.reduction = [as_expr(e) for e in reduction]

    @property
    def order(self) -> int:
        return len(self.reduction)

    def derivative(self, m: int) -> Expr:
        """``f^(m)(argument)`` as an expression."""
        return as_expr(diff(self.function, self.u, m).subs(self.u, self.argument))

    def __repr__(self) -> str:
        return "Factor(%s, %s)" % (self.function.subs(self.u, self.argument), self.order)


#: a basis element: the orders of the derivatives of the factors
Basis = tuple[int, ...]
#: an element of the closure: basis element -> rational coefficient
Vector = dict[Basis, Expr]


def _dfinite_factor(f: Expr, x: Symbol, t: Symbol) -> Optional[Factor]:
    """``f`` as a D-finite function of a polynomial argument."""
    if not isinstance(f, Function) or not f.args:
        return None
    argument = as_expr(f.args[-1])
    if not argument.is_polynomial(x, t) or any(as_expr(a).has(x, t) for a in f.args[:-1]):
        return None
    u = Dummy('u')
    function = as_expr(f.func(*(list(f.args[:-1]) + [u])))
    parameters = sorted(free_symbols(function) - {u}, key=lambda s: s.name)
    domain = QQ.frac_field(*parameters) if parameters else QQ
    try:
        h = attempt(lambda: expr_to_holonomic(function, u, domain=domain), settings.timeout)
    except (PolynomialError, CoercionFailed, GeneratorsError, AttributeError, KeyError, ZeroDivisionError,
            AssertionError, NotImplementedError):
        return None
    if h is None:
        return None
    ring = h.annihilator.parent.base
    coefficients = [as_expr(ring.to_sympy(p)) for p in h.annihilator.listofpoly]
    if len(coefficients) < 2 or coefficients[-1] == 0:
        return None
    leading = coefficients[-1]
    return Factor(function, u, argument, [as_expr(cancel(-c / leading)) for c in coefficients[:-1]])


def closure(F: ExprLike, x: Symbol, t: Symbol) -> Optional[tuple[Expr, list[Factor]]]:
    """The integrand as a rational coefficient times D-finite factors
    ``f_i(argument_i)`` (the argument polynomial in ``x`` and ``t``), or
    ``None`` when a factor is not of this kind.

    Examples
    ========

    >>> from sympy import symbols, exp, besselj
    >>> from sympy_extras.integrals.dfinite import closure
    >>> x, t = symbols('x t')
    >>> closure(x*exp(-x**2)*besselj(0, t*x), x, t)
    (x, [Factor(besselj(0, t*x), 2), Factor(exp(-x**2), 1)])
    """
    F_ = as_expr(sympify(F))
    coefficient: Expr = S.One
    factors: list[Factor] = []
    parts = list(F_.args) if isinstance(F_, Mul) else [F_]
    for part in parts:
        e = as_expr(part)
        if e.is_rational_function(x, t):
            coefficient = coefficient * e
            continue
        count = 1
        if isinstance(e, Pow) and isinstance(e.exp, Integer) and e.exp > 0:
            count = int(e.exp)
            e = as_expr(e.base)
        factor = _dfinite_factor(e, x, t)
        if factor is None:
            expanded = as_expr(expand_func(e))
            if expanded != e:
                inner = closure(expanded, x, t)
                if inner is None:
                    return None
                coefficient = coefficient * inner[0]**count
                factors.extend(inner[1] * count)
                continue
            return None
        factors.extend([factor] * count)
    return coefficient, factors


def _derive(vector: Vector, var: Symbol, factors: Sequence[Factor]) -> Vector:
    """The derivative of an element of the closure with respect to
    ``var`` (``x`` or ``t``), in the basis."""
    result: Vector = {}
    for basis, coefficient in vector.items():
        derived = as_expr(diff(coefficient, var))
        if derived != 0:
            _add_basis(result, basis, derived)
        for k, factor in enumerate(factors):
            chain = as_expr(diff(factor.argument, var))
            if chain == 0:
                continue
            m = basis[k] + 1
            if m < factor.order:
                _add_basis(result, basis[:k] + (m,) + basis[k + 1:], coefficient * chain)
            else:
                # f^(n)(argument) reduced by the differential equation of f
                for m_, e in enumerate(factor.reduction):
                    if e == 0:
                        continue
                    _add_basis(result, basis[:k] + (m_,) + basis[k + 1:],
                               coefficient * chain * e.subs(factor.u, factor.argument))
    return {basis: as_expr(cancel(value)) for basis, value in result.items() if cancel(value) != 0}


def _add_basis(total: Vector, key: Basis, value: Expr) -> None:
    total[key] = as_expr(total.get(key, S.Zero) + value)


def _denominators(vectors: Sequence[Vector], x: Symbol) -> Expr:
    """The product of the irreducible factors in ``x`` of the
    denominators of the coefficients."""
    factors: list[Expr] = []
    for vector in vectors:
        for value in vector.values():
            denominator = as_expr(together(value).as_numer_denom()[1])
            if not denominator.has(x):
                continue
            try:
                _, parts = factor_list(denominator, x)
            except (PolynomialError, CoercionFailed):
                continue
            for part, _ in parts:
                part_ = as_expr(part)
                if part_.has(x) and part_ not in factors:
                    factors.append(part_)
    result: Expr = S.One
    for f in factors:
        result = result * f
    return result


def _nullspace(rows: list[list[Expr]], unknowns: int) -> list[list[Expr]]:
    """The vectors annihilated by the rows (over the rational functions
    of the parameters)."""
    if not rows:
        return [[S.One if k == n else S.Zero for k in range(unknowns)] for n in range(unknowns)]
    matrix = Matrix(rows)
    try:
        domain_matrix = DomainMatrix.from_Matrix(matrix).to_field()
        vectors = domain_matrix.nullspace().to_Matrix()
    except (CoercionFailed, GeneratorsError, NotImplementedError, ValueError):
        vectors = Matrix.hstack(*matrix.nullspace()).T if matrix.nullspace() else Matrix(0, unknowns, [])
    return [[as_expr(vectors[n, k]) for k in range(vectors.cols)] for n in range(vectors.rows)]


class _Closure:
    """The integrand in its closure: the vector of ``F`` and its
    derivatives in ``t``, and the derivatives in ``x`` of the basis."""

    def __init__(self, F: Expr, x: Symbol, t: Symbol, coefficient: Expr, factors: Sequence[Factor]) -> None:
        self.F = F
        self.x = x
        self.t = t
        self.factors = list(factors)
        self.basis: list[Basis] = [()]
        for factor in factors:
            self.basis = [b + (m,) for b in self.basis for m in range(factor.order)]
        zero: Basis = tuple(0 for _ in factors)
        self.vector: Vector = {zero: coefficient}
        self.t_derivatives: list[Vector] = [self.vector]
        self.x_derivatives: dict[Basis, Vector] = {b: _derive({b: S.One}, x, self.factors) for b in self.basis}

    def t_derivative(self, k: int) -> Vector:
        while len(self.t_derivatives) <= k:
            self.t_derivatives.append(_derive(self.t_derivatives[-1], self.t, self.factors))
        return self.t_derivatives[k]

    def element(self, basis: Basis) -> Expr:
        """The basis element as an expression."""
        result: Expr = S.One
        for factor, m in zip(self.factors, basis):
            result = result * factor.derivative(m)
        return result


def _telescope(closure_: _Closure, order: int, degree: int, power: int) -> Optional[DFiniteTelescoper]:
    x, t = closure_.x, closure_.t
    lhs = [closure_.t_derivative(k) for k in range(order + 1)]
    D = _denominators(lhs + list(closure_.x_derivatives.values()) + [closure_.vector], x)**power
    a = [Dummy('a%d' % k) for k in range(order + 1)]
    u: dict[tuple[Basis, int], Symbol] = {(b, m): Dummy('u%d' % m) for b in closure_.basis for m in range(degree + 1)}
    keys = sorted(u, key=lambda key: (closure_.basis.index(key[0]), key[1]))
    unknowns: list[Symbol] = list(a) + [u[key] for key in keys]
    # sum_k a_k d^k_t F - d_x (sum_b g_b b), collected on the basis
    total: Vector = {}
    for k, vector in enumerate(lhs):
        for key, value in vector.items():
            _add_basis(total, key, a[k] * value)
    for b in closure_.basis:
        g = as_expr(Add(*[u[(b, m)] * x**m for m in range(degree + 1)]) / D)
        _add_basis(total, b, -diff(g, x))
        for key, value in closure_.x_derivatives[b].items():
            _add_basis(total, key, -g * value)
    rows: list[list[Expr]] = []
    for value in total.values():
        numerator, _ = together(value).as_numer_denom()
        try:
            polynomial = Poly(numerator, x)
        except (PolynomialError, CoercionFailed, GeneratorsError):
            return None
        for coefficient in polynomial.coeffs():
            coefficient_ = as_expr(coefficient.expand())
            if coefficient_.subs({w: 0 for w in unknowns}) != 0:
                return None
            rows.append([as_expr(cancel(coefficient_.coeff(w))) for w in unknowns])
    for solution in _nullspace(rows, len(unknowns)):
        if all(v == 0 for v in solution[:order + 1]):
            continue
        coefficients, scale = _normalized(solution[:order + 1], t)
        certificate: dict[Expr, Expr] = {}
        for b in closure_.basis:
            g = Add(*[solution[len(a) + keys.index((b, m))] * x**m for m in range(degree + 1)]) / D
            g_ = as_expr(cancel(g * scale))
            if g_ != 0:
                certificate[closure_.element(b)] = g_
        return DFiniteTelescoper(closure_.F, x, t, coefficients, certificate)
    return None


def _normalized(values: Sequence[Expr], t: Symbol) -> tuple[list[Expr], Expr]:
    """The coefficients with denominators cleared and common factors
    removed, and the factor they were multiplied by."""
    combined = as_expr(together(Add(*[v * Dummy() for v in values])))
    _, denominator = combined.as_numer_denom()
    scaled = [as_expr(cancel(v * denominator)) for v in values]
    scale = as_expr(denominator)
    try:
        polys = [Poly(v, t) for v in scaled]
        common = polys[0]
        for p in polys[1:]:
            common = common.gcd(p)
        if common.degree() > 0 or common.LC() != 1:
            scaled = [as_expr(p.quo(common).as_expr()) for p in polys]
            scale = as_expr(denominator / common.as_expr())
    except (PolynomialError, CoercionFailed, GeneratorsError):
        pass
    if scaled and scaled[-1].could_extract_minus_sign():
        scaled = [as_expr(-v) for v in scaled]
        scale = as_expr(-scale)
    return scaled, scale


def _with_fresh_parameter(F: Expr, t: Symbol) -> tuple[Expr, Symbol]:
    """``F`` with the parameter written as a fresh plain symbol: SymPy's
    ``expr_to_holonomic`` keeps a lookup table built for one coefficient
    domain, and after a call with another domain over a symbol of the
    same name it fails on functions it knows (the bug: the telescoper of
    ``exp(-x**2)*besselj(0, t*x)`` was not found after a Laplace
    transform over a plain ``t``)."""
    fresh = Dummy(t.name)
    return as_expr(F.xreplace({t: fresh})), fresh


def chyzak(F: ExprLike, x: Symbol, t: Symbol, max_order: int = 3, max_degree: int = 6) -> Optional[DFiniteTelescoper]:
    """A telescoper and certificate for ``F(x, t)`` by Chyzak's algorithm
    with Koutschan's ansatz (see the module documentation), the order of
    the telescoper and the degree of the certificate increased up to the
    bounds; ``None`` when none is found or ``F`` is not a product of
    D-finite factors of polynomial arguments.

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.dfinite import chyzak
    >>> x, t = symbols('x t')
    >>> chyzak(exp(-x**2)*exp(2*t*x), x, t)
    DFiniteTelescoper([-2*t, 1], {exp(-x**2)*exp(2*t*x): -1})
    >>> chyzak(exp(-t*x**2), x, t).coefficients
    [1, 2*t]
    """
    F_ = as_expr(sympify(F))
    found = closure(F_, x, t)
    parameter = t
    if found is None:
        # SymPy's lookup table may be stale for this symbol: once more
        # with a fresh one, mapped back below
        F_, parameter = _with_fresh_parameter(F_, t)
        found = closure(F_, x, parameter)
        if found is None:
            return None
    coefficient, factors = found
    closure_ = _Closure(F_, x, parameter, coefficient, factors)
    for order in range(max_order + 1):
        for power in (1, 2):
            for degree in range(max_degree + 1):
                telescoper = attempt(lambda: _telescope(closure_, order, degree, power), settings.timeout)
                if telescoper is not None:
                    if parameter is not t:
                        back = {parameter: t}
                        return DFiniteTelescoper(as_expr(telescoper.term.xreplace(back)), x, t,
                                                 [as_expr(c.xreplace(back)) for c in telescoper.coefficients],
                                                 {as_expr(k.xreplace(back)): as_expr(v.xreplace(back))
                                                  for k, v in telescoper.certificate.items()})
                    return telescoper
    return None


# ---------------------------------------------------------------------------
# The differential equation of the integral, and its solution

def dfinite_ode(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                assumptions: Assumptions = None, name: str = 'I') -> Optional[Eq]:
    """The linear differential equation ``sum_k a_k(t) I^(k)(t) = [G]_a^b``
    satisfied by ``I(t) = Integral(F, (x, a, b))``, the boundary values
    of the certificate taken as limits under the assumptions; ``None``
    when no telescoper is found or a limit is not finite.

    Examples
    ========

    >>> from sympy import symbols, exp, sin, oo
    >>> from sympy_extras.integrals.dfinite import dfinite_ode
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> dfinite_ode(exp(-t*x)*sin(x)/x, x, 0, oo, t)
    Eq(Derivative(I(t), t), -1/(t**2 + 1))
    """
    F_ = as_expr(sympify(F))
    a_, b_ = as_expr(sympify(a)), as_expr(sympify(b))
    telescoper = chyzak(F_, x, t)
    if telescoper is None:
        return None
    G = as_expr(telescoper.function().doit())
    upper = _boundary(G, x, b_, '-', assumptions)
    lower = _boundary(G, x, a_, '+', assumptions)
    if upper is None or lower is None:
        return None
    return Eq(telescoper.operator(name), as_expr(simplify(upper - lower)))


def dfinite_integral(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(F, (x, a, b))`` as a function of the parameter ``t`` by
    Chyzak's algorithm: the equation of :func:`dfinite_ode` solved by
    ``dsolve``, its constants fixed by the values of the integral and its
    derivatives at a point (``1``, ``2``, ``1/2``, ``3``) computed by
    :func:`~sympy_extras.integrals.definite_integral`; checked
    numerically when ``settings.numerical_checks`` is on. ``None`` when
    a step fails.

    Examples
    ========

    >>> from sympy import symbols, exp, besselj, oo
    >>> from sympy_extras.integrals.dfinite import dfinite_integral
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> dfinite_integral(x*exp(-x**2)*besselj(0, t*x), x, 0, oo, t)
    ConditionalValue(exp(-t**2/4)/2)
    """
    from .definite import verify_numerically
    from .marichev import tidy
    F_ = as_expr(sympify(F))
    a_, b_ = as_expr(sympify(a)), as_expr(sympify(b))
    equation = dfinite_ode(F_, x, a_, b_, t, assumptions)
    if equation is None:
        return None
    unknown = Function('I')(t)
    if not isinstance(unknown, AppliedUndef):
        return None
    order = _order(equation.lhs, unknown, t)
    general = attempt(lambda: dsolve(equation, unknown), settings.timeout)
    if not isinstance(general, Eq) or general.lhs != unknown:
        return None
    solution = as_expr(general.rhs)
    constants = _constants(solution)
    if len(constants) != order:
        return None
    if constants:
        fixed: Optional[Expr] = None
        for t0 in (S.One, Integer(2), S.Half, Integer(3)):
            values = _initial_values(F_, x, a_, b_, t, t0, order, assumptions)
            if values is None:
                continue
            equations = [as_expr(diff(solution, t, j).subs(t, t0)) - v for j, v in enumerate(values)]
            solved = attempt(lambda: linsolve(equations, constants), settings.timeout)
            if not isinstance(solved, FiniteSet) or len(solved) != 1:
                continue
            assignment = dict(zip(constants, [as_expr(v) for v in list(solved)[0]]))
            fixed = as_expr(solution.xreplace(assignment))
            break
        if fixed is None:
            return None
        solution = fixed
    if free_symbols(solution) - free_symbols(F_) - {x}:
        return None
    value = tidy(solution, assumptions)
    if value.has(nan, zoo, oo, -oo, Piecewise):
        return None
    if settings.numerical_checks and verify_numerically(value, F_, x, a_, b_, assumptions) is False:
        return None
    return ConditionalValue(value)

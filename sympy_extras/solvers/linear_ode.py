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
* **Hyperexponential solutions** `y = \\exp(\\int \\rho)`, `\\rho` rational,
  when the operator is Fuchsian (regular singular points only): the local
  exponents at the finite singularities and at infinity are the roots
  of the indicial polynomials, the rational combinations of them bound
  the ansatz `y = \\prod (x - c)^{e_c} P(x)`, and `P` is again a polynomial
  solution. Each such solution gives a first order right factor
  `D - \\rho` of the operator (Beke's method for first order factors).
  Irregular singularities (exponential parts `e^{p(x)}`) are not
  treated: for second order equations Kovacic's algorithm
  (:mod:`sympy_extras.solvers.kovacic`) covers them.
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
"""
from __future__ import annotations

from itertools import combinations, product as cartesian
from typing import Optional, Sequence

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef, Derivative, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import binomial, ff
from sympy.functions.elementary.exponential import exp, exp_polar
from sympy.functions.elementary.piecewise import Piecewise
from sympy.integrals.integrals import Integral, integrate
from sympy.series.order import Order
from sympy.matrices.dense import Matrix
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError, PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, cancel, factor_list, gcd_list, resultant, lcm_list

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr, free_symbols
from sympy_extras.settings import settings

__all__ = ['LinearOperator', 'polynomial_solutions', 'rational_solutions',
           'hyperexponential_solutions', 'reduce_order_linear', 'dsolve_linear',
           'gcrd', 'lclm', 'log_derivative', 'hyperexponential_search']


class LinearOperator:
    """``a_n D**n + ... + a_1 D + a_0`` with polynomial coefficients
    ``[a_0, ..., a_n]`` in ``x``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    >>> L.coefficients, L.order
    ([2, -x - 2, x], 2)
    >>> L(x**2 + 2*x + 2)
    0
    """

    def __init__(self, coefficients: Sequence[Expr], x: Symbol) -> None:
        coefficients_ = [as_expr(c) for c in coefficients]
        while len(coefficients_) > 1 and coefficients_[-1] == 0:
            coefficients_.pop()
        self.coefficients = coefficients_
        self.x = x

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
        lhs = equation.lhs - equation.rhs if isinstance(equation, Eq) else as_expr(equation)
        expanded = as_expr(expand(as_expr(lhs)))
        n = 0
        for d in expanded.atoms(Derivative):
            if d.expr == f:
                n = max(n, sum(int(as_expr(count)) for v, count in d.variable_count if v == x))
        coefficients: list[Expr] = []
        rest = expanded
        for i in range(n, -1, -1):
            term = Derivative(f, (x, i)) if i > 0 else f
            c = as_expr(rest.coeff(term))
            rest = as_expr(expand(rest - c*term))
            coefficients.append(c)
        if rest != 0 or any(c.has(f) for c in coefficients):
            raise ValueError("%s is not a homogeneous linear equation in %s" % (equation, f))
        coefficients.reverse()
        denominators = [as_expr(cancel(c).as_numer_denom()[1]) for c in coefficients]
        common = as_expr(lcm_list(denominators))
        cleared = [as_expr(cancel(c*common)) for c in coefficients]
        for c in cleared:
            if not c.is_polynomial(x):
                raise ValueError("the coefficients are not rational functions of %s" % (x,))
        return cls(cleared, x)

    def __call__(self, y: Expr) -> Expr:
        x = self.x
        return as_expr(expand(Add(*[c*y.diff(x, i) if i else c*y for i, c in enumerate(self.coefficients)])))

    def __repr__(self) -> str:
        return "LinearOperator(%s, %s)" % (self.coefficients, self.x)

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
            ratios: list[Expr] = [S.One]
            for _ in range(self.order):
                ratios.append(_normal(ratios[-1].diff(x) + rho*ratios[-1]))
            shifted: list[Expr] = [S.Zero]*(self.order + 1)
            for i, a in enumerate(self.coefficients):
                if a == 0:
                    continue
                for l in range(i + 1):
                    shifted[l] = as_expr(shifted[l] + binomial(i, l)*a*ratios[i - l])
            return LinearOperator([_normal(c) for c in shifted], x).cleared()
        u = Function('u')(x)
        image = as_expr(expand(cancel(self(factor*u)/factor)))
        coefficients: list[Expr] = []
        rest = image
        for i in range(self.order, -1, -1):
            term = Derivative(u, (x, i)) if i > 0 else u
            c = as_expr(rest.coeff(term))
            rest = as_expr(expand(rest - c*term))
            coefficients.append(as_expr(cancel(c)))
        coefficients.reverse()
        denominators = [as_expr(cancel(c).as_numer_denom()[1]) for c in coefficients]
        common = as_expr(lcm_list(denominators))
        return LinearOperator([as_expr(cancel(c*common)) for c in coefficients], x)

    # -- arithmetic in the Ore ring K[D], K = Q(x) (or its extension by
    # the constants of the coefficients), D a = a D + a'

    @property
    def leading_coefficient(self) -> Expr:
        return self.coefficients[-1]

    @property
    def is_zero(self) -> bool:
        return len(self.coefficients) == 1 and self.coefficients[0] == 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinearOperator):
            return NotImplemented
        if self.x != other.x or self.order != other.order:
            return False
        return all(_normal(a - b) == 0 for a, b in zip(self.coefficients, other.coefficients))

    def __hash__(self) -> int:
        return hash((self.x, self.order))

    def __add__(self, other: LinearOperator) -> LinearOperator:
        n = max(self.order, other.order) + 1
        a = self.coefficients + [S.Zero]*(n - len(self.coefficients))
        b = other.coefficients + [S.Zero]*(n - len(other.coefficients))
        return LinearOperator([_normal(p + q) for p, q in zip(a, b)], self.x)

    def __neg__(self) -> LinearOperator:
        return LinearOperator([as_expr(-c) for c in self.coefficients], self.x)

    def __sub__(self, other: LinearOperator) -> LinearOperator:
        return self + (-other)

    def __mul__(self, other: LinearOperator) -> LinearOperator:
        """The composition ``self(other(y))``: ``a D^i b D^j = a sum_l
        binomial(i, l) b^(l) D^(i - l + j)`` (Leibniz's rule).

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> D = LinearOperator([0, 1], x)
        >>> D*LinearOperator([x], x)
        LinearOperator([1, x], x)
        >>> (D - LinearOperator([1/x], x))*(D + LinearOperator([1/x], x))
        LinearOperator([-2/x**2, 0, 1], x)
        """
        x = self.x
        top = max((i for i, a in enumerate(self.coefficients) if a != 0), default=0)
        derivatives: list[list[Expr]] = []
        for b in other.coefficients:
            row = [b]
            for _ in range(top):
                row.append(_normal(row[-1].diff(x)))
            derivatives.append(row)
        result: list[Expr] = [S.Zero]*(self.order + other.order + 1)
        for i, a in enumerate(self.coefficients):
            if a == 0:
                continue
            for l in range(i + 1):
                weight = as_expr(binomial(i, l)*a)
                for j in range(len(other.coefficients)):
                    b = derivatives[j][l]
                    if b != 0:
                        result[i - l + j] = as_expr(result[i - l + j] + weight*b)
        return LinearOperator([_normal(c) for c in result], x)

    def scaled(self, c: Expr) -> LinearOperator:
        """The operator ``c L`` (left multiplication by the function
        ``c``)."""
        return LinearOperator([_normal(c*a) for a in self.coefficients], self.x)

    def monic(self) -> LinearOperator:
        """The operator divided by its leading coefficient.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> LinearOperator([1, x, x**2], x).monic()
        LinearOperator([x**(-2), 1/x, 1], x)
        """
        if self.is_zero:
            raise ValueError("the zero operator has no leading coefficient")
        return self.scaled(as_expr(1/self.leading_coefficient))

    def primitive(self) -> LinearOperator:
        """The operator times a rational function so that the
        coefficients are polynomials in ``x`` without common factor (the
        leading coefficient keeps the sign it had).

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> LinearOperator([2/x**2, 2/x, 2], x).primitive()
        LinearOperator([1, x, x**2], x)
        """
        if self.is_zero:
            return self
        x = self.x
        parts = [_normal(c).as_numer_denom() for c in self.coefficients]
        common = as_expr(lcm_list([as_expr(q) for _, q in parts]))
        numerators = [_normal(as_expr(p)*common/as_expr(q)) for p, q in parts]
        content = as_expr(gcd_list([c for c in numerators if c != 0]))
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
        common = as_expr(lcm_list([as_expr(q) for _, q in parts]))
        return LinearOperator([_normal(as_expr(p)*common/as_expr(q)) for p, q in parts], self.x)

    def right_divmod(self, other: LinearOperator) -> tuple[LinearOperator, LinearOperator]:
        """``(Q, R)`` with ``self = Q*other + R`` and ``R`` of order less
        than ``other`` (Euclidean division on the right in the Ore
        ring).

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> L = LinearOperator([0, 0, 0, 1], x)
        >>> L.right_divmod(LinearOperator([-1, x], x))
        (LinearOperator([0, -1/x**2, 1/x], x), LinearOperator([0], x))
        """
        if other.is_zero:
            raise ZeroDivisionError("division by the zero operator")
        x = self.x
        quotient = LinearOperator([S.Zero], x)
        rest = self
        m = other.order
        lc = other.leading_coefficient
        while not rest.is_zero and rest.order >= m:
            d = rest.order - m
            term = LinearOperator([S.Zero]*d + [_normal(rest.leading_coefficient/lc)], x)
            quotient = quotient + term
            difference = rest - term*other
            # the leading coefficient cancels exactly: drop it even when
            # the normal form of its algebraic constants does not show it
            coefficients = difference.coefficients[:rest.order] if difference.order >= rest.order else difference.coefficients
            rest = LinearOperator(coefficients or [S.Zero], x)
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

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> LinearOperator([0, x, 1], x).adjoint()
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
            total: Expr = S.Zero
            for i in range(l, n + 1):
                total = as_expr(total + (-1)**i*binomial(i, l)*derivatives[i][i - l])
            result.append(_normal(total))
        return LinearOperator(result, x)

    def exterior_power(self, k: int) -> LinearOperator:
        """The ``k``-th associated operator: the monic operator of least
        order annihilating the Wronskian of any ``k`` solutions (of
        order ``binomial(n, k)`` at most).

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> LinearOperator([-1, 0, 0, 1], x).exterior_power(2)
        LinearOperator([1, 0, 0, 1], x)
        """
        if not 1 <= k <= self.order:
            raise ValueError("the power must be between 1 and the order")
        subsets, system = _exterior_system(self.monic(), k)
        start: list[Expr] = [S.One if i == 0 else S.Zero for i in range(len(subsets))]
        return _scalar_operator(system, start, self.x)[0]

    def symmetric_power(self, m: int) -> LinearOperator:
        """The ``m``-th symmetric power: the monic operator of least order
        annihilating the products of ``m`` solutions.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy_extras.solvers.linear_ode import LinearOperator
        >>> LinearOperator([1, 0, 1], x).symmetric_power(2)
        LinearOperator([0, 4, 0, 1], x)
        """
        if m < 1:
            raise ValueError("the power must be positive")
        monomials, system = _symmetric_system(self.monic(), m)
        start: list[Expr] = [S.One if i == 0 else S.Zero for i in range(len(monomials))]
        return _scalar_operator(system, start, self.x)[0]


# ---------------------------------------------------------------------------
# helpers of the operator arithmetic

def _normal(e: Expr) -> Expr:
    """The normal form ``p/q`` of a rational function."""
    return as_expr(cancel(e))


def log_derivative(h: Expr, x: Symbol) -> Expr:
    """``h'/h``, computed factor by factor so that it comes out rational
    for a hyperexponential ``h`` written as a product of powers and
    exponentials.

    Examples
    ========

    >>> from sympy import exp, sqrt, Integral
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import log_derivative
    >>> log_derivative(sqrt(x - 1)*exp(x**2)/x, x)
    (4*x**3 - 4*x**2 - x + 2)/(2*x**2 - 2*x)
    >>> log_derivative(exp(Integral(1/(x**3 + 1), x)), x)
    1/(x**3 + 1)
    """
    if not h.has(x):
        return S.Zero
    if isinstance(h, Mul):
        return _normal(Add(*[log_derivative(as_expr(f), x) for f in h.args]))
    if isinstance(h, Pow) and not h.exp.has(x):
        return _normal(h.exp*as_expr(h.base).diff(x)/h.base)
    if isinstance(h, exp):
        return _normal(as_expr(h.args[0]).diff(x))
    return _normal(h.diff(x)/h)


def _nullspace(rows: list[list[Expr]], columns: int) -> list[list[Expr]]:
    """A basis of the vectors annihilated by the rows, over the field of
    rational functions of the entries."""
    if not rows:
        return [[S.One if k == n else S.Zero for k in range(columns)] for n in range(columns)]
    matrix = Matrix(rows)
    try:
        vectors = DomainMatrix.from_Matrix(matrix).to_field().nullspace().to_Matrix()
    except (CoercionFailed, GeneratorsError, NotImplementedError, ValueError):
        basis = matrix.nullspace()
        vectors = Matrix.hstack(*basis).T if basis else Matrix(0, columns, [])
    return [[_normal(as_expr(vectors[n, k])) for k in range(vectors.cols)] for n in range(vectors.rows)]


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
    system: list[list[Expr]] = [[S.Zero]*len(subsets) for _ in subsets]
    for p, I in enumerate(subsets):
        for t, i in enumerate(I):
            if i + 1 < n:
                if i + 1 in I:
                    continue
                J = I[:t] + (i + 1,) + I[t + 1:]
                system[p][position[J]] = as_expr(system[p][position[J]] + 1)
            else:
                base = I[:-1]
                for m in range(n):
                    if m in base or a[m] == 0:
                        continue
                    J = tuple(sorted(base + (m,)))
                    sign = (-1)**sum(1 for b in base if b > m)
                    system[p][position[J]] = as_expr(system[p][position[J]] - sign*a[m])
    return subsets, system


def _symmetric_system(L: LinearOperator, m: int) -> tuple[list[tuple[int, ...]], list[list[Expr]]]:
    """The system of the monomials of degree ``m`` in ``y, ..., y^(n-1)``
    for a solution ``y`` of the monic ``L``, ``y**m`` first."""
    n = L.order
    a = L.coefficients
    monomials: list[tuple[int, ...]] = [e for e in cartesian(range(m + 1), repeat=n) if sum(e) == m]
    monomials.sort(key=lambda e: tuple(-c for c in e))
    position = {e: p for p, e in enumerate(monomials)}
    system: list[list[Expr]] = [[S.Zero]*len(monomials) for _ in monomials]
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
                system[p][q] = as_expr(system[p][q] + e[i])
            else:
                for j in range(n):
                    if a[j] == 0:
                        continue
                    raised = list(lowered)
                    raised[j] += 1
                    q = position[tuple(raised)]
                    system[p][q] = as_expr(system[p][q] - e[i]*a[j])
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
        following = [_normal(v[j].diff(x) + Add(*[v[i]*system[i][j] for i in range(size) if v[i] != 0 and system[i][j] != 0]))
                     for j in range(size)]
        # sum_j c_j v_j + v_N = 0
        columns = rows + [following]
        relations = _nullspace([[columns[j][i] for j in range(len(columns))] for i in range(size)], len(columns))
        for relation in relations:
            if relation[-1] != 0:
                coefficients = [_normal(c/relation[-1]) for c in relation]
                return LinearOperator(coefficients, x), rows
        rows.append(following)


def gcrd(A: LinearOperator, B: LinearOperator) -> LinearOperator:
    """The monic greatest common right divisor of ``A`` and ``B``, by the
    Euclidean algorithm on the right; its solutions are the common
    solutions of ``A`` and ``B``.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, gcrd
    >>> D = LinearOperator([0, 1], x)
    >>> R = D - LinearOperator([1/x], x)
    >>> gcrd((D*D + LinearOperator([1], x))*R, (D - LinearOperator([x], x))*R)
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

    >>> from sympy.abc import x
    >>> from sympy import exp, simplify
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, lclm
    >>> M = lclm(LinearOperator([-1, 1], x), LinearOperator([-1, x], x))
    >>> M.order, simplify(M(exp(x))), M(x)
    (2, 0, 0)
    """
    x = A.x
    A, B = A.monic(), B.monic()
    remainders: list[tuple[LinearOperator, LinearOperator]] = []
    power = LinearOperator([S.One], x)
    D = LinearOperator([S.Zero, S.One], x)
    a, b = A.order, B.order
    while True:
        remainders.append((power.right_remainder(A), power.right_remainder(B)))
        vectors = [r.coefficients + [S.Zero]*(a - len(r.coefficients)) + s.coefficients + [S.Zero]*(b - len(s.coefficients))
                   for r, s in remainders]
        size = a + b
        relations = _nullspace([[v[i] for v in vectors] for i in range(size)], len(vectors))
        for relation in relations:
            if relation[-1] != 0:
                return LinearOperator([_normal(c/relation[-1]) for c in relation], x)
        power = D*power


def _falling(d: Expr, i: int) -> Expr:
    return as_expr(ff(d, i))


def _indicial_at_infinity(L: LinearOperator) -> Poly:
    """The polynomial in ``d`` whose nonnegative integer roots bound the
    degree of the polynomial solutions."""
    x = L.x
    d = Dummy('d')
    data: list[tuple[int, int, Expr]] = []
    for i, c in enumerate(L.coefficients):
        if c == 0:
            continue
        p = Poly(c, x)
        data.append((i, p.degree() - i, as_expr(p.LC())))
    top = max(shift for _, shift, _ in data)
    expression = Add(*[lc*_falling(d, i) for i, shift, lc in data if shift == top])
    return Poly(as_expr(expression), d)


def _nonnegative_integer_roots(p: Poly) -> list[int]:
    found = roots(p)
    result = [int(r) for r in found if isinstance(r, Integer) and int(r) >= 0]
    return sorted(result)


def _negative_integer_roots(p: Poly) -> list[int]:
    found = roots(p)
    return sorted(int(r) for r in found if isinstance(r, Integer) and int(r) < 0)


def polynomial_solutions(L: LinearOperator) -> list[Expr]:
    """A basis of the polynomial solutions of ``L(y) = 0``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, polynomial_solutions
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    >>> polynomial_solutions(L)
    [x**2 + 2*x + 2]
    >>> polynomial_solutions(LinearOperator([-2, 0, 1], x))
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
    unknowns = [Dummy('c%d' % i) for i in range(bound + 1)]
    ansatz = as_expr(Add(*[c*x**i for i, c in enumerate(unknowns)]))
    image = as_expr(expand(L(ansatz)))
    try:
        rows = Poly(image, x).all_coeffs()
    except PolynomialError:
        return []
    M = Matrix([[as_expr(row).coeff(u) for u in unknowns] for row in rows]) if rows else Matrix.zeros(0, len(unknowns))
    solutions: list[Expr] = []
    for vector in M.nullspace():
        solutions.append(as_expr(Add(*[v*x**i for i, v in enumerate(vector)])))
    return solutions


def _indicial_at(L: LinearOperator, q: Poly) -> Poly:
    """The indicial polynomial (in ``m``) at the roots ``c`` of the
    irreducible ``q``, as a polynomial with rational coefficients whose
    integer roots contain those of the true indicial polynomial (the norm
    from ``Q(c)`` when ``q`` is not linear)."""
    x = L.x
    m = Dummy('m')
    c = Dummy('c')
    orders: list[tuple[int, int, Expr]] = []
    for i, coefficient in enumerate(L.coefficients):
        if coefficient == 0:
            continue
        p = Poly(coefficient, x)
        order = 0
        while True:
            quotient, remainder = p.div(q)
            if not remainder.is_zero:
                break
            p = quotient
            order += 1
        # the leading coefficient of a_i = q**order p at c in powers of
        # x - c: q = (x - c) q~ with q~(c) = q'(c), so p(c) q'(c)**order,
        # reduced modulo q(c) (the factor q'(c)**order was missing, and
        # the exponents at the roots of a factor like 2*x + 1 came out
        # wrong)
        cofactor = as_expr(p.as_expr()*q.diff(x).as_expr()**order)
        value = Poly(cofactor.subs(x, c), c).rem(Poly(q.as_expr().subs(x, c), c)).as_expr() \
            if q.degree() > 1 else cofactor.subs(x, as_expr(-q.all_coeffs()[1]/q.all_coeffs()[0]))
        orders.append((i, order - i, as_expr(value)))
    lowest = min(shift for _, shift, _ in orders)
    expression = as_expr(Add(*[value*_falling(m, i) for i, shift, value in orders if shift == lowest]))
    if q.degree() > 1:
        expression = as_expr(resultant(expression, q.as_expr().subs(x, c), c))
    return Poly(expression, m)


def _denominator_bound(L: LinearOperator) -> Expr:
    x = L.x
    leading = Poly(L.coefficients[-1], x)
    _, factors = factor_list(leading.as_expr(), x)
    result: Expr = S.One
    for f, _ in factors:
        q = Poly(f, x)
        if q.degree() == 0:
            continue
        indicial = _indicial_at(L, q)
        if indicial.is_zero:
            continue
        negative = _negative_integer_roots(indicial)
        if negative:
            result = as_expr(result*f**(-negative[0]))
    return result


def rational_solutions(L: LinearOperator) -> list[Expr]:
    """A basis of the rational solutions of ``L(y) = 0``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, rational_solutions
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    >>> rational_solutions(L)
    [x**(-2), 1/x]
    """
    L = _polynomial_operator(L)
    Q = _denominator_bound(L)
    if Q == 1:
        return polynomial_solutions(L)
    M = L.transformed(1/Q)
    return [as_expr(cancel(P/Q)) for P in polynomial_solutions(M)]


# ---------------------------------------------------------------------------
# hyperexponential solutions of Fuchsian operators

def _is_fuchsian(L: LinearOperator) -> bool:
    """Whether every finite singular point is regular singular:
    ``ord_c(a_i) >= ord_c(a_n) - (n - i)``."""
    x = L.x
    n = L.order
    leading = Poly(L.coefficients[-1], x)
    _, factors = factor_list(leading.as_expr(), x)
    for f, multiplicity in factors:
        q = Poly(f, x)
        if q.degree() == 0:
            continue
        for i, c in enumerate(L.coefficients[:-1]):
            if c == 0:
                continue
            order = _order_at(Poly(c, x), q)
            if order < multiplicity - (n - i):
                return False
    return True


def _order_at(p: Poly, q: Poly) -> int:
    order = 0
    while True:
        quotient, remainder = p.div(q)
        if not remainder.is_zero:
            return order
        p = quotient
        order += 1


def _exponents(L: LinearOperator, q: Poly) -> list[Rational]:
    """The rational local exponents at the roots of ``q``."""
    return _exponents_status(L, q)[0]


def _exponents_status(L: LinearOperator, q: Poly) -> tuple[list[Rational], bool]:
    """The rational local exponents at the roots of ``q``, and whether
    they are all the exponents there: every root of the indicial
    polynomial found and rational, and all of them congruent modulo the
    integers when ``q`` is not linear (conjugate points could otherwise
    take exponents of different classes; integer differences are taken
    by the polynomial of the ansatz)."""
    indicial = _indicial_at(L, q)
    if indicial.is_zero:
        return [], False
    found = roots(indicial)
    rational = sorted(set(r for r in found if isinstance(r, Rational)), key=lambda r: (r.p, r.q))
    complete = sum(found.values()) == indicial.degree() and len(rational) == len(found)
    if q.degree() > 1 and any(not (e - rational[0]).is_integer for e in rational):
        complete = False
    return rational, complete


def _newton_polygon_parts(L: LinearOperator) -> tuple[list[Expr], bool]:
    """Candidate leading terms ``c x**s`` (``s`` a nonnegative integer)
    of ``rho = y'/y`` at infinity, from the edges of the Newton polygon
    of the points ``(i, deg a_i)`` with slope ``-s``: on such an edge the
    terms ``a_i rho**i`` balance, ``sum lc(a_i) c**i = 0``; and whether
    every root of those polynomials was found."""
    x = L.x
    points: list[tuple[int, int, Expr]] = []
    for i, c in enumerate(L.coefficients):
        if c != 0:
            p = Poly(c, x)
            points.append((i, p.degree(), as_expr(p.LC())))
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
        slope = Rational(d2 - d1, i2 - i1)
        s = -slope
        if s < 0 or not s.is_integer:
            continue
        on_edge = [(i, lc) for i, d, lc in points if d - d1 == slope*(i - i1)]
        characteristic = Poly(as_expr(Add(*[lc*t**i for i, lc in on_edge])), t)
        found = roots(characteristic)
        if sum(found.values()) != characteristic.degree():
            complete = False
        for root in found:
            r = as_expr(root)
            if r != 0 and not r.has(x):
                candidates.append(as_expr(r*x**int(s)))
            elif r != 0:
                complete = False
    return candidates, complete


def _exponential_parts(L: LinearOperator, depth: int = 0, below: Optional[int] = None) -> tuple[list[Expr], bool]:
    """The polynomial parts ``P`` of ``rho`` at infinity (solutions
    ``exp(Integral(P)) * ...``), found term by term from the Newton
    polygon, each term of lower degree than the one before (``below``);
    ``0`` is always a candidate (no exponential part). The flag says
    whether the candidates are all the possible parts."""
    x = L.x
    parts: list[Expr] = [S.Zero]
    leading_terms, complete = _newton_polygon_parts(L)
    if below is not None:
        leading_terms = [t for t in leading_terms if Poly(t, x).degree() < below]
    if depth > 4:
        return parts, complete and not leading_terms
    for leading in leading_terms:
        exponent = as_expr(integrate(leading, x))
        M = L.transformed(exp(exponent))
        lower_parts, lower_complete = _exponential_parts(M, depth + 1, Poly(leading, x).degree())
        complete = complete and lower_complete
        for lower in lower_parts:
            candidate = as_expr(leading + lower)
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
    """Solutions ``exp(Integral(P)) * prod (x - c)**e_c * Q(x)`` with ``P``
    a polynomial (the exponential part at infinity, from the Newton
    polygon), rational exponents at the finite singular points, which
    must be regular singular, and a polynomial ``Q``. Solutions with
    distinct exponent combinations are independent.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, hyperexponential_solutions
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(4*x**2*y.diff(x, 2) + 4*x*y.diff(x) - y, y)
    >>> hyperexponential_solutions(L)
    [1/sqrt(x), sqrt(x)]
    >>> hyperexponential_solutions(LinearOperator.from_equation(y.diff(x, 2) - 2*x*y.diff(x) + 4*y, y))
    [x**2 - 1/2]
    >>> hyperexponential_solutions(LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y))
    [x**2 + 2*x + 2, exp(x)]
    """
    return hyperexponential_search(L, max_combinations)[0]


def hyperexponential_search(L: LinearOperator, max_combinations: int = 200) -> tuple[list[Expr], bool]:
    """The hyperexponential solutions of :func:`hyperexponential_solutions`
    and whether the search was exhaustive: ``True`` when every
    hyperexponential solution (over the algebraic closure of the
    constants) is a linear combination of the solutions returned. It is
    not when a finite singular point is irregular, when a local exponent
    is not rational or not found, when conjugate singular points could
    take different exponents, or when the exponential part at infinity
    has more terms than the search follows.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, hyperexponential_search
    >>> hyperexponential_search(LinearOperator([-x, 0, 1], x))
    ([], True)
    >>> hyperexponential_search(LinearOperator([-1, 0, x**2], x))
    ([], False)

    The second equation has the solutions ``x**((1 +- sqrt(5))/2)``,
    whose exponents are not rational.
    """
    x = L.x
    L = _polynomial_operator(L)
    solutions: list[Expr] = []
    parts, complete = _exponential_parts(L)
    for part in parts:
        if part == 0:
            M = L
            factor_part: Expr = S.One
        else:
            factor_part = as_expr(exp(integrate(part, x)))
            M = L.transformed(factor_part)
        found, found_complete = _hyperexponential_regular(M, max_combinations)
        complete = complete and found_complete
        for s in found:
            candidate = as_expr(factor_part*s)
            if candidate not in solutions:
                solutions.append(candidate)
    return _independent(solutions, x), complete


def _hyperexponential_regular(L: LinearOperator, max_combinations: int) -> tuple[list[Expr], bool]:
    """The ansatz with the local exponents at the finite singular points,
    which must be regular singular. One exponent per class modulo the
    integers is enough at each point (the smallest: the polynomial
    ``Q`` takes the integer shifts), and a combination is kept only when
    the exponents at infinity allow a polynomial ``Q``."""
    x = L.x
    if not _is_fuchsian(L):
        return [], False
    leading = Poly(L.coefficients[-1], x)
    _, factors = factor_list(leading.as_expr(), x)
    places: list[tuple[Expr, int, list[Rational]]] = []
    complete = True
    for f, _ in factors:
        q = Poly(f, x)
        if q.degree() == 0:
            continue
        # a common exponent for conjugate roots: the factor itself
        exponents, exponents_complete = _exponents_status(L, q)
        complete = complete and exponents_complete
        if not exponents:
            return [], complete
        representatives: list[Rational] = []
        for e in exponents:
            if not any((e - r).is_integer for r in representatives):
                representatives.append(e)
        places.append((as_expr(f), q.degree(), representatives))
    at_infinity = _indicial_at_infinity(L)
    d = at_infinity.gen
    t = Dummy('t')
    solutions: list[Expr] = []
    combinations = list(cartesian(*[e for _, _, e in places]))
    if len(combinations) > max_combinations:
        combinations = combinations[:max_combinations]
        complete = False
    for combination in combinations:
        degree = as_expr(Add(*[e*k for (_, k, _), e in zip(places, combination)]))
        # y ~ x**(degree + deg Q) at infinity
        if not at_infinity.is_zero and not _nonnegative_integer_roots(Poly(at_infinity.as_expr().subs(d, degree + t), t)):
            continue
        factor = as_expr(Mul(*[f**e for (f, _, _), e in zip(places, combination)]))
        M = L.transformed(factor)
        for P in polynomial_solutions(M):
            solutions.append(as_expr(factor*P))
    return _independent(solutions, x), complete


# ---------------------------------------------------------------------------
# reduction of order

def reduce_order_linear(L: LinearOperator, y1: Expr) -> LinearOperator:
    """The operator ``M`` of order ``n - 1`` with ``L(y1 * w) = y1 * M(w')``
    for a solution ``y1`` of ``L``.

    Examples
    ========

    >>> from sympy import Function, exp
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, reduce_order_linear
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(y.diff(x, 2) - 3*y.diff(x) + 2*y, y)
    >>> reduce_order_linear(L, exp(x))
    LinearOperator([-1, 1], x)
    """
    x = L.x
    T = L.transformed(y1)
    if cancel(T.coefficients[0]) != 0:
        raise ValueError("%s is not a solution" % (y1,))
    return LinearOperator(T.coefficients[1:], x)


def _independent(solutions: list[Expr], x: Symbol) -> list[Expr]:
    """Drop the solutions which are constant multiples of earlier ones."""
    kept: list[Expr] = []
    for s in solutions:
        if s == 0:
            continue
        if any(_proportional(s, k, x) for k in kept):
            continue
        kept.append(s)
    return kept


def _proportional(s: Expr, k: Expr, x: Symbol) -> bool:
    """Whether ``s/k`` is constant; a ratio with unevaluated integrals
    is taken as not constant (deciding it would call ``simplify`` on the
    integrals, which may not return)."""
    ratio = as_expr(cancel(s/k))
    if not ratio.has(x):
        return True
    if ratio.has(Integral):
        return False
    try:
        # is_constant substitutes values such as 0 for x; SymPy's
        # evalf of meijerg(..., zoo) raises AttributeError
        # (sympy-extras#25), and a ratio it cannot decide is taken as not
        # constant
        return bool(attempt(lambda: ratio.is_constant(x), settings.timeout))
    except AttributeError:
        return False


def dsolve_linear(equation: Basic, f: AppliedUndef, use_kovacic: bool = True,
                  use_dsolve: bool = True) -> list[Expr]:
    """Independent solutions of a homogeneous linear equation with
    rational coefficients: rational and hyperexponential solutions,
    Kovacic's algorithm for the second order, and reduction of order by
    the solutions found (the reduced equations handed to the same
    solvers and to SymPy's ``dsolve``). The list is empty when nothing
    is found and shorter than the order when the solution space is not
    exhausted.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import dsolve_linear
    >>> y = Function('y')(x)
    >>> dsolve_linear(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    [x**2 + 2*x + 2, exp(x)]
    >>> dsolve_linear(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    [x**(-2), 1/x]
    """
    L = LinearOperator.from_equation(equation, f)
    return _solve_operator(L, f, use_kovacic, use_dsolve)


def _solve_operator(L: LinearOperator, f: AppliedUndef, use_kovacic: bool, use_dsolve: bool,
                    factorize: bool = True) -> list[Expr]:
    x = L.x
    n = L.order
    if n == 0:
        return []
    if n == 1:
        # y' = -a_0/a_1 y
        rho = as_expr(cancel(-L.coefficients[0]/L.coefficients[1]))
        integral = attempt(lambda: integrate(rho, x), settings.timeout)
        if integral is None:
            return [as_expr(Function('exp')(Integral(rho, x)))]
        from sympy.functions.elementary.exponential import exp
        return [as_expr(exp(integral))]
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
        if reduced_found is not None and len(reduced_found) == 2 and not reduced_found[1].has(Integral):
            return reduced_found
    if len(found) < n and n == 2 and use_kovacic:
        from .kovacic import dsolve_kovacic
        y = Function('y')(x)
        equation = as_expr(Add(*[c*y.diff(x, i) if i else c*y for i, c in enumerate(L.coefficients)]))
        liouvillian = attempt(lambda: dsolve_kovacic(equation, y), settings.timeout)
        if liouvillian:
            found = _independent(found + liouvillian, x)
    if len(found) < n and n == 2:
        from .special import special_solutions
        y = Function('y')(x)
        equation = as_expr(Add(*[c*y.diff(x, i) if i else c*y for i, c in enumerate(L.coefficients)]))
        special = attempt(lambda: special_solutions(equation, y), settings.timeout)
        if special:
            found = _independent(found + special, x)
    if found and len(found) < n:
        reduced_found = _by_reduction(L, f, found, use_kovacic, use_dsolve, factorize)
        if reduced_found is None:
            return found
        found = reduced_found
    if len(found) < n and use_dsolve:
        from sympy.solvers.ode import dsolve
        y = Function('y')(x)
        equation = as_expr(Add(*[c*y.diff(x, i) if i else c*y for i, c in enumerate(L.coefficients)]))
        result = attempt(lambda: dsolve(equation, y), settings.timeout)
        # a truncated power series with an O term is not a solution either
        # (sympy-extras#37)
        if isinstance(result, Eq) and not result.rhs.has(Integral, Order):
            rhs = as_expr(expand(as_expr(result.rhs)))
            constants = sorted((s for s in free_symbols(rhs) if s.name.startswith('C')), key=lambda s: s.name)
            for c in constants:
                part = as_expr(rhs.coeff(c))
                if part != 0 and not part.has(*constants):
                    found = _independent(found + [part], x)
    return found[:n]


def _by_reduction(L: LinearOperator, f: AppliedUndef, found: list[Expr], use_kovacic: bool, use_dsolve: bool,
                  factorize: bool) -> Optional[list[Expr]]:
    """``found`` extended by the solutions of the operator reduced by the
    first solution (``y1 * Integral(v)`` for its solutions ``v``), ``None``
    when the first one is not a solution."""
    x = L.x
    y1 = found[0]
    try:
        M = reduce_order_linear(L, y1)
    except ValueError:
        return None
    reduced = _solve_operator(M, f, use_kovacic, use_dsolve, factorize)
    for v in reduced:
        w = attempt(lambda: integrate(v, x, conds='none'), settings.timeout)
        if w is None or w.has(Integral, Piecewise, exp_polar):
            # the quadrature is left unevaluated
            w = Integral(v, x)
        found = _independent(found + [as_expr(cancel(y1*w)) if (y1*w).is_rational_function(x) else as_expr(y1*w)], x)
    return found


def _solve_by_factorization(L: LinearOperator, f: AppliedUndef, use_kovacic: bool, use_dsolve: bool) -> list[Expr]:
    """Independent solutions of ``L`` through a factorisation into at
    least two factors (none when ``L`` does not factor)."""
    from .factorization import factor_operator, solve_factored
    limit = None if settings.timeout is None else 4*settings.timeout
    factorization = attempt(lambda: factor_operator(L), limit)
    if factorization is None or len(factorization.factors) < 2:
        return []

    def solver(M: LinearOperator) -> list[Expr]:
        return _solve_operator(M.primitive(), f, use_kovacic, use_dsolve, False)

    return _independent(solve_factored(factorization.factors, solver), L.x)

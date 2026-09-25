"""First order linear systems ``Y' = A(x) Y + b(x)`` with rational function
coefficients: cyclic vectors, rational, hyperexponential and general
solutions.

**Cyclic vectors.** For a row vector `c_0` the derivatives of `u = c_0 Y`
along the solutions are `u^{(k)} = c_k Y` with `c_{k+1} = c_k A + c_k'`.
The vector is *cyclic* when `c_0, \\ldots, c_{n-1}` are independent over
`K = \\mathbb{Q}(x)`: the matrix `M` of these rows is invertible, `c_n =
\\sum_k a_k c_k` with `(a_k) = c_n M^{-1}`, and `u` satisfies the scalar
equation `u^{(n)} = \\sum_k a_k u^{(k)}`. Conversely every solution `u` of
that equation gives the solution `Y = M^{-1} (u, u', \\ldots,
u^{(n-1)})^T` of the system, so a basis of the scalar solutions is a
fundamental matrix. Every system over a differential field with a
nonconstant element has cyclic vectors, and they are generic (Cope,
Katz; Churchill and Kovacic show that a random choice among the
vectors with polynomial entries of degree less than `n` succeeds with
high probability and survey the deterministic constructions). The unit
vectors are tried first, since they give the simplest scalar equations
(the first unit vector is cyclic for a companion system), then vectors
drawn by a seeded generator, constant ones before polynomial ones; each
is checked by the invertibility of `M`. The scalar equation is solved
by :func:`~.dsolve_linear`'s solvers (rational and hyperexponential
solutions, Beke's factorisation for the order three and more,
Kovacic's algorithm, special functions) and, when they do not give a
full basis, by SymPy's ``linodesolve`` (constant coefficients and
systems with a commuting antiderivative), whose columns are checked.

**Rational solutions** (Barkatou's method, with the local bounds of
Abramov and Bronstein). A rational solution is `Y = P/D` with `P` a
vector of polynomials and `D` a *universal denominator* built from the
finite singularities, the irreducible factors `p` of the denominator
`d` of `A`:

* at a simple pole of `A` the leading term `Y_{-m} (x - \\alpha)^{-m}` of
  a solution with a pole of order `m` gives `R Y_{-m} = -m Y_{-m}`, `R`
  the residue of `A` at the root `\\alpha` of `p`: `-m` is a negative
  integer eigenvalue of `R`. The eigenvalues at all the roots of `p`
  are the roots of the resultant with `p` of `\\det(\\lambda p' q I - N)`,
  `A = N/(p q)`, so the bound is found over `\\mathbb{Q}`;
* at a pole of order `e \\ge 2` the lowest term of `Y' = A Y` is
  `A_{-e} Y_v = 0` for the leading coefficient `Y_v` of any Laurent
  solution, so when `A_{-e}` is invertible (`p` does not divide
  `\\det N`) there is no nonzero rational solution at all; otherwise the
  system would first have to be reduced (Moser's algorithm, Barkatou's
  simple forms), which is not implemented, and the bound is taken from
  the scalar equation of a cyclic vector instead: the pole order of `u =
  c_0 Y` is bounded by its indicial equation at `p`, and `Y = M^{-1} (u,
  \\ldots, u^{(n-1)})` bounds that of `Y`.

At infinity the same leading term argument with `R_\\infty = \\lim x A`
bounds the degree when `A = O(1/x)`; when `A = O(x^q)`, `q \\ge 0`, with
an invertible leading matrix there is no rational solution, and
otherwise the scalar equation bounds the degree. The numerators solve
the linear system `d D P' - d D' P - D N P = 0` for the coefficients of
`P`. The rational solutions of an inhomogeneous system are those of
the augmented system `(Y, z)' = [[A, b], [0, 0]] (Y, z)` with `z = 1`.

**Hyperexponential solutions** `Y = h R`, `h'/h` and `R` rational,
correspond to the hyperexponential solutions `u = c_0 Y` of the scalar
equation (`u^{(k)} = u R_k` with `R_{k+1} = R_k' + (u'/u) R_k`), found by
:func:`~.hyperexponential_search`.

**Inhomogeneous systems** by variation of constants: with a fundamental
matrix `\\Phi`, `Y = \\Phi \\int \\Phi^{-1} b`; the components of `\\Phi^{-1}
b` are quotients of determinants (Cramer's rule) and `\\det \\Phi` is taken
from Liouville's formula `\\det \\Phi = C \\exp \\int \\operatorname{tr} A`
when it matches. A rational particular solution is looked for first.

SymPy's ``dsolve`` solves linear systems with constant coefficients,
systems whose coefficient matrix commutes with its antiderivative, and a
few special types of two and three equations; it has no algorithm for
general rational coefficients.

References
==========

.. [ChurchillKovacic] R. C. Churchill, J. Kovacic, Cyclic vectors, in
   Differential Algebra and Related Topics, World Scientific (2002).
.. [Katz] N. Katz, A simple algorithm for cyclic vectors, American
   Journal of Mathematics 109 (1987).
.. [Barkatou93] M. Barkatou, An algorithm for computing a companion
   block diagonal form for a system of linear differential equations,
   AAECC 4 (1993).
.. [Barkatou99] M. Barkatou, On rational solutions of systems of linear
   differential equations, Journal of Symbolic Computation 28 (1999).
.. [AbramovBronstein] S. A. Abramov, M. Bronstein, On solutions of
   linear functional systems, ISSAC 2001.
.. [Coddington] E. A. Coddington, N. Levinson, Theory of Ordinary
   Differential Equations, McGraw-Hill (1955), chapter 3 (Liouville's
   formula, variation of constants).
"""
from __future__ import annotations

from random import Random
from typing import Optional, Sequence

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, expand
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import Lambda
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices.dense import Matrix, MutableDenseMatrix
from sympy.matrices.matrixbase import MatrixBase
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMError
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError
from sympy.integrals.rationaltools import ratint_logpart, ratint_ratpart
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, cancel, factor_list, lcm_list, resultant
from sympy.polys.rootoftools import RootSum
from sympy.simplify.simplify import simplify

from sympy_extras._timeout import attempt
from sympy_extras._typing import Truth, as_expr, free_symbols
from sympy_extras.settings import settings

from .linear_ode import (LinearOperator, _solve_operator, _nullspace, _indicial_at, _indicial_at_infinity,
    hyperexponential_search, log_derivative)

__all__ = ['CyclicReduction', 'LinearSystemSolution', 'cyclic_vector', 'cyclic_reduction', 'system_to_scalar',
           'rational_system_solutions', 'hyperexponential_system_solutions', 'dsolve_system',
           'dsolve_linear_system']

#: the rows of a matrix of rational functions
Rows = list[list[Expr]]

#: the seed of the generator of candidate cyclic vectors (fixed, so that
#: the scalar equation, and the form of the solutions, are reproducible)
_CYCLIC_SEED = 2002
#: the number of random candidates tried after the unit vectors
_CYCLIC_TRIALS = 24


def _normal(e: Expr) -> Expr:
    return as_expr(cancel(e))


def _entries(A: MutableDenseMatrix, x: Symbol) -> Rows:
    """The entries of the square matrix ``A``, checked to be rational in
    ``x`` and normalised."""
    if A.rows != A.cols or A.rows == 0:
        raise ValueError("the matrix of the system must be square and nonempty")
    rows: Rows = []
    for i in range(A.rows):
        row: list[Expr] = []
        for j in range(A.cols):
            e = as_expr(A[i, j])
            if not e.is_rational_function(x):
                raise ValueError("the entry %s is not a rational function of %s" % (e, x))
            row.append(_normal(e))
        rows.append(row)
    return rows


def _column(b: MutableDenseMatrix, n: int, x: Symbol) -> list[Expr]:
    if not isinstance(b, MatrixBase):
        raise TypeError("the inhomogeneous term must be a Matrix, not %r" % (b,))
    if b.shape != (n, 1):
        raise ValueError("the inhomogeneous term must be a column of length %d" % n)
    return [as_expr(b[i, 0]) for i in range(n)]


def _next_row(A: Rows, c: list[Expr], x: Symbol) -> list[Expr]:
    """``c A + c'``: the row of ``u^(k+1)`` from that of ``u^(k)``."""
    n = len(c)
    return [_normal(c[j].diff(x) + Add(*[c[i]*A[i][j] for i in range(n) if c[i] != 0 and A[i][j] != 0]))
            for j in range(n)]


def _invert(rows: Rows) -> Optional[Rows]:
    """The inverse of a matrix of rational functions, ``None`` when it is
    singular."""
    matrix = Matrix(rows)
    try:
        inverse = DomainMatrix.from_Matrix(matrix).to_field().inv().to_Matrix()
    except DMError:
        # DMNonInvertibleMatrixError: singular
        return None
    except (CoercionFailed, GeneratorsError, NotImplementedError, ValueError, ZeroDivisionError):
        determinant = _normal(as_expr(matrix.det(method='berkowitz')))
        if determinant == 0:
            return None
        inverse = matrix.adjugate(method='berkowitz')/determinant
    return [[_normal(as_expr(inverse[i, j])) for j in range(len(rows))] for i in range(len(rows))]


def _candidates(n: int, x: Symbol) -> list[list[Expr]]:
    """The unit vectors, then random vectors: constant entries first,
    then polynomial entries of degree less than ``n`` (a generic vector of
    this kind is cyclic, Katz, Churchill–Kovacic); the generator is
    seeded so that the choice is reproducible."""
    candidates: list[list[Expr]] = [[S.One if j == i else S.Zero for j in range(n)] for i in range(n)]
    generator = Random(_CYCLIC_SEED)
    for trial in range(_CYCLIC_TRIALS):
        degree = 0 if trial < _CYCLIC_TRIALS//3 else min(n - 1, 1 + trial % n)
        vector = [as_expr(Add(*[Integer(generator.randint(-3, 3))*x**k for k in range(degree + 1)]))
                  for _ in range(n)]
        if any(v != 0 for v in vector):
            candidates.append(vector)
    return candidates


class CyclicReduction:
    """The scalar form of ``Y' = A Y`` given by a cyclic row vector
    ``vector``: ``u = vector * Y`` satisfies ``operator(u) = 0``, and
    ``(u, u', ..., u^(n-1))^T = matrix * Y``, so that
    :meth:`to_system` maps the solutions of the operator to those of the
    system (an isomorphism of the solution spaces).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import cyclic_reduction
    >>> R = cyclic_reduction(Matrix([[0, 1], [2/x**2, 0]]), x)
    >>> R.operator
    LinearOperator([-2, 0, x**2], x)
    >>> R.to_system(x**2).T
    Matrix([[x**2, 2*x]])
    """

    def __init__(self, vector: list[Expr], rows: Rows, inverse: Rows, operator: LinearOperator, x: Symbol
                 ) -> None:
        self.vector: MutableDenseMatrix = Matrix([vector])
        self.matrix: MutableDenseMatrix = Matrix(rows)
        self.operator = operator
        self.x = x
        self._rows = rows
        self._inverse = inverse

    def to_system(self, u: Expr) -> MutableDenseMatrix:
        """The solution ``Y = matrix**-1 (u, ..., u^(n-1))^T`` of the
        system given by a solution ``u`` of the operator. For a
        hyperexponential ``u`` the vector is written as ``u`` times rational
        functions."""
        x = self.x
        n = len(self._rows)
        rho = log_derivative(u, x) if u.has(x) else S.Zero
        if rho.is_rational_function(x):
            # u^(k) = u R_k with R_0 = 1 and R_{k+1} = R_k' + rho R_k
            ratios: list[Expr] = [S.One]
            for _ in range(n - 1):
                ratios.append(_normal(ratios[-1].diff(x) + rho*ratios[-1]))
            parts = [_normal(Add(*[self._inverse[i][k]*ratios[k] for k in range(n)])) for i in range(n)]
            if u.is_rational_function(x):
                return Matrix([_normal(u*p) for p in parts])
            return Matrix([as_expr(u*p) for p in parts])
        derivatives = [u] + [as_expr(u.diff(x, k)) for k in range(1, n)]
        entries: list[Expr] = []
        for i in range(n):
            e = as_expr(Add(*[self._inverse[i][k]*derivatives[k] for k in range(n) if self._inverse[i][k] != 0]))
            entries.append(_simplified(e))
        return Matrix(entries)

    def __repr__(self) -> str:
        return "CyclicReduction(%s, %s)" % (self.vector.tolist(), self.operator)


def _simplified(e: Expr) -> Expr:
    """``simplify(e)`` within the time limit, except for the expressions
    with integrals, which simplify evaluates (into polylogarithms of
    ``exp_polar`` arguments, which the quadratures had refused), and with
    ``RootSum``s, which it writes in radicals."""
    if e.count_ops() >= 400 or e.has(Integral, RootSum):
        return e
    simplified = attempt(lambda: as_expr(simplify(e)), settings.timeout)
    if simplified is None or simplified.has(exp_polar, Piecewise):
        return e
    return simplified


def _reduction(A: Rows, x: Symbol) -> CyclicReduction:
    n = len(A)
    for c0 in _candidates(n, x):
        rows = [c0]
        for _ in range(n - 1):
            rows.append(_next_row(A, rows[-1], x))
        inverse = _invert(rows)
        if inverse is None:
            continue
        following = _next_row(A, rows[-1], x)
        # c_n = sum_k a_k c_k with (a_k) = c_n M**-1
        a = [_normal(Add(*[following[j]*inverse[j][k] for j in range(n) if following[j] != 0])) for k in range(n)]
        operator = LinearOperator([_normal(-e) for e in a] + [S.One], x).primitive()
        return CyclicReduction(c0, rows, inverse, operator, x)
    raise ValueError("no cyclic vector was found among the candidates")


def cyclic_reduction(A: MutableDenseMatrix, x: Symbol) -> CyclicReduction:
    """The scalar equation of a cyclic vector of ``Y' = A Y`` with the
    matrices mapping its solutions to those of the system.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import cyclic_reduction
    >>> R = cyclic_reduction(Matrix([[1, 0], [0, 2]]), x)
    >>> R.vector, R.operator
    (Matrix([[2, -3]]), LinearOperator([2, -3, 1], x))

    The unit vectors are not cyclic for this diagonal system: ``y1`` alone
    satisfies ``y1' = y1``.
    """
    return _reduction(_entries(A, x), x)


def cyclic_vector(A: MutableDenseMatrix, x: Symbol) -> tuple[MutableDenseMatrix, LinearOperator]:
    """A cyclic row vector ``c_0`` for ``Y' = A Y`` and the scalar operator
    satisfied by ``u = c_0 Y``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import cyclic_vector
    >>> c, L = cyclic_vector(Matrix([[0, 1], [-1, 0]]), x)
    >>> c, L.coefficients
    (Matrix([[1, 0]]), [1, 0, 1])
    """
    R = cyclic_reduction(A, x)
    return R.vector, R.operator


def system_to_scalar(A: MutableDenseMatrix, x: Symbol) -> tuple[LinearOperator, MutableDenseMatrix]:
    """The scalar operator of a cyclic vector and the matrix ``M`` with
    ``(u, u', ..., u**(n-1))**T = M Y``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import system_to_scalar
    >>> system_to_scalar(Matrix([[1/x, 1], [0, 1/x]]), x)
    (LinearOperator([2, -2*x, x**2], x), Matrix([
    [  1, 0],
    [1/x, 1]]))
    """
    R = cyclic_reduction(A, x)
    return R.operator, R.matrix


# ---------------------------------------------------------------------------
# rational solutions

def _integer_roots(p: Expr, variable: Symbol) -> list[int]:
    """The integer roots of a polynomial in ``variable`` (whose other
    symbols are parameters: a root depending on them is not an
    integer)."""
    _, factors = factor_list(p, variable)
    found: list[int] = []
    for f, _ in factors:
        q = Poly(f, variable)
        if q.degree() != 1:
            continue
        leading, constant = q.all_coeffs()
        root = _normal(as_expr(-constant/leading))
        if isinstance(root, Integer):
            found.append(int(root))
    return sorted(set(found))


def _multiplicity(q: Poly, p: Poly) -> int:
    """The largest ``k`` with ``p**k`` dividing ``q``."""
    k = 0
    while not q.is_zero and q.degree() >= p.degree():
        quotient, remainder = q.div(p)
        if not remainder.is_zero:
            break
        q = quotient
        k += 1
    return k


def _degree(e: Expr, x: Symbol) -> Optional[int]:
    """``deg numerator - deg denominator``, ``None`` for zero."""
    if e == 0:
        return None
    numerator, denominator = _normal(e).as_numer_denom()
    return int(Poly(numerator, x).degree() - Poly(denominator, x).degree())


class _ScalarBounds:
    """The bounds for the rational solutions of a system taken from the
    scalar equation of a cyclic vector, built when first needed."""

    def __init__(self, A: Rows, x: Symbol) -> None:
        self.A = A
        self.x = x
        self._reduction: Optional[CyclicReduction] = None

    def reduction(self) -> CyclicReduction:
        if self._reduction is None:
            self._reduction = _reduction(self.A, self.x)
        return self._reduction

    def pole(self, p: Poly) -> int:
        """A bound for the order of the pole at the roots of ``p`` of a
        rational solution."""
        R = self.reduction()
        L = R.operator
        x = self.x
        m = 0
        if _multiplicity(Poly(L.leading_coefficient, x), p) > 0:
            indicial = _indicial_at(L, p)
            negative = [r for r in _integer_roots(indicial.as_expr(), indicial.gen) if r < 0]
            if negative:
                m = -negative[0]
        n = len(self.A)
        bound = 0
        for i in range(n):
            for k in range(n):
                e = R._inverse[i][k]
                if e == 0:
                    continue
                denominator = Poly(_normal(e).as_numer_denom()[1], x)
                # the pole of u^(k) is at most m + k when u has one
                bound = max(bound, _multiplicity(denominator, p) + (m + k if m > 0 else 0))
        return bound

    def degree(self) -> Optional[int]:
        """A bound for ``deg numerator - deg denominator`` of the components
        of a rational solution, ``None`` when there is none."""
        R = self.reduction()
        indicial = _indicial_at_infinity(R.operator)
        exponents = _integer_roots(indicial.as_expr(), indicial.gen)
        if not exponents:
            return None
        delta = exponents[-1]
        n = len(self.A)
        bound: Optional[int] = None
        for i in range(n):
            for k in range(n):
                d = _degree(R._inverse[i][k], self.x)
                if d is not None:
                    bound = d + delta - k if bound is None else max(bound, d + delta - k)
        return bound


def _determinant(rows: Rows) -> Expr:
    return as_expr(expand(as_expr(Matrix(rows).det(method='berkowitz'))))


def _residue_bound(N: Rows, d: Poly, p: Poly, x: Symbol) -> int:
    """The pole order bound at a simple pole: minus the least negative
    integer eigenvalue of the residue ``N(c)/(p'(c) q(c))``, ``d = p q``,
    at the roots ``c`` of ``p``."""
    n = len(N)
    lam = Dummy('lambda')
    q = d.quo(p)
    scale = as_expr((p.diff(x)*q).as_expr())
    characteristic = _determinant([[as_expr(lam*scale - N[i][j]) if i == j else as_expr(-N[i][j])
                                    for j in range(n)] for i in range(n)])
    if p.degree() == 1:
        leading, constant = p.all_coeffs()
        eigen = as_expr(expand(characteristic.subs(x, as_expr(-constant/leading))))
    else:
        eigen = as_expr(resultant(characteristic, p.as_expr(), x))
    negative = [r for r in _integer_roots(eigen, lam) if r < 0]
    return -negative[0] if negative else 0


def _rational_basis(A: Rows, x: Symbol) -> list[list[Expr]]:
    """A basis of the rational solutions of ``Y' = A Y`` (Barkatou)."""
    n = len(A)
    parts = [[_normal(e).as_numer_denom() for e in row] for row in A]
    d = Poly(as_expr(lcm_list([as_expr(den) for row in parts for _, den in row])), x)
    N = [[_normal(as_expr(num)*d.as_expr()/as_expr(den)) for num, den in row] for row in parts]
    scalar = _ScalarBounds(A, x)
    # the universal denominator
    D = Poly(1, x)
    determinant: Optional[Poly] = None
    _, factors = factor_list(d.as_expr(), x)
    for f, e in factors:
        p = Poly(f, x)
        if p.degree() == 0:
            continue
        if e == 1:
            m = _residue_bound(N, d, p, x)
        else:
            if determinant is None:
                determinant = Poly(_determinant(N), x)
            if _multiplicity(determinant, p) == 0:
                # the leading matrix at the pole is invertible: no Laurent
                # solution, so no rational one
                return []
            m = scalar.pole(p)
        D = D*p**m
    # the degree at infinity: deg Y = deg P - deg D
    degrees = [_degree(e, x) for row in A for e in row]
    orders = [-k for k in degrees if k is not None]
    # the order of A at infinity
    shift: Optional[int] = min(orders) if orders else None
    delta: Optional[int]
    if shift is None:
        # A = 0: the constant solutions
        delta = 0
    elif shift >= 1:
        # A = O(1/x): the integer eigenvalues of lim x A
        lam = Dummy('lambda')
        residue = [[_normal(as_expr((x*A[i][j]).limit(x, S.Infinity))) if shift == 1 else S.Zero
                    for j in range(n)] for i in range(n)]
        characteristic = _determinant([[as_expr(lam - residue[i][j]) if i == j else as_expr(-residue[i][j])
                                        for j in range(n)] for i in range(n)])
        exponents = _integer_roots(characteristic, lam)
        delta = exponents[-1] if exponents else None
    else:
        # A = O(x**q), q = -shift >= 0: the leading matrix
        q = -shift
        leading = [[_normal(as_expr((A[i][j]/x**q).limit(x, S.Infinity))) for j in range(n)] for i in range(n)]
        if _normal(_determinant(leading)) != 0:
            return []
        delta = scalar.degree()
    if delta is None:
        return []
    top = D.degree() + delta
    if top < 0:
        return []
    return _numerators(A, N, d, D, top, x)


def _numerators(A: Rows, N: Rows, d: Poly, D: Poly, top: int, x: Symbol) -> list[list[Expr]]:
    """The solutions ``P/D`` with ``deg P <= top``: the kernel of the
    linear map ``P -> d (D P' - D' P) - D N P``."""
    n = len(A)
    dD = d*D
    dDprime = d*D.diff(x)
    columns: list[list[Poly]] = []
    for i in range(n):
        for j in range(top + 1):
            monomial = Poly(x**j, x)
            image: list[Poly] = []
            for k in range(n):
                value = -D*Poly(N[k][i], x)*monomial
                if k == i:
                    value = value + dD*monomial.diff(x) - dDprime*monomial
                image.append(value)
            columns.append(image)
    size = max((p.degree() for column in columns for p in column if not p.is_zero), default=0) + 1
    rows: Rows = []
    for k in range(n):
        for power in range(size):
            row = [as_expr(column[k].coeff_monomial(x**power)) for column in columns]
            if any(e != 0 for e in row):
                rows.append(row)
    denominator = D.as_expr()
    basis: list[list[Expr]] = []
    for vector in _nullspace(rows, len(columns)):
        P = [as_expr(Add(*[vector[i*(top + 1) + j]*x**j for j in range(top + 1)])) for i in range(n)]
        basis.append([_normal(p/denominator) for p in P])
    return basis


def rational_system_solutions(A: MutableDenseMatrix, x: Symbol) -> list[MutableDenseMatrix]:
    """A basis of the rational solutions of ``Y' = A Y``: a universal
    denominator from the finite singularities, a degree bound at
    infinity, and a linear system for the numerators (Barkatou's method;
    see the module docstring for the bounds at poles of order two and
    more).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import rational_system_solutions
    >>> rational_system_solutions(Matrix([[1/x, 1], [0, 1/x]]), x)
    [Matrix([
    [x],
    [0]]), Matrix([
    [x**2],
    [   x]])]
    >>> rational_system_solutions(Matrix([[-1/x, 0], [1, 0]]), x)
    [Matrix([
    [0],
    [1]])]
    """
    return [Matrix(v) for v in _rational_basis(_entries(A, x), x)]


def _augmented(A: Rows, b: list[Expr]) -> Rows:
    n = len(A)
    rows = [A[i] + [_normal(b[i])] for i in range(n)]
    rows.append([S.Zero]*(n + 1))
    return rows


def _rational_particular(A: Rows, b: list[Expr], x: Symbol) -> Optional[list[Expr]]:
    """A rational solution of ``Y' = A Y + b``: a rational solution of the
    augmented system whose last component (a constant) is not zero."""
    n = len(A)
    for vector in _rational_basis(_augmented(A, b), x):
        z = vector[n]
        if z != 0 and not z.has(x):
            return [_normal(vector[i]/z) for i in range(n)]
    return None


# ---------------------------------------------------------------------------
# hyperexponential and general solutions

def hyperexponential_system_solutions(A: MutableDenseMatrix, x: Symbol) -> list[MutableDenseMatrix]:
    """Independent solutions ``h R`` of ``Y' = A Y`` with ``h'/h`` and the
    vector ``R`` rational, from the hyperexponential solutions of the
    scalar equation of a cyclic vector (complete when that search is, see
    :func:`~.hyperexponential_search`).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import hyperexponential_system_solutions
    >>> hyperexponential_system_solutions(Matrix([[1, 1/x], [0, 0]]), x)
    [Matrix([
    [exp(x)],
    [     0]])]
    """
    R = cyclic_reduction(A, x)
    found, _ = hyperexponential_search(R.operator)
    return [R.to_system(u) for u in found]


def _verified(A: Rows, Y: list[Expr], x: Symbol) -> bool:
    """Whether ``Y' = A Y``, by simplification within the time limit."""
    n = len(A)
    for i in range(n):
        residual = as_expr(Y[i].diff(x) - Add(*[A[i][j]*Y[j] for j in range(n)]))
        zero = attempt(lambda: as_expr(simplify(residual)) == 0, settings.timeout)
        if not zero:
            return False
    return True


def _sympy_columns(A: Rows, x: Symbol) -> list[list[Expr]]:
    """The fundamental solutions given by SymPy's ``linodesolve``
    (constant coefficients, commuting antiderivatives), each checked."""
    from sympy.solvers.ode.systems import linodesolve
    matrix = Matrix(A)
    result = attempt(lambda: linodesolve(matrix, x), settings.timeout)
    if result is None:
        return []
    vector = [as_expr(e) for e in result]
    known = {s for row in A for e in row for s in free_symbols(e)} | {x}
    constants = sorted({s for e in vector for s in free_symbols(e) if s not in known}, key=str)
    columns: list[list[Expr]] = []
    for c in constants:
        column = [as_expr(expand(e).coeff(c)) for e in vector]
        if any(e != 0 for e in column) and not any(e.has(*constants) for e in column) and _verified(A, column, x):
            columns.append(column)
    return columns if len(columns) == len(A) else []


def _liouville(A: Rows, determinant: Expr, x: Symbol) -> Expr:
    """``det Phi`` as ``C exp(Integral(trace A))`` when the ratio is found
    to be constant, else the simplified determinant."""
    trace = _normal(Add(*[A[i][i] for i in range(len(A))]))
    integral = attempt(lambda: integrate(trace, x, conds='none'), settings.timeout)
    if integral is not None and not integral.has(Integral, Piecewise, exp_polar):
        abel = as_expr(exp(as_expr(integral)))
        ratio = attempt(lambda: as_expr(simplify(determinant/abel)), settings.timeout)
        if ratio is not None and ratio != 0 and not ratio.has(x):
            return as_expr(ratio*abel)
    simplified = attempt(lambda: as_expr(simplify(determinant)), settings.timeout)
    return determinant if simplified is None else simplified


def _rational_integral(f: Expr, x: Symbol) -> Expr:
    """An antiderivative of a rational function: the polynomial and
    rational parts (Hermite's reduction) and the logarithmic part of
    Rothstein and Trager, with the roots of its factors of degree at most
    two written out and a ``RootSum`` for the others (SymPy's
    ``integrate`` writes them in radicals, nested ones for quartics)."""
    numerator, denominator = _normal(f).as_numer_denom()
    P = Poly(numerator, x)
    Q = Poly(denominator, x)
    quotient, remainder = P.div(Q)
    result = as_expr(quotient.integrate().as_expr())
    if remainder.is_zero:
        return result
    rational, logarithmic = ratint_ratpart(remainder, Q, x)
    result = as_expr(result + rational)
    top, bottom = _normal(as_expr(logarithmic)).as_numer_denom()
    if top == 0:
        return result
    t = Dummy('t')
    for argument, resultant_poly in ratint_logpart(Poly(top, x), Poly(bottom, x), x, t):
        S_ = as_expr(argument.as_expr())
        _, factors = factor_list(as_expr(resultant_poly.as_expr()), t)
        for factor, _ in factors:
            F = Poly(factor, t)
            if F.degree() == 0:
                continue
            if F.degree() <= 2:
                for a in roots(F):
                    result = as_expr(result + a*log(as_expr(S_.subs(t, a))))
            else:
                result = as_expr(result + RootSum(F, Lambda(t, t*log(S_))))
    return result


def _variation_of_constants(A: Rows, fundamental: MutableDenseMatrix, b: list[Expr], x: Symbol) -> list[Expr]:
    """``Phi Integral(Phi**-1 b)``, the components of ``Phi**-1 b`` by
    Cramer's rule."""
    from .factorization import _quadrature
    n = len(A)
    W = _liouville(A, as_expr(fundamental.det(method='berkowitz')), x)
    if W == 0:
        raise ValueError("the fundamental matrix is singular")
    total: list[Expr] = [S.Zero]*n
    for j in range(n):
        replaced = fundamental.copy()
        for i in range(n):
            replaced[i, j] = b[i]
        integrand = as_expr(as_expr(replaced.det(method='berkowitz'))/W)
        if integrand.is_rational_function(x):
            c = _rational_integral(integrand, x)
        elif integrand.has(Integral):
            # an integrand with the unevaluated integrals of a fundamental
            # matrix: SymPy does not integrate it, and trying takes the
            # time limit
            c = as_expr(Integral(integrand, x))
        else:
            c = _quadrature(integrand, x)
        for i in range(n):
            total[i] = as_expr(total[i] + as_expr(fundamental[i, j])*c)
    result: list[Expr] = []
    for e in total:
        result.append(_simplified(e))
    return result


class LinearSystemSolution:
    """The solutions of ``Y' = A Y + b`` found by :func:`dsolve_system`.

    ``fundamental`` is an ``n x k`` matrix whose columns are independent
    solutions of the homogeneous system, ``particular`` a solution of the
    inhomogeneous one (the zero vector for a homogeneous system, ``None``
    when none was found), and ``complete`` is ``True`` when ``k = n`` and
    the particular solution was found, so that every solution is
    ``fundamental * C + particular``, and ``None`` otherwise (closed forms
    may exist which the solvers did not find). ``reduction`` is the
    cyclic vector reduction used.

    Examples
    ========

    >>> from sympy import Matrix, symbols
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import dsolve_system
    >>> S = dsolve_system(Matrix([[0, 1], [2/x**2, 0]]), x)
    >>> S.complete
    True
    >>> S.general_solution(symbols('C1 C2'))
    Matrix([[C1/x + C2*x**2], [-C1/x**2 + 2*C2*x]])
    """

    def __init__(self, fundamental: MutableDenseMatrix, particular: Optional[MutableDenseMatrix],
                 complete: Truth, reduction: CyclicReduction) -> None:
        self.fundamental = fundamental
        self.particular = particular
        self.complete = complete
        self.reduction = reduction

    @property
    def rank(self) -> int:
        """The number of independent homogeneous solutions found."""
        return int(self.fundamental.cols)

    def columns(self) -> list[MutableDenseMatrix]:
        """The independent homogeneous solutions."""
        return [Matrix(self.fundamental[:, j]) for j in range(self.rank)]

    def general_solution(self, constants: Sequence[Symbol]) -> MutableDenseMatrix:
        """``fundamental * C + particular`` for the given constants (one per
        column); the particular part is left out when it was not found."""
        if len(constants) != self.rank:
            raise ValueError("%d constants are needed" % self.rank)
        result = Matrix.zeros(self.fundamental.rows, 1)
        for j, c in enumerate(constants):
            result += c*Matrix(self.fundamental[:, j])
        if self.particular is not None:
            result += self.particular
        return result

    def __repr__(self) -> str:
        return "LinearSystemSolution(%s, %s, %s)" % (self.fundamental.tolist(),
            None if self.particular is None else self.particular.tolist(), self.complete)


def dsolve_system(A: MutableDenseMatrix, x: Symbol, b: Optional[MutableDenseMatrix] = None
                  ) -> LinearSystemSolution:
    """The solutions of ``Y' = A Y + b`` with ``A`` rational in ``x``: a fundamental matrix from the scalar equation of a cyclic vector
    (solved by rational and hyperexponential solutions, Beke's
    factorisation, Kovacic's algorithm and special functions, see
    :func:`~.dsolve_linear`), then SymPy's ``linodesolve``; a rational
    particular solution by Barkatou's method on the augmented system, or
    one by variation of constants.

    Parameters
    ==========

    A : the ``n x n`` matrix of the system
    x : the independent variable
    b : the inhomogeneous term, a column of length ``n`` (optional, any
        functions of ``x``; the rational particular solutions are looked
        for when it is rational)

    Returns
    =======

    A :class:`LinearSystemSolution`, whose ``complete`` flag says whether
    it gives every solution.

    Examples
    ========

    >>> from sympy import Matrix, symbols
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import dsolve_system
    >>> S = dsolve_system(Matrix([[1/x, 1], [0, 1/x]]), x, Matrix([1, 0]))
    >>> S.fundamental
    Matrix([[x, x**2], [0, x]])
    >>> S.particular.T
    Matrix([[x*log(x), 0]])
    >>> S.complete
    True
    """
    rows = _entries(A, x)
    n = len(rows)
    vector: Optional[list[Expr]] = None
    if b is not None:
        vector = _column(b, n, x)
        if all(e == 0 for e in vector):
            vector = None
    R = _reduction(rows, x)
    scalar = _solve_operator(R.operator, Function('y')(x), True, True)
    columns = [R.to_system(u) for u in scalar]
    if len(columns) < n:
        found = _sympy_columns(rows, x)
        if len(found) > len(columns):
            columns = [Matrix(c) for c in found]
    fundamental = Matrix.hstack(*columns) if columns else Matrix.zeros(n, 0)
    particular: Optional[MutableDenseMatrix] = Matrix.zeros(n, 1)
    if vector is not None:
        rational = _rational_particular(rows, vector, x) if all(e.is_rational_function(x) for e in vector) \
            else None
        if rational is not None:
            particular = Matrix(rational)
        elif len(columns) == n:
            particular = Matrix(_variation_of_constants(rows, fundamental, vector, x))
        else:
            particular = None
    complete: Truth = True if len(columns) == n and particular is not None else None
    return LinearSystemSolution(fundamental, particular, complete, R)


def dsolve_linear_system(A: MutableDenseMatrix, x: Symbol) -> list[MutableDenseMatrix]:
    """Independent solution vectors of ``Y' = A Y`` (the columns of the
    fundamental matrix of :func:`dsolve_system`).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_systems import dsolve_linear_system
    >>> sorted((list(Y) for Y in dsolve_linear_system(Matrix([[0, 1], [-1, 0]]), x)), key=str)
    [[exp(-I*x), -I*exp(-I*x)], [exp(I*x), I*exp(I*x)]]
    >>> dsolve_linear_system(Matrix([[1/x, 1], [0, 1/x]]), x)
    [Matrix([
    [x],
    [0]]), Matrix([
    [x**2],
    [   x]])]
    """
    return dsolve_system(A, x).columns()


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
full basis, by the matrix exponential for constant coefficients
(Putzer's algorithm, :func:`~sympy_extras.solvers.dae.matrix_exponential`)
and by SymPy's ``linodesolve`` (systems with a commuting
antiderivative), whose columns are checked.

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

The module works on the objects of the core and imports no SymPy: the
matrices are :class:`~sympy_extras.matrices.dense.ExprMatrix` of core
expressions, the scalar equations the native
:class:`~sympy_extras.solvers.linear_ode.LinearOperator` and its solvers
(:mod:`~sympy_extras.solvers.linear_ode`,
:mod:`~sympy_extras.solvers.factorization`), and the polynomials, their
factors and roots, the linear algebra (inverses, null spaces,
determinants) and the simplifications are the native ones;
:mod:`sympy_extras.solvers.linear_systems_sympy` is its face on SymPy
matrices. The antiderivatives of rational functions are the native
ones (``ratint_ratpart`` and ``ratint_logpart`` of
:mod:`sympy_extras.integrals.risch.rationaltools`, with the native
``RootSum`` of :mod:`sympy_extras.polys.rootoftools`). The antiderivative of
the trace in Liouville's formula comes from the native indefinite
integration (:func:`~sympy_extras.integrals.indefinite.closed_antiderivative`), the
quadratures it cannot do are left unevaluated
(:class:`~sympy_extras.integrals.integral.Integral`), and the last
resort for a fundamental matrix, SymPy's ``linodesolve`` of a matrix
``A = g'(x) M`` with ``M`` constant (its type 3), is native:
``exp(M g(x))`` by Putzer's algorithm
(:func:`~sympy_extras.solvers.dae.matrix_exponential`).

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

Examples
========

>>> import sympy_extras.interop  # SymPy's integrate (the *_of_core functions)
>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices.dense import ExprMatrix
>>> from sympy_extras.solvers.linear_systems import dsolve_system
>>> x = Symbol('x')
>>> dsolve_system(ExprMatrix.from_rows([[0, 1], [2/x**2, 0]]), x).fundamental
ExprMatrix([[1/x, x**2], [-1/x**2, 2*x]])
"""
from __future__ import annotations

from fractions import Fraction
from random import Random
from typing import Final, Optional, Sequence

from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import Function, count_ops, expand
from sympy_extras.core.numbers import Integer
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import ExpPolar, exp
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.simplify.simplify import simplify
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

from ._special_values import determinant
from .linear_ode import (_FIELD_FAILURES, LinearOperator, UPoly, _RationalRows, _factor_list, _indicial_at,
                         _indicial_at_infinity, _inverse, _lcm, _normal, _nullspace, _replaced, _resultant,
                         _solve_operator, _sum, has_integral, hyperexponential_search, log_derivative)

__all__ = ['CyclicReduction', 'LinearSystemSolution', 'cyclic_vector', 'cyclic_reduction', 'system_to_scalar',
           'rational_system_solutions', 'hyperexponential_system_solutions', 'dsolve_system',
           'dsolve_linear_system', 'rational_integral']

#: the rows of a matrix of rational functions
Rows = list[list[Expr]]

#: a truth value of three states (``None`` unknown)
Truth = Optional[bool]

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

#: the seed of the generator of candidate cyclic vectors (fixed, so that
#: the scalar equation, and the form of the solutions, are reproducible)
_CYCLIC_SEED: Final[int] = 2002
#: the number of random candidates tried after the unit vectors
_CYCLIC_TRIALS: Final[int] = 24


def _entries(A: ExprMatrix, x: Symbol) -> Rows:
    """The entries of the square matrix ``A``, checked to be rational in
    ``x`` and normalised."""
    if A.rows != A.cols or A.rows == 0:
        raise ValueError("the matrix of the system must be square and nonempty")
    rows: Rows = []
    for i in range(A.rows):
        row: list[Expr] = []
        for j in range(A.cols):
            e = A[i, j]
            if e.is_rational_function(x) is not True:
                raise ValueError("the entry %s is not a rational function of %s" % (e, x))
            row.append(_normal(e))
        rows.append(row)
    return rows


def _column(b: ExprMatrix, n: int) -> list[Expr]:
    if b.shape != (n, 1):
        raise ValueError("the inhomogeneous term must be a column of length %d" % n)
    return list(b.entries)


def _matrix(rows: Rows) -> ExprMatrix:
    return ExprMatrix.from_rows(rows)


def _columns_matrix(columns: Sequence[ExprMatrix], n: int) -> ExprMatrix:
    """The columns side by side (``n`` rows, possibly no column)."""
    return ExprMatrix(n, len(columns), [c[i, 0] for i in range(n) for c in columns])


def _next_row(A: Rows, c: list[Expr], x: Symbol) -> list[Expr]:
    """``c A + c'``: the row of ``u^(k+1)`` from that of ``u^(k)``."""
    n = len(c)
    return [_normal(c[j].diff(x) + _sum([c[i]*A[i][j] for i in range(n) if c[i] != _ZERO and A[i][j] != _ZERO]))
            for j in range(n)]


def _candidates(n: int, x: Symbol) -> list[list[Expr]]:
    """The unit vectors, then random vectors: constant entries first,
    then polynomial entries of degree less than ``n`` (a generic vector of
    this kind is cyclic, Katz, Churchill–Kovacic); the generator is
    seeded so that the choice is reproducible."""
    candidates: list[list[Expr]] = [[_ONE if j == i else _ZERO for j in range(n)] for i in range(n)]
    generator = Random(_CYCLIC_SEED)
    for trial in range(_CYCLIC_TRIALS):
        degree = 0 if trial < _CYCLIC_TRIALS//3 else min(n - 1, 1 + trial % n)
        vector = [_sum([Integer(generator.randint(-3, 3))*x**k if k else Integer(generator.randint(-3, 3))
                        for k in range(degree + 1)]) for _ in range(n)]
        if any(v != _ZERO for v in vector):
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

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.linear_systems import cyclic_reduction
    >>> x = Symbol('x')
    >>> R = cyclic_reduction(ExprMatrix.from_rows([[0, 1], [2/x**2, 0]]), x)
    >>> R.operator
    LinearOperator([-2, 0, x**2], x)
    >>> R.to_system(x**2)
    ExprMatrix([[x**2], [2*x]])
    """

    def __init__(self, vector: list[Expr], rows: Rows, inverse: Rows, operator: LinearOperator, x: Symbol
                 ) -> None:
        self.vector: Final[ExprMatrix] = ExprMatrix(1, len(vector), vector)
        self.matrix: Final[ExprMatrix] = _matrix(rows)
        self.operator: Final[LinearOperator] = operator
        self.x: Final[Symbol] = x
        self.inverse: Final[tuple[tuple[Expr, ...], ...]] = tuple(tuple(row) for row in inverse)

    def to_system(self, u: Expr) -> ExprMatrix:
        """The solution ``Y = matrix**-1 (u, ..., u^(n-1))^T`` of the
        system given by a solution ``u`` of the operator. For a
        hyperexponential ``u`` the vector is written as ``u`` times rational
        functions."""
        x = self.x
        inverse = self.inverse
        n = len(inverse)
        rho = log_derivative(u, x) if u.has(x) else _ZERO
        if rho.is_rational_function(x) is True:
            # u^(k) = u R_k with R_0 = 1 and R_{k+1} = R_k' + rho R_k
            ratios: list[Expr] = [_ONE]
            for _ in range(n - 1):
                ratios.append(_normal(ratios[-1].diff(x) + rho*ratios[-1]))
            parts = [_normal(_sum([inverse[i][k]*ratios[k] for k in range(n)])) for i in range(n)]
            if u.is_rational_function(x) is True:
                return ExprMatrix.column([_normal(u*p) for p in parts])
            return ExprMatrix.column([u*p for p in parts])
        derivatives = [u] + [u.diff(x, k) for k in range(1, n)]
        entries: list[Expr] = []
        for i in range(n):
            e = _sum([inverse[i][k]*derivatives[k] for k in range(n) if inverse[i][k] != _ZERO])
            entries.append(_simplified(e))
        return ExprMatrix.column(entries)

    def __repr__(self) -> str:
        return "CyclicReduction(%s, %s)" % ([list(self.vector.entries)], self.operator)


def rational_integral(f: Expr, x: Symbol) -> Expr:
    """An antiderivative of a rational function: the polynomial and
    rational parts (Hermite's reduction) and the logarithmic part of
    Lazard, Rioboo and Trager, with the roots of its factors of degree at
    most two written out and a ``RootSum`` for the others (SymPy's
    ``integrate`` writes them in radicals, nested ones for quartics).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.solvers.linear_systems import rational_integral
    >>> x = symbols('x')
    >>> rational_integral(1/(x**3 - x - 1) + 1/x**2, x)
    RootSum(23*_t**3 + 3*_t - 1, Lambda(_t, _t*log(-46*_t**2/9 - 23*_t/9 + x - 4/9))) - 1/x
    """
    from sympy_extras.core.function import Lambda
    from sympy_extras.functions.elementary.exponential import log
    from sympy_extras.integrals.risch.rationaltools import polynomial_part, ratint_logpart, ratint_ratpart
    from sympy_extras.polys.polyerrors import BasePolynomialError
    from sympy_extras.polys.polyroots import roots
    from sympy_extras.polys.polytools import cancel, factor_list
    from sympy_extras.polys.rootoftools import polynomial_coefficients, root_sum
    failures = (BasePolynomialError, NotImplementedError, ValueError, TypeError)
    numerator, denominator = cancel(f).as_numer_denom()
    result, remainder = polynomial_part(numerator, denominator, x)
    if remainder == _ZERO:
        return result
    rational, logarithmic = ratint_ratpart(remainder, denominator, x)
    result = result + rational
    top, bottom = cancel(logarithmic).as_numer_denom()
    if top == _ZERO:
        return result
    t = Dummy('t')
    for argument, resultant_poly in ratint_logpart(top, bottom, x, t):
        try:
            _, found = factor_list(resultant_poly, t)
            factors = [g for g, _ in found]
        except failures:
            factors = [resultant_poly]
        for factor in factors:
            degree = len(polynomial_coefficients(factor, t)) - 1
            if degree <= 0:
                continue
            zeros: Optional[dict[Expr, int]] = None
            if degree <= 2:
                try:
                    listed = roots(factor, t)
                    zeros = listed if isinstance(listed, dict) else None
                except failures:
                    zeros = None
            if zeros is not None and sum(zeros.values()) == degree:
                for a in zeros:
                    result = result + a*log(argument.subs(t, a))
            else:
                result = result + root_sum(factor, Lambda(t, t*log(argument)), t)
    return result


def _has_foreign(e: Basic, names: tuple[str, ...]) -> bool:
    """Whether ``e`` has a node of SymPy (wrapped in the core) of one of
    the classes ``names``."""
    return any(node._is_foreign_node() and node._compare_name() in names for node in preorder_traversal(e))


def _ops(e: Expr) -> int:
    count = count_ops(e)
    return int(count) if isinstance(count, Integer) else 0


def _simplified(e: Expr) -> Expr:
    """``simplify(e)`` within the time limit, except for the expressions
    with integrals, which simplify evaluates (into polylogarithms of
    ``exp_polar`` arguments, which the quadratures had refused), and with
    ``RootSum``s, which it writes in radicals."""
    from sympy_extras.polys.rootoftools import RootSum
    if _ops(e) >= 400 or e.has(RootSum) or has_integral(e) or _has_foreign(e, ('RootSum',)):
        return e
    simplified = attempt(lambda: simplify(e), settings.timeout, native=True)
    if simplified is None or simplified.has(ExpPolar, Piecewise) \
            or _has_foreign(simplified, ('exp_polar', 'Piecewise')):
        return e
    return simplified


def _field_inverse(field: SparseFractionField[Fraction], rows: list[list[SparseFracElement[Fraction]]]
                   ) -> Optional[list[list[SparseFracElement[Fraction]]]]:
    """The inverse of a square matrix over the field (the reduced row
    echelon form of ``[M | 1]``), ``None`` when it is singular."""
    n = len(rows)
    sparse: list[dict[int, SparseFracElement[Fraction]]] = []
    for i, row in enumerate(rows):
        entries = {j: c for j, c in enumerate(row) if not field.is_zero(c)}
        entries[n + i] = field.one
        sparse.append(entries)
    echelon = field.sparse_echelon(sparse)
    pivot_rows = {min(r): r for r in echelon}
    if sorted(pivot_rows)[:n] != list(range(n)):
        return None
    return [[pivot_rows[i].get(n + j, field.zero) for j in range(n)] for i in range(n)]


def _field_reduction(A: Rows, x: Symbol) -> Optional[CyclicReduction]:
    """:func:`_reduction` with the rows ``c_k``, the inverse and the
    coefficients of the scalar equation computed in the native field of
    rational functions of the symbols (much faster than normalising
    every entry), ``None`` when an entry is not a rational function with
    rational coefficients."""
    n = len(A)
    try:
        matrix = _RationalRows(A + [[x]])
    except _FIELD_FAILURES:
        return None
    field = matrix.field
    variables = list(matrix.restored.values())
    index = variables.index(x)
    renamed: dict[Basic, Basic] = {s: name for name, s in matrix.restored.items()}
    entries = matrix.rows[:n]

    def following(c: list[SparseFracElement[Fraction]]) -> list[SparseFracElement[Fraction]]:
        result: list[SparseFracElement[Fraction]] = []
        for j in range(n):
            value = field.diff(c[j], index)
            for i in range(n):
                if not field.is_zero(c[i]) and not field.is_zero(entries[i][j]):
                    value = field.add(value, field.mul(c[i], entries[i][j]))
            result.append(value)
        return result

    for c0 in _candidates(n, x):
        rows = [[field.from_expr(_replaced(e, renamed)) for e in c0]]
        for _ in range(n - 1):
            rows.append(following(rows[-1]))
        inverse = _field_inverse(field, rows)
        if inverse is None:
            continue
        last = following(rows[-1])
        # c_n = sum_k a_k c_k with (a_k) = c_n M**-1
        a: list[SparseFracElement[Fraction]] = []
        for k in range(n):
            value = field.zero
            for j in range(n):
                if not field.is_zero(last[j]):
                    value = field.add(value, field.mul(last[j], inverse[j][k]))
            a.append(value)
        operator = LinearOperator([_normal(-matrix.expr(e)) for e in a] + [_ONE], x).primitive()
        return CyclicReduction(c0, [[_normal(matrix.expr(e)) for e in row] for row in rows],
                               [[_normal(matrix.expr(e)) for e in row] for row in inverse], operator, x)
    raise ValueError("no cyclic vector was found among the candidates")


def _reduction(A: Rows, x: Symbol) -> CyclicReduction:
    reduction = _field_reduction(A, x)
    if reduction is not None:
        return reduction
    n = len(A)
    for c0 in _candidates(n, x):
        rows = [c0]
        for _ in range(n - 1):
            rows.append(_next_row(A, rows[-1], x))
        inverse = _inverse(rows)
        if inverse is None:
            continue
        following = _next_row(A, rows[-1], x)
        # c_n = sum_k a_k c_k with (a_k) = c_n M**-1
        a = [_normal(_sum([following[j]*inverse[j][k] for j in range(n) if following[j] != _ZERO]))
             for k in range(n)]
        operator = LinearOperator([_normal(-e) for e in a] + [_ONE], x).primitive()
        return CyclicReduction(c0, rows, inverse, operator, x)
    raise ValueError("no cyclic vector was found among the candidates")


def cyclic_reduction(A: ExprMatrix, x: Symbol) -> CyclicReduction:
    """The scalar equation of a cyclic vector of ``Y' = A Y`` with the
    matrices mapping its solutions to those of the system.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.linear_systems import cyclic_reduction
    >>> x = Symbol('x')
    >>> R = cyclic_reduction(ExprMatrix.from_rows([[1, 0], [0, 2]]), x)
    >>> R.vector, R.operator
    (ExprMatrix([[2, -3]]), LinearOperator([2, -3, 1], x))

    The unit vectors are not cyclic for this diagonal system: ``y1`` alone
    satisfies ``y1' = y1``.
    """
    return _reduction(_entries(A, x), x)


def cyclic_vector(A: ExprMatrix, x: Symbol) -> tuple[ExprMatrix, LinearOperator]:
    """A cyclic row vector ``c_0`` for ``Y' = A Y`` and the scalar operator
    satisfied by ``u = c_0 Y``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.linear_systems import cyclic_vector
    >>> x = Symbol('x')
    >>> c, L = cyclic_vector(ExprMatrix.from_rows([[0, 1], [-1, 0]]), x)
    >>> c, L.coefficients
    (ExprMatrix([[1, 0]]), (1, 0, 1))
    """
    R = cyclic_reduction(A, x)
    return R.vector, R.operator


def system_to_scalar(A: ExprMatrix, x: Symbol) -> tuple[LinearOperator, ExprMatrix]:
    """The scalar operator of a cyclic vector and the matrix ``M`` with
    ``(u, u', ..., u**(n-1))**T = M Y``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.linear_systems import system_to_scalar
    >>> x = Symbol('x')
    >>> system_to_scalar(ExprMatrix.from_rows([[1/x, 1], [0, 1/x]]), x)
    (LinearOperator([2, -2*x, x**2], x), ExprMatrix([[1, 0], [1/x, 1]]))
    """
    R = cyclic_reduction(A, x)
    return R.operator, R.matrix


# ---------------------------------------------------------------------------
# rational solutions

def _integer_roots(p: Expr, variable: Symbol) -> list[int]:
    """The integer roots of a polynomial in ``variable`` (whose other
    symbols are parameters: a root depending on them is not an
    integer)."""
    found: list[int] = []
    for f, _ in _factor_list(p, variable):
        try:
            q = UPoly.of(f, variable)
        except ValueError:
            continue
        if q.degree() != 1:
            continue
        leading, constant = q.all_coeffs()
        root = _normal(-constant/leading)
        if isinstance(root, Integer):
            found.append(int(root))
    return sorted(set(found))


def _multiplicity(q: UPoly, p: UPoly) -> int:
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
    if e == _ZERO:
        return None
    numerator, denominator = _normal(e).as_numer_denom()
    return UPoly.of(numerator, x).degree() - UPoly.of(denominator, x).degree()


def _at_infinity(e: Expr, x: Symbol) -> Expr:
    """The limit at infinity of a rational function of degree at most
    zero (``deg numerator <= deg denominator``)."""
    numerator, denominator = _normal(e).as_numer_denom()
    top, bottom = UPoly.of(numerator, x), UPoly.of(denominator, x)
    if top.is_zero or top.degree() < bottom.degree():
        return _ZERO
    if top.degree() > bottom.degree():
        raise ValueError("%s has no finite limit at infinity" % (e,))
    return _normal(top.LC()/bottom.LC())


class _ScalarBounds:
    """The bounds for the rational solutions of a system taken from the
    scalar equation of a cyclic vector, built when first needed."""

    def __init__(self, A: Rows, x: Symbol) -> None:
        self.A: Final[Rows] = A
        self.x: Final[Symbol] = x
        self._reduction: Optional[CyclicReduction] = None

    def reduction(self) -> CyclicReduction:
        if self._reduction is None:
            self._reduction = _reduction(self.A, self.x)
        return self._reduction

    def pole(self, p: UPoly) -> int:
        """A bound for the order of the pole at the roots of ``p`` of a
        rational solution."""
        R = self.reduction()
        L = R.operator
        x = self.x
        m = 0
        if _multiplicity(UPoly.of(L.leading_coefficient, x), p) > 0:
            indicial = _indicial_at(L, p)
            negative = [r for r in _integer_roots(indicial.as_expr(), indicial.x) if r < 0]
            if negative:
                m = -negative[0]
        n = len(self.A)
        bound = 0
        for i in range(n):
            for k in range(n):
                e = R.inverse[i][k]
                if e == _ZERO:
                    continue
                denominator = UPoly.of(_normal(e).as_numer_denom()[1], x)
                # the pole of u^(k) is at most m + k when u has one
                bound = max(bound, _multiplicity(denominator, p) + (m + k if m > 0 else 0))
        return bound

    def degree(self) -> Optional[int]:
        """A bound for ``deg numerator - deg denominator`` of the components
        of a rational solution, ``None`` when there is none."""
        R = self.reduction()
        indicial = _indicial_at_infinity(R.operator)
        exponents = _integer_roots(indicial.as_expr(), indicial.x)
        if not exponents:
            return None
        delta = exponents[-1]
        n = len(self.A)
        bound: Optional[int] = None
        for i in range(n):
            for k in range(n):
                d = _degree(R.inverse[i][k], self.x)
                if d is not None:
                    bound = d + delta - k if bound is None else max(bound, d + delta - k)
        return bound


def _determinant(rows: Rows) -> Expr:
    return expand(determinant(rows))


def _characteristic(rows: Rows, lam: Symbol, scale: Expr) -> Expr:
    """``det(lam scale I - rows)``, expanded."""
    n = len(rows)
    return _determinant([[lam*scale - rows[i][j] if i == j else -rows[i][j] for j in range(n)] for i in range(n)])


def _residue_bound(N: Rows, d: UPoly, p: UPoly, x: Symbol) -> int:
    """The pole order bound at a simple pole: minus the least negative
    integer eigenvalue of the residue ``N(c)/(p'(c) q(c))``, ``d = p q``,
    at the roots ``c`` of ``p``."""
    lam = Dummy('lambda')
    q = d.div(p)[0]
    scale = expand(p.diff().as_expr()*q.as_expr())
    characteristic = _characteristic(N, lam, scale)
    if p.degree() == 1:
        leading, constant = p.all_coeffs()
        eigen = expand(_replaced(characteristic, {x: _normal(-constant/leading)}))
    else:
        eigen = _resultant(characteristic, p.as_expr(), x)
    negative = [r for r in _integer_roots(eigen, lam) if r < 0]
    return -negative[0] if negative else 0


def _rational_basis(A: Rows, x: Symbol) -> Rows:
    """A basis of the rational solutions of ``Y' = A Y`` (Barkatou)."""
    n = len(A)
    parts = [[_normal(e).as_numer_denom() for e in row] for row in A]
    d_expr = _lcm([den for row in parts for _, den in row])
    d = UPoly.of(d_expr, x)
    N = [[_normal(num*d_expr/den) for num, den in row] for row in parts]
    scalar = _ScalarBounds(A, x)
    # the universal denominator
    D: Expr = _ONE
    determinant_: Optional[UPoly] = None
    for f, e in _factor_list(d_expr, x):
        p = UPoly.of(f, x)
        if p.degree() <= 0:
            continue
        if e == 1:
            m = _residue_bound(N, d, p, x)
        else:
            if determinant_ is None:
                determinant_ = UPoly.of(_determinant(N), x)
            # a zero determinant is divisible by every p (the multiplicity
            # of the zero polynomial came out 0, and the augmented systems
            # of variation of constants, whose last row is zero, never had
            # a rational solution at a double pole)
            if not determinant_.is_zero and _multiplicity(determinant_, p) == 0:
                # the leading matrix at the pole is invertible: no Laurent
                # solution, so no rational one
                return []
            m = scalar.pole(p)
        D = D*p.as_expr()**m if m else D
    D = expand(D)
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
        residue = [[_at_infinity(x*A[i][j], x) if shift == 1 else _ZERO for j in range(n)] for i in range(n)]
        exponents = _integer_roots(_characteristic(residue, lam, _ONE), lam)
        delta = exponents[-1] if exponents else None
    else:
        # A = O(x**q), q = -shift >= 0: the leading matrix
        q = -shift
        leading = [[_at_infinity(A[i][j]/x**q, x) for j in range(n)] for i in range(n)]
        if _normal(_determinant(leading)) != _ZERO:
            return []
        delta = scalar.degree()
    if delta is None:
        return []
    top = UPoly.of(D, x).degree() + delta
    if top < 0:
        return []
    return _numerators(A, N, d_expr, D, top, x)


def _numerators(A: Rows, N: Rows, d: Expr, D: Expr, top: int, x: Symbol) -> Rows:
    """The solutions ``P/D`` with ``deg P <= top``: the kernel of the
    linear map ``P -> d (D P' - D' P) - D N P``."""
    n = len(A)
    dD = expand(d*D)
    dDprime = expand(d*D.diff(x))
    columns: list[list[UPoly]] = []
    for i in range(n):
        for j in range(top + 1):
            monomial = x**j if j else _ONE
            image: list[UPoly] = []
            for k in range(n):
                value = -D*N[k][i]*monomial
                if k == i:
                    value = value + (dD*j*x**(j - 1) if j else _ZERO) - dDprime*monomial
                image.append(UPoly.of(expand(value), x))
            columns.append(image)
    size = max((p.degree() for column in columns for p in column if not p.is_zero), default=0) + 1
    rows: Rows = []
    for k in range(n):
        for power in range(size):
            row = [column[k].coeff(power) for column in columns]
            if any(e != _ZERO for e in row):
                rows.append(row)
    basis: Rows = []
    for vector in _nullspace(rows, len(columns)):
        P = [_sum([vector[i*(top + 1) + j]*x**j if j else vector[i*(top + 1)] for j in range(top + 1)])
             for i in range(n)]
        basis.append([_normal(p/D) for p in P])
    return basis


def rational_system_solutions(A: ExprMatrix, x: Symbol) -> list[ExprMatrix]:
    """A basis of the rational solutions of ``Y' = A Y``: a universal
    denominator from the finite singularities, a degree bound at
    infinity, and a linear system for the numerators (Barkatou's method;
    see the module docstring for the bounds at poles of order two and
    more).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.linear_systems import rational_system_solutions
    >>> x = Symbol('x')
    >>> rational_system_solutions(ExprMatrix.from_rows([[1/x, 1], [0, 1/x]]), x)
    [ExprMatrix([[x], [0]]), ExprMatrix([[x**2], [x]])]
    >>> rational_system_solutions(ExprMatrix.from_rows([[-1/x, 0], [1, 0]]), x)
    [ExprMatrix([[0], [1]])]
    """
    return [ExprMatrix.column(v) for v in _rational_basis(_entries(A, x), x)]


def _augmented(A: Rows, b: list[Expr]) -> Rows:
    n = len(A)
    rows = [A[i] + [_normal(b[i])] for i in range(n)]
    rows.append([_ZERO]*(n + 1))
    return rows


def _rational_particular(A: Rows, b: list[Expr], x: Symbol) -> Optional[list[Expr]]:
    """A rational solution of ``Y' = A Y + b``: a rational solution of the
    augmented system whose last component (a constant) is not zero."""
    n = len(A)
    for vector in _rational_basis(_augmented(A, b), x):
        z = vector[n]
        if z != _ZERO and not z.has(x):
            return [_normal(vector[i]/z) for i in range(n)]
    return None


# ---------------------------------------------------------------------------
# hyperexponential and general solutions

def hyperexponential_system_solutions(A: ExprMatrix, x: Symbol) -> list[ExprMatrix]:
    """Independent solutions ``h R`` of ``Y' = A Y`` with ``h'/h`` and the
    vector ``R`` rational, from the hyperexponential solutions of the
    scalar equation of a cyclic vector (complete when that search is, see
    :func:`~.hyperexponential_search`).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.linear_systems import hyperexponential_system_solutions
    >>> x = Symbol('x')
    >>> hyperexponential_system_solutions(ExprMatrix.from_rows([[1, 1/x], [0, 0]]), x)
    [ExprMatrix([[exp(x)], [0]])]
    """
    R = cyclic_reduction(A, x)
    found, _ = hyperexponential_search(R.operator)
    return [R.to_system(u) for u in found]


def _verified(A: Rows, Y: list[Expr], x: Symbol) -> bool:
    """Whether ``Y' = A Y``, by simplification within the time limit."""
    n = len(A)
    for i in range(n):
        residual = Y[i].diff(x) - _sum([A[i][j]*Y[j] for j in range(n)])
        zero = attempt(lambda: simplify(residual) == _ZERO, settings.timeout, native=True)
        if not zero:
            return False
    return True


def _is_constant(A: Rows, x: Symbol) -> bool:
    return all(not e.has(x) for row in A for e in row)


def _exponential_columns(A: Rows, x: Symbol) -> Rows:
    """The columns of ``exp(A x)`` for a constant matrix (Putzer's
    algorithm), none when its eigenvalues are not found."""
    from .dae import matrix_exponential
    n = len(A)
    try:
        Phi = matrix_exponential(_matrix(A), x)
    except NotImplementedError:
        return []
    return [[Phi[i, j] for i in range(n)] for j in range(n)]


def _commuting_columns(A: Rows, x: Symbol) -> Rows:
    """The fundamental solutions ``exp(M g(x))`` of ``Y' = A Y`` when ``A =
    g'(x) M`` with ``M`` constant (``A`` commutes with its antiderivative:
    SymPy's ``linodesolve`` of type 3), each checked; none when ``A`` is
    not of that form, the antiderivative or the eigenvalues of ``M`` are
    not found."""
    from sympy_extras.integrals.indefinite import closed_antiderivative
    from .dae import matrix_exponential
    from .linear_ode import _normal
    n = len(A)
    pivots = [e for row in A for e in row if e != _ZERO]
    if not pivots or all(not e.has(x) for e in pivots):
        return []
    pivot = pivots[0]
    M = [[_normal(e/pivot) for e in row] for row in A]
    if any(e.has(x) for row in M for e in row):
        return []
    g = closed_antiderivative(pivot, x, closed=True)
    if g is None:
        return []
    s = Dummy('s')
    try:
        Phi = matrix_exponential(_matrix(M), s)
    except NotImplementedError:
        return []
    columns: Rows = []
    for j in range(n):
        column = [_simplified(_replaced_value(Phi[i, j], s, g)) for i in range(n)]
        if _verified(A, column, x):
            columns.append(column)
    return columns if len(columns) == n else []


def _replaced_value(e: Expr, s: Symbol, value: Expr) -> Expr:
    replaced = e._xreplace_evaluated({s: value})
    if not isinstance(replaced, Expr):
        raise TypeError("%s is not an expression" % (replaced,))
    return replaced


def _liouville(A: Rows, determinant_: Expr, x: Symbol) -> Expr:
    """``det Phi`` as ``C exp(Integral(trace A))`` when the ratio is found
    to be constant, else the simplified determinant."""
    from sympy_extras.integrals.indefinite import closed_antiderivative
    trace = _normal(_sum([A[i][i] for i in range(len(A))]))
    integral = attempt(lambda: closed_antiderivative(trace, x, closed=True), settings.timeout, native=True)
    if integral is not None:
        abel = exp(integral)
        ratio = attempt(lambda: simplify(determinant_/abel), settings.timeout, native=True)
        if ratio is not None and ratio != _ZERO and not ratio.has(x):
            return ratio*abel
    simplified = attempt(lambda: simplify(determinant_), settings.timeout, native=True)
    return determinant_ if simplified is None else simplified


def _variation_of_constants(A: Rows, fundamental: ExprMatrix, b: list[Expr], x: Symbol) -> list[Expr]:
    """``Phi Integral(Phi**-1 b)``, the components of ``Phi**-1 b`` by
    Cramer's rule."""
    from .factorization import _quadrature
    from sympy_extras.integrals.integral import Integral
    n = len(A)
    rows = [list(fundamental.row(i)) for i in range(n)]
    W = _liouville(A, determinant(rows), x)
    if W == _ZERO:
        raise ValueError("the fundamental matrix is singular")
    total: list[Expr] = [_ZERO]*n
    for j in range(n):
        replaced = [[b[i] if k == j else rows[i][k] for k in range(n)] for i in range(n)]
        integrand = determinant(replaced)/W
        if integrand.is_rational_function(x) is True:
            c = rational_integral(integrand, x)
        elif has_integral(integrand):
            # an integrand with the unevaluated integrals of a fundamental
            # matrix: SymPy does not integrate it, and trying takes the
            # time limit
            c = Integral(integrand, x)
        else:
            c = _quadrature(integrand, x)
        for i in range(n):
            total[i] = total[i] + rows[i][j]*c
    return [_simplified(e) for e in total]


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

    >>> import sympy_extras.interop  # SymPy's integrate (the *_of_core functions)
    >>> from sympy_extras.core import Symbol, symbols
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.linear_systems import dsolve_system
    >>> x = Symbol('x')
    >>> S = dsolve_system(ExprMatrix.from_rows([[0, 1], [2/x**2, 0]]), x)
    >>> S.complete
    True
    >>> S.general_solution(symbols('C1 C2'))
    ExprMatrix([[C1/x + C2*x**2], [-C1/x**2 + 2*C2*x]])
    """

    def __init__(self, fundamental: ExprMatrix, particular: Optional[ExprMatrix], complete: Truth,
                 reduction: CyclicReduction) -> None:
        self.fundamental: Final[ExprMatrix] = fundamental
        self.particular: Final[Optional[ExprMatrix]] = particular
        self.complete: Final[Truth] = complete
        self.reduction: Final[CyclicReduction] = reduction

    @property
    def rank(self) -> int:
        """The number of independent homogeneous solutions found."""
        return self.fundamental.cols

    def columns(self) -> list[ExprMatrix]:
        """The independent homogeneous solutions."""
        F = self.fundamental
        return [ExprMatrix.column([F[i, j] for i in range(F.rows)]) for j in range(self.rank)]

    def general_solution(self, constants: Sequence[Expr]) -> ExprMatrix:
        """``fundamental * C + particular`` for the given constants (one per
        column); the particular part is left out when it was not found."""
        if len(constants) != self.rank:
            raise ValueError("%d constants are needed" % self.rank)
        F = self.fundamental
        entries = [_sum([constants[j]*F[i, j] for j in range(self.rank)]) for i in range(F.rows)]
        if self.particular is not None:
            entries = [e + p for e, p in zip(entries, self.particular.entries)]
        return ExprMatrix.column(entries)

    def __repr__(self) -> str:
        F = self.fundamental
        particular = None if self.particular is None else [[e] for e in self.particular.entries]
        return "LinearSystemSolution(%s, %s, %s)" % ([list(F.row(i)) for i in range(F.rows)], particular,
                                                     self.complete)


def dsolve_system(A: ExprMatrix, x: Symbol, b: Optional[ExprMatrix] = None) -> LinearSystemSolution:
    """The solutions of ``Y' = A Y + b`` with ``A`` rational in ``x``: a
    fundamental matrix from the scalar equation of a cyclic vector (solved
    by rational and hyperexponential solutions, Beke's factorisation,
    Kovacic's algorithm and special functions, see
    :func:`~.dsolve_linear`), then the matrix exponential for constant
    coefficients and for ``A = g'(x) M``, ``M`` constant; a rational particular
    solution by Barkatou's method on the augmented system, or one by
    variation of constants.

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

    >>> import sympy_extras.interop  # SymPy's integrate (the *_of_core functions)
    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.linear_systems import dsolve_system
    >>> x = Symbol('x')
    >>> S = dsolve_system(ExprMatrix.from_rows([[1/x, 1], [0, 1/x]]), x, ExprMatrix.column([1, 0]))
    >>> S.fundamental
    ExprMatrix([[x, x**2], [0, x]])
    >>> S.particular
    ExprMatrix([[x*log(x)], [0]])
    >>> S.complete
    True
    """
    rows = _entries(A, x)
    n = len(rows)
    vector: Optional[list[Expr]] = None
    if b is not None:
        vector = _column(b, n)
        if all(e == _ZERO for e in vector):
            vector = None
    R = _reduction(rows, x)
    scalar = _solve_operator(R.operator, Function('y')(x), True, True)
    columns = [R.to_system(u) for u in scalar]
    if len(columns) < n:
        found = _exponential_columns(rows, x) if _is_constant(rows, x) else []
        if len(found) < n:
            found = _commuting_columns(rows, x)
        if len(found) > len(columns):
            columns = [ExprMatrix.column(c) for c in found]
    fundamental = _columns_matrix(columns, n)
    particular: Optional[ExprMatrix] = ExprMatrix.zeros(n, 1)
    if vector is not None:
        rational = _rational_particular(rows, vector, x) \
            if all(e.is_rational_function(x) is True for e in vector) else None
        if rational is not None:
            particular = ExprMatrix.column(rational)
        elif len(columns) == n:
            particular = ExprMatrix.column(_variation_of_constants(rows, fundamental, vector, x))
        else:
            particular = None
    complete: Truth = True if len(columns) == n and particular is not None else None
    return LinearSystemSolution(fundamental, particular, complete, R)


def dsolve_linear_system(A: ExprMatrix, x: Symbol) -> list[ExprMatrix]:
    """Independent solution vectors of ``Y' = A Y`` (the columns of the
    fundamental matrix of :func:`dsolve_system`).

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's integrate (the *_of_core functions)
    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.linear_systems import dsolve_linear_system
    >>> x = Symbol('x')
    >>> dsolve_linear_system(ExprMatrix.from_rows([[0, 1], [-1, 0]]), x)
    [ExprMatrix([[exp(-I*x)], [-I*exp(-I*x)]]), ExprMatrix([[exp(I*x)], [I*exp(I*x)]])]
    >>> dsolve_linear_system(ExprMatrix.from_rows([[1/x, 1], [0, 1/x]]), x)
    [ExprMatrix([[x], [0]]), ExprMatrix([[x**2], [x]])]
    """
    return dsolve_system(A, x).columns()

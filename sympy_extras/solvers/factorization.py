"""Factorisation of linear ordinary differential operators with rational
function coefficients (Beke's algorithm), and the solution of the
equations whose operators factor.

The operators ``L = a_n D^n + ... + a_0`` with ``a_i`` in `K = \\mathbb{Q}(x)`
(or its extension by the constants of the coefficients) form the Ore ring
`K[D]`, `D a = a D + a'`, which has a Euclidean division on the right
(:meth:`~.LinearOperator.right_divmod`). ``R`` is a right factor of ``L``
exactly when the solution space of ``R`` is a subspace of that of ``L``
invariant under the differential Galois group.

**Beke's algorithm.** A right factor of order ``k`` of the monic ``L``,
with solutions `y_1, \\ldots, y_k`, is

.. math:: R(y) = \\frac{\\mathrm{Wr}(y_1, \\ldots, y_k, y)}{\\mathrm{Wr}(y_1, \\ldots, y_k)}
    = \\sum_{i=0}^{k} (-1)^{k-i} \\frac{w_{I_i}}{w_{I_k}} y^{(i)},

where `w_I` is the minor of the Wronskian matrix `(y_j^{(r)})` on the rows
`I` and `I_i = \\{0, \\ldots, k\\} \\setminus \\{i\\}`. The vector `w` of all
the ``binomial(n, k)`` minors satisfies a linear system `w' = A w` (the
derivative of a minor moves one row down, and the row ``n`` is reduced
with ``L``): its solutions are the exterior power `\\wedge^k` of the
solution space of ``L``, and the decomposable ones (those satisfying the
Grassmann–Plücker relations) are the ``k``-dimensional subspaces. The
subspace is invariant, so ``R`` has rational coefficients, exactly when
`w` is hyperexponential. A cyclic vector turns the system into the
scalar *associated equation* (Schwarz's and Bronstein's formulation of
Beke's method): its hyperexponential solutions ``u`` give `w` by solving
the linear system of ``u, u', ...``, and the ratios above are the
coefficients of the candidate ``R``. When several hyperexponential
solutions differ by rational factors, the combination `\\sum c_j w_j` must
be decomposable, a system of quadratic equations in the ``c_j``. Every
candidate is checked by an exact right division.

Left factors of order ``k`` are the adjoints of the right factors of the
adjoint operator `L^* = \\sum (-1)^i D^i a_i`, since `(AB)^* = B^* A^*`;
so the right and left factors of order at most ``n/2`` decide whether
``L`` is reducible.

**Irreducibility** is proven only when every search was exhaustive:
the hyperexponential solutions of the associated equations are found by
:func:`~.hyperexponential_search`, which says when it may have missed some
(irregular finite singular points, irrational exponents). For second
order factors Kovacic's algorithm decides it as well: its case 1 is a
right factor of order one, and the other cases, or no Liouvillian
solution, mean that the operator is irreducible.

**Solving.** If `L = F_1 \\cdots F_m`, the solutions of `F_m` are
solutions of ``L``, and for each solution ``z`` of `F_1 \\cdots F_{m-1}`
a solution of `F_m y = z` found by variation of parameters is one too;
together they are a basis when every factor is solved.

Van Hoeij's local method and the eigenring are not implemented.

References
==========

.. [Beke] E. Beke, Die Irreducibilität der homogenen linearen
   Differentialgleichungen, Mathematische Annalen 45 (1894).
.. [Schwarz] F. Schwarz, A factorization algorithm for linear ordinary
   differential equations, ISSAC 1989.
.. [Bronstein94] M. Bronstein, An improved algorithm for factoring
   linear ordinary differential operators, ISSAC 1994.
.. [Kovacic] J. Kovacic, An algorithm for solving second order linear
   homogeneous differential equations, Journal of Symbolic Computation 2
   (1986).
"""
from __future__ import annotations

from itertools import combinations
from random import Random
from typing import Callable, Optional

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import Function
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.hyperbolic import acosh, acoth, asinh, atanh, cosh, coth, sinh, tanh
from sympy.functions.elementary.trigonometric import acos, acot, asin, atan, cos, cot, sin, tan
from sympy.functions.elementary.piecewise import Piecewise
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices.dense import Matrix
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, resultant
from sympy.polys.rationaltools import together
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve

from sympy_extras._timeout import attempt
from sympy_extras._typing import Truth, as_expr
from sympy_extras.settings import settings

from .linear_ode import (LinearOperator, hyperexponential_search, log_derivative, _exterior_system,
    _scalar_operator, _normal)

__all__ = ['OperatorFactorization', 'right_factor', 'left_factor', 'factor_operator',
           'variation_of_parameters', 'solve_factored']

_ELEMENTARY = (exp, log, sin, cos, tan, cot, sinh, cosh, tanh, coth, asin, acos, atan, acot, asinh, acosh,
               atanh, acoth)

#: a solver of homogeneous equations: a list of independent solutions
Solver = Callable[[LinearOperator], list[Expr]]


class OperatorFactorization:
    """``L = unit * factors[0] * ... * factors[-1]`` with monic factors;
    ``irreducible[i]`` is ``True`` when the ``i``-th factor is proven
    irreducible and ``None`` when a search was not exhaustive.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> from sympy_extras.solvers.factorization import factor_operator
    >>> F = factor_operator(LinearOperator([0, -1, 0, 1], x))
    >>> F
    OperatorFactorization(1, [LinearOperator([-1, 1], x), LinearOperator([1, 1], x), LinearOperator([0, 1], x)], [True, True, True])
    >>> F.expand()
    LinearOperator([0, -1, 0, 1], x)
    """

    def __init__(self, unit: Expr, factors: list[LinearOperator], irreducible: list[Truth], x: Symbol) -> None:
        self.unit = unit
        self.factors = factors
        self.irreducible = irreducible
        self.x = x

    def expand(self) -> LinearOperator:
        """The product ``unit * factors[0] * ... * factors[-1]``."""
        product = LinearOperator([self.unit], self.x)
        for factor in self.factors:
            product = product*factor
        return product

    @property
    def is_complete(self) -> bool:
        """Whether every factor is proven irreducible."""
        return all(flag is True for flag in self.irreducible)

    def __repr__(self) -> str:
        return "OperatorFactorization(%s, %s, %s)" % (self.unit, self.factors, self.irreducible)


# ---------------------------------------------------------------------------
# right factors of order one: hyperexponential solutions and Kovacic

def _first_order_factor(M: LinearOperator) -> tuple[Optional[LinearOperator], bool]:
    """A monic right factor ``D - rho`` of the monic ``M``, and whether
    the search for one was exhaustive."""
    x = M.x
    solutions, complete = hyperexponential_search(M)
    for h in solutions:
        rho = log_derivative(h, x)
        if not rho.is_rational_function(x):
            complete = False
            continue
        R = LinearOperator([as_expr(-rho), S.One], x)
        if M.right_remainder(R).is_zero:
            return R, complete
    if M.order == 2:
        return _kovacic_factor(M, complete)
    return None, complete


def _kovacic_factor(M: LinearOperator, complete: bool) -> tuple[Optional[LinearOperator], bool]:
    """Kovacic's algorithm on the monic ``M = D**2 + p D + q``: case 1
    gives the right factor ``D - (omega - p/2)``; no Liouvillian solution
    or cases 2 and 3 prove ``M`` irreducible (when the poles of the
    normal form were all found)."""
    from .kovacic import liouvillian_solution, normal_form
    x = M.x
    q, p = M.coefficients[0], M.coefficients[1]
    r = normal_form(p, q, x)
    outcome = attempt(lambda: (liouvillian_solution(r, x),), settings.timeout)
    if outcome is None:
        return None, complete
    solution = outcome[0]
    if solution is not None and solution.case == 1 and solution.omega is not None:
        rho = _normal(solution.omega - p/2)
        if rho.is_rational_function(x):
            R = LinearOperator([as_expr(-rho), S.One], x)
            if M.right_remainder(R).is_zero:
                return R, complete
        return None, complete
    denominator = Poly(_normal(r).as_numer_denom()[1], x)
    poles_found = sum(roots(denominator).values()) == denominator.degree()
    return None, complete or poles_found


# ---------------------------------------------------------------------------
# right factors of higher order: Beke's method on the associated equation

def _solve_with_inverse(inverse: list[list[Expr]], vector: list[Expr]) -> list[Expr]:
    return [_normal(Add(*[row[j]*vector[j] for j in range(len(vector)) if row[j] != 0 and vector[j] != 0]))
            for row in inverse]


def _inverse(rows: list[list[Expr]]) -> Optional[list[list[Expr]]]:
    matrix = Matrix(rows)
    try:
        inverse = DomainMatrix.from_Matrix(matrix).to_field().inv().to_Matrix()
    except (CoercionFailed, GeneratorsError, NotImplementedError, ValueError, ZeroDivisionError):
        return None
    return [[_normal(as_expr(inverse[i, j])) for j in range(inverse.cols)] for i in range(inverse.rows)]


def _rational_ratio(d: Expr, x: Symbol) -> tuple[Optional[Expr], bool]:
    """A rational function ``f`` with ``f'/f = d`` (``d`` rational), and
    whether that was decided: ``d`` must be proper with a squarefree
    denominator and integer residues, the roots of the Rothstein-Trager
    resultant ``res_x(p - t q', q)``, and then ``f = prod gcd(p - k q',
    q)**k``."""
    d = _normal(d)
    if d == 0:
        return S.One, True
    numerator, denominator = (Poly(as_expr(e), x) for e in d.as_numer_denom())
    if numerator.degree() >= denominator.degree():
        return None, True
    derivative = denominator.diff(x)
    if denominator.gcd(derivative).degree() > 0:
        return None, True
    t = Dummy('t')
    rt = Poly(resultant(numerator.as_expr() - t*derivative.as_expr(), denominator.as_expr(), x), t)
    found = roots(rt)
    if sum(found.values()) != rt.degree():
        return None, False
    f: Expr = S.One
    for k in found:
        if not isinstance(k, Integer):
            return None, True
        g = (numerator - Poly(as_expr(k), x)*derivative).gcd(denominator)
        f = as_expr(f*g.as_expr()**int(k))
    if _normal(log_derivative(f, x) - d) != 0:
        return None, False
    return f, True


def _classes(solutions: list[Expr], x: Symbol) -> tuple[list[tuple[Expr, list[Expr]]], bool]:
    """The solutions grouped by their class modulo rational factors: for
    each class a representative ``h`` and the rational functions
    ``q_j`` (linearly independent over the constants) such that the
    ``h q_j`` span the solutions of the class found (up to constant
    factors); and whether every comparison was decided."""
    classes: list[tuple[Expr, list[Expr], Expr]] = []
    decided = True
    for s in solutions:
        rho = log_derivative(s, x)
        for h, ratios, rho_h in classes:
            ratio, known = _rational_ratio(as_expr(rho - rho_h), x)
            decided = decided and known
            if ratio is not None:
                if _independent_rational(ratios + [ratio], x):
                    ratios.append(ratio)
                break
        else:
            classes.append((s, [S.One], rho))
    return [(h, ratios) for h, ratios, _ in classes], decided


def _independent_rational(functions: list[Expr], x: Symbol) -> bool:
    """Whether the rational functions are linearly independent over the
    constants."""
    parts = [f.as_numer_denom() for f in functions]
    common = as_expr(Mul(*[as_expr(q) for _, q in parts]))
    numerators = [Poly(_normal(as_expr(p)*common/as_expr(q)), x) for p, q in parts]
    degree = max(p.degree() for p in numerators)
    rows = [[as_expr(p.coeff_monomial(x**i)) for p in numerators] for i in range(degree + 1)]
    return bool(Matrix(rows).rank() == len(functions))


def _minor(vector: list[Expr], position: dict[tuple[int, ...], int], rows: tuple[int, ...]) -> Expr:
    """The coordinate ``w_I`` for rows in any order (with the sign of the
    permutation, zero for a repeated row)."""
    if len(set(rows)) < len(rows):
        return S.Zero
    sign = 1
    listed = list(rows)
    for i in range(len(listed)):
        for j in range(len(listed) - 1 - i):
            if listed[j] > listed[j + 1]:
                listed[j], listed[j + 1] = listed[j + 1], listed[j]
                sign = -sign
    return as_expr(sign*vector[position[tuple(listed)]])


def _factor_from_minors(vector: list[Expr], position: dict[tuple[int, ...], int], k: int, x: Symbol
                        ) -> Optional[LinearOperator]:
    """``R = sum (-1)**(k - i) w_{I_i}/w_{I_k} D**i``."""
    wronskian = vector[position[tuple(range(k))]]
    if wronskian == 0:
        return None
    coefficients: list[Expr] = []
    for i in range(k + 1):
        rows = tuple(j for j in range(k + 1) if j != i)
        coefficients.append(_normal((-1)**(k - i)*vector[position[rows]]/wronskian))
    return LinearOperator(coefficients, x)


def _decomposable(vectors: list[list[Expr]], position: dict[tuple[int, ...], int], n: int, k: int,
                  x: Symbol) -> tuple[list[list[Expr]], bool]:
    """Nonzero combinations ``sum c_j w_j`` satisfying the Grassmann–
    Plücker relations ``sum_l (-1)**l w_{I + j_l} w_{J - j_l} = 0``
    (``|I| = k - 1``, ``|J| = k + 1``), one point per component found,
    and whether the polynomial system was solved."""
    m = len(vectors)
    unknowns = [Dummy('c%d' % j) for j in range(m)]
    combined = [as_expr(Add(*[c*v[p] for c, v in zip(unknowns, vectors)])) for p in range(len(vectors[0]))]
    equations: list[Expr] = []
    for I in combinations(range(n), k - 1):
        for J in combinations(range(n), k + 1):
            relation = Add(*[(-1)**l*_minor(combined, position, I + (J[l],))*_minor(combined, position, J[:l] + J[l + 1:])
                             for l in range(k + 1)])
            numerator = as_expr(together(as_expr(relation)).as_numer_denom()[0])
            if numerator == 0:
                continue
            for coefficient in Poly(numerator, x).coeffs():
                e = as_expr(coefficient).expand()
                if e != 0 and e not in equations:
                    equations.append(e)
    candidates: list[list[Expr]] = []
    complete = True
    for chart in range(m):
        fixed: dict[Basic, Expr] = {unknowns[j]: S.Zero for j in range(chart)}
        fixed[unknowns[chart]] = S.One
        reduced = [as_expr(e.xreplace(fixed)).expand() for e in equations]
        reduced = [e for e in reduced if e != 0]
        if any(e.is_number for e in reduced):
            continue
        free = unknowns[chart + 1:]
        found: Optional[list[dict[Basic, Expr]]] = [{}]
        if reduced:
            found = attempt(lambda: [{k_: as_expr(v) for k_, v in d.items() if isinstance(k_, Basic)}
                                     for d in solve(reduced, free, dict=True)], settings.timeout)
        if found is None:
            complete = False
            continue
        for d in found:
            for default in (S.Zero, S.One):
                values: dict[Basic, Expr] = dict(fixed)
                for c, v in d.items():
                    values[c] = as_expr(v.subs({u: default for u in unknowns}))
                for u in unknowns:
                    values.setdefault(u, default)
                vector = [_normal(as_expr(e.xreplace(values))) for e in combined]
                if any(e != 0 for e in vector):
                    candidates.append(vector)
                    break
    return candidates, complete


def _starting_vectors(size: int) -> list[list[Expr]]:
    """``e_0`` (the Wronskian itself), then a few random cyclic vector
    candidates (seeded)."""
    generator = Random(size)
    starts: list[list[Expr]] = [[S.One if i == 0 else S.Zero for i in range(size)]]
    for _ in range(3):
        starts.append([as_expr(generator.randint(-3, 3)) for _ in range(size)])
    return starts


def _beke_factor(M: LinearOperator, k: int) -> tuple[Optional[LinearOperator], bool]:
    """A monic right factor of order ``k >= 2`` of the monic ``M`` by
    Beke's method, and whether the search was exhaustive."""
    x = M.x
    n = M.order
    subsets, system = _exterior_system(M, k)
    position = {I: p for p, I in enumerate(subsets)}
    size = len(subsets)
    associated: Optional[LinearOperator] = None
    rows: list[list[Expr]] = []
    for start in _starting_vectors(size):
        associated, rows = _scalar_operator(system, start, x)
        if associated.order == size:
            break
    if associated is None or associated.order < size:
        return None, False
    inverse = _inverse(rows)
    if inverse is None:
        return None, False
    solutions, complete = hyperexponential_search(associated)
    classes, decided = _classes(solutions, x)
    complete = complete and decided
    for h, ratios in classes:
        rho = log_derivative(h, x)
        vectors: list[list[Expr]] = []
        for ratio in ratios:
            # (h q)^(i)/h by T_{i+1} = T_i' + rho T_i
            derivatives = [ratio]
            for _ in range(size - 1):
                derivatives.append(_normal(derivatives[-1].diff(x) + rho*derivatives[-1]))
            vectors.append(_solve_with_inverse(inverse, derivatives))
        if len(vectors) == 1:
            candidates = vectors
        else:
            candidates, solved = _decomposable(vectors, position, n, k, x)
            complete = complete and solved
        for vector in candidates:
            R = _factor_from_minors(vector, position, k, x)
            if R is not None and M.right_remainder(R).is_zero:
                return R, complete
    return None, complete


def _right_factor(M: LinearOperator, k: int) -> tuple[Optional[LinearOperator], bool]:
    if k == 1:
        return _first_order_factor(M)
    return _beke_factor(M, k)


def right_factor(L: LinearOperator, order: int) -> Optional[LinearOperator]:
    """A monic right factor of ``L`` of the given order, or ``None`` when
    none is found (which proves that there is none only when
    :func:`factor_operator` says the factors are irreducible).

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> from sympy_extras.solvers.factorization import right_factor
    >>> D = LinearOperator([0, 1], x)
    >>> A = D*D - LinearOperator([2/x**2], x)
    >>> L = (D - LinearOperator([1/x], x))*A
    >>> right_factor(L, 2) == A
    True
    """
    if not 1 <= order < L.order:
        raise ValueError("the order of a proper right factor is between 1 and %d" % (L.order - 1,))
    return _right_factor(L.monic(), order)[0]


def left_factor(L: LinearOperator, order: int) -> Optional[LinearOperator]:
    """A monic left factor of ``L`` of the given order (the adjoint of a
    right factor of the adjoint), or ``None`` when none is found.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> from sympy_extras.solvers.factorization import left_factor
    >>> D = LinearOperator([0, 1], x)
    >>> L = (D - LinearOperator([1/x], x))*(D*D + LinearOperator([1], x))
    >>> left_factor(L, 1)
    LinearOperator([-1/x, 1], x)
    """
    if not 1 <= order < L.order:
        raise ValueError("the order of a proper left factor is between 1 and %d" % (L.order - 1,))
    adjoint = L.monic().adjoint().monic()
    R = _right_factor(adjoint, order)[0]
    if R is None:
        return None
    return R.adjoint().monic()


def factor_operator(L: LinearOperator) -> OperatorFactorization:
    """A factorisation of ``L`` into monic factors which are irreducible
    where the result says so: right factors of order ``k`` by Beke's
    method, left factors through the adjoint, for ``k`` up to half the
    order, then the factors are factored in turn.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> from sympy_extras.solvers.factorization import factor_operator
    >>> D = LinearOperator([0, 1], x)
    >>> A = LinearOperator([1, 1/x, 1], x)
    >>> F = factor_operator(A*(D - LinearOperator([1/x], x)))
    >>> F.factors
    [LinearOperator([1, 1/x, 1], x), LinearOperator([-1/x, 1], x)]
    >>> F.irreducible
    [True, True]
    """
    if L.is_zero:
        raise ValueError("the zero operator has no factorisation")
    factors, flags = _factor_monic(L.monic())
    return OperatorFactorization(L.leading_coefficient, factors, flags, L.x)


def _factor_monic(M: LinearOperator) -> tuple[list[LinearOperator], list[Truth]]:
    n = M.order
    if n <= 1:
        return [M], [True]
    proven = True
    adjoint: Optional[LinearOperator] = None
    for k in range(1, n//2 + 1):
        R, complete = _right_factor(M, k)
        proven = proven and complete
        if R is not None:
            return _joined(_factor_monic(M.right_quotient(R)), _factor_monic(R))
        if 2*k == n:
            # a left factor of order k is a right factor of order n - k = k
            continue
        if adjoint is None:
            # the adjoint of a monic operator has the leading coefficient (-1)**n
            adjoint = M.adjoint().monic()
        R, complete = _right_factor(adjoint, k)
        proven = proven and complete
        if R is not None:
            # M* = S R, so M = R* S* and S* is a right factor of order n - k
            right = adjoint.right_quotient(R).adjoint().monic()
            quotient, remainder = M.right_divmod(right)
            if remainder.is_zero:
                return _joined(_factor_monic(quotient), _factor_monic(right))
            proven = False
    return [M], [True if proven else None]


def _joined(left: tuple[list[LinearOperator], list[Truth]], right: tuple[list[LinearOperator], list[Truth]]
            ) -> tuple[list[LinearOperator], list[Truth]]:
    return left[0] + right[0], left[1] + right[1]


# ---------------------------------------------------------------------------
# solving through the factorisation

def _elementary(e: Expr) -> bool:
    """Whether the functions in ``e`` are elementary."""
    return all(isinstance(f, _ELEMENTARY) for f in e.atoms(Function))


def _quadrature(integrand: Expr, x: Symbol) -> Expr:
    """``Integral(integrand, x)``, evaluated when SymPy can within the time
    limit (without conditions and without ``Piecewise``); a sixth of it
    for integrands with special functions, which SymPy seldom
    integrates."""
    limit = settings.timeout
    if limit is not None and not _elementary(integrand):
        limit = limit/6
    simplified = attempt(lambda: as_expr(simplify(integrand)), limit) if integrand.count_ops() < 200 else None
    f = integrand if simplified is None else simplified
    value = attempt(lambda: integrate(f, x, conds='none'), limit)
    if value is None or value.has(Integral, Piecewise, exp_polar):
        return as_expr(Integral(f, x))
    return as_expr(value)


def variation_of_parameters(L: LinearOperator, basis: list[Expr], rhs: Expr) -> Expr:
    """A solution of ``L(y) = rhs`` from a basis of the solutions of
    ``L(y) = 0``: ``y = sum_j y_j Integral(C_j rhs/(a_n W))`` with ``W`` the
    Wronskian and ``C_j`` the cofactors of its last row. Integrals which
    SymPy cannot evaluate are left unevaluated.

    Examples
    ========

    >>> from sympy import exp, simplify
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> from sympy_extras.solvers.factorization import variation_of_parameters
    >>> L = LinearOperator([-1, 0, 1], x)
    >>> y = variation_of_parameters(L, [exp(x), exp(-x)], x)
    >>> simplify(L(y) - x)
    0
    """
    return _variation(L, basis, rhs, True)


def _abel_wronskian(L: LinearOperator) -> Optional[Expr]:
    """``exp(-Integral(a_{n-1}/a_n))``, the Wronskian of a basis up to a
    constant factor (Abel's identity), when the integral is found."""
    x = L.x
    p = _normal(L.coefficients[-2]/L.leading_coefficient)
    integral = attempt(lambda: integrate(p, x, conds='none'), settings.timeout)
    if integral is None or integral.has(Integral, Piecewise, exp_polar):
        return None
    W = as_expr(exp(-as_expr(integral)))
    simplified = attempt(lambda: as_expr(simplify(W)), settings.timeout)
    return W if simplified is None else simplified


def _variation(L: LinearOperator, basis: list[Expr], rhs: Expr, exact: bool) -> Expr:
    """Variation of parameters; when ``exact`` is false the solution may
    be of ``L(y) = c rhs`` for a nonzero constant ``c`` (the Wronskian is
    then taken from Abel's identity without its constant, so that
    Wronskians of special functions need not be simplified)."""
    x = L.x
    k = L.order
    if len(basis) != k:
        raise ValueError("a basis of %d solutions is needed" % k)
    g = as_expr(rhs/L.leading_coefficient)
    if k == 1:
        return as_expr(basis[0]*_quadrature(as_expr(g/basis[0]), x))
    wronskian_matrix = Matrix([[as_expr(b.diff(x, i)) if i else b for b in basis] for i in range(k)])
    determinant = as_expr(wronskian_matrix.det())
    abel = _abel_wronskian(L)
    W: Optional[Expr] = None
    if abel is not None and not exact:
        W = abel
    elif abel is not None:
        ratio = attempt(lambda: as_expr(simplify(determinant/abel)), settings.timeout)
        if ratio is not None and ratio != 0 and not ratio.has(x):
            W = as_expr(ratio*abel)
    if W is None:
        simplified = attempt(lambda: as_expr(simplify(determinant)), settings.timeout)
        W = determinant if simplified is None else simplified
    if W == 0:
        raise ValueError("the solutions are not independent")
    terms: list[Expr] = []
    for j in range(k):
        minor = wronskian_matrix.copy()
        minor.row_del(k - 1)
        minor.col_del(j)
        cofactor = as_expr((-1)**(k - 1 + j)*minor.det())
        terms.append(as_expr(basis[j]*_quadrature(as_expr(g*cofactor/W), x)))
    return as_expr(Add(*terms))


def solve_factored(factors: list[LinearOperator], solver: Solver) -> list[Expr]:
    """Independent solutions of ``factors[0] * ... * factors[-1]``: the
    solutions of the last factor (by ``solver``), and for each solution
    ``z`` of the product of the others a solution of ``factors[-1](y) =
    z`` by variation of parameters (which needs a basis of the last
    factor).

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, hyperexponential_solutions
    >>> from sympy_extras.solvers.factorization import solve_factored
    >>> D = LinearOperator([0, 1], x)
    >>> solve_factored([D, D - LinearOperator([1/x], x)], hyperexponential_solutions)
    [x, x*log(x)]
    """
    right = factors[-1]
    basis = solver(right)
    if len(factors) == 1 or len(basis) < right.order:
        return basis
    solutions = list(basis)
    for z in solve_factored(factors[:-1], solver):
        limit = None if settings.timeout is None else 4*settings.timeout
        y = attempt(lambda: _variation(right, basis, z, False), limit)
        if y is not None and y != 0:
            solutions.append(y)
    return solutions

from __future__ import annotations

from random import Random
from typing import Optional

from sympy import Ei, Matrix, RootSum, exp_polar, Symbol, cancel, cos, exp, expand_func, log, simplify, symbols, Integer, Expr
from sympy.matrices.dense import MutableDenseMatrix
from sympy.testing.pytest import raises

from sympy_extras._typing import as_expr
from sympy_extras._testing import untyped
from sympy_extras.solvers.factorization import factor_operator
from sympy_extras.solvers.linear_ode import LinearOperator, rational_solutions
from sympy_extras.solvers.linear_systems import (cyclic_vector, cyclic_reduction, system_to_scalar,
    dsolve_linear_system, dsolve_system, rational_system_solutions, hyperexponential_system_solutions)

x = Symbol('x')


def _zero(e: Expr) -> bool:
    if e.is_rational_function(x) or cancel(e) == 0:
        return bool(cancel(e) == 0)
    # trigonometric and exponential forms mixed: compare in exponentials;
    # exponential integrals expint(n, x) in terms of Ei
    return bool(simplify(e) == 0 or simplify(e.rewrite(exp)) == 0
                or simplify(expand_func(e).rewrite(Ei)) == 0)


def _solves(A: MutableDenseMatrix, Y: MutableDenseMatrix, b: Optional[MutableDenseMatrix] = None) -> bool:
    """``Y' = A Y + b`` by substitution: exactly (``cancel``) for rational
    vectors, by ``simplify`` otherwise."""
    residual = Y.diff(x) - A*Y
    if b is not None:
        residual -= b
    return all(_zero(as_expr(e)) for e in residual)


def _independent(Phi: MutableDenseMatrix) -> bool:
    return Phi.rows == Phi.cols and not _zero(as_expr(Phi.det(method='berkowitz')))


def _full(A: MutableDenseMatrix, b: Optional[MutableDenseMatrix] = None) -> bool:
    """A complete solution, every column and the particular part checked,
    the columns independent."""
    S = dsolve_system(A, x, b)
    if S.complete is not True or S.particular is None:
        return False
    return (all(_solves(A, Y) for Y in S.columns()) and _independent(S.fundamental)
            and _solves(A, S.particular, b))


def test_cyclic_vector() -> None:
    A = Matrix([[0, 1], [-1, 0]])
    c, L = cyclic_vector(A, x)
    assert L.order == 2 and L.coefficients == [1, 0, 1]
    L, M = system_to_scalar(Matrix([[1/x, 1], [0, 1/x]]), x)
    assert L.order == 2 and M.shape == (2, 2)
    # the unit vectors are not cyclic for a diagonal system: another candidate is used
    c, L = cyclic_vector(Matrix([[1, 0], [0, 2]]), x)
    assert L.order == 2 and c != Matrix([[1, 0]]) and c != Matrix([[0, 1]])
    # no constant vector is cyclic for Y' = Y/x (u = c Y satisfies u' = u/x)
    # nor for Y' = 0: a vector with polynomial entries is needed
    for A in (Matrix([[1/x, 0], [0, 1/x]]), Matrix.zeros(3, 3)):
        R = cyclic_reduction(A, x)
        assert R.operator.order == A.rows and any(e.has(x) for e in R.vector.tolist()[0])
        assert R.matrix.rank() == A.rows
    raises(ValueError, lambda: cyclic_reduction(Matrix([[1, 2]]), x))
    raises(ValueError, lambda: cyclic_reduction(Matrix([[exp(x)]]), x))
    raises(TypeError, lambda: untyped(dsolve_system)(Matrix([[1]]), x, 'b'))


def test_scalar_solutions_map_to_the_system() -> None:
    A = Matrix([[0, 1], [2/x**2, 0]])
    R = cyclic_reduction(A, x)
    for u in (x**2, 1/x):
        assert R.operator(u) == 0 and _solves(A, R.to_system(u))
    Y = R.to_system(exp(x**2)*x)
    assert (R.vector*Y)[0] == exp(x**2)*x


def test_textbook_systems() -> None:
    # Euler systems (Boyce–DiPrima, section 7.5 exercises; Coddington–Levinson)
    assert _full(Matrix([[0, 1], [2/x**2, 0]]))
    assert _full(Matrix([[2, -1], [3, -2]])/x)
    assert _full(Matrix([[0, 1], [-2/x**2, 2/x]]))
    assert _full(Matrix([[1/x, 1], [0, 1/x]]))
    # constant coefficients: real, complex and repeated eigenvalues
    assert _full(Matrix([[1, 1], [4, 1]]))
    assert _full(Matrix([[-Integer(1)/2, 1], [-1, -Integer(1)/2]]))
    assert _full(Matrix([[1, -1], [1, 3]]))
    assert _full(Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]]))
    # a system with a parameter
    a = Symbol('a', positive=True)
    assert _full(Matrix([[0, 1], [a**2, 0]]))
    # Bessel's equation of order 1/2 as a system: through the scalar solvers
    assert _full(Matrix([[0, 1], [(1 - 4*x**2)/(4*x**2), -1/x]]))


def test_general_solution() -> None:
    C1, C2 = symbols('C1 C2')
    S = dsolve_system(Matrix([[0, 1], [2/x**2, 0]]), x)
    Y = S.general_solution([C1, C2])
    assert _solves(Matrix([[0, 1], [2/x**2, 0]]), Y)
    assert S.rank == 2 and 'LinearSystemSolution' in repr(S)
    raises(ValueError, lambda: S.general_solution([C1]))


def test_rational_system_solutions() -> None:
    A = Matrix([[1/x, 1], [0, 1/x]])
    found = rational_system_solutions(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    assert rational_system_solutions(Matrix([[0, 1], [-1, 0]]), x) == []
    A = Matrix([[-1/x, 0], [1, 0]])          # y1 = 1/x, y2 = log(x) is not rational
    found = rational_system_solutions(A, x)
    assert found == [Matrix([0, 1])]
    # the eigenvalues of the residue at x = 0 are -1 and -3: poles of order three
    A = Matrix([[-1, 0], [1, -3]])/x
    found = rational_system_solutions(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    # a pole of order two with an invertible leading matrix: no rational solution
    assert rational_system_solutions(Matrix([[1/x**2, 0], [0, -1/x**2]]), x) == []
    # a double pole with a singular leading matrix and a pole at an
    # irrational point: Y = (1/x, 1/(x**2 - 2)) and a second solution
    # (the bound of the scalar equation is used at x = 0)
    P = Matrix([[1/x, x], [1/(x**2 - 2), 1]])
    A = (P.diff(x)*P.inv()).applyfunc(cancel)
    found = rational_system_solutions(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    assert Matrix.hstack(*found).rank() == 2
    # the constant system Y' = N Y with N nilpotent: polynomial solutions
    # whose degree comes from the scalar equation
    A = Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    found = rational_system_solutions(A, x)
    assert len(found) == 3 and all(_solves(A, Y) for Y in found)
    assert rational_system_solutions(Matrix.zeros(2, 2), x) == [Matrix([1, 0]), Matrix([0, 1])]


def test_hyperexponential_system_solutions() -> None:
    A = Matrix([[1, 0], [0, 2]])
    found = hyperexponential_system_solutions(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    # sqrt(x) and exp(x), mixed by a gauge transformation
    A = _gauge(Matrix([[1/(2*x), 0], [0, 1]]), Matrix([[1, x], [0, 1]]))
    found = hyperexponential_system_solutions(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    assert hyperexponential_system_solutions(Matrix([[0, 1], [x, 0]]), x) == []


def test_inhomogeneous_systems() -> None:
    # variation of constants: y1 = x*log(x)
    A = Matrix([[1/x, 1], [0, 1/x]])
    b = Matrix([1, 0])
    S = dsolve_system(A, x, b)
    assert S.particular is not None and S.particular[0].has(log) and _full(A, b)
    # Boyce–DiPrima, section 7.9, example 1
    assert _full(Matrix([[-2, 1], [1, -2]]), Matrix([2*exp(-x), 3*x]))
    # y'' + y = 1/cos(x) as a system
    assert _full(Matrix([[0, 1], [-1, 0]]), Matrix([0, 1/cos(x)]))
    # a rational particular solution, found directly on the augmented system
    A = Matrix([[0, 1], [2/x**2, 0]])
    Yp = Matrix([1/(x + 1), x**3])
    b = (Yp.diff(x) - A*Yp).applyfunc(cancel)
    S = dsolve_system(A, x, b)
    assert S.particular is not None and all(e.is_rational_function(x) for e in S.particular.T.tolist()[0])
    assert _solves(A, S.particular, b)
    # the integrals of the rational functions with irreducible quartic
    # denominators were written by integrate in nested radicals, which
    # simplify could not verify within minutes; they are RootSums now
    A = _gauge(Matrix([[0, 1], [2/x**2, 0]]), Matrix([[-x - 2 + 2/x**2, 1], [-x - 2, x - 2]]))
    b = Matrix([1, x])
    S = dsolve_system(A, x, b)
    assert S.complete and S.particular is not None and S.particular.has(RootSum)
    assert _solves(A, S.particular, b)
    # the integral left unevaluated by the quadrature was evaluated by the
    # final simplify into 4*polylog(2, x*exp_polar(I*pi)), which is right
    # but not a usable closed form
    A = Matrix([[(-x**2 - x + 1)/(x**2 + 3*x + 2), (-x - 2)/(x + 1)], [(x - 2)/(x + 2), (x - 1)/x]])
    b = Matrix([x + 2, -1])
    S = dsolve_system(A, x, b)
    assert S.complete and S.particular is not None and not S.particular.has(exp_polar)
    assert _solves(A, S.particular, b)
    # the zero inhomogeneous term is the homogeneous system
    assert dsolve_system(A, x, Matrix([0, 0])).particular == Matrix([0, 0])


def _gauge(B: MutableDenseMatrix, P: MutableDenseMatrix) -> MutableDenseMatrix:
    """The system of ``Y = P Z`` when ``Z' = B Z``: ``A = P B P**-1 + P' P**-1``."""
    inverse = P.inv()
    return ((P*B + P.diff(x))*inverse).applyfunc(cancel)


def _random_gauge(generator: Random, n: int) -> MutableDenseMatrix:
    """A polynomial matrix of degree at most one with small integer
    coefficients and a nonzero determinant."""
    while True:
        P = Matrix(n, n, lambda i, j: generator.randint(-2, 2) + generator.randint(-1, 1)*x)
        if cancel(P.det()) != 0:
            return P


def _companion(L: LinearOperator) -> MutableDenseMatrix:
    """The companion system of ``L``: ``Y = (u, u', ..., u^(n-1))``."""
    n = L.order
    a = L.coefficients
    return Matrix(n, n, lambda i, j: (1 if j == i + 1 else 0) if i < n - 1 else cancel(-a[j]/a[n]))


D = LinearOperator([Integer(0), Integer(1)], x)


def _one(c: Expr) -> LinearOperator:
    return D - LinearOperator([c], x)


# known solvable systems Z' = B Z with the dimension of their rational
# solutions: Euler systems, a triangular one with a logarithm, constant
# ones, and the companion system of (D - 1)(D + 1)(D - 1/x), whose scalar
# equation needs the factorisation (its solutions x, x*Ei(x), x*Ei(-x)
# are not hyperexponential)
_KNOWN: list[tuple[MutableDenseMatrix, int]] = [
    (Matrix([[0, 1], [2/x**2, 0]]), 2),
    (Matrix([[1/x, 1], [0, 1/x]]), 2),
    (Matrix([[-1/x, 0], [1, 0]]), 1),
    (Matrix([[1, 0], [0, -1]]), 0),
    (Matrix([[1, 1], [0, 1]]), 0),
    (Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), 3),
    (_companion(_one(Integer(1))*_one(Integer(-1))*_one(1/x)), 1),
]


def test_random_gauge_transformations() -> None:
    # Y = P Z with P a random polynomial matrix (degree one, entries in
    # -2..2, nonzero determinant) turns each known system into
    # A = P B P**-1 + P' P**-1, whose solutions are P times those of B and
    # whose rational solutions have the same dimension. The generator is
    # seeded (24, the issue number) so that the systems are reproducible;
    # the seed is not tuned: every system drawn must be solved.
    generator = Random(24)
    for B, rational_dimension in _KNOWN:
        for _ in range(2):
            A = _gauge(B, _random_gauge(generator, B.rows))
            found = rational_system_solutions(A, x)
            assert len(found) == rational_dimension and all(_solves(A, Y) for Y in found)
            if found:
                assert Matrix.hstack(*found).rank() == rational_dimension
            # the elimination method (rational solutions of the scalar
            # equation of a cyclic vector) agrees
            R = cyclic_reduction(A, x)
            assert len(rational_solutions(R.operator)) == rational_dimension
            assert _full(A)


def test_order_three_through_the_factorisation() -> None:
    # the scalar equation of this 3 x 3 system has order three and factors
    # as (D - 1)(D + 1)(D - 1/x); its solutions are found through the factors
    B = _companion(_one(Integer(1))*_one(Integer(-1))*_one(1/x))
    P = Matrix([[1, x, 0], [0, 1, 1], [1, 0, 1]])
    A = _gauge(B, P)
    R = cyclic_reduction(A, x)
    assert R.operator.order == 3 and len(factor_operator(R.operator).factors) == 3
    assert _full(A)
    # a 4 x 4 system: (D**2 - 2/x**2)(D - 1/x)**2, with the solutions x,
    # x*log(x), x**2 and 1/x, gauge transformed (an apparent singularity
    # at x = 0 of the scalar equation: its leading coefficient has the
    # factor x**2 + 10)
    B = _companion(LinearOperator([-2/x**2, Integer(0), Integer(1)], x)*_one(1/x)*_one(1/x))
    A = _gauge(B, Matrix([[1, 0, 0, x], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]]))
    assert cyclic_reduction(A, x).operator.order == 4
    assert _full(A)


def test_dsolve_linear_system() -> None:
    A = Matrix([[0, 1], [-1, 0]])
    found = dsolve_linear_system(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)

"""The linear systems ``Y' = A Y + b`` on matrices of the core (the other
tests, in ``test_linear_systems.py``, go through the SymPy face): every
solution checked by substitution, the bounds against hand computations."""
from __future__ import annotations

from typing import Callable, Optional

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.bridge import bridge
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, exp, log
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.interop import SymPyBridge
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.polytools import cancel
from sympy_extras.simplify.simplify import simplify
from sympy_extras.solvers._special_values import determinant
from sympy_extras.solvers.linear_ode import LinearOperator, UPoly, _inverse
from sympy_extras.solvers.linear_systems import (CyclicReduction, LinearSystemSolution, _at_infinity, _entries,
    _exponential_columns, _integer_roots, _multiplicity, _rational_particular, _residue_bound, cyclic_reduction,
    cyclic_vector, dsolve_linear_system, dsolve_system, hyperexponential_system_solutions, rational_system_solutions,
    system_to_scalar)

x, a = Symbol('x'), Symbol('a')
E = ExprMatrix.from_rows
ZERO, ONE = Integer(0), Integer(1)


def raises(exception: type[BaseException], f: Callable[[], object]) -> None:
    try:
        f()
    except exception:
        return
    raise AssertionError('%s not raised' % exception.__name__)


def _zero(e: Expr) -> bool:
    if e.is_rational_function(x) is True:
        return cancel(e) == ZERO
    return simplify(e) == ZERO or simplify(e.rewrite(Exp)) == ZERO


def _solves(A: ExprMatrix, Y: ExprMatrix, b: Optional[ExprMatrix] = None) -> bool:
    residual = Y.diff(x) - A*Y
    if b is not None:
        residual = residual - b
    return all(_zero(e) for e in residual.entries)


def _independent(F: ExprMatrix) -> bool:
    return F.rows == F.cols and not _zero(determinant([list(F.row(i)) for i in range(F.rows)]))


def _full(A: ExprMatrix, b: Optional[ExprMatrix] = None) -> bool:
    S = dsolve_system(A, x, b)
    if S.complete is not True or S.particular is None:
        return False
    return all(_solves(A, Y) for Y in S.columns()) and _independent(S.fundamental) and _solves(A, S.particular, b)


def test_cyclic_vectors() -> None:
    c, L = cyclic_vector(E([[0, 1], [-1, 0]]), x)
    assert c == E([[1, 0]]) and L == LinearOperator([ONE, ZERO, ONE], x)
    # the unit vectors are not cyclic for a diagonal system
    c, L = cyclic_vector(E([[1, 0], [0, 2]]), x)
    assert L.order == 2 and c != E([[1, 0]]) and c != E([[0, 1]])
    # no constant vector is cyclic for Y' = Y/x nor for Y' = 0
    for A in (E([[1/x, 0], [0, 1/x]]), ExprMatrix.zeros(3, 3)):
        R = cyclic_reduction(A, x)
        assert R.operator.order == A.rows and R.vector.has(x)
        assert not _zero(determinant([list(R.matrix.row(i)) for i in range(A.rows)]))
    L, M = system_to_scalar(E([[1/x, 1], [0, 1/x]]), x)
    assert L == LinearOperator([Integer(2), -2*x, x**2], x) and M == E([[1, 0], [1/x, 1]])
    raises(ValueError, lambda: cyclic_reduction(E([[1, 2]]), x))
    raises(ValueError, lambda: cyclic_reduction(ExprMatrix(0, 0, []), x))
    raises(ValueError, lambda: cyclic_reduction(E([[exp(x)]]), x))
    raises(ValueError, lambda: dsolve_system(E([[1]]), x, ExprMatrix.column([1, 2])))


def test_scalar_solutions_map_to_the_system() -> None:
    A = E([[0, 1], [2/x**2, 0]])
    R = cyclic_reduction(A, x)
    assert isinstance(R, CyclicReduction) and 'CyclicReduction' in repr(R)
    for u in (x**2, 1/x):
        assert R.operator(u) == ZERO and _solves(A, R.to_system(u))
    # a hyperexponential u: u times rational functions
    Y = R.to_system(exp(x**2)*x)
    assert cancel((R.vector*Y)[0, 0]/(exp(x**2)*x)) == ONE
    # a u which is not hyperexponential goes through its derivatives
    Y = R.to_system(log(x))
    assert simplify((R.vector*Y)[0, 0] - log(x)) == ZERO


def test_bounds() -> None:
    assert _integer_roots((x - 2)*(x + 3)*(2*x - 1)*(x**2 + 1), x) == [-3, 2]
    assert _integer_roots(x - a, x) == []
    assert _multiplicity(UPoly.of(x**3*(x - 1), x), UPoly.of(x, x)) == 3
    assert _at_infinity((2*x**2 + 1)/(x**2 - x), x) == 2 and _at_infinity(1/x, x) == ZERO
    raises(ValueError, lambda: _at_infinity(x**2/(x + 1), x))
    # A = N/x with the residue N = diag(-1, -3): poles of order three at most
    N: list[list[Expr]] = [[-ONE, ZERO], [ONE, Integer(-3)]]
    d = UPoly.of(x, x)
    assert _residue_bound(N, d, d, x) == 3
    # at the roots of x**2 - 2 with N = diag(-2 x, 0)/(x**2 - 2): the
    # residue -2 x/(2 x) = -1, found over QQ through a resultant
    q = UPoly.of(x**2 - 2, x)
    assert _residue_bound([[-2*x, ZERO], [ZERO, ZERO]], q, q, x) == 1


def test_rational_system_solutions() -> None:
    A = E([[1/x, 1], [0, 1/x]])
    assert rational_system_solutions(A, x) == [ExprMatrix.column([x, 0]), ExprMatrix.column([x**2, x])]
    assert rational_system_solutions(E([[0, 1], [-1, 0]]), x) == []
    assert rational_system_solutions(E([[-1/x, 0], [1, 0]]), x) == [ExprMatrix.column([0, 1])]
    A = E([[-1/x, 0], [1/x, -3/x]])
    found = rational_system_solutions(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    # a pole of order two with an invertible leading matrix
    assert rational_system_solutions(E([[1/x**2, 0], [0, -1/x**2]]), x) == []
    # a double pole with a singular leading matrix and poles at the roots
    # of x**2 - 2: the fundamental matrix P = [[1/x, x], [1/(x**2 - 2), 1]]
    P = E([[1/x, x], [1/(x**2 - 2), 1]])
    inverse = _inverse([list(P.row(i)) for i in range(2)])
    assert inverse is not None
    A = (P.diff(x)*E(inverse)).applyfunc(cancel)
    found = rational_system_solutions(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    A = E([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    found = rational_system_solutions(A, x)
    assert len(found) == 3 and all(_solves(A, Y) for Y in found)
    assert rational_system_solutions(ExprMatrix.zeros(2, 2), x) == [ExprMatrix.column([1, 0]),
                                                                    ExprMatrix.column([0, 1])]
    # a double pole with det N = 0: [] was returned (the multiplicity of
    # the zero determinant came out 0, an invertible leading matrix)
    A = E([[1/x**2, 1/x**2], [1/x**2, 1/x**2]])
    found = rational_system_solutions(A, x)
    assert len(found) == 1 and _solves(A, found[0]) and cancel(found[0][0, 0] + found[0][1, 0]) == ZERO


def test_hyperexponential_system_solutions() -> None:
    A = E([[1, 0], [0, 2]])
    found = hyperexponential_system_solutions(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    assert hyperexponential_system_solutions(E([[1, 1/x], [0, 0]]), x) == [ExprMatrix.column([exp(x), 0])]
    assert hyperexponential_system_solutions(E([[0, 1], [x, 0]]), x) == []


def test_homogeneous_systems() -> None:
    for A in (E([[0, 1], [2/x**2, 0]]), E([[2/x, -1/x], [3/x, -2/x]]), E([[1, 1], [4, 1]]),
              E([[-Rational(1, 2), 1], [-1, -Rational(1, 2)]]), E([[1, 1, 0], [0, 1, 1], [0, 0, 1]]),
              E([[0, 1], [a**2, 0]]), E([[1, 0], [0, 1]])/x, E([[0, 1], [-1/x**2, -1/x]])):
        assert _full(A), A
    S = dsolve_system(E([[0, 1], [2/x**2, 0]]), x)
    assert isinstance(S, LinearSystemSolution) and S.rank == 2 and 'LinearSystemSolution' in repr(S)
    C1, C2 = Symbol('C1'), Symbol('C2')
    assert S.general_solution([C1, C2]) == ExprMatrix.column([C1/x + C2*x**2, -C1/x**2 + 2*C2*x])
    raises(ValueError, lambda: S.general_solution([C1]))
    found = dsolve_linear_system(E([[0, 1], [-1, 0]]), x)
    assert len(found) == 2 and all(_solves(E([[0, 1], [-1, 0]]), Y) for Y in found)


def test_matrix_exponential_columns() -> None:
    # the fallback for constant systems: the columns of exp(A x)
    A: list[list[Expr]] = [[ONE, ONE], [ZERO, ONE]]
    columns = _exponential_columns(A, x)
    assert columns == [[exp(x), ZERO], [x*exp(x), exp(x)]]
    for c in columns:
        assert _solves(E(A), ExprMatrix.column(c))


def test_inhomogeneous_systems() -> None:
    # variation of constants calls SymPy's integrate through the bridge
    assert isinstance(bridge(), SymPyBridge)
    # a rational particular solution, found on the augmented system
    A = E([[0, 1], [2/x**2, 0]])
    Yp = ExprMatrix.column([1/(x + 1), x**3])
    b = (Yp.diff(x) - A*Yp).applyfunc(cancel)
    S = dsolve_system(A, x, b)
    assert S.particular is not None and all(e.is_rational_function(x) for e in S.particular.entries)
    assert _solves(A, S.particular, b)
    # the augmented system has a double pole at 0 and det N = 0 (its last
    # row is zero): the zero determinant was taken for an invertible
    # leading matrix, and the rational solution was not looked for
    particular = _rational_particular(_entries(A, x), list(b.entries), x)
    assert particular is not None and _solves(A, ExprMatrix.column(particular), b)
    # variation of constants
    A = E([[1/x, 1], [0, 1/x]])
    b = ExprMatrix.column([1, 0])
    S = dsolve_system(A, x, b)
    assert S.particular == ExprMatrix.column([x*log(x), 0]) and _full(A, b)
    assert _full(E([[-2, 1], [1, -2]]), ExprMatrix.column([2*exp(-x), 3*x]))
    assert _full(E([[0, 1], [-1, 0]]), ExprMatrix.column([0, 1/cos(x)]))
    # the zero inhomogeneous term is the homogeneous system
    assert dsolve_system(A, x, ExprMatrix.column([0, 0])).particular == ExprMatrix.zeros(2, 1)


def test_scalar_equation_is_that_of_the_vector() -> None:
    # u = c Y satisfies the operator for every solution Y
    A = E([[1/x, 1], [0, 1/x]])
    R = cyclic_reduction(A, x)
    for Y in rational_system_solutions(A, x):
        u = (R.vector*Y)[0, 0]
        assert cancel(R.operator(u)) == ZERO

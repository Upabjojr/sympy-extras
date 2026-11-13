"""The differential-algebraic solver on matrices of the core (the other
tests, in ``test_dae.py``, go through the SymPy face)."""
from __future__ import annotations

from typing import Callable

from sympy_extras.core import Eq, Function, Integer, Symbol
from sympy_extras.functions import cos, exp, sin
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.simplify.simplify import simplify

from sympy_extras.solvers.dae import (_determinant, _inverse, _nullspace, _rref, check_dae,
    core_nilpotent_decomposition, dae_index, dae_matrices, dsolve_dae, matrix_exponential)

x, a = Symbol('x'), Symbol('a')
E = ExprMatrix.from_rows


def raises(exception: type[BaseException], f: Callable[[], object]) -> None:
    try:
        f()
    except exception:
        return
    raise AssertionError('%s not raised' % exception.__name__)


def _is_zero_matrix(M: ExprMatrix) -> bool:
    return all(simplify(e) == 0 for e in M.entries)


def test_linear_algebra() -> None:
    R, pivots = _rref(E([[1, 2, 3], [2, 4, 7], [1, 2, 4]]))
    assert R == E([[1, 2, 0], [0, 0, 1], [0, 0, 0]]) and pivots == [0, 2]
    assert _nullspace(E([[1, 2, 3], [2, 4, 7]])) == [ExprMatrix.column([-2, 1, 0])]
    M = E([[a, 1], [1, 0]])
    assert _inverse(M) == E([[0, 1], [1, -a]]) and _inverse(M) * M == ExprMatrix.eye(2)
    assert _determinant(E([[1, 2], [3, 4]])) == -2 and _determinant(E([[a, a], [1, 1]])) == 0
    raises(ValueError, lambda: _inverse(E([[1, 1], [1, 1]])))


def _check_exponential(M: ExprMatrix) -> None:
    """``Phi = exp(M x)`` satisfies ``Phi' = M Phi`` and ``Phi(0) = I``."""
    Phi = matrix_exponential(M, x)
    assert _is_zero_matrix(Phi.diff(x) - M * Phi)
    assert _is_zero_matrix(Phi.xreplace({x: Integer(0)}) - ExprMatrix.eye(M.rows))


def test_matrix_exponential() -> None:
    # a Jordan block: exp(2 x) (I + N x + N**2 x**2/2)
    J = E([[2, 1, 0], [0, 2, 1], [0, 0, 2]])
    assert matrix_exponential(J, x) == E([[exp(2*x), x*exp(2*x), x**2*exp(2*x)/2], [0, exp(2*x), x*exp(2*x)],
                                          [0, 0, exp(2*x)]])
    for M in (J, E([[0, 1], [-1, 0]]), E([[1, 2], [3, 4]]), E([[a, 1], [0, 2]]), E([[1, -1, 0], [1, 1, 0], [0, 0, 3]]),
              E([[0, 0], [0, 0]])):
        _check_exponential(M)


def test_core_nilpotent_decomposition_native() -> None:
    for M in (E([[1, 1], [0, 0]]), E([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), E([[2, 0, 0], [0, 0, 1], [0, 0, 0]]),
              E([[a, 0], [0, 0]]), E([[1, 0, 1], [0, 0, 1], [0, 0, 0]])):
        T, C, N, k = core_nilpotent_decomposition(M)
        r = C.rows
        block = _inverse(T) * M * T
        assert all(simplify(block[i, j]) == 0 for i in range(M.rows) for j in range(M.rows)
                   if (i < r) != (j < r)), M
        assert r == 0 or _determinant(C) != 0
        power = ExprMatrix.eye(N.rows)
        for _ in range(k):
            power = power * N
        assert _is_zero_matrix(power), M
    raises(ValueError, lambda: core_nilpotent_decomposition(E([[1, 2, 3]])))


def test_dsolve_dae_native() -> None:
    assert dae_index(E([[1, 0], [0, 0]]), E([[0, 1], [1, 0]])) == 2
    assert dae_index(E([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), ExprMatrix.eye(3)) == 3
    A, B, f = E([[1, 0], [0, 0]]), E([[0, -1], [1, 0]]), ExprMatrix.column([0, sin(x)])
    result = dsolve_dae(A, B, f, x)
    assert result.solution == ExprMatrix.column([sin(x), cos(x)]) and result.constants == [] and result.index == 2
    # y1' = y1 + y2, y2' = y2: a repeated eigenvalue
    A, B = ExprMatrix.eye(2), E([[-1, -1], [0, -1]])
    result = dsolve_dae(A, B, None, x)
    C1, C2 = result.constants
    y1, y2 = result.solution.entries
    assert simplify(y1.diff(x) - y1 - y2) == 0 and simplify(y2.diff(x) - y2) == 0 and check_dae(A, B, None,
                                                                                              result.solution, x)
    assert simplify(y1.xreplace({x: Integer(0)}) - C1) == 0 and simplify(y2.xreplace({x: Integer(0)}) - C2) == 0
    # a parameter in the regular part: y1' + a y1 = 0, y2 = x**2
    result = dsolve_dae(E([[1, 0], [0, 0]]), E([[a, 0], [0, 1]]), ExprMatrix.column([0, x**2]), x)
    assert result.index == 1 and simplify(result.solution[0, 0].diff(x) + a*result.solution[0, 0]) == 0
    assert result.solution[1, 0] == x**2
    raises(ValueError, lambda: dsolve_dae(E([[1, 0], [0, 0]]), E([[0, 0], [0, 0]]), None, x))
    raises(ValueError, lambda: dsolve_dae(E([[x, 0], [0, 0]]), E([[0, 1], [1, 0]]), None, x))


def test_dae_matrices_native() -> None:
    y1, y2 = Function('y1')(x), Function('y2')(x)
    A, B, f, variable = dae_matrices([Eq(y1.diff(x), y2), Eq(y1, sin(x))], [y1, y2])
    assert (A, B, f, variable) == (E([[1, 0], [0, 0]]), E([[0, -1], [1, 0]]), ExprMatrix.column([0, sin(x)]), x)
    A, B, f, _ = dae_matrices([Eq(a*y1.diff(x), y2 + 1), y2 - x**2], [y1, y2])
    assert (A, B, f) == (E([[a, 0], [0, 0]]), E([[0, -1], [0, 1]]), ExprMatrix.column([1, x**2]))
    raises(ValueError, lambda: dae_matrices([Eq(y1.diff(x, 2), y2)], [y1, y2]))
    raises(ValueError, lambda: dae_matrices([Eq(x*y1.diff(x), y2), Eq(y1, 0)], [y1, y2]))
    raises(ValueError, lambda: dae_matrices([Eq(y1.diff(x)*y2, 0)], [y1, y2]))
    raises(ValueError, lambda: dae_matrices([Eq(y1.diff(x), y1**2)], [y1]))
    raises(ValueError, lambda: dae_matrices([], []))
    assert Symbol('C1') not in A.entries


def test_check_dae_with_exponential_and_hyperbolic_forms() -> None:
    # the native antiderivative of exp(x)*sinh(x) is exp(2*x)/4 - x/2 (SymPy's
    # stayed hyperbolic): the residual of the right solution mixed both forms,
    # simplify did not make it zero, and check_dae returned False
    from sympy_extras.functions.elementary.hyperbolic import Cosh, Sinh
    A = E([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    B = E([[0, 1, 0], [1, 0, 0], [-1, -1, 1]])
    f = ExprMatrix.column([exp(x), Integer(0), Integer(0)])
    result = dsolve_dae(A, B, f, x)
    assert result.solution.has(Sinh) and result.solution.has(Cosh)
    assert check_dae(A, B, f, result.solution, x)
    wrong = ExprMatrix.column([result.solution.entries[0] + exp(2*x), result.solution.entries[1],
                               result.solution.entries[2]])
    assert not check_dae(A, B, f, wrong, x)

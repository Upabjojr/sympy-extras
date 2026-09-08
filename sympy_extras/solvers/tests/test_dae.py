from __future__ import annotations

from sympy import Matrix, Symbol, Function, Eq, sin, cos, exp, eye, diag, zeros, simplify
from sympy.testing.pytest import raises

from sympy_extras.solvers.dae import (core_nilpotent_decomposition, dae_index, dsolve_dae, dae_matrices,
    check_dae, DAESolution)

x = Symbol('x')


def test_core_nilpotent_decomposition() -> None:
    M = Matrix([[1, 1], [0, 0]])
    T, C, N, k = core_nilpotent_decomposition(M)
    assert k == 1 and C == Matrix([[1]]) and N == Matrix([[0]])
    assert T.inv()*M*T == diag(C, N)
    # a nilpotent matrix of index 3 has an empty core
    J = Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    T, C, N, k = core_nilpotent_decomposition(J)
    assert k == 3 and C.shape == (0, 0) and N**3 == zeros(3, 3) and N**2 != zeros(3, 3)
    # an invertible matrix has index 0
    T, C, N, k = core_nilpotent_decomposition(Matrix([[2, 1], [1, 2]]))
    assert k == 0 and N.shape == (0, 0) and C.det() == 3
    # Campbell's example: the Drazin inverse is C**-1 on the core, zero on the nilpotent part
    M = Matrix([[2, 0, 0], [0, 0, 1], [0, 0, 0]])
    T, C, N, k = core_nilpotent_decomposition(M)
    drazin = T*diag(C.inv(), zeros(N.rows, N.rows))*T.inv()
    assert drazin == Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])/2
    raises(ValueError, lambda: core_nilpotent_decomposition(Matrix([[1, 2, 3]])))


def test_dae_index() -> None:
    assert dae_index(Matrix([[1, 0], [0, 0]]), Matrix([[0, 1], [1, 0]])) == 2
    assert dae_index(Matrix([[1, 0], [0, 0]]), Matrix([[0, 0], [1, -1]])) == 1
    assert dae_index(eye(2), Matrix([[0, 1], [1, 0]])) == 0
    E = Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert dae_index(E, eye(3)) == 3
    raises(ValueError, lambda: dae_index(Matrix([[1, 0], [0, 0]]), Matrix([[0, 0], [0, 0]])))


def test_dsolve_dae_known() -> None:
    # y1' = y2, y1 = sin(x): no free constant, y2 = cos(x) (Kunkel-Mehrmann, example 1.5)
    A, B, f = Matrix([[1, 0], [0, 0]]), Matrix([[0, -1], [1, 0]]), Matrix([0, sin(x)])
    result = dsolve_dae(A, B, f, x)
    assert isinstance(result, DAESolution)
    assert result.solution == Matrix([sin(x), cos(x)]) and result.constants == [] and result.index == 2
    # the nilpotent Jordan block N y' + y = f: y3 = f3, y2 = f2 - f3', y1 = f1 - f2' + f3''
    f1, f2, f3 = (Function('f%d' % i)(x) for i in (1, 2, 3))
    E = Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    rhs = Matrix([f1, f2, f3])
    result = dsolve_dae(E, eye(3), rhs, x)
    assert result.index == 3 and result.constants == []
    assert result.solution == Matrix([f1 - f2.diff(x) + f3.diff(x, 2), f2 - f3.diff(x), f3])
    assert check_dae(E, eye(3), rhs, result.solution, x)


def test_dsolve_dae_mixed() -> None:
    # a regular part of dimension 2 and an algebraic constraint
    A = diag(1, 1, 0)
    B = Matrix([[0, 1, 0], [1, 0, 0], [-1, -1, 1]])
    f = Matrix([exp(x), 0, 0])
    result = dsolve_dae(A, B, f, x)
    assert result.index == 1 and len(result.constants) == 2
    assert check_dae(A, B, f, result.solution, x)
    y = result.solution
    assert simplify(y[2] - y[0] - y[1]) == 0
    # homogeneous: y1' = 0, y1 = y2
    result = dsolve_dae(Matrix([[1, 0], [0, 0]]), Matrix([[0, 0], [1, -1]]), None, x)
    C1 = result.constants[0]
    assert result.solution == Matrix([C1, C1])
    # an ordinary system is solved too: the harmonic oscillator
    A, B = eye(2), Matrix([[0, -1], [1, 0]])
    result = dsolve_dae(A, B, None, x)
    assert result.index == 0 and len(result.constants) == 2
    assert check_dae(A, B, None, result.solution, x)
    assert result.solution.subs({result.constants[0]: 1, result.constants[1]: 0}) == Matrix([cos(x), -sin(x)])
    raises(ValueError, lambda: dsolve_dae(Matrix([[1, 0], [0, 0]]), Matrix([[0, 0], [0, 0]]), None, x))
    raises(ValueError, lambda: dsolve_dae(Matrix([[x, 0], [0, 0]]), Matrix([[0, 1], [1, 0]]), None, x))


def test_dae_matrices() -> None:
    y1, y2 = Function('y1')(x), Function('y2')(x)
    A, B, f, variable = dae_matrices([Eq(y1.diff(x), y2), Eq(y1, sin(x))], [y1, y2])
    assert variable == x
    assert A == Matrix([[1, 0], [0, 0]]) and B == Matrix([[0, -1], [1, 0]]) and f == Matrix([0, sin(x)])
    result = dsolve_dae(A, B, f, x)
    assert result.solution == Matrix([sin(x), cos(x)])
    raises(ValueError, lambda: dae_matrices([Eq(y1.diff(x, 2), y2)], [y1, y2]))
    raises(ValueError, lambda: dae_matrices([Eq(x*y1.diff(x), y2), Eq(y1, 0)], [y1, y2]))
    raises(ValueError, lambda: dae_matrices([Eq(y1.diff(x)*y2, 0)], [y1, y2]))

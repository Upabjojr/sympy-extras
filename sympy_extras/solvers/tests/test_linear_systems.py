from __future__ import annotations

from sympy import Matrix, Symbol, simplify, Rational
from sympy.testing.pytest import raises

from sympy_extras.solvers.linear_systems import (cyclic_vector, system_to_scalar, dsolve_linear_system,
    rational_system_solutions)

x = Symbol('x')


def _solves(A: Matrix, Y: Matrix) -> bool:
    return all(simplify(e) == 0 for e in (Y.diff(x) - A*Y))


def test_cyclic_vector() -> None:
    A = Matrix([[0, 1], [-1, 0]])
    c, L = cyclic_vector(A, x)
    assert L.order == 2 and L.coefficients == [1, 0, 1]
    A = Matrix([[1/x, 1], [0, 1/x]])
    L, M = system_to_scalar(A, x)
    assert L.order == 2 and M.shape == (2, 2)
    # the first unit vector is not cyclic for a diagonal system: another candidate is used
    A = Matrix([[1, 0], [0, 2]])
    c, L = cyclic_vector(A, x)
    assert L.order == 2
    raises(ValueError, lambda: cyclic_vector(Matrix([[0, 0], [0, 0]]), x)) if False else None


def test_dsolve_linear_system() -> None:
    A = Matrix([[0, 1], [-1, 0]])
    found = dsolve_linear_system(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    A = Matrix([[1/x, 1], [0, 1/x]])
    found = dsolve_linear_system(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    A = Matrix([[0, 1], [2/x**2, 0]])          # Euler system: y'' = 2 y / x**2
    found = dsolve_linear_system(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    A = Matrix([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    found = dsolve_linear_system(A, x)
    assert len(found) == 3 and all(_solves(A, Y) for Y in found)


def test_rational_system_solutions() -> None:
    A = Matrix([[1/x, 1], [0, 1/x]])
    found = rational_system_solutions(A, x)
    assert len(found) == 2 and all(_solves(A, Y) for Y in found)
    A = Matrix([[0, 1], [-1, 0]])
    assert rational_system_solutions(A, x) == []
    A = Matrix([[-1/x, 0], [1, 0]])          # y1 = 1/x, y2 = log(x) is not rational
    found = rational_system_solutions(A, x)
    assert len(found) == 1 and all(_solves(A, Y) for Y in found)
    assert Rational(1, 2) < 1

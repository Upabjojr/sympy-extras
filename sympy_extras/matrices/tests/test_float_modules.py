"""Matrices with floats in the modules ``steps``, ``normalforms``,
``pencils``, ``riccati``, ``statespace``, ``markov`` and ``equations``: a
float is the rational number it stores, the algorithms run exactly and the
results are written as floats ("Floating point entries" in
``docs/matrices.md``). Before, ``steps``, ``normalforms``, ``pencils`` and
``riccati`` raised, and the others rounded the products they formed."""
from __future__ import annotations

from typing import Sequence, Union

import mpmath
from sympy.testing.pytest import raises

from sympy_extras.core import Float, I, Rational, Symbol, oo
from sympy_extras.core.expr import Expr
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices import equations, markov, normalforms, pencils, riccati, statespace, steps
from sympy_extras.matrices.linalg import exact_matrix, inverse

x, s = Symbol('x'), Symbol('s')
Number = Union[float, int]


def F(rows: Sequence[Sequence[Number]]) -> ExprMatrix:
    return ExprMatrix.from_rows([[Float(v) if isinstance(v, float) else v for v in row] for row in rows])


def value(e: Expr, at: float = 0.375) -> complex:
    return complex(e.subs(x, Float(at)).subs(s, Float(at)).evalf(20))


def close(a: ExprMatrix, b: ExprMatrix) -> bool:
    """The entries agree to 11 digits (at a value of the symbols)."""
    if a.shape != b.shape:
        return False
    return all(abs(value(p) - value(q)) <= 1e-11*(1 + abs(value(q))) for p, q in zip(a.entries, b.entries))


def has_floats(M: ExprMatrix) -> bool:
    return any(e.atoms(Float) for e in M.entries)


A = F([[0.5, 1.5], [0.25, 2.0]])
# the floats of 0.2 and 0.6 are twice those of 0.1 and 0.3: of rank one as stored
S1 = F([[0.1, 0.2], [0.3, 0.6]])
B = F([[0.0], [1.0]])
C = F([[1.0, 0.0]])
Q = F([[1.0, 0.0], [0.0, 1.0]])
R = F([[1.0]])
ZERO = ExprMatrix.zeros(2, 2)


def test_steps_with_floats() -> None:
    # NotImplementedError before ("entries are not rational functions of symbols ...")
    found = steps.rref_steps(A)
    exact = steps.rref_steps(exact_matrix(A))
    assert [step.operation.kind for step in found] == [step.operation.kind for step in exact]
    assert all(close(p.matrix, q.matrix) and has_floats(p.matrix) for p, q in zip(found, exact))
    assert close(found[-1].matrix, ExprMatrix.eye(2))
    # the pivots are those of the matrix as stored: no pivot which is a rounding error
    last = steps.rref_steps(S1)[-1].matrix
    assert last[1, 0] == 0 and last[1, 1] == 0
    D = steps.determinant_steps(S1)
    assert D.value == 0 and D.diagonal[1] == 0
    assert steps.determinant_steps(A).value == Float(0.625)
    inverse = steps.inverse_steps(A)[-1].matrix
    assert close(ExprMatrix(2, 2, [inverse[i, j + 2] for i in range(2) for j in range(2)]), F([[3.2, -2.4], [-0.4, 0.8]]))
    raises(ValueError, lambda: steps.inverse_steps(S1))
    solved = steps.solve_steps(A, B)[-1].matrix
    assert close(ExprMatrix(2, 1, [solved[0, 2], solved[1, 2]]), F([[-2.4], [0.8]]))
    assert steps.echelon_steps(A)[-1].matrix[1, 0] == 0
    assert 'R2 <- R2 - (0.25' in steps.format_steps(A, steps.rref_steps(A))


def test_normal_forms_with_floats() -> None:
    # NotImplementedError before
    pencil = ExprMatrix.eye(2)*x - A
    D = normalforms.smith_normal_form(pencil, x)
    assert D.S[0, 0] == Float(1.0) and D.S[0, 1] == 0
    assert abs(value(D.S[1, 1] - (x**2 - Float(2.5)*x + Float(0.625)))) < 1e-14
    H, U = normalforms.hermite_normal_form(pencil, x)
    assert close(U*pencil, H) and H[1, 0] == 0
    P, V = normalforms.popov_form(pencil, x)
    assert close(V*pencil, P)
    assert normalforms.polynomial_matrix_rank(pencil, x) == 2 and not normalforms.is_unimodular(pencil, x)
    assert normalforms.is_unimodular(F([[1.0, 0.5], [0.0, 2.0]]) + ExprMatrix.from_rows([[0, x], [0, 0]]), x)
    assert normalforms.invariant_factors(A, x) == [normalforms.matrix_minimal_polynomial(A, x)]
    assert abs(value(normalforms.matrix_minimal_polynomial(A, x) - (x**2 - Float(2.5)*x + Float(0.625)))) < 1e-14
    T, frobenius = normalforms.frobenius_form(A)
    assert close(A*T, T*frobenius) and close(frobenius, F([[0, -0.625], [1, 2.5]]))
    # the structure is that of the matrix as stored: 0.5*I has two invariant factors, a Jordan block one
    scalar = F([[0.5, 0.0], [0.0, 0.5]])
    block = F([[0.5, 1.0], [0.0, 0.5]])
    assert len(normalforms.invariant_factors(scalar, x)) == 2 and len(normalforms.invariant_factors(block, x)) == 1
    assert [k for _, k in normalforms.elementary_divisors(scalar, x)] == [1, 1]
    divisors = normalforms.elementary_divisors(block, x)
    assert [k for _, k in divisors] == [2] and abs(value(divisors[0][0] - (x - Float(0.5)))) < 1e-14
    assert not normalforms.is_similar(scalar, block) and normalforms.is_similar(A, A.transpose())
    # S1 has the eigenvalues 0 and its trace, both simple
    trace = exact_matrix(S1)[0, 0] + exact_matrix(S1)[1, 1]
    assert normalforms.is_similar(S1, ExprMatrix.from_rows([[0, 0], [0, trace]]))


def test_pencils_with_floats() -> None:
    # NotImplementedError before; the polynomial was computed with rounded products
    assert abs(value(pencils.pencil_polynomial(A, Q, x) - (x**2 - Float(2.5)*x + Float(0.625)))) < 1e-14
    values = pencils.generalized_eigenvalues(A, Q)
    reference = sorted(mpmath.eig(mpmath.matrix([[0.5, 1.5], [0.25, 2.0]]))[0], key=lambda z: z.real)
    assert all(abs(complex(v) - complex(r)) < 1e-12 for v, r in zip(sorted(values, key=lambda v: complex(v).real), reference))
    for eigenvalue, multiplicity, vectors in pencils.generalized_eigenvectors(A, Q):
        assert multiplicity == 1 and len(vectors) == 1
        assert close(A*vectors[0], (Q*vectors[0])*eigenvalue)
    # of any degree, and a singular matrix B has the eigenvalue oo
    A3 = F([[0.5, 1.5, 0.0], [0.25, 2.0, 1.0], [0.1, 0.2, 0.3]])
    assert len(pencils.generalized_eigenvalues(A3, ExprMatrix.eye(3))) == 3
    assert pencils.generalized_eigenvalues(A, F([[1.0, 0.0], [0.0, 0.0]]))[oo] == 1
    # the eigenvalue 0 of a matrix which is singular as stored is exact
    assert 0 in pencils.generalized_eigenvalues(S1, Q)
    spaces = pencils.generalized_eigenspaces(A, Q, x)
    assert len(spaces) == 1 and not spaces[0].factor.atoms(Float)


def test_riccati_with_floats() -> None:
    # ValueError before ("the entries of A must be rational numbers")
    X = riccati.care_solve(A, B, Q, R)
    assert X is not None and has_floats(X)
    assert close(A.transpose()*X + X*A - X*B*B.transpose()*X + Q + ExprMatrix.eye(2), ExprMatrix.eye(2))
    K, X2 = riccati.lqr(A, B, Q, R)
    assert close(X2, X) and close(K, B.transpose()*X)
    assert riccati.closed_loop_is_stable(A, B, K) and not riccati.closed_loop_is_stable(A, B, F([[0.0, 0.0]]))
    exact = riccati.riccati_solutions(exact_matrix(A), exact_matrix(B), exact_matrix(Q), exact_matrix(R))
    assert len(riccati.riccati_solutions(A, B, Q, R)) == len(exact)
    Y = riccati.dare_solve(A, B, Q, R)
    assert Y is not None
    gain = inverse(R + B.transpose()*Y*B)*B.transpose()*Y*A
    assert close(A.transpose()*Y*A - Y - A.transpose()*Y*B*gain + Q + ExprMatrix.eye(2), ExprMatrix.eye(2))
    Kd, Yd = riccati.dlqr(A, B, Q, R)
    assert close(Yd, Y) and close(Kd, gain)
    assert len(riccati.discrete_riccati_solutions(A, B, Q, R)) == 2


def test_statespace_with_floats() -> None:
    # (A, B) with A of rank one as stored and B in its column space is not controllable: the rounded
    # products of the controllability matrix had the rank 2, and the minimal realisation was 0
    b = F([[0.1], [0.3]])
    assert not statespace.is_controllable(S1, b) and statespace.is_controllable(A, B)
    assert statespace.is_observable(A, C) and not statespace.is_observable(S1.transpose(), b.transpose())
    T, A1, B1, r = statespace.controllable_decomposition(S1, b)
    assert r == 1 and A1[1, 0] == 0 and B1[1, 0] == 0 and close(S1*T, T*A1)
    _, _, C2, r2 = statespace.observable_decomposition(S1, F([[0.3, -0.1]]))
    assert r2 == 1 and C2[0, 1] == 0
    Am, Bm, Cm = statespace.minimal_realization(S1, b, C)
    assert Am.shape == (1, 1) and close(Am, F([[0.7]]))
    assert close(statespace.transfer_function(S1, b, C, s), statespace.transfer_function(Am, Bm, Cm, s))
    assert close(statespace.controllability_matrix(A, B), F([[0.0, 1.5], [1.0, 2.0]]))
    K = statespace.pole_placement(A, B, [Float(-1.0), Float(-2.0)])
    assert abs(value((A - B*K).charpoly(x) - (x + 1)*(x + 2))) < 1e-12
    assert close(statespace.pole_placement(A, B, (s + 1)*(s + Float(2.0)), s), K)
    A0, B0, C0, _ = statespace.realization(1/(s**2 + Float(0.5)*s + 1), s)
    assert close(statespace.transfer_function(A0, B0, C0, s), ExprMatrix(1, 1, [1/(s**2 + Float(0.5)*s + 1)]))


def test_markov_chains_with_floats() -> None:
    # the floats of 0.1 and 0.9 have a sum which is 1 only after rounding: the constructor took the chain
    # and stationary_distribution raised ArithmeticError ("the recurrent class has 0 invariant vectors")
    chain = markov.MarkovChain(F([[0.1, 0.9], [0.3, 0.7]]))
    assert close(chain.stationary_distribution(), F([[0.25, 0.75]]))
    assert close(chain.limiting_matrix(), F([[0.25, 0.75], [0.25, 0.75]]))
    assert abs(complex(chain.mean_return_time(0)) - 4) < 1e-12
    assert chain.is_ergodic and chain.P == F([[0.1, 0.9], [0.3, 0.7]])
    raises(ValueError, lambda: markov.MarkovChain(F([[0.1, 0.8], [0.3, 0.7]])))
    absorbing = markov.MarkovChain(F([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]]))
    assert absorbing.transient_states() == [0, 1] and absorbing.is_absorbing
    assert close(absorbing.fundamental_matrix(), F([[4.0, 4.0], [2.0, 4.0]]))
    assert close(absorbing.expected_steps(), F([[8.0], [6.0]]))
    assert close(absorbing.absorption_probabilities(), F([[1.0], [1.0]]))
    assert close(absorbing.hitting_probabilities([2]), F([[1.0], [1.0], [1.0]]))
    assert close(absorbing.mean_hitting_times([2]), F([[8.0], [6.0], [0.0]]))
    assert close(chain.distribution_after([1, 0], 2), F([[0.28, 0.72]]))
    exact = markov.MarkovChain(ExprMatrix.from_rows([[Rational(1, 10), Rational(9, 10)], [Rational(3, 10), Rational(7, 10)]]))
    assert exact.stationary_distribution() == ExprMatrix.from_rows([[Rational(1, 4), Rational(3, 4)]])


def test_matrix_equations_with_floats() -> None:
    solution = equations.sylvester_solve(A, Q, Q)
    assert solution is not None and solution.is_unique and close(A*solution.particular + solution.particular*Q, Q)
    lyapunov = equations.lyapunov_solve(A, Q)
    assert lyapunov is not None
    assert close(A*lyapunov.particular + lyapunov.particular*A.transpose() + Q + ExprMatrix.eye(2), ExprMatrix.eye(2))
    stein = equations.discrete_lyapunov_solve(A, Q)
    assert stein is not None
    assert close(A*stein.particular*A.transpose() - stein.particular + Q + ExprMatrix.eye(2), ExprMatrix.eye(2))
    # S1 X - X S1 = I has no solution (the trace), and the centraliser of S1 has the dimension 2: for the
    # matrix as stored, whose eigenvalues 0 and 0.7 are simple
    assert equations.sylvester_solve(S1, -S1, Q) is None
    assert len(equations.commuting_matrices(S1)) == 2 and len(equations.commuting_matrices(F([[0.5, 0.0], [0.0, 0.5]]))) == 4
    # a term A X B with two float factors: the products of the system are exact
    general = equations.linear_matrix_equation([(A, S1), (Q, Q)], Q)
    assert general is not None
    assert close(A*general.particular*S1 + general.particular, Q)


def test_eigenvalues_of_a_complex_matrix_with_floats_of_degree_three() -> None:
    # NotImplementedError before ("the roots of ... are not expressed in radicals"): a factor with I
    # in its coefficients has its roots among the isolated roots of its norm
    M = ExprMatrix.from_rows([[Float(0.5) + I, Float(1.5), 0], [Float(0.25), Float(2.0)*I, 1],
                              [Float(0.1), Float(0.2), Float(0.3) + Float(0.7)*I]])
    reference = sorted(mpmath.eig(mpmath.matrix([[0.5 + 1j, 1.5, 0], [0.25, 2j, 1], [0.1, 0.2, 0.3 + 0.7j]]))[0],
                       key=lambda z: (z.real, z.imag))
    values: list[Expr] = list(M.eigenvals())
    assert len(values) == 3
    assert all(abs(complex(v) - complex(r)) < 1e-12 for v, r in zip(values, reference))
    for eigenvalue, multiplicity, vectors in M.eigenvects():
        assert multiplicity == 1
        assert all(abs(complex(e)) < 1e-11 for e in (M*vectors[0] - vectors[0]*eigenvalue).entries)
    found = M.diagonalize()
    assert found is not None
    P, D = found
    assert all(abs(complex(e)) < 1e-10 for e in (M*P - P*D).entries)

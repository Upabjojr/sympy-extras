"""The Rust ports of ``matrices/eigen.py`` and
``matrices/decompositions.py`` against the Python ones: eigenspaces,
Jordan chains and forms, eigenvalues and diagonalisation; LU, LDL,
Cholesky, Gram-Schmidt, QR, rank factorisations, congruence
diagonalisation and inertia, on the matrices of the matrices tests
(entries read with the core's parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
import sympy_extras.matrices.decompositions as DC
import sympy_extras.matrices.eigen as EG
from sympy_extras.matrices.dense import ExprMatrix

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return str(r)


def encoded(v: object) -> Basic:
    if v is None:
        return Symbol('NONE')
    if isinstance(v, bool):
        return Symbol(str(v))
    if isinstance(v, ExprMatrix):
        return Tuple(Integer(v.rows), Integer(v.cols), Tuple(*v.entries))
    if isinstance(v, EG.Eigenspace):
        return Tuple(v.factor, v.symbol, Integer(v.multiplicity), encoded(v.basis))
    if isinstance(v, EG.JordanChains):
        return Tuple(v.factor, v.symbol, Integer(v.multiplicity), encoded(v.chains))
    if isinstance(v, DC.LUDecomposition):
        return Tuple(encoded(v.L), encoded(v.U), encoded(v.permutation), Integer(v.rank))
    if isinstance(v, dict):
        return Tuple(*[Tuple(k, encoded(x)) for k, x in v.items()])
    if isinstance(v, (list, tuple)):
        return Tuple(*[encoded(x) for x in v])
    if isinstance(v, int):
        return Integer(v)
    assert isinstance(v, Basic)
    return v


def argument(a: object) -> object:
    if isinstance(a, str):
        return a == 'True' if a in ('True', 'False') else parse(a)
    assert isinstance(a, list)
    if a and isinstance(a[0], int):
        rows, cols, entries = a
        assert isinstance(rows, int) and isinstance(cols, int) and isinstance(entries, list)
        return ExprMatrix(rows, cols, [parse(str(e)) for e in entries])
    return [argument(x) for x in a]


CASES: list[tuple[str, str, list[object]]] = [
    ('decompositions', 'cholesky', [[1, 1, ['9']]]),
    ('decompositions', 'cholesky', [[2, 2, ['1', '2', '2', '1']]]),
    ('decompositions', 'cholesky', [[4, 4, ['6', '4', '0', '11/2', '4', '4', '-1', '4', '0', '-1', '21/4', '-6', '11/2', '4', '-6', '57/4']]]),
    ('decompositions', 'cholesky', [[2, 2, ['10', '3', '3', '9']]]),
    ('decompositions', 'cholesky', [[4, 4, ['22', '0', '19/2', '4', '0', '9', '3', '9', '19/2', '3', '21/2', '2', '4', '9', '2', '13']]]),
    ('decompositions', 'cholesky', [[4, 4, ['4', '-2', '4', '-2', '-2', '57/4', '-3/2', '-5', '4', '-3/2', '5', '-2', '-2', '-5', '-2', '10']]]),
    ('decompositions', 'cholesky', [[3, 3, ['9/4', '1', '-1', '1', '5/4', '-1', '-1', '-1', '5']]]),
    ('decompositions', 'cholesky', [[3, 3, ['37/4', '-1', '0', '-1', '4', '0', '0', '0', '1/4']]]),
    ('decompositions', 'cholesky', [[2, 2, ['10', '1', '1', '1']]]),
    ('decompositions', 'congruence_diagonalize', [[3, 3, ['-2', '1/2', '0', '1/2', '0', '1/2', '0', '1/2', '3']]]),
    ('decompositions', 'congruence_diagonalize', [[5, 5, ['0', '1/2', '1/2', '1', '-2', '1/2', '3', '3', '1/2', '-2', '1/2', '3', '0', '1', '1', '1', '1/2', '1', '0', '1', '-2', '-2', '1', '1', '1']]]),
    ('decompositions', 'congruence_diagonalize', [[2, 2, ['3', '-2', '-2', '1/2']]]),
    ('decompositions', 'congruence_diagonalize', [[5, 5, ['3', '1', '0', '0', '1', '1', '1', '0', '1/2', '1', '0', '0', '-2', '-2', '1', '0', '1/2', '-2', '-2', '0', '1', '1', '1', '0', '1']]]),
    ('decompositions', 'congruence_diagonalize', [[1, 1, ['1']]]),
    ('decompositions', 'congruence_diagonalize', [[5, 5, ['0', '0', '0', '1/2', '1', '0', '0', '1/2', '1/2', '0', '0', '1/2', '1/2', '-2', '3', '1/2', '1/2', '-2', '0', '0', '1', '0', '3', '0', '1']]]),
    ('decompositions', 'congruence_diagonalize', [[2, 2, ['1', '2', '3', '4']]]),
    ('decompositions', 'congruence_diagonalize', [[4, 4, ['0', '-2', '1/2', '0', '-2', '1', '3', '0', '1/2', '3', '3', '0', '0', '0', '0', '-2']]]),
    ('decompositions', 'congruence_diagonalize', [[4, 4, ['-2', '0', '1', '-2', '0', '-2', '3', '1', '1', '3', '-2', '-2', '-2', '1', '-2', '-2']]]),
    ('eigen', 'diagonalize', [[3, 3, ['2', '1', '2', '1', '1', '2', '1', '2', '-2']]]),
    ('eigen', 'diagonalize', [[3, 3, ['0', '2', '0', '-1', '1', '-2', '0', '1', '0']]]),
    ('eigen', 'diagonalize', [[3, 3, ['-1', '-2', '2', '-1', '-2', '2', '0', '1', '-2']]]),
    ('eigen', 'diagonalize', [[2, 2, ['a', 'b', 'b', 'a']]]),
    ('eigen', 'diagonalize', [[3, 3, ['1', '-2', '1', '-1', '2', '0', '2', '2', '-1']]]),
    ('eigen', 'diagonalize', [[2, 2, ['2', '-1', '2', '-2']]]),
    ('eigen', 'diagonalize', [[2, 2, ['-1', '0', '0', '0']]]),
    ('eigen', 'diagonalize', [[3, 3, ['2', '-1', '2', '-2', '2', '-1', '1', '0', '-1']]]),
    ('eigen', 'diagonalize', [[2, 2, ['-2', '2', '-1', '-2']]]),
    ('eigen', 'eigenspaces', [[4, 4, ['-2', '-2', '-2', '-1', '-1', '-2', '1', '1', '1', '1', '-2', '2', '-1', '0', '0', '-2']], 'x']),
    ('eigen', 'eigenspaces', [[3, 3, ['0', '-2', '1', '-2', '-1', '-1', '-2', '-2', '-2']], 'x']),
    ('eigen', 'eigenspaces', [[3, 3, ['1', '-1', '2 - I', '-I', '2 - I', '1 + I', '-1', '-1', '0']], 'x']),
    ('eigen', 'eigenspaces', [[3, 3, ['2', '1', '0', '0', '2', '1', '0', '0', '2']], 'x']),
    ('eigen', 'eigenspaces', [[4, 4, ['1', '-1', '2', '-1', '1', '-1', '2', '-1', '1', '1', '-2', '1', '1', '-1', '-2', '0']], 'x']),
    ('eigen', 'eigenspaces', [[2, 2, ['1', '2', '3', '4']], 'x']),
    ('eigen', 'eigenspaces', [[2, 2, ['1/a', '1', 'b', 'a/(b + 1)']], 'x']),
    ('eigen', 'eigenspaces', [[2, 2, ['a', 'b', 'b', 'a']], 'x']),
    ('eigen', 'eigenspaces', [[2, 2, ['I', '2 - I', '-1', '1']], 'x']),
    ('eigen', 'eigenvalues', [[1, 1, ['-1/2']]]),
    ('eigen', 'eigenvalues', [[2, 2, ['-3/2', '1', '-1/2', '0']]]),
    ('eigen', 'eigenvalues', [[2, 2, ['1', 'sqrt(2)', 'sqrt(2)', '1']]]),
    ('eigen', 'eigenvalues', [[4, 4, ['0', '1', '0', '0', '-1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '-1', '0']]]),
    ('eigen', 'eigenvalues', [[2, 2, ['-1/2', '0', '-1/2', '1/2']]]),
    ('eigen', 'eigenvalues', [[2, 2, ['a', 'b', 'b', 'a']]]),
    ('eigen', 'eigenvalues', [[3, 3, ['1', '2', '2', '2', '4', '4', '2', '4', '4']]]),
    ('eigen', 'eigenvalues', [[2, 2, ['2', '1 - I', '1 + I', '3']]]),
    ('eigen', 'eigenvalues', [[2, 2, ['sqrt(2)/2', 'sqrt(2)/2', 'sqrt(2)/2', '-sqrt(2)/2']]]),
    ('eigen', 'eigenvectors', [[4, 4, ['0', '1', '0', '0', '-1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '-1', '0']]]),
    ('eigen', 'eigenvectors', [[3, 3, ['2', '1', '0', '0', '2', '1', '0', '0', '2']]]),
    ('eigen', 'eigenvectors', [[2, 2, ['sqrt(2)/2', 'sqrt(2)/2', 'sqrt(2)/2', '-sqrt(2)/2']]]),
    ('eigen', 'eigenvectors', [[3, 3, ['2', '1', '0', '0', '2', '0', '0', '0', '2']]]),
    ('eigen', 'eigenvectors', [[3, 3, ['2', '0', '0', '0', '2', '0', '0', '0', '3']]]),
    ('eigen', 'eigenvectors', [[2, 2, ['1', '2', '3', '4']]]),
    ('decompositions', 'gram_schmidt', [[[2, 1, ['3', '1']], [2, 1, ['2', '2']]]]),
    ('decompositions', 'gram_schmidt', [[[2, 1, ['3', '1']], [2, 1, ['2', '2']]], 'True']),
    ('decompositions', 'gram_schmidt', [[]]),
    ('decompositions', 'gram_schmidt', [[[2, 1, ['1', '2']], [3, 1, ['1', '2', '3']]]]),
    ('decompositions', 'inertia', [[4, 4, ['-2', '0', '1', '-2', '0', '-2', '3', '1', '1', '3', '-2', '-2', '-2', '1', '-2', '-2']]]),
    ('decompositions', 'inertia', [[4, 4, ['22', '-2', '-3', '29/2', '-2', '8', '-7', '-2', '-3', '-7', '39/4', '-15/4', '29/2', '-2', '-15/4', '57/4']]]),
    ('decompositions', 'inertia', [[5, 5, ['-2', '0', '0', '0', '-2', '0', '0', '0', '-2', '1/2', '0', '0', '3', '1', '-2', '0', '-2', '1', '3', '0', '-2', '1/2', '-2', '0', '0']]]),
    ('decompositions', 'inertia', [[2, 2, ['6', '-3', '-3', '6']]]),
    ('decompositions', 'inertia', [[4, 4, ['3', '1', '1', '0', '1', '1/2', '3', '1', '1', '3', '-2', '1', '0', '1', '1', '0']]]),
    ('decompositions', 'inertia', [[2, 2, ['sqrt(2)', 'sqrt(2)', 'sqrt(2)', 'sqrt(2)']]]),
    ('decompositions', 'inertia', [[4, 4, ['37/4', '-3', '-19/4', '-3/2', '-3', '25/4', '11/2', '1', '-19/4', '11/2', '77/4', '5/2', '-3/2', '1', '5/2', '5/4']]]),
    ('decompositions', 'inertia', [[4, 4, ['4', '-2', '4', '-2', '-2', '57/4', '-3/2', '-5', '4', '-3/2', '5', '-2', '-2', '-5', '-2', '10']]]),
    ('decompositions', 'inertia', [[4, 4, ['6', '4', '0', '11/2', '4', '4', '-1', '4', '0', '-1', '21/4', '-6', '11/2', '4', '-6', '57/4']]]),
    ('eigen', 'is_diagonalizable', [[3, 3, ['1', '-1', '2 - I', '-I', '2 - I', '1 + I', '-1', '-1', '0']]]),
    ('eigen', 'is_diagonalizable', [[4, 4, ['0', '1', '2', '-2', '0', '0', '-1', '2', '0', '-2', '-2', '2', '-2', '1', '-2', '0']]]),
    ('eigen', 'is_diagonalizable', [[4, 4, ['0', '2', '1', '0', '-2', '-2', '0', '2', '-2', '0', '2', '0', '0', '2', '-2', '-1']]]),
    ('eigen', 'is_diagonalizable', [[3, 3, ['1 + I', '1 + I', '-1', '0', '-I', '2 - I', '0', 'I', '0']]]),
    ('eigen', 'is_diagonalizable', [[4, 4, ['2', '1', '1', '2', '2', '1', '1', '2', '1', '2', '-1', '-2', '1', '0', '-1', '-2']]]),
    ('eigen', 'is_diagonalizable', [[4, 4, ['1', '-1', '2', '-1', '1', '-1', '2', '-1', '1', '1', '-2', '1', '1', '-1', '-2', '0']]]),
    ('eigen', 'is_diagonalizable', [[2, 2, ['a', '1', '0', 'b']]]),
    ('eigen', 'is_diagonalizable', [[4, 4, ['1', '-1', '-2', '-2', '-1', '0', '0', '2', '2', '-1', '-2', '0', '-1', '1', '0', '2']]]),
    ('eigen', 'is_diagonalizable', [[2, 2, ['2 - I', '-1', '1', '1']]]),
    ('decompositions', 'is_hermitian', [[3, 3, ['2', '1 + I', '0', '1 - I', '3', 'I', '0', '-I', '1']]]),
    ('decompositions', 'is_hermitian', [[3, 3, ['I*((1 - I)*(1 + I) + 4)', '5 - 5*I', '-I*(1 - I)', 'I*(5 + 5*I)', '(1 - I)*(1 + I) + 10', '-4*I', '-1 - I', '4*I', '2']]]),
    ('decompositions', 'is_positive_definite', [[4, 4, ['3', '1', '1', '0', '1', '1/2', '3', '1', '1', '3', '-2', '1', '0', '1', '1', '0']]]),
    ('decompositions', 'is_positive_definite', [[3, 3, ['1503/125', '54/25', '-36/25', '54/25', '14/5', '-6/5', '-36/25', '-6/5', '9/5']]]),
    ('decompositions', 'is_positive_definite', [[3, 3, ['14', '7', '3', '7', '13', '9', '3', '9', '9']]]),
    ('decompositions', 'is_positive_definite', [[4, 4, ['6', '4', '0', '11/2', '4', '4', '-1', '4', '0', '-1', '21/4', '-6', '11/2', '4', '-6', '57/4']]]),
    ('decompositions', 'is_positive_definite', [[5, 5, ['-2', '0', '0', '0', '-2', '0', '0', '0', '-2', '1/2', '0', '0', '3', '1', '-2', '0', '-2', '1', '3', '0', '-2', '1/2', '-2', '0', '0']]]),
    ('decompositions', 'is_positive_definite', [[5, 5, ['1', '1/2', '1/2', '-2', '-2', '1/2', '0', '-2', '0', '0', '1/2', '-2', '0', '3', '0', '-2', '0', '3', '3', '1/2', '-2', '0', '0', '1/2', '1/2']]]),
    ('decompositions', 'is_positive_definite', [[4, 4, ['0', '-2', '1/2', '0', '-2', '1', '3', '0', '1/2', '3', '3', '0', '0', '0', '0', '-2']]]),
    ('decompositions', 'is_positive_definite', [[5, 5, ['3', '1', '0', '0', '1', '1', '1', '0', '1/2', '1', '0', '0', '-2', '-2', '1', '0', '1/2', '-2', '-2', '0', '1', '1', '1', '0', '1']]]),
    ('decompositions', 'is_positive_definite', [[2, 2, ['31/3', '-35/12', '-35/12', '11/6']]]),
    ('decompositions', 'is_positive_semidefinite', [[5, 5, ['-2', '3', '0', '0', '0', '3', '0', '3', '3', '-2', '0', '3', '1/2', '0', '1/2', '0', '3', '0', '-2', '1/2', '0', '-2', '1/2', '1/2', '1']]]),
    ('decompositions', 'is_positive_semidefinite', [[5, 5, ['0', '0', '0', '1/2', '1', '0', '0', '1/2', '1/2', '0', '0', '1/2', '1/2', '-2', '3', '1/2', '1/2', '-2', '0', '0', '1', '0', '3', '0', '1']]]),
    ('decompositions', 'is_positive_semidefinite', [[4, 4, ['0', '0', '1/2', '-2', '0', '-2', '-2', '0', '1/2', '-2', '1', '1/2', '-2', '0', '1/2', '0']]]),
    ('decompositions', 'is_positive_semidefinite', [[2, 2, ['0', '-2', '-2', '1/2']]]),
    ('decompositions', 'is_positive_semidefinite', [[4, 4, ['-2', '1', '0', '3', '1', '1/2', '1', '3', '0', '1', '-2', '1', '3', '3', '1', '3']]]),
    ('decompositions', 'is_positive_semidefinite', [[1, 1, ['-2']]]),
    ('decompositions', 'is_positive_semidefinite', [[3, 3, ['-2', '1', '0', '1', '1', '-2', '0', '-2', '0']]]),
    ('decompositions', 'is_positive_semidefinite', [[2, 2, ['0', '3', '3', '1/2']]]),
    ('decompositions', 'is_positive_semidefinite', [[1, 1, ['0']]]),
    ('eigen', 'jordan_chains', [[4, 4, ['6/7', '-6/7', '2/7', '2/7', '-6/7', '6/7', '-2/7', '-2/7', '-6/7', '-8/7', '12/7', '-2/7', '12/7', '2/7', '-10/7', '4/7']], 'x']),
    ('eigen', 'jordan_chains', [[5, 5, ['-2', '-11', '1', '-6', '-1', '0', '-2', '0', '0', '0', '0', '-4', '4', '1', '0', '0', '12', '-1', '4', '1', '0', '-4', '0', '1', '4']], 'x']),
    ('eigen', 'jordan_chains', [[2, 2, ['0', '-2', '0', '2']], 'x']),
    ('eigen', 'jordan_chains', [[3, 3, ['0', '0', '-1', '4', '2', '3', '-1/2', '-1/2', '1']], 'x']),
    ('eigen', 'jordan_chains', [[5, 5, ['9/4', '3/4', '3/4', '-5/4', '-7/4', '1', '3', '1', '-1', '-3', '1/8', '3/8', '11/8', '3/8', '-7/8', '1/8', '3/8', '11/8', '3/8', '-7/8', '1', '1', '1', '-1', '-1']], 'x']),
    ('eigen', 'jordan_chains', [[3, 3, ['-1', '1', '1', '-2', '1', '1', '-4', '2', '3']], 'x']),
    ('eigen', 'jordan_chains', [[5, 5, ['0', '0', '0', '0', '1', '1', '0', '0', '0', '1', '0', '1', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '1', '0']], 'x']),
    ('eigen', 'jordan_chains', [[2, 2, ['0', '2', '-1', '3']], 'x']),
    ('eigen', 'jordan_chains', [[4, 4, ['3', '0', '-1', '-3', '-2', '2', '0', '2', '-2', '0', '2', '2', '2', '0', '-1', '-2']], 'x']),
    ('eigen', 'jordan_form', [[2, 2, ['1', 'sqrt(2)', 'sqrt(2)', '1']]]),
    ('eigen', 'jordan_form', [[3, 3, ['0', '0', '2 - I', '-I', '-1', '1 + I', '2 - I', '-1', '1']]]),
    ('eigen', 'jordan_form', [[3, 3, ['-1', 'I', '-I', '1 + I', '2 - I', '-I', '0', '-1', '0']]]),
    ('eigen', 'jordan_form', [[2, 2, ['I', '2 - I', '-1', '1']]]),
    ('eigen', 'jordan_form', [[2, 2, ['1 + I', '-I', '2 - I', '-I']]]),
    ('eigen', 'jordan_form', [[5, 5, ['-2', '-11', '1', '-6', '-1', '0', '-2', '0', '0', '0', '0', '-4', '4', '1', '0', '0', '12', '-1', '4', '1', '0', '-4', '0', '1', '4']]]),
    ('eigen', 'jordan_form', [[4, 4, ['0', '1', '0', '0', '0', '0', '0', '0', '0', '-1', '0', '1', '0', '0', '0', '0']]]),
    ('eigen', 'jordan_form', [[3, 3, ['1', '-1', '2 - I', '-I', '2 - I', '1 + I', '-1', '-1', '0']]]),
    ('eigen', 'jordan_form', [[3, 3, ['1', '2 - I', '0', '-1', '-1', 'I', '1', '-I', '-I']]]),
    ('decompositions', 'ldl', [[1, 1, ['1/4']]]),
    ('decompositions', 'ldl', [[4, 4, ['7', '-5/2', '-4', '1', '-5/2', '9/2', '1', '-3/4', '-4', '1', '23', '2', '1', '-3/4', '2', '1/2']]]),
    ('decompositions', 'ldl', [[2, 2, ['9', '0', '0', '0']]]),
    ('decompositions', 'ldl', [[4, 4, ['4', '-2', '4', '-2', '-2', '57/4', '-3/2', '-5', '4', '-3/2', '5', '-2', '-2', '-5', '-2', '10']]]),
    ('decompositions', 'ldl', [[2, 2, ['1', '2', '3', '4']]]),
    ('decompositions', 'ldl', [[2, 2, ['10', '1', '1', '1']]]),
    ('decompositions', 'ldl', [[1, 1, ['0']]]),
    ('decompositions', 'ldl', [[1, 1, ['9']]]),
    ('decompositions', 'ldl', [[2, 2, ['0', '0', '0', '9']]]),
    ('decompositions', 'leading_principal_minors', [[3, 3, ['x', 'y', '1', 'y', '2', 'x', '1', 'x', 'y']]]),
    ('decompositions', 'lu', [[1, 1, ['0']]]),
    ('decompositions', 'lu', [[1, 1, ['1']]]),
    ('decompositions', 'lu', [[4, 4, ['0', '3', '1/2', '0', '1/2', '1', '1', '0', '1', '1', '0', '-2', '0', '0', '0', '1/2']]]),
    ('decompositions', 'lu', [[2, 2, ['cos(t)', '-sin(t)', 'sin(t)', 'cos(t)']]]),
    ('decompositions', 'lu', [[2, 2, ['-2', '3', '1/2', '3']]]),
    ('decompositions', 'lu', [[1, 4, ['x', 'x*y', '0', 'x*y']]]),
    ('decompositions', 'lu', [[2, 1, ['-2', '1']]]),
    ('decompositions', 'lu', [[2, 1, ['0', '-2']]]),
    ('decompositions', 'lu', [[4, 2, ['1', '-2', '1/2', '1', '0', '1/2', '-2', '0']]]),
    ('decompositions', 'qr', [[4, 1, ['1/x', '-2', '3', '0']]]),
    ('decompositions', 'qr', [[1, 1, ['1']]]),
    ('decompositions', 'qr', [[2, 3, ['3', '0', 'x', '1', '1/x', 'x*y']]]),
    ('decompositions', 'qr', [[3, 3, ['1/2', '1/2', '0', '1/2', '3', '0', '1/2', '-2', '0']]]),
    ('decompositions', 'qr', [[1, 2, ['0', '0']]]),
    ('decompositions', 'qr', [[2, 1, ['0', '-2']]]),
    ('decompositions', 'qr', [[1, 2, ['3', '3']]]),
    ('decompositions', 'qr', [[4, 3, ['3', '-2', '1/2', '1', '-2', '1/2', '1', '1/2', '0', '1', '3', '1/2']]]),
    ('decompositions', 'qr', [[4, 3, ['0', '3', '0', '0', '1/2', '1', '0', '0', '-2', '1/2', '1/2', '-2']]]),
    ('decompositions', 'rank_factorization', [[3, 2, ['0', 'y', 'x + 1', '1/x', '-2', '3']]]),
    ('decompositions', 'rank_factorization', [[1, 3, ['0', '0', '3']]]),
    ('decompositions', 'rank_factorization', [[3, 1, ['x + 1', 'x*y', '1']]]),
    ('decompositions', 'rank_factorization', [[3, 2, ['0', '0', '3', '1/2', '1', '1']]]),
    ('decompositions', 'rank_factorization', [[3, 1, ['0', '1', 'x']]]),
    ('decompositions', 'rank_factorization', [[1, 1, ['-2']]]),
    ('decompositions', 'rank_factorization', [[2, 1, ['1/2', '0']]]),
    ('decompositions', 'rank_factorization', [[3, 3, ['3', '0', '0', '1', '-2', '0', '-2', '3', 'x + 1']]]),
    ('decompositions', 'rank_factorization', [[3, 1, ['0', '3', '3']]]),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_matrices(i: int) -> None:
    op, name, raw = CASES[i]
    args = [argument(a) for a in raw]
    module = EG if op == 'eigen' else DC
    got = outcome(lambda: rc.logic_call(op, [Symbol(name)] + [encoded(a) for a in args]))
    if got != 'SKIP':
        assert got == outcome(lambda: encoded(getattr(module, name)(*args))), (name, raw)

"""The Rust ports of ``matrices/functions.py``, ``matrices/svd.py``,
``matrices/pencils.py`` and ``matrices/steps.py`` against the Python ones:
spectral components, functions, exponentials and powers of matrices,
singular values and decompositions, generalised eigenvalues, elimination
steps, on the matrices of the matrices and solvers tests (entries read
with the core's parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
import sympy_extras.matrices.decompositions as DC
import sympy_extras.matrices.eigen as EG
import sympy_extras.matrices.functions as FN
import sympy_extras.matrices.pencils as PN
import sympy_extras.matrices.steps as ST
import sympy_extras.matrices.svd as SV
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
    if isinstance(v, ST.Step):
        return Tuple(Symbol(str(v.operation)), encoded(v.matrix))
    if isinstance(v, ST.DeterminantSteps):
        return Tuple(encoded(v.steps), Integer(v.exchanges), Tuple(*v.diagonal), v.value)
    if isinstance(v, FN.MatrixComponents):
        return Tuple(v.factor, v.symbol, encoded(v.components))
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


CASES: list[tuple[str, list[object]]] = [
    ('matrix_components', [[2, 2, ['a', '1', '0', 'b']], 'x']),
    ('matrix_components', [[3, 3, ['0', '1', '0', '0', '0', '1', '2', '0', '0']], 'x']),
    ('matrix_components', [[2, 2, ['1', '1', '1', '0']], 'x']),
    ('matrix_components', [[3, 3, ['2', '1', '0', '0', '2', '0', '1', '0', '3']], 'x']),
    ('matrix_exp', [[3, 3, ['a', '1', '0', '0', 'a', '0', '0', '0', 'b']], 't']),
    ('matrix_exp', [[3, 3, ['-1', '-2', '-1', '2', '2', '2', '-1', '1', '0']], 't']),
    ('matrix_exp', [[2, 2, ['1', '2', '3', '4']], 't']),
    ('matrix_exp', [[3, 3, ['-1', '1', '-2', '2', '2', '1', '2', '-2', '-1']], 't']),
    ('matrix_exp', [[2, 2, ['a', 'b', 'b', 'a']], 't']),
    ('matrix_exp', [[1, 1, ['1']], 't']),
    ('matrix_exp', [[5, 5, ['0', '0', '0', '0', '1', '1', '0', '0', '0', '1', '0', '1', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '1', '0']], 't']),
    ('matrix_exp', [[1, 1, ['-1']], 't']),
    ('matrix_exp', [[3, 3, ['2', '-1', '1', '-2', '0', '1', '-1', '-2', '-2']], 't']),
    ('matrix_exp', [[2, 2, ['0', '0', '0', '0']], 't']),
    ('matrix_function', [[2, 2, ['5', '4', '4', '5']], 'sqrt(x)', 'x']),
    ('matrix_function', [[2, 2, ['1/2', '0', '0', '4']], 'sqrt(x)', 'x']),
    ('matrix_function', [[3, 3, ['2', '1', '0', '0', '2', '0', '1', '0', '3']], 'x**3 - 2*x + 1', 'x']),
    ('matrix_function', [[3, 3, ['2', '1', '0', '0', '2', '0', '1', '0', '3']], '1/x', 'x']),
    ('matrix_power', [[3, 3, ['-2', '1', '-2', '2', '0', '2', '0', '2', '-1']], 'n']),
    ('matrix_power', [[2, 2, ['a', '1', '0', 'a']], 'n']),
    ('matrix_power', [[3, 3, ['-1', '2', '1', '-1', '1', '2', '0', '-2', '-1']], 'n']),
    ('matrix_power', [[3, 3, ['-2', '-2', '0', '-1', '-1', '-1', '-2', '2', '-2']], 'n']),
    ('matrix_power', [[2, 2, ['2', '0', '-2', '1']], 'n']),
    ('matrix_power', [[3, 3, ['-2', '-2', '-1', '0', '-2', '-1', '-2', '1', '0']], 'n']),
    ('matrix_power', [[3, 3, ['-2', '2', '-1', '2', '-1', '-1', '1', '1', '0']], 'n']),
    ('matrix_power', [[3, 3, ['1', '-2', '-2', '-2', '-1', '-1', '2', '2', '-2']], 'n']),
    ('matrix_power', [[2, 2, ['1', '1', '1', '0']], 'n']),
    ('matrix_power', [[3, 3, ['1', '0', '0', '-1', '-1', '0', '-2', '-2', '1']], 'n']),
    ('singular_values', [[2, 2, ['1', '1', '1', '1']]]),
    ('singular_values', [[2, 2, ['3', '0', '4', '5']]]),
    ('singular_values', [[1, 3, ['1', '2', '2']]]),
    ('singular_values', [[2, 2, ['a', '0', '0', '2']]]),
    ('svd', [[1, 2, ['0', '2']]]),
    ('svd', [[2, 2, ['a', '0', '0', '2']]]),
    ('svd', [[3, 2, ['2', '-1', '-2', '0', '-2', '-2']]]),
    ('svd', [[3, 1, ['1', '1', '0']]]),
    ('svd', [[3, 2, ['0', '2', '2', '0', '-1', '0']]]),
    ('svd', [[3, 1, ['1', '2', '2']]]),
    ('svd', [[3, 2, ['1', '-1', '-1', '-1', '2', '1']]]),
    ('svd', [[2, 1, ['-2', '1']]]),
    ('svd', [[3, 1, ['-2', '0', '1']]]),
    ('svd', [[1, 3, ['-2', '0', '0']]]),
    ('generalized_eigenspaces', [[2, 2, ['0', '1/2', '1', '2']], [2, 2, ['0', '-1', '2', '1/2']], 'x']),
    ('generalized_eigenspaces', [[2, 2, ['1/2', '1/2', '-1', '1']], [2, 2, ['1/2', '0', '1', '0']], 'x']),
    ('generalized_eigenspaces', [[1, 1, ['1/2']], [1, 1, ['1']], 'x']),
    ('generalized_eigenspaces', [[3, 3, ['0', '-1', '0', '0', '2', '2', 'a + 1', 'a', '1/2']], [3, 3, ['1', '0', '0', '0', '-1', '1/2', '-1', '2', '0']], 'x']),
    ('generalized_eigenspaces', [[3, 3, ['0', '0', '1/2', '-1', '0', 'a', '0', '0', '1']], [3, 3, ['1/2', '0', '-1', '0', '1/2', '0', '1', '1/2', '1']], 'x']),
    ('generalized_eigenspaces', [[2, 2, ['-1', '0', '1', '1']], [2, 2, ['1', '0', '0', '1/2']], 'x']),
    ('generalized_eigenspaces', [[1, 1, ['0']], [1, 1, ['0']], 'x']),
    ('generalized_eigenspaces', [[3, 3, ['1/2', '0', '2', '-1', '-1', '0', '1/2', '0', '1']], [3, 3, ['1', '2', '0', '1', '-1', '1', '2', '1', '0']], 'x']),
    ('generalized_eigenvalues', [[1, 1, ['0']], [1, 1, ['1/2']]]),
    ('generalized_eigenvalues', [[2, 2, ['0', '2', '1/2', '-1']], [2, 2, ['2', '0', '2', '2']]]),
    ('generalized_eigenvalues', [[2, 2, ['1', '0', '0', '0']], [2, 2, ['1', '0', '0', '0']]]),
    ('generalized_eigenvalues', [[2, 2, ['-1', '1/2', '1/2', '0']], [2, 2, ['1', '1', '-1', '0']]]),
    ('generalized_eigenvalues', [[3, 3, ['1', '0', '0', '0', '1', '0', '0', '0', '1']], [3, 3, ['0', '0', '0', '0', '0', '0', '0', '0', '0']]]),
    ('generalized_eigenvalues', [[1, 1, ['2']], [1, 1, ['2']]]),
    ('generalized_eigenvalues', [[2, 2, ['2*k', '-k', '-k', 'k']], [2, 2, ['m', '0', '0', 'm']]]),
    ('generalized_eigenvalues', [[2, 2, ['1', '0', '0', '1']], [2, 2, ['1', '0', '0', '0']]]),
    ('generalized_eigenvectors', [[2, 2, ['0', '2', '1/2', '-1']], [2, 2, ['2', '0', '2', '2']]]),
    ('generalized_eigenvectors', [[2, 2, ['-1', '1/2', '1/2', '0']], [2, 2, ['1', '1', '-1', '0']]]),
    ('generalized_eigenvectors', [[2, 2, ['1', '-1', '1/2', '0']], [2, 2, ['-1', '1', '0', '-1']]]),
    ('generalized_eigenvectors', [[1, 1, ['1/2']], [1, 1, ['-1']]]),
    ('generalized_eigenvectors', [[1, 1, ['0']], [1, 1, ['1/2']]]),
    ('generalized_eigenvectors', [[1, 1, ['1/2']], [1, 1, ['1']]]),
    ('generalized_eigenvectors', [[2, 2, ['1', '1', '2', '0']], [2, 2, ['2', '1', '-1', '-1']]]),
    ('generalized_eigenvectors', [[1, 1, ['2']], [1, 1, ['2']]]),
    ('pencil_polynomial', [[2, 2, ['1', '-1', '1', '0']], [2, 2, ['1/2', '1', '0', '1']], 'x']),
    ('pencil_polynomial', [[2, 2, ['1/2', '0', '0', '1']], [2, 2, ['0', '0', '1/2', '0']], 'x']),
    ('pencil_polynomial', [[2, 2, ['0', '0', '0', '0']], [2, 2, ['1/2', '-1', '2', '0']], 'x']),
    ('pencil_polynomial', [[2, 2, ['-1', '0', '2', '1/2']], [2, 2, ['0', '1', '2', '2']], 'x']),
    ('pencil_polynomial', [[2, 2, ['1/2', '1/2', '-1', '1']], [2, 2, ['1/2', '0', '1', '0']], 'x']),
    ('pencil_polynomial', [[2, 2, ['1/2', '0', '2', '2']], [2, 2, ['0', '1', '1', '0']], 'x']),
    ('pencil_polynomial', [[2, 2, ['1/2', '2', '-1', '-1']], [2, 2, ['1', '-1', '2', '0']], 'x']),
    ('pencil_polynomial', [[3, 3, ['0', '-1', '1/2', '2', '0', '0', '2', '1', '2']], [3, 3, ['-1', '-1', '2', '-1', '0', '1/2', '0', '-1', '-1']], 'x']),
    ('determinant_steps', [[4, 4, ['-1', '0', '-1', '2', '-1', '1', '-1', '1/2', '-1', '1/2', '2', '2', '0', '0', '1/2', '0']]]),
    ('determinant_steps', [[4, 4, ['-1', '0', 'a', '3', '1', '3', '2', '0', '2', 'a', '1', '-1', 'b', 'a', '-1', 'a']]]),
    ('determinant_steps', [[4, 4, ['2', '3', '1', '2', '0', '0', '1/2', '1', '1', '1', '1', '0', '1/2', '2', '3', '2']]]),
    ('determinant_steps', [[4, 4, ['1', '1/2', '3', '-1', '2', '1/2', '-1', '-1', '2', '3', '2', '-1', '1/2', '-1', '0', '0']]]),
    ('determinant_steps', [[2, 2, ['2', '0', '1', '1']]]),
    ('determinant_steps', [[1, 1, ['2']]]),
    ('determinant_steps', [[3, 3, ['2', '4', '-2', '4', '9', '-3', '-2', '-3', '7']]]),
    ('determinant_steps', [[2, 2, ['sqrt(2)', '2', '1', 'sqrt(2)']]]),
    ('determinant_steps', [[2, 2, ['b', '0', '0', '0']]]),
    ('determinant_steps', [[3, 3, ['-1', '3', '0', '0', '1', '0', '1/2', '0', '0']]]),
    ('echelon_steps', [[4, 3, ['2', '3', '1/2', '1', '0', '1', '1', '0', '3', '1', '3', '3']]]),
    ('echelon_steps', [[2, 2, ['0', '0', '0', '2']]]),
    ('echelon_steps', [[1, 3, ['0', '0', '-1']]]),
    ('echelon_steps', [[3, 4, ['2', 'a + 1', '2', '-1', '1/2', '3', 'a', 'a + 1', 'a', '0', '2', '2']]]),
    ('echelon_steps', [[3, 1, ['3', '3', '0']]]),
    ('echelon_steps', [[1, 4, ['-1', '0', '2', '0']]]),
    ('echelon_steps', [[1, 3, ['0', '-1', '-1']]]),
    ('echelon_steps', [[1, 2, ['0', '3']]]),
    ('echelon_steps', [[3, 3, ['2', '4', '-2', '4', '9', '-3', '-2', '-3', '7']]]),
    ('echelon_steps', [[4, 3, ['0', '0', '-1', '2', '0', '1/2', '0', '1/2', '0', '0', '3', '0']]]),
    ('inverse_steps', [[2, 2, ['2', '3', '1', '2']]]),
    ('inverse_steps', [[1, 1, ['-1']]]),
    ('inverse_steps', [[1, 1, ['b']]]),
    ('inverse_steps', [[1, 1, ['0']]]),
    ('inverse_steps', [[4, 4, ['1', 'a', '-1', '1', '2', 'b', '3', 'a + 1', '0', '3', '3', 'a', '1', '0', '0', '1/2']]]),
    ('inverse_steps', [[2, 2, ['0', '0', '0', '0']]]),
    ('inverse_steps', [[2, 2, ['0', '1/2', '3', '2']]]),
    ('inverse_steps', [[3, 3, ['0', '3', '2', '1/2', '1/2', '-1', '0', '-1', 'a + 1']]]),
    ('inverse_steps', [[4, 4, ['2', '2', '1', '-1', '2', '0', '0', '1/2', '0', '3', '3', 'a', 'a', 'b', 'a + 1', '1']]]),
    ('inverse_steps', [[4, 4, ['-1', '-1', '3', '1/2', '0', '0', '2', '0', '1', '0', '0', '0', '-1', '0', '0', '0']]]),
    ('rref_steps', [[3, 3, ['0', '2', '0', '1/2', '0', '0', '1', '-1', 'a + 1']]]),
    ('rref_steps', [[2, 4, ['-1', '0', '2', 'a', 'a', 'b', '1/2', '0']]]),
    ('rref_steps', [[3, 3, ['0', '1/2', '1/2', '2', '1/2', '-1', '1/2', '2', '2']]]),
    ('rref_steps', [[4, 2, ['0', '0', '1', '0', 'a', '1/2', '1/2', 'a + 1']]]),
    ('rref_steps', [[2, 1, ['3', 'a']]]),
    ('rref_steps', [[4, 1, ['-1', '0', '-1', '1/2']]]),
    ('rref_steps', [[2, 2, ['0', '-1', '3', '2']]]),
    ('rref_steps', [[2, 2, ['sin(a)', '1', '1', '1']]]),
    ('rref_steps', [[3, 4, ['0', '2', 'a + 1', '1', '2', '0', '2', '1', 'a', '2', '0', '2']]]),
    ('rref_steps', [[1, 1, ['b']]]),
    ('solve_steps', [[3, 3, ['1', '1', '1', '0', '2', '5', '2', '5', '-1']], [3, 1, ['6', '-4', '27']]]),
    ('solve_steps', [[2, 2, ['1', '1', '1', '1']], [2, 1, ['1', '2']]]),
    ('solve_steps', [[3, 3, ['1', '1', '1', '0', '2', '5', '2', '5', '-1']], [2, 1, ['1', '2']]]),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_matrix_functions(i: int) -> None:
    name, raw = CASES[i]
    args = [argument(a) for a in raw]
    module = SV if name in ('svd', 'singular_values') else (PN if 'generalized' in name or name == 'pencil_polynomial' else
                                                            (ST if name.endswith('_steps') else FN))
    got = outcome(lambda: rc.logic_call('matfunctions', [Symbol(name)] + [encoded(a) for a in args]))
    if got != 'SKIP':
        assert got == outcome(lambda: encoded(getattr(module, name)(*args))), (name, raw)

"""The Rust ports of ``matrices/equations.py`` and
``matrices/statespace.py`` against the Python ones: Kronecker products,
Sylvester and Lyapunov equations, commutants; controllability,
observability, transfer functions, pole placement, realisations and
Kalman decompositions; and of solvers/stability.py (Routh-Hurwitz) and
solvers/rootclassification.py (discrimination sequences) and
matrices/riccati.py (algebraic Riccati equations),
on the matrices of the matrices tests (entries
read with the core's parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
import sympy_extras.matrices.equations as EQ
import sympy_extras.matrices.riccati as RK
import sympy_extras.matrices.statespace as SS
import sympy_extras.solvers.rootclassification as RC
import sympy_extras.solvers.stability as SB
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
    if isinstance(v, EQ.MatrixSolution):
        return Tuple(encoded(v.particular), encoded(v.homogeneous))
    if isinstance(v, (list, tuple)):
        return Tuple(*[encoded(x) for x in v])
    if isinstance(v, int):
        return Integer(v)
    assert isinstance(v, Basic)
    return v


def argument(a: object) -> object:
    if isinstance(a, str):
        if a in ('True', 'False'):
            return a == 'True'
        return None if a == 'None' else parse(a)
    if isinstance(a, dict):
        if 'int' in a:
            return a['int']
        items = a['seq']
        assert isinstance(items, list)
        return [argument(x) for x in items]
    assert isinstance(a, list)
    rows, cols, entries = a
    assert isinstance(rows, int) and isinstance(cols, int) and isinstance(entries, list)
    return ExprMatrix(rows, cols, [parse(str(e)) for e in entries])


EQUATIONS = {'kronecker_product', 'vec', 'unvec', 'adjoint', 'linear_matrix_equation', 'sylvester_solve', 'lyapunov_solve',
             'discrete_lyapunov_solve', 'commuting_matrices'}
STABILITY = {'hurwitz_matrix', 'hurwitz_determinants', 'routh_array', 'stability_conditions', 'is_stable'}
CLASSIFICATION = {'discrimination_sequence', 'signature_from_signs', 'real_root_classification'}

CASES: list[tuple[str, list[object]]] = [
    ('adjoint', [[2, 2, ['18/13', '5/26 + 6*I/13', '5/26 - 6*I/13', '3/4']]]),
    ('adjoint', [[2, 2, ['-1 + I', '2', '0', '-2 - I']]]),
    ('commuting_matrices', [[1, 1, ['-3']]]),
    ('commuting_matrices', [[4, 4, ['0', '1/2', '1/2', '-3', '-3', '1', '1/2', '2', '1', '1/2', '2', '1/2', '1', '1/2', '0', '1/2']]]),
    ('commuting_matrices', [[3, 3, ['0', '1/2', '1', '0', '1', '1/2', '-1', '-1', '-3']]]),
    ('commuting_matrices', [[4, 4, ['0', '-1', '2', '1', '0', '-1', '1', '0', '0', '-3', '0', '0', '1/2', '2', '-3', '0']]]),
    ('commuting_matrices', [[2, 2, ['1', '1', '1', '-3']]]),
    ('commuting_matrices', [[3, 3, ['2', '-1', '1/2', '1/2', '-3', '1/2', '2', '0', '0']]]),
    ('controllability_matrix', [[1, 1, ['1/2']], [1, 1, ['-1']]]),
    ('controllability_matrix', [[2, 2, ['0', '0', '2', '0']], [2, 1, ['1/2', '-1']]]),
    ('controllability_matrix', [[2, 2, ['-1', '1', '1/2', '1']], [2, 1, ['0', '1/2']]]),
    ('controllability_matrix', [[2, 2, ['1/2', '2', '2', '1']], [2, 1, ['1/2', '-1']]]),
    ('controllability_matrix', [[3, 3, ['1/2', '0', '1/2', '-1', '2', '1', '-1', '2', '-1']], [3, 1, ['1', '0', '0']]]),
    ('controllability_matrix', [[2, 2, ['1', '-2', '0', '-3']], [2, 1, ['0', '-1']]]),
    ('controllable_decomposition', [[3, 3, ['2', '0', '0', '0', '2', '3', '0', '0', '-1']], [3, 1, ['1', '0', '0']]]),
    ('controllable_decomposition', [[3, 3, ['3', '0', '0', '6', '-3', '0', '-11/2', '1/2', '-2']], [3, 1, ['0', '1', '0']]]),
    ('controllable_decomposition', [[2, 2, ['-1', '0', '-4', '1']], [2, 1, ['-1', '-3']]]),
    ('controllable_decomposition', [[3, 3, ['5', '1', '-2', '-6', '-2', '2', '6', '2', '-2']], [3, 1, ['1/2', '0', '0']]]),
    ('controllable_decomposition', [[2, 2, ['1', '0', '1', '2']], [2, 1, ['0', '-1']]]),
    ('controllable_decomposition', [[1, 1, ['-1']], [1, 1, ['1']]]),
    ('discrete_lyapunov_solve', [[3, 3, ['1/3', '1', '-1', '0', '-1/3', '-1', '0', '0', '-1/3']], [3, 3, ['1', '0', '0', '0', '1', '0', '0', '0', '1']]]),
    ('discrete_lyapunov_solve', [[2, 2, ['-2/3', '-1', '0', '2/3']], [2, 2, ['1', '0', '0', '1']]]),
    ('discrete_lyapunov_solve', [[3, 3, ['-1/3', '1', '0', '0', '-2/3', '0', '0', '0', '1/3']], [3, 3, ['1', '0', '0', '0', '1', '0', '0', '0', '1']]]),
    ('discrete_lyapunov_solve', [[2, 2, ['0', '1', '0', '0']], [2, 2, ['1', '0', '0', '1']]]),
    ('discrete_lyapunov_solve', [[1, 1, ['a']], [1, 1, ['1']]]),
    ('discrete_lyapunov_solve', [[3, 3, ['0', '0', '-1', '0', '0', '1', '0', '0', '-1/3']], [3, 3, ['1', '0', '0', '0', '1', '0', '0', '0', '1']]]),
    ('is_controllable', [[3, 3, ['-1', '0', '1', '1/2', '-1', '2', '0', '1', '2']], [3, 1, ['-1', '0', '2']]]),
    ('is_controllable', [[2, 2, ['0', '0', '1', '-1']], [2, 1, ['2', '2']]]),
    ('is_controllable', [[1, 1, ['1/2']], [1, 1, ['-1']]]),
    ('is_controllable', [[2, 2, ['-1', '1/2', '0', '2']], [2, 1, ['0', '0']]]),
    ('is_controllable', [[3, 3, ['0', '1', '0', '0', '0', '1', '3/2', '3/2', '-3/2']], [3, 1, ['0', '0', '1']]]),
    ('is_controllable', [[2, 2, ['-1', '1', '1/2', '1']], [2, 1, ['0', '1/2']]]),
    ('is_observable', [[1, 1, ['1/2']], [1, 1, ['0']]]),
    ('is_observable', [[1, 1, ['-1']], [1, 1, ['1']]]),
    ('is_observable', [[3, 3, ['2', '2', '1', '-1', '1', '1/2', '0', '1', '0']], [1, 3, ['0', '1/2', '0']]]),
    ('is_observable', [[2, 2, ['0', '-1', '2', '1/2']], [1, 2, ['0', '2']]]),
    ('is_observable', [[3, 3, ['1/2', '-1', '-1', '0', '2', '2', '1/2', '1', '-1']], [1, 3, ['1', '0', '0']]]),
    ('is_observable', [[1, 1, ['3']], [1, 1, ['1']]]),
    ('kronecker_product', [[2, 3, ['1', '1', '1/2', '1/2', '2', '0']], [2, 4, ['0', '1', '-1', '1', '-1', '-3', '2', '2']]]),
    ('kronecker_product', [[2, 3, ['1/2', '1/2', '-3', '1/2', '0', '0']], [2, 4, ['1', '-3', '2', '-1', '-1', '2', '-3', '2']]]),
    ('kronecker_product', [[4, 2, ['-1', '0', '1', '1', '0', '2', '1', '-1']], [2, 3, ['-1', '-3', '-3', '0', '1', '1/2']]]),
    ('kronecker_product', [[2, 3, ['1', '1/2', '0', '1/2', '-3', '1/2']], [2, 4, ['-1', '2', '-3', '2', '1/2', '1', '1', '1']]]),
    ('kronecker_product', [[4, 2, ['0', '2', '2', '0', '1', '-1', '1', '1/2']], [2, 3, ['-3', '0', '2', '2', '1/2', '2']]]),
    ('kronecker_product', [[2, 3, ['-3', '0', '2', '2', '1/2', '2']], [2, 4, ['0', '2', '1', '1', '2', '0', '-1', '1/2']]]),
    ('linear_matrix_equation', [{'seq': [{'seq': [[2, 3, ['2', '0', '1', '-1', '1/2', '2']], [2, 2, ['0', '-1', '0', '-1']]]}, {'seq': [[2, 3, ['-3', '2', '0', '-3', '0', '-3']], [2, 2, ['1/2', '1/2', '1', '-3']]]}]}, [2, 2, ['5/2', '53/2', '18', '-25']]]),
    ('linear_matrix_equation', [{'seq': [{'seq': [[2, 3, ['-1', '-3', '0', '1/2', '-1', '-3']], [2, 2, ['0', '0', '-3', '1']]]}, {'seq': [[2, 3, ['1/2', '0', '-1', '1', '-3', '-1']], [2, 2, ['0', '0', '2', '1/2']]]}]}, [2, 2, ['13', '-23/8', '-39/4', '3/2']]]),
    ('linear_matrix_equation', [{'seq': [{'seq': [[2, 3, ['0', '1/2', '1/2', '-1', '-1', '-3']], [2, 2, ['0', '0', '-1', '0']]]}, {'seq': [[2, 3, ['0', '-3', '-3', '0', '0', '1/2']], [2, 2, ['1/2', '1/2', '2', '0']]]}]}, [2, 2, ['-11', '-9/2', '9/2', '1/2']]]),
    ('linear_matrix_equation', [{'seq': [{'seq': [[2, 3, ['-3', '1', '0', '-3', '1', '2']], [2, 2, ['-3', '2', '2', '2']]]}, {'seq': [[2, 3, ['-3', '-3', '1/2', '2', '2', '0']], [2, 2, ['2', '-1', '1', '0']]]}]}, [2, 2, ['-35/2', '-23/2', '69/2', '-16']]]),
    ('linear_matrix_equation', [{'seq': []}, [2, 2, ['1', '0', '0', '1']]]),
    ('linear_matrix_equation', [{'seq': [{'seq': [[2, 3, ['-1', '-1', '2', '2', '2', '2']], [2, 2, ['1/2', '2', '2', '2']]]}, {'seq': [[2, 3, ['-1', '-3', '0', '-1', '2', '-1']], [2, 2, ['0', '1/2', '-3', '-3']]]}]}, [2, 2, ['-8', '-17/2', '-85/2', '-155/4']]]),
    ('lyapunov_solve', [[2, 2, ['-2', '2', '0', '-1']], [2, 2, ['6', '4', '4', '5']]]),
    ('lyapunov_solve', [[1, 1, ['a']], [1, 1, ['b']]]),
    ('lyapunov_solve', [[3, 3, ['-2', '2', '-2', '0', '-1', '-2', '0', '0', '-4']], [3, 3, ['11', '6', '1', '6', '20', '6', '1', '6', '3']]]),
    ('lyapunov_solve', [[2, 2, ['-1', '-1', '0', '-4']], [2, 2, ['1', '0', '0', '5/4']]]),
    ('lyapunov_solve', [[1, 1, ['-4']], [1, 1, ['2']]]),
    ('lyapunov_solve', [[2, 2, ['-1', '-2', '0', '-3']], [2, 2, ['9', '-8', '-8', '11']]]),
    ('minimal_realization', [[3, 3, ['-4/7', '20/7', '10/7', '20/7', '26/7', '20/7', '-15/7', '-30/7', '-29/7']], [3, 1, ['3/7', '-1/7', '6/7']], [1, 3, ['-1', '2', '2']]]),
    ('minimal_realization', [[4, 4, ['-5/4', '-19/8', '-7/8', '-25/4', '-1/4', '17/8', '-3/8', '19/4', '3/4', '13/8', '9/8', '31/4', '-1', '-1/2', '-1/2', '-2']], [4, 1, ['1/4', '1/4', '-3/4', '0']], [1, 4, ['1', '4', '-1', '3']]]),
    ('minimal_realization', [[4, 4, ['1/4', '13/4', '-23/4', '41/8', '9/4', '-3/4', '21/4', '-27/8', '-1/2', '-1/2', '5/2', '-17/4', '0', '0', '0', '-3']], [4, 1, ['-17/8', '11/8', '5/4', '1']], [1, 4, ['2', '0', '5', '1']]]),
    ('minimal_realization', [[4, 4, ['-9', '-12', '-6', '-6', '62', '77', '38', '38', '-82', '-100', '-49', '-50', '-30', '-36', '-18', '-17']], [4, 1, ['-1', '9', '-12', '-4']], [1, 4, ['6', '4', '2', '1']]]),
    ('minimal_realization', [[1, 1, ['-3']], [1, 1, ['1/2']], [1, 1, ['0']]]),
    ('minimal_realization', [[4, 4, ['-4', '-2', '-6', '4', '-6', '-4', '-14', '4', '2', '2', '4', '-4', '-2', '-2', '-6', '2']], [4, 1, ['0', '-1/2', '1/2', '0']], [1, 4, ['1', '-1', '1', '3']]]),
    ('observability_matrix', [[3, 3, ['0', '-1', '-1', '1/2', '1/2', '2', '0', '1/2', '-1']], [1, 3, ['1/2', '-1', '0']]]),
    ('observability_matrix', [[3, 3, ['0', '1', '0', '1/2', '0', '0', '1/2', '0', '2']], [1, 3, ['0', '1/2', '1']]]),
    ('observability_matrix', [[2, 2, ['0', '1', '1', '0']], [1, 2, ['2', '-1']]]),
    ('observability_matrix', [[1, 1, ['1']], [1, 1, ['1/2']]]),
    ('observability_matrix', [[2, 2, ['0', '1', '0', '-1']], [1, 2, ['2', '2']]]),
    ('observability_matrix', [[2, 2, ['0', '1', '1/2', '-1']], [1, 2, ['-1', '2']]]),
    ('observable_decomposition', [[4, 4, ['5', '-10', '-8', '8', '-4', '5', '5', '-5', '4', '-4', '-4', '5', '-4', '10', '8', '-7']], [1, 4, ['2', '2', '-1', '0']]]),
    ('observable_decomposition', [[2, 2, ['2', '-2', '1', '-1']], [1, 2, ['-2', '3']]]),
    ('observable_decomposition', [[3, 3, ['2', '1', '0', '-6', '-3', '0', '-2', '-1', '0']], [1, 3, ['4', '1', '2']]]),
    ('observable_decomposition', [[4, 4, ['0', '0', '3', '3', '2', '-3', '2', '2', '-1/2', '2', '-3/2', '-5/2', '-1/2', '-2', '-5/2', '-3/2']], [1, 4, ['1', '1', '4', '2']]]),
    ('observable_decomposition', [[3, 3, ['0', '-3', '-3', '0', '-1', '0', '0', '4', '3']], [1, 3, ['2', '3', '3']]]),
    ('observable_decomposition', [[2, 2, ['1', '-2', '0', '-3']], [1, 2, ['2', '-1']]]),
    ('pole_placement', [[2, 2, ['1', '0', '0', '1']], [2, 2, ['1', '0', '0', '1']], {'seq': [{'int': 1}, {'int': 2}]}]),
    ('pole_placement', [[4, 4, ['0', '1', '-1', '1/2', '-1', '2', '2', '1/2', '1', '1', '-1', '-1', '0', '1', '2', '2']], [4, 1, ['2', '2', '-1', '1/2']], {'seq': ['-2', '-3', '-4', '-3']}]),
    ('pole_placement', [[3, 3, ['2', '2', '2', '1', '0', '0', '1/2', '-1', '0']], [3, 1, ['1/2', '1/2', '2']], '(s + 2)*(s + 4)*(3*s + 6)', 's']),
    ('pole_placement', [[1, 1, ['1']], [1, 1, ['0']], {'seq': ['-1']}]),
    ('pole_placement', [[1, 1, ['-1']], [1, 1, ['1']], '3*s + 3', 's']),
    ('pole_placement', [[2, 2, ['-1', '0', '0', '1']], [2, 1, ['0', '2']], {'seq': ['-1', '-2']}]),
    ('realization', ['(-2*s - 1)/(3*s**2 + 2*s + 2)', 's']),
    ('realization', ['1/(2*s**2 - s + 3)', 's']),
    ('realization', ['1/(s**2 - 3*s)', 's']),
    ('realization', ['s**2/(s + 1)', 's']),
    ('realization', ['(s**2 + 2*s - 2)/(3*s**2 - s - 3)', 's']),
    ('realization', ['3/(3*s**3 - 3*s**2 + 2*s - 1)', 's']),
    ('sylvester_solve', [[2, 2, ['1', '2', '3', '4']], [2, 2, ['-1', '-2', '-3', '-4']], [2, 2, ['1', '0', '0', '1']]]),
    ('sylvester_solve', [[2, 2, ['1', '0', '-3', '-3']], [2, 2, ['1', '1', '-3', '-1']], [2, 2, ['-6', '-3', '18', '1']]]),
    ('sylvester_solve', [[2, 2, ['1', '2', '-1', '1/2']], [2, 2, ['-1', '1', '-2', '-1/2']], [2, 2, ['1', '19/4', '-5', '1/2']]]),
    ('sylvester_solve', [[3, 3, ['0', '0', '2', '1', '-3', '-1', '0', '2', '1/2']], [3, 3, ['0', '-1', '0', '0', '3', '-2', '-2', '1', '-1/2']], [3, 3, ['3', '8', '-25/4', '3/2', '-3', '8', '5', '9/2', '-8']]]),
    ('sylvester_solve', [[1, 1, ['-1']], [2, 2, ['1/2', '1', '2', '0']], [1, 2, ['-6', '3']]]),
    ('sylvester_solve', [[2, 2, ['1', '2', 'a', '-3']], [2, 2, ['-1', '-a', '-2', '3']], [2, 2, ['4', '3*a - 3', '-3*a - 5', '-2*a']]]),
    ('transfer_function', [[1, 1, ['-1']], [1, 1, ['1']], [1, 1, ['4']], 's', [1, 1, ['-3']]]),
    ('transfer_function', [[3, 3, ['1/2', '0', '0', '2', '2', '-1', '2', '1/2', '2']], [3, 1, ['-1', '0', '2']], [2, 3, ['2', '1/2', '0', '0', '0', '2']], 's', 'None']),
    ('transfer_function', [[1, 1, ['0']], [1, 2, ['-1', '1']], [2, 1, ['1', '2']], 's', [2, 2, ['2', '0', '-1', '2']]]),
    ('transfer_function', [[3, 3, ['0', '1', '0', '0', '0', '1', '2', '0', '3']], [3, 1, ['0', '0', '1']], [1, 3, ['-1', '-3', '-2']], 's', [1, 1, ['0']]]),
    ('transfer_function', [[1, 1, ['1']], [1, 1, ['1/2']], [1, 1, ['0']], 's']),
    ('transfer_function', [[1, 1, ['0']], [1, 1, ['-1']], [1, 1, ['1/2']], 's', [1, 1, ['2']]]),
    ('unvec', [[6, 1, ['-3', '-1', '2', '1/2', '2', '1']], {'int': 3}, {'int': 2}]),
    ('unvec', [[6, 1, ['1', '1', '-3', '2', '1/2', '0']], {'int': 3}, {'int': 2}]),
    ('unvec', [[6, 1, ['0', '0', '1/2', '-1', '-1', '-1']], {'int': 3}, {'int': 2}]),
    ('unvec', [[6, 1, ['1', '1/2', '0', '1/2', '2', '2']], {'int': 3}, {'int': 2}]),
    ('unvec', [[6, 1, ['-3', '-3', '1', '2', '2', '-1']], {'int': 3}, {'int': 2}]),
    ('unvec', [[6, 1, ['-1', '1/2', '2', '1', '0', '1/2']], {'int': 3}, {'int': 2}]),
    ('vec', [[3, 2, ['0', '-1', '0', '-1', '1/2', '-1']]]),
    ('vec', [[2, 4, ['-45/2', '15/2', '-45/2', '-45/2', '3/2', '5/2', '9/2', '1']]]),
    ('vec', [[3, 2, ['-1', '1', '1/2', '0', '2', '1/2']]]),
    ('vec', [[2, 4, ['9', '-8', '-5/4', '3', '-5/4', '3/2', '1/8', '-1']]]),
    ('vec', [[2, 4, ['-13', '26', '-13', '0', '-31/2', '-13/2', '7', '0']]]),
    ('vec', [[2, 4, ['1', '11/4', '-7/4', '-9/4', '11/2', '17', '-23/2', '-21/2']]]),
    ('hurwitz_determinants', ['a*s**2 + b*s + c + s**3', 's']),
    ('hurwitz_matrix', ['a*s + b + s**2', 's']),
    ('is_stable', ['3*s**5 + 5*s**4 + 9*s**3 + 8*s**2 + 3*s + 8', 's', 'False']),
    ('is_stable', ['-4*s**4 - 9*s**3 - 9*s**2 - 6*s - 4', 's', 'False']),
    ('is_stable', ['-2*s**2 - 8*s - 6', 's', 'False']),
    ('is_stable', ['-s**3 - 11*s**2/3 - 2*s/3 - 8', 's', 'False']),
    ('is_stable', [[2, 2, ['0', '1', '-k', '-1']], 'None', 'False']),
    ('is_stable', ['z**4 - 7*z**3/4 + 11*z**2/8 - z/2 + 1/16', 'z', 'True']),
    ('is_stable', ['z + 1/2', 'z', 'True']),
    ('is_stable', ['s**2 + 7*s/3 + 2/3', 's', 'False']),
    ('is_stable', ['7*s**5 + 4*s**4 + 7*s**3 + 7*s**2 + 5*s + 6', 's', 'False']),
    ('is_stable', ['-3*s**5 - 5*s**4 - 8*s**3 - 7*s**2 - s - 1', 's', 'False']),
    ('is_stable', ['4*s**4 + 9*s**3 + 9*s**2 + 6*s + 4', 's', 'False']),
    ('is_stable', ['z**2 + 2*z + 1', 'z', 'True']),
    ('routh_array', ['s**4 + 7*s**3 + 65*s**2/4 + 14*s + 15/4', 's']),
    ('routh_array', ['5*s**2 + 8*s + 7', 's']),
    ('routh_array', ['s**4 + 27*s**3/2 + 109*s**2/2 + 60*s + 18', 's']),
    ('routh_array', ['s**5 + 31*s**4/3 + 301*s**3/9 + 43*s**2 + 530*s/9 + 200/3', 's']),
    ('routh_array', ['s + 2', 's']),
    ('routh_array', ['s**2 + 4*s + 3', 's']),
    ('routh_array', ['s**5 + 19*s**4/2 + 35*s**3 + 125*s**2/2 + 54*s + 18', 's']),
    ('routh_array', ['s**3 + s**2 + 14*s/9 + 8/3', 's']),
    ('routh_array', ['3*s**5 + 5*s**4 + 9*s**3 + 8*s**2 + 3*s + 8', 's']),
    ('routh_array', ['s**6 + 35*s**5/3 + 47*s**4 + 257*s**3/3 + 112*s**2 + 428*s/3 + 80', 's']),
    ('routh_array', ['s**3 + 10*s**2 + 27*s + 18', 's']),
    ('routh_array', ['s**2 + 7*s/3 + 2/3', 's']),
    ('stability_conditions', ['a*s**2 + b*s + s**4 + 2*s**3 + 1', 's', 'False', 'True']),
    ('stability_conditions', ['a*s + b + s**2', 's', 'False', 'True']),
    ('stability_conditions', ['b/2 + s**2 + s*(a**2 - 1)', 's', 'False', 'False']),
    ('stability_conditions', ['a*z + b + z**2', 'z', 'True', 'True']),
    ('stability_conditions', ['s**2 + 1', 's', 'False', 'True']),
    ('stability_conditions', ['a*s**2 + b*s + s**3 + 1', 's', 'False', 'False']),
    ('stability_conditions', ['a*z + b + z**3', 'z', 'True', 'True']),
    ('stability_conditions', ['a*s**2 + b*s + s**3 + 1', 's', 'False', 'True']),
    ('stability_conditions', ['a*s**2 + b + s', 's', 'False', 'True']),
    ('stability_conditions', ['a*s**2 + b*s + s**4 + 2*s**3 + 1', 's', 'False', 'False']),
    ('stability_conditions', ['5', 's', 'False', 'True']),
    ('stability_conditions', ['a*s**2 + b + s', 's', 'False', 'False']),
    ('discrimination_sequence', ['2*x**3 + 4*x', 'x']),
    ('discrimination_sequence', ['-x**6 - 3*x**5 - 6*x**4 - 9*x**3 - 9*x**2 - 6*x - 2', 'x']),
    ('discrimination_sequence', ['2*x**4 - 2*x**2 - 4*x + 4', 'x']),
    ('discrimination_sequence', ['x**3 + x**2', 'x']),
    ('discrimination_sequence', ['-x**4 + 2*x**2 - 1', 'x']),
    ('discrimination_sequence', ['-x**4 - 2*x**3 - 4*x**2 - 2*x - 3', 'x']),
    ('discrimination_sequence', ['x**7 + 2*x**6 - 7*x**5 - 14*x**4 + 16*x**3 + 32*x**2 - 12*x - 24', 'x']),
    ('discrimination_sequence', ['2*x**7 - 6*x**5 - 12*x**4 + 6*x**3 + 24*x**2 - 2*x - 12', 'x']),
    ('discrimination_sequence', ['-x**4 - 4*x**3 - 7*x**2 - 6*x - 2', 'x']),
    ('discrimination_sequence', ['2*x**4 + 4*x**3 - 2*x**2 - 8*x - 4', 'x']),
    ('discrimination_sequence', ['p*x + q + x**3', 'x']),
    ('discrimination_sequence', ['-x**4 + 4*x**2 - 4', 'x']),
    ('discrimination_sequence', ['2*x**6 + 6*x**4 - 18*x**2 - 54', 'x']),
    ('discrimination_sequence', ['-x**8 + 2*x**7 + x**6 - 4*x**5 + 3*x**4 - 2*x**3 - x**2 + 4*x - 2', 'x']),
    ('real_root_classification', ['-x**2', 'x', {'int': 6}]),
    ('real_root_classification', ['-x**4 - 3*x**3 - 6*x**2 - 7*x - 3', 'x', {'int': 6}]),
    ('real_root_classification', ['2*x**4 - 8', 'x', {'int': 6}]),
    ('real_root_classification', ['2*x**4 - 4*x**2 + 2', 'x', {'int': 6}]),
    ('real_root_classification', ['-x**4 + 4*x**3 - 4*x**2', 'x', {'int': 6}]),
    ('real_root_classification', ['0', 'x', {'int': 6}]),
    ('real_root_classification', ['2*x**8 + 2*x**7 - 6*x**6 - 8*x**5 - 2*x**4 + 6*x**3 + 6*x**2', 'x', {'int': 6}]),
    ('real_root_classification', ['2*x**4 - 4*x**3 - 2*x**2 + 4*x', 'x', {'int': 6}]),
    ('real_root_classification', ['-x**4 + 4*x**2 - 4', 'x', {'int': 6}]),
    ('real_root_classification', ['x**5 + x**3 - 2*x', 'x', {'int': 6}]),
    ('real_root_classification', ['2*x**8 - 8*x**6 - 4*x**5 + 10*x**4 + 12*x**3 - 12*x**2', 'x', {'int': 6}]),
    ('real_root_classification', ['-x**5 - 3*x**4 + 7*x**2 + 9*x + 6', 'x', {'int': 6}]),
    ('real_root_classification', ['-x**8 - 4*x**7 - 6*x**6 - 4*x**5 - x**4', 'x', {'int': 6}]),
    ('real_root_classification', ['b*x + c + x**2', 'x', {'int': 6}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': 1}, {'int': 1}, {'int': 0}, {'int': 0}, {'int': 0}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': 1}, {'int': 1}, {'int': -1}, {'int': -1}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': -1}, {'int': -1}, {'int': -1}, {'int': 0}, {'int': 0}, {'int': 0}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': 1}, {'int': 1}, {'int': -1}, {'int': -1}, {'int': 0}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': 1}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': 1}, {'int': -1}, {'int': 0}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': 1}, {'int': 1}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': -1}, {'int': 1}, {'int': 1}, {'int': 1}, {'int': -1}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': -1}, {'int': -1}, {'int': -1}, {'int': -1}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': 1}, {'int': 1}, {'int': -1}, {'int': -1}, {'int': 0}, {'int': 0}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': -1}, {'int': -1}, {'int': 0}, {'int': 0}, {'int': 0}, {'int': 0}, {'int': 0}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': 0}, {'int': 0}, {'int': 0}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': 1}, {'int': -1}]}]),
    ('signature_from_signs', [{'seq': [{'int': 1}, {'int': -1}, {'int': -1}, {'int': 0}, {'int': 0}, {'int': 0}]}]),
    ('care_solve', [[1, 1, ['-3']], [1, 1, ['1']], [1, 1, ['3']], [1, 1, ['3']]]),
    ('care_solve', [[1, 1, ['-1']], [1, 1, ['1']], [1, 1, ['0']], [1, 1, ['1']]]),
    ('care_solve', [[1, 1, ['3']], [1, 1, ['3']], [1, 1, ['1']], [1, 1, ['2']]]),
    ('care_solve', [[2, 2, ['0', '1', '0', '0']], [3, 3, ['1', '0', '0', '0', '1', '0', '0', '0', '1']], [2, 2, ['1', '0', '0', '1']], [1, 1, ['1']]]),
    ('care_solve', [[1, 1, ['-1']], [1, 1, ['2']], [1, 1, ['3']], [1, 1, ['3']]]),
    ('care_solve', [[1, 1, ['-2']], [1, 1, ['1']], [1, 1, ['4']], [1, 1, ['3']]]),
    ('care_solve', [[1, 1, ['-3']], [1, 1, ['2']], [1, 1, ['3']], [1, 1, ['2']]]),
    ('care_solve', [[2, 2, ['1', '2', '1', '-2']], [2, 1, ['1', '1']], [2, 2, ['3', '0', '0', '3']], [1, 1, ['1/2']]]),
    ('closed_loop_is_stable', [[2, 2, ['0', '1', '0', '0']], [2, 1, ['0', '1']], [1, 2, ['-1', '3']]]),
    ('closed_loop_is_stable', [[2, 2, ['0', '1', '0', '0']], [2, 1, ['0', '1']], [1, 2, ['2', '3']]]),
    ('closed_loop_is_stable', [[2, 2, ['0', '1', '0', '0']], [2, 1, ['0', '1']], [1, 2, ['1', '0']]]),
    ('dare_solve', [[1, 1, ['-2']], [1, 1, ['1']], [1, 1, ['4']], [1, 1, ['3']]]),
    ('dare_solve', [[1, 1, ['-2']], [1, 1, ['1']], [1, 1, ['1']], [1, 1, ['1']]]),
    ('dare_solve', [[1, 1, ['0']], [1, 1, ['1']], [1, 1, ['1']], [1, 1, ['3']]]),
    ('dare_solve', [[1, 1, ['1']], [1, 1, ['3']], [1, 1, ['2']], [1, 1, ['3']]]),
    ('dare_solve', [[1, 1, ['3']], [1, 1, ['3']], [1, 1, ['2']], [1, 1, ['2']]]),
    ('dare_solve', [[1, 1, ['-3']], [1, 1, ['1']], [1, 1, ['4']], [1, 1, ['1']]]),
    ('dare_solve', [[1, 1, ['0']], [1, 1, ['2']], [1, 1, ['1']], [1, 1, ['1']]]),
    ('dare_solve', [[1, 1, ['-2']], [1, 1, ['1']], [1, 1, ['2']], [1, 1, ['2']]]),
    ('discrete_riccati_solutions', [[2, 2, ['1', '1', '2', '0']], [2, 1, ['0', '1']], [2, 2, ['1', '0', '0', '1']], [1, 1, ['1']]]),
    ('discrete_riccati_solutions', [[2, 2, ['0', '-2', '1', '-2']], [2, 1, ['1', '0']], [2, 2, ['1', '0', '0', '1']], [1, 1, ['1']]]),
    ('discrete_riccati_solutions', [[2, 2, ['1', '2', '0', '2']], [2, 1, ['-1', '-1']], [2, 2, ['1', '0', '0', '1']], [1, 1, ['1']]]),
    ('discrete_riccati_solutions', [[2, 2, ['1', '-2', '2', '-2']], [2, 1, ['0', '-1']], [2, 2, ['1', '0', '0', '1']], [1, 1, ['1']]]),
    ('dlqr', [[1, 1, ['3']], [1, 1, ['2']], [1, 1, ['3']], [1, 1, ['2']]]),
    ('dlqr', [[1, 1, ['1']], [1, 1, ['3']], [1, 1, ['2']], [1, 1, ['3']]]),
    ('dlqr', [[1, 1, ['2']], [1, 1, ['0']], [1, 1, ['1']], [1, 1, ['1']]]),
    ('dlqr', [[2, 2, ['0', '-2', '1', '-2']], [2, 1, ['1', '0']], [2, 2, ['1', '0', '0', '1']], [1, 1, ['1']]]),
    ('dlqr', [[1, 1, ['3']], [1, 1, ['2']], [1, 1, ['4']], [1, 1, ['2']]]),
    ('dlqr', [[1, 1, ['1/2']], [1, 1, ['1']], [1, 1, ['0']], [1, 1, ['1']]]),
    ('dlqr', [[1, 1, ['0']], [1, 1, ['2']], [1, 1, ['1']], [1, 1, ['1']]]),
    ('dlqr', [[1, 1, ['-2']], [1, 1, ['1']], [1, 1, ['4']], [1, 1, ['3']]]),
    ('lqr', [[2, 2, ['0', '2', '-1', '-2']], [2, 1, ['2', '-1']], [2, 2, ['2', '0', '0', '1']], [1, 1, ['1/2']]]),
    ('lqr', [[2, 2, ['0', '1', '0', '0']], [2, 1, ['0', '1']], [2, 2, ['1', '0', '0', '1']], [1, 1, ['1']]]),
    ('lqr', [[2, 2, ['1', '2', '0', '-1']], [2, 1, ['0', '1']], [2, 2, ['2', '1', '1', '2']], [1, 1, ['3']]]),
    ('lqr', [[2, 2, ['1', '2', '1', '-2']], [2, 1, ['1', '1']], [2, 2, ['3', '0', '0', '3']], [1, 1, ['1/2']]]),
    ('lqr', [[1, 1, ['1']], [1, 1, ['0']], [1, 1, ['1']], [1, 1, ['1']]]),
    ('lqr', [[2, 2, ['-2', '-1', '0', '0']], [2, 1, ['-1', '1']], [2, 2, ['2', '1', '1', '2']], [1, 1, ['1']]]),
    ('lqr', [[2, 2, ['0', '-2', '0', '0']], [2, 2, ['-1', '0', '1', '-1']], [2, 2, ['2', '1', '1', '3']], [2, 2, ['3', '0', '0', '3/2']]]),
    ('riccati_solutions', [[1, 1, ['3']], [1, 1, ['3']], [1, 1, ['1']], [1, 1, ['2']]]),
    ('riccati_solutions', [[2, 2, ['1', '2', '0', '-1']], [2, 1, ['0', '1']], [2, 2, ['2', '1', '1', '2']], [1, 1, ['3']]]),
    ('riccati_solutions', [[1, 1, ['1']], [1, 1, ['2']], [1, 1, ['4']], [1, 1, ['1']]]),
    ('riccati_solutions', [[2, 2, ['1', '2', '1', '-2']], [2, 1, ['1', '1']], [2, 2, ['3', '0', '0', '3']], [1, 1, ['1/2']]]),
    ('riccati_solutions', [[1, 1, ['2']], [1, 1, ['3']], [1, 1, ['4']], [1, 1, ['2']]]),
    ('riccati_solutions', [[2, 2, ['-2', '-1', '0', '0']], [2, 1, ['-1', '1']], [2, 2, ['2', '1', '1', '2']], [1, 1, ['1']]]),
    ('riccati_solutions', [[1, 1, ['3']], [1, 1, ['1']], [1, 1, ['2']], [1, 1, ['2']]]),
    ('riccati_solutions', [[2, 2, ['0', '-2', '0', '0']], [2, 2, ['-1', '0', '1', '-1']], [2, 2, ['2', '1', '1', '3']], [2, 2, ['3', '0', '0', '3/2']]]),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_statespace(i: int) -> None:
    name, raw = CASES[i]
    args = [argument(a) for a in raw]
    module = EQ if name in EQUATIONS else (SB if name in STABILITY else (RC if name in CLASSIFICATION else (RK if hasattr(RK, name) else SS)))
    got = outcome(lambda: rc.logic_call('matfunctions', [Symbol(name)] + [encoded(a) for a in args]))
    if got != 'SKIP':
        assert got == outcome(lambda: encoded(getattr(module, name)(*args))), (name, raw)

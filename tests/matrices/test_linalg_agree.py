"""The Rust port of ``matrices/linalg.py`` against the Python one: the
determinants, characteristic polynomials, adjugates, inverses, ranks,
echelon forms, kernels, linear solutions and pseudoinverses of the
matrices of the matrices tests (entries read with the core's parser),
over the integers, the polynomials in the kernels and the multiquadratic
fields."""
from __future__ import annotations

from typing import Callable, Union

import pytest

from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
import sympy_extras.matrices.linalg as LA
from sympy_extras.matrices.dense import ExprMatrix

from ..conftest import load

rc = load("sympy_extras_rs.core")

Argument = Union[str, list[object]]


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else str(r)


def encoded(v: object) -> Basic:
    if v is None:
        return Symbol('NONE')
    if isinstance(v, ExprMatrix):
        return Tuple(Integer(v.rows), Integer(v.cols), Tuple(*v.entries))
    if isinstance(v, LA.LinearSolution):
        return Tuple(encoded(v.particular), Tuple(*[encoded(x) for x in v.nullspace]))
    if isinstance(v, (list, tuple)):
        return Tuple(*[encoded(x) for x in v])
    if isinstance(v, int):
        return Integer(v)
    assert isinstance(v, Basic)
    return v


def argument(a: Argument) -> object:
    if isinstance(a, list):
        rows, cols, entries = a
        assert isinstance(rows, int) and isinstance(cols, int) and isinstance(entries, list)
        return ExprMatrix(rows, cols, [parse(str(e)) for e in entries])
    return parse(a)


CASES: list[tuple[str, list[Argument]]] = [
    ('adjugate', [[1, 1, ['2 - I']]]),
    ('adjugate', [[2, 2, ['2 - I', '-I', '0', '1 + I']]]),
    ('adjugate', [[1, 1, ['1/2 + I/3']]]),
    ('adjugate', [[1, 1, ['I']]]),
    ('adjugate', [[2, 2, ['x', 'I*x', 'I*x', '-x']]]),
    ('adjugate', [[3, 3, ['-I', '0', '0', '1 + I', '1 + I', '1', '-I', '2 - I', '1/2 + I/3']]]),
    ('adjugate', [[3, 3, ['y', 'x', '-2', '1', '-2', 'x', 'x*y - 1', '-2', '0']]]),
    ('adjugate', [[3, 3, ['y', 'x**2', '0', 'x*y - 1', '1/3', '0', '-2', '1', 'y']]]),
    ('adjugate', [[3, 3, ['x**2', 'y', 'x**2', 'x', 'x*y - 1', '1', '-x + 2*y', 'y', 'x']]]),
    ('adjugate', [[2, 2, ['1/3', 'x + y', 'x', '0']]]),
    ('adjugate', [[1, 1, ['1/3']]]),
    ('adjugate', [[4, 4, ['1', '-2', 'x + y', 'y', '-2', '0', 'x + y', 'x', '-2', 'x*y - 1', '-x + 2*y', '-2', 'x**2', 'x*y - 1', 'x*y - 1', 'y/(x + 1)']]]),
    ('adjugate', [[1, 1, ['-2']]]),
    ('adjugate', [[3, 3, ['x*y - 1', 'x', 'x', 'x*y - 1', 'x + y', '-2', '1', 'x + y', 'x**2']]]),
    ('charpoly', [[2, 2, ['x', 'I*x', 'I*x', '-x']], 't']),
    ('charpoly', [[2, 2, ['0', '1 + I', '1/2 + I/3', '1']], 't']),
    ('charpoly', [[2, 2, ['2 - I', '-I', '0', '1 + I']], 't']),
    ('charpoly', [[1, 1, ['1/2 + I/3']], 't']),
    ('charpoly', [[3, 3, ['-I', '0', '0', '1 + I', '1 + I', '1', '-I', '2 - I', '1/2 + I/3']], 't']),
    ('charpoly', [[1, 1, ['2 - I']], 't']),
    ('charpoly', [[3, 3, ['y', 'x**2', '0', 'x*y - 1', '1/3', '0', '-2', '1', 'y']], 't']),
    ('charpoly', [[4, 4, ['-2', '0', 'x', '0', 'y', 'x + y', '0', 'x**2', 'x + y', 'y', 'x + y', '-x + 2*y', '0', 'x*y - 1', '0', '-2']], 't']),
    ('charpoly', [[2, 2, ['(-4*b - 6)/b', '6/b', '(-b**2 - 5*b - 6)/b', '(b + 6)/b']], 's']),
    ('charpoly', [[3, 3, ['x + y', '-2', '-2', 'x', '1', 'y', 'x', '-x + 2*y', '-x + 2*y']], 't']),
    ('charpoly', [[1, 1, ['-x + 2*y']], 't']),
    ('charpoly', [[2, 2, ['1/3', 'x + y', 'x', '0']], 't']),
    ('charpoly', [[1, 1, ['-1']], 's']),
    ('charpoly', [[4, 4, ['-2', '-2', '0', '0', '1/3', '(x - y)/(y + 2)', '1/3', '-2', '-2', 'x', 'y', '1/3', 'x**2', '1/x', '1/x', '1/3']], 't']),
    ('charpoly_coefficients', [[2, 2, ['-1', '1 + I', '2 - I', 'I']]]),
    ('charpoly_coefficients', [[3, 3, ['0', '1', '1', '0', '-1', '-I', 'I', '2 - I', '0']]]),
    ('charpoly_coefficients', [[3, 3, ['-1', 'I', '-I', '1 + I', '2 - I', '-I', '0', '-1', '0']]]),
    ('charpoly_coefficients', [[3, 3, ['I', 'I', '2 - I', '-1', '1', '0', '1 + I', '1', '-1']]]),
    ('charpoly_coefficients', [[2, 2, ['I', '1', '0', 'I']]]),
    ('charpoly_coefficients', [[3, 3, ['0', 'I', 'I', 'I', 'I', '-I', '1 + I', '1 + I', '2 - I']]]),
    ('charpoly_coefficients', [[2, 2, ['2', '0', '-2', '1']]]),
    ('charpoly_coefficients', [[3, 3, ['1', '0', '0', '-1', '-1', '0', '-2', '-2', '1']]]),
    ('charpoly_coefficients', [[1, 1, ['5']]]),
    ('charpoly_coefficients', [[4, 4, ['0', '-2', '0', '0', '-1', '2', '-1', '2', '-1', '0', '1', '1', '-1', '0', '1', '0']]]),
    ('charpoly_coefficients', [[2, 2, ['2', '0', '0', '0']]]),
    ('charpoly_coefficients', [[3, 3, ['12', '1', '2', '1', '3', '-1', '2', '-1', '6']]]),
    ('charpoly_coefficients', [[3, 3, ['4', '-4', '-2', '-4', '4', '2', '-2', '2', '1']]]),
    ('charpoly_coefficients', [[4, 4, ['0', '2', '1', '0', '-2', '-2', '0', '2', '-2', '0', '2', '0', '0', '2', '-2', '-1']]]),
    ('det', [[2, 2, ['0', '1 + I', '1/2 + I/3', '1']]]),
    ('det', [[2, 2, ['sqrt(2)', 'I', 'I', 'sqrt(2)']]]),
    ('det', [[3, 3, ['-I', '0', '0', '1 + I', '1 + I', '1', '-I', '2 - I', '1/2 + I/3']]]),
    ('det', [[2, 2, ['x', 'I*x', 'I*x', '-x']]]),
    ('det', [[2, 2, ['sqrt(2)*x', '1/(x + sqrt(3))', 'sqrt(2)', '0']]]),
    ('det', [[2, 2, ['I', '1', '1', 'I']]]),
    ('det', [[1, 1, ['-2']]]),
    ('det', [[2, 2, ['sin(t)**2 + cos(t)**2', '1', '1', '1']]]),
    ('det', [[3, 3, ['1', 'x', 'y', 'x', 'x*y - 1', 'y/(x + 1)', 'y', '-2', 'x*y - 1']]]),
    ('det', [[3, 3, ['cos(t)', '-sin(t)', '0', 'sin(t)', 'cos(t)', '0', '0', '0', '1']]]),
    ('det', [[1, 1, ['2 - 2*_lambda']]]),
    ('det', [[2, 2, ['-_lambda*m + 2*k', '-k', '-k', '-_lambda*m + k']]]),
    ('det', [[3, 3, ['a + 1', '-_lambda + a', '_lambda + 1/2', '2 - _lambda/2', '_lambda + 2', '-2*_lambda + a', 'a', '-_lambda/2 + a', '_lambda + a + 1']]]),
    ('det', [[4, 4, ['-1', '0', 'a', '3', '1', '3', '2', '0', '2', 'a', '1', '-1', 'b', 'a', '-1', 'a']]]),
    ('inverse', [[2, 2, ['sqrt(2)*x', '1/(x + sqrt(3))', 'sqrt(2)', '0']]]),
    ('inverse', [[1, 1, ['sqrt(2)/2']]]),
    ('inverse', [[2, 2, ['sqrt(2)/2', '1', 'sqrt(2)*x', 'x']]]),
    ('inverse', [[2, 2, ['sqrt(6)', 'sqrt(2)*x', 'sqrt(6)', 'sqrt(2)/2']]]),
    ('inverse', [[2, 2, ['sqrt(2)/2', 'sqrt(2)/2', 'sqrt(2)/2', '-sqrt(2)/2']]]),
    ('inverse', [[2, 2, ['sqrt(3)*I', 'sqrt(2)/2', 'sqrt(3)', 'sqrt(2)*x']]]),
    ('inverse', [[3, 3, ['4', '1', '2', '-2', '-1', '0', '1', '0', '0']]]),
    ('inverse', [[3, 3, ['1', '1', '2', '-1', '0', '0', '-1', '1', '0']]]),
    ('inverse', [[3, 3, ['4', '2', '-1', '2', '-2', '-2', '1', '0', '0']]]),
    ('inverse', [[4, 4, ['0', '1', '2', '2', '2', '1', '1/2', '1', '3', '-1', '2', '2', '2', '2', '2', '0']]]),
    ('inverse', [[2, 2, ['1', '1', '-1', '0']]]),
    ('inverse', [[2, 2, ['s', '-1', '-1', 's - 1/3']]]),
    ('inverse', [[2, 2, ['b', '3', '0', '3']]]),
    ('inverse', [[3, 3, ['x*y - 1', 'x', 'x', 'x*y - 1', 'x + y', '-2', '1', 'x + y', 'x**2']]]),
    ('linear_solve', [[3, 4, ['1 + I', 'I', '-I', '1/2 + I/3', 'I*(1 + I)', '-1', '1', 'I*(1/2 + I/3)', '1 + I', '-I', '1', '0']], [3, 1, ['1/2 + I/3', '1 + I', '2 - I']]]),
    ('linear_solve', [[2, 1, ['1/2 + I/3', '1']], [2, 1, ['1', '1']]]),
    ('linear_solve', [[3, 4, ['0', '1/2 + I/3', 'I', '1', '1 + I', 'I', 'I', '1/2 + I/3', '0', '1', '-I', '0']], [3, 1, ['1 + I', '0', '1 + I']]]),
    ('linear_solve', [[2, 1, ['2 - I', 'I*(2 - I)']], [2, 1, ['-I', '1']]]),
    ('linear_solve', [[1, 4, ['1', '-I', '1 + I', '-I']], [1, 1, ['1 + I']]]),
    ('linear_solve', [[2, 2, ['2 - I', '-I', '0', '1 + I']], [2, 1, ['1/2 + I/3', '2 - I']]]),
    ('linear_solve', [[1, 1, ['1']], [1, 1, ['1']]]),
    ('linear_solve', [[6, 6, ['1', '2', '-1', '0', '2', '0', '-3', 'a + 1', '0', '-1', '0', '2', '1', '0', '1', '2', '-3', '0', '0', '1', '-3', 'a + 1', '0', '-3', '0', '0', '-3', '0', '2', '2', '0', '0', '0', '-3', '-3', 'a + 2']], [6, 1, ['4', 'a/2 - 1/2', '-5', '-3*a - 5/2', '5', 'a + 11']]]),
    ('linear_solve', [[3, 3, ['9/10', '-1/5', '-3/10', '-2/9', '5/9', '-2/9', '-3/8', '-1/8', '3/4']], [3, 1, ['2/5', '1/9', '1/4']]]),
    ('linear_solve', [[4, 4, ['2/3', '0', '-1/3', '-1/3', '0', '1', '0', '0', '0', '0', '4/5', '-1/5', '0', '0', '0', '1']], [4, 1, ['0', '1', '3/5', '1']]]),
    ('linear_solve', [[4, 4, ['7/8', '-1/8', '-3/8', '0', '-2/5', '4/5', '-1/5', '-1/5', '-1/10', '0', '9/10', '-2/5', '-3/14', '-2/7', '-1/14', '5/7']], [4, 1, ['3/8', '0', '2/5', '1/7']]]),
    ('linear_solve', [[2, 2, ['sin(t)**2 + cos(t)**2', '1', '1', '1']], [2, 1, ['1', '2']]]),
    ('linear_solve', [[4, 4, ['0', '2', '-2', '0', '-1', '-1/2', '0', '-2', '1', '0', '1/2', '2', '0', '1', '-1', '0']], [4, 1, ['1', '-5', '19/4', '1/2']]]),
    ('linear_solve', [[9, 9, ['0', '1/2', '2', '-2', '0', '0', '0', '0', '0', '2', '3/2', '1', '0', '-2', '0', '0', '0', '0', '0', '-1', '2', '0', '0', '-2', '0', '0', '0', '-1/2', '0', '0', '-3/2', '1/2', '2', '1', '0', '0', '0', '-1/2', '0', '2', '0', '1', '0', '1', '0', '0', '0', '-1/2', '0', '-1', '1/2', '0', '0', '1', '-2', '0', '0', '-1', '0', '0', '-2', '1/2', '2', '0', '-2', '0', '0', '-1', '0', '2', '-1/2', '1', '0', '0', '-2', '0', '0', '-1', '0', '-1', '0']], [9, 1, ['0', '0', '0', '0', '0', '0', '0', '0', '0']]]),
    ('normal', ['I']),
    ('normal', ['sqrt(2)']),
    ('normal', ['-2 - (-I)']),
    ('normal', ['(-I)/2']),
    ('normal', ['-1 - I']),
    ('normal', ['5/26 + 6*(-I)/13']),
    ('normal', ['-12*(-2*a**2/3 + a*b/3 - 29*a/12 + 3*b/2 - 1/2)/(8*a**2 - 4*a*b + 29*a - 18*b + 6) - 1']),
    ('normal', ['74']),
    ('normal', ['193']),
    ('normal', ['130']),
    ('normal', ['5*a/(2*a - b) - 1']),
    ('normal', ['(-3*a - b)/(2*a - b) + (3*a + b)/(2*a - b)']),
    ('normal', ['318/973']),
    ('normal', ['(-a**2 - 2*a*b - 2*a - 4*b - 1)/(2*a*b - 3*a + 2*b - 3)']),
    ('nullspace', [[1, 4, ['1', '-I', '1 + I', '-I']]]),
    ('nullspace', [[3, 1, ['sqrt(3)*I', 'sqrt(6)*I + 3*I', 'x']]]),
    ('nullspace', [[2, 3, ['I', '1', '1', '-1', 'I', 'I']]]),
    ('nullspace', [[2, 2, ['x', 'I*x', 'I*x', '-x']]]),
    ('nullspace', [[3, 2, ['x', 'sqrt(2)', '1', '1/(x + sqrt(3))', '1/(x + sqrt(3))', '1']]]),
    ('nullspace', [[2, 1, ['x', '-1 + 2*sqrt(2)']]]),
    ('nullspace', [[2, 2, ['-2/3', '3/7', '2/3', '-3/7']]]),
    ('nullspace', [[5, 5, ['-9/10', '3/8', '0', '2/5', '1/7', '2/5', '-7/8', '2/5', '1/10', '3/14', '1/10', '1/8', '-4/5', '0', '2/7', '1/10', '3/8', '1/5', '-9/10', '1/14', '3/10', '0', '1/5', '2/5', '-5/7']]]),
    ('nullspace', [[4, 4, ['-1', '1/2', '0', '0', '1', '-1', '1/2', '0', '0', '1/2', '-1', '1', '0', '0', '1/2', '-1']]]),
    ('nullspace', [[2, 2, ['sin(t)**2 + cos(t)**2', '1', '1', '1']]]),
    ('nullspace', [[1, 1, ['0']]]),
    ('nullspace', [[2, 2, ['-a', 'b', 'a', '-b']]]),
    ('nullspace', [[4, 4, ['-8/11', '2/5', '1/9', '1/4', '3/11', '-9/10', '2/9', '3/8', '2/11', '1/5', '-5/9', '1/8', '3/11', '3/10', '2/9', '-3/4']]]),
    ('nullspace', [[5, 5, ['-1', '0', '0', '0', '2/7', '0', '-1', '0', '1/2', '1/7', '1', '0', '-1', '0', '0', '0', '1', '1', '-7/8', '0', '0', '0', '0', '3/8', '-3/7']]]),
    ('pseudoinverse', [[2, 3, ['1', '-1', '-1', '2', '-2', '-2']]]),
    ('pseudoinverse', [[2, 2, ['2', '1', '1', '1']]]),
    ('pseudoinverse', [[2, 4, ['-2', '-1', '-2', '-1', '-1', '0', '-2', '-2']]]),
    ('pseudoinverse', [[4, 4, ['-2', '-2', '-1', '2', '-1', '1', '1', '-1', '1', '-2', '-2', '2', '1', '-1', '1', '2']]]),
    ('pseudoinverse', [[2, 3, ['-1', '0', '2', '0', '2', '-2']]]),
    ('pseudoinverse', [[2, 1, ['-1', '-2']]]),
    ('pseudoinverse', [[2, 3, ['1', '2', '-2', '0', '0', '0']]]),
    ('pseudoinverse', [[3, 1, ['1', '-1', '-2']]]),
    ('rank', [[3, 2, ['1 + sqrt(2)', '1', 'sqrt(3)', '-2', '1 + sqrt(2)', '0']]]),
    ('rank', [[1, 1, ['1/2 + I/3']]]),
    ('rank', [[2, 1, ['2 - I', 'I*(2 - I)']]]),
    ('rank', [[3, 2, ['1/2 + I/3', '1', '1/2 + I/3', '1 + I', '1/(x - I)', '1']]]),
    ('rank', [[2, 2, ['sqrt(2)*x', '1/(x + sqrt(3))', 'sqrt(2)', '0']]]),
    ('rank', [[1, 2, ['1 + sqrt(2)', 'sqrt(3)']]]),
    ('rank', [[3, 2, ['2', '1', '0', 'a + 1', '0', 'b']]]),
    ('rank', [[4, 2, ['1', '-2', '1/2', '1', '0', '1/2', '-2', '0']]]),
    ('rank', [[3, 3, ['0', '-b', '-a*b - b**2 + 1', '-1', '-a', '-a**2 - 2*b + 2', '0', '1', 'a + b']]]),
    ('rank', [[2, 2, ['0', '0', '0', '2']]]),
    ('rank', [[2, 2, ['cos(t)', '-sin(t)', 'sin(t)', 'cos(t)']]]),
    ('rank', [[1, 2, ['x*y', '1/x']]]),
    ('rank', [[3, 1, ['1', '1/2', '1/2']]]),
    ('rank', [[4, 4, ['0', '0', '0', '1', '0', '0', '1', '-2/3', '0', '1', '-2/3', '4/9', '1', '-2/3', '4/9', '10/27']]]),
    ('rref', [[2, 1, ['1/2 + I/3', '1']]]),
    ('rref', [[2, 2, ['x', 'sqrt(2)*x', '1', '-1 + 2*sqrt(2)']]]),
    ('rref', [[2, 2, ['x', 'I*x', 'I*x', '-x']]]),
    ('rref', [[1, 2, ['1/2 + I/3', '1']]]),
    ('rref', [[2, 2, ['sqrt(3)', '0', '-1 + 2*sqrt(2)', '-2']]]),
    ('rref', [[2, 1, ['x', 'sqrt(2)*x + sqrt(3)*x']]]),
    ('rref', [[3, 4, ['1/2', '2', '3', '1/2', '0', '0', '-1', '2', '-1', '2', '0', '2']]]),
    ('rref', [[4, 8, ['1', '1/3', '19/3', '1/3', '1', '0', '0', '0', '-2', '-6', '-18', '-54', '0', '1', '0', '0', '3', '3', '27', '27', '0', '0', '1', '0', '3', '17/3', '89/3', '161/3', '0', '0', '0', '1']]]),
    ('rref', [[2, 4, ['2', '-3', '1', '0', '-3', '5', '0', '1']]]),
    ('rref', [[3, 2, ['0', '0', '3', '1/2', '3', '1']]]),
    ('rref', [[2, 4, ['2', '-2', '1', '0', '-2', '4', '0', '1']]]),
    ('rref', [[0, 1, []]]),
    ('rref', [[3, 6, ['0', '0', '0', '1', '0', '0', '1', '-1', '1', '0', '1', '0', '-1', '1', '-1', '0', '0', '1']]]),
    ('rref', [[3, 6, ['0', '0', '0', '1', '0', '0', '1', '-3', '9', '0', '1', '0', '0', '1/2', '-5/2', '0', '0', '1']]]),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_linalg(i: int) -> None:
    name, raw = CASES[i]
    args = [argument(a) for a in raw]
    got = outcome(lambda: rc.logic_call('linalg', [Symbol(name)] + [encoded(a) for a in args]))
    if got != 'SKIP':
        assert got == outcome(lambda: encoded(getattr(LA, name)(*args))), (name, raw)

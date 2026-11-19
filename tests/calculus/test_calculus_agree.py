"""The Rust ports of ``calculus/interpolation.py`` and
``calculus/optimization.py`` against the Python ones: divided
differences, Newton and Hermite interpolation, cubic splines, least
squares; value ranges, minima and maxima, parametric optima, on the
cases of their tests (read with the core's parser)."""
from __future__ import annotations

import ast
from typing import Callable

import pytest

from sympy_extras.core import Eq, Ge, Gt, Integer, Le, Lt, Ne, Symbol, Tuple
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import And, Boolean, Not, Or, false, true
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
import sympy_extras.calculus.interpolation as IP
import sympy_extras.calculus.optimization as OP

from ..conftest import load

rc = load("sympy_extras_rs.core")

BOUNDARIES = ('natural', 'clamped', 'not-a-knot', 'periodic')


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
    if isinstance(v, str):
        return Symbol(v)
    if isinstance(v, OP.Optimum):
        return Tuple(v.value, Symbol(str(v.attained)), encoded(v.points))
    if isinstance(v, dict):
        return Tuple(*[Tuple(k, encoded(x)) for k, x in v.items()])
    if isinstance(v, (list, tuple)):
        return Tuple(*[encoded(x) for x in v])
    if isinstance(v, int):
        return Integer(v)
    assert isinstance(v, Basic)
    return v


RELATIONS = {ast.Lt: Lt, ast.LtE: Le, ast.Gt: Gt, ast.GtE: Ge, ast.Eq: Eq, ast.NotEq: Ne}


def term(node: ast.expr) -> Expr:
    e = parse(ast.unparse(node))
    assert isinstance(e, Expr)
    return e


def formula(node: ast.expr) -> Basic:
    """A boolean formula of relations, ``&``, ``|``, ``~``, ``Eq`` and
    ``Ne``, the arithmetic read with the core's parser."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        return And(boolean(node.left), boolean(node.right))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return Or(boolean(node.left), boolean(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return Not(boolean(node.operand))
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return RELATIONS[type(node.ops[0])](term(node.left), term(node.comparators[0]))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Eq':
        return Eq(term(node.args[0]), term(node.args[1]))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Ne':
        return Ne(term(node.args[0]), term(node.args[1]))
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return true if node.value else false
    return term(node)


def boolean(node: ast.expr) -> Boolean:
    f = formula(node)
    assert isinstance(f, Boolean)
    return f


def argument(a: object) -> object:
    if isinstance(a, str):
        if a in ('True', 'False'):
            return a == 'True'
        if a == 'None':
            return None
        return a if a in BOUNDARIES else formula(ast.parse(a, mode='eval').body)
    assert isinstance(a, dict)
    if 'int' in a:
        return a['int']
    items = a['seq']
    assert isinstance(items, list)
    return [argument(x) for x in items]


CASES: list[tuple[str, list[object]]] = [
    ('cubic_spline', [{'seq': [{'seq': ['-5', '-1/3']}, {'seq': ['1', '8/3']}, {'seq': ['4', '-3']}, {'seq': ['5', '-1/3']}]}, 'x', 'periodic', 'None']),
    ('cubic_spline', [{'seq': [{'seq': ['-6', '-1']}, {'seq': ['-4', '5/2']}, {'seq': ['3', '-7']}, {'seq': ['4', '-1']}, {'seq': ['5', '-3/2']}]}, 'x', 'clamped', {'seq': ['2', '0']}]),
    ('cubic_spline', [{'seq': [{'seq': ['-6', '-1']}, {'seq': ['-4', '5/2']}, {'seq': ['3', '-7']}, {'seq': ['4', '-1']}, {'seq': ['5', '-3/2']}]}, 'x', 'natural', 'None']),
    ('cubic_spline', [{'seq': [{'seq': ['-3', '0']}, {'seq': ['-1', '8/3']}, {'seq': ['1', '8']}, {'seq': ['2', '2']}, {'seq': ['5', '-6']}]}, 'x', 'not-a-knot', 'None']),
    ('cubic_spline', [{'seq': [{'seq': ['-4', '-1/2']}, {'seq': ['-3', '-1/3']}, {'seq': ['4', '-4']}, {'seq': ['5', '-5']}]}, 'x', 'natural', 'None']),
    ('cubic_spline', [{'seq': [{'seq': ['-6', '4']}, {'seq': ['-5', '3']}, {'seq': ['-3', '4/3']}, {'seq': ['-2', '-4/3']}, {'seq': ['0', '-2/3']}, {'seq': ['1', '4']}]}, 'x', 'periodic', 'None']),
    ('cubic_spline', [{'seq': [{'seq': ['-2', '-1']}, {'seq': ['4', '1']}]}, 'x', 'natural', 'None']),
    ('cubic_spline', [{'seq': [{'seq': ['-2', '-3']}, {'seq': ['-1', '2']}, {'seq': ['0', '1']}, {'seq': ['1', '0']}, {'seq': ['2', '5']}]}, 'x', 'not-a-knot', 'None']),
    ('cubic_spline', [{'seq': [{'seq': [{'int': 0}, {'int': 0}]}, {'seq': [{'int': 1}, {'int': 1}]}]}, 'x', 'periodic', 'None']),
    ('cubic_spline', [{'seq': [{'seq': ['-5', '-1/3']}, {'seq': ['1', '8/3']}, {'seq': ['4', '-3']}, {'seq': ['5', '-5']}]}, 'x', 'clamped', {'seq': ['1', '-2']}]),
    ('cubic_spline', [{'seq': [{'seq': ['-6', '-4']}, {'seq': ['-3', '1/2']}, {'seq': ['-2', '-4']}]}, 'x', 'not-a-knot', 'None']),
    ('cubic_spline', [{'seq': [{'seq': ['-4', '-1/2']}, {'seq': ['-3', '-1/3']}, {'seq': ['4', '-4']}, {'seq': ['5', '-5']}]}, 'x', 'not-a-knot', 'None']),
    ('divided_differences', [{'seq': [{'seq': [{'int': 0}, {'int': 0}]}, {'seq': [{'int': 1}, {'int': 1}]}, {'seq': [{'int': 2}, {'int': 8}]}, {'seq': [{'int': 3}, {'int': 27}]}]}]),
    ('hermite_interpolation', [{'seq': [{'seq': ['-1', {'seq': ['1']}]}, {'seq': ['0', {'seq': ['0', '3']}]}, {'seq': ['1', {'seq': ['-3', '0']}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': ['3', {'seq': ['0', '-3', '-2']}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': [{'int': 0}, {'seq': []}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': [{'int': 0}, {'seq': [{'int': 1}, {'int': 0}]}]}, {'seq': [{'int': 1}, {'seq': [{'int': 0}, {'int': 0}]}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': ['-2', {'seq': ['3', '-2']}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': ['-3', {'seq': ['-3', '-1', '-1']}]}, {'seq': ['0', {'seq': ['1', '-1', '1']}]}, {'seq': ['2', {'seq': ['-2', '-1', '2']}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': ['-3', {'seq': ['-3']}]}, {'seq': ['0', {'seq': ['3']}]}, {'seq': ['2', {'seq': ['3']}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': ['-2', {'seq': ['-2', '3']}]}, {'seq': ['2', {'seq': ['1']}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': [{'int': 0}, {'seq': [{'int': 0}, {'int': 1}, {'int': 0}, {'int': -1}]}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': ['-2', {'seq': ['-2', '-2']}]}, {'seq': ['1', {'seq': ['-3', '-1', '2']}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': ['1', {'seq': ['-3', '-2']}]}]}, 'x']),
    ('hermite_interpolation', [{'seq': [{'seq': ['-3', {'seq': ['2']}]}, {'seq': ['-1', {'seq': ['1', '0']}]}, {'seq': ['2', {'seq': ['-1', '-1']}]}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-1', '-5']}, {'seq': ['3', '-2']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-4', '3']}, {'seq': ['3', '8']}, {'seq': ['5', '-2']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-6', '7']}, {'seq': ['-3', '2']}, {'seq': ['0', '3/2']}, {'seq': ['2', '-4']}, {'seq': ['3', '5/3']}, {'seq': ['6', '0']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-6', '-8/3']}, {'seq': ['-4', '5']}, {'seq': ['-3', '1']}, {'seq': ['-1', '-2']}, {'seq': ['2', '-4']}, {'seq': ['6', '7']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-6', '2']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-5', '8']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-3', '-5/3']}, {'seq': ['0', '2']}, {'seq': ['3', '-7/2']}, {'seq': ['6', '7']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-5', '4']}, {'seq': ['1', '0']}, {'seq': ['3', '0']}, {'seq': ['5', '1']}, {'seq': ['6', '8/3']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-3', '1']}, {'seq': ['-2', '-5/2']}, {'seq': ['-1', '7/3']}, {'seq': ['0', '-3']}, {'seq': ['4', '-5/2']}, {'seq': ['5', '1/3']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-1', '0']}, {'seq': ['0', '1']}, {'seq': ['4', '4']}, {'seq': ['6', '-6']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['4', '-2/3']}]}, 'x']),
    ('interpolating_polynomial', [{'seq': [{'seq': ['-4', '-5/2']}, {'seq': ['-3', '8/3']}, {'seq': ['1', '8/3']}, {'seq': ['3', '4/3']}, {'seq': ['4', '-5/2']}]}, 'x']),
    ('least_squares', [{'seq': [{'seq': [{'int': 0}, {'int': 1}]}]}, 'x', {'seq': []}]),
    ('least_squares', [{'seq': [{'seq': [{'int': 0}, {'int': 1}]}, {'seq': [{'int': 0}, {'int': 2}]}]}, 'x', {'seq': [{'int': 1}, 'x']}]),
    ('least_squares', [{'seq': [{'seq': ['0', '-1']}, {'seq': ['1', '-cos(1) + 2*sin(1)']}, {'seq': ['2', '-cos(2) + 2*sin(2)']}, {'seq': ['3', '2*sin(3) - cos(3)']}]}, 'x', {'seq': ['sin(x)', 'cos(x)']}]),
    ('maximize', ['2*x**4 + 4*x**3 + 4*x**2 + x + 3', 'True', 'None']),
    ('maximize', ['x**2 - 3*x + 5/2', 'True', {'seq': ['x']}]),
    ('maximize', ['3*x**4 + 3*x**3 - 4*x**2 - 2*x', 'True', 'None']),
    ('maximize', ['x**2 - 3*x/2 - 1/2', 'True', {'seq': ['x']}]),
    ('maximize', ['1/(x**2 + 1)', 'True', 'None']),
    ('maximize', ['x**2 + x', 'True', 'None']),
    ('maximize', ['x**4 - x**2/2', 'True', {'seq': ['x']}]),
    ('maximize', ['x**2 + y**2', 'Eq(x**2 + y**2, 4)', 'None']),
    ('maximize', ['x**2 + x/2 - 2', 'True', {'seq': ['x']}]),
    ('maximize', ['3*x**2/2 + x', 'True', {'seq': ['x']}]),
    ('maximize', ['x**2 + y**2', 'Eq(x + y, 0)', {'seq': ['x', 'y']}]),
    ('maximize', ['x**4 + 2*x**3 - 4*x + 3', 'True', 'None']),
    ('maximize_parametric', ['a*x + b + x**2', 'True', 'None', {'seq': ['a', 'b']}]),
    ('maximize_parametric', ['x**2 + y**2', 'Eq(x + y, a)', 'None', {'seq': ['a']}]),
    ('maximize_parametric', ['-a*x**2 + x**4', 'True', 'None', {'seq': ['a']}]),
    ('maximize_parametric', ['x**2', 'True', 'None', {'seq': ['a']}]),
    ('maximize_parametric', ['x*y', 'Eq(x**2 + y**2, a)', 'None', {'seq': ['a']}]),
    ('maximize_parametric', ['x', 'Eq(x**2, a)', 'None', {'seq': ['a']}]),
    ('maximize_parametric', ['a*x**2 + x', 'True', 'None', {'seq': ['a']}]),
    ('minimize', ['x**6 - 3*x**4 - x**3 - 3*x**2 - 3*x - 4', 'True', 'None']),
    ('minimize', ['x**2 - 5*x/2 + 1/2', 'True', {'seq': ['x']}]),
    ('minimize', ['x*z', 'True', {'seq': ['x']}]),
    ('minimize', ['x**2 - 3*x + 5/2', 'True', {'seq': ['x']}]),
    ('minimize', ['x', 'Eq(x**2, -3)', {'seq': ['x']}]),
    ('minimize', ['x**4 - x**2/2', 'True', {'seq': ['x']}]),
    ('minimize', ['x**2 + x', 'True', 'None']),
    ('minimize', ['x**2 + y**2', 'Eq(x + y, 3/2)', {'seq': ['x', 'y']}]),
    ('minimize', ['3*x**2 + 4*x*y - x + 3*y**2 + 2*y + 2', 'True', 'None']),
    ('minimize', ['3*x**2 - 6*x*y + 4*y**2 - 3*y + 3', 'True', 'None']),
    ('minimize', ['2*x**4 + x**3 + 4*x**2 - 4*x + 4', 'True', 'None']),
    ('minimize', ['x**4 - 4*x**3 - 3*x**2 + 3*x + 2', 'True', 'None']),
    ('minimize_parametric', ['x', 'Eq(x**2, a)', 'None', {'seq': ['a']}]),
    ('minimize_parametric', ['x*y', 'Eq(x**2 + y**2, a)', 'None', {'seq': ['a']}]),
    ('minimize_parametric', ['x**2', 'True', 'None', {'seq': []}]),
    ('minimize_parametric', ['a*x**2 + x', 'True', 'None', {'seq': ['a']}]),
    ('minimize_parametric', ['a/(x**2 + 1)', 'True', 'None', {'seq': ['a']}]),
    ('minimize_parametric', ['a + x*z', 'True', {'seq': ['x']}, {'seq': ['a']}]),
    ('minimize_parametric', ['x**2 + y**2', 'Eq(x + y, a)', 'None', {'seq': ['a']}]),
    ('minimize_parametric', ['-a*x**2 + x**4', 'True', 'None', {'seq': ['a']}]),
    ('minimize_parametric', ['a*x + b + x**2', 'True', 'None', {'seq': ['a', 'b']}]),
    ('polynomial_fit', [{'seq': [{'seq': ['-6', '-1']}, {'seq': ['-3', '-8']}, {'seq': ['-2', '2']}, {'seq': ['0', '-1/3']}]}, 'x', {'int': 0}]),
    ('polynomial_fit', [{'seq': [{'seq': ['-5', '-5/3']}, {'seq': ['-3', '-4']}, {'seq': ['1', '-2']}, {'seq': ['6', '-1/2']}]}, 'x', {'int': 1}]),
    ('polynomial_fit', [{'seq': [{'seq': [{'int': 0}, {'int': 1}]}]}, 'x', {'int': -1}]),
    ('polynomial_fit', [{'seq': [{'seq': ['-6', '0']}, {'seq': ['-4', '8/3']}, {'seq': ['-3', '2']}, {'seq': ['2', '-4']}]}, 'x', {'int': 0}]),
    ('polynomial_fit', [{'seq': [{'seq': ['-5', '-1']}, {'seq': ['0', '3']}, {'seq': ['1', '-8']}, {'seq': ['2', '5/2']}, {'seq': ['4', '1']}]}, 'x', {'int': 1}]),
    ('polynomial_fit', [{'seq': [{'seq': ['0', '3']}, {'seq': ['1', '3']}, {'seq': ['2', '5']}, {'seq': ['3', '9']}, {'seq': ['4', '15']}]}, 'x', {'int': 2}]),
    ('polynomial_fit', [{'seq': [{'seq': ['-6', '-1/3']}, {'seq': ['-3', '2']}, {'seq': ['-1', '0']}, {'seq': ['0', '4']}, {'seq': ['1', '1/3']}, {'seq': ['5', '-1']}]}, 'x', {'int': 0}]),
    ('polynomial_fit', [{'seq': [{'seq': ['-5', '-1/2']}, {'seq': ['-2', '-2/3']}, {'seq': ['1', '1']}, {'seq': ['4', '0']}]}, 'x', {'int': 1}]),
    ('polynomial_fit', [{'seq': [{'seq': ['-5', '-1']}, {'seq': ['0', '4']}, {'seq': ['1', '-8/3']}, {'seq': ['3', '-4']}]}, 'x', {'int': 1}]),
    ('polynomial_fit', [{'seq': [{'seq': ['-6', '4']}, {'seq': ['-5', '-2']}, {'seq': ['-4', '1/2']}, {'seq': ['-3', '1/3']}, {'seq': ['2', '-5/3']}, {'seq': ['4', '-3/2']}]}, 'x', {'int': 2}]),
    ('polynomial_fit', [{'seq': [{'seq': ['-3', '8']}, {'seq': ['0', '3']}, {'seq': ['1', '2']}, {'seq': ['4', '-7/3']}, {'seq': ['6', '2']}]}, 'x', {'int': 0}]),
    ('polynomial_fit', [{'seq': [{'seq': ['-5', '1']}, {'seq': ['-4', '-7/2']}, {'seq': ['0', '1']}, {'seq': ['1', '7']}, {'seq': ['2', '-7/3']}, {'seq': ['3', '0']}, {'seq': ['5', '-3']}]}, 'x', {'int': 2}]),
    ('value_range', ['x/(x**2 + 1)', 'True', 'None']),
    ('value_range', ['x**2', 'True', 'None']),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_calculus(i: int) -> None:
    name, raw = CASES[i]
    args = [argument(a) for a in raw]
    module = IP if hasattr(IP, name) else OP
    got = outcome(lambda: rc.logic_call('calculus', [Symbol(name)] + [encoded(a) for a in args]))
    if got != 'SKIP':
        assert got == outcome(lambda: encoded(getattr(module, name)(*args))), (name, raw)

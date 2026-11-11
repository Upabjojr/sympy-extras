"""Matrices of derivatives against SymPy's, and the second derivative
test on functions with known critical points."""
from __future__ import annotations

import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.functions import cos, exp, log, sin, sqrt
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.matrices.calculus import classify_critical_point, gradient, hessian, jacobian, wronskian

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
sx, sy, sz = sympy.symbols('x y z')


def _sympy_matrix(M: ExprMatrix) -> sympy.Matrix:
    return sympy.Matrix(M.rows, M.cols, [to_sympy(e) for e in M.entries])


def test_jacobian_and_hessian_against_sympy() -> None:
    functions = [x**2*sin(y) + z, exp(x*y)/(1 + z**2), log(x + y)*z]
    J = jacobian(functions, [x, y, z])
    expected = sympy.Matrix([to_sympy(f) for f in functions]).jacobian([sx, sy, sz])
    assert (_sympy_matrix(J) - expected).applyfunc(sympy.simplify) == sympy.zeros(3, 3)
    assert jacobian([x, y], [x, y, z]) == ExprMatrix.from_rows([[1, 0, 0], [0, 1, 0]])
    for f in functions:
        H = hessian(f, [x, y, z])
        assert (_sympy_matrix(H) - sympy.hessian(to_sympy(f), [sx, sy, sz])).applyfunc(sympy.simplify) == sympy.zeros(3, 3)
        assert gradient(f, [x, y, z]) == ExprMatrix.column([f.diff(x), f.diff(y), f.diff(z)])
    bordered = hessian(x**2 + y**2 + z**2, [x, y, z], [x + y + z - 1, x*y - z])
    expected = sympy.hessian(sx**2 + sy**2 + sz**2, [sx, sy, sz], [sx + sy + sz - 1, sx*sy - sz])
    assert _sympy_matrix(bordered) == expected


def test_wronskian() -> None:
    # Abel's formula for y'' + y = 0 and y'' - 3 y' + 2 y = 0
    assert wronskian([cos(x), sin(x)], x) == 1
    assert wronskian([exp(x), exp(2*x)], x) == exp(3*x)
    assert wronskian([Integer(1), x, x**2, x**3], x) == 12
    assert wronskian([sin(x)**2, cos(x)**2, Integer(1)], x) == 0          # dependent functions
    assert wronskian([], x) == 1 and wronskian([sqrt(x)], x) == sqrt(x)
    expected = sympy.wronskian([sx, sympy.exp(sx), sx*sympy.exp(sx)], sx)
    assert sympy.simplify(to_sympy(wronskian([x, exp(x), x*exp(x)], x)) - expected) == 0


def test_second_derivative_test() -> None:
    # Marsden, Tromba, Vector Calculus, Section 3.3
    assert classify_critical_point(x**2 + y**2, [x, y], [0, 0]) == 'minimum'
    assert classify_critical_point(-x**2 - x*y - y**2, [x, y], [0, 0]) == 'maximum'
    assert classify_critical_point(x*y, [x, y], [0, 0]) == 'saddle'
    assert classify_critical_point(x**4 + y**4, [x, y], [0, 0]) is None
    assert classify_critical_point(x**2 + y**3, [x, y], [0, 0]) is None
    f = x**3 + y**3 - 3*x*y
    assert classify_critical_point(f, [x, y], [1, 1]) == 'minimum' and classify_critical_point(f, [x, y], [0, 0]) == 'saddle'
    # the monkey saddle of three variables, a point with rational coordinates, functions evaluated at the point
    assert classify_critical_point(x**2 - y**2 + z**2, [x, y, z], [0, 0, 0]) == 'saddle'
    assert classify_critical_point((x - Rational(1, 2))**2 + (y + 3)**2, [x, y], [Rational(1, 2), -3]) == 'minimum'
    assert classify_critical_point(cos(x) + cos(y), [x, y], [0, 0]) == 'maximum'
    # irrational Hessians: the signs come from the assumptions of the core
    assert classify_critical_point(x*exp(-x**2 - y**2), [x, y], [-sqrt(2)/2, 0]) == 'minimum'
    assert classify_critical_point(x*exp(-x**2 - y**2), [x, y], [sqrt(2)/2, 0]) == 'maximum'
    assert classify_critical_point(sqrt(2)*x**2 - y**2, [x, y], [0, 0]) == 'saddle'
    p = Symbol('p', positive=True)
    assert classify_critical_point(p*x**2 + y**2, [x, y], [0, 0]) == 'minimum'
    with pytest.raises(ValueError):
        classify_critical_point(x**2 + y, [x, y], [0, 0])
    with pytest.raises(ValueError):
        classify_critical_point(x**2, [x, y], [0])
    # a parameter of unknown sign
    with pytest.raises(ValueError):
        classify_critical_point(z*x**2 + y**2, [x, y], [0, 0])
    assert sympy_expr(to_sympy(hessian(x*y, [x, y])[0, 1])) == 1

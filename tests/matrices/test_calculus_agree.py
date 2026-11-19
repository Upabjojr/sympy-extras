"""The Rust ports of ``matrices/calculus.py``, ``vector/orthogonal.py``,
``solvers/elimination.py``, ``series/fouriercoefficients.py`` and
``diffgeom/classical.py`` against the Python ones on the calls of their
tests (the default simplifier only; read with the core's parser)."""
from __future__ import annotations

import ast
from typing import Callable

import pytest

from sympy_extras.core import Eq, Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
import sympy_extras.diffgeom.classical as DG
import sympy_extras.matrices.calculus as MC
import sympy_extras.series.fouriercoefficients as FC
import sympy_extras.solvers.elimination as EL
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.vector.orthogonal import OrthogonalCoordinates

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
    if isinstance(v, str):
        return Symbol(v)
    if isinstance(v, ExprMatrix):
        return Tuple(Integer(v.rows), Integer(v.cols), Tuple(*v.entries))
    if isinstance(v, (list, tuple)):
        return Tuple(*[encoded(x) for x in v])
    if isinstance(v, int):
        return Integer(v)
    assert isinstance(v, Basic)
    return v


def argument(a: object) -> object:
    if isinstance(a, str):
        if a == 'None':
            return None
        node = ast.parse(a, mode='eval').body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Eq':
            return Eq(parse(ast.unparse(node.args[0])), parse(ast.unparse(node.args[1])))
        return parse(a)
    assert isinstance(a, dict)
    if 'int' in a:
        return a['int']
    items = a['seq']
    assert isinstance(items, list)
    return [argument(x) for x in items]


def python(name: str, args: list[object]) -> object:
    if name.startswith(('curve.', 'surface.')):
        kind, method = name.split('.', 1)
        components = args[0]
        assert isinstance(components, list)
        if kind == 'curve':
            assert isinstance(args[1], Symbol)
            found = getattr(DG.Curve(components, args[1]), method)(*args[2:])
        else:
            assert isinstance(args[1], Symbol) and isinstance(args[2], Symbol)
            found = getattr(DG.Surface(components, args[1], args[2]), method)(*args[3:])
        return found
    if name.startswith('orthogonal.'):
        method = name.split('.', 1)[1]
        coordinates, rest = args[0], args[1:]
        assert isinstance(coordinates, list)
        if method == 'from_transformation':
            assert isinstance(rest[0], list)
            c = OrthogonalCoordinates.from_transformation(coordinates, rest[0])
            return [list(c.coordinates), list(c.scale_factors)]
        assert isinstance(rest[0], list)
        system = OrthogonalCoordinates(coordinates, rest[0])
        return getattr(system, method)(*rest[1:])
    if hasattr(FC, name):
        found = getattr(FC, name)(*args)
        if isinstance(found, FC.FourierCoefficients):
            return [found.a0, found.an, found.bn, found.n, found.omega]
        return found
    return getattr(EL if hasattr(EL, name) else MC, name)(*args)


CASES: list[tuple[str, list[object]]] = [
    ('jacobian', [{'seq': ['x**2*sin(y) + z', 'exp(x*y)/(z**2 + 1)', 'z*log(x + y)']}, {'seq': ['x', 'y', 'z']}]),
    ('jacobian', [{'seq': ['x', 'y']}, {'seq': ['x', 'y', 'z']}]),
    ('hessian', ['x**2*sin(y) + z', {'seq': ['x', 'y', 'z']}, {'seq': []}]),
    ('gradient', ['x**2*sin(y) + z', {'seq': ['x', 'y', 'z']}]),
    ('hessian', ['exp(x*y)/(z**2 + 1)', {'seq': ['x', 'y', 'z']}, {'seq': []}]),
    ('gradient', ['exp(x*y)/(z**2 + 1)', {'seq': ['x', 'y', 'z']}]),
    ('hessian', ['z*log(x + y)', {'seq': ['x', 'y', 'z']}, {'seq': []}]),
    ('gradient', ['z*log(x + y)', {'seq': ['x', 'y', 'z']}]),
    ('hessian', ['x**2 + y**2 + z**2', {'seq': ['x', 'y', 'z']}, {'seq': ['x + y + z - 1', 'x*y - z']}]),
    ('wronskian', [{'seq': ['cos(x)', 'sin(x)']}, 'x']),
    ('wronskian', [{'seq': ['exp(x)', 'exp(2*x)']}, 'x']),
    ('wronskian', [{'seq': ['1', 'x', 'x**2', 'x**3']}, 'x']),
    ('wronskian', [{'seq': ['sin(x)**2', 'cos(x)**2', '1']}, 'x']),
    ('wronskian', [{'seq': []}, 'x']),
    ('wronskian', [{'seq': ['sqrt(x)']}, 'x']),
    ('wronskian', [{'seq': ['x', 'exp(x)', 'x*exp(x)']}, 'x']),
    ('classify_critical_point', ['x**2 + y**2', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['-x**2 - x*y - y**2', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['x*y', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['x**4 + y**4', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['x**2 + y**3', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['x**3 - 3*x*y + y**3', {'seq': ['x', 'y']}, {'seq': [{'int': 1}, {'int': 1}]}]),
    ('classify_critical_point', ['x**3 - 3*x*y + y**3', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['x**2 - y**2 + z**2', {'seq': ['x', 'y', 'z']}, {'seq': [{'int': 0}, {'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['(x - 1/2)**2 + (y + 3)**2', {'seq': ['x', 'y']}, {'seq': ['1/2', {'int': -3}]}]),
    ('classify_critical_point', ['cos(x) + cos(y)', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['x*exp(-x**2 - y**2)', {'seq': ['x', 'y']}, {'seq': ['-sqrt(2)/2', {'int': 0}]}]),
    ('classify_critical_point', ['x*exp(-x**2 - y**2)', {'seq': ['x', 'y']}, {'seq': ['sqrt(2)/2', {'int': 0}]}]),
    ('classify_critical_point', ['sqrt(2)*x**2 - y**2', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['p*x**2 + y**2', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['x**2 + y', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('classify_critical_point', ['x**2', {'seq': ['x', 'y']}, {'seq': [{'int': 0}]}]),
    ('classify_critical_point', ['x**2*z + y**2', {'seq': ['x', 'y']}, {'seq': [{'int': 0}, {'int': 0}]}]),
    ('hessian', ['x*y', {'seq': ['x', 'y']}, {'seq': []}]),
    ('orthogonal.gradient', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, 'f(r, theta, phi)']),
    ('orthogonal.laplacian', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, 'f(r, theta, phi)']),
    ('orthogonal.divergence', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, {'seq': ['Fr(r, theta, phi)', 'Ft(r, theta, phi)', 'Fp(r, theta, phi)']}]),
    ('orthogonal.curl', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, {'seq': ['Fr(r, theta, phi)', 'Ft(r, theta, phi)', 'Fp(r, theta, phi)']}]),
    ('orthogonal.laplacian', [{'seq': ['rho', 'phi', 'z']}, {'seq': ['1', 'rho', '1']}, 'g(rho, phi, z)']),
    ('orthogonal.volume_element', [{'seq': ['rho', 'phi', 'z']}, {'seq': ['1', 'rho', '1']}]),
    ('orthogonal.volume_element', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}]),
    ('orthogonal.laplacian', [{'seq': ['x', 'y', 'z']}, {'seq': ['1', '1', '1']}, 'x**2*y + z**3']),
    ('orthogonal.from_transformation', [{'seq': ['u', 'v', 'z']}, {'seq': ['u**2/2 - v**2/2', 'u*v', 'z']}]),
    ('orthogonal.gradient', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, 'phi*r + r**2*sin(theta) + theta*exp(phi)']),
    ('orthogonal.curl', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, {'seq': ['phi + 2*r*sin(theta)', '(r**2*cos(theta) + exp(phi))/r', '(r + theta*exp(phi))/(r*sin(theta))']}]),
    ('orthogonal.curl', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, {'seq': ['r*theta', 'phi*sin(theta) + r**2', 'phi**2*r + theta']}]),
    ('orthogonal.divergence', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, {'seq': ['(phi**2*r*cos(theta) + theta*cos(theta))/(r*sin(theta))', '(-2*phi**2*r - theta)/r', '(phi*sin(theta) + 3*r**2 - r)/r']}]),
    ('orthogonal.laplacian', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, 'phi*r + r**2*sin(theta) + theta*exp(phi)']),
    ('orthogonal.divergence', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, {'seq': ['phi + 2*r*sin(theta)', '(r**2*cos(theta) + exp(phi))/r', '(r + theta*exp(phi))/(r*sin(theta))']}]),
    ('orthogonal.divergence', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, {'seq': ['r*theta', 'phi*sin(theta) + r**2', 'phi**2*r + theta']}]),
    ('orthogonal.gradient', [{'seq': ['rho', 'phi', 'z']}, {'seq': ['1', 'rho', '1']}, 'phi*exp(z) + rho**2*sin(phi) + rho*z']),
    ('orthogonal.curl', [{'seq': ['rho', 'phi', 'z']}, {'seq': ['1', 'rho', '1']}, {'seq': ['2*rho*sin(phi) + z', '(rho**2*cos(phi) + exp(z))/rho', 'phi*exp(z) + rho']}]),
    ('orthogonal.curl', [{'seq': ['rho', 'phi', 'z']}, {'seq': ['1', 'rho', '1']}, {'seq': ['phi*rho', 'rho**2 + z*sin(phi)', 'phi + rho*z**2']}]),
    ('orthogonal.divergence', [{'seq': ['rho', 'phi', 'z']}, {'seq': ['1', 'rho', '1']}, {'seq': ['(-rho*sin(phi) + 1)/rho', '-z**2', '(3*rho**2 - rho + z*sin(phi))/rho']}]),
    ('orthogonal.laplacian', [{'seq': ['rho', 'phi', 'z']}, {'seq': ['1', 'rho', '1']}, 'phi*exp(z) + rho**2*sin(phi) + rho*z']),
    ('orthogonal.divergence', [{'seq': ['rho', 'phi', 'z']}, {'seq': ['1', 'rho', '1']}, {'seq': ['2*rho*sin(phi) + z', '(rho**2*cos(phi) + exp(z))/rho', 'phi*exp(z) + rho']}]),
    ('orthogonal.divergence', [{'seq': ['rho', 'phi', 'z']}, {'seq': ['1', 'rho', '1']}, {'seq': ['phi*rho', 'rho**2 + z*sin(phi)', 'phi + rho*z**2']}]),
    ('orthogonal.gradient', [{'seq': ['u', 'v', 'z']}, {'seq': ['sqrt(u**2 + v**2)', 'sqrt(u**2 + v**2)', '1']}, 'u**2*sin(v) + u*z + v*exp(z)']),
    ('orthogonal.curl', [{'seq': ['u', 'v', 'z']}, {'seq': ['sqrt(u**2 + v**2)', 'sqrt(u**2 + v**2)', '1']}, {'seq': ['(2*u*sin(v) + z)/sqrt(u**2 + v**2)', '(u**2*cos(v) + exp(z))/sqrt(u**2 + v**2)', 'u + v*exp(z)']}]),
    ('orthogonal.curl', [{'seq': ['u', 'v', 'z']}, {'seq': ['sqrt(u**2 + v**2)', 'sqrt(u**2 + v**2)', '1']}, {'seq': ['u*v', 'u**2 + z*sin(v)', 'u*z**2 + v']}]),
    ('orthogonal.divergence', [{'seq': ['u', 'v', 'z']}, {'seq': ['sqrt(u**2 + v**2)', 'sqrt(u**2 + v**2)', '1']}, {'seq': ['(-sqrt(u**2 + v**2)*sin(v) + 1)/sqrt(u**2 + v**2)', '-z**2/sqrt(u**2 + v**2)', '(2*u**3 + u*z*sin(v))/(u**2*sqrt(u**2 + v**2) + v**2*sqrt(u**2 + v**2))']}]),
    ('orthogonal.laplacian', [{'seq': ['u', 'v', 'z']}, {'seq': ['sqrt(u**2 + v**2)', 'sqrt(u**2 + v**2)', '1']}, 'u**2*sin(v) + u*z + v*exp(z)']),
    ('orthogonal.divergence', [{'seq': ['u', 'v', 'z']}, {'seq': ['sqrt(u**2 + v**2)', 'sqrt(u**2 + v**2)', '1']}, {'seq': ['(2*u*sin(v) + z)/sqrt(u**2 + v**2)', '(u**2*cos(v) + exp(z))/sqrt(u**2 + v**2)', 'u + v*exp(z)']}]),
    ('orthogonal.divergence', [{'seq': ['u', 'v', 'z']}, {'seq': ['sqrt(u**2 + v**2)', 'sqrt(u**2 + v**2)', '1']}, {'seq': ['u*v', 'u**2 + z*sin(v)', 'u*z**2 + v']}]),
    ('orthogonal.laplacian', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, 'r**3*sin(phi)*sin(theta)**3*cos(phi)**2 + r*cos(theta)']),
    ('orthogonal.gradient', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, 'r**3*sin(phi)*sin(theta)**3*cos(phi)**2 + r*cos(theta)']),
    ('orthogonal.curl', [{'seq': ['rho', 'phi', 'z']}, {'seq': ['1', 'rho', '1']}, {'seq': [{'int': 0}, 'rho', {'int': 0}]}]),
    ('orthogonal.vector_laplacian', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['1', 'r', 'r*sin(theta)']}, {'seq': ['r', {'int': 0}, {'int': 0}]}]),
    ('orthogonal.from_transformation', [{'seq': ['r', 'theta', 'phi']}, {'seq': ['r*sin(theta)*cos(phi)', 'r*sin(phi)*sin(theta)', 'r*cos(theta)']}]),
    ('orthogonal.from_transformation', [{'seq': ['u', 'v']}, {'seq': ['u**2/2 - v**2/2', 'u*v']}]),
    ('orthogonal.laplacian', [{'seq': ['u', 'v']}, {'seq': ['sqrt(u**2 + v**2)', 'sqrt(u**2 + v**2)']}, 'u**3 - 3*u*v**2']),
    ('orthogonal.from_transformation', [{'seq': ['u', 'v']}, {'seq': ['u + v', 'v']}]),
    ('orthogonal.curl', [{'seq': ['u', 'v']}, {'seq': ['sqrt(u**2 + v**2)', 'sqrt(u**2 + v**2)']}, {'seq': [{'int': 1}, {'int': 0}]}]),
    ('orthogonal.divergence', [{'seq': ['u', 'v']}, {'seq': ['sqrt(u**2 + v**2)', 'sqrt(u**2 + v**2)']}, {'seq': [{'int': 1}, {'int': 0}, {'int': 0}]}]),
    ('implicitize', [{'seq': ['t', 't**2', 't**3']}, {'seq': ['t']}, {'seq': ['x', 'y', 'z']}]),
    ('implicitize', [{'seq': ['t**2', 't**3']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['(1 - t**2)/(t**2 + 1)', '2*t/(t**2 + 1)']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['cos(t)', 'sin(t)']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['3*t/(t**3 + 1)', '3*t**2/(t**3 + 1)']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['cos(t)/(sin(t)**2 + 1)', 'sin(t)*cos(t)/(sin(t)**2 + 1)']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['cos(t)', 'sin(2*t)']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['cos(t)**3', 'sin(t)**3']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['s*t', 's*t**2', 's**2']}, {'seq': ['s', 't']}, {'seq': ['x', 'y', 'z']}]),
    ('implicitize', [{'seq': ['sin(s)*cos(t)', 'sin(s)*sin(t)', 'cos(s)']}, {'seq': ['s', 't']}, {'seq': ['x', 'y', 'z']}]),
    ('implicitize', [{'seq': ['cos(t)*cosh(s)', 'sin(t)*cosh(s)', 'sinh(s)']}, {'seq': ['s', 't']}, {'seq': ['x', 'y', 'z']}]),
    ('implicitize', [{'seq': ['(cos(s) + 2)*cos(t)', '(cos(s) + 2)*sin(t)', 'sin(s)']}, {'seq': ['s', 't']}, {'seq': ['x', 'y', 'z']}]),
    ('implicitize', [{'seq': ['s*t', 's', 't']}, {'seq': ['s', 't']}, {'seq': ['x', 'y', 'z']}]),
    ('implicitize', [{'seq': ['a*cos(t)', 'b*sin(t)']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['t + 1/t', 't - 1/t']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['sqrt(t)', 't**2 + 1']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['t**3 - t', 'a + t**2']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['tan(t)', '1/cos(t)']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['cos(t) + cos(2*t)', 'sin(t) - sin(2*t)']}, {'seq': ['t']}, {'seq': ['x', 'y']}]),
    ('implicitize', [{'seq': ['t**2 - 1', 't*(t**2 - 1)', 't**4']}, {'seq': ['t']}, {'seq': ['x', 'y', 'z']}]),
    ('eliminate', [{'seq': ['2*x**2 + x*y - x - y**2 + 2*y - 2', '-x**2 - x*y + x + y**2 + y + 1']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['2*x**2 - x*y - x + y**2 + 2', 'x**2 + 3*x + y**2 - 2*y - 2']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['x*y + 2*x + y**2 + 2', '-x**2 + 2*x*y + x + y**2 - y']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['2*x**2 + x*y - 2*x + 2*y**2 - 2*y', 'x**2 + 2*x*y - x - y**2 + 2*y - 1']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['3*x**2 - x*y + 2*y**2 + 1', '-x**2 + 2*x*y + 2*x + y + 2']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['2*x**2 + 2*x*y - 2*x + y', '2*x**2 + 2*x*y + 2*y - 1']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['-x**2 + x*y - 2*x + 2*y**2 - y + 2', '2*x**2 - 2*x*y + 2*y**2 - 2']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['x**2 - x*y + 2*y**2 - 2*y + 1', '2*x**2 - x*y + y**2 - 2*y']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['x**2 + x*y - x + 2*y**2 + y + 2', '-x**2 - 2*x*y + 2']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['x*y + 2*x + 2*y**2 + 2*y - 2', 'x*y + x + y**2 + y - 2']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['x**2 - x*y - x + y**2 + 2*y + 1', '-2*x**2 + x*y + 3*x - 2*y**2 - y + 2']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['x**2 + x + 2*y**2 - y - 1', 'x - 2*y**2 + 2*y + 1']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['Eq(P*V, R*T*n)', 'Eq(U, T*c*n)']}, {'seq': ['T']}]),
    ('eliminate', [{'seq': ['x + y + z - 1', 'x - y + 2*z']}, {'seq': ['x', 'y']}]),
    ('eliminate', [{'seq': ['x + y + z - 1', 'x - y + 2*z', 'x - 3']}, {'seq': ['x', 'y']}]),
    ('eliminate', [{'seq': ['Eq(x, t*sin(a))', 'Eq(y, t**2)']}, {'seq': ['t']}]),
    ('eliminate', [{'seq': ['Eq(x, exp(t))', 'Eq(y, exp(2*t) + 1)']}, {'seq': ['t']}]),
    ('eliminate', [{'seq': ['Eq(x, log(t))', 'Eq(y, 2*log(t) + 1)']}, {'seq': ['t']}]),
    ('eliminate', [{'seq': ['Eq(2*x, exp(t) + exp(-t))', 'Eq(2*y, exp(t) - exp(-t))']}, {'seq': ['t']}]),
    ('eliminate', [{'seq': ['Eq(x, exp(t/2))', 'Eq(y, exp(t/3))']}, {'seq': ['t']}]),
    ('eliminate', [{'seq': []}, {'seq': ['t']}]),
    ('eliminate', [{'seq': ['0']}, {'seq': ['t']}]),
    ('eliminate', [{'seq': ['x - y']}, {'seq': ['z']}]),
    ('eliminate', [{'seq': ['x', 'x - 1']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['x**2 - 1']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['1']}, {'seq': ['x']}]),
    ('eliminate', [{'seq': ['Eq(x, 1)']}, {'seq': ['t']}]),
    ('eliminate', [{'seq': ['Eq(t*x, 1)', 'Eq(t*y, 1)']}, {'seq': ['t']}]),
    ('eliminate', [{'seq': ['-I*t + x']}, {'seq': ['t']}]),
    ('implicitize', [{'seq': ['t', 't**2']}, {'seq': ['t']}, {'seq': ['x']}]),
    ('implicitize', [{'seq': ['t', 't**2']}, {'seq': ['t']}, {'seq': ['t', 'y']}]),
    ('fourier_coefficients', ['x', 'x', {'seq': ['-pi', 'pi']}, 'n']),
    ('fourier_coefficients', ['x**2', 'x', {'seq': ['-pi', 'pi']}, 'n']),
    ('fourier_partial_sum', ['x**2', 'x', {'seq': ['-pi', 'pi']}, {'int': 2}]),
    ('fourier_coefficients', ['x**3 - x', 'x', {'seq': ['-1', '1']}, 'None']),
    ('fourier_coefficients', ['x**2 + 1', 'x', {'seq': ['0', '3']}, 'None']),
    ('fourier_coefficients', ['x*(pi - x)', 'x', {'seq': ['0', 'pi']}, 'None']),
    ('fourier_coefficients', ['x', 'x', {'seq': ['-pi', 'pi']}, 'None']),
    ('fourier_coefficients', ['x', 'x', {'seq': [{'int': 1}, {'int': 0}]}, 'None']),
    ('fourier_coefficients', ['x', 'x', {'seq': [{'int': 0}, {'int': 1}]}, 'm']),
    ('fourier_coefficients', ['sin(x**2)', 'x', {'seq': [{'int': 0}, {'int': 1}]}, 'None']),
    ('curve.speed', [{'seq': ['a*cos(t)', 'a*sin(t)', 'b*t']}, 't']),
    ('curve.curvature', [{'seq': ['a*cos(t)', 'a*sin(t)', 'b*t']}, 't']),
    ('curve.torsion', [{'seq': ['a*cos(t)', 'a*sin(t)', 'b*t']}, 't']),
    ('curve.frenet_frame', [{'seq': ['a*cos(t)', 'a*sin(t)', 'b*t']}, 't']),
    ('curve.curvature', [{'seq': ['R*cos(t)', 'R*sin(t)']}, 't']),
    ('curve.signed_curvature', [{'seq': ['R*cos(t)', 'R*sin(t)']}, 't']),
    ('curve.torsion', [{'seq': ['R*cos(t)', 'R*sin(t)']}, 't']),
    ('curve.signed_curvature', [{'seq': ['R*cos(t)', '-R*sin(t)']}, 't']),
    ('curve.tangent', [{'seq': ['R*cos(t)', 'R*sin(t)']}, 't']),
    ('curve.curvature', [{'seq': ['t', 't**2']}, 't']),
    ('curve.curvature', [{'seq': ['t', 'cosh(t)']}, 't']),
    ('curve.curvature', [{'seq': ['a*cos(t)', 'b*sin(t)']}, 't']),
    ('curve.curvature', [{'seq': ['t', '2*t', '3*t']}, 't']),
    ('curve.torsion', [{'seq': ['t', '2*t', '3*t']}, 't']),
    ('curve.frenet_frame', [{'seq': ['t', '2*t', '3*t']}, 't']),
    ('curve.signed_curvature', [{'seq': ['t', '2*t', '3*t']}, 't']),
    ('curve.curvature', [{'seq': ['t', 't**2', 't**3']}, 't']),
    ('curve.torsion', [{'seq': ['t', 't**2', 't**3']}, 't']),
    ('curve.curvature', [{'seq': ['c*cos(t)', 'c*sin(t)', 't']}, 't']),
    ('surface.first_fundamental_form', [{'seq': ['R*sin(u)*cos(v)', 'R*sin(u)*sin(v)', 'R*cos(u)']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['R*sin(u)*cos(v)', 'R*sin(u)*sin(v)', 'R*cos(u)']}, 'u', 'v']),
    ('surface.mean_curvature', [{'seq': ['R*sin(u)*cos(v)', 'R*sin(u)*sin(v)', 'R*cos(u)']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['(R + r*cos(u))*cos(v)', '(R + r*cos(u))*sin(v)', 'r*sin(u)']}, 'u', 'v']),
    ('surface.mean_curvature', [{'seq': ['cos(v)*cosh(u)', 'sin(v)*cosh(u)', 'u']}, 'u', 'v']),
    ('surface.mean_curvature', [{'seq': ['u*cos(v)', 'u*sin(v)', 'v']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['cos(v)*cosh(u)', 'sin(v)*cosh(u)', 'u']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['u*cos(v)', 'u*sin(v)', 'v']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['cos(v)/cosh(u)', 'sin(v)/cosh(u)', 'u - sinh(u)/cosh(u)']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['cos(u)', 'sin(u)', 'v']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['v*cos(u)', 'v*sin(u)', 'v']}, 'u', 'v']),
    ('surface.first_fundamental_form', [{'seq': ['(R + r*cos(u))*cos(v)', '(R + r*cos(u))*sin(v)', 'r*sin(u)']}, 'u', 'v']),
    ('surface.first_fundamental_form', [{'seq': ['cos(v)*cosh(u)', 'sin(v)*cosh(u)', 'u']}, 'u', 'v']),
    ('surface.first_fundamental_form', [{'seq': ['u*cos(v)', 'u*sin(v)', 'v']}, 'u', 'v']),
    ('surface.first_fundamental_form', [{'seq': ['cos(v)/cosh(u)', 'sin(v)/cosh(u)', 'u - sinh(u)/cosh(u)']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['u', 'v', 'u*v']}, 'u', 'v']),
    ('surface.first_fundamental_form', [{'seq': ['u', 'v', 'u*v']}, 'u', 'v']),
    ('surface.second_fundamental_form', [{'seq': ['u', 'v', 'u*v']}, 'u', 'v']),
    ('surface.normal', [{'seq': ['u', 'v', 'u*v']}, 'u', 'v']),
    ('surface.principal_curvatures', [{'seq': ['u', 'v', 'u*v']}, 'u', 'v']),
    ('surface.mean_curvature', [{'seq': ['u', 'v', 'u*v']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['u', 'v', 'u**2 - v**2']}, 'u', 'v']),
    ('surface.first_fundamental_form', [{'seq': ['u', 'v', 'u**2 - v**2']}, 'u', 'v']),
    ('surface.second_fundamental_form', [{'seq': ['u', 'v', 'u**2 - v**2']}, 'u', 'v']),
    ('surface.normal', [{'seq': ['u', 'v', 'u**2 - v**2']}, 'u', 'v']),
    ('surface.principal_curvatures', [{'seq': ['u', 'v', 'u**2 - v**2']}, 'u', 'v']),
    ('surface.mean_curvature', [{'seq': ['u', 'v', 'u**2 - v**2']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['u', 'v', 'u**3 + u*v**2']}, 'u', 'v']),
    ('surface.first_fundamental_form', [{'seq': ['u', 'v', 'u**3 + u*v**2']}, 'u', 'v']),
    ('surface.second_fundamental_form', [{'seq': ['u', 'v', 'u**3 + u*v**2']}, 'u', 'v']),
    ('surface.normal', [{'seq': ['u', 'v', 'u**3 + u*v**2']}, 'u', 'v']),
    ('surface.principal_curvatures', [{'seq': ['u', 'v', 'u**3 + u*v**2']}, 'u', 'v']),
    ('surface.mean_curvature', [{'seq': ['u', 'v', 'u**3 + u*v**2']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['u', 'v', 'v*exp(u)']}, 'u', 'v']),
    ('surface.first_fundamental_form', [{'seq': ['u', 'v', 'v*exp(u)']}, 'u', 'v']),
    ('surface.second_fundamental_form', [{'seq': ['u', 'v', 'v*exp(u)']}, 'u', 'v']),
    ('surface.normal', [{'seq': ['u', 'v', 'v*exp(u)']}, 'u', 'v']),
    ('surface.principal_curvatures', [{'seq': ['u', 'v', 'v*exp(u)']}, 'u', 'v']),
    ('surface.mean_curvature', [{'seq': ['u', 'v', 'v*exp(u)']}, 'u', 'v']),
    ('surface.area_element', [{'seq': ['u', 'v', 'u*v']}, 'u', 'v']),
    ('surface.gaussian_curvature', [{'seq': ['u + v', 'u + v', 'u + v']}, 'u', 'v']),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_calculus(i: int) -> None:
    name, raw = CASES[i]
    args = [argument(a) for a in raw]
    head: list[Basic] = [Symbol(name.split('.', 1)[0]), Symbol(name.split('.', 1)[1])] if '.' in name else [Symbol(name)]
    got = outcome(lambda: rc.logic_call('calculus', head + [encoded(a) for a in args]))
    if got != 'SKIP':
        assert got == outcome(lambda: encoded(python(name, args))), (name, raw)

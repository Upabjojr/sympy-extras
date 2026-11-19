"""The Rust port of ``diffgeom/tensors.py`` against the Python one: the
methods of ``Tensor`` and the tensors of a metric, on the calls of its
tests (the default simplifier only; read with the core's parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
import sympy_extras.diffgeom.tensors as TN
from sympy_extras.diffgeom.metric import Metric
from sympy_extras.matrices.dense import ExprMatrix

from ..conftest import load

rc = load("sympy_extras_rs.core")

METHODS = {'getitem': '__getitem__', 'add': '__add__', 'sub': '__sub__', 'mul': '__mul__', 'neg': '__neg__'}


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return str(r)


def encoded(v: object) -> Basic:
    if isinstance(v, TN.Tensor):
        return Tuple(Symbol(v.positions), Tuple(*v.components))
    if isinstance(v, bool):
        return Symbol(str(v))
    if isinstance(v, str):
        return Symbol(v)
    if isinstance(v, (list, tuple)):
        return Tuple(*[encoded(x) for x in v])
    if isinstance(v, int):
        return Integer(v)
    assert isinstance(v, Basic)
    return v


def argument(a: object, metric: Metric) -> object:
    if isinstance(a, str):
        return a if set(a) <= set('ud') and a else parse(a)
    assert isinstance(a, dict)
    if 'int' in a:
        return a['int']
    if 'tensor' in a:
        positions, components = a['tensor']
        assert isinstance(positions, str) and isinstance(components, list)
        return TN.Tensor(metric, [parse(str(c)) for c in components], positions)
    items = a['seq']
    assert isinstance(items, list)
    return [argument(x, metric) for x in items]


CASES: list[list[object]] = [
    ['add', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['ud', ['1', '2', '3', '4']]}, [{'tensor': ['dd', ['1', '2', '3', '4']]}]],
    ['add', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['dd', ['theta', 'phi/2 + 1/2', 'phi/2 + 1/2', 'phi*theta']]}, [{'tensor': ['dd', ['0', 'phi/2 - 1/2', '1/2 - phi/2', '0']]}]],
    ['antisymmetrize', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['dd', ['theta', 'phi', '1', 'phi*theta']]}, [{'int': 0}, {'int': 1}]],
    ['contract', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], {'tensor': ['du', ['12*m**2/r**6', '0', '0', '0', '0', '12*m**2/r**6', '0', '0', '0', '0', '12*m**2/r**6', '0', '0', '0', '0', '12*m**2/r**6']]}, [{'int': 0}, {'int': 1}]],
    ['contract', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['ud', ['phi*theta**3', 'phi**2*theta', 'theta**2*sin(theta)', 'phi*sin(theta)']]}, [{'int': 0}, {'int': 1}]],
    ['contract', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['udd', ['3*phi*theta**2', '(-phi**2*theta*cos(theta) + theta**3*sin(theta) - theta**2*sin(theta)**3*cos(theta))/sin(theta)', '(-phi**2*theta*cos(theta) + phi**2*sin(theta))/sin(theta)', 'phi*theta**3*sin(theta)*cos(theta) + 2*phi*theta - phi*sin(theta)**2*cos(theta)', '2*theta**2*cos(theta) + 2*theta*sin(theta)', '(phi*theta**3*cos(theta) - phi*sin(theta)*cos(theta))/sin(theta)', 'phi*cos(theta)', '(phi**2*theta*cos(theta) + theta**2*sin(theta)**3*cos(theta) + sin(theta)**2)/sin(theta)']]}, [{'int': 0}, {'int': 1}]],
    ['contract', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['dd', ['1', '2', '3', '4']]}, [{'int': 0}, {'int': 1}]],
    ['covariant_derivative', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['u', ['x*y', 'x + z', 'y**2']]}, []],
    ['covariant_derivative', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['u', ['phi*theta', 'sin(theta)']]}, []],
    ['covariant_derivative', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['d', ['0', 'sin(theta)**2']]}, []],
    ['covariant_derivative', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], {'tensor': ['dd', ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']]}, []],
    ['covariant_derivative', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['d', ['y', 'x*z', '1']]}, []],
    ['covariant_derivative', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['ud', ['phi*theta**3', 'phi**2*theta', 'theta**2*sin(theta)', 'phi*sin(theta)']]}, []],
    ['einstein_tensor', [[4, 4, ['-1', '0', '0', '0', '0', 'a(t)**2', '0', '0', '0', '0', 'a(t)**2', '0', '0', '0', '0', 'a(t)**2']], ['t', 'x', 'y', 'z']], None, []],
    ['einstein_tensor', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], None, []],
    ['getitem', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['ud', ['1', '2', '3', '4']]}, [{'seq': [{'int': 0}, {'int': 5}]}]],
    ['getitem', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['ud', ['1', '2', '3', '4']]}, [{'seq': [{'int': 1}, {'int': 0}]}]],
    ['getitem', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['ud', ['-1', '-2', '-3', '-4']]}, [{'seq': [{'int': 0}, {'int': 0}]}]],
    ['lower_index', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['uu', ['(y**2 - 2*z**2 - 2)/(x**2*y**2*z**2 + 2*x**2*y**2 - 2*x**2*z**2 - 2*x**2 + y**2 - 2*z**2 - 2)', '(x*y*z**2 + x*y)/(x**2*y**2*z**2 + 2*x**2*y**2 - 2*x**2*z**2 - 2*x**2 + y**2 - 2*z**2 - 2)', '-x*y**2/(x**2*y**2*z**2 + 2*x**2*y**2 - 2*x**2*z**2 - 2*x**2 + y**2 - 2*z**2 - 2)', '(x*y*z**2 + x*y)/(x**2*y**2*z**2 + 2*x**2*y**2 - 2*x**2*z**2 - 2*x**2 + y**2 - 2*z**2 - 2)', '(-x**2*z**2 - x**2 - z**2 - 1)/(x**2*y**2*z**2 + 2*x**2*y**2 - 2*x**2*z**2 - 2*x**2 + y**2 - 2*z**2 - 2)', '(x**2*y + y)/(x**2*y**2*z**2 + 2*x**2*y**2 - 2*x**2*z**2 - 2*x**2 + y**2 - 2*z**2 - 2)', '-x*y**2/(x**2*y**2*z**2 + 2*x**2*y**2 - 2*x**2*z**2 - 2*x**2 + y**2 - 2*z**2 - 2)', '(x**2*y + y)/(x**2*y**2*z**2 + 2*x**2*y**2 - 2*x**2*z**2 - 2*x**2 + y**2 - 2*z**2 - 2)', '(x**2*y**2 - 2*x**2 - 2)/(x**2*y**2*z**2 + 2*x**2*y**2 - 2*x**2*z**2 - 2*x**2 + y**2 - 2*z**2 - 2)']]}, [{'int': 0}]],
    ['lower_index', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], {'tensor': ['uu', ['-r/(-2*m + r)', '0', '0', '0', '0', '(-2*m + r)/r', '0', '0', '0', '0', 'r**(-2)', '0', '0', '0', '0', '1/(r**2*sin(theta)**2)']]}, [{'int': 0}]],
    ['lower_index', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['u', ['1', '0']]}, [{'int': 0}]],
    ['lower_index', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['u', ['0', '1']]}, [{'int': 0}]],
    ['lower_index', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['ud', ['x', 'y*z', '1', 'x*y', '0', 'z', 'y', 'x + z', 'z**2']]}, [{'int': 0}]],
    ['lower_index', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['du', ['1', '0', '0', '0', '1', '0', '0', '0', '1']]}, [{'int': 1}]],
    ['metric_tensor', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], None, []],
    ['metric_tensor', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], None, []],
    ['mul', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['ud', ['phi', 'theta - sin(theta)**2*cos(theta)', '2*cos(theta)', 'phi*theta*cos(theta)/sin(theta)']]}, [{'tensor': ['d', ['theta**2', 'phi']]}]],
    ['mul', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['u', ['phi*theta', 'sin(theta)']]}, [{'tensor': ['d', ['theta**2', 'phi']]}]],
    ['mul', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['u', ['phi*theta', 'sin(theta)']]}, [{'tensor': ['dd', ['2*theta', '-phi*cos(theta)/sin(theta)', '-phi*cos(theta)/sin(theta)', 'theta**2*sin(theta)*cos(theta) + 1']]}]],
    ['neg', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['dd', ['0', 'phi/2 - 1/2', '1/2 - phi/2', '0']]}, []],
    ['neg', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['ud', ['1', '2', '3', '4']]}, []],
    ['raise_index', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['dd', ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']]}, [{'int': 0}]],
    ['raise_index', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['dd', ['x**3 + x**2*y**2 + x', 'x**2*y*z + y*z', 'x**2 + x*y*z + 1', 'x**2*y + 2*x*y + y**2', 'x*y**2*z + x*y + y*z', 'x*y + y*z**2 + 2*z', 'x*y**2 + y*z**2 + y', 'x*z**2 + x + z**3 + z', 'y*z + z**4 + z**2']]}, [{'int': 0}]],
    ['raise_index', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], {'tensor': ['dd', ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']]}, [{'int': 0}]],
    ['ricci_tensor', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], None, []],
    ['ricci_tensor', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], None, []],
    ['riemann_tensor', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], None, []],
    ['riemann_tensor', [[2, 2, ['x**2 + 1', 'x*y', 'x*y', 'y**4 + 2']], ['x', 'y']], None, []],
    ['riemann_tensor', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], None, []],
    ['scalar', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['ud', ['1', '2', '3', '4']]}, []],
    ['scalar', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], {'tensor': ['', ['48*m**2/r**6']]}, []],
    ['scalar', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['', ['phi*theta**3 + phi*sin(theta)']]}, []],
    ['scale', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['dd', ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']]}, ['(-4*x**4*z - 8*x**2*z + 4*y**2*z + 4*z**4 + 8*z**2 - 4*z + 4)/(2*(x**4*y**4*z**4 + 4*x**4*y**4*z**2 + 4*x**4*y**4 - 4*x**4*y**2*z**4 - 12*x**4*y**2*z**2 - 8*x**4*y**2 + 4*x**4*z**4 + 8*x**4*z**2 + 4*x**4 + 2*x**2*y**4*z**2 + 4*x**2*y**4 - 4*x**2*y**2*z**4 - 16*x**2*y**2*z**2 - 12*x**2*y**2 + 8*x**2*z**4 + 16*x**2*z**2 + 8*x**2 + y**4 - 4*y**2*z**2 - 4*y**2 + 4*z**4 + 8*z**2 + 4))']],
    ['scale', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['ud', ['1', '2', '3', '4']]}, [{'int': 2}]],
    ['sub', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['udd', ['2*phi*theta**2', '-phi**2*theta*cos(theta)/sin(theta)', '-phi**2*theta*cos(theta)/sin(theta)', 'phi*theta**3*sin(theta)*cos(theta) + phi*theta', '2*theta*sin(theta)', '-phi*cos(theta)', '-phi*cos(theta)', 'theta**2*sin(theta)**2*cos(theta) + sin(theta)']]}, [{'tensor': ['udd', ['2*phi*theta**2', '-phi**2*theta*cos(theta)/sin(theta)', '-phi**2*theta*cos(theta)/sin(theta)', 'phi*theta**3*sin(theta)*cos(theta) + phi*theta', '2*theta*sin(theta)', '-phi*cos(theta)', '-phi*cos(theta)', 'theta**2*sin(theta)**2*cos(theta) + sin(theta)']]}]],
    ['sub', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], {'tensor': ['dd', ['(2*m - r)/r', '0', '0', '0', '0', '-r/(2*m - r)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']]}, [{'tensor': ['dd', ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']]}]],
    ['sub', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['udd', ['3*phi*theta**2', '(-phi**2*theta*cos(theta) + theta**3*sin(theta) - theta**2*sin(theta)**3*cos(theta))/sin(theta)', '(-phi**2*theta*cos(theta) + phi**2*sin(theta))/sin(theta)', 'phi*theta**3*sin(theta)*cos(theta) + 2*phi*theta - phi*sin(theta)**2*cos(theta)', '2*theta**2*cos(theta) + 2*theta*sin(theta)', '(phi*theta**3*cos(theta) - phi*sin(theta)*cos(theta))/sin(theta)', 'phi*cos(theta)', '(phi**2*theta*cos(theta) + theta**2*sin(theta)**3*cos(theta) + sin(theta)**2)/sin(theta)']]}, [{'tensor': ['udd', ['phi*theta**2', 'theta**3 - theta**2*sin(theta)**2*cos(theta)', 'phi**2', 'phi*theta - phi*sin(theta)**2*cos(theta)', '2*theta**2*cos(theta)', 'phi*theta**3*cos(theta)/sin(theta)', '2*phi*cos(theta)', 'phi**2*theta*cos(theta)/sin(theta)']]}]],
    ['sub', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['dd', ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']]}, [{'tensor': ['dd', ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']]}]],
    ['sub', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['d', ['3*phi*theta**2 + phi*cos(theta)', 'theta**3 + sin(theta)']]}, [{'tensor': ['d', ['3*phi*theta**2 + phi*cos(theta)', 'theta**3 + sin(theta)']]}]],
    ['symmetrize', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['dd', ['theta', 'phi', '1', 'phi*theta']]}, [{'int': 0}, {'int': 1}]],
    ['symmetrize', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['ud', ['1', '2', '3', '4']]}, [{'int': 0}, {'int': 1}]],
    ['symmetrize', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['dd', ['0', '-sin(theta)*cos(theta)', 'sin(theta)*cos(theta)', '0']]}, [{'int': 0}, {'int': 1}]],
    ['symmetrize', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['dd', ['0', '0', '0', 'sin(theta)*cos(theta)']]}, [{'int': 0}, {'int': 1}]],
    ['trace', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], {'tensor': ['dd', ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']]}, []],
    ['trace', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['u', ['1', '2']]}, []],
    ['trace', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['dd', ['x**3 + x**2*y**2 + x', 'x**2*y*z + y*z', 'x**2 + x*y*z + 1', 'x**2*y + 2*x*y + y**2', 'x*y**2*z + x*y + y*z', 'x*y + y*z**2 + 2*z', 'x*y**2 + y*z**2 + y', 'x*z**2 + x + z**3 + z', 'y*z + z**4 + z**2']]}, []],
    ['trace', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['dd', ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']]}, []],
    ['trace', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['ud', ['x', 'y*z', '1', 'x*y', '0', 'z', 'y', 'x + z', 'z**2']]}, []],
    ['transpose', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['du', ['x', 'x*y', 'y', 'y*z', '0', 'x + z', '1', 'z', 'z**2']]}, [{'int': 0}, {'int': 1}]],
    ['transpose', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['udd', ['phi*theta**2', 'phi**2', 'theta**3 - theta**2*sin(theta)**2*cos(theta)', 'phi*theta - phi*sin(theta)**2*cos(theta)', '2*theta**2*cos(theta)', '2*phi*cos(theta)', 'phi*theta**3*cos(theta)/sin(theta)', 'phi**2*theta*cos(theta)/sin(theta)']]}, [{'int': 1}, {'int': 2}]],
    ['transpose', [[2, 2, ['1', '0', '0', 'sin(theta)**2']], ['theta', 'phi']], {'tensor': ['dd', ['0', 'phi/2 - 1/2', '1/2 - phi/2', '0']]}, [{'int': 0}, {'int': 1}]],
    ['transpose', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['ud', ['x', 'y*z', '1', 'x*y', '0', 'z', 'y', 'x + z', 'z**2']]}, [{'int': 0}, {'int': 1}]],
    ['with_positions', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['dd', ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']]}, ['uu']],
    ['with_positions', [[2, 2, ['1', '0', '0', '1']], ['x', 'y']], {'tensor': ['ud', ['1', '2', '3', '4']]}, ['dd']],
    ['with_positions', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['ud', ['x', 'y*z', '1', 'x*y', '0', 'z', 'y', 'x + z', 'z**2']]}, ['du']],
    ['with_positions', [[4, 4, ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']], ['t', 'r', 'theta', 'phi']], {'tensor': ['dd', ['2*m/r - 1', '0', '0', '0', '0', '1/(-2*m/r + 1)', '0', '0', '0', '0', 'r**2', '0', '0', '0', '0', 'r**2*sin(theta)**2']]}, ['uu']],
    ['with_positions', [[3, 3, ['x**2 + 1', 'x*y', '0', 'x*y', '2', 'y', '0', 'y', 'z**2 + 1']], ['x', 'y', 'z']], {'tensor': ['ud', ['x', 'y*z', '1', 'x*y', '0', 'z', 'y', 'x + z', 'z**2']]}, ['dd']],
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_tensor(i: int) -> None:
    name, raw_metric, raw_tensor, raw_args = CASES[i]
    assert isinstance(name, str) and isinstance(raw_metric, list) and isinstance(raw_args, list)
    (rows, cols, entries), coordinates = raw_metric
    assert isinstance(entries, list) and isinstance(coordinates, list) and isinstance(rows, int) and isinstance(cols, int)
    g = ExprMatrix(rows, cols, [parse(str(e)) for e in entries])
    xs = [parse(str(c)) for c in coordinates]
    metric = Metric(g, [x for x in xs if isinstance(x, Symbol)])
    head: list[Basic] = [Symbol('tensor'), Symbol(name), Tuple(Integer(rows), Integer(cols), Tuple(*g.entries)), Tuple(*xs)]
    if raw_tensor is None:
        got = outcome(lambda: rc.logic_call('calculus', head))
        expected = outcome(lambda: encoded(getattr(TN, name)(metric)))
    else:
        tensor = argument(raw_tensor, metric)
        assert isinstance(tensor, TN.Tensor)
        args = [argument(a, metric) for a in raw_args]
        extra = [encoded(a) for a in args]
        got = outcome(lambda: rc.logic_call('calculus', head + [Tuple(*tensor.components), Symbol(tensor.positions)] + extra))
        expected = outcome(lambda: encoded(getattr(tensor, METHODS.get(name, name))(*args)))
    if got != 'SKIP':
        assert got == expected, (name, raw_args)

"""The Rust port of series/formal.py against the Python one: the formal
power series of the Python tests, their truncations, the operations on
them and the algorithms of the coefficients (the cases read by ``eval`` in
the core's namespace, the integers exact)."""
from __future__ import annotations

import re
from typing import Any, Callable

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.core import Derivative, Function, Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.series import formal as fm

from ..conftest import load

rc = load("sympy_extras_rs.core")


class _Namespace(dict):  # type: ignore[type-arg]
    def __missing__(self, name: str) -> Symbol:
        return Symbol(name)


_NS = _Namespace()
for _mod in (core, functions):
    for _k in dir(_mod):
        if not _k.startswith('_'):
            _NS[_k] = getattr(_mod, _k)
_NS.update(Integer=Integer, Derivative=Derivative, f=Function('f'), r=Function('r'))


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return str(r)


x, k = Symbol('x'), Symbol('k')

#: (operation, function, options): the series of the function by fps(f, x,
#: x0, dir, hyper, rational, full), then the operation
FPS: list[tuple[str, str, dict[str, Any]]] = [
    ('truncate', 'x**2 + x + 1', {}),
    ('truncate', 'log(1 + x)', {}),
    ('truncate', 'log(1 + x)', {'hyper': False, 'rational': False}),
    ('truncate', '1/x + 1/x**2', {}),
    ('truncate', '1/x', {}),
    ('truncate', '(x**2 + x + 1)/x**3', {'dir': -1}),
    ('truncate', '1/((x - 1)**2*(x - 2))', {}),
    ('truncate', '(1 + x + x**2 + x**3)/((x - 1)*(x - 2))', {}),
    ('truncate', 'x/(1 - x - x**2)', {'full': True}),
    ('truncate', '1/(x**2 + 2*x + 2)', {'full': True}),
    ('truncate', '1/(x**2 + x + 1)', {}),
    ('truncate', '1/(x**2 + x + 1)', {'full': True}),
    ('truncate', 'log(1 + x)', {'x0': 2}),
    ('truncate', 'log(1 + x)', {'x0': 2, 'dir': -1}),
    ('truncate', 'atan(x)', {'full': True}),
    ('truncate', 'atan(x)', {'x0': 2, 'full': True}),
    ('truncate', 'x*atan(x) - log(1 + x**2)/2', {'full': True}),
    ('truncate', 'sin(x)', {}),
    ('truncate', 'cos(x)', {}),
    ('truncate', 'exp(x)', {}),
    ('truncate', 'atan(x)', {}),
    ('truncate', 'exp(acos(x))', {}),
    ('truncate', 'exp(acosh(x))', {}),
    ('truncate', 'atan(1/x)', {}),
    ('truncate', 'x*atan(x) - log(1 + x**2)/2', {'rational': False}),
    ('truncate', 'log(1 + x)', {'rational': False}),
    ('truncate', 'exp(x)*sin(x)', {}),
    ('truncate', 'exp(x)*sin(x)/x', {}),
    ('truncate', 'sin(x)*cos(x)', {}),
    ('truncate', 'x**(-5)*sin(x)', {}),
    ('truncate', 'x**2*atan(x)', {'rational': False}),
    ('truncate', 'cos(sqrt(x))*x', {}),
    ('truncate', 'x**2*cos(sqrt(x))', {}),
    ('truncate', 'x*atan(x) - log(1 + x**2)/2', {}),
    ('truncate', 'exp(x)', {'x0': 'oo'}),
    ('truncate', '1/(1 - x)', {'x0': 'oo'}),
    ('truncate', 'asech(x)', {}),
    ('truncate', 'y*exp(x)', {'x': 'x'}),
    ('truncate', 'x**n*sin(x**2)', {'x': 'x'}),
    ('polynomial', 'cos(x)', {}),
    ('polynomial', 'exp(x)', {}),
    ('derivative', 'sin(x)', {}),
    ('derivative', 'exp(x)', {}),
    ('derivative', 'log(1 + x)', {}),
    ('integrate', 'sin(x)', {}),
    ('integrate', 'exp(x)', {}),
    ('integrate', 'log(1 + x)', {}),
    ('integrate01', 'sin(x)', {}),
    ('neg', 'sin(x)', {}),
    ('scale', 'exp(x)', {}),
    ('plus_const', 'exp(x)', {}),
    ('add', 'sin(x)|cos(x)', {}),
    ('add', 'sin(x)|-sin(x)', {}),
    ('add', 'exp(x)|log(1 + x)', {}),
    ('sub', 'exp(x)|sin(x)', {}),
    ('product', 'sin(x)|exp(x)', {}),
    ('product', 'cos(x)|log(1 + x)', {}),
    ('compose', 'exp(x)|sin(x)', {}),
    ('compose', 'sin(x)|log(1 + x)', {}),
    ('compose', 'exp(x)|cos(x)', {}),
    ('inverse', 'cos(x)', {}),
    ('inverse', 'exp(x)', {}),
    ('inverse', 'sin(x)', {}),
    ('terms', 'exp(x)', {}),
    ('terms', 'log(1 + x)', {}),
    ('infinite', 'exp(x)', {}),
]

#: (algorithm, arguments)
ALGORITHMS: list[tuple[str, list[str]]] = [
    ('rational_algorithm', ['1/(1 - x)', 'False']),
    ('rational_algorithm', ['log(1 + x)', 'False']),
    ('rational_algorithm', ['(x + 1)/(x**2 - 3*x + 2)', 'False']),
    ('rational_algorithm', ['atan(x)', 'False']),
    ('rational_algorithm', ['atan(x)', 'True']),
    ('rational_algorithm', ['cos(x)', 'False']),
    ('rational_algorithm', ['1/(x**2 + x + 1)', 'False']),
    ('rational_algorithm', ['z/((x - y)*(x - m))', 'False']),
    ('hyper_algorithm', ['exp(x)']),
    ('hyper_algorithm', ['sin(x)']),
    ('hyper_algorithm', ['cos(x)']),
    ('hyper_algorithm', ['atan(x)']),
    ('hyper_algorithm', ['exp(x)*sin(x)']),
    ('simpleDE', ['exp(x)']),
    ('simpleDE', ['sin(x)']),
    ('simpleDE', ['exp(x)*sin(x)']),
    ('simpleDE', ['atan(x)']),
    ('exp_re', ['-f(x) + Derivative(f(x), x)']),
    ('exp_re', ['f(x) + Derivative(f(x), (x, 2))']),
    ('exp_re', ['Derivative(f(x), (x, 3)) + Derivative(f(x), (x, 4)) + Derivative(f(x), x)']),
    ('hyper_re', ['f(x) + Derivative(f(x), (x, 2))']),
    ('hyper_re', ['-x*f(x) + Derivative(f(x), (x, 2))']),
    ('hyper_re', ['2*f(x) - 2*Derivative(f(x), x) + Derivative(f(x), (x, 2))']),
    ('hyper_re', ['2*n*f(x) + (x**2 - 1)*Derivative(f(x), x)']),
    ('hyper_re', ['(x**2 - 1)*Derivative(f(x), (x, 3)) + 3*x*Derivative(f(x), (x, 2)) + Derivative(f(x), x)']),
    ('coeff_of', ['x**2 + 3*x + 1', '1']),
    ('coeff_of', ['x**2 + 3*x + 1', 'k']),
    ('coeff_of', ['x**2 + 3*x + 1', '0']),
    ('coeff_of', ['x**2 + 3*x + 1', '-1']),
    ('bell_incomplete', ['4', '2', 'x1, x2, x3']),
    ('bell_incomplete', ['5', '3', 'x1, x2, x3']),
]


def _fps_args(f: str, o: dict[str, Any]) -> tuple[Any, Any, Any, int, bool, bool, bool]:
    xs = _read(o['x']) if 'x' in o else None
    x0 = _read(str(o.get('x0', 0)))
    return (_read(f), xs, x0, int(o.get('dir', 1)), bool(o.get('hyper', True)), bool(o.get('rational', True)),
            bool(o.get('full', False)))


def python_fps(op: str, f: str, o: dict[str, Any]) -> object:
    if op in ('add', 'sub', 'product', 'compose'):
        a, b = f.split('|')
        s, t = fm.fps(_read(a)), fm.fps(_read(b))
        assert isinstance(s, fm.FormalPowerSeries) and isinstance(t, fm.FormalPowerSeries)
        if op == 'add' or op == 'sub':
            r = s + t if op == 'add' else s - t
            return r.truncate() if isinstance(r, fm.FormalPowerSeries) else Tuple(Symbol('expr'), r)
        return (s.product(t, x) if op == 'product' else s.compose(t, x)).truncate()
    e, xs, x0, d, hyper, rational, full = _fps_args(f, o)
    s = fm.fps(e, xs, x0, d, hyper=hyper, rational=rational, full=full)
    if not isinstance(s, fm.FormalPowerSeries):
        return Tuple(Symbol('expr'), s)
    if op == 'truncate':
        return s.truncate(8) if Symbol('n') in e.free_symbols else s.truncate()
    if op == 'polynomial':
        return s.polynomial(5)
    if op == 'derivative':
        return s.diff(s.x).truncate()
    if op == 'integrate':
        return s.integrate(s.x).truncate()
    if op == 'integrate01':
        return s.integrate((s.x, 0, 1))
    if op == 'neg':
        return (-s).truncate()
    if op == 'scale':
        return (s * Symbol('c')).truncate()
    if op == 'plus_const':
        return (s + Symbol('c')).truncate()
    if op == 'inverse':
        return s.inverse(x).truncate(8)
    if op == 'terms':
        return Tuple(*s[:6])
    return s.infinite


def rust_fps(op: str, f: str, o: dict[str, Any]) -> object:
    if op in ('add', 'sub', 'product', 'compose'):
        a, b = f.split('|')
        return rc.logic_call('calculus', [Symbol('formal'), Symbol(op), _read(a), _read(b)])
    e, xs, x0, d, hyper, rational, full = _fps_args(f, o)
    flags = [Symbol(str(v)) for v in (hyper, rational, full)]
    return rc.logic_call('calculus', [Symbol('formal'), Symbol(op), e, Symbol('NONE') if xs is None else xs, x0,
                                      Integer(d), *flags])


def python_algorithm(name: str, raw: list[str]) -> object:
    if name == 'rational_algorithm':
        found = fm.rational_algorithm(_read(raw[0]), x, k, full=raw[1] == 'True')
        return Symbol('NONE') if found is None else Tuple(*found)
    if name == 'hyper_algorithm':
        found = fm.hyper_algorithm(_read(raw[0]), x, k)
        return Symbol('NONE') if found is None else Tuple(*found)
    if name == 'simpleDE':
        return Tuple(*[Tuple(de, Integer(i)) for de, i in fm.simpleDE(_read(raw[0]), x, Function('g'))])
    if name == 'exp_re':
        return fm.exp_re(_read(raw[0]), Function('r'), k)
    if name == 'hyper_re':
        return fm.hyper_re(_read(raw[0]), Function('r'), k)
    if name == 'coeff_of':
        return fm.coeff_of(_read(raw[0]), x, _read(raw[1]))
    return fm.bell_incomplete(int(raw[0]), int(raw[1]), list(_read('(%s,)' % raw[2])))


def rust_algorithm(name: str, raw: list[str]) -> object:
    if name == 'bell_incomplete':
        args: list[Basic] = [Integer(int(raw[0])), Integer(int(raw[1])), Tuple(*_read('(%s,)' % raw[2]))]
    else:
        args = [Symbol(a) if a in ('True', 'False') else _read(a) for a in raw]
    return rc.logic_call('calculus', [Symbol('formal'), Symbol(name), *args])


@pytest.mark.parametrize('i', range(len(FPS)))
def test_fps(i: int) -> None:
    op, f, o = FPS[i]
    expected = outcome(lambda: python_fps(op, f, o))
    got = outcome(lambda: rust_fps(op, f, o))
    if got == 'SKIP':
        pytest.skip('not ported')
    assert got == expected, (op, f, o)


@pytest.mark.parametrize('i', range(len(ALGORITHMS)))
def test_algorithms(i: int) -> None:
    name, raw = ALGORITHMS[i]
    expected = outcome(lambda: python_algorithm(name, raw))
    got = outcome(lambda: rust_algorithm(name, raw))
    if got == 'SKIP':
        pytest.skip('not ported')
    assert got == expected, (name, raw)

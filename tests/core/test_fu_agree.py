"""``sympy_extras.simplify.fu`` against its Rust port: every TR rule on
random trigonometric expressions, ``fu`` on the identities of its tests,
``trig_split``, ``as_f_sign_1`` and ``hyper_as_trig``; the outputs
compared as strings."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load

RULES = ['TR0', 'TR1', 'TR2', 'TR2i', 'TR3', 'TR4', 'TR5', 'TR6', 'TR7', 'TR8', 'TR9', 'TR10', 'TR10i', 'TR11', '_TR11',
         'TR12', 'TR12i', 'TR13', 'TRmorrie', 'TR14', 'TR15', 'TR16', 'TR111', 'TR22', 'TRpower', 'L', 'sincos_to_sum']


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('fu', [Symbol(op), *args])


def _outcome(f: Any) -> Any:
    try:
        return ('ok', str(f()))
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _python(name: str, e: Any) -> Any:
    import sympy_extras.simplify.fu as F
    extra = {'TR2i_half': ('TR2i', {'half': True}), 'TR6_pow': ('TR6', {'max': 6, 'pow': True})}
    fname, kwargs = extra.get(name, (name, {}))
    return getattr(F, fname)(e, **kwargs)


def _cases(seed: int) -> list[Any]:
    from sympy_extras.core import Integer, Symbol, pi
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.functions import cos, cot, csc, sec, sin, tan
    x, y = Symbol('x'), Symbol('y')
    rng = random.Random(seed)
    args = [x, y, 2*x, x + y, x - y, -x, pi/3 - x, pi/2 + x, x/2]
    funcs = [sin, cos, tan, cot, sec, csc]

    def atom() -> Any:
        return pow(rng.choice(funcs)(rng.choice(args)), Integer(rng.choice([1, 1, 2, 3, 4, -1, 6])))

    out = []
    for _ in range(30):
        e = add(*[mul(Integer(rng.randint(-2, 2)), *[atom() for _ in range(rng.randint(1, 2))]) for _ in range(rng.randint(1, 3))])
        if rng.random() < 0.3:
            e = e/add(atom(), Integer(rng.randint(1, 2)))
        out.append(e)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rules(rust: Any, seed: int) -> None:
    different = []
    for e in _cases(seed):
        for name in RULES + ['TR2i_half', 'TR6_pow']:
            got = _outcome(lambda: _call(name, e))
            expected = _outcome(lambda: _python(name, e))
            if got != expected:
                different.append((name, e, got, expected))
    assert not different, different[:3]


def _identities() -> list[Any]:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.functions import cos, cot, sec, sin, sqrt, tan, sinh, cosh, tanh
    x, y = Symbol('x'), Symbol('y')
    return [sin(x)**2 + cos(x)**2, sin(x)/cos(x), 1 - 2*sin(x)**2, sin(x)*cos(x), cos(x)**4 - sin(x)**4, tan(x)*cot(x),
            sin(x + y) - sin(x)*cos(y), (1 - cos(x))*(1 + cos(x)), sec(x)**2 - tan(x)**2, cos(x)*cos(2*x)*cos(4*x),
            2*sin(x)*cos(x)/(cos(x)**2 - sin(x)**2), sin(3*x)/sin(x), cos(x) - cos(y), sqrt(Integer(3))*cos(x)/2 + sin(x)/2,
            (tan(x) + tan(y))/(1 - tan(x)*tan(y)), sinh(x)**2 + cosh(x)**2, tanh(x)*cosh(x), sin(x)*sin(y) + cos(x)*cos(y)]


def test_fu(rust: Any) -> None:
    import sympy_extras.simplify.fu as F
    different = []
    for e in _identities():
        got, expected = _outcome(lambda: _call('fu', e)), _outcome(lambda: F.fu(e))
        if got != expected:
            different.append((e, got, expected))
    assert not different, different[:3]


def test_helpers(rust: Any) -> None:
    import sympy_extras.simplify.fu as F
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.logic import false, true
    from sympy_extras.functions import cos, cosh, sin, sinh, sqrt, tanh
    x, y = Symbol('x'), Symbol('y')
    pairs = [(cos(x), cos(y)), (cos(x), -cos(y)), (sin(x), sin(y)), (cos(x)*sin(y), sin(x)*cos(y)), (2*cos(x), 2*sin(y)),
             (sqrt(Integer(2))*cos(x), sqrt(Integer(2))*sin(x)), (sqrt(Integer(3))*cos(x), sin(x)), (cos(x), sin(x)),
             (-cos(x)*cos(y), sin(x)*sin(y))]
    for a, b in pairs:
        for two in (False, True):
            got = _outcome(lambda: _call('trig_split', a, b, true if two else false))
            got = ('ok', '()') if got == ('ok', '()') else got
            r = F.trig_split(a, b, two=two)
            expected = ('ok', '()' if r is None else str((r[0], r[1], r[2], r[3], r[4], r[5])).replace('True', 'True'))
            if got[0] == 'ok' and got[1] != '()':
                t = _call('trig_split', a, b, true if two else false).args
                got = ('ok', str((t[0], int(t[1]), int(t[2]), t[3], t[4], bool(t[5]))))
            assert got == expected, (a, b, two)
    for e in [x - 1, 1 - x, 2*x - 2, cos(x) + 1, -sin(x) - 1, 2*cos(x)*y - 2*y, x + y]:
        r = F.as_f_sign_1(e)
        got = _call('as_f_sign_1', e)
        assert (str(tuple(got.args)) if got.args else 'None') == (str(r) if r is not None else 'None'), e
    for e in [sinh(x)**2 + cosh(x)**2, tanh(x) + sin(y)*cosh(x), sinh(x + y)]:
        expr, back = _call('hyper_as_trig', e).args
        pe, inverse = F.hyper_as_trig(e)
        assert str(back) == str(inverse(pe)), e


def _targeted() -> dict[str, list[Any]]:
    from sympy_extras.core import Integer, Symbol, pi
    from sympy_extras.functions import cos, cot, sin, sqrt, tan
    from sympy_extras.functions.elementary.trigonometric import Cos, Sin, Tan
    x, y = Symbol('x'), Symbol('y')
    p = Symbol('p', positive=True)
    return {
        'TR2i': [sin(x)/cos(x), sin(x)**2/cos(x)**2, cos(y)/sin(y), sin(x)*cos(y)/(cos(x)*sin(y)), sin(p)**y/cos(p)**y],
        'TR2i_half': [sin(x)/(1 + cos(x)), (1 + cos(x))/sin(x), sin(x)**2/(cos(x) + 1)**2],
        'TR3': [cos(-x), sin(-x + y), cos(pi/2 - x), sin(3*pi/7), tan(-y), cos(pi/3 + x)],
        'TR4': [Cos(pi/3), Sin(pi/4) + Cos(pi/6), Tan(pi/3)*x, Sin(2*pi/3), Cos(pi), Cos(pi/5)],
        'TR9': [cos(x) + cos(y), cos(x) - cos(y), sin(x) + sin(y), sin(x) - sin(y), 2*cos(x) + 2*cos(3*x) + y,
                cos(x) + cos(y) + sin(x) + sin(y)],
        'TR10i': [cos(x)*cos(y) - sin(x)*sin(y), sin(x)*cos(y) + cos(x)*sin(y), cos(x)*cos(y) + sin(x)*sin(y),
                  sqrt(Integer(2))*cos(x)/2 + sqrt(Integer(2))*sin(x)/2, sqrt(Integer(3))*cos(x)/2 + sin(x)/2,
                  cos(x)*sin(y) - sin(x)*cos(y) + 2*y],
        '_TR11': [sin(x)/cos(x/2), sin(2*x)/sin(x), cos(x/2)/sin(x)],
        'TR12i': [(tan(x) + tan(y))/(1 - tan(x)*tan(y)), -(tan(x) + tan(y))/(tan(x)*tan(y) - 1),
                  (tan(x) + tan(y))**2/(1 - tan(x)*tan(y))**2],
        'TR14': [(1 - cos(x))*(1 + cos(x)), (1 - sin(x))*(1 + sin(x)), (cos(x) - 1)*(cos(x) + 1)*y,
                 (1 - cos(x))**2*(1 + cos(x))**3, x/((1 - cos(x))*(1 + cos(x)))],
        'TR15': [sin(x)**-2, sin(x)**-4, sin(x)**-3, 1 - sin(x)**-2],
        'TR16': [cos(x)**-2, cos(x)**-4, cos(x)**-3, 1 - cos(x)**-2],
        'TR13': [tan(x)*tan(y), cot(x)*cot(y), tan(x)*tan(y)*tan(2*x)],
        'TRmorrie': [cos(x)*cos(2*x), cos(x)*cos(2*x)*cos(4*x), cos(3*x)*cos(6*x)*y, cos(x)**2*cos(2*x)**2],
    }


def test_targeted(rust: Any) -> None:
    different, unchanged = [], []
    for name, cases in _targeted().items():
        changed = 0
        for e in cases:
            got, expected = _outcome(lambda: _call(name, e)), _outcome(lambda: _python(name, e))
            changed += expected[0] == 'ok' and expected[1] != str(e)
            if got != expected:
                different.append((name, e, got, expected))
        if not changed:
            unchanged.append(name)
    assert not different, different[:3]
    assert not unchanged, unchanged

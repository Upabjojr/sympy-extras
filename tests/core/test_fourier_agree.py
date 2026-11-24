"""The Rust port of series/fourier.py against the Python one: the Fourier
series of its tests, with the operations on them, compared by their
truncations, terms and coefficients (read with the core's parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Symbol, Tuple, oo, pi
from sympy_extras.core.relational import Lt
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
from sympy_extras.functions import piecewise
from sympy_extras.series.fourier import FiniteFourierSeries, FourierSeries, fourier_series

from ..conftest import load

rc = load("sympy_extras_rs.core")

#: a series: ('series', f, [limits], finite), (op, s, c) for op in shift,
#: shiftx, scale, scalex; ('neg', s); ('add'|'sub', s, t)
Spec = tuple[object, ...]
Series = object


def ex(v: object) -> Basic:
    """An expression of a case: a node as it is, a string parsed."""
    return v if isinstance(v, Basic) else parse(str(v))


def python_series(spec: Spec) -> Series:
    kind = spec[0]
    if kind == 'series':
        lims = spec[2]
        assert isinstance(lims, list)
        limits = [parse(str(v)) for v in lims] if lims else None
        return fourier_series(ex(spec[1]), limits, finite=bool(spec[3]))
    s = python_series(spec[1])  # type: ignore[arg-type]
    if kind == 'neg':
        assert isinstance(s, (FourierSeries, FiniteFourierSeries))
        return -s
    if kind in ('add', 'sub'):
        t = python_series(spec[2])  # type: ignore[arg-type]
        assert isinstance(s, (FourierSeries, FiniteFourierSeries))
        return s + t if kind == 'add' else s - t  # type: ignore[operator]
    assert isinstance(s, (FourierSeries, FiniteFourierSeries)) and isinstance(kind, str)
    method: Callable[[object], Series] = getattr(s, kind)
    return method(parse(str(spec[2])))


def encoded(spec: Spec) -> Basic:
    kind = spec[0]
    assert isinstance(kind, str)
    head = Symbol(kind)
    if kind == 'series':
        lims = spec[2]
        assert isinstance(lims, list)
        return Tuple(head, ex(spec[1]), Tuple(*[parse(str(v)) for v in lims]), Symbol(str(bool(spec[3]))))
    inner = spec[1]
    assert isinstance(inner, tuple)
    if kind == 'neg':
        return Tuple(head, encoded(inner))
    if kind in ('add', 'sub'):
        other = spec[2]
        assert isinstance(other, tuple)
        return Tuple(head, encoded(inner), encoded(other))
    return Tuple(head, encoded(inner), parse(str(spec[2])))


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return str(r)


Query = tuple[object, ...]


def python(spec: Spec, query: Query) -> object:
    s = python_series(spec)
    if not isinstance(s, (FourierSeries, FiniteFourierSeries)):
        assert isinstance(s, Basic)
        return Tuple(Symbol('expr'), s)
    assert isinstance(s, (FourierSeries, FiniteFourierSeries))
    name = query[0]
    if name == 'kind':
        return Symbol('series' if isinstance(s, FourierSeries) else 'finite')
    if name == 'truncate':
        k = query[1]
        if k is None:
            return s.truncate(oo)
        assert isinstance(k, int)
        return s.truncate(k)
    if name == 'term':
        return s.term(parse(str(query[1])))
    if name == 'leading':
        x = s.x
        return s._eval_as_leading_term(x, None, parse('1'))
    if name == 'sigma':
        k = query[1]
        assert isinstance(k, int)
        return s.sigma_approximation(k)
    if name == 'a0':
        return s.a0
    if name == 'length':
        return s.length
    assert isinstance(s, FiniteFourierSeries)
    return Tuple(Tuple(*[Tuple(k, v) for k, v in s.an.items()]), Tuple(*[Tuple(k, v) for k, v in s.bn.items()]))


def rust(spec: Spec, query: Query) -> object:
    name = query[0]
    assert isinstance(name, str)
    args: list[Basic] = [Symbol('fourier_series'), encoded(spec), Symbol(name)]
    if name in ('truncate', 'sigma'):
        k = query[1]
        args.append(Symbol('NONE') if k is None else parse(str(k)))
    elif name == 'term':
        args.append(parse(str(query[1])))
    return rc.logic_call('calculus', args)


def S(f: object, *lims: str, finite: bool = True) -> Spec:
    return ('series', f, list(lims), finite)


FO = S('x', 'x', '-pi', 'pi')
FE = S('x**2', '-pi', 'pi')
FP = S(piecewise((0, Lt(Symbol('x'), 0)), (pi, True)), 'x', '-pi', 'pi')
SIN6 = S('sin(x)**6')

CASES: list[tuple[Spec, Query]] = [
    (FO, ('truncate', 3)),
    (FO, ('truncate', 4)),
    (FE, ('truncate', 3)),
    (FP, ('truncate', 3)),
    (FP, ('term', '3')),
    (FO, ('term', '0')),
    (FE, ('a0',)),
    (FO, ('kind',)),
    (FO, ('length',)),
    (FO, ('leading',)),
    (S('1', '-pi', 'pi'), ('truncate', 3)),
    (S('x', '0', '1', '2'), ('truncate', 3)),
    (S('x', 'x', '0', 'oo'), ('truncate', 3)),
    (S('x*y', '0', 'oo'), ('truncate', 3)),
    (S('x*y', '0', '1'), ('truncate', 3)),
    (S(piecewise((0, Lt(Symbol('x'), 0)), (parse('x'), True)), 'x', '-2', '2'), ('term', '3')),
    (S(piecewise((0, Lt(Symbol('x'), 0)), (parse('x'), True)), 'x', '-2', '2'), ('truncate', 3)),
    (S(piecewise((1, Lt(Symbol('x'), pi)), (-1, True)), 'x', '0', '2*pi'), ('truncate', 3)),
    (S(piecewise((1, Lt(Symbol('x'), pi)), (-1, True)), 'x', '0', '2*pi'), ('sigma', 4)),
    (S('x', 'x', '0', 'pi'), ('truncate', 4)),
    (S('x', 'x', '0', '1'), ('truncate', 4)),
    (S('x**3', 'x', '-1', '1'), ('truncate', 3)),
    (S('exp(x)', 'x', '-pi', 'pi'), ('truncate', 2)),
    (S('Abs(x)', 'x', '-pi', 'pi'), ('truncate', 3)),
    (('shift', ('scale', FE, '-1'), 'pi**2'), ('truncate', 3)),
    (('shift', FP, '-pi/2'), ('truncate', 3)),
    (('scale', FO, '3'), ('truncate', 3)),
    (('shiftx', ('scalex', FE, '2'), '1'), ('truncate', 3)),
    (('scale', ('shiftx', ('scalex', ('shift', ('scalex', FE, '3'), '-pi'), '2'), '1'), '4'), ('truncate', 3)),
    (('shift', FO, 'x'), ('truncate', 3)),
    (('shiftx', FO, 'sin(x)'), ('truncate', 3)),
    (('scale', FO, 'x*y'), ('truncate', 3)),
    (('scalex', FO, 'x**2'), ('truncate', 3)),
    (('neg', FO), ('truncate', 3)),
    (('neg', FE), ('truncate', 3)),
    (('add', FO, FO), ('truncate', 3)),
    (('sub', ('neg', FE), FE), ('truncate', 3)),
    (('add', FO, FE), ('truncate', 3)),
    (('sub', FO, FE), ('truncate', 3)),
    (('sub', FO, FO), ('truncate', 3)),
    (('add', FO, S('x', 'x', '0', '2')), ('truncate', 3)),
    (S('sin(x)'), ('truncate', 1)),
    (S('sin(x)'), ('kind',)),
    (S('sin(x)*log(y)*exp(z)', 'x', 'pi', '-pi'), ('truncate', 3)),
    (SIN6, ('truncate', None)),
    (SIN6, ('truncate', 3)),
    (SIN6, ('coefficients',)),
    (SIN6, ('length',)),
    (SIN6, ('term', '4')),
    (SIN6, ('leading',)),
    (S('sin(4*x + 3) + cos(3*x + 4)'), ('truncate', None)),
    (S('sin(4*x + 3) + cos(3*x + 4)'), ('coefficients',)),
    (S('sin(x) + cos(x)*tan(x)'), ('truncate', None)),
    (S('cos(pi*x)', 'x', '-1', '1'), ('truncate', None)),
    (S('cos(3*pi*x + 4) - sin(4*pi*x)*log(pi*y)', 'x', '-1', '1'), ('truncate', None)),
    (S('cos(3*pi*x + 4) - sin(4*pi*x)*log(pi*y)', 'x', '-1', '1'), ('coefficients',)),
    (S('sin(x)*cos(x)'), ('coefficients',)),
    (S('x*sin(x)'), ('kind',)),
    (S('sin(x)**2', finite=False), ('truncate', 3)),
    (S('sin(x)**2', finite=False), ('kind',)),
    (('scale', SIN6, '2'), ('truncate', None)),
    (('scalex', SIN6, '2'), ('truncate', None)),
    (('shiftx', SIN6, '1'), ('truncate', None)),
    (('neg', SIN6), ('truncate', None)),
    (('shift', SIN6, '3'), ('coefficients',)),
    (('add', S('sin(x)**2'), S('cos(2*x)')), ('truncate', None)),
    (('add', S('sin(x)'), S('-sin(x)')), ('truncate', None)),
    (('add', S('sin(x)'), FO), ('truncate', 3)),
    (('add', FO, S('sin(x)')), ('truncate', 3)),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_fourier(i: int) -> None:
    spec, query = CASES[i]
    expected = outcome(lambda: python(spec, query))
    got = outcome(lambda: rust(spec, query))
    if got == 'SKIP':
        pytest.skip('not ported')
    assert got == expected, (spec, query)

"""The Rust port of series/sequences.py against the Python one: the
sequences of its tests built the same way, compared by their str, terms,
coefficients, intervals and linear recurrences (read with the core's
parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Function, Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
from sympy_extras.series.sequences import RecursiveSeq, SeqBase, sequence

from ..conftest import load

rc = load("sympy_extras_rs.core")

#: a sequence: ('formula', f, limits), ('per', [elements], limits),
#: ('rec', f, start, [initial]) on y(n), ('add'|'sub'|'mul', s, t),
#: ('neg', s), ('coeff_mul', s, c); limits a list of 0, 2 or 3 strings
Spec = tuple[object, ...]

y, n = Function('y'), Symbol('n')


def python_seq(spec: Spec) -> SeqBase:
    kind = spec[0]
    if kind == 'formula' or kind == 'per':
        lims = spec[2]
        assert isinstance(lims, list)
        limits = [parse(str(v)) for v in lims] if lims else None
        if kind == 'formula':
            return sequence(parse(str(spec[1])), limits)
        items = spec[1]
        assert isinstance(items, list)
        return sequence(tuple(parse(str(v)) for v in items), limits)
    if kind == 'rec':
        initial = spec[3]
        assert isinstance(initial, list)
        return RecursiveSeq(parse(str(spec[1])), y(n), n, [parse(str(v)) for v in initial], parse(str(spec[2])))
    a = spec[1]
    assert isinstance(a, tuple)
    if kind == 'neg':
        return -python_seq(a)
    if kind == 'coeff_mul':
        return python_seq(a).coeff_mul(parse(str(spec[2])))
    b = spec[2]
    assert isinstance(b, tuple)
    if kind == 'add':
        return python_seq(a) + python_seq(b)
    if kind == 'sub':
        return python_seq(a) - python_seq(b)
    return python_seq(a) * python_seq(b)


def encoded(spec: Spec) -> Basic:
    kind = spec[0]
    assert isinstance(kind, str)
    head = Symbol(kind)
    if kind == 'formula' or kind == 'per':
        lims = spec[2]
        assert isinstance(lims, list)
        limits = Tuple(*[parse(str(v)) for v in lims])
        if kind == 'formula':
            return Tuple(head, parse(str(spec[1])), limits)
        items = spec[1]
        assert isinstance(items, list)
        return Tuple(head, Tuple(*[parse(str(v)) for v in items]), limits)
    if kind == 'rec':
        initial = spec[3]
        assert isinstance(initial, list)
        return Tuple(head, parse(str(spec[1])), y(n), n, Tuple(*[parse(str(v)) for v in initial]),
                     parse(str(spec[2])))
    a = spec[1]
    assert isinstance(a, tuple)
    if kind == 'neg':
        return Tuple(head, encoded(a))
    if kind == 'coeff_mul':
        return Tuple(head, encoded(a), parse(str(spec[2])))
    b = spec[2]
    assert isinstance(b, tuple)
    return Tuple(head, encoded(a), encoded(b))


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return str(r)


Action = tuple[object, ...]


def python(spec: Spec, action: Action) -> object:
    s = python_seq(spec)
    name = action[0]
    if name == 'str':
        return str(s)
    if name == 'terms':
        k = action[1]
        assert isinstance(k, int)
        return Tuple(*s[:k])
    if name == 'coeff':
        return s.coeff(parse(str(action[1])))
    if name == 'length':
        return s.length
    if name == 'interval':
        return Tuple(s.start, s.stop)
    k, d, g = action[1], action[2], action[3]
    assert isinstance(k, int) and (d is None or isinstance(d, int))
    if g is None:
        return Tuple(Tuple(*s.find_linear_recurrence(k, d)), Symbol('NONE'))
    coeffs, gf = s.find_linear_recurrence(k, d, parse(str(g)))
    return Tuple(Tuple(*coeffs), Symbol('NONE') if gf is None else gf)


def rust(spec: Spec, action: Action) -> object:
    name = action[0]
    assert isinstance(name, str)
    args: list[Basic] = [Symbol('sequence'), encoded(spec), Symbol(name)]
    if name == 'terms':
        k = action[1]
        assert isinstance(k, int)
        args.append(Integer(k))
    elif name == 'coeff':
        args.append(parse(str(action[1])))
    elif name == 'recurrence':
        k, d, g = action[1], action[2], action[3]
        assert isinstance(k, int)
        args += [Integer(k), Symbol('NONE') if d is None else Integer(int(str(d))),
                 Symbol('NONE') if g is None else parse(str(g))]
    r = rc.logic_call('calculus', args)
    if name == 'str':
        assert isinstance(r, Symbol)
        return r.name
    return r


F: Callable[..., Spec] = lambda f, *lims: ('formula', f, list(lims))
P: Callable[..., Spec] = lambda items, *lims: ('per', list(items), list(lims))

CASES: list[tuple[Spec, Action]] = [
    (F('n**2', 'n', '0', 'oo'), ('str',)),
    (F('n**2', 'n', '0', 'oo'), ('terms', 4)),
    (F('n**2', 'n', '0', 'oo'), ('coeff', '5')),
    (F('n**2', 'n', '0', '5'), ('length',)),
    (F('n**2', 'n', '0', '5'), ('terms', 6)),
    (F('n**2', 'n', '0', '5'), ('coeff', '7')),
    (F('n**2', 'n', '3', '1'), ('str',)),
    (F('n**2', '0', '5'), ('str',)),
    (F('m*n**2', 'n', '0', '5'), ('terms', 3)),
    (F('m*n**2', '0', '5'), ('str',)),
    (F('n', 'n', '-oo', '0'), ('terms', 4)),
    (F('n', 'n', '-oo', 'oo'), ('str',)),
    (F('n**2', 'n', 'a', 'oo'), ('interval',)),
    (('add', F('n**2', 'n', '0', 'oo'), F('n', 'n', '2', 'oo')), ('terms', 3)),
    (('add', F('n**2', 'n', '0', 'oo'), F('n**3', 'n', '0', 'oo')), ('str',)),
    (('add', F('n**2', 'n', '0', '5'), F('n**2', 'n', '6', '10')), ('str',)),
    (('mul', F('n**2', 'n', '0', 'oo'), F('n**3', 'n', '0', 'oo')), ('str',)),
    (('mul', F('n**2', 'n', '0', 'oo'), P(['1', '2'], 'n', '0', 'oo')), ('terms', 4)),
    (('mul', F('n**2', 'n', '0', 'oo'), P(['1', '2'], 'n', '0', 'oo')), ('str',)),
    (('add', F('n**2', 'n', '0', 'oo'), P(['1', '2'], 'n', '1', 'oo')), ('str',)),
    (('add', F('n**2', 'n', '0', 'oo'), P(['1', '2'], 'n', '1', 'oo')), ('terms', 4)),
    (('sub', F('n**2', 'n', '0', 'oo'), P(['1', '2'], 'n', '1', 'oo')), ('terms', 4)),
    (('coeff_mul', ('add', F('n**2', 'n', '0', 'oo'), P(['1', '2'], 'n', '0', 'oo')), '3'), ('terms', 4)),
    (('coeff_mul', ('mul', F('n**2', 'n', '0', 'oo'), P(['1', '2'], 'n', '0', 'oo')), 'x'), ('terms', 4)),
    (('add', P(['1', '2'], 'n', '0', 'oo'), P(['1', '2', '3'], 'n', '0', 'oo')), ('str',)),
    (('mul', P(['1', 'n'], 'n', '0', 'oo'), P(['1', '2', '3'], 'n', '0', 'oo')), ('terms', 6)),
    (('neg', P(['1', '2'], 'n', '0', 'oo')), ('terms', 3)),
    (('neg', F('n**2', 'n', '0', 'oo')), ('str',)),
    (P(['1', 'n', '3'], 'n', '0', 'oo'), ('terms', 6)),
    (P(['1', 'n', '3'], 'n', '0', 'oo'), ('str',)),
    (P(['1', '2', '3'], 'n', '0', '5'), ('str',)),
    (P(['1', '2', '3'], 'n', '-oo', '0'), ('terms', 5)),
    (P(['1', '2', '3'], 'n', '0', '5'), ('coeff', '1/2')),
    (P(['1', '2'], 'n', '0', 'oo'), ('coeff', '-1')),
    (('add', ('add', F('n', 'n', '0', 'oo'), P(['1', '2'], 'n', '0', 'oo')), F('n**2', 'n', '0', 'oo')), ('str',)),
    (('rec', 'y(n - 1) + y(n - 2)', '0', ['0', '1']), ('str',)),
    (('rec', 'y(n - 1) + y(n - 2)', '0', ['0', '1']), ('terms', 10)),
    (('rec', 'y(n - 1) + y(n - 2)', '0', ['0', '1']), ('coeff', '3')),
    (('rec', 'y(n - 1) + y(n - 2)', '2', ['0', '1']), ('terms', 5)),
    (('rec', 'n*y(n - 1)', '1', ['1']), ('terms', 6)),
    (('rec', 'y(n - 2)**2 + n', '0', ['a', 'b']), ('terms', 5)),
    (('rec', 'y(n - 1) + y(n - 2)', '0', ['0', '1']), ('recurrence', 10, None, None)),
    (F('n**2'), ('recurrence', 10, 2, None)),
    (F('n**2'), ('recurrence', 10, None, None)),
    (F('2**n'), ('recurrence', 10, None, None)),
    (F('sqrt(5)*(((1 + sqrt(5))/2)**n - (-(1 + sqrt(5))/2)**(-n))/5'), ('recurrence', 10, None, None)),
    (F('x + y*(-2)**(-n)', 'n', '0', 'oo'), ('recurrence', 30, None, None)),
    (F('3*5**n + 12'), ('recurrence', 20, None, 'x')),
    (F('n**2'), ('recurrence', 10, None, 'x')),
    (F('n**3 - n'), ('recurrence', 12, None, 'x')),
    (F('n**2'), ('recurrence', 10, 2, 'x')),
    (P(['1', '2', '3'], 'n', '0', 'oo'), ('recurrence', 12, None, 'x')),
    (F('factorial(n)'), ('recurrence', 10, None, None)),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_sequences(i: int) -> None:
    spec, action = CASES[i]
    expected = outcome(lambda: python(spec, action))
    got = outcome(lambda: rust(spec, action))
    if got == 'SKIP':
        pytest.skip('not ported')
    assert got == expected, (spec, action)

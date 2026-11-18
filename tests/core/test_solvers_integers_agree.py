"""``sympy_extras.solvers.integers`` against its Rust port: the Hermite
normal form with its transform, linear Diophantine systems, Hilbert bases
and minimal nonnegative solutions (Contejean and Devie), Cooper's
elimination and Presburger quantifier elimination. The Rust port visits
the relations in the order of their first occurrence (the Python iterates
a set): Cooper's results are compared as strings, otherwise by their
truth at integer points."""

from __future__ import annotations

import itertools
import random
from typing import Any

import pytest

from ..conftest import load


def _symbols() -> Any:
    from sympy_extras.core import Symbol
    return Symbol('x'), Symbol('y'), Symbol('z')


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('int_linear', [Symbol(op), *args])


def _tuple(m: Any) -> Any:
    from sympy_extras.core import Integer, Tuple
    if isinstance(m, list):
        return Tuple(*[_tuple(r) for r in m])
    return Integer(m)


def _lists(t: Any) -> Any:
    from sympy_extras.core import Integer
    if isinstance(t, Integer):
        return t.p
    return [_lists(a) for a in t.args]


def _outcome(f: Any) -> Any:
    try:
        return ('ok', f())
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


MATRICES = [[[2, 4, 4], [-6, 6, 12], [10, -4, -16]], [[3, 5]], [[1, 2], [4, 0]], [[0, 0], [0, 3]],
            [[6, 10, 15]], [[2, 0, 0], [0, 1, 0]], [[1, -1, 2, 3], [0, 4, -2, 5]], [[-4, 6], [2, -3]]]


def test_hermite_normal_form(rust: Any) -> None:
    from sympy_extras.solvers.integers import hermite_normal_form_with_transform
    for A in MATRICES:
        h, u = _call('hnf', _tuple(A)).args
        assert [_lists(h), _lists(u)] == [list(map(list, m)) for m in hermite_normal_form_with_transform(A)], A


def test_linear_diophantine_system(rust: Any) -> None:
    from sympy_extras.core import Eq, Tuple
    from sympy_extras.solvers.integers import linear_diophantine_system
    x, y, z = _symbols()
    cases = [([Eq(3*x + 5*y, 7)], [x, y]), ([Eq(2*x + 4*y, 3)], [x, y]), ([x + y - 1, x + y - 2], [x, y]),
             ([x + y - 3, x - y - 1], [x, y]), ([6*x + 10*y + 15*z - 1], [x, y, z]), ([x*y - 1], [x, y]),
             ([x/2 - 1], [x]), ([Eq(x, 1)], []), ([x + 2*y - 3*z, 4*x - z - 8], [x, y, z]), ([2*x - 6], [x])]
    for equations, symbols in cases:
        got = _outcome(lambda: _call('diophantine', Tuple(*equations), Tuple(*symbols)))
        expected = _outcome(lambda: linear_diophantine_system(equations, symbols))
        assert got[0] == expected[0], (equations, got, expected)
        if got[0] != 'ok':
            continue
        if expected[1] is None:
            assert got[1] == Tuple(), equations
            continue
        values, parameters = got[1].args
        assert (list(values.args), list(parameters.args)) == (expected[1][0], expected[1][1]), equations


def test_hilbert_basis_and_minimal_solutions(rust: Any) -> None:
    from sympy_extras.solvers.integers import hilbert_basis, minimal_nonnegative_solutions
    for A in [[[1, 1, -1]], [[2, -3]], [[1, 1]], [[1, 1, -1, -1]], [[1, 2, -3], [2, -1, -1]], [[3, 2, -5, -1]]]:
        assert _lists(_call('hilbert', _tuple(A), _tuple(10000))) == hilbert_basis(A), A
    got = _outcome(lambda: _call('hilbert', _tuple([[1, -1]]), _tuple(1)))
    assert got[0] == _outcome(lambda: hilbert_basis([[1, -1]], limit=1))[0] == 'ERROR RuntimeError'
    for A, b in [([[3, 5]], [22]), ([[1, 1]], [2]), ([[3, -5]], [1]), ([[2, 4]], [3]), ([[1, 2, -1]], [3])]:
        p, h = _call('minimal', _tuple(A), _tuple(b)).args
        assert (_lists(p), _lists(h)) == minimal_nonnegative_solutions(A, b), (A, b)


def _formulas() -> list[Any]:
    from sympy_extras.core import Eq, Mod, Ne
    from sympy_extras.core.logic import Implies, Not, and_, or_
    x, y, z = _symbols()
    fs = [Eq(2*x, y), and_(x > y, x < y + 2), and_(x > y, x < y + 1), x > y, and_(Eq(x, y), Eq(x, z)),
          and_(3*x > y, 3*x < y + 3), and_(x > y, x < z), and_(2*x >= y, 3*x <= z), Eq(2*x + 3*y, z),
          and_(Ne(x, y), x > y - 1, x < y + 2), and_(Eq(Mod(x + y, 3), 0), x > z, x < z + 2),
          and_(Not(Eq(Mod(x, 2), 0)), x >= y, x <= y + 1), and_(or_(Eq(x, y), Eq(x, z)), x > 0),
          and_(Implies(x > y, x > z), x <= y + 1), and_(Eq(Mod(2*x + y, 4), 0), x > z),
          and_(5*x > 2*y + 1, 5*x < 2*y + 7), and_(Eq(3*x, y), Ne(x, z)), x**2 > y, Eq(Mod(x, 3), 5)]
    rng = random.Random(7)
    for _ in range(40):
        atoms = []
        for _ in range(rng.randint(1, 3)):
            e = rng.choice([1, 2, 3, -1, -2])*x + rng.randint(-2, 2)*y + rng.randint(-3, 3)
            kind = rng.choice(['lt', 'gt', 'le', 'ge', 'eq', 'ne', 'mod'])
            atoms.append({'lt': lambda: e < 0, 'gt': lambda: e > 0, 'le': lambda: e <= 0, 'ge': lambda: e >= 0,
                          'eq': lambda: Eq(e, 0), 'ne': lambda: Ne(e, 0),
                          'mod': lambda: Eq(Mod(e, rng.choice([2, 3])), 0)}[kind]())
        f = rng.choice([and_, or_])(*atoms)
        if rng.random() < 0.3:
            f = Not(f)
        fs.append(f)
    return fs


def _same(r: Any, p: Any) -> bool:
    from sympy_extras.core import Integer
    if str(r) == str(p):
        return True
    symbols = sorted(r.free_symbols | p.free_symbols, key=lambda s: s.name)
    for values in itertools.product(range(-6, 7), repeat=len(symbols)):
        rule = {s: Integer(v) for s, v in zip(symbols, values)}
        if bool(r._xreplace_evaluated(rule)) != bool(p._xreplace_evaluated(rule)):
            return False
    return True


@pytest.mark.parametrize("index", range(59))
def test_cooper(rust: Any, index: int) -> None:
    from sympy_extras.solvers.integers import cooper
    f = _formulas()[index]
    x = _symbols()[0]
    got = _outcome(lambda: _call('cooper', f, x))
    expected = _outcome(lambda: cooper(f, x))
    assert got[0] == expected[0], (f, got, expected)
    if got[0] == 'ok':
        assert _same(got[1], expected[1]), (f, got[1], expected[1])


def test_presburger(rust: Any) -> None:
    from sympy_extras.core import Eq, Mod, Symbol, Tuple
    from sympy_extras.core.logic import and_, or_
    from sympy_extras.solvers.integers import is_presburger, presburger_quantifier_elimination
    x, y, z = _symbols()
    cases = [(x < y, [('forall', x), ('exists', y)]), (x < y, [('exists', y), ('forall', x)]),
             (or_(Eq(3*y, x), Eq(3*y, x + 1), Eq(3*y, x + 2)), [('forall', x), ('exists', y)]),
             (or_(Eq(3*y, x), Eq(3*y, x + 1)), [('forall', x), ('exists', y)]),
             (and_(2*x > y, y > z), [('exists', x)]), (and_(Eq(2*x, y), Eq(3*z, y)), [('exists', x), ('exists', z)])]
    for f, prefix in cases:
        got = _call('presburger', f, Tuple(*[Tuple(Symbol(k), v) for k, v in prefix]))
        assert _same(got, presburger_quantifier_elimination(f, prefix)), (f, prefix)
    for f, ints in [(and_(2*x > y, Eq(Mod(x, 3), 0)), [x, y]), (x*y > 0, [x, y]), (x > y, [x]), (x/2 > y, [x, y])]:
        assert bool(_call('is_presburger', f, Tuple(*ints))) == is_presburger(f, set(ints)), f

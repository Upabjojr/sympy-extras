"""``sympy_extras.polys.virtual_substitution`` against its Rust port: the
linear and quadratic eliminations of Loos and Weispfenning, the quantifier
sequences, the negation normal form and ``with_sides``. The Rust port
visits the relations in the order of their first occurrence (the Python
iterates a set): the results are compared as strings, and otherwise as
formulas at random rational points of their free variables."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load


def _symbols() -> Any:
    from sympy_extras.core import Symbol
    return [Symbol(n) for n in 'xyzabc']


def _linear() -> list[Any]:
    from sympy_extras.core import And, Eq, Ne, Or
    x, y, z, a, b, c = _symbols()
    return [(x > y) & (x < 1), Eq(a*x, 1), (x > 0) & (x < 0), Or(x > y, x < z), (x >= y) & (x <= z), y > 0,
            (x < 2) & (a*x + b > 0), (x > 0) & Eq(a*x + b, 0), (x > 2) & (x*y < 1), Or(And(x > a, x < b), Eq(x, 3)),
            (a*x > b) & (x <= 0) & Ne(x, -1), (x - a > 0) & (b - x >= 0) & (x*y - 1 < 0), x**2 > 1,
            ~((x > y) | (x < 0)), Eq(2*x + y, 3) & (x > z)]


def _quadratic() -> list[Any]:
    from sympy_extras.core import Eq, Ne
    x, y, z, a, b, c = _symbols()
    return [Eq(x**2 + b*x + c, 0), (x**2 < a) & (x > 1), x**2 < 0, x**2 + 1 > 0, x**3 > y, Eq(a*x**2 + b*x + c, 0),
            (x**2 + b*x + c > 0) & (x > 0), (x**2 < y) & (x > y - 1), (a*x**2 + 1 <= 0) | (x < a),
            Ne(x**2 - a, 0) & (x >= 0) & (x <= 1), (x**2 - 2*a*x + b > 0) & (x < a), Eq(x**2, a) & Eq(x, b),
            (x**2 >= a) & (x**2 <= b), (x**2 + a*x < 0) & (x**2 - b*x > 0), (a*x**2 + b*x + 1 >= 0) & (x < 0)]


def _same(r: Any, p: Any) -> bool:
    """Equal strings, or the same truth at random rational points."""
    from sympy_extras.core import Rational
    if str(r) == str(p):
        return True
    symbols = sorted(r.free_symbols | p.free_symbols, key=lambda s: s.name)
    rng = random.Random(11)
    for _ in range(60):
        values = {s: Rational(rng.randint(-12, 12), rng.randint(1, 4)) for s in symbols}
        if bool(r._xreplace_evaluated(values)) != bool(p._xreplace_evaluated(values)):
            return False
    return True


def _outcome(f: Any) -> Any:
    try:
        return ('ok', f())
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


@pytest.mark.parametrize("index", range(15))
def test_linear(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.polys.virtual_substitution import eliminate_linear, to_nnf, with_sides
    rc = load("sympy_extras_rs.core")
    f = _linear()[index]
    x = _symbols()[0]
    got = _outcome(lambda: rc.logic_call('virtual_sub', [Symbol('linear'), f, x]))
    expected = _outcome(lambda: eliminate_linear(f, x))
    assert got[0] == expected[0], (f, got, expected)
    if got[0] == 'ok':
        assert _same(got[1], expected[1]), (f, got[1], expected[1])
    assert str(rc.logic_call('virtual_sub', [Symbol('nnf'), f])) == str(to_nnf(f)), f
    assert _same(rc.logic_call('virtual_sub', [Symbol('with_sides'), f]), with_sides(f)), f


@pytest.mark.parametrize("index", range(15))
def test_quadratic(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.polys.virtual_substitution import eliminate_quadratic
    rc = load("sympy_extras_rs.core")
    f = _quadratic()[index]
    x = _symbols()[0]
    got = _outcome(lambda: rc.logic_call('virtual_sub', [Symbol('quadratic'), f, x]))
    expected = _outcome(lambda: eliminate_quadratic(f, x))
    assert got[0] == expected[0], (f, got, expected)
    if got[0] == 'ok':
        assert _same(got[1], expected[1]), (f, got[1], expected[1])


def test_quantifier_sequences(rust: Any) -> None:
    from sympy_extras.core import Eq, Symbol, Tuple
    from sympy_extras.polys.virtual_substitution import linear_quantifier_elimination, virtual_substitution_elimination
    rc = load("sympy_extras_rs.core")
    x, y, z, a, b, c = _symbols()
    cases = [('linear_qe', (x > y) & (y > z), [('exists', x)]), ('linear_qe', (x > y) & (y > z), [('forall', z), ('exists', x)]),
             ('linear_qe', (x > y) & (y > z), [('exists', z), ('exists', x)]),
             ('linear_qe', (x**2 > y) & (y > z), [('exists', x), ('exists', z)]),
             ('qe', Eq(x**2 + b*x + c, 0), [('exists', x)]), ('qe', x**2 + b*x + c > 0, [('forall', x)]),
             ('qe', x**3 > y, [('exists', x)]), ('qe', (x**2 < y) & (y < z), [('exists', y), ('exists', x)])]
    for op, f, prefix in cases:
        python = (linear_quantifier_elimination if op == 'linear_qe' else virtual_substitution_elimination)(f, prefix)
        r = rc.logic_call('virtual_sub', [Symbol(op), f, Tuple(*[Tuple(Symbol(k), v) for k, v in prefix])])
        formula, rest = r.args
        assert _same(formula, python[0]), (f, formula, python[0])
        assert [(str(k), v) for k, v in (q.args for q in rest.args)] == list(python[1]), f

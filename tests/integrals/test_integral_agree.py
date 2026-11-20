"""The Rust port of the ``Integral`` node (``integrals/integral.py``)
against the Python one: the node through the bridge both ways, its
printing, free symbols, ``is_number``, derivatives and substitutions."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol, Tuple, oo
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.functions import exp, log, sin
from sympy_extras.integrals.integral import Integral

from ..conftest import load

rc = load("sympy_extras_rs.core")

x, y, z, t = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('t')
a, b = Symbol('a'), Symbol('b')


def outcome(f: Callable[[], object]) -> str:
    try:
        return str(f())
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__


NODES: list[Integral] = [
    Integral(exp(x**2) * y, x),
    Integral(exp(-y * x**2), x, Integer(0), oo),
    Integral(sin(x * y), x, a, b),
    Integral(x * y, x, y, y**2),
    Integral(log(x + z), x, Integer(1), Integer(2)),
    Integral(sin(x), x, Integer(0), Integer(1)),
    Integral(Integer(1), x),
    Integral(x**t, x, Integer(0), y),
    Integral(Integral(x * y, x), y),
    Integral(Integral(x * y, x, Integer(0), y), y, Integer(0), Integer(1)),
]

#: the substitutions (old, new)
SUBS: list[tuple[Expr, Expr]] = [(y, Integer(2)), (x, z), (a, Integer(0)), (y, z**2), (z, x), (t, Integer(3))]

#: the variables of the derivatives
WRT: list[Symbol] = [x, y, a, b, z]

#: (the node, the operation, the index of its derivative variable or substitution)
CASES: list[tuple[int, str, int]] = (
    [(k, 'node', 0) for k in range(len(NODES))]
    + [(k, 'free', 0) for k in range(len(NODES))]
    + [(k, 'number', 0) for k in range(len(NODES))]
    + [(k, 'diff', j) for k in range(len(NODES)) for j in range(len(WRT))]
    + [(k, 'subs', j) for k in range(len(NODES)) for j in range(len(SUBS))]
)


@pytest.mark.parametrize('k, op, j', CASES)
def test_agrees(k: int, op: str, j: int) -> None:
    node = NODES[k]
    rs_args: list[Basic] = [Symbol(op), node]
    if op == 'node':
        py = outcome(lambda: node)
    elif op == 'free':
        py = outcome(lambda: Tuple(*sorted(node.free_symbols, key=str)))
    elif op == 'number':
        py = str(node.is_number).lower()
    elif op == 'diff':
        s = WRT[j]
        py = outcome(lambda: node.diff(s))
        rs_args.append(s)
    else:
        old, new = SUBS[j]
        py = outcome(lambda: node.subs(old, new))
        rs_args += [old, new]
    rs = outcome(lambda: rc.logic_call('integral', rs_args))
    if op == 'number':
        rs = rs.lower()
    assert py == rs

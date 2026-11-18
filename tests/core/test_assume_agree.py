"""``sympy_extras.core.assume`` against its Rust port: the predicate
nodes (``Q``'s and the user's, equal on the name and that flag), the
applied predicates, their round trip through the bridge, printing,
canonical order, kind and substitution."""

from __future__ import annotations

from typing import Any

from ..conftest import load


def _nodes() -> list[Any]:
    from sympy_extras.core import Predicate, Q, Symbol
    x, y = Symbol('x'), Symbol('y')
    P = Predicate('mypred')
    return [Q.positive, Q.extended_nonnegative, Q.gt, P, Predicate('positive'), Q.positive(x), Q.real(x + y),
            Q.gt(x, y), P(x), Q.positive(x) & Q.real(y), ~Q.integer(x), Q.prime(2)]


def test_round_trip_and_printing(rust: Any) -> None:
    rc = load("sympy_extras_rs.core")
    for node in _nodes():
        back = rc.roundtrip(node)
        assert back == node and type(back) is type(node), node
        assert rc.sstr(node) == str(node), node


def test_equality_order_and_kind(rust: Any) -> None:
    from sympy_extras.core import Predicate, Q, Symbol
    from sympy_extras.core.sorting import default_sort_key
    rc = load("sympy_extras_rs.core")
    nodes = _nodes()
    for a in nodes:
        for b in nodes:
            assert rc.equal(a, b) == (a == b), (a, b)
    assert [str(n) for n in rc.sorted_by_key(nodes)] == [str(n) for n in sorted(nodes, key=default_sort_key)]
    for node in nodes:
        assert str(rc.logic_call('kind', [node])) == str(node.kind), node
    x = Symbol('x')
    assert str(rc.subs(Q.positive(x + 1), x, Symbol('z'))) == str(Q.positive(x + 1).subs(x, Symbol('z')))
    assert not rc.equal(Predicate('positive'), Q.positive)


def test_arity(rust: Any) -> None:
    import pytest
    from sympy_extras.core import Q, Symbol
    with pytest.raises(TypeError):
        Q.positive(Symbol('x'), Symbol('y'))

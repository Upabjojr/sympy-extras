"""The ``rewrite`` rules of the orthogonal polynomials
(``sympy_extras.functions.special._rewrites_polynomials``) against their
Rust port: every polynomial with symbolic, integer and invalid degrees,
rewritten as a ``Sum`` (and its old name ``polynomial``) and the Hermite
polynomials in terms of each other; the outcomes compared as trees, the
summation dummies numbered by their first appearance."""

from __future__ import annotations

import re
from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _nodes() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.functions.special import polynomials as p
    x, a, b = Symbol('x'), Symbol('a'), Symbol('b')
    n = Symbol('n')
    m = Symbol('m', integer=True, nonnegative=True)
    k = Symbol('k', negative=True)
    degrees = [n, m, k, Rational(1, 2)]
    builders: list[Any] = []
    for d in degrees:
        builders += [lambda d=d: p.jacobi(d, a, b, x), lambda d=d: p.gegenbauer(d, a, x),
                     lambda d=d: p.chebyshevt(d, x), lambda d=d: p.chebyshevu(d, x), lambda d=d: p.legendre(d, x),
                     lambda d=d: p.hermite(d, x), lambda d=d: p.hermite_prob(d, x), lambda d=d: p.laguerre(d, x),
                     lambda d=d: p.assoc_laguerre(d, a, x)]
    builders += [lambda: p.assoc_legendre(n, Symbol('j'), x), lambda: p.assoc_legendre(n, Integer(1), x)]
    nodes: list[Any] = []
    for build in builders:
        try:
            nodes.append(build())
        except ValueError:
            # the degrees which the polynomial refuses (laguerre(1/2, x))
            pass
    return nodes


def _targets() -> list[tuple[str, Any]]:
    from sympy_extras.functions.special import polynomials as p
    return [('Sum', 'Sum'), ('polynomial', 'polynomial'), ('hermite_prob', p.HermiteProb), ('hermite', p.Hermite)]


def _renumbered(value: object) -> str:
    text = str(value)
    names: dict[str, str] = {}
    return re.sub(r'_k\d*', lambda g: names.setdefault(g.group(0), '_k%d' % len(names)), text)


@pytest.mark.parametrize("index", range(len(_nodes())))
def test_rewrite(rust: Any, index: int) -> None:
    from sympy_extras.core import Integer, Symbol
    rc = load("sympy_extras_rs.core")
    e = _nodes()[index]
    for name, target in _targets():
        got = _renumbered(outcome(lambda: rc.logic_call('rewrite', [e, Symbol(name), Integer(1)])))
        expected = _renumbered(outcome(lambda: e.rewrite(target)))
        assert got == expected, (e, name, got, expected)

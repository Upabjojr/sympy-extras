"""The ``rewrite`` rules of the Bessel, spherical Bessel and Airy
functions and of ``Ynm``
(``sympy_extras.functions.special._rewrites_bessel``) against their Rust
port: every head with several orders and arguments, rewritten to every
target; the outcomes compared as trees (``NotImplementedError`` on both
sides is a skip; the rules of ``NOT_PORTED`` are a skip on the Rust side)."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _nodes() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.functions.special import bessel as b
    x, z = Symbol('x'), Symbol('z')
    p = Symbol('p', positive=True)
    n = Symbol('n', integer=True)
    nu = Symbol('nu')
    orders = [Rational(1, 3), Rational(1, 2), n, Integer(2), nu]
    args = [z, p, -p]
    nodes = []
    for f in (b.besselj, b.bessely, b.besseli, b.besselk, b.jn, b.yn, b.hn1, b.hn2):
        for o in orders:
            for a in args:
                nodes.append(f(o, a))
    for f in (b.airyai, b.airybi, b.airyaiprime, b.airybiprime):
        for a in (z, p, -p, x + 1):
            nodes.append(f(a))
    from sympy_extras.functions.special.spherical_harmonics import Ynm
    theta, phi = Symbol('theta'), Symbol('phi')
    for nn, m in [(1, 0), (2, 1), (2, -1), (3, 2)]:
        nodes.append(Ynm(Integer(nn), Integer(m), theta, phi))
    return nodes


def _targets() -> list[tuple[str, Any]]:
    from sympy_extras.functions.elementary.trigonometric import Cos, Sin
    from sympy_extras.functions.special import bessel as b
    return [('besselj', b.BesselJ), ('bessely', b.BesselY), ('besseli', b.BesselI), ('jn', b.SphericalJ),
            ('yn', b.SphericalY), ('hankel1', b.Hankel1), ('hankel2', b.Hankel2), ('hyper', 'hyper'),
            ('polynomial', 'polynomial'), ('cos', Cos), ('sin', Sin)]


@pytest.mark.parametrize("index", range(len(_nodes())))
def test_rewrite(rust: Any, index: int) -> None:
    from sympy_extras.core import Integer, Symbol
    rc = load("sympy_extras_rs.core")
    e = _nodes()[index]
    for name, target in _targets():
        got = outcome(lambda: rc.logic_call('rewrite', [e, Symbol(name), Integer(1)]))
        expected = outcome(lambda: e.rewrite(target))
        if got == 'SKIP' and expected != 'SKIP' and _not_ported(e, name):
            continue
        assert got == expected, (e, name, got, expected)


#: the rules the Rust port does not have: Ynm into cos and sin (simplify)
NOT_PORTED = {('YnmNode', 'cos'), ('YnmNode', 'sin')}


def _not_ported(e: Any, target: str) -> bool:
    from sympy_extras.core.basic import preorder_traversal
    return any((type(node).__name__, target) in NOT_PORTED for node in preorder_traversal(e))

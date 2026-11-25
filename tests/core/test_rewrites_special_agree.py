"""The ``rewrite`` rules of the error, exponential and trigonometric
integral, incomplete gamma, polygamma, zeta, beta, elliptic and Marcum
functions and of ``hyper`` (``sympy_extras.functions.special._rewrites``)
against their Rust port: every node rewritten in terms of every target,
the outcomes compared as trees, the fresh dummies numbered by their first
appearance."""

from __future__ import annotations

import re
from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _nodes() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras import functions as f
    x, y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')
    s, m, z = Symbol('s'), Symbol('m'), Symbol('z')
    p = Symbol('p', positive=True)
    n = Symbol('n', integer=True, positive=True)
    k = Symbol('k', integer=True, nonpositive=True)
    w = Symbol('w', negative=True)
    builders: list[Any] = [
        lambda: f.erf(x), lambda: f.erfc(x), lambda: f.erfi(x), lambda: f.erf2(x, y), lambda: f.erfinv(x),
        lambda: f.erfcinv(x), lambda: f.fresnels(x), lambda: f.fresnelc(x),
        lambda: f.expint(Integer(1), x), lambda: f.expint(Integer(3), x), lambda: f.expint(s, x),
        lambda: f.expint(Rational(1, 2), x),
        lambda: f.Ei(x), lambda: f.Ei(f.log(x)), lambda: f.Ei(p), lambda: f.Ei(w),
        lambda: f.li(x), lambda: f.Li(x), lambda: f.Si(x), lambda: f.Ci(x), lambda: f.Shi(x), lambda: f.Chi(x),
        lambda: f.lowergamma(s, x), lambda: f.lowergamma(k, x), lambda: f.uppergamma(s, x),
        lambda: f.polygamma(n, x), lambda: f.polygamma(Integer(0), x), lambda: f.polygamma(s, x),
        lambda: f.digamma(x), lambda: f.trigamma(x),
        lambda: f.lerchphi(z, s, a), lambda: f.lerchphi(Integer(-1), s, a), lambda: f.polylog(s, x),
        lambda: f.zeta(s), lambda: f.zeta(s, a), lambda: f.zeta(Integer(4)), lambda: f.zeta(Integer(-3)),
        lambda: f.dirichlet_eta(s), lambda: f.dirichlet_eta(s, a), lambda: f.riemann_xi(s),
        lambda: f.beta(x, y), lambda: f.betainc(a, b, x, y), lambda: f.betainc_regularized(a, b, x, y),
        lambda: f.elliptic_k(m), lambda: f.elliptic_f(z, m), lambda: f.elliptic_e(m), lambda: f.elliptic_e(z, m),
        lambda: f.elliptic_pi(n, m), lambda: f.elliptic_pi(x, z, m),
        lambda: f.marcumq(m, a, b), lambda: f.marcumq(Integer(1), a, a), lambda: f.marcumq(Integer(3), a, a),
        lambda: f.hyper([a, b], [s], x), lambda: f.hyper([Integer(1)], [Integer(2)], x),
        lambda: f.hyper([a], [b], x)]
    nodes: list[Any] = []
    for build in builders:
        try:
            nodes.append(build())
        except (ValueError, TypeError):
            pass
    return nodes


def _targets() -> list[tuple[str, Any]]:
    from sympy_extras.functions.combinatorial import numbers as nb
    from sympy_extras.functions.special import bessel as b, error_functions as er, gamma_functions as g
    from sympy_extras.functions.special import zeta_functions as z
    return [('erf', er.Erf), ('erfc', er.Erfc), ('erfi', er.Erfi), ('fresnels', er.FresnelS),
            ('fresnelc', er.FresnelC), ('meijerg', 'meijerg'), ('hyper', 'hyper'), ('uppergamma', g.UpperGamma),
            ('lowergamma', g.LowerGamma), ('expint', er.Expint), ('erfinv', er.Erfinv), ('erfcinv', er.Erfcinv),
            ('Integral', 'Integral'), ('Ei', er.EiNode), ('li', er.LiNode), ('Li', er.OffsetLiNode),
            ('Si', er.SiNode), ('Ci', er.CiNode), ('Shi', er.ShiNode), ('Chi', er.ChiNode), ('zeta', z.Zeta),
            ('harmonic', nb.Harmonic), ('polygamma', g.PolyGamma), ('lerchphi', z.LerchPhi),
            ('polylog', z.PolyLog), ('bernoulli', nb.Bernoulli), ('dirichlet_eta', z.DirichletEta),
            ('genocchi', nb.Genocchi), ('besseli', b.BesselI), ('Sum', 'Sum')]


def _renumbered(value: object) -> str:
    text = str(value)
    names: dict[str, str] = {}
    return re.sub(r'\b_(?:t\d*|k|n|x)\b', lambda g: names.setdefault(g.group(0), '_d%d' % len(names)), text)


@pytest.mark.parametrize("index", range(len(_nodes())))
def test_rewrite(rust: Any, index: int) -> None:
    from sympy_extras.core import Integer, Symbol
    rc = load("sympy_extras_rs.core")
    e = _nodes()[index]
    for name, target in _targets():
        got = _renumbered(outcome(lambda: rc.logic_call('rewrite', [e, Symbol(name), Integer(1)])))
        if got == 'SKIP':
            continue
        expected = _renumbered(outcome(lambda: e.rewrite(target)))
        assert got == expected, (e, name, got, expected)

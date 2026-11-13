"""The ΠΣ-fields of the Rust port against ``sympy_extras.concrete.pisigma``:
the shifts, levels, telescoped sums and Karr's solver of the
parameterized first order equation in rational and parametric towers
with Σ-extensions (harmonic numbers) and Π-extensions (``2**k``,
``k!``, ``binomial(n, k)``), compared through ``to_expr`` with the
generators as symbols."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load


def python_report(F: Any, a: Any, fs: list[Any]) -> dict[str, Any]:
    elements = [F.from_expr(f) for f in fs]

    def shown(e: Any) -> Any:
        return F.to_expr(e, substitute=False)

    telescoped = []
    for f in elements:
        g = F.telescope(f)
        telescoped.append(None if g is None else shown(g))
    report: dict[str, Any] = {
        "sigma": [x for f in elements for x in (shown(F.sigma(f)), shown(F.sigma(f, -2)))],
        "level_of": [F.level_of(f) for f in elements],
        "telescope": telescoped,
    }
    sols = F.solve(F.from_expr(a), elements)
    report["solve"] = None if sols is None else [([F.constant_to_expr(c) for c in cs], shown(g)) for cs, g in sols]
    return report


def check(rc: Any, F: Any, a: Any, fs: list[Any]) -> None:
    extensions = [(e.kind, e.symbol, e.value, e.expr) for e in F.extensions]
    got = rc.pisigma_report_ops(F.k, list(F.params), extensions, a, fs)
    expected = python_report(F, a, fs)
    assert repr(got["sigma"]) == repr(expected["sigma"])
    assert got["level_of"] == expected["level_of"]
    assert repr(got["telescope"]) == repr(expected["telescope"]), (fs, got["telescope"], expected["telescope"])
    assert repr(got["solve"]) == repr(expected["solve"]), (a, fs)


def towers() -> list[tuple[str, Any]]:
    from sympy_extras.concrete.pisigma import parametric_pisigma_field, rational_pisigma_field
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.functions import binomial, factorial, harmonic
    k, n = Symbol('k'), Symbol('n')
    result: list[tuple[str, Any]] = [("rational", rational_pisigma_field(k))]
    F, _ = rational_pisigma_field(k).add_sigma(1/(k + 1), harmonic(k))
    result.append(("harmonic", F))
    F2, _ = F.add_sigma(1/(k + 1)**2, harmonic(k, 2))
    result.append(("harmonic2", F2))
    G, _ = rational_pisigma_field(k).add_pi(2, Integer(2)**k)
    result.append(("power", G))
    H, _ = rational_pisigma_field(k).add_pi(k + 1, factorial(k))
    H, _ = H.add_sigma(1/(k + 1), harmonic(k))
    result.append(("factorial-harmonic", H))
    P, _ = parametric_pisigma_field(k, [n]).add_pi((n - k)/(k + 1), binomial(n, k))
    result.append(("binomial", P))
    return result


@pytest.mark.parametrize("index", range(6))
def test_towers(rust: Any, index: int) -> None:
    from sympy_extras.core import Integer, Symbol
    rc = load("sympy_extras_rs.core")
    k, n = Symbol('k'), Symbol('n')
    name, F = towers()[index]
    t = [e.symbol for e in F.extensions]
    fs: list[Any] = [k, 1/(k*(k + 1)), k**2 + 3*k, Integer(1)]
    if name.startswith("harmonic"):
        fs += [t[0], k*t[0], t[0]**2, t[0]/(k + 1)]
        if name == "harmonic2":
            fs += [t[1], t[0]*t[1]]
    elif name == "power":
        fs += [t[0], k*t[0], t[0]/(k + 1), k**2/t[0]]
    elif name == "factorial-harmonic":
        fs += [k*t[0], t[0]*t[1], t[0]*(k + 1)*t[1]]
    elif name == "binomial":
        fs += [t[0], k*t[0], (n - 2*k)*t[0], n/(k + n)]
    for a in [Integer(1), k + 1, Integer(2)]:
        check(rc, F, a, fs)

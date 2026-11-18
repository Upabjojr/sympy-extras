"""``sympy_extras.integrals.curves`` (part 3) against its Rust port: the
whole of Trager's algorithm (``integrate_on_curve``): the places above
the roots of the denominator and their residues by the Puiseux
expansions, the field of the residues, the functions of the divisors
found as solutions of linear systems, the exact check, and the verdict;
the elementary part and the remainder written back in `x` and the
original radicals."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def integrands() -> list[Any]:
    from sympy_extras.core import as_expr, symbols
    from sympy_extras.functions import cbrt, log, sqrt
    from sympy_extras.polys.polytools import cancel
    x = symbols('x')
    derivatives = [
        sqrt(x + 1)*sqrt(x - 1) + log(sqrt(x + 1) + sqrt(x - 1)),
        x*cbrt(x**2 + 1) + log(cbrt(x**2 + 1) + 1),
        log(sqrt(x + sqrt(x)) + sqrt(x)) + sqrt(x + sqrt(x))/x,
        (x**2 + 1)/sqrt(x**3 + 2) + 2*log(1 + sqrt(x)),
        log(x + sqrt(x**2 + 1)) - sqrt(x**2 + 1)/x + log(sqrt(x**2 + 1) + sqrt(x**2 + 2)),
    ]
    direct = [1/(x*sqrt(1 - x**3)), 1/sqrt(x**3 + 1), x/sqrt(1 - x**3), 1/sqrt(x**5 + 1), x**2/sqrt(x**4 + 1),
              x/cbrt(x**3 + 1), 1/cbrt(x**3 + 1), cbrt(x)/(x + 1), (x**3 + 1)/((x - 1)**2*sqrt(x**3 + 1)),
              1/(x*sqrt(x**3 + 1)), sqrt(x)/(x + 1)**2, 1/((x**2 + 1)*sqrt(x**2 + 2)), 1/(x*sqrt(x**2 - 2))]
    return [as_expr(cancel(F.diff(x))) for F in derivatives] + direct


@pytest.mark.parametrize("index", range(18))
def test_integrate_on_curve(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.integrals.curves import integrate_on_curve
    rc = load("sympy_extras_rs.core")
    x = Symbol('x')
    f = integrands()[index]

    def expected() -> Any:
        result = integrate_on_curve(f, x)
        return (str(result.elementary_part()), str(result.remainder_expr()), result.nonelementary)

    def got() -> Any:
        r = rc.logic_call('curve_integrate', [f, x])
        verdict = r.args[2]
        return (str(r.args[0]), str(r.args[1]), None if verdict == 0 else bool(verdict))
    assert outcome(got) == outcome(expected), f

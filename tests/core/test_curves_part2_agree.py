"""``sympy_extras.integrals.curves`` (part 2) against its Rust port: the
regular integer of the change of variable, Trager's Hermite reduction in
the integral basis normal at infinity (its elementary part and the
remaining integrand, in `x` and the original radicals) and Trager's
criterion at infinity (``integrate_on_curve(f, x, logarithms=False)``)."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def integrands() -> list[Any]:
    from sympy_extras.core import Rational, symbols
    from sympy_extras.functions import sqrt
    x = symbols('x')
    return [x/sqrt(1 - x**3), 1/(sqrt(x + 1) + sqrt(x - 1)), 1/(x*sqrt(x**3 + 1)), x/sqrt(x**2 + 1),
            1/(x**2*sqrt(x**2 - 1)), sqrt(x)/(x + 1)**2, (x + 1)/(x**2*sqrt(x**2 + 1)), 1/(x**3*sqrt(x + 1)),
            x/(x**2 - 2)**Rational(1, 3), 1/((x - 1)**2*sqrt(x**3 + x)), sqrt(x + 1)/x**3, 1/sqrt(x**4 + 1),
            (x**2 + 1)**Rational(1, 3)/x**4, 1/((x**2 + 1)**2*sqrt(x**2 + 2))]


@pytest.mark.parametrize("index", range(14))
def test_hermite(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.integrals.curves import integrate_on_curve
    rc = load("sympy_extras_rs.core")
    x = Symbol('x')
    f = integrands()[index]

    def expected() -> Any:
        result = integrate_on_curve(f, x, logarithms=False)
        return (result.transformed.N, str(result.elementary_part()), str(result.remainder_expr()),
                result.nonelementary)

    def got() -> Any:
        r = rc.logic_call('curve_hermite', [f, x])
        verdict = r.args[3]
        return (int(r.args[0]), str(r.args[1]), str(r.args[2]), None if verdict == 0 else bool(verdict))
    assert outcome(got) == outcome(expected), f

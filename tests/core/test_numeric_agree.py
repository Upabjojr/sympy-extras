"""``sympy_extras._numeric`` against its Rust port: the hidden zeros
written 0, the sums refused when their value hangs on a rounding, the
arguments on a cut, the parts which are exactly zero, the small values
which need more precision, the values at the values of the symbols, and
the proofs of zero."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from ..conftest import load


def _cases() -> list[tuple[Any, int, dict[Any, Any]]]:
    from sympy_extras.core import I, Integer, Rational, Symbol, pi, sqrt
    from sympy_extras.functions import Log, asin, atan, cos, exp, gamma, log, sin
    x = Symbol('x')
    Z = -6 + (-2 + sqrt(2))**2 + 4*sqrt(2)
    W = sin(Integer(1))**2 + cos(Integer(1))**2 - 1
    logarithms = log(Integer(2)) + log(Integer(3)) - log(Integer(6))
    base = I*(log(1 + 3*I) - log(1 - 3*I))/2
    tiny = (1 + exp(Integer(-100))) - 1
    cases: list[tuple[Any, int, dict[Any, Any]]] = []
    for digits in (15, 20, 30):
        cases += [(Log(-2*sqrt(2) - 2*sqrt(Z)), digits, {}), (sqrt(Z) + 1, digits, {}), (sqrt(W) + 1, digits, {}),
                  (log(-1 - sqrt(W)), digits, {}), (W, digits, {}), (asin(Integer(-1)) + pi/2, digits, {}),
                  (Log(-1 - sqrt(Z)), digits, {}), (sqrt(logarithms) + 1, digits, {}), (logarithms, digits, {}),
                  (base**Rational(1, 3), digits, {}), (log(-1 + I*exp(Integer(-200))), digits, {}),
                  (Log(-1 + I*tiny), digits, {}), (atan(x)**Rational(1, 3), digits, {x: Integer(2)}),
                  (Log(-2*sqrt(2)*x - 2*sqrt(Z)), digits, {x: Integer(3)}), (sqrt(x), digits, {x: Integer(-4)}),
                  (exp(-1000000*x**2 - 50), digits, {x: Integer(2)}),
                  (x*gamma(x) - gamma(x + 1), digits, {x: Rational(-7, 50)}), (exp(I*pi/7) + 2, digits, {})]
    return cases


def _close(a: Optional[complex], b: Optional[complex], digits: int) -> bool:
    if a is None or b is None:
        return a is b
    # a hidden zero evaluated by itself is rounding noise on both sides
    # (harmless, see reliable_form): an absolute floor of the digits asked
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) <= max(scale*10.0**(-digits + 3), 10.0**(-digits))


@pytest.mark.parametrize("index", range(54))
def test_reliable(rust: Any, index: int) -> None:
    from sympy_extras._numeric import proved_zero, reliable_complex, reliable_form, reliable_value
    from sympy_extras.core import Integer, Symbol, Tuple
    from sympy_extras.core.evalf import numeric_value
    rc = load("sympy_extras_rs.core")
    e, digits, values = _cases()[index]
    pairs = Tuple(*[Tuple(k, v) for k, v in values.items()])

    def call(op: str) -> Any:
        return rc.logic_call('reliable', [Symbol(op), e, Integer(digits), pairs])
    form = reliable_form(e, digits)
    got_form = call('form')
    assert (None if isinstance(got_form, Tuple) else str(got_form)) == (None if form is None else str(form)), e
    value = reliable_value(e, digits, values or None)
    got = call('value')
    python_value = None if value is None else complex(numeric_value(value, 200))
    rust_value = None if not got.args else complex(float(got.args[0]), float(got.args[1]))
    assert _close(rust_value, python_value, min(digits, 15)), (e, rust_value, python_value)
    c = reliable_complex(e, digits, values or None)
    got_c = call('complex')
    rust_c = None if not got_c.args else complex(float(got_c.args[0]), float(got_c.args[1]))
    assert _close(rust_c, c, min(digits, 15)), (e, rust_c, c)
    assert bool(rc.logic_call('reliable', [Symbol('proved_zero'), e])) == proved_zero(e), e

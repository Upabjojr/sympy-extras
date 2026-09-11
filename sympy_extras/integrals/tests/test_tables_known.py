"""Every entry of the table checked numerically at its sample parameter
values against mpmath's quadrature (the values are transcribed from
Gradshteyn and Ryzhik; a transcription error would show here)."""
from __future__ import annotations

import mpmath

from sympy import Expr, symbols, oo, lambdify
from sympy.core.symbol import Wild

from sympy_extras._typing import as_expr
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.tables import TABLE, TableEntry

x = symbols('x')


def _frequency(entry: TableEntry, sample: dict[Wild, Expr]) -> float:
    """The largest scale of the oscillation, for ``quadosc``."""
    scales = [abs(float(v)) for v in sample.values() if as_expr(v).is_number] + [1.0]
    return max(scales)


def _quadrature(f: Expr, lower: Expr, upper: Expr, oscillatory: bool, omega: float) -> complex:
    g = lambdify(x, f, 'mpmath')
    lo = float(lower)
    if upper == oo:
        if oscillatory:
            head = mpmath.quad(g, [lo, 1, 10, 40])
            tail = mpmath.quadosc(g, [40, mpmath.inf], omega=omega)
            return complex(head + tail)
        return complex(mpmath.quad(g, [lo, 1, 10, 100, mpmath.inf]))
    if lower == -oo:
        return complex(mpmath.quad(g, [-mpmath.inf, -10, 0, 10, mpmath.inf]))
    return complex(mpmath.quad(g, [lo, float(upper)]))


def _value(entry: TableEntry, sample: dict[Wild, Expr]) -> complex:
    return complex(as_expr(entry.value.xreplace(dict(sample))).evalf(20))


def test_every_entry_numerically() -> None:
    mpmath.mp.dps = 20
    for entry in TABLE:
        for sample in entry.samples:
            f = entry.integrand(x, sample)
            expected = _value(entry, sample)
            approx = _quadrature(f, entry.lower, entry.upper, entry.oscillatory, _frequency(entry, sample))
            # the oscillatory tails are extrapolated: a looser tolerance,
            # still far below any transcription error (a factor 2 or pi)
            tolerance = 1e-4 if entry.oscillatory else 1e-6
            assert abs(approx - expected) < tolerance * (1 + abs(expected)), (entry, sample, approx, expected)


def test_finite_ranges_with_the_package_check() -> None:
    # the same check through verify_numerically for the entries whose
    # quadrature the package trusts
    for entry in TABLE:
        if entry.oscillatory or entry.upper == oo or entry.lower == -oo:
            continue
        for sample in entry.samples:
            f = entry.integrand(x, sample)
            value = as_expr(entry.value.xreplace(dict(sample)))
            assert verify_numerically(value, f, x, entry.lower, entry.upper) is not False, (entry, sample)

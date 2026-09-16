"""Every entry of the table checked numerically at its sample parameter
values against mpmath's quadrature (the values are transcribed from
Gradshteyn and Ryzhik; a transcription error would show here)."""
from __future__ import annotations

import mpmath

from typing import Optional

from sympy import Expr, symbols, oo, lambdify, exp, I
from sympy.core.power import Pow
from sympy.core.symbol import Wild

from sympy_extras._typing import as_expr
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.tables import TABLE, TableEntry

x = symbols('x')


def _frequency(entry: TableEntry, sample: dict[Wild, Expr]) -> float:
    """The largest scale of the oscillation, for ``quadosc``."""
    scales = [abs(float(v)) for v in sample.values() if as_expr(v).is_number] + [1.0]
    return max(scales)


def _chirp(f: Expr) -> Optional[tuple[float, float]]:
    """``(c, n)`` for an integrand oscillating as ``exp(+-I*c*x**n)`` with
    ``n != 1`` (the zeros of the oscillation are then ``(k*pi/c)**(1/n)``,
    not equally spaced)."""
    for node in f.atoms(exp):
        argument = as_expr(node.args[0])
        coefficient, rest = argument.as_independent(x, as_Add=False)
        if not as_expr(coefficient).has(I) or not isinstance(rest, Pow) or rest.base != x:
            continue
        n = float(as_expr(rest.exp))
        if n != 1:
            return abs(float(as_expr(coefficient / I))), n
    return None


def _quadrature(f: Expr, lower: Expr, upper: Expr, oscillatory: bool, omega: float) -> complex:
    g = lambdify(x, f, 'mpmath')
    lo = float(lower)
    if upper == oo:
        chirp = _chirp(f)
        if chirp is not None:
            # u = x**n makes the oscillation periodic, with a decaying
            # amplitude, which quadosc extrapolates well
            c, n = chirp

            def h(u: float) -> complex:
                return g(u**(1 / n)) * u**(1 / n - 1) / n

            # the end at 0 (a singular amplitude) by plain quadrature
            real = mpmath.quad(lambda u: mpmath.re(h(u)), [lo**n, 1]) \
                + mpmath.quadosc(lambda u: mpmath.re(h(u)), [1, mpmath.inf], omega=c)
            imaginary = mpmath.quad(lambda u: mpmath.im(h(u)), [lo**n, 1]) \
                + mpmath.quadosc(lambda u: mpmath.im(h(u)), [1, mpmath.inf], omega=c)
            return complex(real + 1j * imaginary)
        if oscillatory:
            head = mpmath.quad(g, [lo] + [p for p in (1, 10, 40) if p > lo])
            tail = mpmath.quadosc(g, [40, mpmath.inf], omega=omega)
            return complex(head + tail)
        # the breakpoints beyond the lower end only (a range from k = 3)
        return complex(mpmath.quad(g, [lo] + [p for p in (1, 10, 100) if p > lo] + [mpmath.inf]))
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
            lower, upper = entry.range(sample)
            approx = _quadrature(f, lower, upper, entry.oscillatory, _frequency(entry, sample))
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
            lower, upper = entry.range(sample)
            assert verify_numerically(value, f, x, lower, upper) is not False, (entry, sample)

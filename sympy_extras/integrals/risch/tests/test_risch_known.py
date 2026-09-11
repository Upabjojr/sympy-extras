"""Integrals from Bronstein's *Symbolic Integration I* and from Aaron
Meurer's pull requests which the ported algorithm computes; each result is
checked by differentiation, and SymPy 1.14's ``risch_integrate`` is noted
where it gives up (``NotImplementedError``) or returns the integral."""
from __future__ import annotations

from sympy import symbols, tan, exp, log, atan, cancel, together, diff, S, sin, cos
from sympy.integrals.integrals import Integral

from sympy_extras._typing import as_expr
from sympy_extras.integrals.risch import risch_antiderivative, is_nonelementary

x = symbols('x')


def _check(f: object, expected: object) -> None:
    found = risch_antiderivative(as_expr(f), x)
    assert found is not None, f
    assert cancel(together(diff(found, x) - as_expr(f))) == 0, (f, found)
    if expected is not None:
        assert found == as_expr(expected), (found, expected)


def test_hypertangent_towers() -> None:
    # Bronstein section 5.10; SymPy 1.14 raises NotImplementedError on
    # every tangent tower ("Couldn't find an elementary transcendental
    # extension for tan")
    _check(tan(x), log(tan(x)**2 + 1) / 2)
    _check(tan(x)**2, tan(x) - x)
    _check(tan(x)**5, log(tan(x)**2 + 1) / 2 + tan(x)**4 / 4 - tan(x)**2 / 2)
    # Exercise 5.6 f): the residues give a real arc-tangent
    _check((2 + tan(x)**2) / (1 + (tan(x) + x)**2), atan(x + tan(x)))
    # Example 5.10.3
    _check((tan(x)**5 + tan(x)**3 + x**2 * tan(x) + 1) / (tan(x)**2 + 1)**3, None)
    # an exponential over a tangent: the Risch differential equation at the tangent level
    _check(exp(tan(x)) * (1 + x * (1 + tan(x)**2)), x * exp(tan(x)))
    # Example 5.10.1: x*tan(x) has no elementary integral
    assert is_nonelementary(x * tan(x), x) is True
    assert risch_antiderivative(tan(x)**2 + x * tan(x) + 1, x) is None


def test_real_arc_tangents_from_complex_residues() -> None:
    # Rioboo's rewriting inside the Risch algorithm (Bronstein section 2.8):
    # SymPy 1.14 returns complex logarithms or RootSums here
    _check(exp(x) / ((exp(x) + 1)**2 + 1), atan(exp(x) + 1))
    _check(exp(x) / (exp(2 * x) + 1), atan(exp(x)))
    _check(1 / (x * (log(x)**2 + 1)), atan(log(x)))
    _check((exp(2 * x) + 2 * exp(x) + 7) * exp(x) / (2 * (exp(x) + 3) * (exp(2 * x) + 1)),
           log(exp(x) + 3) / 2 + atan(exp(x)))
    _check((exp(x) + 4) * exp(x) / ((exp(x) + 1)**2 + 1),
           log(exp(2 * x) + 2 * exp(x) + 2) / 2 + 3 * atan(exp(x) + 1))


def test_cancellation_cases() -> None:
    # the parametric Liouvillian cancellation case at the exponential level
    # (SymPy 1.14 hangs or raises NotImplementedError)
    _check(exp(x) * log(exp(x) + 1), exp(x) * log(exp(x) + 1) - exp(x) + log(exp(x) + 1))
    # the structure theorem fallback of parametric_log_deriv
    for F in [(x - 1) * log(exp(x) + log(x)), exp(x) * log(exp(x) + log(x))]:
        _check(cancel(diff(F, x)), None)


def test_classical_exp_log_cases() -> None:
    _check(exp(x**2) * 2 * x, exp(x**2))
    _check(1 / (x * log(x)), log(log(x)))
    _check(exp(x) / x + exp(x) * log(x), exp(x) * log(x))
    assert is_nonelementary(exp(x**2), x) is True
    assert is_nonelementary(exp(x) / x, x) is True
    assert is_nonelementary(1 / log(x), x) is True
    assert is_nonelementary(sin(x) / x, x) is True
    # sin and cos are rewritten through tan(x/2): the tangent tower
    _check(sin(x), -cos(x))


def test_gives_up_quietly() -> None:
    # an algebraic function is outside the transcendental algorithm
    assert risch_antiderivative(x**S.Half, x) is None
    assert is_nonelementary(x**S.Half, x) is None
    assert risch_antiderivative(Integral(x, x), x) is None

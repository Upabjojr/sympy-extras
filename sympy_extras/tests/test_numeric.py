"""Tests of the numerical values which do not depend on a rounding error,
and of the numerical checks of the package which go through them."""
from __future__ import annotations

from sympy import CRootOf, Gt, I, Integer, Rational, asin, atan, cos, exp, im, log, pi, sin, sqrt, symbols

from sympy_extras._numeric import reliable_form, reliable_value
from sympy_extras._typing import as_boolean, as_expr

x = symbols('x')

#: a sum which is exactly zero, and SymPy proves it
Z = -6 + (-2 + sqrt(2))**2 + 4 * sqrt(2)
#: one which SymPy does not decide
W = sin(1)**2 + cos(1)**2 - 1
#: log(2*sqrt(2)) + I*pi, which evalf puts above or below the cut according
#: to the precision (below it at 20, 30 and 100 digits)
L = log(-2 * sqrt(2) - 2 * sqrt(Z), evaluate=False)
RIGHT = log(2 * sqrt(2)) + I * pi
WRONG = log(2 * sqrt(2)) - I * pi


def test_hidden_zeros_are_written_zero() -> None:
    assert reliable_form(as_expr(L)) == RIGHT
    assert reliable_form(as_expr(sqrt(Z) + 1)) == 1
    for digits in (15, 20, 30, 50):
        value = reliable_value(as_expr(L), digits)
        assert value is not None and abs(complex(value) - complex(RIGHT.evalf(20))) < 1e-12
    # a root object: the residual of its polynomial
    root = CRootOf(x**5 - x - 1, 0)
    assert reliable_form(as_expr(sqrt(root**5 - root - 1) + 2)) == 2


def test_an_undecided_sum_is_refused_where_it_matters() -> None:
    assert reliable_form(as_expr(sqrt(W) + 1)) is None
    assert reliable_value(as_expr(log(-1 - sqrt(W))), 20) is None
    # a value which is itself zero is not exposed to a cut
    assert reliable_form(as_expr(W)) is not None
    root = CRootOf(x**2 - 2, 0)
    assert reliable_form(as_expr(asin(sqrt(2) * root / 2) + pi / 2)) is not None


def test_a_part_which_is_exactly_zero() -> None:
    # the real part of log(1 + 3*I) - log(1 - 3*I) is zero, and evalf gives
    # it as 5.7e-29 with twenty digits "of precision", as -1.3e-48 with forty
    base = I * (log(1 + 3 * I) - log(1 - 3 * I)) / 2
    third = Rational(1, 3)
    value = reliable_value(as_expr(base**third - atan(-3)**third), 30)
    assert value is not None and abs(complex(value)) < 1e-25
    # a small part which is there is not a rounding error
    assert reliable_value(as_expr(log(-1 + I * exp(-200))), 20) is not None
    small = reliable_value(as_expr(sqrt(exp(-200) - exp(-201))), 20)
    assert small is not None and abs(complex(small) - 2.9577e-44) < 1e-47


def test_symbols_and_their_values() -> None:
    # the hidden zeros are written 0 before the substitution, which
    # evaluates SymPy's log again: RecursionError, or zoo, in some runs
    e = log(-2 * sqrt(2) * x - 2 * sqrt(Z), evaluate=False)
    value = reliable_value(as_expr(e), 20, {x: Integer(3)})
    assert value is not None and abs(complex(value) - complex((log(6 * sqrt(2)) + I * pi).evalf(20))) < 1e-12
    assert reliable_value(as_expr(e), 20) is None             # not a number


def test_the_checks_of_the_package_go_through_them() -> None:
    # each of these gave the wrong answer, or raised RecursionError, in
    # some runs: the thirty digits of numerically_equal put the argument of
    # the logarithm below the cut
    from sympy_extras.integrals.conditions import numerically_equal
    assert numerically_equal(as_expr(L), as_expr(RIGHT)) is True
    assert numerically_equal(as_expr(L), as_expr(WRONG)) is False
    positive = symbols('p', positive=True)
    parametric = log(-2 * sqrt(2) * positive - 2 * sqrt(Z), evaluate=False)
    assert numerically_equal(as_expr(parametric), as_expr(log(2 * sqrt(2) * positive) + I * pi)) is True
    assert numerically_equal(as_expr(parametric), as_expr(log(2 * sqrt(2) * positive) - I * pi)) is False
    from sympy_extras.assumptions.ask import _clearly_not_zero
    from sympy_extras.assumptions.facts import Facts
    assert _clearly_not_zero(as_expr(parametric - log(2 * sqrt(2) * positive) - I * pi), Facts([])) is False
    from sympy_extras.assumptions.solve import _clearly_not_real
    assert _clearly_not_real(as_expr(L - I * pi)) is False
    assert _clearly_not_real(as_expr(L)) is True
    from sympy_extras.solvers.transcendental import _real_root
    t = symbols('t', real=True)
    assert _real_root(as_expr(t - log(2 * sqrt(2))), t, as_expr(L - I * pi)) is True
    from sympy_extras.integrals.residues import _numerically
    # (unevaluated: SymPy's own im(L) > 0 is decided by the rounding, or
    # raises RecursionError)
    assert _numerically(as_boolean(Gt(im(L, evaluate=False), 0, evaluate=False))) is True

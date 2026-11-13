"""Tests of the native integration of delta functions
(:mod:`sympy_extras.integrals.deltafunctions`)."""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/tests/test_deltafunctions.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: on the core.
# ---------------------------------------------------------------------
from __future__ import annotations

from sympy_extras.core import Integer, Rational, Symbol, oo
from sympy_extras.functions import DiracDelta, cos, dirac_delta, exp, heaviside, sin
from sympy_extras.integrals.deltafunctions import change_mul, deltaintegrate, expand_diracdelta, is_simple

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


def test_change_mul() -> None:
    assert change_mul(x, x) == (None, None)
    assert change_mul(x * y, x) == (None, None)
    assert change_mul(x * y * dirac_delta(x), x) == (DiracDelta(x), x * y)
    assert change_mul(x * y * dirac_delta(x) * dirac_delta(y), x) == (DiracDelta(x), x * y * DiracDelta(y))
    assert change_mul(dirac_delta(x)**2, x) == (DiracDelta(x), DiracDelta(x))


def test_deltaintegrate() -> None:
    assert deltaintegrate(x, x) is None
    assert deltaintegrate(x + dirac_delta(x), x) is None
    assert deltaintegrate(dirac_delta(x, Integer(0)), x) == heaviside(x)
    for n in range(10):
        assert deltaintegrate(dirac_delta(x, Integer(n + 1)), x) == dirac_delta(x, Integer(n))
    assert deltaintegrate(dirac_delta(x), x) == heaviside(x)
    assert deltaintegrate(dirac_delta(-x), x) == heaviside(x)
    assert deltaintegrate(dirac_delta(x - y), x) == heaviside(x - y)
    assert deltaintegrate(dirac_delta(y - x), x) == heaviside(x - y)
    assert deltaintegrate(x * dirac_delta(x), x) == 0
    assert deltaintegrate((x - y) * dirac_delta(x - y), x) == 0
    assert deltaintegrate(dirac_delta(x)**2, x) == dirac_delta(Integer(0)) * heaviside(x)
    assert deltaintegrate(y * dirac_delta(x)**2, x) == y * dirac_delta(Integer(0)) * heaviside(x)
    assert deltaintegrate(dirac_delta(x, Integer(1)), x) == dirac_delta(x)
    assert deltaintegrate(y * dirac_delta(x, Integer(1)), x) == y * dirac_delta(x)
    assert deltaintegrate(dirac_delta(x) * x * sin(x) * cos(x), x) == 0
    assert deltaintegrate(x * sin(x) * cos(x) * dirac_delta(x - 1), x) == sin(1) * cos(1) * heaviside(x - 1)
    assert deltaintegrate(y**2 * dirac_delta(x - z) * dirac_delta(y - z), y) == \
        z**2 * dirac_delta(x - z) * heaviside(y - z)


def test_expansion_by_the_roots() -> None:
    assert is_simple(DiracDelta(2 * x - 1), x) and not is_simple(DiracDelta(x**2), x)
    assert expand_diracdelta(DiracDelta(x**2 - 1), x) == DiracDelta(x - 1) / 2 + DiracDelta(x + 1) / 2
    # a double root is not expanded (the derivative vanishes there)
    assert expand_diracdelta(DiracDelta(x**2), x) == DiracDelta(x**2)
    assert deltaintegrate(dirac_delta(x**2 - 4) * exp(x), x) == \
        exp(2) * heaviside(x - 2) / 4 + exp(-2) * heaviside(x + 2) / 4


def test_definite_integrals_of_delta_functions() -> None:
    # SymPy's integrate answered these for the native definite integrator
    # (the last of its methods); now the native one does
    from sympy_extras.integrals.definite import definite_integral
    assert definite_integral(exp(x) * dirac_delta(x - 1), (x, Integer(0), Integer(2))) == exp(1)
    assert definite_integral(dirac_delta(x - Rational(1, 2)) * x**2, (x, -oo, oo)) == Rational(1, 4)
    assert definite_integral(dirac_delta(x - 3) * x, (x, Integer(0), Integer(1))) == 0

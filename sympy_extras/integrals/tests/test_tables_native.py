"""Tests of the native table of definite integrals
(:mod:`sympy_extras.integrals.tables`) on the objects of the core: the
matching, the conditions decided by the native ``ask``, a sample of entries
against mpmath's quadrature (an independent source; every entry is checked
through the face in ``test_tables_known.py``), and the agreement with the
SymPy face."""
from __future__ import annotations

import mpmath
import sympy

from sympy_extras.core import Catalan, Float, Integer, Rational, S, Symbol, oo, pi
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.logic import BooleanTrue
from sympy_extras.functions import atan, besselj, cos, log, sin
from sympy_extras.integrals import tables_sympy
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.tables import TABLE, A, X, lookup, table_integral
from sympy_extras.interop import from_sympy, to_sympy

x, a, b = Symbol('x'), Symbol('a'), Symbol('b')
p = Symbol('p', positive=True)


def test_lookup_and_conditions() -> None:
    found = lookup(sin(3 * x)**2 / x**2, x, 0, oo)
    assert found is not None
    entry, match, condition = found
    assert entry.reference == 'GR 3.821.9' and match[A] == 3 and isinstance(condition, BooleanTrue)
    found = lookup(sin(a * x)**2 / x**2, x, 0, oo)
    assert found is not None and found[2] == (a > 0)
    assert lookup(sin(x)**2 / x**2, x, 0, 1) is None


def test_values() -> None:
    assert table_integral(sin(p * x)**3 / x**3, x, 0, oo) == ConditionalValue(3 * pi * p**2 / 8)
    assert table_integral(atan(x) / x, x, 0, 1) == ConditionalValue(Catalan)
    # the two cases of GR 4.224.14, the refuted one skipped
    assert table_integral(log(1 - 2 * a * cos(x) + a**2), x, 0, pi, a > 1) == ConditionalValue(2 * pi * log(a))
    found = table_integral(log(1 - 2 * a * cos(x) + a**2), x, 0, pi)
    assert found is not None and found.condition == (a**2 < 1)
    # the value is evaluated at the parameters (1/(2*a) at a = 2 is 1/4)
    assert table_integral(besselj(2, x)**2 / x, x, 0, oo) == ConditionalValue(Rational(1, 4))


def test_sample_entries_numerically() -> None:
    mpmath.mp.dps = 20
    checked = 0
    for entry in TABLE:
        if entry.upper == oo or entry.lower != S.Zero or entry.lower.has(X):
            continue
        for sample in entry.samples:
            f = entry.integrand(x, sample)
            lower, upper = entry.range(sample)
            values: dict[Basic, Basic] = {key: v for key, v in sample.items()}
            value = numeric_value(entry.value.subs(values), 100)
            assert value is not None, entry

            if numeric_value(f.subs(x, Rational(1, 3)), 100) is None:
                continue   # (a function the native evaluation does not take)

            def integrand(t: mpmath.mpf) -> mpmath.mpc:
                found = numeric_value(f.subs(x, Float(float(t))), 100)
                return mpmath.mpc(0) if found is None else found

            hi = numeric_value(upper, 100)
            assert hi is not None
            quadrature = mpmath.quad(integrand, [0, hi.real])
            assert abs(quadrature - value) < 1e-8, entry
            checked += 1
    assert checked >= 10


def test_the_face_agrees() -> None:
    sx, sa = sympy.Symbol('x'), sympy.Symbol('a')
    assert len(tables_sympy.TABLE) == len(TABLE)
    assert tables_sympy.A == to_sympy(A)
    found = tables_sympy.table_integral(sympy.log(1 - 2 * sa * sympy.cos(sx) + sa**2), sx, 0, sympy.pi, sa > 1)
    native = table_integral(log(1 - 2 * a * cos(x) + a**2), x, 0, pi, a > 1)
    assert found is not None and native is not None
    assert from_sympy(found.value) == native.value
    entry = tables_sympy.lookup(sympy.sin(3 * sx)**2 / sx**2, sx, 0, sympy.oo)
    assert entry is not None and entry[0] is tables_sympy.TABLE[[e.reference for e in TABLE].index('GR 3.821.9')]
    assert entry[1][tables_sympy.A] == 3
    assert Integer(3) == from_sympy(entry[1][tables_sympy.A])

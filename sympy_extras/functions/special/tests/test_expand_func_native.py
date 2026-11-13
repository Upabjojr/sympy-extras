"""The ``expand_func`` rules which went to SymPy through the bridge:
``hyper`` and ``meijerg`` (hyperexpand, Gauss's sum at 1), ``polygamma``
of a non-integer order (the derivative of ``zeta``) and ``harmonic`` at
a rational number (the sums of Choi's formula). Checked by the values."""
from __future__ import annotations

from sympy_extras._numeric import reliable_complex
from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import Derivative, Subs, expand_func
from sympy_extras.functions import exp, harmonic, log, polygamma
from sympy_extras.functions.special.hyper import Hyper, MeijerG, hyper, meijerg

z = Symbol('z')


def _close(a: Expr, b: Expr) -> None:
    va, vb = reliable_complex(a, 15), reliable_complex(b, 15)
    assert va is not None and vb is not None, (a, b)
    assert abs(va - vb) <= 1e-10 * max(1.0, abs(va)), (a, b, va, vb)


def test_hyper_and_meijerg_expand_natively() -> None:
    e = expand_func(hyper([Integer(1), Integer(1)], [Integer(2)], z))
    assert not e.has(Hyper)
    # 2F1(1, 1; 2; z) is -log(1 - z)/z
    _close(e.subs(z, Rational(1, 3)), -log(Rational(2, 3)) * 3)
    # Gauss's sum at z = 1
    g = expand_func(Hyper((Rational(1, 3), Rational(1, 2)), (Integer(3),), Integer(1)))
    assert not g.has(Hyper)
    from sympy_extras.functions import gamma
    _close(g, gamma(Integer(3)) * gamma(Integer(3) - Rational(5, 6)) / gamma(Integer(3) - Rational(1, 3))
           / gamma(Integer(3) - Rational(1, 2)))
    m = expand_func(meijerg([], [], [Integer(0)], [], z))
    assert not m.has(MeijerG) and m == exp(-z)


def test_polygamma_of_a_noninteger_order_and_harmonic_of_a_rational() -> None:
    # SymPy's rules with a derivative of zeta and with Sums (they were
    # left to SymPy: None from the native table)
    e = expand_func(polygamma(Rational(1, 2), z))
    assert e.has(Subs) and e.has(Derivative)
    h = expand_func(harmonic(Rational(7, 3)))
    from sympy_extras.concrete.expr_with_limits import Sum
    assert h.has(Sum)
    # H(x) = digamma(x + 1) + EulerGamma
    from sympy_extras.core.numbers import EulerGamma
    _close(h.doit(), polygamma(Integer(0), Rational(10, 3)) + EulerGamma)
    # psi(1/2, z) at z = 2 against the definition by zeta, numerically
    # (the derivative in the first argument of zeta has no value here)
    assert e.subs(z, 2).has(Subs)

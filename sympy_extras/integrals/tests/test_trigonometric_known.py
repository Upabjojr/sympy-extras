"""The trigonometric integrator against the closed forms of Gradshteyn
and Ryzhik (2.51-2.53, 2.41-2.43, 2.66, 2.81) and classical values."""
from __future__ import annotations

from sympy import symbols, sin, cos, tan, sinh, cosh, tanh, exp, log, asin, atan, sqrt, simplify, pi, N, Integral, Expr, S

from sympy_extras._typing import as_expr
from sympy_extras.integrals.trigonometric import trigonometric_antiderivative

x = symbols('x')
a, b = symbols('a b', positive=True)


def _same(u: object, v: object) -> bool:
    return simplify(as_expr(u) - as_expr(v)) == 0


def _definite(f: Expr, lower: Expr, upper: Expr) -> Expr:
    F = trigonometric_antiderivative(f, x)
    assert F is not None
    return as_expr(F.subs(x, upper) - F.subs(x, lower))


def test_gradshteyn_ryzhik_closed_forms() -> None:
    # GR 2.513.1: sin**2 (a x); 2.526.2: sec**3; 2.523: tan**2; 2.511: sin**4
    assert _same(trigonometric_antiderivative(sin(a * x)**2, x), x / 2 - sin(2 * a * x) / (4 * a))
    assert _same(trigonometric_antiderivative(sin(x)**4, x), 3 * x / 8 - sin(2 * x) / 4 + sin(4 * x) / 32)
    assert _same(trigonometric_antiderivative(tan(x)**2, x), tan(x) - x)
    # GR 2.663.1, 2.663.4: e**(bx) sin(ax), x e**(bx) sin(ax)
    assert _same(trigonometric_antiderivative(exp(b * x) * sin(a * x), x),
                 exp(b * x) * (b * sin(a * x) - a * cos(a * x)) / (a**2 + b**2))
    # GR 2.551.3: 1/(a + b cos x) with a > b, through tan(x/2)
    c = symbols('c', positive=True)
    F = trigonometric_antiderivative(1 / (2 * c + c * cos(x)), x)
    assert F is not None and _same(F.diff(x), 1 / (2 * c + c * cos(x)))
    # GR 2.412: sinh**2, cosh**3; 2.423: tanh**3
    assert _same(trigonometric_antiderivative(cosh(x)**3, x), sinh(x) + sinh(x)**3 / 3)
    assert _same(trigonometric_antiderivative(tanh(x)**3, x), log(cosh(x)) - tanh(x)**2 / 2)
    # GR 2.813, 2.822: arcsines and arctangents times powers
    assert _same(trigonometric_antiderivative(x * asin(x), x), x**2 * asin(x) / 2 + x * sqrt(1 - x**2) / 4 - asin(x) / 4)
    assert _same(trigonometric_antiderivative(x * atan(x), x), (x**2 + 1) * atan(x) / 2 - x / 2)


def test_classical_definite_values() -> None:
    # Wallis: Integral(sin**6, (x, 0, pi)) = 5 pi/16; Integral(sec**3, (0, pi/4)) = (sqrt 2 + log(1 + sqrt 2))/2
    assert _same(_definite(sin(x)**6, S.Zero, pi), 5 * pi / 16)
    assert _same(_definite(1 / cos(x)**3, S.Zero, pi / 4), (sqrt(2) + log(1 + sqrt(2))) / 2)
    assert _same(_definite(x * sin(x), S.Zero, pi), pi)
    assert _same(_definite(exp(-x) * sin(x), S.Zero, pi), (1 + exp(-pi)) / 2)
    # 1/(2 + cos x) over a half period, where tan(x/2) is finite
    value = _definite(1 / (2 + cos(x)), S.Zero, pi / 2)
    assert abs(float(N(as_expr(value))) - float(N(Integral(1 / (2 + cos(x)), (x, 0, pi / 2))))) < 1e-12

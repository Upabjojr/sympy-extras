"""The native rewrites of the trigonometric, hyperbolic, inverse and
exponential functions in terms of each other, of square roots, powers and
Bessel functions (ported from SymPy's ``_eval_rewrite_as_<name>``):
checked by the numerical values of the node and of its rewritten form at
sample points, and by the exact values of the cosines in square roots."""
from __future__ import annotations

from typing import Optional, Union

from sympy_extras._numeric import reliable_complex
from sympy_extras.core import E, I, Integer, Rational, Symbol, pi
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import DefinedFunction, FunctionClass
from sympy_extras.functions.elementary import hyperbolic as h, trigonometric as t
from sympy_extras.functions.elementary.exponential import Exp, exp, log
from sympy_extras.functions.special.bessel import BesselJ, SphericalJ

x, y = Symbol('x'), Symbol('y')

Target = Union[Basic, FunctionClass, type, str]


def _value(e: Expr, at: Expr) -> Optional[complex]:
    # substituted first: the conditions of the Piecewise rewrites decide
    substituted = e.subs({x: at, y: Rational(3, 7)})
    assert isinstance(substituted, Expr)
    return reliable_complex(substituted, 15)


def _agree(node: Expr, target: Target, points: list[Expr]) -> None:
    """The node and its rewrite have the same values at the points."""
    rewritten = node.rewrite(target)
    for p in points:
        a, b = _value(node, p), _value(rewritten, p)
        assert a is not None and b is not None, (node, target, p)
        assert abs(a - b) < 1e-10 * max(1.0, abs(a)), (node, target, p, rewritten)


POINTS = [Rational(3, 10), Rational(13, 10), Rational(1, 5) + Rational(2, 5) * I]


def test_trigonometric_rewrites_agree_numerically() -> None:
    nodes: list[Expr] = [t.Sin(x), t.Cos(x), t.Tan(x), t.Cot(x), t.Sec(x), t.Csc(x)]
    targets: list[Target] = [t.Sin, t.Cos, t.Tan, t.Cot, t.Sec, t.Csc, BesselJ, t.Sinc, 'sincos']
    for node in nodes:
        for target in targets:
            if target is type(node) or (target == 'sincos' and isinstance(node, (t.Sin, t.Cos, t.Tan))):
                continue
            if target is t.Sinc and not isinstance(node, t.Sin):
                continue
            if target is t.Cot and isinstance(node, (t.Sin, t.Cos)):
                # the condition Eq(Mod(x, pi), 0) of the Piecewise is not
                # decided at a real point (the core's mod does not reduce
                # 3/10 modulo pi): the complex point only
                _agree(node, target, POINTS[2:])
                continue
            _agree(node, target, POINTS)
    for node in [t.Sin(log(x)), t.Cos(log(x)), t.Tan(log(x)), t.Cot(log(x)), t.Sec(log(x)), t.Csc(log(x))]:
        _agree(node, 'Pow', POINTS)
    _agree(t.Sinc(x), SphericalJ, POINTS)
    _agree(t.Sinc(2 * x + 1), t.Sin, POINTS)


def test_cosines_of_rational_angles_in_square_roots() -> None:
    # the Fermat primes, the factors of 2 and the products of the cos_table
    for q in [3, 5, 8, 12, 15, 16, 17, 24, 51, 85]:
        for p in [1, 2, 3]:
            cosine = t.Cos(Rational(p, q) * pi)
            for target in ['sqrt', 'pow']:
                roots = cosine.rewrite(target)
                assert not roots.has(t.Cos, t.Sin), (cosine, roots)
                a, b = _value(cosine, Integer(0)), _value(roots, Integer(0))
                assert a is not None and b is not None and abs(a - b) < 1e-12, (cosine, roots)
    others: list[Expr] = [t.Sin(pi / 5), t.Tan(pi / 12), t.Cot(pi / 16), t.Sec(3 * pi / 17), t.Csc(pi / 15)]
    for node in others:
        rewritten = node.rewrite('sqrt')
        assert not rewritten.has(t.Cos, t.Sin, t.Tan, t.Cot, t.Sec, t.Csc), (node, rewritten)
        a, b = _value(node, Integer(0)), _value(rewritten, Integer(0))
        assert a is not None and b is not None and abs(a - b) < 1e-12, (node, rewritten)
    # no square roots of cos(pi/7) (not constructible): the node stays
    assert t.Cos(pi / 7).rewrite('sqrt') == t.cos(pi / 7)
    assert t.Tan(x).rewrite('sqrt') == t.tan(x)


def test_hyperbolic_rewrites_agree_numerically() -> None:
    nodes: list[Expr] = [h.Sinh(x), h.Cosh(x), h.Tanh(x), h.Coth(x), h.Sech(x), h.Csch(x)]
    targets: list[Target] = [h.Sinh, h.Cosh, h.Tanh, h.Coth, h.Sech, h.Csch, t.Csc, t.Sec, t.Tan, t.Cot]
    pairs = {(h.Sinh, h.Cosh), (h.Sinh, h.Tanh), (h.Sinh, h.Coth), (h.Sinh, h.Csch), (h.Sinh, t.Csc),
             (h.Cosh, t.Sec), (h.Cosh, h.Sinh), (h.Cosh, h.Tanh), (h.Cosh, h.Coth), (h.Cosh, h.Sech),
             (h.Tanh, t.Tan), (h.Tanh, t.Cot), (h.Tanh, h.Sinh), (h.Tanh, h.Cosh), (h.Tanh, h.Coth),
             (h.Coth, h.Sinh), (h.Coth, h.Cosh), (h.Coth, h.Tanh), (h.Csch, h.Tanh), (h.Csch, h.Coth),
             (h.Sech, h.Tanh), (h.Sech, h.Coth), (h.Csch, t.Csc), (h.Csch, h.Cosh), (h.Csch, h.Sinh),
             (h.Sech, t.Sec), (h.Sech, h.Sinh), (h.Sech, h.Cosh)}
    count = 0
    for node in nodes:
        for target in targets:
            if (type(node), target) in pairs:
                _agree(node, target, POINTS)
                count += 1
    assert count == len(pairs)


def test_inverse_rewrites_agree_numerically() -> None:
    # SymPy's rules are identities on the principal branches inside the
    # domains sampled here (the real interval (0, 1) for the inverse
    # trigonometric functions and the inverse tanh, (1, oo) for acosh,
    # acoth, acsc and asec, (0, 1) for asech, the positive reals for asinh
    # and acsch)
    inside: list[Expr] = [Rational(3, 10), Rational(7, 10)]
    outside: list[Expr] = [Rational(13, 10), Rational(5, 2)]
    trig: list[type[DefinedFunction]] = [t.Asin, t.Acos, t.Atan, t.Acot, t.Asec, t.Acsc]
    for cls in trig:
        for target in trig:
            if target is not cls:
                _agree(cls(x), target, outside if cls in (t.Asec, t.Acsc) else inside)
    hyperbolic: list[tuple[type[DefinedFunction], list[type[DefinedFunction]], list[Expr]]] = [
        (h.Asinh, [h.Atanh, h.Acosh, t.Asin, t.Acos], inside + outside),
        (h.Acosh, [t.Acos, t.Asin, h.Asinh, h.Atanh], outside),
        (h.Atanh, [h.Asinh], inside),
        (h.Acoth, [h.Atanh, h.Asinh], outside),
        (h.Asech, [h.Acosh, h.Asinh, h.Atanh, h.Acsch], inside),
        (h.Acsch, [h.Asinh, h.Acosh, h.Atanh], inside + outside)]
    for cls, targets, points in hyperbolic:
        for target in targets:
            _agree(cls(x), target, points)


def test_exponential_and_power_rewrites() -> None:
    _agree(Exp(x), h.Tanh, POINTS)
    _agree(E**x, h.Tanh, POINTS)
    # exp(I*pi*c) in square roots when the cosine and the sine evaluate
    for c in [Rational(1, 3), Rational(2, 5), Rational(1, 12)]:
        for node in [Exp(I * pi * c), E**(I * pi * c)]:
            rewritten = node.rewrite('sqrt')
            assert not rewritten.has(Exp, t.Cos, t.Sin) and rewritten != node, (node, rewritten)
            a, b = _value(node, Integer(0)), _value(rewritten, Integer(0))
            assert a is not None and b is not None and abs(a - b) < 1e-12
    assert Exp(I * pi / 7).rewrite('sqrt') == exp(I * pi / 7)
    _agree(Exp(3 * log(x)), 'Pow', POINTS)
    _agree(Exp(y * log(x)), 'Pow', POINTS)
    # E itself in sines and cosines
    for target in [t.Sin, t.Cos]:
        rewritten = E.rewrite(target)
        a = _value(rewritten, Integer(0))
        assert rewritten != E and a is not None and abs(a - 2.718281828459045) < 1e-12

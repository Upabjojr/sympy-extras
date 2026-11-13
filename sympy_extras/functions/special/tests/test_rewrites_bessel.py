"""The native rewrites of the Bessel, spherical Bessel and Airy functions
and of the spherical harmonics between themselves, and the native
``expand_func`` of the Airy functions and of ``Ynm``: each result agrees
numerically with the node it comes from (reliable values at sample
points), and no bridge is asked."""
from __future__ import annotations

from typing import Callable, Optional, Union

from sympy_extras._numeric import reliable_complex
from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import ExpandHints
from sympy_extras.functions.elementary.trigonometric import Cos, Sin
from sympy_extras.functions.special import bessel as b
from sympy_extras.functions.special._expand_func_bessel import rules as expand_rules
from sympy_extras.functions.special.spherical_harmonics import YnmNode

z = Symbol('z')
th, ph = Symbol('theta'), Symbol('phi')
POINTS: list[Expr] = [Rational(7, 10), Rational(23, 10), Rational(3, 10) + Rational(2, 5) * I, Rational(-13, 10)]


def _agree(before: Expr, after: Basic, symbols: list[Symbol], points: Optional[list[Expr]] = None) -> None:
    assert isinstance(after, Expr)
    assert after != before, before
    checked = 0
    for k, point in enumerate(points if points is not None else POINTS):
        values: dict[Symbol, Expr] = {s: point + Integer(i) / 7 for i, s in enumerate(symbols)}
        v0 = reliable_complex(before, 15, values)
        v1 = reliable_complex(after, 15, values)
        if v0 is None or v1 is None:
            continue
        checked += 1
        assert abs(v0 - v1) <= 1e-10 * max(1.0, abs(v0)), (before, after, values, v0, v1)
    assert checked, (before, after)


Target = Union[type, str]


def test_bessel_rewrites_agree_numerically() -> None:
    cases: list[tuple[Callable[[Expr, Expr], Expr], list[Target]]] = [
        (b.BesselJ, [b.BesselY]), (b.BesselY, [b.BesselI, b.SphericalY]), (b.BesselI, [b.BesselY]),
        (b.BesselK, [b.BesselY, b.SphericalY])]
    for cls, targets in cases:
        for target in targets:
            for nu in (Rational(1, 3), Rational(5, 2), Rational(-7, 4)):
                node = cls(nu, z)
                _agree(node, node.rewrite(target), [z], POINTS[:3])


def test_spherical_bessel_rewrites_agree_numerically() -> None:
    cases: list[tuple[Callable[[Expr, Expr], Expr], list[Target]]] = [
        (b.SphericalJ, [b.BesselJ, b.BesselY, b.SphericalY]), (b.SphericalY, [b.BesselJ, b.BesselY, b.SphericalJ]),
        (b.SphericalHankel1, [b.BesselJ, b.BesselY, b.SphericalJ, b.SphericalY, b.Hankel1]),
        (b.SphericalHankel2, [b.BesselJ, b.BesselY, b.SphericalJ, b.SphericalY, b.Hankel2])]
    for cls, targets in cases:
        for target in targets:
            for n in (Integer(0), Integer(2), Integer(-3)):
                node = cls(n, z)
                _agree(node, node.rewrite(target), [z], POINTS[:3])


def test_spherical_hankel_as_hankel_has_order_n_plus_half() -> None:
    # SymPy 1.14's hn1(n, z).rewrite(hankel1) is sqrt(pi/(2*z))*hankel1(n, z),
    # which is wrong (hn1(0, z) is -I*exp(I*z)/z, the order is n + 1/2)
    for cls, target in ((b.SphericalHankel1, b.Hankel1), (b.SphericalHankel2, b.Hankel2)):
        node = cls(Integer(1), z)
        rewritten = node.rewrite(target)
        assert rewritten.has(target(Rational(3, 2), z))
        _agree(node, rewritten, [z], POINTS[:3])


def test_airy_rewrites_agree_numerically() -> None:
    # the Bessel forms hold for the real part of the argument of the sign
    # SymPy's rule asks for, the hyper form everywhere
    p = Symbol('p', positive=True)
    for cls in (b.AiryAi, b.AiryBi, b.AiryAiPrime, b.AiryBiPrime):
        for target in (b.BesselJ, b.BesselI, 'hyper'):
            for arg in (-p, p, z):
                node = cls(arg)
                rewritten = node.rewrite(target)
                if rewritten == node:
                    assert target == b.BesselJ and arg != -p
                    continue
                points = POINTS[:2] if arg != z else POINTS
                _agree(node, rewritten, [p] if arg != z else [z], points)


def test_airyaiprime_as_besseli_off_the_right_half_plane() -> None:
    # SymPy 1.14's airyaiprime(z).rewrite(besseli) has besseli(-1/3, ...) in
    # its second term where re(z) is not positive: -0.1577 + 0.1401*I at
    # z = -7/10, where airyaiprime is -0.1446
    p = Symbol('p', positive=True)
    node = b.AiryAiPrime(-p)
    _agree(node, node.rewrite(b.BesselI), [p], POINTS[:2])


def test_ynm_rewrites_agree_numerically() -> None:
    for n, m in ((0, 0), (1, 1), (2, 1), (3, -2)):
        node = YnmNode(Integer(n), Integer(m), th, ph)
        for target in ('polynomial', Cos, Sin):
            # theta in (0, pi): SymPy's rules replace |sin(theta)| by sin(theta)
            _agree(node, node.rewrite(target), [th, ph], [Rational(7, 10), Rational(23, 10)])


def test_airy_and_ynm_expand_func_agree_numerically() -> None:
    hints = ExpandHints(True, True, True, True, False, False)
    table = expand_rules()
    third = Rational(1, 3)
    for cls in (b.AiryAi, b.AiryBi, b.AiryAiPrime, b.AiryBiPrime):
        rule = table[cls]
        for arg in ((3 * z**3)**third, 2 * (5 * z**6)**third, (z**2)**third):
            node = cls(arg)
            expanded = rule(node, hints)
            assert expanded is not None
            _agree(node, expanded, [z])
    node_y = YnmNode(Integer(2), Integer(1), th, ph)
    expanded_y = table[YnmNode](node_y, hints)
    assert expanded_y is not None
    _agree(node_y, expanded_y, [th, ph], [Rational(7, 10), Rational(23, 10)])

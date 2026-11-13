"""The rewrite rules of the special functions between themselves and into
integrals, sums, hypergeometric and Meijer G-functions (ported from SymPy's
``_eval_rewrite_as_<name>``): each rewrite agrees numerically with the
node at sample points."""
from __future__ import annotations

from typing import Optional, Union

import mpmath

from sympy_extras._numeric import reliable_complex
from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.numbers import I
from sympy_extras.core.rewriting import rewrite
from sympy_extras.functions.special import error_functions as er, gamma_functions as g, zeta_functions as zf
from sympy_extras.functions.combinatorial.numbers import Bernoulli, Genocchi, Harmonic

x, y = Symbol('x'), Symbol('y')
R = Rational
Target = Union[type, str]
POINTS: list[dict[Symbol, Expr]] = [{x: R(7, 10), y: R(-13, 10)}, {x: R(3, 10) + R(2, 5) * I, y: R(23, 10)},
                                    {x: R(11, 5), y: R(1, 3) - I}]


def _quadratures(e: Basic) -> Basic:
    """``e`` with its definite integrals (in one variable, of numbers)
    replaced by their values by quadrature."""
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.numbers import Float
    from sympy_extras.integrals.integral import Integral
    if not e.args:
        return e
    args = tuple(_quadratures(a) for a in e.args)
    rebuilt = e._evaluated(args) if args != e.args else e
    if isinstance(rebuilt, Integral) and rebuilt.bounds is not None:
        f, t = rebuilt.function, rebuilt.variable
        def bound(b: Expr) -> mpmath.mpf:
            from sympy_extras.core.numbers import Infinity, NegativeInfinity
            if isinstance(b, Infinity):
                return mpmath.inf
            if isinstance(b, NegativeInfinity):
                return -mpmath.inf
            c = reliable_complex(b, 15)
            assert c is not None and c.imag == 0
            return mpmath.mpf(c.real)

        def integrand(u: mpmath.mpf) -> mpmath.mpc:
            # the integrand at a quadrature node: its plain value (the sums
            # agree to the tolerance of the test, nothing is decided by it)
            v = numeric_value(as_expr(f.xreplace({t: Float(str(u), 20)})), 60)
            assert v is not None
            return v

        lower, upper = rebuilt.bounds
        value = complex(mpmath.quad(integrand, [bound(lower), bound(upper)]))
        return Float(str(value.real), 20) + I * Float(str(value.imag), 20)
    return rebuilt


def _agrees(e: Expr, target: Target, points: list[dict[Symbol, Expr]] = POINTS,
            reference: Optional[Expr] = None) -> None:
    rewritten = rewrite(e, (), target)
    assert isinstance(rewritten, Expr)
    assert rewritten != e, (e, target)
    numeric = _quadratures(rewritten)
    assert isinstance(numeric, Expr)
    checked = 0
    for values in points:
        theirs = reliable_complex(e if reference is None else reference, 15, values)
        ours = reliable_complex(numeric, 15, values)
        if theirs is None or ours is None:
            continue
        checked += 1
        assert abs(theirs - ours) <= 1e-9 * max(1.0, abs(theirs)), (e, target, rewritten, values, theirs, ours)
    assert checked, (e, target, rewritten)


def _all(nodes: list[Expr], targets: list[Target], points: list[dict[Symbol, Expr]] = POINTS) -> None:
    for e in nodes:
        for t in targets:
            _agrees(e, t, points)


def _positive() -> tuple[Symbol, list[dict[Symbol, Expr]]]:
    p = Symbol('p', positive=True)
    return p, [{p: R(7, 10)}, {p: R(23, 10)}]


def test_error_functions_between_themselves() -> None:
    _all([er.Erf(x)], [er.Erfc, er.Erfi, er.FresnelS, er.FresnelC, 'hyper', 'meijerg', g.UpperGamma, er.Expint])
    _all([er.Erfc(x)], [er.Erfi, er.FresnelS, er.FresnelC, 'hyper', 'meijerg', g.UpperGamma, er.Expint])
    _all([er.Erfi(x)], [er.Erfc, er.FresnelS, er.FresnelC, 'hyper', 'meijerg', g.UpperGamma, er.Expint])
    _all([er.Erf2(x, y)], [er.Erfc, er.Erfi, er.FresnelS, er.FresnelC, 'hyper', g.UpperGamma, er.Expint])
    _all([er.FresnelS(x), er.FresnelC(x)], [er.Erf, 'hyper', 'meijerg'])
    _agrees(er.Erfinv(R(1, 3)), er.Erfcinv)
    _agrees(er.Erfcinv(R(1, 3)), er.Erfinv)


def test_erfi_as_expint_is_right_on_the_real_line() -> None:
    # SymPy 1.14's erfi(x).rewrite(expint), sqrt(-x**2)/x - x*expint(1/2, -x**2)/sqrt(pi),
    # is erfi(x) + 2*I at x = 7/10 (the sign of its first term); ported with the sign fixed
    points: list[dict[Symbol, Expr]] = [{x: R(7, 10)}, {x: R(-13, 10)}, {x: R(3, 10) + R(2, 5) * I}]
    _agrees(er.Erfi(x), er.Expint, points)


def test_exponential_and_trigonometric_integrals() -> None:
    p, positive = _positive()
    _all([er.EiNode(p)], [er.ChiNode, er.ShiNode, er.LiNode, g.UpperGamma], positive)
    _all([er.Expint(R(1, 2), p), er.Expint(R(1, 3), p)], [g.UpperGamma], positive)
    _all([er.Expint(Integer(1), p)], [er.SiNode], positive)
    _all([er.LiNode(p + 2)], [er.EiNode, er.OffsetLiNode, er.SiNode, er.ShiNode, 'hyper', 'meijerg', g.UpperGamma],
         positive)
    _all([er.SiNode(p), er.CiNode(p), er.ShiNode(p), er.ChiNode(p)], [er.EiNode, er.Expint, g.UpperGamma], positive)


def test_li_as_shi_and_chi_is_li() -> None:
    # SymPy 1.14's li(z).rewrite(Shi), Chi(log(z)) - Shi(log(z)) - (log(1/log(z)) - log(log(z)))/2,
    # is li(27/10) - 2.105...; ported as Ei(log(z)) in Chi and Shi, on every branch
    z = Symbol('z')
    points: list[dict[Symbol, Expr]] = [{z: R(27, 10)}, {z: R(1, 3)}, {z: 3 + I}, {z: Integer(-2)}, {z: -2 - I}]
    _all([er.LiNode(z)], [er.ShiNode, er.ChiNode], points)


def test_gamma_and_zeta_families() -> None:
    p, positive = _positive()
    _all([g.LowerGamma(R(1, 2), p)], [g.UpperGamma, er.Expint], positive)
    _all([g.UpperGamma(R(5, 2), p)], [g.LowerGamma, er.Expint], positive)
    _all([g.PolyGamma(Integer(2), p)], [zf.Zeta, Harmonic], positive)
    _all([g.PolyGamma(Integer(0), p)], [Harmonic], positive)
    _all([g.DiGamma(p)], [Harmonic, g.PolyGamma], positive)
    _all([g.TriGamma(p)], [Harmonic, g.PolyGamma, zf.Zeta], positive)
    _all([zf.Zeta(R(1, 2))], [zf.DirichletEta, zf.LerchPhi])
    _agrees(zf.Zeta(Integer(4)), Bernoulli)
    _agrees(zf.DirichletEta(Integer(0)), Genocchi)
    _agrees(zf.PolyLog(Integer(3), R(1, 3)), zf.LerchPhi)
    _agrees(zf.LerchPhi(R(1, 3), Integer(2), R(1, 2)), zf.PolyLog)
    _agrees(zf.RiemannXi(R(1, 2)), zf.Zeta)


def test_integral_and_hypergeometric_forms() -> None:
    from sympy_extras.concrete.expr_with_limits import Sum
    from sympy_extras.core.add import add
    from sympy_extras.functions.special import beta_functions as bf, elliptic_integrals as el
    from sympy_extras.functions.special.bessel import BesselI, MarcumQ
    numbers: list[dict[Symbol, Expr]] = [{}]
    # Ei over (-oo, z) is a principal value for z > 0: a negative argument
    _all([er.FresnelS(R(7, 10)), er.FresnelC(R(7, 10)), er.EiNode(R(-7, 10)), er.SiNode(R(7, 10)),
          er.CiNode(R(7, 10)), er.Expint(Integer(2), R(7, 10)), bf.Beta(R(1, 2), R(3, 2)), el.EllipticK(R(1, 3)),
          el.EllipticE(R(1, 3)), el.EllipticE(R(1, 2), R(1, 3)), el.EllipticF(R(1, 2), R(1, 3)),
          el.EllipticPi(R(1, 4), R(1, 3)), el.EllipticPi(R(1, 4), R(1, 2), R(1, 3))], ['Integral'], numbers)
    _all([bf.BetaInc(Integer(2), Integer(3), R(1, 5), R(1, 2)),
          bf.BetaIncRegularized(Integer(2), Integer(3), R(1, 5), R(1, 2))], ['Integral', 'hyper'], numbers)
    _all([el.EllipticK(R(1, 3)), el.EllipticE(R(1, 3))], ['hyper', 'meijerg'], numbers)
    # Marcum's Q-function has no numerical value of its own: its Bessel form at a = b is the reference
    for m in (Integer(1), Integer(2)):
        q = MarcumQ(m, R(1, 2), R(1, 2))
        closed = rewrite(q, (), BesselI)
        assert isinstance(closed, Expr) and closed != q
        _agrees(q, 'Integral', numbers, closed)
        summed = rewrite(q, (), 'Sum')
        assert isinstance(summed, Expr)
        sums = [a for a in summed.atoms(Sum) if isinstance(a, Sum)]
        assert len(sums) == 1
        (k, lower, _), = sums[0].limits
        partial = add(*[as_expr(sums[0].function.xreplace({k: lower + i})) for i in range(40)])
        replacement: dict[Expr, Expr] = {sums[0]: partial}
        value = reliable_complex(as_expr(summed.xreplace(replacement)), 15)
        reference = reliable_complex(closed, 15)
        assert value is not None and reference is not None and abs(value - reference) < 1e-9


def test_hyper_as_sum_and_hyperrep_closed_forms() -> None:
    from sympy_extras.core.rewriting import RewriteHints
    from sympy_extras.functions.special._rewrites import node_rules
    from sympy_extras.functions.special.hyper import Hyper
    from sympy_extras.functions.special.hyperrep import HyperRep_atanh, HyperRep_power1
    h = Hyper([R(1, 2), Integer(1)], [R(5, 2)], R(1, 3))
    rule = node_rules()[(Hyper, 'Sum')]
    summed = rule(h, h.args, RewriteHints())
    assert isinstance(summed, Expr)
    theirs, ours = reliable_complex(h, 15), reliable_complex(summed, 15)
    assert theirs is not None and ours is not None and abs(theirs - ours) < 1e-9
    small: list[dict[Symbol, Expr]] = [{x: R(1, 3)}, {x: R(-1, 2)}]
    _agrees(HyperRep_atanh(x), 'nonrepsmall', small, Hyper([R(1, 2), Integer(1)], [R(3, 2)], x))
    _agrees(HyperRep_power1(R(1, 3), x), 'nonrepsmall', small, Hyper([-R(1, 3)], [], x))
    _agrees(HyperRep_atanh(x), 'nonrep', small, Hyper([R(1, 2), Integer(1)], [R(3, 2)], x))
